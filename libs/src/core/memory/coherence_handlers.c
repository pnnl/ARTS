/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message handlers and senders.
 *
 * The protocol's drop-discipline (lookup -> destroy_state precheck ->
 * either OoO defer or explicit wake-up reply on failure) is
 * implemented via the `home_lookup_or_defer` helper at the top of this
 * file; every home-side handler entry funnels through it.  Sharer-side
 * response handlers do their own lookup + cache.destroy_state check
 * inline (no requester to notify back -- the message *is* the
 * requester's own context).
 *
 * Memory ordering: ARTS atomics are __sync_*-based (full fence) and
 * the per-rank network thread (S1) is the sole writer of home.*
 * state, so the trickier orderings are confined to:
 *   - cache.writer_count (worker ↔ network handler)
 *   - cache.buffer       (worker ↔ network handler installs)
 *   - cache.pending_count (worker push ↔ marker mark)
 *   - cache.destroy_state (forward-only tri-state)
 * All accessed via arts_atomic_*, all matching the algorithm in the
 * coherence design plan.
 */

#include "arts/memory/coherence_handlers.h"

#include <stdlib.h>
#include <string.h>

#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/memory/coherence.h"
#include "arts/memory/coherence_acquire.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_home.h"
#include "arts/memory/db.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== Sender helpers ============================================== */

/* Single-node note: arts_remote_send_request_async drops messages
 * whose destination is the local rank (self_send_check rejects).  v3
 * uses uniform "send to home" semantics including home == self, so
 * we dispatch handlers directly when rank == self instead of going
 * over the network. */

void arts_coh_send_lock_req(unsigned int home_rank, arts_guid_t db_guid) {
  struct arts_remote_lock_req_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), ARTS_REMOTE_LOCK_REQ_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_coh_handle_lock_req(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_coh_send_grant(unsigned int requester_rank, arts_guid_t db_guid,
                         uint64_t version, bool has_next, const void *data,
                         uint64_t data_size) {
  struct arts_remote_grant_packet_s p;
  uint64_t total = sizeof(p) + (data ? data_size : 0);
  arts_fill_packet_header(&p.header, total, ARTS_REMOTE_GRANT_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.has_next = has_next ? 1u : 0u;
  p.data_present = data ? 1u : 0u;
  memset(p.pad, 0, sizeof(p.pad));
  if (requester_rank == arts_global_rank_id) {
    arts_coh_handle_grant(&p, data, data ? data_size : 0);
    return;
  }
  if (data && data_size > 0) {
    arts_remote_send_request_payload_async((int)requester_rank, (char *)&p,
                                           sizeof(p), (char *)data, data_size);
  } else {
    arts_remote_send_request_async((int)requester_rank, (char *)&p, sizeof(p));
  }
}

void arts_coh_send_writeback(unsigned int home_rank, arts_guid_t db_guid,
                             uint64_t version, uint64_t seq,
                             arts_writeback_flag_t flag, const void *data,
                             uint64_t data_size) {
  struct arts_remote_writeback_packet_s p;
  uint64_t total = sizeof(p) + data_size;
  arts_fill_packet_header(&p.header, total, ARTS_REMOTE_WRITEBACK_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.seq = seq;
  p.flag = (uint8_t)flag;
  memset(p.pad, 0, sizeof(p.pad));
  if (home_rank == arts_global_rank_id) {
    arts_coh_handle_writeback(&p, data, data_size);
    return;
  }
  /* Sentinel DBs (db_size==0) still need WRITEBACK for ownership
   * transfer / R3 ordering, but the payload-async path errors on
   * zero-size payload — route via the no-payload async send. */
  if (data == NULL || data_size == 0) {
    arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
    return;
  }
  arts_remote_send_request_payload_async((int)home_rank, (char *)&p, sizeof(p),
                                         (char *)data, data_size);
}

void arts_coh_send_writeback_ack(unsigned int releaser_rank,
                                 arts_guid_t db_guid, uint64_t seq) {
  struct arts_remote_writeback_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), ARTS_REMOTE_WRITEBACK_ACK_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.seq = seq;
  if (releaser_rank == arts_global_rank_id) {
    arts_coh_handle_writeback_ack(&p);
    return;
  }
  arts_remote_send_request_async((int)releaser_rank, (char *)&p, sizeof(p));
}

void arts_coh_send_invalidate_notice(unsigned int owner_rank,
                                     arts_guid_t db_guid) {
  struct arts_remote_invalidate_notice_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p),
                          ARTS_REMOTE_INVALIDATE_NOTICE_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (owner_rank == arts_global_rank_id) {
    arts_coh_handle_invalidate_notice(&p);
    return;
  }
  arts_remote_send_request_async((int)owner_rank, (char *)&p, sizeof(p));
}

void arts_coh_send_release_ownership(unsigned int home_rank,
                                     arts_guid_t db_guid) {
  struct arts_remote_release_ownership_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p),
                          ARTS_REMOTE_RELEASE_OWNERSHIP_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_coh_handle_release_ownership(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_coh_send_get_data(unsigned int home_rank, arts_guid_t db_guid,
                            void *waiter_addr) {
  struct arts_remote_get_data_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), ARTS_REMOTE_GET_DATA_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.waiter_addr = (uint64_t)(uintptr_t)waiter_addr;
  if (home_rank == arts_global_rank_id) {
    arts_coh_handle_get_data(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_coh_send_data_response(unsigned int requester_rank,
                                 arts_guid_t db_guid, uint64_t version,
                                 void *waiter_addr, const void *data,
                                 uint64_t data_size) {
  struct arts_remote_data_response_packet_s p;
  uint64_t total = sizeof(p) + (data ? data_size : 0);
  arts_fill_packet_header(&p.header, total, ARTS_REMOTE_DATA_RESPONSE_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.waiter_addr = (uint64_t)(uintptr_t)waiter_addr;
  p.data_present = data ? 1u : 0u;
  memset(p.pad, 0, sizeof(p.pad));
  if (requester_rank == arts_global_rank_id) {
    arts_coh_handle_data_response(&p, data, data ? data_size : 0);
    return;
  }
  if (data && data_size > 0) {
    arts_remote_send_request_payload_async((int)requester_rank, (char *)&p,
                                           sizeof(p), (char *)data, data_size);
  } else {
    arts_remote_send_request_async((int)requester_rank, (char *)&p, sizeof(p));
  }
}

void arts_coh_send_db_create_coherent(unsigned int home_rank,
                                      arts_guid_t db_guid, uint64_t db_size,
                                      uint16_t flags, uint16_t db_type) {
  struct arts_remote_db_create_coherent_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p),
                          ARTS_REMOTE_DB_CREATE_COHERENT_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.db_size = db_size;
  p.flags = flags;
  p.db_type = db_type;
  memset(p.pad, 0, sizeof(p.pad));
  if (home_rank == arts_global_rank_id) {
    arts_coh_handle_db_create_coherent(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_coh_send_destroy_req(unsigned int home_rank, arts_guid_t db_guid) {
  struct arts_remote_destroy_req_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), ARTS_REMOTE_DESTROY_REQ_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_coh_handle_destroy_req(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_coh_send_destroy_notify(unsigned int sharer_rank,
                                  arts_guid_t db_guid) {
  struct arts_remote_destroy_notify_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), ARTS_REMOTE_DESTROY_NOTIFY_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (sharer_rank == arts_global_rank_id) {
    arts_coh_handle_destroy_notify(&p);
    return;
  }
  arts_remote_send_request_async((int)sharer_rank, (char *)&p, sizeof(p));
}

/* ===== Forward decls: handler wiring with B4/B5/B11 ================= */

/* These are implemented in the acquire/release/destroy modules and
 * called from the response handlers below.  Forward-declared here to
 * avoid a circular include with coherence_acquire / coherence_destroy
 * (which themselves include this header for the sender helpers). */
void arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                           uint64_t version, bool has_next);
void arts_coh_drain_pending_ro(struct arts_db_cache_s *cache, uint64_t version);
void arts_coh_trigger_ro_waiter(struct arts_db_cache_s *cache,
                                struct arts_db_ro_waiter_s *w,
                                uint64_t version);
void arts_coh_fail_trigger_pending(struct arts_db_cache_s *cache);
void arts_coh_try_finalize_destroy(struct arts_db_cache_s *cache);

/* ===== home_lookup_or_defer helper (Phase 2.2 OoO defer) ============ */

/* Reply kind selector for the helper.  Mirrors the design spec's
 * REPLY_DESTROY_NOTIFY / REPLY_WB_ACK / REPLY_NONE. */
typedef enum {
  COH_REPLY_NONE = 0,
  COH_REPLY_DESTROY_NOTIFY,
  COH_REPLY_WB_ACK,
} coh_reply_kind_t;

/* Look up cache by guid; if not yet installed, defer the wire message
 * via the OoO list so it re-issues once DB_CREATE arrives.  Spec §4.9.
 *
 * Behaviour matrix:
 *   cache != NULL, destroy_state == NONE    -> return cache (caller body)
 *   cache != NULL, destroy_state != NONE    -> reply per reply_kind, NULL
 *   cache == NULL, packet_for_oo == NULL    -> reply per reply_kind, NULL
 *   cache == NULL, ENQUEUED                 -> NULL (fire_oo will retry)
 *   cache == NULL, FIRED_BY_DRAIN           -> NULL (handler already ran via
 *                                              the drain triggered inside
 *                                              add_oo_ex; payload freed)
 *   cache == NULL, AVAILABLE_NOW (race)     -> re-lookup; same precheck
 *
 * The OoO payload is a heap copy of `packet_for_oo` whose first field is
 * forced to `oo_type` (every OO_COH_* payload begins with oo_type_t). */
static struct arts_db_cache_s *
home_lookup_or_defer(arts_guid_t guid, unsigned int requester,
                     oo_type_t oo_type, void *packet_for_oo, size_t packet_size,
                     coh_reply_kind_t reply_kind) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(guid);

  if (cache != NULL) {
    if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
      if (reply_kind == COH_REPLY_DESTROY_NOTIFY) {
        arts_coh_send_destroy_notify(requester, guid);
      }
      return NULL;
    }
    return cache;
  }

  /* cache == NULL — defer via OoO or reply immediately. */
  if (packet_for_oo == NULL) {
    if (reply_kind == COH_REPLY_DESTROY_NOTIFY) {
      arts_coh_send_destroy_notify(requester, guid);
    }
    return NULL;
  }

  /* Allocate OoO payload (the network packet itself will soon be
   * overwritten when the receiver moves on to the next message). */
  void *oo_data = arts_malloc(packet_size);
  memcpy(oo_data, packet_for_oo, packet_size);
  /* Stamp the OO_COH_* tag.  First field of every OO payload struct is
   * oo_type_t (matches oo_generic_s dispatch). */
  *(oo_type_t *)oo_data = oo_type;

  oo_add_result_t r = arts_route_table_add_oo_ex(guid, oo_data);
  if (r == OO_RESULT_FIRED_BY_DRAIN) {
    /* The OO_COH_* dispatcher inside fire_oo already re-issued the
     * handler and freed our payload.  Caller must do nothing further. */
    return NULL;
  }
  if (r == OO_RESULT_AVAILABLE_NOW) {
    /* Race: data became non-NULL on the pre-push or push-time check, so
     * the payload was never inserted into the OO list.  Free our heap
     * copy and re-lookup so the caller can re-issue inline. */
    arts_free(oo_data);
    cache = arts_coh_route_table_lookup_cache(guid);
    if (cache != NULL &&
        arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
      if (reply_kind == COH_REPLY_DESTROY_NOTIFY) {
        arts_coh_send_destroy_notify(requester, guid);
      }
      return NULL;
    }
    return cache;
  }
  /* OO_RESULT_ENQUEUED — the future fire_oo on DB_CREATE arrival will
   * re-issue the handler with the deferred payload. */
  return NULL;
}

/* ===== home.last_sent_version atomic-monotonic helpers ============== */

/* update_last_sent_max: the GET_DATA reply path.  Decide send-with-
 * data vs send-no-data based on the home watermark, then advance the
 * watermark.  Under single-threaded handler dispatch the "atomic
 * CAS-loop" the design plan specifies collapses to a plain compare/
 * advance — but we keep the helper signature so future MPMC upgrades
 * are localized. */
static void update_last_sent_max(struct arts_db_cache_s *cache,
                                 unsigned int requester, uint64_t master_v,
                                 const void *data, uint64_t data_size,
                                 void *waiter_addr) {
  /* Design plan §update_last_sent_max line 919-928: monotonic dedup —
   * if the requester already received this version (cur >= master_v),
   * send NO_DATA.  Cache_s lifetime invariant guarantees user_data
   * persists until destroy (route_table ref tracking). */
  uint64_t cur =
      arts_rank_u64_map_get(cache->home->last_sent_version, requester);
  if (cur >= master_v) {
    arts_coh_send_data_response(requester, cache->db_guid, master_v,
                                waiter_addr, NULL, 0);
    return;
  }
  arts_rank_u64_map_set(cache->home->last_sent_version, requester, master_v);
  arts_coh_send_data_response(requester, cache->db_guid, master_v, waiter_addr,
                              data, data_size);
}

/* grant_to: same monotonic-CAS logic for the GRANT path used by
 * local_transfer / WRITEBACK_AND_TRANSFER / RELEASE_OWNERSHIP. */
static void grant_to(struct arts_db_cache_s *cache, unsigned int new_owner,
                     uint64_t master_v, bool has_next, const void *data,
                     uint64_t data_size) {
  /* Design plan §grant_to line 930-939: monotonic dedup.  Cache_s
   * persistence (route_table ref) ensures user_data stays valid until
   * destroy, so NO_DATA is safe when the receiver already has the
   * version. */
  uint64_t cur =
      arts_rank_u64_map_get(cache->home->last_sent_version, new_owner);
  if (cur >= master_v) {
    arts_coh_send_grant(new_owner, cache->db_guid, master_v, has_next, NULL, 0);
    return;
  }
  arts_rank_u64_map_set(cache->home->last_sent_version, new_owner, master_v);
  arts_coh_send_grant(new_owner, cache->db_guid, master_v, has_next, data,
                      data_size);
}

/* ===== Home-side handlers ========================================== */

void arts_coh_handle_lock_req(struct arts_remote_lock_req_packet_s *p) {
  unsigned int requester = p->header.rank;

  /* Stack-built OoO defer payload — heap-copied by home_lookup_or_defer
   * if it actually has to defer. */
  struct oo_coh_lock_req_s oo_payload = {
      .type = OO_COH_LOCK_REQ,
      .requester = requester,
      .db_guid = p->db_guid,
  };

  struct arts_db_cache_s *cache =
      home_lookup_or_defer(p->db_guid, requester, OO_COH_LOCK_REQ, &oo_payload,
                           sizeof(oo_payload), COH_REPLY_DESTROY_NOTIFY);
  if (cache == NULL) {
    return;
  }
  arts_pending_rw_enqueue(cache->home->pending_rw, requester);
  /* Active-directory invariant: AT MOST ONE INVALIDATE_NOTICE in flight
   * to the current rw_holder per ownership-transfer round.  CAS 0->1
   * gates the dispatch — only the thread that flips the bit sends.
   * Concurrent LOCK_REQs whose CAS loses simply piggyback on the
   * outstanding round; their requester is queued in pending_rw and is
   * served by the chain that the round's GRANT (with has_next=true)
   * triggers in the new owner's drain.  The flag is cleared by
   * handle_writeback (WB_AND_TRANSFER), handle_release_ownership, or
   * local_transfer_now once the round completes.  Replaces the pre-fix
   * `was_empty` heuristic which over-sent on concurrent enqueues. */
  if (arts_atomic_cswap(&cache->home->invalidate_in_flight, 0, 1) == 0) {
    arts_coh_send_invalidate_notice(cache->home->rw_holder, p->db_guid);
  }
}

void arts_coh_handle_get_data(struct arts_remote_get_data_packet_s *p) {
  unsigned int requester = p->header.rank;

  struct oo_coh_get_data_s oo_payload = {
      .type = OO_COH_GET_DATA,
      .requester = requester,
      .db_guid = p->db_guid,
      .waiter_addr = p->waiter_addr,
  };

  struct arts_db_cache_s *cache =
      home_lookup_or_defer(p->db_guid, requester, OO_COH_GET_DATA, &oo_payload,
                           sizeof(oo_payload), COH_REPLY_DESTROY_NOTIFY);
  if (cache == NULL) {
    return;
  }
  struct arts_db_buffer_s *master = arts_coherence_acquire_buf(cache);
  if (master == NULL) {
    /* Two cases produce master==NULL post-precheck:
     *   (a) Sentinel DB (db_size==0): no buffer is ever installed.
     *   (b) HOME_RECV pre-WRITEBACK: cross-rank create has happened but
     *       the creator's first WRITEBACK has not landed; we have a
     *       cache but no buffer (lazy install per OCR pattern).
     * Both cases: respond with version=0, NULL data.  The requester's
     * handle_data_response will deliver ptr=NULL to the parked RO waiter
     * (per spec, "value is undefined" before any writer publishes).
     * NOT a destroy condition -- the precheck above (destroy_state) is
     * authoritative for that. */
    arts_coh_send_data_response(requester, p->db_guid, /*version=*/0,
                                (void *)(uintptr_t)p->waiter_addr,
                                /*data=*/NULL, /*data_size=*/0);
    return;
  }
  uint64_t master_v = master->version;
  update_last_sent_max(cache, requester, master_v, master->data, cache->db_size,
                       (void *)(uintptr_t)p->waiter_addr);
  arts_coherence_release_buf(cache, master);
}

void arts_coh_handle_writeback(struct arts_remote_writeback_packet_s *p,
                               const void *data, uint64_t data_size) {
  unsigned int releaser = p->header.rank;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);

  /* Phase 2.2: defer-on-no-cache.  WRITEBACK can race ahead of DB_CREATE
   * on the home rank when the producer EDT releases very early; we must
   * (a) preserve the trailing data payload in the OoO entry and (b) ACK
   * the releaser immediately so its await_writeback_ack returns instead
   * of stalling.  When DB_CREATE finally arrives, fire_oo re-issues this
   * handler with the deferred buffer. */
  if (cache == NULL) {
    size_t total = sizeof(struct oo_coh_writeback_s) + data_size;
    struct oo_coh_writeback_s *oo_payload =
        (struct oo_coh_writeback_s *)arts_malloc(total);
    oo_payload->type = OO_COH_WRITEBACK;
    oo_payload->releaser = releaser;
    oo_payload->db_guid = p->db_guid;
    oo_payload->version = p->version;
    oo_payload->seq = p->seq;
    oo_payload->flag = p->flag;
    oo_payload->data_size = data_size;
    if (data_size > 0 && data != NULL) {
      memcpy(oo_payload->data, data, data_size);
    }
    oo_add_result_t r = arts_route_table_add_oo_ex(p->db_guid, oo_payload);
    if (r == OO_RESULT_FIRED_BY_DRAIN) {
      /* The OO_COH_* dispatcher inside fire_oo already re-issued this
       * handler on the now-installed cache; that re-issue path sent the
       * WB_ACK and installed the buffer.  We must not re-issue or free. */
      return;
    }
    if (r == OO_RESULT_AVAILABLE_NOW) {
      /* Race: data became non-NULL on the pre-push or push-time check;
       * payload was never inserted.  Drop our copy and fall through to
       * normal handling on the now-installed cache. */
      arts_free(oo_payload);
      cache = arts_coh_route_table_lookup_cache(p->db_guid);
      if (cache == NULL) {
        arts_coh_send_writeback_ack(releaser, p->db_guid, p->seq);
        return;
      }
      /* fall through to normal handling */
    } else {
      /* OO_RESULT_ENQUEUED — release ACK immediately so the releaser
       * doesn't stall on await_writeback_ack; the deferred OoO will
       * install the buffer when DB_CREATE arrives. */
      arts_coh_send_writeback_ack(releaser, p->db_guid, p->seq);
      return;
    }
  }

  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    arts_coh_send_writeback_ack(releaser, p->db_guid, p->seq);
    return;
  }
  arts_coherence_install_buffer(cache, p->version, data, data_size);
  arts_coh_send_writeback_ack(releaser, p->db_guid, p->seq);
  /* No pending_ro drain here: home's RO acquires hit case 1/3 and
   * never park.  Foreign ROs are served by GET_DATA, not by drain. */

  if (p->flag == ARTS_WB_AND_TRANSFER) {
    unsigned int new_owner;
    if (!arts_pending_rw_dequeue(cache->home->pending_rw, &new_owner)) {
      /* No queued waiter — home becomes the new owner.  Set the
       * sentinel so any subsequent foreign LOCK_REQ has a real
       * rw_holder to invalidate. */
      cache->writer_count = 1;
      cache->home->rw_holder = arts_global_rank_id;
      /* Round complete with no successor; clear the in-flight gate so
       * the next foreign LOCK_REQ can re-arm it. */
      arts_atomic_swap(&cache->home->invalidate_in_flight, 0);
      return;
    }
    cache->home->rw_holder = new_owner;
    bool has_next = !arts_pending_rw_empty(cache->home->pending_rw);
    struct arts_db_buffer_s *master = arts_coherence_acquire_buf(cache);
    if (master == NULL) {
      /* master==NULL has two distinct causes: destroy raced past
       * precheck, or sentinel DB (db_size==0) never installed a buffer.
       * Distinguish by destroy_state and emit either DESTROY_NOTIFY or
       * a no-data GRANT so the new owner's waiter still fires. */
      if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
        arts_coh_send_destroy_notify(new_owner, p->db_guid);
      } else {
        arts_coh_send_grant(new_owner, p->db_guid, /*version=*/0, has_next,
                            NULL, 0);
      }
    } else {
      grant_to(cache, new_owner, master->version, has_next, master->data,
               cache->db_size);
      arts_coherence_release_buf(cache, master);
    }
    /* Round complete: rw_holder has been advanced to new_owner and the
     * GRANT (or DESTROY_NOTIFY) has been dispatched.  Clear the
     * in-flight gate.  When has_next was true the GRANT itself encodes
     * the relay instruction — the new owner's drain
     * (drain_pending_rw_after_grant) withdraws the sentinel and
     * forwards ownership to the next queued requester via its own
     * release path.  We do NOT send a fresh INVALIDATE_NOTICE here:
     * the chain is self-driving from this point. */
    arts_atomic_swap(&cache->home->invalidate_in_flight, 0);
  }
}

void arts_coh_handle_release_ownership(
    struct arts_remote_release_ownership_packet_s *p) {
  /* RELEASE_OWNERSHIP is one-way; on destroy the silent drop is fine
   * (caller doesn't await any reply).  Phase 2.2: no OoO defer either —
   * RELEASE_OWNERSHIP only flows from a current owner whose acquire
   * implied DB_CREATE already landed at home, so cache==NULL here means
   * the DB was already torn down. */
  struct arts_db_cache_s *cache =
      home_lookup_or_defer(p->db_guid, p->header.rank,
                           OO_COH_LOCK_REQ /*unused*/, NULL, 0, COH_REPLY_NONE);
  if (cache == NULL) {
    return;
  }
  unsigned int new_owner;
  if (!arts_pending_rw_dequeue(cache->home->pending_rw, &new_owner)) {
    /* RELEASE_OWNERSHIP arrived with no queued waiter — home reclaims
     * ownership so a future foreign LOCK_REQ has a holder to invalidate. */
    cache->writer_count = 1;
    cache->home->rw_holder = arts_global_rank_id;
    /* Round complete with no successor; clear the in-flight gate. */
    arts_atomic_swap(&cache->home->invalidate_in_flight, 0);
    return;
  }
  cache->home->rw_holder = new_owner;
  bool has_next = !arts_pending_rw_empty(cache->home->pending_rw);
  struct arts_db_buffer_s *master = arts_coherence_acquire_buf(cache);
  if (master == NULL) {
    arts_coh_send_destroy_notify(new_owner, p->db_guid);
  } else {
    grant_to(cache, new_owner, master->version, has_next, master->data,
             cache->db_size);
    arts_coherence_release_buf(cache, master);
  }
  /* Round complete: rw_holder advanced, GRANT dispatched.  Clear the
   * in-flight gate.  When has_next was true the GRANT carries the
   * relay instruction; the new owner's drain self-forwards ownership
   * to the next queued requester. */
  arts_atomic_swap(&cache->home->invalidate_in_flight, 0);
}

void arts_coh_handle_destroy_req(struct arts_remote_destroy_req_packet_s *p) {
  /* Phase 2.2 / spec §4.11-4.12: home-side DESTROY_REQ.
   *
   * Symmetric with LOCK_REQ / GET_DATA / WRITEBACK: when the cache_s
   * has not yet been installed on home (DB_CREATE_COHERENT raced behind
   * the destroy), we must defer via the OoO list rather than treating
   * the NULL data slot as "already destroyed".  fire_oo on
   * DB_CREATE_COHERENT arrival re-issues this handler with the now-
   * installed cache.
   *
   * Fast path: cache is installed and reachable through item->data.  We
   * proceed with the legacy three-phase destroy (xchg data->NULL, drop
   * ooList, run the cache_s self-destroy protocol).  Order matters —
   *   [1] xchg item->data to NULL FIRST so new lookups can no longer
   *       enter (they observe NULL -> enqueue to ooList).
   *   [2] drop the OoO list (silent free; user-error path).
   *   [3] PIN/CXL DBs have no coherence_cache -> step 1 + arts_db_free.
   *   [4] For coherent DBs, run the cache_s self-destroy protocol:
   *       fail_trigger_pending wakes parked waiters;
   *       try_finalize_destroy single-flights the buffer detach. */
  unsigned int requester = p->header.rank;
  struct oo_coh_destroy_req_s oo_payload = {
      .type = OO_COH_DESTROY_REQ,
      .requester = requester,
      .db_guid = p->db_guid,
  };
  struct arts_db_cache_s *cache =
      home_lookup_or_defer(p->db_guid, requester, OO_COH_DESTROY_REQ,
                           &oo_payload, sizeof(oo_payload), COH_REPLY_NONE);
  if (cache == NULL) {
    /* Either cache had destroy_state != NONE (already torn down -
     * destroy is idempotent, nothing else to do) or the request was
     * deferred via OoO (will replay after DB_CREATE_COHERENT). */
    return;
  }

  /* Cache is live — perform the three-phase destroy.  Re-derive item +
   * db from the cache back-pointer so the xchg sees the same descriptor
   * the lookup observed. */
  struct arts_db_s *db = (struct arts_db_s *)cache->db_owner;
  arts_route_item_t *item = NULL;
  arts_route_table_reserve_or_lookup(p->db_guid, &item);
  if (item == NULL || db == NULL) {
    return;
  }

  /* [1] data NULL store first — block new lookups. */
  void *prev =
      atomic_exchange_explicit(&item->data, NULL, memory_order_acq_rel);
  if (prev == NULL) {
    return; /* concurrent destroyer already advanced past step 1. */
  }
  if (db->coherence_cache == NULL) {
    /* PIN/CXL — coherence-irrelevant.  But we should not have reached
     * this branch via the home_lookup_or_defer fast path: that helper
     * only returns a non-NULL cache, which implies db->coherence_cache
     * was set.  Treat as defensive guard for future PIN/CXL evolution
     * routing through this entry point. */
    arts_route_table_drop_oo(p->db_guid);
    arts_db_free(db);
    return;
  }
  /* cache already bound from home_lookup_or_defer above; the
   * db->coherence_cache pointer must match by construction (cache_s
   * back-pointer invariant).  Re-asserting via an assignment would
   * shadow the outer variable, so we just sanity-check identity. */
  /* [2] cleanup ooList memory (silent — destroy after enqueue is user
   * error and graceful-fail is out of scope per spec §4.7). */
  arts_route_table_drop_oo(p->db_guid);

  /* [3-5] cache_s self-destroy protocol — single-flight via destroy_state
   * CAS-gate, fail every parked waiter, hand finalize to the buffer/
   * pending_count drain. */
  if (arts_atomic_cswap(&cache->destroy_state, ARTS_DB_DESTROY_NONE,
                        ARTS_DB_DESTROY_MARKED) != ARTS_DB_DESTROY_NONE) {
    return; /* concurrent destroy raced — single-flight winner runs. */
  }

  /* Notify ranks with cached copies + queued LOCK_REQ requesters so
   * remote sharers wake up and observe DB_DESTROYED (legacy roster
   * build preserved from prior destroy_req — necessary for foreign-rank
   * progress, the spec's home-only protocol assumes this fan-out). */
  unsigned int self = arts_global_rank_id;
  unsigned int n = arts_global_rank_count;
  for (unsigned int r = 0; r < n; r++) {
    if (r == self) {
      continue;
    }
    if (arts_rank_u64_map_get(cache->home->last_sent_version, r) > 0) {
      arts_coh_send_destroy_notify(r, p->db_guid);
    }
  }
  unsigned int q_rank;
  while (arts_pending_rw_dequeue(cache->home->pending_rw, &q_rank)) {
    if (q_rank != self) {
      arts_coh_send_destroy_notify(q_rank, p->db_guid);
    }
  }

  arts_coh_fail_trigger_pending(cache);
  arts_coh_try_finalize_destroy(cache);
}

void arts_coh_handle_db_create_coherent(
    struct arts_remote_db_create_coherent_packet_s *p) {
  /* Home-side init for non-home creator.  Per coherence design plan
   * §968-988: install zero-init buffer, home struct with rw_holder =
   * creator_rank, writer_count = 0 (home is non-owner). */
  unsigned int creator_rank = (unsigned int)p->header.rank;
  arts_guid_t db_guid = p->db_guid;
  uint64_t db_size = p->db_size;

  /* Race against lazy_install or another path that already set up an
   * empty cache_s on this rank — coalesce by promoting the existing
   * lazy entry rather than allocating a duplicate. */
  struct arts_db_s *existing = (struct arts_db_s *)arts_route_table_lookup_db(
      db_guid, NULL, /*aquire=*/false);
  if (existing != NULL && existing->coherence_cache != NULL) {
    struct arts_db_cache_s *cache =
        (struct arts_db_cache_s *)existing->coherence_cache;
    if (cache->buffer == NULL && db_size > 0) {
      arts_coherence_install_buffer(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (cache->home == NULL) {
      cache->home = arts_db_home_create(creator_rank, arts_global_rank_count);
    } else {
      cache->home->rw_holder = creator_rank;
    }
    return;
  }

  /* No existing entry -- allocate stub + cache_s, install in route_table.
   *
   * Lazy buffer install (OCR pattern): HOME_RECV does NOT install a
   * buffer here.  cache->buffer stays NULL with version 0 -- "metadata
   * only" state.  The first WRITEBACK from the creator's release_rw
   * installs the buffer at home (version 1+, with the creator's
   * payload).  Cross-rank GET_DATA before that point is served as a
   * no-payload DATA_RESPONSE (handle_get_data); the requesting rank
   * sees ptr=NULL (per OCR spec ch2:832-839 "value of the created data
   * block is undefined" -- ARTS interprets this as "before any writer
   * has published, no data exists; reading is application's
   * responsibility"). */
  struct arts_db_s *stub =
      (struct arts_db_s *)arts_malloc_align(sizeof(struct arts_db_s), 16);
  memset(stub, 0, sizeof(struct arts_db_s));
  stub->header.type = ARTS_DB;
  stub->header.size = sizeof(struct arts_db_s);
  stub->guid = db_guid;
  stub->db_type = (arts_db_types_t)p->db_type;
  stub->coherence_cache = arts_coh_alloc_cache_s(
      db_guid, db_size, ARTS_COH_INIT_HOME_RECV, creator_rank);
  /* Phase 3.1: back-pointer for try_finalize_destroy direct-free. */
  ((struct arts_db_cache_s *)stub->coherence_cache)->db_owner = stub;

  if (arts_route_table_add_item_race(stub, db_guid, arts_global_rank_id,
                                     /*used=*/true)) {
    arts_route_table_fire_oo(db_guid, arts_out_of_order_handler);
    return;
  }

  /* Lost race — coalesce into the existing entry. */
  struct arts_db_cache_s *new_cache =
      (struct arts_db_cache_s *)stub->coherence_cache;
  arts_db_free(stub);
  struct arts_db_s *winner =
      (struct arts_db_s *)arts_route_table_lookup_db(db_guid, NULL, false);
  if (winner != NULL && winner->coherence_cache != NULL) {
    struct arts_db_cache_s *cache =
        (struct arts_db_cache_s *)winner->coherence_cache;
    if (cache->buffer == NULL && db_size > 0) {
      arts_coherence_install_buffer(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (cache->home == NULL) {
      cache->home = arts_db_home_create(creator_rank, arts_global_rank_count);
    } else {
      cache->home->rw_holder = creator_rank;
    }
  }
  (void)new_cache;
}

/* ===== Sharer-side response handlers =============================== */

void arts_coh_handle_grant(struct arts_remote_grant_packet_s *p,
                           const void *data, uint64_t data_size) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    return;
  }
  if (p->data_present) {
    arts_coherence_install_buffer(cache, p->version, data, data_size);
  }
  /* Always install sentinel = 1; post-drain withdraw if has_next. */
  cache->writer_count = 1;
  cache->lock_req_in_flight = 0;
  /* Drain pending_rw — pop every queued waiter in FIFO order via the
   * Vyukov MPSC consumer path.  Implementation lives in the acquire
   * module (B4). */
  arts_coh_drain_pending_rw_after_grant(cache, p->version, p->has_next != 0);
  /* Drain pending_ro: any RO whose target_version is now satisfied. */
  arts_coh_drain_pending_ro(cache, p->version);
}

void arts_coh_handle_data_response(struct arts_remote_data_response_packet_s *p,
                                   const void *data, uint64_t data_size) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    return;
  }
  if (p->data_present) {
    arts_coherence_install_buffer(cache, p->version, data, data_size);
  }
  struct arts_db_ro_waiter_s *w =
      (struct arts_db_ro_waiter_s *)(uintptr_t)p->waiter_addr;
  arts_coh_trigger_ro_waiter(cache, w, p->version);
  if (p->data_present) {
    arts_coh_drain_pending_ro(cache, p->version);
  }
}

void arts_coh_handle_invalidate_notice(
    struct arts_remote_invalidate_notice_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    return;
  }
  /* Plain sentinel withdrawal: writer_count -= 1.
   *
   * Architectural invariant (post-fix): home's invalidate_in_flight gate
   * guarantees AT MOST ONE INVALIDATE_NOTICE is in flight to this rank
   * per ownership-transfer round.  Because home only sends INVALIDATE
   * after rw_holder is set to a rank that holds the sentinel +1, the
   * holder's writer_count is always >= 1 when the notice arrives.
   * Underflow is therefore impossible by construction.
   *
   * If wc==0 ever observed here, it is a real bug (lost release-notice
   * pair, double-INVALIDATE, etc.) — investigate, do not paper over.
   *
   * The thread whose fetch_sub returns 1 (post-decrement value 0) is
   * the unique transfer actor.  Otherwise (rest > 0), the last local
   * writer's release will see rest=0 in release_rw and drive the
   * transfer instead. */
  unsigned int rest = arts_atomic_sub(&cache->writer_count, 1);
  if (rest == 0) {
    extern void arts_coh_invalidate_transfer(struct arts_db_cache_s * cache);
    arts_coh_invalidate_transfer(cache);
  }
}

void arts_coh_handle_writeback_ack(
    struct arts_remote_writeback_ack_packet_s *p) {
  /* Wakes any release_rw thread parked on await-WRITEBACK_ACK whose
   * outstanding seq <= p->seq.  Implementation in coherence_release.c. */
  extern void arts_coh_writeback_ack_signal(arts_guid_t db_guid, uint64_t seq);
  arts_coh_writeback_ack_signal(p->db_guid, p->seq);
}

void arts_coh_handle_destroy_notify(
    struct arts_remote_destroy_notify_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return; /* already torn down on this rank. */
  }
  if (arts_atomic_cswap(&cache->destroy_state, ARTS_DB_DESTROY_NONE,
                        ARTS_DB_DESTROY_MARKED) != ARTS_DB_DESTROY_NONE) {
    return; /* already marked; idempotent. */
  }
  arts_coh_fail_trigger_pending(cache);
  arts_coh_try_finalize_destroy(cache);
}
