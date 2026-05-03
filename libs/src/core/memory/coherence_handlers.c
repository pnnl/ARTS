/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message handlers and senders.
 *
 * The protocol's drop-discipline (lookup → destroy_state precheck →
 * explicit wake-up reply on failure) is implemented via the
 * `home_lookup_or_notify` helper at the top of this file; every
 * home-side handler entry funnels through it.  Sharer-side response
 * handlers do their own lookup + cache.destroy_state check inline
 * (no requester to notify back — the message *is* the requester's
 * own context).
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

/* ===== home_lookup_or_notify helper ================================= */

/* Reply kind selector for the helper.  Mirrors the design plan's
 * REPLY_DESTROY_NOTIFY / REPLY_WB_ACK / REPLY_NONE. */
typedef enum {
  COH_REPLY_NONE = 0,
  COH_REPLY_DESTROY_NOTIFY,
  COH_REPLY_WB_ACK,
} coh_reply_kind_t;

/* Look up cache by guid, atomically check destroy_state, and emit
 * the appropriate wake-up reply if either lookup fails or destroy is
 * non-NONE.  On success: returns the cache pointer; caller MUST call
 * arts_route_table_return_db when done.  On failure: returns NULL,
 * route_table ref already dropped (or never taken). */
static struct arts_db_cache_s *home_lookup_or_notify(arts_guid_t guid,
                                                     unsigned int requester,
                                                     coh_reply_kind_t kind) {
  /* helper handles DESTROY_NOTIFY / NONE replies; WRITEBACK_ACK
   * needs the per-message seq and is wired inline in handle_writeback. */
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(guid);
  if (cache == NULL) {
    if (kind == COH_REPLY_DESTROY_NOTIFY) {
      arts_coh_send_destroy_notify(requester, guid);
    }
    return NULL;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    if (kind == COH_REPLY_DESTROY_NOTIFY) {
      arts_coh_send_destroy_notify(requester, guid);
    }
    arts_route_table_return_db(guid, false);
    return NULL;
  }
  return cache;
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
  struct arts_db_cache_s *cache =
      home_lookup_or_notify(p->db_guid, requester, COH_REPLY_DESTROY_NOTIFY);
  if (cache == NULL) {
    return;
  }
  bool was_empty = arts_pending_rw_empty(cache->home->pending_rw);
  arts_pending_rw_enqueue(cache->home->pending_rw, requester);
  if (was_empty) {
    arts_coh_send_invalidate_notice(cache->home->rw_holder, p->db_guid);
  }
  arts_route_table_return_db(p->db_guid, false);
}

void arts_coh_handle_get_data(struct arts_remote_get_data_packet_s *p) {
  unsigned int requester = p->header.rank;
  struct arts_db_cache_s *cache =
      home_lookup_or_notify(p->db_guid, requester, COH_REPLY_DESTROY_NOTIFY);
  if (cache == NULL) {
    return;
  }
  struct arts_db_buffer_s *master = arts_coherence_acquire_buf(cache);
  if (master == NULL) {
    /* destroy raced past precheck. */
    arts_coh_send_destroy_notify(requester, p->db_guid);
    arts_route_table_return_db(p->db_guid, false);
    return;
  }
  uint64_t master_v = master->version;
  update_last_sent_max(cache, requester, master_v, master->data, cache->db_size,
                       (void *)(uintptr_t)p->waiter_addr);
  arts_coherence_release_buf(cache, master);
  arts_route_table_return_db(p->db_guid, false);
}

void arts_coh_handle_writeback(struct arts_remote_writeback_packet_s *p,
                               const void *data, uint64_t data_size) {
  unsigned int releaser = p->header.rank;
  /* Inline lookup + destroy precheck: WRITEBACK's drop reply needs
   * the seq from the request, so we don't go through the helper. */
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    arts_coh_send_writeback_ack(releaser, p->db_guid, p->seq);
    return;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    arts_coh_send_writeback_ack(releaser, p->db_guid, p->seq);
    arts_route_table_return_db(p->db_guid, false);
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
      arts_route_table_return_db(p->db_guid, false);
      return;
    }
    cache->home->rw_holder = new_owner;
    bool has_next = !arts_pending_rw_empty(cache->home->pending_rw);
    struct arts_db_buffer_s *master = arts_coherence_acquire_buf(cache);
    if (master == NULL) {
      /* destroy raced; notify the new_owner so its parked waiter wakes. */
      arts_coh_send_destroy_notify(new_owner, p->db_guid);
    } else {
      grant_to(cache, new_owner, master->version, has_next, master->data,
               cache->db_size);
      arts_coherence_release_buf(cache, master);
    }
  }
  arts_route_table_return_db(p->db_guid, false);
}

void arts_coh_handle_release_ownership(
    struct arts_remote_release_ownership_packet_s *p) {
  /* RELEASE_OWNERSHIP is one-way; on destroy the silent drop is
   * fine (caller doesn't await any reply). */
  struct arts_db_cache_s *cache =
      home_lookup_or_notify(p->db_guid, p->header.rank, COH_REPLY_NONE);
  if (cache == NULL) {
    return;
  }
  unsigned int new_owner;
  if (!arts_pending_rw_dequeue(cache->home->pending_rw, &new_owner)) {
    /* RELEASE_OWNERSHIP arrived with no queued waiter — home reclaims
     * ownership so a future foreign LOCK_REQ has a holder to invalidate. */
    cache->writer_count = 1;
    cache->home->rw_holder = arts_global_rank_id;
    arts_route_table_return_db(p->db_guid, false);
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
  arts_route_table_return_db(p->db_guid, false);
}

void arts_coh_handle_destroy_req(struct arts_remote_destroy_req_packet_s *p) {
  /* DESTROY_REQ is forwarded to home from any caller (uniform path).
   * Idempotent via destroy_state CAS-gate. */
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return; /* already torn down or never existed. */
  }
  if (arts_atomic_cswap(&cache->destroy_state, ARTS_DB_DESTROY_NONE,
                        ARTS_DB_DESTROY_MARKED) != ARTS_DB_DESTROY_NONE) {
    arts_route_table_return_db(p->db_guid, false);
    return; /* already destroying. */
  }

  /* Build the destroy roster: ranks with cached copies + ranks with
   * outstanding LOCK_REQs.  A rank in the latter but not the former
   * (sent LOCK_REQ but never received GRANT) still needs notification. */
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
  arts_route_table_return_db(p->db_guid, false);
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
      cache->home =
          arts_db_home_create(creator_rank, arts_global_rank_count);
    } else {
      cache->home->rw_holder = creator_rank;
    }
    arts_route_table_return_db(db_guid, false);
    return;
  }

  /* No existing entry — allocate stub + cache_s, install in route_table. */
  struct arts_db_s *stub =
      (struct arts_db_s *)arts_malloc_align(sizeof(struct arts_db_s), 16);
  memset(stub, 0, sizeof(struct arts_db_s));
  stub->header.type = ARTS_DB;
  stub->header.size = sizeof(struct arts_db_s);
  stub->guid = db_guid;
  stub->db_type = (arts_db_types_t)p->db_type;
  stub->copy_count = 1;
  stub->coherence_cache = arts_coh_alloc_cache_s(
      db_guid, db_size, ARTS_COH_INIT_HOME_RECV, creator_rank);

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
      cache->home =
          arts_db_home_create(creator_rank, arts_global_rank_count);
    } else {
      cache->home->rw_holder = creator_rank;
    }
    arts_route_table_return_db(db_guid, false);
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
    arts_route_table_return_db(p->db_guid, false);
    return;
  }
  if (p->data_present) {
    arts_coherence_install_buffer(cache, p->version, data, data_size);
  }
  /* Always install sentinel = 1; post-drain withdraw if has_next. */
  cache->writer_count = 1;
  cache->lock_req_in_flight = 0;
  /* Drain pending_rw — visible waiters are claimed via the marked-list
   * traversal.  Implementation lives in the acquire module (B4). */
  arts_coh_drain_pending_rw_after_grant(cache, p->version, p->has_next != 0);
  /* Drain pending_ro: any RO whose target_version is now satisfied. */
  arts_coh_drain_pending_ro(cache, p->version);
  arts_route_table_return_db(p->db_guid, false);
}

void arts_coh_handle_data_response(struct arts_remote_data_response_packet_s *p,
                                   const void *data, uint64_t data_size) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    arts_route_table_return_db(p->db_guid, false);
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
  arts_route_table_return_db(p->db_guid, false);
}

void arts_coh_handle_invalidate_notice(
    struct arts_remote_invalidate_notice_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    arts_route_table_return_db(p->db_guid, false);
    return;
  }
  unsigned int rest = arts_atomic_sub(&cache->writer_count, 1);
  if (rest == 0) {
    /* Sentinel went to 0 with no active writers — we are the
     * transfer actor.  Implementation in B5 (release path) supplies
     * local_transfer / send WRITEBACK_AND_TRANSFER. */
    extern void arts_coh_invalidate_transfer(struct arts_db_cache_s * cache);
    arts_coh_invalidate_transfer(cache);
  }
  /* rest != 0: per design, the INVALIDATE_NOTICE fetch_sub withdrew the
   * sentinel.  The last local writer's release will see rest=0 and trigger
   * transfer.  No invalidate_pending flag — design plan line 1018-1028. */
  arts_route_table_return_db(p->db_guid, false);
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
    arts_route_table_return_db(p->db_guid, false);
    return; /* already marked; idempotent. */
  }
  arts_coh_fail_trigger_pending(cache);
  arts_coh_try_finalize_destroy(cache);
  arts_route_table_return_db(p->db_guid, false);
}
