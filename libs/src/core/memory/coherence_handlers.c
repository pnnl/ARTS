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
 * whose destination is the local rank (self_send_check rejects). The RC
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
                                     arts_guid_t db_guid,
                                     unsigned int new_owner_rank) {
  struct arts_remote_invalidate_notice_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p),
                          ARTS_REMOTE_INVALIDATE_NOTICE_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.new_owner_rank = new_owner_rank;
  memset(p.pad, 0, sizeof(p.pad));
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

#ifdef ARTS_MEMORY_MODEL_LRC

/* Forward declarations for handlers called inline (self-transfer fast path)
 * and for acquire/destroy helpers defined in other translation units. */
void arts_coh_handle_transfer_ownership(void *payload, size_t size);
void arts_coh_handle_install_ack(struct arts_remote_install_ack_packet_s *p);
void arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                           uint64_t version, bool has_next);
void arts_coh_drain_pending_ro(struct arts_db_cache_s *cache, uint64_t version);
void arts_coh_try_finalize_destroy(struct arts_db_cache_s *cache);

/* ===== LRC sender helpers (TRANSFER_OWNERSHIP, INSTALL_ACK) ============ */

void arts_coh_send_transfer_ownership(unsigned int new_owner_rank,
                                      arts_guid_t db_guid, uint64_t version,
                                      const void *map_buf, size_t map_size,
                                      const void *data, size_t data_size) {
  struct arts_remote_transfer_ownership_packet_s hdr;
  uint32_t entry_count = (map_buf != NULL && map_size >= sizeof(uint32_t) * 2)
                             ? ((const uint32_t *)map_buf)[0]
                             : 0u;
  uint64_t total =
      (uint64_t)sizeof(hdr) + (uint64_t)map_size + (uint64_t)data_size;
  arts_fill_packet_header(&hdr.header, total,
                          ARTS_REMOTE_TRANSFER_OWNERSHIP_MSG);
  hdr.header.rank = arts_global_rank_id;
  hdr.db_guid = db_guid;
  hdr.version = version;
  hdr.map_entry_count = entry_count;
  hdr.pad = 0;

  if (new_owner_rank == arts_global_rank_id) {
    /* Self-transfer: construct a contiguous buffer and call handler inline. */
    void *buf = arts_malloc((size_t)total);
    memcpy(buf, &hdr, sizeof(hdr));
    if (map_buf != NULL && map_size > 0) {
      memcpy((char *)buf + sizeof(hdr), map_buf, map_size);
    }
    if (data != NULL && data_size > 0) {
      memcpy((char *)buf + sizeof(hdr) + map_size, data, data_size);
    }
    arts_coh_handle_transfer_ownership(buf, (size_t)total);
    arts_free(buf);
    return;
  }

  /* Remote path: send header + optional map payload + optional data payload.
   * Use the two-part payload helper so we avoid an extra memcpy. */
  if ((map_size > 0 || data_size > 0) && (map_buf != NULL || data != NULL)) {
    /* Combine map + data into one contiguous payload for the send helper. */
    size_t payload_size = map_size + data_size;
    void *payload = arts_malloc(payload_size);
    if (map_buf != NULL && map_size > 0) {
      memcpy(payload, map_buf, map_size);
    }
    if (data != NULL && data_size > 0) {
      memcpy((char *)payload + map_size, data, data_size);
    }
    arts_remote_send_request_payload_async_free(
        (int)new_owner_rank, (char *)&hdr, sizeof(hdr), (char *)payload,
        /*offset=*/0, (uint64_t)payload_size, arts_free);
  } else {
    arts_remote_send_request_async((int)new_owner_rank, (char *)&hdr,
                                   sizeof(hdr));
  }
}

void arts_coh_send_install_ack(unsigned int home_rank, arts_guid_t db_guid,
                               uint64_t version) {
  struct arts_remote_install_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), ARTS_REMOTE_INSTALL_ACK_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  if (home_rank == arts_global_rank_id) {
    arts_coh_handle_install_ack(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

/* ===== LRC ship_transfer / start_invalidate_round ==================== */

void arts_coh_lrc_ship_transfer(struct arts_db_cache_s *cache) {
  unsigned int new_owner = cache->incoming_new_owner;
  struct arts_db_buffer_s *buf = arts_coherence_acquire_buf(cache);
  if (buf == NULL) {
    /* Sentinel DB (db_size==0) or pre-publication: send empty transfer. */
    arts_coh_send_transfer_ownership(new_owner, cache->db_guid,
                                     /*version=*/0,
                                     /*map_buf=*/NULL, /*map_size=*/0,
                                     /*data=*/NULL, /*data_size=*/0);
    return;
  }

  /* Serialize the owner-side dedup map for transfer. */
  size_t map_max =
      sizeof(uint32_t) * 2 + (size_t)arts_global_rank_count *
                                 sizeof(struct arts_remote_rank_version_pair_s);
  void *map_buf = arts_malloc(map_max);
  size_t map_size;
  if (cache->last_sent_version != NULL) {
    map_size = arts_rank_u64_map_serialize(cache->last_sent_version, map_buf);
  } else {
    /* No map yet: emit an empty map (count=0, pad=0). */
    uint32_t *p = (uint32_t *)map_buf;
    p[0] = 0u;
    p[1] = 0u;
    map_size = sizeof(uint32_t) * 2;
  }

  arts_coh_send_transfer_ownership(new_owner, cache->db_guid, buf->version,
                                   map_buf, map_size, buf->data,
                                   cache->db_size);
  arts_free(map_buf);

  /* Release buffer ref (sentinel was withdrawn by INVALIDATE_NOTICE
   * handler's atomic_sub); the buffer recycles once all readers drop. */
  arts_coherence_release_buf(cache, buf);

  /* Destroy local ownership-tracking map: new owner will reconstruct. */
  if (cache->last_sent_version != NULL) {
    arts_rank_u64_map_destroy(cache->last_sent_version);
    cache->last_sent_version = NULL;
  }
}

void arts_coh_lrc_start_invalidate_round(struct arts_db_cache_s *cache,
                                         unsigned int new_owner) {
  unsigned int current_owner =
      atomic_load_explicit(&cache->home->rw_holder, memory_order_acquire);
  arts_coh_send_invalidate_notice(current_owner, cache->db_guid, new_owner);
}

/* ===== LRC TRANSFER_OWNERSHIP handler (new owner C) =================== */

void arts_coh_handle_transfer_ownership(void *payload, size_t size) {
  struct arts_remote_transfer_ownership_packet_s *hdr =
      (struct arts_remote_transfer_ownership_packet_s *)payload;
  arts_guid_t db_guid = hdr->db_guid;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(db_guid);
  if (cache == NULL) {
    cache = arts_coh_lazy_install_cache_s(db_guid, /*db_size=*/0);
    if (cache == NULL) {
      return; /* DB destroyed before we became owner — drop. */
    }
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    return;
  }

  /* Wire layout: header | map (count pairs) | data bytes */
  char *map_start = (char *)payload + sizeof(*hdr);
  size_t map_size =
      sizeof(uint32_t) * 2 + (size_t)hdr->map_entry_count *
                                 sizeof(struct arts_remote_rank_version_pair_s);
  char *data_start = map_start + map_size;
  size_t data_size = size - sizeof(*hdr) - map_size;

  /* Reconstruct the owner-side dedup map so this rank can skip
   * redundant DATA_RESPONSE sends to readers that already hold a
   * sufficiently fresh copy. */
  if (cache->last_sent_version != NULL) {
    arts_rank_u64_map_destroy(cache->last_sent_version);
  }
  cache->last_sent_version = arts_rank_u64_map_deserialize(
      map_start, map_size, arts_global_rank_count);

  /* Install the transferred buffer. */
  if (data_size > 0) {
    arts_coherence_install_buffer(cache, hdr->version, data_start, data_size);
    if (cache->db_size == 0) {
      cache->db_size = data_size;
    }
  }

  /* Install ownership sentinel: writer_count = 1. */
  arts_atomic_swap(&cache->writer_count, 1u);
  /* Clear lock_req_in_flight so subsequent RW acquires can kick new
   * LOCK_REQ rounds if needed. */
  cache->lock_req_in_flight = 0;

  /* Drain local RW waiters and RO waiters that were parked before we
   * held the buffer. */
  arts_coh_drain_pending_rw_after_grant(cache, hdr->version,
                                        /*has_next=*/false);
  arts_coh_drain_pending_ro(cache, hdr->version);

  /* Confirm installation to home so home can update rw_holder and
   * serve pending_ro_forwards. */
  unsigned int home_rank = (unsigned int)arts_guid_get_rank(db_guid);
  arts_coh_send_install_ack(home_rank, db_guid, hdr->version);
}

/* ===== LRC INSTALL_ACK handler (home A) =============================== */

void arts_coh_handle_install_ack(struct arts_remote_install_ack_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    return;
  }

  /* Publish the new rw_holder (visible to GET_DATA redirect path). */
  unsigned int new_owner = cache->home->pending_install_owner;
  atomic_store_explicit(&cache->home->rw_holder, new_owner,
                        memory_order_release);

  /* Forward any RO acquires that were deferred while the transfer was
   * in progress. */
  arts_coh_drain_pending_ro_forwards(cache);

  /* Drain-or-release retry loop: try to start the next transfer round
   * if there are pending_rw requests, otherwise release the baton. */
  while (1) {
    unsigned int next_owner;
    if (arts_home_lockreq_queue_pop(&cache->home->pending_rw, &next_owner)) {
      cache->home->pending_install_owner = next_owner;
      arts_coh_lrc_start_invalidate_round(cache, next_owner);
      return;
    }
    /* No pending requester — release the baton. */
    atomic_store_explicit(&cache->home->invalidate_in_flight, 0u,
                          memory_order_release);
    /* Re-check for a freshly-enqueued requester that raced the baton
     * release.  If the queue is still empty, we're done. */
    if (arts_home_lockreq_queue_empty(&cache->home->pending_rw)) {
      return;
    }
    /* There is a new requester; try to re-acquire the baton. */
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &cache->home->invalidate_in_flight, &expected, 1u,
            memory_order_acq_rel, memory_order_acquire)) {
      /* Another LOCK_REQ handler already picked up the baton (race);
       * that thread will drain the queue. */
      return;
    }
    /* Re-acquired the baton; loop to pop and start the next round. */
  }
}

void arts_coh_send_redirect_ro(unsigned int owner_rank, arts_guid_t db_guid,
                               unsigned int requester_rank, void *waiter_addr) {
  struct arts_remote_redirect_ro_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), ARTS_REMOTE_REDIRECT_RO_MSG);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.requester_rank = requester_rank;
  p.pad = 0;
  p.waiter_addr = (uint64_t)(uintptr_t)waiter_addr;
  if (owner_rank == arts_global_rank_id) {
    arts_coh_handle_redirect_ro(&p);
    return;
  }
  arts_remote_send_request_async((int)owner_rank, (char *)&p, sizeof(p));
}
#endif /* ARTS_MEMORY_MODEL_LRC */

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

/* ===== home_lookup_or_defer helper ============ */

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

/* ===== home.last_sent_version atomic-monotonic helpers (RC only) ==== */
#ifndef ARTS_MEMORY_MODEL_LRC

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
  /* Monotonic dedup — if the requester already received this version
   * (cur >= master_v), send NO_DATA.  Cache_s lifetime invariant
   * guarantees user_data persists until destroy (route_table ref). */
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
  /* Monotonic dedup.  Cache_s persistence (route_table ref) ensures
   * user_data stays valid until destroy, so NO_DATA is safe when the
   * receiver already has the version. */
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
#endif /* !ARTS_MEMORY_MODEL_LRC */

/* ===== Home-side handlers ========================================== */

void arts_coh_handle_lock_req(struct arts_remote_lock_req_packet_s *p) {
  /* LOCK_REQ is only sent in RC and LRC builds (LC routes all acquires
   * through GET_DATA / DATA_RESPONSE instead of LOCK_REQ / GRANT).  The
   * handler is still compiled in LC to satisfy the wire dispatcher table,
   * but it will never be called at runtime. */
#ifndef ARTS_MEMORY_MODEL_LC
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
  arts_home_lockreq_queue_push(&cache->home->pending_rw, requester);

#ifdef ARTS_MEMORY_MODEL_LRC
  /* LRC: guard against the narrow window between destroy_state CAS
   * (NONE→MARKED) and the destroy fan-out that drains pending_rw.
   * If destroy_in_flight is already set the LOCK_REQ requester will
   * never receive a GRANT; send DESTROY_NOTIFY so the requester can
   * surface ARTS_DB_DESTROYED to its EDT. */
  if (atomic_load_explicit(&cache->home->destroy_in_flight,
                           memory_order_acquire) != 0) {
    /* Note: the requester was just pushed to pending_rw.  The destroy
     * fan-out path (handle_destroy_req) drains pending_rw after setting
     * destroy_in_flight; if it already ran we must notify here.  If it
     * has not yet drained, our push is harmless — the drain will pick it
     * up and send DESTROY_NOTIFY then.  Send unconditionally here to
     * cover the "drain already finished before our push" race. */
    arts_coh_send_destroy_notify(requester, p->db_guid);
    return;
  }
#endif /* ARTS_MEMORY_MODEL_LRC */

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
  unsigned int iif_zero = 0;
  if (!atomic_compare_exchange_strong_explicit(
          &cache->home->invalidate_in_flight, &iif_zero, 1u,
          memory_order_acq_rel, memory_order_acquire)) {
    return; /* another round is in flight; requester stays queued */
  }
#ifdef ARTS_MEMORY_MODEL_LRC
  /* LRC: pop the OLDEST requester (FIFO) to be the transfer target and
   * embed its rank in the INVALIDATE_NOTICE so the current holder ships
   * TRANSFER_OWNERSHIP directly, without a home round-trip.
   * pending_install_owner is only written by the baton holder (single
   * writer invariant), so no atomic needed. */
  unsigned int next_owner;
  if (!arts_home_lockreq_queue_pop(&cache->home->pending_rw, &next_owner)) {
    /* Defensive: we just pushed, so empty is impossible under correct
     * usage.  Release the baton and return. */
    atomic_store_explicit(&cache->home->invalidate_in_flight, 0u,
                          memory_order_release);
    return;
  }
  cache->home->pending_install_owner = next_owner;
  arts_coh_lrc_start_invalidate_round(cache, next_owner);
#else  /* RC */
  arts_coh_send_invalidate_notice(
      atomic_load_explicit(&cache->home->rw_holder, memory_order_acquire),
      p->db_guid, /*new_owner_rank=*/0u);
#endif /* ARTS_MEMORY_MODEL_LRC */
#else  /* ARTS_MEMORY_MODEL_LC */
  (void)p; /* LC: LOCK_REQ is never sent; stub for completeness. */
#endif /* !ARTS_MEMORY_MODEL_LC */
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

#ifdef ARTS_MEMORY_MODEL_LRC
  /* LRC home-side RO routing.
   *
   * Home does not hold the canonical data copy under LRC — the current
   * owner does.  Home's job is to redirect the requester to the owner
   * (via REDIRECT_RO) so the owner can send DATA_RESPONSE directly,
   * applying the owner-side last_sent_version dedup.
   *
   * Gate ordering (must be stable): check destroy_in_flight BEFORE
   * recording the requester in the readers set; if destroy is already
   * in flight the requester should receive DESTROY_NOTIFY and NOT be
   * added to the readers roster (it will never receive data).  Once
   * recorded, check invalidate_in_flight: if an ownership transfer
   * round is in progress, defer to pending_ro_forwards; the drain that
   * follows INSTALL_ACK will re-issue the redirect to the new owner. */
  if (atomic_load_explicit(&cache->home->destroy_in_flight,
                           memory_order_acquire) != 0) {
    arts_coh_send_destroy_notify(requester, cache->db_guid);
    return;
  }
  arts_readers_bits_set(&cache->home->readers, requester);
  if (atomic_load_explicit(&cache->home->invalidate_in_flight,
                           memory_order_acquire) != 0) {
    arts_home_pending_ro_queue_push(&cache->home->pending_ro_forwards,
                                    requester,
                                    (void *)(uintptr_t)p->waiter_addr);
    return;
  }
  unsigned int owner =
      atomic_load_explicit(&cache->home->rw_holder, memory_order_acquire);
  arts_coh_send_redirect_ro(owner, cache->db_guid, requester,
                            (void *)(uintptr_t)p->waiter_addr);
#else  /* !ARTS_MEMORY_MODEL_LRC */
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
#endif /* ARTS_MEMORY_MODEL_LRC */
}

void arts_coh_handle_writeback(struct arts_remote_writeback_packet_s *p,
                               const void *data, uint64_t data_size) {
  unsigned int releaser = p->header.rank;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);

  /* defer-on-no-cache.  WRITEBACK can race ahead of DB_CREATE
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

  /* WB_AND_TRANSFER: ownership-chain relay.  LC uses WRITEBACK_NORMAL only
   * (no exclusive owner to transfer to), so this branch is RC/LRC-only. */
#ifndef ARTS_MEMORY_MODEL_LC
  if (p->flag == ARTS_WB_AND_TRANSFER) {
    unsigned int new_owner;
    if (!arts_home_lockreq_queue_pop(&cache->home->pending_rw, &new_owner)) {
      /* No queued waiter — home becomes the new owner.  Set the
       * sentinel so any subsequent foreign LOCK_REQ has a real
       * rw_holder to invalidate. */
      cache->writer_count = 1;
      atomic_store_explicit(&cache->home->rw_holder, arts_global_rank_id,
                            memory_order_release);
      /* Round complete with no successor; clear the in-flight gate so
       * the next foreign LOCK_REQ can re-arm it. */
      atomic_store_explicit(&cache->home->invalidate_in_flight, 0,
                            memory_order_release);
      return;
    }
    atomic_store_explicit(&cache->home->rw_holder, new_owner,
                          memory_order_release);
    bool has_next = !arts_home_lockreq_queue_empty(&cache->home->pending_rw);
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
#ifndef ARTS_MEMORY_MODEL_LRC
      grant_to(cache, new_owner, master->version, has_next, master->data,
               cache->db_size);
#else
      /* LRC: grant without home-side dedup (owner-side dedup via
       * cache->last_sent_version handles redundant sends). */
      arts_coh_send_grant(new_owner, cache->db_guid, master->version, has_next,
                          master->data, cache->db_size);
#endif
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
    atomic_store_explicit(&cache->home->invalidate_in_flight, 0,
                          memory_order_release);
  }
#endif /* !ARTS_MEMORY_MODEL_LC */
}

void arts_coh_handle_release_ownership(
    struct arts_remote_release_ownership_packet_s *p) {
  /* RELEASE_OWNERSHIP is only sent in RC and LRC builds.  In LC there is
   * no exclusive ownership chain, so this handler is never called at
   * runtime.  Stub it for LC to satisfy the dispatcher table. */
#ifndef ARTS_MEMORY_MODEL_LC
  /* RELEASE_OWNERSHIP is one-way; on destroy the silent drop is fine
   * (caller doesn't await any reply).  no OoO defer either —
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
  if (!arts_home_lockreq_queue_pop(&cache->home->pending_rw, &new_owner)) {
    /* RELEASE_OWNERSHIP arrived with no queued waiter — home reclaims
     * ownership so a future foreign LOCK_REQ has a holder to invalidate. */
    cache->writer_count = 1;
    atomic_store_explicit(&cache->home->rw_holder, arts_global_rank_id,
                          memory_order_release);
    /* Round complete with no successor; clear the in-flight gate. */
    atomic_store_explicit(&cache->home->invalidate_in_flight, 0,
                          memory_order_release);
    return;
  }
  atomic_store_explicit(&cache->home->rw_holder, new_owner,
                        memory_order_release);
  bool has_next = !arts_home_lockreq_queue_empty(&cache->home->pending_rw);
  struct arts_db_buffer_s *master = arts_coherence_acquire_buf(cache);
  if (master == NULL) {
    arts_coh_send_destroy_notify(new_owner, p->db_guid);
  } else {
#ifndef ARTS_MEMORY_MODEL_LRC
    grant_to(cache, new_owner, master->version, has_next, master->data,
             cache->db_size);
#else
    /* LRC: grant without home-side dedup (owner-side dedup via
     * cache->last_sent_version handles redundant sends). */
    arts_coh_send_grant(new_owner, cache->db_guid, master->version, has_next,
                        master->data, cache->db_size);
#endif
    arts_coherence_release_buf(cache, master);
  }
  /* Round complete: rw_holder advanced, GRANT dispatched.  Clear the
   * in-flight gate.  When has_next was true the GRANT carries the
   * relay instruction; the new owner's drain self-forwards ownership
   * to the next queued requester. */
  atomic_store_explicit(&cache->home->invalidate_in_flight, 0,
                        memory_order_release);
#else
  (void)p; /* LC: RELEASE_OWNERSHIP is never sent. */
#endif /* !ARTS_MEMORY_MODEL_LC */
}

#ifdef ARTS_MEMORY_MODEL_LRC
/* Fan-out callback for arts_readers_bits_for_each during destroy.
 * ctx carries the db_guid encoded as uintptr_t (no heap allocation
 * needed since the callback is synchronous). */
static void lrc_destroy_fanout_cb(unsigned int rank, void *ctx) {
  arts_guid_t db_guid = (arts_guid_t)(uintptr_t)ctx;
  unsigned int self = arts_global_rank_id;
  if (rank != self) {
    arts_coh_send_destroy_notify(rank, db_guid);
  }
}
#endif /* ARTS_MEMORY_MODEL_LRC */

void arts_coh_handle_destroy_req(struct arts_remote_destroy_req_packet_s *p) {
  /* Spec §4.11-4.12: home-side DESTROY_REQ.
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
  /* [1.5] set DELETE bit so subsequent acquire_item / add_oo_ex observe
   * destroyed state and short-circuit (lookup → NULL, add_oo →
   * AVAILABLE_NOW → NULL-callback inline).  Without this, a late
   * arts_add_dependence racing AFTER our NULL-store would push to
   * ooList and stall forever — the entry's data is gone but the
   * ooList is still walkable, so the request lands in a queue with
   * no future drainer. */
  arts_route_table_set_destroyed(p->db_guid);
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
  /* [2] drain ooList — wake parked EDT-DB-request waiters with
   * NULL_DB (destroyed semantic) and free their payloads.  Silently
   * dropping the queue would leave the waiters parked forever. */
  arts_route_table_drop_oo(p->db_guid);

  /* [3-5] cache_s self-destroy protocol — single-flight via destroy_state
   * CAS-gate, fail every parked waiter, hand finalize to the buffer/
   * pending_count drain. */
  if (arts_atomic_cswap(&cache->destroy_state, ARTS_DB_DESTROY_NONE,
                        ARTS_DB_DESTROY_MARKED) != ARTS_DB_DESTROY_NONE) {
    return; /* concurrent destroy raced — single-flight winner runs. */
  }

  /* Notify ranks with cached copies + queued LOCK_REQ requesters so
   * remote sharers wake up and observe DB_DESTROYED (necessary for
   * foreign-rank progress). */
  unsigned int self = arts_global_rank_id;

  /* Claim destroy_in_flight before iterating readers so that any
   * concurrent GET_DATA (RO_REQ) handler that arrives after the CAS
   * sees destroy_in_flight=1 and returns DESTROY_NOTIFY instead of
   * adding itself to the readers set (per gate-ordering in GET_DATA
   * handler: check destroy_in_flight BEFORE readers_bits_set). */
  unsigned int dif_zero = 0;
  if (!atomic_compare_exchange_strong_explicit(
          &cache->home->destroy_in_flight, &dif_zero, 1u, memory_order_acq_rel,
          memory_order_acquire)) {
    /* Another destroy already claimed the baton — idempotent drop. */
    return;
  }

#if defined(ARTS_MEMORY_MODEL_LC)
  /* LC: use home->last_sent_version as the readers roster (same as RC). */
  {
    unsigned int n = arts_global_rank_count;
    for (unsigned int r = 0; r < n; r++) {
      if (r == self) {
        continue;
      }
      if (arts_rank_u64_map_get(cache->home->last_sent_version, r) > 0) {
        arts_coh_send_destroy_notify(r, p->db_guid);
      }
    }
  }
  /* LC has no pending_rw queue: skip the lockreq drain. */
#elif defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC: notify the current RW owner first (tracked by rw_holder, not
   * the readers bit-set), then iterate the RO readers bit-set.  The
   * destroy_in_flight baton (claimed above) prevents new bits from
   * being set after this scan, and ensures incoming LOCK_REQs see
   * destroy_in_flight=1 and are rejected rather than queued. */
  {
    unsigned int holder =
        atomic_load_explicit(&cache->home->rw_holder, memory_order_acquire);
    if (holder != self) {
      arts_coh_send_destroy_notify(holder, p->db_guid);
    }
  }
  arts_readers_bits_for_each(&cache->home->readers, lrc_destroy_fanout_cb,
                             (void *)(uintptr_t)p->db_guid);
  {
    unsigned int q_rank;
    while (arts_home_lockreq_queue_pop(&cache->home->pending_rw, &q_rank)) {
      if (q_rank != self) {
        arts_coh_send_destroy_notify(q_rank, p->db_guid);
      }
    }
  }
#else
  /* RC: use home->last_sent_version as the readers roster. */
  {
    unsigned int n = arts_global_rank_count;
    for (unsigned int r = 0; r < n; r++) {
      if (r == self) {
        continue;
      }
      if (arts_rank_u64_map_get(cache->home->last_sent_version, r) > 0) {
        arts_coh_send_destroy_notify(r, p->db_guid);
      }
    }
  }
  {
    unsigned int q_rank;
    while (arts_home_lockreq_queue_pop(&cache->home->pending_rw, &q_rank)) {
      if (q_rank != self) {
        arts_coh_send_destroy_notify(q_rank, p->db_guid);
      }
    }
  }
#endif

  arts_coh_fail_trigger_pending(cache);
  arts_coh_try_finalize_destroy(cache);
}

void arts_coh_handle_db_create_coherent(
    struct arts_remote_db_create_coherent_packet_s *p) {
  /* Home-side init for non-home creator.  Per coherence design plan
   * §968-988: install zero-init buffer, home struct with rw_holder =
   * creator_rank, writer_count = 0 (home is non-owner). */
  unsigned int creator_rank = p->header.rank;
  arts_guid_t db_guid = p->db_guid;
  uint64_t db_size = p->db_size;

  /* Race against lazy_install or another path that already set up an
   * empty cache_s on this rank — coalesce by promoting the existing
   * lazy entry rather than allocating a duplicate. */
  struct arts_db_s *existing = arts_route_table_lookup_db_safe(db_guid);
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
#ifndef ARTS_MEMORY_MODEL_LC
      atomic_store_explicit(&cache->home->rw_holder, creator_rank,
                            memory_order_release);
#endif
    }
    arts_route_table_release(db_guid);
    return;
  }
  if (existing != NULL) {
    /* existing without coherence_cache (race with a stub install) — drop
     * the ref and proceed to the install/coalesce branch below. */
    arts_route_table_release(db_guid);
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
  arts_shared_init(&stub->shared, arts_db_get_deleter());
  stub->header.type = ARTS_GUID_DB;
  stub->header.size = sizeof(struct arts_db_s);
  stub->guid = db_guid;
  stub->db_type = (arts_db_types_t)p->db_type;
  stub->coherence_cache = arts_coh_alloc_cache_s(
      db_guid, db_size, ARTS_COH_INIT_HOME_RECV, creator_rank);
  /* back-pointer for try_finalize_destroy direct-free. */
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
  struct arts_db_s *winner = arts_route_table_lookup_db_safe(db_guid);
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
#ifndef ARTS_MEMORY_MODEL_LC
      atomic_store_explicit(&cache->home->rw_holder, creator_rank,
                            memory_order_release);
#endif
    }
  }
  if (winner != NULL) {
    arts_route_table_release(db_guid);
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
  /* Always install sentinel = 1; post-drain withdraw if has_next.
   * GRANT is only sent in RC and LRC builds; LC uses DATA_RESPONSE for
   * all acquires.  lock_req_in_flight and pending_rw are RC/LRC fields. */
  cache->writer_count = 1;
#ifndef ARTS_MEMORY_MODEL_LC
  cache->lock_req_in_flight = 0;
  /* Drain pending_rw — pop every queued waiter in FIFO order via the
   * Vyukov MPSC consumer path.  Implementation lives in the acquire
   * module (B4). */
  arts_coh_drain_pending_rw_after_grant(cache, p->version, p->has_next != 0);
#endif
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
#ifdef ARTS_MEMORY_MODEL_LRC
  /* LRC: record the new owner before decrementing.  This is safe because
   * only one INVALIDATE_NOTICE is ever in flight per round (baton gate
   * at home), so there is no concurrent writer to incoming_new_owner. */
  cache->incoming_new_owner = p->new_owner_rank;
  unsigned int rest = arts_atomic_sub(&cache->writer_count, 1);
  if (rest > 0) {
    /* Local writers still active; set transfer_pending so that the last
     * release_rw picks it up and ships TRANSFER_OWNERSHIP. */
    atomic_store_explicit(&cache->transfer_pending, 1u, memory_order_release);
    return;
  }
  /* rest == 0: we are the unique transfer actor. */
  arts_coh_lrc_ship_transfer(cache);
#elif defined(ARTS_MEMORY_MODEL_LC)
  /* LC: INVALIDATE_NOTICE is never sent.  This handler should not be
   * reachable in LC; stub to satisfy the dispatcher table. */
  (void)p;
#else  /* RC */
  unsigned int rest = arts_atomic_sub(&cache->writer_count, 1);
  if (rest == 0) {
    extern void arts_coh_invalidate_transfer(struct arts_db_cache_s * cache);
    arts_coh_invalidate_transfer(cache);
  }
#endif /* ARTS_MEMORY_MODEL_LRC / ARTS_MEMORY_MODEL_LC */
}

void arts_coh_handle_writeback_ack(
    struct arts_remote_writeback_ack_packet_s *p) {
#ifndef ARTS_MEMORY_MODEL_LRC
  /* Wakes any release_rw thread parked on await-WRITEBACK_ACK whose
   * outstanding seq <= p->seq.  Implementation in coherence_release.c. */
  extern void arts_coh_writeback_ack_signal(arts_guid_t db_guid, uint64_t seq);
  arts_coh_writeback_ack_signal(p->db_guid, p->seq);
#else
  /* LRC does not use WRITEBACK_ACK; this message type is not sent in
   * LRC builds.  Drop silently. */
  (void)p;
#endif
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
  /* Clear the route_table slot's data pointer before chaining into
   * try_finalize_destroy.  try_finalize_destroy frees db_owner directly
   * via arts_db_free; without this NULL-store the slot keeps pointing
   * at the freed stub, and once the malloc pool reuses that address
   * for a fresh stub on a different GUID two route_table slots end up
   * holding the same pointer.  Shutdown's arts_clean_up_route_table
   * would then arts_db_free the reused memory, dereferencing garbage
   * in coherence_cache.  The home-side handle_destroy_req does the
   * same NULL-store; destroy_notify must be symmetric on non-home
   * ranks.  Use claim_item (lookup + atomic_exchange → NULL); it does
   * not allocate or grow the table. */
  (void)arts_route_table_claim_item(p->db_guid);
  /* Symmetric with handle_destroy_req: set the DELETE bit so a late
   * acquire_item / add_oo_ex on this DB short-circuits instead of
   * parking on a queue with no future drainer. */
  arts_route_table_set_destroyed(p->db_guid);
  arts_coh_fail_trigger_pending(cache);
  arts_coh_try_finalize_destroy(cache);
}

/* ===== LRC-only: REDIRECT_RO handler (owner side) ==================== */

#ifdef ARTS_MEMORY_MODEL_LRC
void arts_coh_handle_redirect_ro(struct arts_remote_redirect_ro_packet_s *p) {
  unsigned int requester = p->requester_rank;
  void *waiter_addr = (void *)(uintptr_t)p->waiter_addr;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    /* DB has been destroyed or not yet installed on this rank. */
    arts_coh_send_destroy_notify(requester, p->db_guid);
    return;
  }
  if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
    arts_coh_send_destroy_notify(requester, p->db_guid);
    return;
  }

  struct arts_db_buffer_s *buf = arts_coherence_acquire_buf(cache);
  if (buf == NULL) {
    /* No buffer installed yet (pre-publication or sentinel DB).
     * Respond with version=0, no data — requester's RO waiter fires
     * with undefined content (per spec). */
    arts_coh_send_data_response(requester, p->db_guid, /*version=*/0,
                                waiter_addr, /*data=*/NULL, /*data_size=*/0);
    return;
  }

  /* Lazy-allocate last_sent_version on first ownership. */
  if (cache->last_sent_version == NULL) {
    cache->last_sent_version = arts_rank_u64_map_create(arts_global_rank_count);
  }

  uint64_t cur_v = (uint64_t)buf->version;
  uint64_t last_sent =
      arts_rank_u64_map_get(cache->last_sent_version, requester);

  if (last_sent >= cur_v) {
    /* Requester already holds this version — send no-data response. */
    arts_coh_send_data_response(requester, p->db_guid, cur_v, waiter_addr, NULL,
                                0);
  } else {
    /* Advance dedup watermark (monotonic max) then send data. */
    arts_rank_u64_map_advance(cache->last_sent_version, requester, cur_v);
    arts_coh_send_data_response(requester, p->db_guid, cur_v, waiter_addr,
                                buf->data, cache->db_size);
  }
  arts_coherence_release_buf(cache, buf);
}

/* Drain all deferred GET_DATA requests from the home-side RO forward queue.
 * Called by the INSTALL_ACK handler (Task 14) once invalidate_in_flight
 * is cleared and rw_holder reflects the new owner.  Single consumer. */
void arts_coh_drain_pending_ro_forwards(struct arts_db_cache_s *cache) {
  unsigned int requester;
  void *waiter_addr;
  while (arts_home_pending_ro_queue_pop(&cache->home->pending_ro_forwards,
                                        &requester, &waiter_addr)) {
    arts_readers_bits_set(&cache->home->readers, requester);
    unsigned int owner =
        atomic_load_explicit(&cache->home->rw_holder, memory_order_acquire);
    arts_coh_send_redirect_ro(owner, cache->db_guid, requester, waiter_addr);
  }
}
#endif /* ARTS_MEMORY_MODEL_LRC */
