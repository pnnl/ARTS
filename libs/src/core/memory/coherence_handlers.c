/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message handlers (receive side) + home-side
 * dedup / ownership-transfer helpers.
 *
 * The matching arts_send_db_* SENDERS (packet-fill + outbox enqueue) live
 * in coherence_senders.c.
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

#include <assert.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h" /* ARTS_DB_PROP_NO_ACQUIRE */
#include "arts/gas/route_table.h"
#include "arts/memory/coherence.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_home.h"
#include "arts/memory/db.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

#ifdef ARTS_MEMORY_MODEL_LRC

/* Forward declarations for the drain helpers defined in coherence.c, called
 * from the LRC handlers below. */
void arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                           uint64_t version, bool has_next);
void arts_coh_drain_pending_snapshot(struct arts_db_cache_s *cache);

/* ===== LRC ship_transfer / start_invalidate_round ==================== */

void arts_coh_lrc_send_ownership_response(struct arts_db_cache_s *cache) {
  unsigned int new_owner = cache->incoming_new_owner;
  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf == NULL) {
    /* Sentinel DB (db_size==0) or pre-publication: send an empty transfer.
     * Still emit the 8-byte map count-header (count=0, pad=0): the receiver
     * unconditionally reconstructs map_size >= 8, so omitting it would
     * underflow data_size to (size_t)-8 and corrupt the install. */
    uint32_t empty_map[2] = {0u, 0u};
    arts_send_db_ownership_response(new_owner, cache->db_guid,
                                    /*version=*/0, empty_map, sizeof(empty_map),
                                    /*data=*/NULL, /*data_size=*/0);
    /* Re-arm the sentinel: a future INVALIDATE round publishes a fresh target.
     */
    cache->incoming_new_owner = ARTS_LRC_NO_PENDING_OWNER;
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

  arts_send_db_ownership_response(new_owner, cache->db_guid, buf->version,
                                  map_buf, map_size, buf->data, cache->db_size);
  arts_free(map_buf);

  /* Release the local buffer ref taken above.  The slot's cache-hold ref keeps
   * the buffer alive: the old RW owner retains its buffer + last_sent_version
   * map permanently (until DB destroy), so in-flight RO REDIRECTs that still
   * name this rank as owner are served from its own copy. */
  arts_coh_release_buf(&buf_h);
  /* Re-arm the sentinel: this owner has fully shipped; a future INVALIDATE
   * round will publish a fresh target. */
  cache->incoming_new_owner = ARTS_LRC_NO_PENDING_OWNER;
}

void arts_coh_lrc_start_invalidate_round(struct arts_db_cache_s *cache,
                                         unsigned int new_owner) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  unsigned int current_owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  arts_send_db_ownership_invalidate(current_owner, cache->db_guid, new_owner);
}

/* ===== LRC TRANSFER_OWNERSHIP handler (new owner C) =================== */

void arts_handler_db_ownership_response(void *payload, size_t size) {
  struct arts_remote_ownership_response_packet_s *hdr =
      (struct arts_remote_ownership_response_packet_s *)payload;
  arts_guid_t db_guid = hdr->db_guid;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(db_guid);
  if (cache == NULL) {
    cache = arts_coh_lazy_install_cache_s(db_guid, /*db_size=*/0);
    if (cache == NULL) {
      return; /* DB destroyed before we became owner — drop. */
    }
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
    arts_coh_install_buffer(cache, hdr->version, data_start, data_size);
    if (cache->db_size == 0) {
      cache->db_size = data_size;
    }
  }

  /* Install ownership sentinel: writer_count = 1. */
  arts_atomic_swap(&cache->writer_count, 1u);
  /* Clear ownership_req_in_flight so subsequent RW acquires can kick new
   * LOCK_REQ rounds if needed. */
  cache->ownership_req_in_flight = 0;

  /* Drain local RW waiters + any case-3 snapshot reorder-buffer waiters that
   * this install now satisfies. */
  arts_coh_drain_pending_rw_after_grant(cache, hdr->version,
                                        /*has_next=*/false);
  arts_coh_drain_pending_snapshot(cache);

  /* Confirm installation to home so home can update rw_holder. */
  unsigned int home_rank = (unsigned int)arts_guid_get_rank(db_guid);
  arts_send_db_ownership_response_ack(home_rank, db_guid, hdr->version);
}

/* ===== LRC INSTALL_ACK handler (home A) =============================== */

void arts_handler_db_ownership_response_ack(
    struct arts_remote_install_ack_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  struct arts_db_s *db = arts_db_of_cache(cache);

  /* Publish the new rw_holder (visible to GET_DATA redirect path). */
  unsigned int new_owner = db->pending_install_owner;
  atomic_store_explicit(&db->rw_holder, new_owner, memory_order_release);

  /* Drain-or-release retry loop: try to start the next transfer round
   * if there are pending_rw requests, otherwise release the baton. */
  while (1) {
    unsigned int next_owner;
    if (arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
      db->pending_install_owner = next_owner;
      arts_coh_lrc_start_invalidate_round(cache, next_owner);
      return;
    }
    /* No pending requester — release the baton. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    /* Re-check for a freshly-enqueued requester that raced the baton
     * release.  If the queue is still empty, we're done. */
    if (arts_home_lockreq_queue_empty(&db->pending_rw)) {
      return;
    }
    /* There is a new requester; try to re-acquire the baton. */
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      /* Another LOCK_REQ handler already picked up the baton (race);
       * that thread will drain the queue. */
      return;
    }
    /* Re-acquired the baton; loop to pop and start the next round. */
  }
}
#endif /* ARTS_MEMORY_MODEL_LRC */

/* ===== Forward decls: handler wiring with B4/B5/B11 ================= */

/* These are implemented in coherence.c (acquire/release/destroy) and
 * called from the response handlers below.  Forward-declared here to
 * avoid a circular include with coherence.c (which itself includes this
 * header for the sender helpers). */
void arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                           uint64_t version, bool has_next);
void arts_coh_drain_pending_snapshot(struct arts_db_cache_s *cache);
void arts_coh_fail_trigger_pending(struct arts_db_cache_s *cache);
/* Defined in coherence.c — resume a parked EDT by (edt_guid, slot). */
void mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot);

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
 *   cache != NULL                           -> return cache (caller body)
 *   cache == NULL, packet_for_oo == NULL    -> reply per reply_kind, NULL
 *   cache == NULL, ENQUEUED                 -> NULL (fire_oo will retry)
 *   cache == NULL, FIRED_BY_DRAIN           -> NULL (handler already ran via
 *                                              the drain triggered inside
 *                                              add_oo_ex; payload freed)
 *   cache == NULL, AVAILABLE_NOW (race)     -> re-lookup; same precheck
 *
 * The OoO payload is a heap copy of `packet_for_oo` whose first field is
 * forced to `oo_type` (every OOO_DB_* payload begins with oo_type_t). */
static struct arts_db_cache_s *
home_lookup_or_defer(arts_guid_t guid, unsigned int requester,
                     ooo_kind_t oo_type, void *packet_for_oo,
                     size_t packet_size, coh_reply_kind_t reply_kind) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(guid);

  if (cache != NULL) {
    return cache;
  }

  /* cache == NULL — defer via OoO or reply immediately. */
  if (packet_for_oo == NULL) {
    if (reply_kind == COH_REPLY_DESTROY_NOTIFY) {
      arts_send_db_cache_destroy(requester, guid);
    }
    return NULL;
  }

  /* Defer the replay.  dispatch_or_defer copies `packet_for_oo` (the kind's
   * args struct) into the OoO payload; on a concurrent install it re-issues
   * this handler inline (which re-enters here, now finds the cache, and runs
   * the core).  Either way the caller does nothing further. */
  (void)requester;
  arts_ooo_dispatch_or_defer_guid(guid, oo_type, packet_for_oo,
                                  (uint32_t)packet_size);
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
                                 arts_guid_t edt_guid, uint32_t slot) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  /* Monotonic dedup — if the requester already received this version
   * (cur >= master_v), send NO_DATA.  Cache_s lifetime invariant
   * guarantees user_data persists until destroy (route_table ref). */
  uint64_t cur = arts_rank_u64_map_get(db->last_sent_version, requester);
  if (cur >= master_v) {
    arts_send_db_snapshot_response(requester, cache->db_guid, master_v,
                                   edt_guid, slot, NULL, 0);
    return;
  }
  arts_rank_u64_map_set(db->last_sent_version, requester, master_v);
  arts_send_db_snapshot_response(requester, cache->db_guid, master_v, edt_guid,
                                 slot, data, data_size);
}

#endif /* !ARTS_MEMORY_MODEL_LRC */

/* ===== Home-side handlers ========================================== */

void arts_handler_db_ownership_request(
    struct arts_remote_ownership_request_packet_s *p) {
  /* LOCK_REQ is only sent in RC and LRC builds (LC routes all acquires
   * through GET_DATA / DATA_RESPONSE instead of LOCK_REQ / GRANT).  The
   * handler is still compiled in LC to satisfy the wire dispatcher table,
   * but it will never be called at runtime. */
#ifndef ARTS_MEMORY_MODEL_LC
  unsigned int requester = p->header.rank;

  /* Stack-built OoO defer payload — heap-copied by home_lookup_or_defer
   * if it actually has to defer. */
  struct arts_ooo_args_db_ownership_request_s oo_payload = {
      .requester = requester,
      .db_guid = p->db_guid,
  };

  struct arts_db_cache_s *cache = home_lookup_or_defer(
      p->db_guid, requester, OOO_DB_OWNERSHIP_REQUEST, &oo_payload,
      sizeof(oo_payload), COH_REPLY_DESTROY_NOTIFY);
  if (cache == NULL) {
    return;
  }
  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_home_lockreq_queue_push(&db->pending_rw, requester);

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
          &db->invalidate_in_flight, &iif_zero, 1u, memory_order_acq_rel,
          memory_order_acquire)) {
    return; /* another round is in flight; requester stays queued */
  }
#ifdef ARTS_MEMORY_MODEL_LRC
  /* LRC: pop the OLDEST requester (FIFO) to be the transfer target and
   * embed its rank in the INVALIDATE_NOTICE so the current holder ships
   * TRANSFER_OWNERSHIP directly, without a home round-trip.
   * pending_install_owner is only written by the baton holder (single
   * writer invariant), so no atomic needed. */
  unsigned int next_owner;
  if (!arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
    /* Defensive: we just pushed, so empty is impossible under correct
     * usage.  Release the baton and return. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    return;
  }
  db->pending_install_owner = next_owner;
  arts_coh_lrc_start_invalidate_round(cache, next_owner);
#else  /* RC */
  arts_send_db_ownership_invalidate(
      atomic_load_explicit(&db->rw_holder, memory_order_acquire), p->db_guid,
      /*new_owner_rank=*/0u);
#endif /* ARTS_MEMORY_MODEL_LRC */
#else  /* ARTS_MEMORY_MODEL_LC */
  (void)p; /* LC: LOCK_REQ is never sent; stub for completeness. */
#endif /* !ARTS_MEMORY_MODEL_LC */
}

void arts_handler_db_snapshot_request(
    struct arts_remote_snapshot_request_packet_s *p) {
  unsigned int requester = p->header.rank;

  struct arts_ooo_args_db_snapshot_request_s oo_payload = {
      .requester = requester,
      .db_guid = p->db_guid,
      .edt_guid = p->edt_guid,
      .slot = p->slot,
  };

  struct arts_db_cache_s *cache = home_lookup_or_defer(
      p->db_guid, requester, OOO_DB_SNAPSHOT_REQUEST, &oo_payload,
      sizeof(oo_payload), COH_REPLY_DESTROY_NOTIFY);
  if (cache == NULL) {
    return;
  }
  struct arts_db_s *db = arts_db_of_cache(cache);

#ifdef ARTS_MEMORY_MODEL_LRC
  /* LRC home-side RO routing.
   *
   * Home does not hold the canonical data copy under LRC — the current
   * owner does.  Home's job is to redirect the requester to the owner
   * (via REDIRECT_RO) so the owner can send DATA_RESPONSE directly,
   * applying the owner-side last_sent_version dedup.
   *
   * Record the requester in the cached-ranks set, then redirect to the current
   * owner.  A destroyed DB is handled by the route_table lookup miss (slot
   * value NULL-stored before destroy) + the handler single-actor invariant —
   * no per-home destroy flag. */
  arts_rank_bitset_set(&db->cached_ranks, requester);
  /* Forward to the current owner unconditionally: even mid-transfer, rw_holder
   * still names the OLD owner, which retains its buffer + last_sent_version
   * permanently and serves the REDIRECT from its own copy.  Client-side
   * monotonic version compare keeps stale snapshots safe.  No defer queue. */
  unsigned int owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  arts_send_db_snapshot_redirect(owner, cache->db_guid, requester, p->edt_guid,
                                 p->slot);
#else  /* !ARTS_MEMORY_MODEL_LRC */
  arts_shared_ptr_t master_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *master =
      (struct arts_db_buffer_s *)arts_shared_get(master_h);
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
    arts_send_db_snapshot_response(requester, p->db_guid, /*version=*/0,
                                   p->edt_guid, p->slot,
                                   /*data=*/NULL, /*data_size=*/0);
    return;
  }
  uint64_t master_v = master->version;
  update_last_sent_max(cache, requester, master_v, master->data, cache->db_size,
                       p->edt_guid, p->slot);
  arts_coh_release_buf(&master_h);
#endif /* ARTS_MEMORY_MODEL_LRC */
}

void arts_handler_db_writeback(struct arts_remote_writeback_packet_s *p,
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
    /* WRITEBACK raced ahead of DB_CREATE on the home rank (the producer
     * released before the home db_s installed — common once sender/receiver
     * threads reorder the wire).  Per the master plan this is a pure Cat-B OoO
     * defer: push the writeback (trailing data preserved in the payload) and
     * return WITHOUT acking.  The single WRITEBACK_ACK is sent exactly once
     * when the handler re-runs after DB_CREATE installs the cache (the
     * install's drain re-issues it).  Acking here too would post the
     * releaser's stack-local cv twice — the second post lands on a sem the
     * releaser already sem_destroy'd → glibc futex_fatal_error / SIGABRT.  If
     * the home db_s is never created (shutdown), await_writeback_ack's
     * shutdown re-check returns the worker, so it cannot stall forever. */
    uint32_t asz =
        (uint32_t)sizeof(struct arts_ooo_args_db_writeback_s) + data_size;
    char *buf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_writeback_s *a =
        (struct arts_ooo_args_db_writeback_s *)buf;
    a->releaser = releaser;
    a->db_guid = p->db_guid;
    a->version = p->version;
    a->cv = p->cv;
    a->flag = p->flag;
    a->data_size = data_size;
    if (data_size > 0 && data != NULL) {
      memcpy(buf + sizeof(*a), data, data_size);
    }
    arts_ooo_dispatch_or_defer_guid(p->db_guid, OOO_DB_WRITEBACK, buf, asz);
    arts_free(buf);
    return;
  }

  arts_coh_install_buffer(cache, p->version, data, data_size);
  arts_send_db_writeback_ack(releaser, p->db_guid, p->cv);
  /* No snapshot drain here: home's RO acquires hit case 1 (resume self) and
   * never park.  Foreign ROs are served by GET_DATA, not by drain. */

  /* WB_AND_TRANSFER: ownership-chain relay.  LC uses WRITEBACK_NORMAL only
   * (no exclusive owner to transfer to), so this branch is RC/LRC-only. */
#ifndef ARTS_MEMORY_MODEL_LC
  if (p->flag == ARTS_WB_AND_TRANSFER) {
#if defined(ARTS_MEMORY_MODEL_RC)
    /* RC: the writing owner returned the buffer; advance the transfer chain via
     * the shared helper (baton held across has_next, cleared only at the
     * queue-empty race-recovery point). */
    arts_coh_rc_advance_chain(cache);
#else
    /* LRC has no synchronous writeback (the dispatcher fatals on
     * MSG_DB_WRITEBACK), so this branch is unreachable in LRC; ownership moves
     * owner→owner via arts_coh_lrc_send_ownership_response. */
#endif
  }
#endif /* !ARTS_MEMORY_MODEL_LC */
}

void arts_handler_db_ownership_return(
    struct arts_remote_ownership_return_packet_s *p) {
  /* RELEASE_OWNERSHIP is only sent in RC and LRC builds.  In LC there is
   * no exclusive ownership chain, so this handler is never called at
   * runtime.  Stub it for LC to satisfy the dispatcher table. */
#ifndef ARTS_MEMORY_MODEL_LC
  /* RELEASE_OWNERSHIP is one-way; on destroy the silent drop is fine
   * (caller doesn't await any reply).  no OoO defer either —
   * RELEASE_OWNERSHIP only flows from a current owner whose acquire
   * implied DB_CREATE already landed at home, so cache==NULL here means
   * the DB was already torn down. */
  struct arts_db_cache_s *cache = home_lookup_or_defer(
      p->db_guid, p->header.rank, OOO_DB_OWNERSHIP_REQUEST /*unused*/, NULL, 0,
      COH_REPLY_NONE);
  if (cache == NULL) {
    return;
  }
#if defined(ARTS_MEMORY_MODEL_RC)
  /* RC: advance the transfer chain via the shared helper (baton held across
   * has_next, cleared only at the queue-empty race-recovery point).  The
   * helper's master==NULL path sends a no-data GRANT so the new owner's waiter
   * still fires, rather than tearing the new owner's cache mid-chain. */
  arts_coh_rc_advance_chain(cache);
#else
  /* LRC transfers ownership owner→owner via TRANSFER_OWNERSHIP and never sends
   * RELEASE_OWNERSHIP to home, so this is unreachable in LRC.  (void)cache. */
  (void)cache;
#endif
#else
  (void)p; /* LC: RELEASE_OWNERSHIP is never sent. */
#endif /* !ARTS_MEMORY_MODEL_LC */
}

#ifdef ARTS_MEMORY_MODEL_LRC
/* Fan-out callback for arts_rank_bitset_for_each during destroy.
 * ctx carries the db_guid encoded as uintptr_t (no heap allocation
 * needed since the callback is synchronous). */
static void lrc_destroy_fanout_cb(unsigned int rank, void *ctx) {
  arts_guid_t db_guid = (arts_guid_t)(uintptr_t)ctx;
  unsigned int self = arts_global_rank_id;
  if (rank != self) {
    arts_send_db_cache_destroy(rank, db_guid);
  }
}
#endif /* ARTS_MEMORY_MODEL_LRC */

void arts_handler_db_destroy(struct arts_remote_destroy_packet_s *p) {
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
   *   [3] PIN/CXL DBs have no embedded cache -> step 1 + arts_db_free.
   *   [4] For coherent DBs, run the cache_s self-destroy protocol:
   *       fail_trigger_pending wakes parked waiters;
   *       try_finalize_destroy single-flights the buffer detach. */
  unsigned int requester = p->header.rank;
  struct arts_ooo_args_db_destroy_s oo_payload = {
      .requester = requester,
      .db_guid = p->db_guid,
  };
  struct arts_db_cache_s *cache =
      home_lookup_or_defer(p->db_guid, requester, OOO_DB_DESTROY, &oo_payload,
                           sizeof(oo_payload), COH_REPLY_NONE);
  if (cache == NULL) {
    /* Either cache had destroy_state != NONE (already torn down -
     * destroy is idempotent, nothing else to do) or the request was
     * deferred via OoO (will replay after DB_CREATE_COHERENT). */
    return;
  }

  /* Cache is live — perform the three-phase destroy.  Re-derive the db_s
   * from the cache (cache is the first member of db_s) so the xchg sees the
   * same descriptor the lookup observed. */
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }

  /* cb-model destroy (single-actor on home + cb deferred-free).  No separate
   * destroy gate: mark_delete's atomic_exchange of the slot cb IS the
   * single-flight, and the handler single-actor invariant serializes destroy
   * with snapshot/ownership/writeback on this cache.  Run
   * the waiter-wake + fan-out FIRST (the cache stays alive — only the install
   * ref is dropped, by mark_delete at the very end); the cb deleter
   * (arts_db_deleter → arts_db_free → cache_destructor) then frees the cache
   * once the last outstanding lookup ref drains.  A second DESTROY_REQ finds
   * the slot absent (lookup → NULL) and is a no-op. */

  /* Any OoO replays still queued on this slot are preserved across destroy:
   * destroy is invoked only after explicit event synchronization, so a
   * straggler that arrives later legitimately waits for a labeled-GUID
   * reinstall (route-table teardown frees anything never replayed). */

  /* Notify ranks with cached copies + queued ownership requesters so remote
   * sharers wake and observe DB_DESTROYED.  Single-actor keeps the roster
   * stable across this scan. */
  unsigned int self = arts_global_rank_id;

#if defined(ARTS_MEMORY_MODEL_LC)
  /* LC: use home->last_sent_version as the readers roster (same as RC). */
  {
    unsigned int n = arts_global_rank_count;
    for (unsigned int r = 0; r < n; r++) {
      if (r == self) {
        continue;
      }
      if (arts_rank_u64_map_get(db->last_sent_version, r) > 0) {
        arts_send_db_cache_destroy(r, p->db_guid);
      }
    }
  }
  /* LC has no pending_rw queue: skip the lockreq drain. */
#elif defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC: notify the current RW owner first (tracked by rw_holder, not
   * the cached-ranks bit-set), then iterate the RO cached-ranks bit-set.  The
   * handler single-actor invariant keeps the readers roster stable across this
   * scan; a later LOCK_REQ/RO_REQ for a destroyed DB finds the slot absent
   * (lookup → NULL) and is a no-op. */
  {
    unsigned int holder =
        atomic_load_explicit(&db->rw_holder, memory_order_acquire);
    if (holder != self) {
      arts_send_db_cache_destroy(holder, p->db_guid);
    }
  }
  arts_rank_bitset_for_each(&db->cached_ranks, lrc_destroy_fanout_cb,
                            (void *)(uintptr_t)p->db_guid);
  {
    unsigned int q_rank;
    while (arts_home_lockreq_queue_pop(&db->pending_rw, &q_rank)) {
      if (q_rank != self) {
        arts_send_db_cache_destroy(q_rank, p->db_guid);
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
      if (arts_rank_u64_map_get(db->last_sent_version, r) > 0) {
        arts_send_db_cache_destroy(r, p->db_guid);
      }
    }
  }
  {
    unsigned int q_rank;
    while (arts_home_lockreq_queue_pop(&db->pending_rw, &q_rank)) {
      if (q_rank != self) {
        arts_send_db_cache_destroy(q_rank, p->db_guid);
      }
    }
  }
#endif

  arts_coh_fail_trigger_pending(cache);
  /* mark_delete LAST: detach the slot cb + drop the install ref.  The cache
   * stayed alive through the fan-out above (single-actor + install ref); its
   * cb deleter (arts_db_deleter → cache_destructor) frees it once outstanding
   * lookup refs drain. */
  (void)arts_route_table_mark_delete(p->db_guid);
}

void arts_handler_db_create(struct arts_remote_db_create_coherent_packet_s *p) {
  /* Home-side init for non-home creator.  Per coherence design plan
   * §968-988: install zero-init buffer, home struct with rw_holder =
   * creator_rank, writer_count = 0 (home is non-owner). */
  unsigned int creator_rank = p->header.rank;
  arts_guid_t db_guid = p->db_guid;
  uint64_t db_size = p->db_size;
  /* NO_ACQUIRE: the creator neither acquires nor writes back, so home is the
   * sole idle owner (not a non-owner awaiting a creator writeback). */
  bool no_acquire = (p->flags & ARTS_DB_PROP_NO_ACQUIRE) != 0;

  /* This handler installs/initializes the home directory, so it is only ever
   * dispatched to the GUID's home rank — where the descriptor is always a
   * full allocation.  A cache-only stub (which omits the home-arm fields)
   * is only ever installed on non-home ranks, so the home-field writes below
   * (arts_db_home_init / rw_holder) stay in bounds. */
  assert((unsigned int)arts_guid_get_rank(db_guid) == arts_global_rank_id &&
         "home-directory init must run on the GUID home rank");

  /* Race against lazy_install or another path that already set up an
   * empty cache_s on this rank — coalesce by promoting the existing
   * lazy entry rather than allocating a duplicate. */
  arts_shared_ptr_t existing_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *existing = (struct arts_db_s *)arts_shared_get(existing_h);
  if (existing != NULL && existing->db_type == ARTS_DB) {
    struct arts_db_cache_s *cache = &existing->cache;
    struct arts_db_s *db = existing;
    if (arts_coh_buffer_peek(cache) == NULL && db_size > 0) {
      arts_coh_install_buffer(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (!db->home_initialized) {
      arts_db_home_init(db, creator_rank, arts_global_rank_count);
      db->home_initialized = true;
    } else {
#ifndef ARTS_MEMORY_MODEL_LC
      atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
#endif
    }
    arts_shared_release(&existing_h);
    return;
  }
  if (existing != NULL) {
    /* existing non-ARTS_DB entry (no coherence cache) — drop the ref and
     * proceed to the install/coalesce branch below. */
    arts_shared_release(&existing_h);
  }

  /* No existing entry -- allocate the db_s stub (cache embedded), install in
   * route_table.
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
  stub->db_type = (arts_db_types_t)p->db_type;
  if (no_acquire) {
    /* Home is the idle RW owner from creation — identical to a locally created
     * DB (CREATOR_HOME: rw_holder = self, writer_count > 0).  Install a
     * zero-init buffer so the first consumer's RW acquire is granted locally
     * with a writable payload.  HOME_RECV (rw_holder = creator) would route
     * that acquire's INVALIDATE_NOTICE to a creator that holds no cache — a
     * phantom holder — and the acquire would stall forever. */
    arts_coh_init_cache_s(&stub->cache, db_guid, db_size,
                          ARTS_COH_INIT_CREATOR_HOME, creator_rank);
    if (db_size > 0) {
      arts_coh_install_buffer(&stub->cache, /*new_version=*/1,
                              /*data_payload=*/NULL, db_size);
    }
  } else {
    arts_coh_init_cache_s(&stub->cache, db_guid, db_size,
                          ARTS_COH_INIT_HOME_RECV, creator_rank);
  }

  if (arts_route_table_install_if_absent(stub, db_guid, arts_global_rank_id,
                                         /*used=*/true)) {
    arts_ooo_drain_guid(db_guid);
    return;
  }

  /* Lost race — free our stub and coalesce into the existing entry. */
  arts_db_free(stub);
  arts_shared_ptr_t winner_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *winner = (struct arts_db_s *)arts_shared_get(winner_h);
  if (winner != NULL && winner->db_type == ARTS_DB) {
    struct arts_db_cache_s *cache = &winner->cache;
    struct arts_db_s *db = winner;
    if (arts_coh_buffer_peek(cache) == NULL && db_size > 0) {
      arts_coh_install_buffer(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (!db->home_initialized) {
      arts_db_home_init(db, creator_rank, arts_global_rank_count);
      db->home_initialized = true;
    } else {
#ifndef ARTS_MEMORY_MODEL_LC
      atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
#endif
    }
  }
  if (winner != NULL) {
    arts_shared_release(&winner_h);
  }
}

/* ===== Sharer-side response handlers =============================== */

#ifndef ARTS_MEMORY_MODEL_LRC
/* RC GRANT handler (also compiled — but never dispatched — in LC, which fatals
 * on OWNERSHIP_RESPONSE).  LRC's TRANSFER_OWNERSHIP rides the (void*, size)
 * overload defined in the LRC block above. */
void arts_handler_db_ownership_response(
    struct arts_remote_ownership_response_packet_s *p, const void *data,
    uint64_t data_size) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (p->data_present) {
    arts_coh_install_buffer(cache, p->version, data, data_size);
  }
  /* Always install sentinel = 1; post-drain withdraw if has_next.
   * GRANT is only sent in RC and LRC builds; LC uses DATA_RESPONSE for
   * all acquires.  ownership_req_in_flight and pending_rw are RC/LRC fields. */
  cache->writer_count = 1;
#ifndef ARTS_MEMORY_MODEL_LC
  cache->ownership_req_in_flight = 0;
  /* Drain pending_rw — pop every queued waiter in FIFO order via the
   * Vyukov MPSC consumer path.  Implementation lives in the acquire
   * module (B4). */
  arts_coh_drain_pending_rw_after_grant(cache, p->version, p->has_next != 0);
#endif
  /* Drain pending_snapshot: any case-3 reorder-buffer waiter the new buffer
   * install now satisfies. */
  arts_coh_drain_pending_snapshot(cache);
}
#endif /* !ARTS_MEMORY_MODEL_LRC */

void arts_handler_db_snapshot_response(
    struct arts_remote_snapshot_response_packet_s *p, const void *data,
    uint64_t data_size) {
  /* No acquire-time list registration: this 1:1 response resumes the parked
   * EDT (p->edt_guid, p->slot) directly.  Three cases over a monotonic version:
   *   1. p->version <= buf->version : nothing newer to install — resume self
   *      against the live buffer.
   *   2. data + p->version > buf->version : install + drain-all the reorder
   *      buffer + resume self.
   *   3. NO_DATA + p->version > buf->version : the with-data reply was
   *      reordered behind us — push self onto pending_snapshot (a future
   *      case-2 install drains us) + re-check (race recovery).
   * Shared verbatim by RC/LRC/LC (LC routes RW through here too). */
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    /* destroyed (route_item NULL-stored) — drop. */
    return;
  }
  arts_guid_t edt_guid = p->edt_guid;
  uint32_t slot = p->slot;

  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t buf_v = buf ? (uint64_t)buf->version : 0;
  if (buf != NULL) {
    arts_coh_release_buf(&buf_h);
  }

  if (p->version <= buf_v) {
    /* Case 1: nothing newer to install — resume self. */
    mark_edt_ready_by_guid(edt_guid, slot);
    return;
  }
  if (p->data_present) {
    /* Case 2: install (version-conditional publish inside install_buffer;
     * stale installs retreat) + drain-all + resume self. */
    arts_coh_install_buffer(cache, p->version, data, data_size);
    arts_coh_drain_pending_snapshot(cache);
    mark_edt_ready_by_guid(edt_guid, slot);
    return;
  }
  /* Case 3: NO_DATA arrived ahead of the with-data reply.  Park a
   * reorder-buffer node; a later case-2 install drains it. */
  struct arts_db_snapshot_waiter_s *w =
      (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt_guid;
  w->slot = slot;
  w->target_version = p->version;
  arts_lf_stack_push(&cache->pending_snapshot, &w->link);
  /* Race recovery: a concurrent case-2 install may have published the buffer
   * between our version read and the push.  If so, drain (our own node
   * included) so we don't park forever.  The atomic_exchange drain is the
   * single-actor primitive — a concurrent installer's drain and ours cannot
   * both claim the same node. */
  arts_shared_ptr_t rch = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *rbuf =
      (struct arts_db_buffer_s *)arts_shared_get(rch);
  uint64_t rv = rbuf ? (uint64_t)rbuf->version : 0;
  if (rbuf != NULL) {
    arts_coh_release_buf(&rch);
  }
  if (rv >= p->version) {
    arts_coh_drain_pending_snapshot(cache);
  }
}

void arts_handler_db_ownership_invalidate(
    struct arts_remote_ownership_invalidate_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    /* cache==NULL here means the DB was destroyed: the route_item value is
     * atomic-exchanged to NULL only by the destroy fan-out.  An INVALIDATE is
     * sent exclusively to the rank that currently holds the DB (home publishes
     * rw_holder before any INVALIDATE can name a target, and that holder
     * installed its cache at acquire time), so a live, never-destroyed target
     * always has a cache.  A missing cache therefore means the holder is gone
     * and the sentinel withdrawal is moot — dropping is correct.  Deferring
     * onto the OoO list would be WRONG: a sharer-side message gets no
     * CREATE-driven drain, so it could only replay on a future same-GUID
     * re-creation and underflow that fresh DB's writer_count. */
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
  /* LRC: publish the transfer target BEFORE withdrawing the sentinel.  This
   * ordering is the dedup (no separate transfer_pending flag): a concurrent
   * release_rw that observes rest==0 is guaranteed to see incoming_new_owner
   * already published, so exactly one of {this handler, the last releaser}
   * ships.  Only one INVALIDATE_NOTICE is in flight per round (home baton
   * gate), so there is no concurrent writer to incoming_new_owner. */
  cache->incoming_new_owner = p->new_owner_rank;
  unsigned int rest = arts_atomic_sub(&cache->writer_count, 1);
  if (rest > 0) {
    /* Local writers still active; the last release_rw will see rest==0 with
     * incoming_new_owner already published and ship TRANSFER_OWNERSHIP. */
    return;
  }
  /* rest == 0: we are the unique transfer actor. */
  arts_coh_lrc_send_ownership_response(cache);
#elif defined(ARTS_MEMORY_MODEL_LC)
  /* LC: INVALIDATE_NOTICE is never sent.  This handler should not be
   * reachable in LC; stub to satisfy the dispatcher table. */
  (void)p;
#else  /* RC */
  /* The chain-lifetime baton (home.invalidate_in_flight, held at 1 across the
   * whole has_next chain and cleared only at the single queue-empty
   * race-recovery point) guarantees AT MOST ONE INVALIDATE reaches this holder
   * per transfer round, and only after home advanced rw_holder to a rank that
   * already holds the sentinel (+1).  So writer_count >= 1 on arrival; a wrap
   * to UINT_MAX is a real protocol defect (lost release pair / double-
   * INVALIDATE), caught here in debug builds rather than silently corrupting
   * the count. */
  unsigned int rest = arts_atomic_sub(&cache->writer_count, 1);
  assert(rest != (unsigned int)-1 &&
         "OWNERSHIP_INVALIDATE underflowed writer_count — ownership-transfer "
         "chain baton invariant violated");
  if (rest == 0) {
    extern void arts_coh_invalidate_transfer(struct arts_db_cache_s * cache);
    arts_coh_invalidate_transfer(cache);
  }
#endif /* ARTS_MEMORY_MODEL_LRC / ARTS_MEMORY_MODEL_LC */
}

void arts_handler_db_writeback_ack(
    struct arts_remote_writeback_ack_packet_s *p) {
#ifndef ARTS_MEMORY_MODEL_LRC
  /* Pointer-identity wakeup: p->cv is the address of the releaser's
   * stack-local sem_t (valid on this rank — the ACK always returns to the
   * rank that sent the WRITEBACK).  Post it to wake the parked release_rw.
   * No route_table lookup, no seq matching. */
  sem_t *cv = (sem_t *)(uintptr_t)p->cv;
  if (cv != NULL) {
    sem_post(cv);
  }
#else
  /* LRC does not use WRITEBACK_ACK; this message type is not sent in
   * LRC builds.  Drop silently. */
  (void)p;
#endif
}

void arts_handler_db_cache_destroy(
    struct arts_remote_cache_destroy_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return; /* already torn down on this rank (cb-NULL = idempotent). */
  }
  /* cb-model destroy (single-actor): no destroy_state gate — a second
   * DESTROY_NOTIFY finds the slot absent (cache == NULL above).  fail_trigger
   * FIRST so the cache is alive while we wake waiters (mark_delete drops only
   * the install ref; the cb deleter frees once outstanding lookup refs
   * drain). */
  arts_coh_fail_trigger_pending(cache);
  (void)arts_route_table_mark_delete(p->db_guid);
}

/* ===== LRC-only: REDIRECT_RO handler (owner side) ==================== */

#ifdef ARTS_MEMORY_MODEL_LRC
void arts_handler_db_snapshot_redirect(
    struct arts_remote_snapshot_redirect_packet_s *p) {
  unsigned int requester = p->requester_rank;
  arts_guid_t edt_guid = p->edt_guid;
  uint32_t slot = p->slot;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    /* DB has been destroyed or not yet installed on this rank. */
    arts_send_db_cache_destroy(requester, p->db_guid);
    return;
  }

  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf == NULL) {
    /* No buffer installed yet (pre-publication or sentinel DB).
     * Respond with version=0, no data — requester's RO waiter fires
     * with undefined content (per spec). */
    arts_send_db_snapshot_response(requester, p->db_guid, /*version=*/0,
                                   edt_guid, slot, /*data=*/NULL,
                                   /*data_size=*/0);
    return;
  }
  /* Invariant: last_sent_version is created at ownership-install
   * (TRANSFER_OWNERSHIP, retained permanently thereafter).  The INITIAL
   * owner (the creator, which never received a transfer) has no map yet, so
   * lazily create it on its first served REDIRECT — otherwise the producer-on-
   * home + RO-consumers-elsewhere DAG would be served no-data (NULL/stale).
   * Single-actor here (the redirect handler runs on the network thread). */
  if (cache->last_sent_version == NULL) {
    cache->last_sent_version = arts_rank_u64_map_create(arts_global_rank_count);
  }

  uint64_t cur_v = (uint64_t)buf->version;
  uint64_t last_sent =
      arts_rank_u64_map_get(cache->last_sent_version, requester);

  if (last_sent >= cur_v) {
    /* Requester already holds this version — send no-data response. */
    arts_send_db_snapshot_response(requester, p->db_guid, cur_v, edt_guid, slot,
                                   NULL, 0);
  } else {
    /* Advance dedup watermark (monotonic max) then send data. */
    arts_rank_u64_map_advance(cache->last_sent_version, requester, cur_v);
    arts_send_db_snapshot_response(requester, p->db_guid, cur_v, edt_guid, slot,
                                   buf->data, cache->db_size);
  }
  arts_coh_release_buf(&buf_h);
}
#endif /* ARTS_MEMORY_MODEL_LRC */
