/* SPDX-License-Identifier: Apache-2.0
 *
 * Release-consistency family (RC + LRC) shared coherence code.
 *
 * Compiled only when ARTS_MEMORY_MODEL is RC or LRC (selected in
 * libs/src/core/CMakeLists.txt).  Holds the home-directory / single-owner
 * ownership machinery that RC and LRC share but LC does not have (LC keeps the
 * canonical buffer at home via synchronous WRITEBACK and has no LOCK_REQ /
 * GRANT / pending_rw round).  The small points where RC and LRC themselves
 * differ are delegated to per-model hooks defined in db_coherence_rc.c /
 * db_coherence_lrc.c.  Contains NO ARTS_MEMORY_MODEL_* preprocessor logic.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/db.h"
#include "arts/db_coherence.h"
#include "arts/db_coherence_handlers.h"
#include "arts/db_coherence_home.h" /* arts_home_lockreq_queue_push */
#include "arts/db_coherence_model.h"
#include "arts/edt.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"     /* arts_global_rank_id */
#include "arts/transport/outbox.h"   /* arts_remote_send_request_async */
#include "arts/transport/protocol.h" /* arts_fill_packet_header, MSG_* */

/* ===== Case 2/6: RW local fast path ================================ */

typedef enum { CASE26_OK = 0, CASE26_FAIL_FALLBACK } case26_result_t;

static case26_result_t acquire_rw_local_fast(struct arts_db_cache_s *cache) {
  /* CAS-loop "increment if positive": never bump from 0. */
  while (1) {
    unsigned int wc = cache->writer_count;
    if (wc == 0) {
      return CASE26_FAIL_FALLBACK; /* ownership invalidated. */
    }
    if (arts_atomic_cswap(&cache->writer_count, wc, wc + 1) == wc) {
      return CASE26_OK;
    }
  }
}

/* ===== Case 4/8: remote-RW path ==================================== */

static arts_db_acquire_result_t acquire_remote_rw(struct arts_db_cache_s *cache,
                                                  arts_guid_t edt_guid,
                                                  unsigned int slot) {
  /* No destroy_state precheck: per spec 4.11, handle_destroy_req NULL-stores
   * route_item->data BEFORE flipping destroy_state, so route_table_lookup_db
   * already misses and the caller's OoO defer handles "DB destroyed".  If
   * we did get here with destroy_state advancing concurrently, the
   * fail_trigger_pending pop/wake-with-NULL chain will catch our waiter.
   */
  /* Allocate + push waiter into MPSC queue. */
  struct arts_db_rw_waiter_s *w =
      (struct arts_db_rw_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt_guid;
  w->slot = slot;
  /* Note: under MPSC there is no per-node "mark" — the consumer simply
   * pops in FIFO order and wakes each popped waiter (in
   * drain_pending_rw_after_grant / fail_trigger_pending / destroy
   * fan-out).  The post-push destroy re-check is folded into the
   * consumer path: if destroy_state advances past NONE while we are
   * pushing, fail_trigger_pending will pop us and wake the EDT with
   * NULL ptr. */
  arts_pending_rw_queue_push(&cache->pending_rw, w);

  /* Kick LOCK_REQ if no one else has — GRANT is what eventually
   * triggers our drain in FIFO order. */
  if (arts_atomic_cswap(&cache->ownership_req_in_flight, 0, 1) == 0) {
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_ownership_request(home_rank, cache->db_guid);
  }
  return ARTS_DB_ACQUIRE_PARK;
}

/* ===== 8-case dispatch (RC/LRC arm) ================================ */

arts_db_acquire_result_t
arts_coh_model_acquire_dispatch(struct arts_db_cache_s *cache,
                                arts_edt_dep_t *dep, arts_guid_t edt_guid,
                                unsigned int slot, arts_db_access_mode_t mode,
                                bool is_home, bool is_owner) {
  if (mode == DB_MODE_RO) {
    /* RC: home always holds current data (sync WRITEBACK) -> is_home||is_owner.
     * LRC: only the current owner has an installed buffer -> is_owner.  A
     * home-but-not-owner LRC rank goes through acquire_remote_ro so home
     * forwards to the owner via REDIRECT_RO. */
    if (arts_coh_model_ro_has_local_data(is_home, is_owner)) {
      /* acquire_local returns NULL when buffer is not installed (sentinel or
       * version-0 pre-install).  That is OK -- caller treats NULL as "no
       * payload".  No DESTROYED claim. */
      dep->ptr = arts_coh_acquire_local(cache);
      return ARTS_DB_ACQUIRE_OK;
    }
    /* Case 7: remote-RO. */
    return arts_coh_acquire_remote_ro(cache, edt_guid, slot);
  }

  /* mode == DB_MODE_RW (or RW-equivalent) */
  if (is_owner) {
    if (acquire_rw_local_fast(cache) == CASE26_OK) {
      /* writer_count bumped.  acquire_local NULL is fine (sentinel /
       * version-0); release_rw will decrement the matching bump.  No undo, no
       * DESTROYED. */
      dep->ptr = arts_coh_acquire_local(cache);
      return ARTS_DB_ACQUIRE_OK;
    }
    /* FAIL_FALLBACK: writer_count went to 0 between dispatch and CAS; fall
     * through to remote-RW. */
  }
  return acquire_remote_rw(cache, edt_guid, slot);
}

/* ===== GRANT drain (called from db_coherence_handlers.c) =========== */

/* Drain callback context for the RW MPSC pop loop. */
struct rw_drain_ctx_s {
  struct arts_db_cache_s *cache;
};

static void rw_drain_cb(arts_guid_t edt_guid, unsigned int slot, void *vctx) {
  struct rw_drain_ctx_s *ctx = (struct rw_drain_ctx_s *)vctx;
  /* Each popped waiter claims exactly one writer_count slot (FIFO) and
   * wakes its parked EDT. */
  arts_atomic_add(&ctx->cache->writer_count, 1);
  /* Sentinel DBs (db_size==0) have cache->buffer==NULL by design;
   * mark_edt_ready_by_guid handles that cleanly (depv[slot].ptr=NULL,
   * still decrements depc_needed). */
  mark_edt_ready_by_guid(edt_guid, slot);
}

void arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                           uint64_t version, bool has_next) {
  (void)version;
  (void)
      has_next; /* chain continuation is home-driven (advance_chain INVALIDATEs
                 * the new owner when the queue is still non-empty); the owner
                 * no longer self-withdraws its sentinel. */
  struct rw_drain_ctx_s ctx = {.cache = cache};
  arts_pending_rw_queue_drain(&cache->pending_rw, rw_drain_cb, &ctx);
}

/* ===== ownership-transfer trigger ================================== */

void arts_coh_invalidate_transfer(struct arts_db_cache_s *cache) {
  bool is_home =
      ((unsigned int)arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  if (is_home) {
    arts_coh_local_transfer_now(cache);
    return;
  }
  unsigned int home_rank = (unsigned int)arts_guid_get_rank(cache->db_guid);
  /* The transfer trigger MUST carry the owner's data.  A data-less
   * ownership_return could overtake the releasing worker's in-flight WRITEBACK
   * (release_rw decrements writer_count BEFORE it sends its writeback, so this
   * INVALIDATE-driven decrement can bring the count to 0 while that data is
   * still in flight) and make home GRANT the next owner a stale version.  Ship
   * the current buffer as a one-way WB_AND_TRANSFER instead: home installs it
   * before advancing the chain, so the next owner always sees this owner's
   * write regardless of arrival order vs the worker's own WRITEBACK (identical
   * version => idempotent install).  cv==0 marks it fire-and-forget: home skips
   * the ACK, so the network receiver thread running this handler does not block
   * on an ACK it would itself have to dispatch (a self-deadlock under a single
   * receiver thread). */
  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf != NULL) {
    arts_send_db_writeback(home_rank, cache->db_guid, buf->version, /*cv=*/0,
                           ARTS_WB_AND_TRANSFER, buf->data, cache->db_size);
    arts_coh_release_buf(&buf_h);
  } else {
    /* Zero-size sentinel DB (no buffer): no data can be stale, so the data-less
     * ownership_return is correct. */
    arts_send_db_ownership_return(home_rank, cache->db_guid);
  }
}

/* ===== destroy/fail fan-out of pending_rw ========================== */

static void fail_trigger_rw_cb(arts_guid_t edt_guid, unsigned int slot,
                               void *vctx) {
  (void)vctx;
  mark_edt_ready_by_guid(edt_guid, slot);
}

void arts_coh_model_fail_trigger_pending_rw(struct arts_db_cache_s *cache) {
  arts_pending_rw_queue_drain(&cache->pending_rw, fail_trigger_rw_cb, NULL);
}

/* ===== Home-side ownership handlers (RC+LRC; moved from handlers.c) ==
 * LOCK_REQ / RELEASE_OWNERSHIP exist only under RC and LRC (LC routes all
 * acquires through GET_DATA / DATA_RESPONSE), so these handlers are compiled
 * only in the release-consistency family.  The home-directory machinery they
 * touch (pending_rw, invalidate_in_flight, rw_holder) is RC+LRC-shared; the
 * point where RC and LRC diverge is delegated to per-model seams in
 * db_coherence_rc.c / db_coherence_lrc.c. */

/* OoO replay wrapper for a deferred LOCK_REQ: rebuild the packet from the OoO
 * args and re-issue the handler.  Registered in the route_table OoO dispatch
 * table for OOO_DB_OWNERSHIP_REQUEST (RC/LRC only). */
void arts_coh_ooo_replay_ownership_request(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_db_ownership_request_s *a =
      (struct arts_ooo_args_db_ownership_request_s *)vargs;
  struct arts_remote_ownership_request_packet_s p;
  p.header.rank = a->requester;
  p.db_guid = a->db_guid;
  arts_handler_db_ownership_request(&p);
}

void arts_handler_db_ownership_request(
    struct arts_remote_ownership_request_packet_s *p) {
  unsigned int requester = p->header.rank;

  /* Stack-built OoO defer payload — heap-copied by
   * arts_coh_home_lookup_or_defer if it actually has to defer. */
  struct arts_ooo_args_db_ownership_request_s oo_payload = {
      .requester = requester,
      .db_guid = p->db_guid,
  };

  struct arts_db_cache_s *cache = arts_coh_home_lookup_or_defer(
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
  /* Baton won: RC INVALIDATEs the current rw_holder; LRC pops the FIFO transfer
   * target, publishes pending_install_owner, and starts the invalidate round.
   */
  arts_coh_model_start_ownership_round(cache, db, requester);
}

void arts_handler_db_ownership_return(
    struct arts_remote_ownership_return_packet_s *p) {
  /* RELEASE_OWNERSHIP is one-way; on destroy the silent drop is fine
   * (caller doesn't await any reply).  no OoO defer either —
   * RELEASE_OWNERSHIP only flows from a current owner whose acquire
   * implied DB_CREATE already landed at home, so cache==NULL here means
   * the DB was already torn down. */
  struct arts_db_cache_s *cache = arts_coh_home_lookup_or_defer(
      p->db_guid, p->header.rank, OOO_DB_OWNERSHIP_REQUEST /*unused*/, NULL, 0,
      COH_REPLY_NONE);
  if (cache == NULL) {
    return;
  }
  /* RC advances the transfer chain; LRC never receives this message (no-op). */
  arts_coh_model_ownership_return(cache);
}

/* ===== Ownership wire senders (RC+LRC; moved from db_coherence_senders.c) ===
 * LOCK_REQ / RELEASE_OWNERSHIP / INVALIDATE_NOTICE exist only under the
 * release-consistency family.  Self-sends dispatch the matching handler inline
 * (request/return defined above; invalidate defined per model in rc.c/lrc.c).
 */

void arts_send_db_ownership_request(unsigned int home_rank,
                                    arts_guid_t db_guid) {
  struct arts_remote_ownership_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_ownership_request(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_ownership_invalidate(unsigned int owner_rank,
                                       arts_guid_t db_guid,
                                       unsigned int new_owner_rank) {
  struct arts_remote_ownership_invalidate_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_INVALIDATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.new_owner_rank = new_owner_rank;
  memset(p.pad, 0, sizeof(p.pad));
  if (owner_rank == arts_global_rank_id) {
    arts_handler_db_ownership_invalidate(&p);
    return;
  }
  arts_remote_send_request_async((int)owner_rank, (char *)&p, sizeof(p));
}

void arts_send_db_ownership_return(unsigned int home_rank,
                                   arts_guid_t db_guid) {
  struct arts_remote_ownership_return_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_RETURN);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_ownership_return(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}
