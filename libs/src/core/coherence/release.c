/* SPDX-License-Identifier: Apache-2.0
 *
 * Release-consistency family (RC + LRC) shared coherence code.
 *
 * Compiled only when ARTS_MEMORY_MODEL is RC or LRC (selected in
 * libs/src/core/CMakeLists.txt).  Holds the home-directory / single-owner
 * ownership machinery that RC and LRC share but LC does not have (LC keeps the
 * canonical buffer at home via synchronous WRITEBACK and has no
 * OWNERSHIP_REQUEST / GRANT / pending_rw round).  The small points where RC and
 * LRC themselves differ are delegated to per-model seams
 * (arts_db_start_ownership_round / arts_db_ownership_return) defined in
 * coherence/rc.c / coherence/lrc.c — family→model calls.  Contains NO
 * ARTS_MEMORY_MODEL_* preprocessor logic.
 */
#include <assert.h> /* LRC INVALIDATE direct-call invariant assert */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h" /* arts_db_buf_acquire (invalidate xfer) */
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/home.h" /* arts_home_lockreq_queue_push */
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C self-send) */
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"     /* arts_global_rank_id */
#include "arts/transport/outbox.h"   /* arts_transport_send_async */
#include "arts/transport/protocol.h" /* arts_fill_packet_header, MSG_* */

/* ===== Case 2/6: RW local fast path ================================
 * Shared by the RC and LRC arts_handler_db_acquire bodies (coherence/rc.c /
 * coherence/lrc.c).  CAS-increments writer_count "if positive"; on success
 * writes dep->ptr (acquire_local) and returns true; returns false when
 * writer_count went to 0 (ownership invalidated) so the caller falls through to
 * arts_db_acquire_remote_rw. */
bool arts_db_acquire_rw_local_fast(struct arts_db_cache_s *cache,
                                   arts_edt_dep_t *dep) {
  /* CAS-loop "increment if positive": never bump from <= 0.  The comparison
   * MUST be signed: the commutative counter is transiently NEGATIVE when an
   * INVALIDATE races ahead of its GRANT (two messages on separate wires under
   * multiple receiver threads), and a negative value means NOT owner exactly
   * like 0.  Bumping from a negative value would cancel the invalidation's
   * decrement: the settled count then carries a surplus +1, the owner's last
   * release reads 1 instead of 0, the ownership transfer never fires, and
   * every queued acquirer on every rank is stranded. */
  while (1) {
    unsigned int wc = cache->writer_count;
    if ((int)wc <= 0) {
      return false; /* not owner (incl. transient negative) — go remote-RW. */
    }
    if (arts_atomic_cswap(&cache->writer_count, wc, wc + 1) == wc) {
      /* writer_count bumped.  acquire_local NULL is fine (sentinel /
       * version-0); release_rw will decrement the matching bump. */
      dep->ptr = arts_db_acquire_local(cache);
      return true;
    }
  }
}

/* ===== Case 4/8: remote-RW path ==================================== */

arts_db_acquire_result_t
arts_db_acquire_remote_rw(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
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

  /* Kick OWNERSHIP_REQUEST if no one else has — GRANT is what eventually
   * triggers our drain in FIFO order. */
  if (arts_atomic_cswap(&cache->ownership_req_in_flight, 0, 1) == 0) {
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_ownership_request(home_rank, cache->db_guid);
  }
  return ARTS_DB_ACQUIRE_PARK;
}

/* The RC/LRC arts_handler_db_acquire 8-case body lives per-model in
 * coherence/{rc,lrc}.c — the two builds differ only on the RO-has-local-data
 * predicate (RC is_home||is_owner; LRC is_owner), which the C-preprocessor seam
 * forbids in a shared TU.  Both call the shared arts_db_acquire_rw_local_fast /
 * arts_db_acquire_remote_rw above and arts_db_acquire_remote_ro
 * (coherence/coherence.c).
 */

/* ===== GRANT drain (called from coherence/{rc,lrc}.c) =========== */

/* Drain callback context for the RW MPSC pop loop. */
struct rw_drain_ctx_s {
  struct arts_db_cache_s *cache;
};

static void rw_drain_cb(arts_guid_t edt_guid, unsigned int slot, void *vctx) {
  struct rw_drain_ctx_s *ctx = (struct rw_drain_ctx_s *)vctx;
  /* Each popped waiter claims exactly one writer_count slot (FIFO). */
  arts_atomic_add(&ctx->cache->writer_count, 1);
  /* Advance the RW cursor first (position-idempotent; never schedules), THEN
   * deliver data (may schedule + let another worker run/free the EDT).
   * Sentinel DBs (db_size==0) have cache->buffer==NULL by design;
   * mark_edt_ready_by_guid handles that cleanly (depv[slot].ptr=NULL, still
   * accounts the dep). */
  mark_edt_secured_by_guid(edt_guid, slot);
  mark_edt_ready_by_guid(edt_guid, slot);
}

void arts_db_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
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

void arts_db_invalidate_transfer(struct arts_db_cache_s *cache) {
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  if (is_home) {
    arts_db_local_transfer_now(cache);
    return;
  }
  unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
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
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf != NULL) {
    arts_send_db_writeback(home_rank, cache->db_guid, buf->version, /*cv=*/0,
                           ARTS_WB_AND_TRANSFER, buf->data, cache->db_size);
    arts_db_buf_release(&buf_h);
  } else {
    /* Zero-size sentinel DB (no buffer): no data can be stale, so the data-less
     * ownership_return is correct. */
    arts_send_db_ownership_return(home_rank, cache->db_guid);
  }
}

/* ===== destroy/fail wake of parked waiters (RC+LRC) ================
 * Wake every parked waiter with a NULL ptr so the EDT observes the destroyed
 * DB (mark_edt_ready_by_guid delivers NULL when the cache buffer is gone): the
 * RW Vyukov MPSC FIFO first, then the snapshot reorder buffer.  LC has no
 * pending_rw queue so it defines its own arts_db_fail_trigger_pending
 * (coherence/lc.c) draining only pending_snapshot. */

static void fail_trigger_rw_cb(arts_guid_t edt_guid, unsigned int slot,
                               void *vctx) {
  (void)vctx;
  mark_edt_ready_by_guid(edt_guid, slot);
}

void arts_db_fail_trigger_pending(struct arts_db_cache_s *cache) {
  arts_pending_rw_queue_drain(&cache->pending_rw, fail_trigger_rw_cb, NULL);
  arts_db_drain_pending_snapshot(cache);
}

/* ===== Home-side ownership handlers (RC+LRC; moved from handlers.c) ==
 * OWNERSHIP_REQUEST / RELEASE_OWNERSHIP exist only under RC and LRC (LC routes
 * all acquires through GET_DATA / DATA_RESPONSE), so these handlers are
 * compiled only in the release-consistency family.  The home-directory
 * machinery they touch (pending_rw, invalidate_in_flight, rw_holder) is
 * RC+LRC-shared; the point where RC and LRC diverge is delegated to per-model
 * seams in coherence/rc.c / coherence/lrc.c. */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_OWNERSHIP_REQUEST]): the OoO engine
 * has already acquired the home db_s for db_guid and pinned a ref across this
 * call, so there is no lookup / NULL-check / defer here.  cache is the FIRST
 * member of arts_db_s (offset 0), so the slot object the engine hands us IS the
 * cache.  The wire dispatcher decodes OWNERSHIP_REQUEST into the args struct
 * and routes through the engine via OOO_DB_OWNERSHIP_REQUEST; a missing home
 * db_s defers the args and re-issues this body once DB_CREATE installs and
 * drains.  (LC never enqueues this kind — coherence/lc.c provides a no-op
 * definition that satisfies the single g_ooo_table slot in the LC build.) */
void arts_handler_db_ownership_request(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_ownership_request_s *a =
      (struct arts_ooo_args_db_ownership_request_s *)args_v;
  unsigned int requester = a->requester;

  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_home_lockreq_queue_push(&db->pending_rw, requester);

  /* Active-directory invariant: AT MOST ONE INVALIDATE_NOTICE in flight
   * to the current rw_holder per ownership-transfer round.  CAS 0->1
   * gates the dispatch — only the thread that flips the bit sends.
   * Concurrent OWNERSHIP_REQUESTs whose CAS loses simply piggyback on the
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
  arts_db_start_ownership_round(cache, db, requester);
}

/* Cat-C pure body (RELEASE_OWNERSHIP).  The wire dispatcher / self-send
 * shortcut has already looked the home db_s up with a held ref and passes it as
 * item_v (cache is its FIRST member, offset 0).  No lookup/NULL-check here —
 * the dispatcher's MISS branch SILENTLY DROPS: RELEASE_OWNERSHIP is one-way and
 * only flows from a current owner whose acquire implied DB_CREATE already
 * landed at home, so a missing cache means the DB was already torn down (caller
 * awaits no reply, and there is no OoO defer — this message is never deferred).
 * RC advances the transfer chain; LRC never receives this message (no-op). */
void arts_handler_db_ownership_return(void *item_v, void *args_v) {
  (void)args_v;
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  arts_db_ownership_return(cache);
}

/* ===== Ownership wire senders (RC+LRC; moved from coherence/senders.c) ===
 * OWNERSHIP_REQUEST / RELEASE_OWNERSHIP / INVALIDATE_NOTICE exist only under
 * the release-consistency family.  Self-sends dispatch the matching handler
 * inline (request/return defined above; invalidate defined per model in
 * rc.c/lrc.c).
 */

void arts_send_db_ownership_request(unsigned int home_rank,
                                    arts_guid_t db_guid) {
  struct arts_msg_ownership_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine exactly as the wire RX
     * dispatcher does — HIT runs the OWNERSHIP_REQUEST body inline, MISS defers
     * the args and replays once the home db_s is installed + drained.  (The
     * handler is now a pure (item, args) body; it no longer does its own
     * lookup-or-defer, so the inline shortcut must enter through
     * dispatch_or_defer.) */
    struct arts_ooo_args_db_ownership_request_s args = {
        .requester = p.header.rank,
        .db_guid = db_guid,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_OWNERSHIP_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_ownership_invalidate(unsigned int owner_rank,
                                       arts_guid_t db_guid,
                                       unsigned int new_owner_rank) {
  struct arts_msg_ownership_invalidate_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_INVALIDATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.new_owner_rank = new_owner_rank;
  memset(p.pad, 0, sizeof(p.pad));
  if (owner_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's model-split exactly. */
    struct arts_ooo_args_db_ownership_invalidate_s args = {
        .db_guid = db_guid,
        .new_owner_rank = new_owner_rank,
    };
#if defined(ARTS_MEMORY_MODEL_LRC)
    /* LRC never defers INVALIDATE: home publishes rw_holder (the target) only
     * after that rank's cache install, so the cache is provably present here.
     * Call the pure handler body directly. */
    struct arts_db_cache_s *cache = arts_db_cache_lookup(db_guid);
    assert(cache != NULL); /* target is rw_holder, published post-install */
    arts_handler_db_ownership_invalidate(arts_db_of_cache(cache), &args);
#else /* RC: Cat-B — defer on miss, replay on the install's drain. */
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_OWNERSHIP_INVALIDATE, &args,
                                    sizeof(args));
#endif
    return;
  }
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

void arts_send_db_ownership_return(unsigned int home_rank,
                                   arts_guid_t db_guid) {
  struct arts_msg_ownership_return_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_RETURN);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop.
     * HIT advances the transfer chain on the ref-pinned home db_s; MISS (DB
     * already torn down) silently drops (RELEASE_OWNERSHIP awaits no reply). */
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_ownership_return(db, NULL);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

static void proceed_cb(arts_guid_t edt_guid, unsigned int slot, void *ctx) {
  (void)ctx;
  mark_edt_secured_by_guid(edt_guid, slot);
}

void arts_handler_db_ownership_proceed(arts_guid_t db_guid) {
  struct arts_db_cache_s *cache = arts_db_cache_lookup(db_guid);
  if (cache == NULL) {
    return; /* DB destroyed — parked EDTs are woken by the destroy fail path */
  }
  arts_pending_rw_queue_for_each(&cache->pending_rw, proceed_cb, NULL);
}

void arts_send_db_ownership_proceed(unsigned int new_owner,
                                    arts_guid_t db_guid) {
  if (new_owner == arts_global_rank_id) {
    arts_handler_db_ownership_proceed(db_guid); /* self: run inline */
    return;
  }
  struct arts_msg_ownership_proceed_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_PROCEED);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)new_owner, (char *)&p, sizeof(p));
}
