/* SPDX-License-Identifier: Apache-2.0
 *
 * The RETAIN release policy's half of the migrating grant.
 *
 * An unasked holder keeps the grant forever.  Every hand-over is therefore
 * demand-driven and home-initiated: a queued request opens a revocation round,
 * the round's INVALIDATE names the next owner, the holder ships owner->owner,
 * and the round closes at the home's CONFIRM by either revoking again for the
 * next queued requester or putting the baton down.  Nothing a holder does at
 * its own idle edge is visible to the home — its release only decrements, and
 * ships only where a round has already named a target.
 *
 * Both write policies link this TU; the one place they differ is the message
 * that carries a round's start, which is why the round close is split below.
 */
#include <stdbool.h>
#include <stdint.h>

#include "arts/coherence/coherence.h"
#include "arts/coherence/directory.h"
#include "arts/coherence/handlers.h"
#include "arts/db.h"
#include "arts/runtime_state.h"
#include "arts/system/threads.h"
#include "arts/runtime_types.h"
#include "arts/system/schedfuzz.h"
#include "arts/utils/atomics.h"
#include "arts/counter/Preamble.h"

/* ===== Home: a request arrived ========================================= */

void arts_db_grant_request_arrived(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester) {
  /* Active-directory invariant: AT MOST ONE GRANT_INVALIDATE in flight
   * to the current rw_holder per ownership-transfer round.  CAS 0->1
   * gates the dispatch — only the thread that flips the bit sends.
   * Concurrent GRANT_REQUESTs whose CAS loses simply piggyback on the
   * outstanding round; their requester is queued in pending_rw and is served by
   * the next CONFIRM-driven round.  The baton is held across the INVALIDATE →
   * owner→owner transfer → CONFIRM round-trip and cleared in the CONFIRM
   * handler once the queue drains. */
  unsigned int iif_zero = 0;
  if (!atomic_compare_exchange_strong_explicit(
          &db->invalidate_in_flight, &iif_zero, 1u, memory_order_acq_rel,
          memory_order_acquire)) {
    return; /* another round is in flight; requester stays queued */
  }
  /* Baton won: the WT write policy INVALIDATEs the current rw_holder; the WB
   * write policy pops the FIFO transfer target, publishes pending_install_owner,
   * and starts the invalidate round. */
  arts_db_start_grant_round(cache, db, requester);
}

/* ===== Home: a round closed ============================================ */

#ifdef ARTS_WRITE_POLICY_WT
void arts_db_grant_round_close(struct arts_db_cache_s *cache,
                               struct arts_db_s *db) {
  /* Drain-or-release retry loop: start the next transfer round if there are
   * pending_rw requests, otherwise release the baton (same loop as WB, minus
   * the CONFIRM_ACK send). */
  while (1) {
    unsigned int next_owner;
    struct arts_rdzv_landing_s next_rdzv;
    if (arts_home_grantreq_queue_pop(&db->pending_rw, &next_owner, &next_rdzv, NULL)) {
      db->pending_install_owner = next_owner;
      unsigned int current =
          atomic_load_explicit(&db->rw_holder, memory_order_acquire);
      arts_send_db_grant_invalidate(current, cache->db_guid, next_owner,
                                        &next_rdzv);
      return;
    }
    arts_sched_fuzz_point(); /* widen the pop-empty<->release window */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    /* Dekker-style publication: the baton-release store must be globally
     * visible BEFORE the emptiness re-check loads, or a requester that
     * pushed and lost its baton CAS inside the window is missed — a plain
     * release-store followed by loads permits exactly that StoreLoad
     * reordering. */
    atomic_thread_fence(memory_order_seq_cst);
    if (!arts_home_grantreq_queue_pending(&db->pending_rw)) {
      return;
    }
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      return;
    }
    INCREMENT_NUM_GRANT_BATON_RECLAIM_BY(1);
  }
}
#else
void arts_db_grant_round_close(struct arts_db_cache_s *cache,
                               struct arts_db_s *db) {
  /* Release the baton with the freshly-enqueued-racer recheck retry loop.  The
   * caller has already tried to advance the round by piggybacking the next
   * target on the CONFIRM_ACK, so this runs only with nothing queued. */
  while (1) {
    /* Release the baton. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    /* Dekker-style publication: the baton-release store must be globally
     * visible BEFORE the emptiness re-check loads, or a requester that
     * pushed and lost its baton CAS inside the window is missed — a plain
     * release-store followed by loads permits exactly that StoreLoad
     * reordering. */
    atomic_thread_fence(memory_order_seq_cst);
    /* Re-check for a freshly-enqueued requester that raced the baton
     * release.  If the queue is still empty, we're done. */
    if (!arts_home_grantreq_queue_pending(&db->pending_rw)) {
      return;
    }
    /* There is a new requester; try to re-acquire the baton. */
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      /* Another GRANT_REQUEST handler already picked up the baton (race);
       * that thread will drain the queue. */
      return;
    }
    INCREMENT_NUM_GRANT_BATON_RECLAIM_BY(1);
    /* Re-acquired the baton: pop the racer and start a fresh round.  This path
     * starts AFTER the just-confirmed owner is already running (no in-flight
     * CONFIRM_ACK to piggyback on), so it issues a STANDALONE INVALIDATE to the
     * current rw_holder, exactly like the first-round request-handler path. */
    unsigned int next_owner;
    struct arts_rdzv_landing_s next_rdzv;
    if (arts_home_grantreq_queue_pop(&db->pending_rw, &next_owner,
                                    &next_rdzv, NULL)) {
      db->pending_install_owner = next_owner;
      arts_db_owner_start_invalidate_round(cache, next_owner, &next_rdzv);
      return;
    }
    /* The racer was already consumed by whoever we contended with; loop to
     * release and recheck. */
  }
}
#endif /* ARTS_WRITE_POLICY_WT */

void arts_db_grant_home_idle_transition(struct arts_db_s *db) {
  /* Demand reaches a holder only when the home forwards it, and the home
   * forwards from whatever the directory names — so naming this rank IS the
   * whole transition.  The first request revokes the idle grant from here
   * exactly as it would from any other holder. */
  atomic_store_explicit(&db->rw_holder, arts_global_rank_id,
                        memory_order_seq_cst);
}

/* A grant is never handed back unasked here, so there is no obligation to
 * arm, to carry, or to settle: every release takes the ordinary
 * count-dropping edge, and a home-bound publish carries payload alone. */
bool arts_db_grant_release_claim(struct arts_db_cache_s *cache,
                                 bool will_publish) {
  (void)cache;
  (void)will_publish;
  return false;
}

bool arts_db_grant_return_claim_leg(struct arts_db_cache_s *cache) {
  (void)cache;
  return false;
}

void arts_db_grant_release_settle(struct arts_db_cache_s *cache) {
  (void)cache;
}

/* The shared publish paths ask the home to accept a hand-back that rode a
 * release.  No release here carries one, so the flag they unwrap is never
 * set and this is the arm's answer to a question it never gets asked. */
void arts_db_grant_return_arrived(struct arts_db_s *db,
                                  unsigned int returner) {
  (void)db;
  (void)returner;
}

/* ===== Holder: the word's possession-setting and count-dropping edges ==== */

void arts_db_grant_install(struct arts_db_cache_s *cache) {
  /* Sentinel(+1) + drain guard(+1), single op (0->2). */
  arts_atomic_add(&cache->writer_count, 2u);
}

bool arts_db_grant_release_skip(struct arts_db_cache_s *cache) {
  /* An all-zero word is the only state with nothing to drop: possession is a
   * hold of its own here, so any positive word owes a decrement.  Atomic
   * acquire-load avoids a TSan race against concurrent writes. */
  return arts_atomic_read(&cache->writer_count) == 0;
}

void arts_db_grant_release_commit(struct arts_db_cache_s *cache) {
  /* writer_count is non-negative (post-install flip + install guard absorb any
   * INVALIDATE that lands during install).  A true 1->0 release reads 0 and
   * ships the transfer; the (int) cast is defensive. */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1); /* post value */
  if (rest == 0 && cache->incoming_new_owner != ARTS_NO_PENDING_OWNER) {
    /* An INVALIDATE published a transfer target while writers were live; this
     * (last) releaser is the unique actor that ships the owner→owner transfer.
     * Identical for home and non-home owners. */
    arts_db_grant_ship_pending(cache);
  }
}

unsigned int arts_db_grant_commit_drain(struct arts_db_cache_s *cache,
                                       uint64_t version) {
  /* The commit's own hold stays on the word for the whole walk here: a
   * revocation can arrive at any moment and decrement it, and the hold is what
   * stops that decrement reaching the edge while waiters are still being
   * counted in.  Each waiter takes its hold as it is woken. */
  arts_db_drain_pending_rw_after_grant(cache, version, /*has_next=*/false);
  return 0u;
}

void arts_db_grant_commit_finish(struct arts_db_cache_s *cache,
                                 unsigned int drained) {
  (void)drained;
  /* Drop the commit's hold.  If a next-round INVALIDATE already withdrew the
   * sentinel and no local writer remains, we are the unique actor that ships
   * the next transfer. */
  if ((int)arts_atomic_sub(&cache->writer_count, 1) == 0 &&
      cache->incoming_new_owner != ARTS_NO_PENDING_OWNER) {
    arts_db_grant_ship_pending(cache);
  }
}
