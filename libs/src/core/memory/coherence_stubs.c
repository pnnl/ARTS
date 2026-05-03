/* SPDX-License-Identifier: Apache-2.0
 *
 * Stubs for coherence-protocol entry points whose full implementation
 * lives in later phase modules (acquire/B4, release/B5, destroy/B11).
 * Linking these as weak / no-op now lets B3+B10 (handlers + senders)
 * land in the tree without forward dependencies.  Each stub is
 * replaced by the real definition when the corresponding phase
 * module is wired in.
 *
 * Forward-decls match coherence_handlers.c's `extern` references. */

#include <stdbool.h>
#include <stdint.h>

#include "arts/memory/coherence.h"
#include "arts/system/print.h"

/* Acquire path (B4): drain visible pending_rw waiters after GRANT. */
__attribute__((weak)) void
arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                      uint64_t version, bool has_next) {
  (void)cache;
  (void)version;
  (void)has_next;
  ARTS_INFO("coherence stub: drain_pending_rw_after_grant (B4)");
}

/* Acquire path (B4): drain pending_ro waiters whose target is met. */
__attribute__((weak)) void
arts_coh_drain_pending_ro(struct arts_db_cache_s *cache, uint64_t version) {
  (void)cache;
  (void)version;
  ARTS_INFO("coherence stub: drain_pending_ro (B4)");
}

/* Acquire path (B4): trigger a single RO waiter via DATA_RESPONSE. */
__attribute__((weak)) void
arts_coh_trigger_ro_waiter(struct arts_db_cache_s *cache,
                           struct arts_db_ro_waiter_s *w, uint64_t version) {
  (void)cache;
  (void)w;
  (void)version;
  ARTS_INFO("coherence stub: trigger_ro_waiter (B4)");
}

/* Release path (B5): when INVALIDATE_NOTICE brings writer_count to 0,
 * the receiver becomes the transfer actor.  Implementation issues
 * local_transfer (home == self) or RELEASE_OWNERSHIP (one-way). */
__attribute__((weak)) void
arts_coh_invalidate_transfer(struct arts_db_cache_s *cache) {
  (void)cache;
  ARTS_INFO("coherence stub: invalidate_transfer (B5)");
}

/* Destroy path (B11): wake every still-unmarked waiter so parked
 * EDTs resume, observe destroy_state != NONE, and return DESTROYED. */
__attribute__((weak)) void
arts_coh_fail_trigger_pending(struct arts_db_cache_s *cache) {
  (void)cache;
  ARTS_INFO("coherence stub: fail_trigger_pending (B11)");
}

/* Destroy path (B11): single-flight finalization gated by
 * destroy_state.CAS(MARKED, CLEANING). */
__attribute__((weak)) void
arts_coh_try_finalize_destroy(struct arts_db_cache_s *cache) {
  (void)cache;
  ARTS_INFO("coherence stub: try_finalize_destroy (B11)");
}

/* Release path (B5): home-side ownership transfer.  Called from the
 * GRANT-handler defensive K=0 path when has_next=true with no local
 * waiters; also from INVALIDATE_NOTICE when home == self and
 * rest_count == 0. */
__attribute__((weak)) void
arts_coh_local_transfer_now(struct arts_db_cache_s *cache) {
  (void)cache;
  ARTS_INFO("coherence stub: local_transfer_now (B5)");
}
