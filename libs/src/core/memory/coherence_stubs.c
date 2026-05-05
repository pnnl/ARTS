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
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/memory/coherence.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_home.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/malloc.h"

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

/* Phase 2.2 stub: cache_s allocator.  The full version (per coherence
 * design plan §1006-1031 / §968-988) wires up writer_count / home /
 * buffer based on `kind`.  Until the strong definition lands (Phase 3),
 * this weak stub provides a minimal cache_s with an empty home struct
 * sized to arts_global_rank_count so home->last_sent_version /
 * home->pending_rw / home->rw_holder are at least well-defined.  Does
 * NOT install a buffer (callers that need one go through
 * arts_coherence_install_buffer separately). */
__attribute__((weak)) struct arts_db_cache_s *
arts_coh_alloc_cache_s(arts_guid_t db_guid, uint64_t db_size,
                       arts_coh_init_kind_t kind, unsigned int creator_rank) {
  struct arts_db_cache_s *c =
      (struct arts_db_cache_s *)arts_calloc(1, sizeof(struct arts_db_cache_s));
  c->db_guid = db_guid;
  c->db_size = db_size;
  c->destroy_state = ARTS_DB_DESTROY_NONE;
  /* Vyukov MPSC queue cannot be zero-initialized: head and tail must
   * point at the embedded stub.  Initialize before any push could
   * land. */
  arts_pending_rw_queue_init(&c->pending_rw);
  /* Phase 3.1: caller wires db_owner right after install (e.g.
   * arts_db_create_internal, arts_coh_lazy_install_cache_s).  arts_calloc
   * already zeroed the field, but make the contract explicit. */
  c->db_owner = NULL;
  /* Allocate home metadata only on the rank that owns this DB's GUID
   * home; non-home ranks leave c->home == NULL.  init_kind selects the
   * initial rw_holder. */
  unsigned int self = arts_global_rank_id;
  unsigned int n = arts_global_rank_count;
  if (n == 0) {
    n = 1;
  }
  if (kind == ARTS_COH_INIT_HOME_RECV) {
    c->home = arts_db_home_create(creator_rank, n);
  } else if (kind == ARTS_COH_INIT_CREATOR_HOME) {
    c->home = arts_db_home_create(self, n);
    c->writer_count = 2; /* sentinel + creator EDT */
  } else if (kind == ARTS_COH_INIT_CREATOR_REMOTE) {
    c->writer_count = 2;
  }
  return c;
}
