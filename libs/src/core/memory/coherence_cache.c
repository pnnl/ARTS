/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_coh_init_cache_s — in-place cache_s initializer.
 *
 * One-stop constructor for arts_db_cache_s.  Wires writer_count / home /
 * pending_rw / pending_ro based on `kind` (creator-home, creator-remote,
 * home-recv, lazy).  The cache is embedded by value as the first member of
 * struct arts_db_s; callers allocate the db_s and pass &db->cache here.
 * Buffer install is a separate step: callers that need one go through
 * arts_coherence_install_buffer.
 *
 * Called from:
 *   db.c                  — DB_CREATE local (creator == home / remote)
 *   coherence_acquire.c   — lazy install (RO acquire on non-home)
 *   coherence_handlers.c  — DB_CREATE_HANDLER on non-creator ranks
 *
 * The strong dispatcher entry points (drain_pending_rw_after_grant,
 * drain_pending_ro, trigger_ro_waiter, invalidate_transfer,
 * fail_trigger_pending, try_finalize_destroy, local_transfer_now) live in
 * coherence_acquire.c / coherence_release.c / coherence_destroy.c — earlier
 * weak stubs in this file (B3+B10 phase-staging placeholders) have been
 * removed now that all strong definitions are in place. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/memory/coherence.h"
#include "arts/memory/coherence_home.h"
#include "arts/system/threads.h"
#include "arts/utils/malloc.h"
#include "arts/utils/marked_list.h"

void arts_coh_init_cache_s(struct arts_db_cache_s *c, arts_guid_t db_guid,
                           uint64_t db_size, arts_coh_init_kind_t kind,
                           unsigned int creator_rank) {
  /* Caller provides a zeroed cache (embedded in a zeroed/calloc'd db_s, or
   * memset by the stub path).  We do not zero it here — the embedding db_s
   * owns the storage. */
  c->db_guid = db_guid;
  c->db_size = db_size;
  /* Vyukov MPSC queue cannot be zero-initialized: head and tail must
   * point at the embedded stub.  Initialize before any push could
   * land.  LC routes all RW acquires through the RO path and never
   * pushes to pending_rw, so skip in LC builds. */
#ifndef ARTS_MEMORY_MODEL_LC
  arts_pending_rw_queue_init(&c->pending_rw);
#endif
  /* Marked-list pending_ro queue: element_size MUST equal
   * sizeof(arts_db_ro_waiter_s) so arts_marked_list_alloc returns a
   * buffer big enough to hold the waiter struct (not a 1-byte stub
   * from calloc(1, 0) which would smash the heap on first acquire). */
  arts_marked_list_init(&c->pending_ro, sizeof(struct arts_db_ro_waiter_s));
  /* Initialize the embedded home metadata only on the rank that owns this
   * DB's GUID home; non-home ranks leave it zeroed (home_initialized stays
   * false).  init_kind selects the initial rw_holder. */
  unsigned int self = arts_global_rank_id;
  unsigned int n = arts_global_rank_count;
  if (n == 0) {
    n = 1;
  }
  if (kind == ARTS_COH_INIT_HOME_RECV) {
    arts_db_home_init(&c->home, creator_rank, n);
    c->home_initialized = true;
  } else if (kind == ARTS_COH_INIT_CREATOR_HOME) {
    arts_db_home_init(&c->home, self, n);
    c->home_initialized = true;
    c->writer_count = 2; /* sentinel + creator EDT */
  } else if (kind == ARTS_COH_INIT_CREATOR_REMOTE) {
    c->writer_count = 2;
  }
  /* RC/LC WRITEBACK ACK rendezvous is now a stack-local sem_t per release_rw
   * (pointer-identity match) — no per-cache seq fields to initialize. */
#if defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC owner-side fields: dedup map allocated lazily on first ownership
   * grant; arts_calloc already zeroed last_sent_version / incoming_new_owner,
   * but make the contract explicit. */
  c->last_sent_version = NULL;
  atomic_store_explicit(&c->transfer_pending, 0, memory_order_relaxed);
  c->incoming_new_owner = 0;
#endif
}
