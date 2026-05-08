/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_coh_alloc_cache_s — cache_s allocator + initializer.
 *
 * One-stop constructor for arts_db_cache_s.  Wires writer_count / home /
 * pending_rw / pending_ro / db_owner based on `kind` (creator-home,
 * creator-remote, home-recv, lazy).  Buffer install is a separate step:
 * callers that need one go through arts_coherence_install_buffer.
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

struct arts_db_cache_s *
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
  /* Marked-list pending_ro queue: element_size MUST equal
   * sizeof(arts_db_ro_waiter_s) so arts_marked_list_alloc returns a
   * buffer big enough to hold the waiter struct (not a 1-byte stub
   * from calloc(1, 0) which would smash the heap on first acquire). */
  arts_marked_list_init(&c->pending_ro,
                        sizeof(struct arts_db_ro_waiter_s));
  /* Caller wires db_owner right after install (e.g. arts_db_create_internal,
   * arts_coh_lazy_install_cache_s).  arts_calloc already zeroed the field,
   * but make the contract explicit. */
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
