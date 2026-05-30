/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence destroy lifecycle.
 *
 * Two primitives, plus the user-visible destroy entry:
 *
 *   arts_coh_db_destroy(g)         — public API.  Forwards
 *                                    DESTROY_REQ to home (uniform
 *                                    path even when home == self).
 *
 *   fail_trigger_pending(cache)    — wake every still-unmarked
 *                                    waiter so the parked EDT can
 *                                    resume and surface
 *                                    ARTS_DB_DESTROYED.  Also
 *                                    decrements pending_count once
 *                                    per successful mark.
 *
 * Final teardown is driven by the cb (shared-ptr) deferred-free model:
 * destroy fans out, then arts_route_table_mark_delete frees the cache_s
 * via arts_coh_cache_destructor once all refs drain.
 *
 * The full design (race table, drop discipline, etc.) lives in the
 * coherence design plan; this file realizes the algorithm. */

#include <stdlib.h>

#include "arts/memory/coherence.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/coherence_home.h"
#include "arts/memory/db.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== fail_trigger_pending (strong override) ===================== */

/* Visit ctx: just the cache (so visitors can update pending_count). */
struct fail_trigger_ctx_s {
  struct arts_db_cache_s *cache;
};

/* Defined in coherence_acquire.c — wakes a parked EDT by stamping
 * depv[slot].ptr with cache->user_data and decrementing depc_needed.
 * On destroy-fail we still call it: cache->user_data may already be
 * NULL or freed, but the EDT just needs depc_needed advanced so it
 * can run and observe ARTS_DB_DESTROY semantics through its own dep
 * resolution. */
extern void arts_coh_mark_edt_ready_by_guid(arts_guid_t edt_guid,
                                            unsigned int slot);

static void fail_trigger_rw_cb(arts_guid_t edt_guid, unsigned int slot,
                               void *vctx) {
  struct fail_trigger_ctx_s *ctx = (struct fail_trigger_ctx_s *)vctx;
  arts_coh_mark_edt_ready_by_guid(edt_guid, slot);
  arts_atomic_sub(&ctx->cache->pending_count, 1);
}

static void fail_trigger_visit_ro(arts_marked_list_node_t *node, void *vctx) {
  struct fail_trigger_ctx_s *ctx = (struct fail_trigger_ctx_s *)vctx;
  struct arts_db_ro_waiter_s *w = (struct arts_db_ro_waiter_s *)node;
  arts_guid_t edt_local = w->edt_guid;
  unsigned int slot_local = w->slot;
  if (arts_marked_list_mark(node)) {
    arts_coh_mark_edt_ready_by_guid(edt_local, slot_local);
    arts_atomic_sub(&ctx->cache->pending_count, 1);
  }
}

void arts_coh_fail_trigger_pending(struct arts_db_cache_s *cache) {
  struct fail_trigger_ctx_s ctx = {.cache = cache};
  /* Pure FIFO drain: pop every queued RW waiter and wake.  No mark/
   * traverse needed — single consumer, no concurrent claims.
   * LC has no pending_rw queue (RW acquires use pending_ro). */
#ifndef ARTS_MEMORY_MODEL_LC
  arts_pending_rw_queue_drain(&cache->pending_rw, fail_trigger_rw_cb, &ctx);
#endif
  arts_marked_list_traverse(&cache->pending_ro, fail_trigger_visit_ro, &ctx);
}

/* ===== arts_coh_db_destroy public API ============================= */

/* Public destroy: forward DESTROY_REQ to home (uniform path; home ==
 * self gets the message via self-loop).  Caller is responsible for
 * the OCR-spec contract: no concurrent acquires/uses in flight. */
void arts_coh_db_destroy(arts_guid_t db_guid) {
  unsigned int home_rank = (unsigned int)arts_guid_get_rank(db_guid);
  arts_send_db_destroy(home_rank, db_guid);
}

/* ===== cache_s destructor (chained from arts_db_free, Phase 3.1) === */

/* Called from arts_db_free for ARTS_DB descriptors.  Drains the recycle pool
 * and tears down home_s in place; the cache is embedded by value as the first
 * member of db_s, so the caller (arts_db_free) frees the wrapping db_s — the
 * cache is not freed separately.  Order matters because step 1 covers the
 * rare race where a wire handler installed a buffer past
 * try_finalize_destroy's NULL-swap. */
void arts_coh_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  /* 1. Release the cache-hold on the buffer (store NULL into the shared
   *    slot).  If no acquirer holds a ref the cb deleter frees the buffer
   *    now; otherwise the buffer survives until the last in-flight acquirer
   *    releases (deferred free via the cb).  The buffer carries no back-ref
   *    to this cache, so it safely outlives us — eliminating the old
   *    destroy-vs-release use-after-free without a recycle pool. */
  arts_atomic_shared_store(&cache->buffer, NULL);
  /* 3. Walk pending_rw / pending_ro chains + private pool, freeing
   *    every node and the sentinels (sentinels live in the queue/list
   *    struct so they're freed implicitly).  RW uses Vyukov MPSC; RO
   *    still uses the marked-list (selective drain semantics).
   *    LC has no pending_rw queue. */
#ifndef ARTS_MEMORY_MODEL_LC
  arts_pending_rw_queue_destroy(&cache->pending_rw);
#endif
  arts_marked_list_destroy(&cache->pending_ro);
  /* 4. Home metadata (embedded by value; tear down sub-resources in place). */
  if (cache->home_initialized) {
    arts_db_home_teardown(&cache->home);
    cache->home_initialized = false;
  }
  /* 5. cache_s itself is freed by the route_table after this routine
   *    returns.  Buffers (FAM data lives there) are recycled to the
   *    pool / freed in step 1-3. */
}
