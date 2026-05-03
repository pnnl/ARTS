/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence destroy lifecycle.
 *
 * Three primitives, plus the user-visible destroy entry:
 *
 *   arts_coh_db_destroy(g)         — public API.  Forwards
 *                                    DESTROY_REQ to home (uniform
 *                                    path even when home == self).
 *
 *   fail_trigger_pending(cache)    — wake every still-unmarked
 *                                    waiter so the parked EDT can
 *                                    resume, observe destroy_state
 *                                    != NONE on retry, and surface
 *                                    ARTS_DB_DESTROYED.  Also
 *                                    decrements pending_count once
 *                                    per successful mark.
 *
 *   try_finalize_destroy(cache)    — single-flight cleanup gated by
 *                                    destroy_state.CAS(MARKED → CLEANING).
 *                                    Detaches cache.buffer (NULL-
 *                                    swap) and hands cache_s
 *                                    lifecycle to the route_table
 *                                    via mark_delete.
 *
 * The full design (race table, drop discipline, etc.) lives in the
 * coherence design plan; this file realizes the algorithm. */

#include <stdlib.h>

#include "arts/gas/route_table.h"
#include "arts/memory/coherence.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/coherence_home.h"
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

static void fail_trigger_visit_rw(arts_marked_list_node_t *node, void *vctx) {
  struct fail_trigger_ctx_s *ctx = (struct fail_trigger_ctx_s *)vctx;
  struct arts_db_rw_waiter_s *w = (struct arts_db_rw_waiter_s *)node;
  /* Read payload BEFORE mark — module may recycle the node soon. */
  arts_guid_t edt_local = w->edt_guid;
  unsigned int slot_local = w->slot;
  if (arts_marked_list_mark(node)) {
    arts_coh_mark_edt_ready_by_guid(edt_local, slot_local);
    arts_atomic_sub(&ctx->cache->pending_count, 1);
  }
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
  arts_marked_list_traverse(&cache->pending_rw, fail_trigger_visit_rw, &ctx);
  arts_marked_list_traverse(&cache->pending_ro, fail_trigger_visit_ro, &ctx);
}

/* ===== try_finalize_destroy (strong override) ===================== */

void arts_coh_try_finalize_destroy(struct arts_db_cache_s *cache) {
  /* Cheap reject: if pending_count != 0 there are live waiters and
   * cleanup must wait for them to mark/decrement.  Saves the CAS in
   * the common case. */
  if (cache->pending_count != 0) {
    return;
  }
  /* CAS(MARKED → CLEANING).  Forward-only: only one thread wins;
   * losers see CLEANING and bail. */
  if (arts_atomic_cswap(&cache->destroy_state, ARTS_DB_DESTROY_MARKED,
                        ARTS_DB_DESTROY_CLEANING) != ARTS_DB_DESTROY_MARKED) {
    return;
  }

  /* 1. Detach the buffer.  Outstanding readers retire via release_buf;
   *    no synchronous wait here. */
  struct arts_db_buffer_s *old =
      (struct arts_db_buffer_s *)arts_atomic_swap_ptr(
          (volatile void **)&cache->buffer, NULL);
  if (old != NULL && arts_atomic_sub(&old->ref_count, 1) == 0) {
    arts_lockfree_stack_push(&cache->buffer_pool, &old->pool_link);
  }

  /* 2. Mark route_table entry for delete.  Lifecycle of cache_s
   *    (and any embedded home_s) is handed to the existing route_
   *    table lock-field ref-counting: when count → 0 with DELETE
   *    set, route_table calls our destructor (registered separately
   *    in B9). */
  arts_route_table_mark_delete(cache->db_guid);
}

/* ===== arts_coh_db_destroy public API ============================= */

/* Public destroy: forward DESTROY_REQ to home (uniform path; home ==
 * self gets the message via self-loop).  Caller is responsible for
 * the OCR-spec contract: no concurrent acquires/uses in flight. */
void arts_coh_db_destroy(arts_guid_t db_guid) {
  unsigned int home_rank = (unsigned int)arts_guid_get_rank(db_guid);
  arts_coh_send_destroy_req(home_rank, db_guid);
}

/* ===== cache_s destructor (route_table-invoked, count → 0) ======== */

/* Called by the route_table when an entry's count reaches 0 with
 * DELETE set.  Performs the *only* real `free` calls of the DB's
 * lifetime.  Order matters because step 1 covers the rare race
 * where a wire handler installed a buffer past try_finalize_destroy's
 * NULL-swap. */
void arts_coh_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  /* 1. Detach cache.buffer (NULL-swap).  We do NOT free it here yet —
   *    in some races the same buffer ends up in the pool too, and freeing
   *    it twice would be a use-after-free.  We capture it as `leftover`
   *    and free at the end if and only if the pool drain didn't already
   *    free it. */
  struct arts_db_buffer_s *leftover =
      (struct arts_db_buffer_s *)arts_atomic_swap_ptr(
          (volatile void **)&cache->buffer, NULL);
  /* 2. Drain buffer_pool — frees every recycled buffer.
   *    Single-threaded at this point (cache_destructor runs only after
   *    the route_table ref count hits 0, i.e. no in-flight acquires).
   *    Walk the chain directly via node->next (stable: we are the only
   *    writer) instead of CAS-popping, which would race against arts_free. */
  arts_lockfree_stack_node_t *link =
      (arts_lockfree_stack_node_t *)(uintptr_t)(arts_atomic_swap_u64(
                                                    &cache->buffer_pool.top,
                                                    0) &
                                                ((1ULL << 48) - 1));
  bool leftover_in_pool = false;
  while (link != NULL) {
    arts_lockfree_stack_node_t *next = link->next;
    struct arts_db_buffer_s *buf = (struct arts_db_buffer_s *)link;
    if (buf == leftover) {
      leftover_in_pool = true;
    }
    arts_free(buf);
    link = next;
  }
  if (leftover != NULL && !leftover_in_pool) {
    arts_free(leftover);
  }
  /* 3. Walk pending_rw / pending_ro chains + private pool, freeing
   *    every node and the sentinels (sentinels live in the marked-
   *    list struct so they're freed implicitly). */
  arts_marked_list_destroy(&cache->pending_rw);
  arts_marked_list_destroy(&cache->pending_ro);
  /* 4. Home metadata. */
  if (cache->home != NULL) {
    arts_db_home_destroy(cache->home);
    cache->home = NULL;
  }
  /* 5. cache_s itself is freed by the route_table after this routine
   *    returns.  Buffers (FAM data lives there) are recycled to the
   *    pool / freed in step 1-3. */
}
