/* SPDX-License-Identifier: Apache-2.0
 *
 * The grant's cache-side RW waiter queue.
 *
 * A Treiber stack of parked write acquires.  Order is immaterial — every
 * waiter is woken when the grant arrives, so there is nothing to be fair
 * about — which is why a LIFO push/drain-all suffices here while the home's
 * request FIFO (drain-one, and therefore order-bearing) stays Vyukov.
 *
 * Kept in its own translation unit so it can be linked, and unit-tested,
 * without dragging in the rest of the grant plane.
 */
#include "arts/coherence/types.h"
#include "arts/utils/lockfree_lifo.h"

/*--- per-cache pending_rw stack (Treiber) -------------------------------
 *
 * The per-cache RW waiter chain.  A Treiber stack (arts_lf_stack_t): each
 * waiter embeds an arts_lf_link_t as its first member.
 *
 * Concurrency model:
 *   Producers — foreign-rank acquire_remote_rw, multi-threaded; push prepends
 *               to the head via release-CAS.
 *   Consumer  — single home-side dispatcher
 *               (drain_pending_rw_after_grant /
 *                handle_destroy_req).  drain atomic-exchanges the whole chain
 *                out; for_each walks the live chain non-destructively.
 *
 * Memory: heap-allocated waiters (one malloc per producer).  drain / destroy
 * free each waiter after the callback.  The consume order is LIFO and
 * immaterial — every waiter is woken regardless of order — which is why a
 * stack suffices here while the home grantreq queue (drain-one) stays Vyukov. */

void arts_pending_rw_queue_init(arts_lf_stack_t *q) { arts_lf_stack_init(q); }

void arts_pending_rw_queue_push(arts_lf_stack_t *q,
                                struct arts_db_rw_waiter_s *w) {
  arts_lf_stack_push(q, &w->link);
}

void arts_pending_rw_queue_drain(arts_lf_stack_t *q,
                                 void (*cb)(arts_guid_t edt_guid,
                                            unsigned int slot, void *ctx),
                                 void *ctx) {
  /* Atomic-exchange the whole chain out (the drain's acquire pairs with each
   * producer's release-CAS push, so every detached node->next is visible),
   * then walk + wake + free.  LIFO order; the consume is order-free, so the
   * reversal is immaterial.  A producer prepending concurrently with the
   * exchange forms a fresh stack picked up by the next drain — no waiter is
   * lost. */
  arts_lf_link_t *node = arts_lf_stack_drain(q);
  while (node != NULL) {
    arts_lf_link_t *next =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    struct arts_db_rw_waiter_s *w =
        ARTS_CONTAINER_OF(node, struct arts_db_rw_waiter_s, link);
    cb(w->edt_guid, w->slot, ctx);
    arts_free(w);
    node = next;
  }
}

void arts_pending_rw_queue_for_each(arts_lf_stack_t *q,
                                    void (*cb)(arts_guid_t edt_guid,
                                               unsigned int slot, void *ctx),
                                    void *ctx) {
  /* Non-destructive walk of the live stack (single consumer; no pop, no free).
   * A concurrent producer prepends a new head, so walking from the head we
   * load may miss an in-flight push — PROCEED tolerates this (best-effort
   * wake-ahead; the eventual GRANT drain wakes every waiter). */
  arts_lf_link_t *cur = atomic_load_explicit(&q->head, memory_order_acquire);
  while (cur != NULL) {
    struct arts_db_rw_waiter_s *w =
        ARTS_CONTAINER_OF(cur, struct arts_db_rw_waiter_s, link);
    cb(w->edt_guid, w->slot, ctx);
    cur = atomic_load_explicit(&cur->next, memory_order_acquire);
  }
}

void arts_pending_rw_queue_destroy(arts_lf_stack_t *q) {
  /* Single-threaded at destroy: drain the chain and free every waiter. */
  arts_lf_link_t *node = arts_lf_stack_drain(q);
  while (node != NULL) {
    arts_lf_link_t *next =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    arts_free(ARTS_CONTAINER_OF(node, struct arts_db_rw_waiter_s, link));
    node = next;
  }
}
