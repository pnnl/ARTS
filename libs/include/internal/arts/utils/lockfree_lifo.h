/* SPDX-License-Identifier: Apache-2.0
 *
 * 8-byte head CAS Treiber LIFO (header-only, inline).
 *
 * Differs from the legacy `arts/utils/lockfree_stack.h` (tagged-pointer
 * ABA defense, separate impl file): this variant assumes the caller
 * upholds the "no re-entry into the same stack" invariant documented
 * below, which is sufficient for ABA-freedom without packed counters.
 * Used by the event subsystem rewrite — event->deps, etc.
 *
 * Kept as a separate header (lockfree_lifo.h) because the API names —
 * arts_lf_stack_t / arts_lf_link_t — are distinct from the legacy
 * arts_lockfree_stack_t / arts_lockfree_stack_node_t in lockfree_stack.h.
 * Both can coexist in arts/utils/.
 */

#ifndef ARTS_UTILS_LOCKFREE_LIFO_H
#define ARTS_UTILS_LOCKFREE_LIFO_H

#include <stddef.h>

/* runtime_types.h pulls this header transitively into .cu translation
 * units (via arts_event_s).  C11 _Atomic isn't available in C++/nvcc, so
 * fall back to plain pointers for layout-only visibility there.  The
 * inline producer/consumer helpers below only compile in C. */
#ifdef __cplusplus

/** Common intrusive link node.  Embed as the first member of any
 *  caller struct that participates in lock-free stacks/pools. */
typedef struct arts_lf_link {
  struct arts_lf_link *next;
} arts_lf_link_t;

/** 8-byte head CAS Treiber stack.
 *
 * INVARIANT: a node may belong to at most one arts_lf_stack_t at a
 * time.  After pop/drain, the node must EITHER be freed OR pushed onto a
 * DIFFERENT stack (e.g., the per-rank arts_lockfree_pool_t).  Pushing the
 * same node back onto the same stack invalidates the ABA-free guarantee. */
typedef struct {
  arts_lf_link_t *head;
} arts_lf_stack_t;

#else /* C path with _Atomic + inline helpers */

#include <stdatomic.h>

/** Common intrusive link node.  Embed as the first member of any
 *  caller struct that participates in lock-free stacks/pools. */
typedef struct arts_lf_link {
  _Atomic(struct arts_lf_link *) next;
} arts_lf_link_t;

/** 8-byte head CAS Treiber stack.
 *
 * INVARIANT: a node may belong to at most one arts_lf_stack_t at a
 * time.  After pop/drain, the node must EITHER be freed OR pushed onto a
 * DIFFERENT stack (e.g., the per-rank arts_lockfree_pool_t).  Pushing the
 * same node back onto the same stack invalidates the ABA-free guarantee. */
typedef struct {
  _Atomic(arts_lf_link_t *) head;
} arts_lf_stack_t;

static inline void arts_lf_stack_init(arts_lf_stack_t *s) {
  atomic_store_explicit(&s->head, NULL, memory_order_relaxed);
}

static inline void arts_lf_stack_push(arts_lf_stack_t *s,
                                      arts_lf_link_t *node) {
  arts_lf_link_t *old = atomic_load_explicit(&s->head, memory_order_relaxed);
  do {
    atomic_store_explicit(&node->next, old, memory_order_relaxed);
  } while (!atomic_compare_exchange_weak_explicit(
      &s->head, &old, node, memory_order_release, memory_order_relaxed));
}

/** Atomically detach the entire chain.  Returns LIFO order (top of stack
 *  first). */
static inline arts_lf_link_t *arts_lf_stack_drain(arts_lf_stack_t *s) {
  return atomic_exchange_explicit(&s->head, NULL, memory_order_acquire);
}

/** Drain + reverse — returns FIFO order (oldest push first). */
static inline arts_lf_link_t *arts_lf_stack_reverse_drain(arts_lf_stack_t *s) {
  arts_lf_link_t *lifo = arts_lf_stack_drain(s);
  arts_lf_link_t *fifo = NULL;
  while (lifo) {
    arts_lf_link_t *next =
        atomic_load_explicit(&lifo->next, memory_order_relaxed);
    atomic_store_explicit(&lifo->next, fifo, memory_order_relaxed);
    fifo = lifo;
    lifo = next;
  }
  return fifo;
}

#endif /* __cplusplus */

#endif /* ARTS_UTILS_LOCKFREE_LIFO_H */
