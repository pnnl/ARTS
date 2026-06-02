/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_mpsc_t — Vyukov intrusive multi-producer / single-consumer queue
 * with a try-acquire-bail drain gate.
 *
 * Producers append with a single atomic_exchange on the head (wait-free);
 * the linking store that follows is what a concurrent consumer may briefly
 * observe as "inconsistent" (a producer has claimed the tail slot but not
 * yet linked it) — pop reports empty in that window and the next attempt
 * succeeds.  Exactly one consumer may run pop at a time; the drain gate
 * (try_drain_begin / drain_end) elects that consumer without spinning:
 * a thread that loses the CAS returns to its caller and lets the winner
 * drain everything visible up to the winner's drain_end.  Forward progress
 * is always made by some thread, so the gate is lock-free.
 *
 * Node type: the queue rides the common intrusive link `arts_lf_link_t`
 * (a single `{ _Atomic next }` cell, also used by arts_lf_stack_t).  Embed
 * it as the first member of any payload and recover the payload by casting
 * the popped node.  A node belongs to at most one queue (mpsc OR lf_stack)
 * at a time; after a node is popped it must be freed or pushed onto a
 * *different* queue before being reused — the same invariant arts_lf_stack_t
 * documents.
 */

#ifndef ARTS_UTILS_MPSC_H
#define ARTS_UTILS_MPSC_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/utils/lockfree_lifo.h" /* arts_lf_link_t */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Layout-mirror: the C++/nvcc TU drops _Atomic on head/drain_lock so the
 * struct layout stays visible without C11 atomics, exactly the same pattern
 * arts_lf_stack_t uses.  The inline producer/consumer helpers below only
 * compile in C. */
#ifdef __cplusplus
typedef struct {
  arts_lf_link_t *head;
  arts_lf_link_t *tail;
  arts_lf_link_t stub;
  uint8_t drain_lock;
} arts_mpsc_t;
#else
#include <stdatomic.h>

typedef struct {
  _Atomic(arts_lf_link_t *) head; /* producers tail-append via xchg */
  arts_lf_link_t *tail;           /* single consumer reads/advances */
  arts_lf_link_t stub;            /* embedded sentinel */
  _Atomic(uint8_t) drain_lock;    /* try-acquire drain election */
} arts_mpsc_t;

static inline void arts_mpsc_init(arts_mpsc_t *q) {
  atomic_store_explicit(&q->stub.next, NULL, memory_order_relaxed);
  atomic_store_explicit(&q->head, &q->stub, memory_order_relaxed);
  q->tail = &q->stub;
  atomic_store_explicit(&q->drain_lock, 0u, memory_order_relaxed);
}

/* Lock-free; any number of producers may run concurrently. */
static inline void arts_mpsc_push(arts_mpsc_t *q, arts_lf_link_t *n) {
  atomic_store_explicit(&n->next, NULL, memory_order_relaxed);
  arts_lf_link_t *prev =
      atomic_exchange_explicit(&q->head, n, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, n, memory_order_release);
}

/* Single-consumer pop; caller MUST hold the drain gate.  Returns NULL when
 * the queue is empty or transiently inconsistent (a producer is mid-link). */
static inline arts_lf_link_t *arts_mpsc_pop(arts_mpsc_t *q) {
  arts_lf_link_t *tail = q->tail;
  arts_lf_link_t *next =
      atomic_load_explicit(&tail->next, memory_order_acquire);
  if (tail == &q->stub) {
    if (!next)
      return NULL; /* empty */
    q->tail = next;
    tail = next;
    next = atomic_load_explicit(&tail->next, memory_order_acquire);
  }
  if (next) {
    q->tail = next;
    return tail;
  }
  arts_lf_link_t *head = atomic_load_explicit(&q->head, memory_order_acquire);
  if (tail != head)
    return NULL; /* producer mid-link — retry later */
  /* Re-thread the stub to make the single remaining node poppable. */
  arts_mpsc_push(q, &q->stub);
  next = atomic_load_explicit(&tail->next, memory_order_acquire);
  if (next) {
    q->tail = next;
    return tail;
  }
  return NULL;
}

/* Elect the single consumer.  Returns false if another drainer holds the
 * gate (caller does NOT spin — it returns and lets the winner finish). */
static inline bool arts_mpsc_try_drain_begin(arts_mpsc_t *q) {
  uint8_t expected = 0u;
  return atomic_compare_exchange_strong_explicit(&q->drain_lock, &expected, 1u,
                                                 memory_order_acq_rel,
                                                 memory_order_relaxed);
}

static inline void arts_mpsc_drain_end(arts_mpsc_t *q) {
  atomic_store_explicit(&q->drain_lock, 0u, memory_order_release);
}

/*
 * arts_mpsc_drain_remaining — single-consumer cleanup.  Pops every node
 * currently linked and returns them as a NULL-terminated chain (the popped
 * node's `next` field re-threaded to the following popped node) for batch
 * release back to a per-rank pool.  The embedded `stub` is never returned.
 *
 * Caller MUST be the sole consumer (hold the drain gate, or be in a
 * teardown path where no producer can push and no other consumer runs).
 * Because pop() can transiently report empty while a producer is mid-link,
 * this is intended for a quiescent queue (object teardown): all producers
 * have stopped, so a single pass drains everything.
 */
static inline arts_lf_link_t *arts_mpsc_drain_remaining(arts_mpsc_t *q) {
  arts_lf_link_t *chain = NULL;
  arts_lf_link_t *tail = NULL;
  for (;;) {
    arts_lf_link_t *n = arts_mpsc_pop(q);
    if (!n) {
      break;
    }
    atomic_store_explicit(&n->next, NULL, memory_order_relaxed);
    if (tail) {
      atomic_store_explicit(&tail->next, n, memory_order_relaxed);
    } else {
      chain = n;
    }
    tail = n;
  }
  return chain;
}
#endif /* !__cplusplus */

#ifdef __cplusplus
}
#endif
#endif /* ARTS_UTILS_MPSC_H */
