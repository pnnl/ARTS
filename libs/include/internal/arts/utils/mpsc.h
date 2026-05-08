/*
 * arts_mpsc_t — Multi-producer / single-consumer FIFO queue.
 *
 * Implementation: thin wrapper over arts_lf_stack_t (Treiber LIFO push)
 * + drainer-local cached chain.  Producers `push` with a single CAS;
 * the consumer `pop` lazily detaches the entire stack via
 * reverse_drain (LIFO -> FIFO) and yields one node per call.  When the
 * cached chain is empty, the next pop pulls a fresh chain.
 *
 * Concurrency contract:
 *   - `arts_mpsc_push` is fully lock-free; multiple producers may push
 *     concurrently.
 *   - `arts_mpsc_pop` is **single-consumer only**.  The caller MUST
 *     serialize pop with an external single-flight gate (e.g. the
 *     event's `draining` sentinel CAS).  Concurrent pops would race on
 *     `cached_chain`.
 *
 * FIFO ordering is preserved across producers that have a
 * sequenced-before / happens-before relationship (per OCR 1.2 §B.5
 * channel-event semantics).  Concurrent pushes without a hb relation
 * land in implementation-defined order — same guarantee xsocr's
 * channel ring buffer provides.
 */
#ifndef ARTS_UTILS_MPSC_H
#define ARTS_UTILS_MPSC_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/utils/lockfree_lifo.h"

/* Layout-mirror trick: the C++/nvcc TU drops _Atomic on cached_chain so
 * struct layout stays visible without C11 atomics, exactly the same
 * pattern arts_lf_stack_t uses. */
#ifdef __cplusplus
typedef struct {
  arts_lf_stack_t stack;
  arts_lf_link_t *cached_chain;
} arts_mpsc_t;
#else
typedef struct {
  arts_lf_stack_t stack;        /* Treiber LIFO; producers push here */
  arts_lf_link_t *cached_chain; /* drainer-local FIFO chain (single-consumer) */
} arts_mpsc_t;
#endif

#ifndef __cplusplus
static inline void arts_mpsc_init(arts_mpsc_t *q) {
  arts_lf_stack_init(&q->stack);
  q->cached_chain = NULL;
}

static inline void arts_mpsc_push(arts_mpsc_t *q, arts_lf_link_t *node) {
  arts_lf_stack_push(&q->stack, node);
}

/*
 * arts_mpsc_pop — single-consumer pop.  Caller MUST hold the
 * single-flight gate (e.g. event->channel.draining == 1).  Returns
 * NULL when the queue is empty.
 */
static inline arts_lf_link_t *arts_mpsc_pop(arts_mpsc_t *q) {
  if (q->cached_chain == NULL) {
    /* Detach the whole pushed stack and reverse it into FIFO order.
     * If no producer has pushed since the last drain, returns NULL. */
    q->cached_chain = arts_lf_stack_reverse_drain(&q->stack);
    if (q->cached_chain == NULL) {
      return NULL;
    }
  }
  arts_lf_link_t *n = q->cached_chain;
  q->cached_chain = atomic_load_explicit(&n->next, memory_order_relaxed);
  return n;
}

/*
 * arts_mpsc_drain_remaining — consumer-side cleanup.  Returns the chain
 * of all remaining nodes (cached + freshly-detached stack) for batch
 * release back to a per-rank pool.  Caller MUST hold the single-flight
 * gate; typical use is from the event_deleter at object teardown.
 */
static inline arts_lf_link_t *arts_mpsc_drain_remaining(arts_mpsc_t *q) {
  arts_lf_link_t *cached = q->cached_chain;
  q->cached_chain = NULL;
  arts_lf_link_t *fresh = arts_lf_stack_reverse_drain(&q->stack);
  if (cached == NULL) {
    return fresh;
  }
  /* Stitch fresh chain at the tail of cached (FIFO append). */
  arts_lf_link_t *tail = cached;
  arts_lf_link_t *next;
  for (;;) {
    next = atomic_load_explicit(&tail->next, memory_order_relaxed);
    if (next == NULL) {
      break;
    }
    tail = next;
  }
  atomic_store_explicit(&tail->next, fresh, memory_order_relaxed);
  return cached;
}
#endif /* !__cplusplus */

#ifdef __cplusplus
}
#endif
#endif /* ARTS_UTILS_MPSC_H */
