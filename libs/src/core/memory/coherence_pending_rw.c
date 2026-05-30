/* SPDX-License-Identifier: Apache-2.0
 *
 * Vyukov MPSC queue for cache->pending_rw — the per-cache RW waiter
 * chain.  Replaces the prior Harris-style marked-list.
 *
 * Algorithm: see libs/src/core/gas/out_of_order_list.c for the full
 * narrative.  This file applies the same algorithm to a typed waiter
 * (arts_db_rw_waiter_s) instead of the generic (data, next) node.
 *
 * Concurrency model:
 *   Producers — foreign-rank acquire_remote_rw, multi-threaded.
 *   Consumer  — single home-side dispatcher
 *               (drain_pending_rw_after_grant / fail_trigger_pending /
 *                handle_destroy_req).  Pop is plain head advance.
 *
 * Memory: heap-allocated waiters (one malloc per producer).  Vyukov
 * pop frees the OLD head on each step; the popped item lives on the
 * new head until the NEXT pop or destroy.  We expose pop as a
 * copy-out API (edt_guid + slot) so callers never see the
 * about-to-be-freed pointer.
 */

#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>

#include "arts/memory/coherence.h"
#include "arts/utils/malloc.h"

void arts_pending_rw_queue_init(struct arts_pending_rw_queue_s *q) {
  atomic_store_explicit(&q->stub.next, (struct arts_db_rw_waiter_s *)NULL,
                        memory_order_relaxed);
  q->stub.edt_guid = 0;
  q->stub.slot = 0;
  atomic_store_explicit(&q->head, &q->stub, memory_order_relaxed);
  atomic_store_explicit(&q->tail, &q->stub, memory_order_relaxed);
}

void arts_pending_rw_queue_push(struct arts_pending_rw_queue_s *q,
                                struct arts_db_rw_waiter_s *w) {
  atomic_store_explicit(&w->next, (struct arts_db_rw_waiter_s *)NULL,
                        memory_order_relaxed);
  struct arts_db_rw_waiter_s *prev =
      atomic_exchange_explicit(&q->tail, w, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, w, memory_order_release);
}

bool arts_pending_rw_queue_pop(struct arts_pending_rw_queue_s *q,
                               arts_guid_t *out_edt, unsigned int *out_slot) {
  for (;;) {
    struct arts_db_rw_waiter_s *head =
        atomic_load_explicit(&q->head, memory_order_relaxed);
    struct arts_db_rw_waiter_s *next =
        atomic_load_explicit(&head->next, memory_order_acquire);
    if (next == NULL) {
      struct arts_db_rw_waiter_s *tail =
          atomic_load_explicit(&q->tail, memory_order_acquire);
      if (tail == head) {
        return false; /* truly empty */
      }
      /* Producer mid-link between xchg(tail) and store_release(prev->next).
       * Tight retry until the in-flight link lands (single consumer, ns
       * window).  Lock-free — no scheduler yield. */
      continue;
    }
    /* Copy payload out of `next` (it stays alive as the new head).
     * Then advance head to `next` and free the old head — except on
     * the first pop where the old head IS the embedded stub. */
    *out_edt = next->edt_guid;
    *out_slot = next->slot;
    atomic_store_explicit(&q->head, next, memory_order_relaxed);
    if (head != &q->stub) {
      arts_free(head);
    }
    return true;
  }
}

void arts_pending_rw_queue_drain(struct arts_pending_rw_queue_s *q,
                                 void (*cb)(arts_guid_t edt_guid,
                                            unsigned int slot, void *ctx),
                                 void *ctx) {
  arts_guid_t edt;
  unsigned int slot;
  while (arts_pending_rw_queue_pop(q, &edt, &slot)) {
    cb(edt, slot, ctx);
  }
}

void arts_pending_rw_queue_destroy(struct arts_pending_rw_queue_s *q) {
  /* Single-threaded at destroy: no concurrent producer/consumer.  Walk
   * head chain and free everything except the embedded stub. */
  struct arts_db_rw_waiter_s *h =
      atomic_load_explicit(&q->head, memory_order_relaxed);
  while (h != NULL) {
    struct arts_db_rw_waiter_s *n =
        atomic_load_explicit(&h->next, memory_order_relaxed);
    if (h != &q->stub) {
      arts_free(h);
    }
    h = n;
  }
  atomic_store_explicit(&q->head, (struct arts_db_rw_waiter_s *)NULL,
                        memory_order_relaxed);
  atomic_store_explicit(&q->tail, (struct arts_db_rw_waiter_s *)NULL,
                        memory_order_relaxed);
}
