/* SPDX-License-Identifier: Apache-2.0
 *
 * MRSW per-cache RW-waiter FIFO (cache.pending_rw).
 *
 * A standard Vyukov MPSC FIFO storing an (edt_guid, slot) pair per waiter.
 * Mirrors the home lockreq Vyukov MPSC (see coherence/mrsw/home.c) but is
 * consumed one waiter at a time in FIFO order — the releaser pops exactly the
 * next waiter to grant — and exposes peek_empty for a release-time Dekker
 * re-check (has the producer enqueued a waiter we must wake?).  The embedded
 * stub sentinel in arts_db_rw_waiter_queue_s is the permanent queue sentinel;
 * it is never malloc'd or free'd separately.
 *
 * Field convention (standard Vyukov MPSC naming):
 *   tail — producer end; push() swaps the new node in here (acq_rel).
 *   head — consumer end; pop() reads head->next and advances head.
 *
 * Memory ordering:
 *   push       — acq_rel on the tail exchange (linearization point), then
 *                release on prev->next store (makes payload visible to the
 *                consumer after the tail swap).
 *   pop        — acquire on head->next load (pairs with push's release store);
 *                acquire on head load (defensive: guards against future
 *                weakening of the single-consumer invariant).
 *   peek_empty — acquire on head load + acquire on next load (conservative
 *                snapshot; safe for a single consumer).
 */

#include "arts/coherence/home.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>

#include "arts/coherence/coherence.h"

void arts_db_rw_waiter_queue_init(struct arts_db_rw_waiter_queue_s *q) {
  atomic_store_explicit(&q->stub.next, (struct arts_db_rw_waiter_node_s *)NULL,
                        memory_order_relaxed);
  q->stub.edt_guid = NULL_GUID;
  q->stub.slot = 0;
  atomic_store_explicit(&q->tail, &q->stub, memory_order_relaxed);
  atomic_store_explicit(&q->head, &q->stub, memory_order_relaxed);
}

void arts_db_rw_waiter_queue_push(struct arts_db_rw_waiter_queue_s *q,
                                  arts_guid_t edt_guid, unsigned int slot) {
  struct arts_db_rw_waiter_node_s *n =
      (struct arts_db_rw_waiter_node_s *)malloc(sizeof(*n));
  n->edt_guid = edt_guid;
  n->slot = slot;
  atomic_store_explicit(&n->next, (struct arts_db_rw_waiter_node_s *)NULL,
                        memory_order_relaxed);
  /* Swap n onto the tail (producer end).  The previous tail becomes our
   * predecessor; link it forward to n so the consumer can reach n once
   * it advances past the predecessor. */
  struct arts_db_rw_waiter_node_s *prev =
      atomic_exchange_explicit(&q->tail, n, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, n, memory_order_release);
}

bool arts_db_rw_waiter_queue_pop(struct arts_db_rw_waiter_queue_s *q,
                                 arts_guid_t *edt_guid_out,
                                 unsigned int *slot_out) {
  for (;;) {
    /* Acquire on head load: defensive barrier so a new consumer sees all
     * prior consumer writes to head, even if the consumer ordering ever
     * weakens. */
    struct arts_db_rw_waiter_node_s *head =
        atomic_load_explicit(&q->head, memory_order_acquire);
    struct arts_db_rw_waiter_node_s *next =
        atomic_load_explicit(&head->next, memory_order_acquire);

    if (next == NULL) {
      /* Either truly empty, or a producer is mid-push between its
       * atomic_exchange on tail and its store_release on prev->next.
       * Check if tail == head to distinguish. */
      struct arts_db_rw_waiter_node_s *tail =
          atomic_load_explicit(&q->tail, memory_order_acquire);
      if (head == tail) {
        return false; /* truly empty */
      }
      /* Producer mid-link: tight retry until the in-flight
       * store_release(prev->next) lands (single consumer, ns window).
       * Lock-free — no scheduler yield. */
      continue;
    }

    /* Copy the payload from `next` (the node that will become the new
     * head sentinel after we advance).  Advance head to `next`, then
     * free `head` — unless head IS the stub sentinel, which is embedded
     * in the queue struct and must never be freed. */
    *edt_guid_out = next->edt_guid;
    *slot_out = next->slot;
    atomic_store_explicit(&q->head, next, memory_order_release);
    if (head != &q->stub) {
      free(head);
    }
    return true;
  }
}

bool arts_db_rw_waiter_queue_peek_empty(
    const struct arts_db_rw_waiter_queue_s *q) {
  /* Cast away const for atomic load — the queue isn't mutated (no head
   * advance, no node free).  Single consumer.  A producer mid-link reads as
   * "no front yet" (next == NULL) and is reported empty: conservative for a
   * release-time re-check, which re-runs after the producer's link lands. */
  struct arts_db_rw_waiter_node_s *head = atomic_load_explicit(
      (_Atomic(struct arts_db_rw_waiter_node_s *) *)&q->head,
      memory_order_acquire);
  struct arts_db_rw_waiter_node_s *next =
      atomic_load_explicit(&head->next, memory_order_acquire);
  return next == NULL;
}

void arts_db_rw_waiter_queue_destroy(struct arts_db_rw_waiter_queue_s *q) {
  /* Single-threaded at destroy: walk the head chain freeing every node
   * except the embedded stub sentinel. */
  struct arts_db_rw_waiter_node_s *cur =
      atomic_load_explicit(&q->head, memory_order_relaxed);
  while (cur != NULL) {
    struct arts_db_rw_waiter_node_s *nxt =
        atomic_load_explicit(&cur->next, memory_order_relaxed);
    if (cur != &q->stub) {
      free(cur);
    }
    cur = nxt;
  }
  /* Poison the pointers to catch use-after-destroy. */
  atomic_store_explicit(&q->tail, (struct arts_db_rw_waiter_node_s *)NULL,
                        memory_order_relaxed);
  atomic_store_explicit(&q->head, (struct arts_db_rw_waiter_node_s *)NULL,
                        memory_order_relaxed);
}
