/* SPDX-License-Identifier: Apache-2.0
 *
 * Home-side coherence state implementations (MRSW).  See coherence/home.h.
 *
 * MRSW is rank-granular on the MRNEW engine, so this home-side state is
 * byte-for-byte MRNEW:
 *   - home OWNERSHIP_REQUEST FIFO (lockreq Vyukov MPSC queue, rank-granular) +
 *     home-directory init/teardown;
 *   - the bit-packed atomic rank bit-set (lazy destroy fan-out roster).
 * The per-cache RW-waiter FIFO (cache.pending_rw) is a SEPARATE Vyukov MPSC
 * consumed pop-one — it lives in coherence/mrsw/waiter_queue.c, not here.  The
 * protocol-agnostic last_sent_version dense map lives in rank_u64_map.c.
 */

#include "arts/coherence/home.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>

#include "arts/coherence/coherence.h"
#include "arts/utils/malloc.h"

/*--- pending_rw home FIFO (Vyukov MPSC, rank-granular) ------------------
 *
 * Standard Vyukov MPSC FIFO storing an unsigned int requester rank.  The baton
 * holder pops exactly one requester at a time (the next ownership target) — an
 * ordering a LIFO stack cannot provide.  The embedded stub sentinel in
 * arts_home_lockreq_queue_s is the permanent queue sentinel; it is never
 * malloc'd or free'd separately.
 *
 * Field convention (standard Vyukov MPSC naming):
 *   tail — producer end; push() swaps the new node in here (acq_rel).
 *   head — consumer end; pop() reads head->next and advances head.
 *
 * Memory ordering:
 *   push  — acq_rel on the tail exchange (linearization point), then
 *            release on prev->next store (makes payload visible to
 *            consumer after the tail swap).
 *   pop   — acquire on head->next load (pairs with push's release store);
 *            acquire on head load (defensive: guards against future
 *            weakening of the single-consumer baton invariant).
 *   empty — acquire on head load + acquire on next load (conservative
 *            snapshot; safe for a single consumer).
 */

void arts_home_lockreq_queue_init(struct arts_home_lockreq_queue_s *q) {
  atomic_store_explicit(&q->stub.next, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  q->stub.rank = 0;
  atomic_store_explicit(&q->tail, &q->stub, memory_order_relaxed);
  atomic_store_explicit(&q->head, &q->stub, memory_order_relaxed);
}

void arts_home_lockreq_queue_push(struct arts_home_lockreq_queue_s *q,
                                  unsigned int rank) {
  struct arts_home_lockreq_node_s *n =
      (struct arts_home_lockreq_node_s *)malloc(sizeof(*n));
  n->rank = rank;
  atomic_store_explicit(&n->next, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  /* Swap n onto the tail (producer end).  The previous tail becomes our
   * predecessor; link it forward to n so the consumer can reach n once
   * it advances past the predecessor. */
  struct arts_home_lockreq_node_s *prev =
      atomic_exchange_explicit(&q->tail, n, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, n, memory_order_release);
}

bool arts_home_lockreq_queue_pop(struct arts_home_lockreq_queue_s *q,
                                 unsigned int *out_rank) {
  for (;;) {
    /* Acquire on head load: defensive barrier so a new baton holder sees
     * all prior consumer writes to head, even if the baton CAS ordering
     * ever weakens. */
    struct arts_home_lockreq_node_s *head =
        atomic_load_explicit(&q->head, memory_order_acquire);
    struct arts_home_lockreq_node_s *next =
        atomic_load_explicit(&head->next, memory_order_acquire);

    if (next == NULL) {
      /* Either truly empty, or a producer is mid-push between its
       * atomic_exchange on tail and its store_release on prev->next.
       * Check if tail == head to distinguish. */
      struct arts_home_lockreq_node_s *tail =
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
    *out_rank = next->rank;
    atomic_store_explicit(&q->head, next, memory_order_release);
    if (head != &q->stub) {
      free(head);
    }
    return true;
  }
}

bool arts_home_lockreq_queue_peek(const struct arts_home_lockreq_queue_s *q,
                                  unsigned int *out_rank) {
  /* Cast away const for atomic load — the queue isn't mutated (no head
   * advance, no node free).  Single consumer (the baton holder). */
  struct arts_home_lockreq_node_s *head = atomic_load_explicit(
      (_Atomic(struct arts_home_lockreq_node_s *) *)&q->head,
      memory_order_acquire);
  struct arts_home_lockreq_node_s *next =
      atomic_load_explicit(&head->next, memory_order_acquire);
  if (next == NULL) {
    return false; /* empty, or a producer mid-link — treat as no front yet */
  }
  *out_rank = next->rank;
  return true;
}

bool arts_home_lockreq_queue_empty(const struct arts_home_lockreq_queue_s *q) {
  /* Cast away const for atomic load — the value isn't modified. */
  struct arts_home_lockreq_node_s *head = atomic_load_explicit(
      (_Atomic(struct arts_home_lockreq_node_s *) *)&q->head,
      memory_order_acquire);
  struct arts_home_lockreq_node_s *next =
      atomic_load_explicit(&head->next, memory_order_acquire);
  if (next != NULL) {
    return false;
  }
  struct arts_home_lockreq_node_s *tail = atomic_load_explicit(
      (_Atomic(struct arts_home_lockreq_node_s *) *)&q->tail,
      memory_order_acquire);
  return head == tail;
}

void arts_home_lockreq_queue_destroy(struct arts_home_lockreq_queue_s *q) {
  /* Single-threaded at destroy: walk the head chain freeing every node
   * except the embedded stub sentinel. */
  struct arts_home_lockreq_node_s *cur =
      atomic_load_explicit(&q->head, memory_order_relaxed);
  while (cur != NULL) {
    struct arts_home_lockreq_node_s *nxt =
        atomic_load_explicit(&cur->next, memory_order_relaxed);
    if (cur != &q->stub) {
      free(cur);
    }
    cur = nxt;
  }
  /* Poison the pointers to catch use-after-destroy. */
  atomic_store_explicit(&q->tail, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  atomic_store_explicit(&q->head, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
}

/*--- rank bit-set ----------------------------------------------------
 *
 * Bit-packed atomic rank bit-set.  See coherence/home.h / rank_bitset.h.
 * Used only in lazy builds — the eager protocol reuses the per-rank version
 * map for the same purpose (set membership = nonzero entry). */

void arts_rank_bitset_init(struct arts_rank_bitset_s *r, unsigned int nranks) {
  r->nranks = nranks;
  r->nwords = (nranks + 63) / 64;
  r->words = (_Atomic(uint64_t) *)calloc(r->nwords, sizeof(_Atomic(uint64_t)));
}

void arts_rank_bitset_destroy(struct arts_rank_bitset_s *r) {
  free(r->words);
  r->words = NULL;
  r->nwords = 0;
}

bool arts_rank_bitset_set(struct arts_rank_bitset_s *r, unsigned int rank) {
  if (rank >= r->nranks) {
    return false;
  }
  unsigned int word_idx = rank / 64;
  uint64_t bit = (uint64_t)1 << (rank % 64);
  uint64_t prev =
      atomic_fetch_or_explicit(&r->words[word_idx], bit, memory_order_acq_rel);
  return (prev & bit) == 0;
}

void arts_rank_bitset_for_each(const struct arts_rank_bitset_s *r,
                               void (*cb)(unsigned int rank, void *ctx),
                               void *ctx) {
  for (unsigned int w = 0; w < r->nwords; w++) {
    uint64_t snap = atomic_load_explicit((&r->words[w]), memory_order_acquire);
    while (snap) {
      unsigned int b = (unsigned int)__builtin_ctzll(snap);
      cb((w * 64) + b, ctx);
      snap &= snap - 1;
    }
  }
}
