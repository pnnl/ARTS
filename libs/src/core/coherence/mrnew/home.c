/* SPDX-License-Identifier: Apache-2.0
 *
 * Home-side coherence state implementations.  See coherence_home.h.
 *
 * Consolidates the ownership-protocol home-side state (MRNEW today, MRSW later;
 * not linked under MRMW, which carries no ownership lease):
 *   - home OWNERSHIP_REQUEST FIFO (lockreq Vyukov MPSC queue) + home-directory
 *     init/teardown
 *   - the per-cache pending_rw Treiber stack (cache-side RW waiter chain)
 *   - the bit-packed atomic rank bit-set (lazy destroy fan-out roster)
 * The protocol-agnostic last_sent_version dense map moved to rank_u64_map.c
 * (linked into every build, including MRMW).
 */

#include "arts/coherence/home.h"

#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>

#include "arts/coherence/coherence.h"
#include "arts/utils/malloc.h"

/*--- pending_rw home FIFO (Vyukov MPSC) ---------------------------------
 *
 * Standard Vyukov MPSC FIFO storing an unsigned int requester rank.  The
 * cache-side RW waiter chain (arts_pending_rw_queue_* below) is a separate
 * Treiber stack; this home queue stays Vyukov because the baton holder pops
 * exactly one requester at a time (the next ownership target) — an ordering a
 * LIFO stack cannot provide.  The embedded stub sentinel in
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
 * Bit-packed atomic rank bit-set.  See coherence_home.h / rank_bitset.h.
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
 * stack suffices here while the home lockreq queue (drain-one) stays Vyukov. */

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
