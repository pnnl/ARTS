/* SPDX-License-Identifier: Apache-2.0
 *
 * Home-side coherence state implementations.  See coherence_home.h.
 *
 * Consolidates three home-side concerns:
 *   - home metadata queues / maps / lifecycle (lockreq queue,
 *     last_sent_version dense map, home-directory init/teardown)
 *   - the per-cache pending_rw Vyukov MPSC queue (cache-side RW waiter chain)
 *   - the bit-packed atomic rank bit-set (LRC destroy fan-out roster)
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
 * Algorithm mirrors the cache-side pending_rw queue (see
 * arts_pending_rw_queue_* below) but stores only an unsigned int rank
 * instead of edt_guid+slot.  The embedded stub sentinel in
 * arts_home_lockreq_queue_s is the permanent queue sentinel; it is never
 * malloc'd or free'd separately.
 *
 * Field convention (standard Vyukov MPSC naming, matching cache-side):
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

/*--- last_sent_version dense map ----------------------------------------*/

struct arts_rank_to_u64_map_s *arts_rank_u64_map_create(unsigned int nranks) {
  struct arts_rank_to_u64_map_s *m =
      (struct arts_rank_to_u64_map_s *)malloc(sizeof(*m));
  m->nranks = nranks;
  m->slots =
      (_Atomic(uint64_t) *)calloc((size_t)nranks, sizeof(_Atomic(uint64_t)));
  return m;
}

void arts_rank_u64_map_destroy(struct arts_rank_to_u64_map_s *m) {
  if (m == NULL) {
    return;
  }
  free(m->slots);
  free(m);
}

uint64_t arts_rank_u64_map_get(const struct arts_rank_to_u64_map_s *m,
                               unsigned int rank) {
  if (rank >= m->nranks) {
    return 0;
  }
  return atomic_load_explicit(&m->slots[rank], memory_order_acquire);
}

void arts_rank_u64_map_set(struct arts_rank_to_u64_map_s *m, unsigned int rank,
                           uint64_t value) {
  if (rank >= m->nranks) {
    return;
  }
  atomic_store_explicit(&m->slots[rank], value, memory_order_release);
}

bool arts_rank_u64_map_advance(struct arts_rank_to_u64_map_s *m,
                               unsigned int rank, uint64_t value) {
  if (rank >= m->nranks) {
    return false;
  }
  uint64_t old = atomic_load_explicit(&m->slots[rank], memory_order_acquire);
  while (1) {
    if (value <= old) {
      return false;
    }
    if (atomic_compare_exchange_weak_explicit(&m->slots[rank], &old, value,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      return true;
    }
    /* old refreshed by failed CAS; retry with new snapshot */
  }
}

/*--- rank bit-set ----------------------------------------------------
 *
 * Bit-packed atomic rank bit-set.  See coherence_home.h / rank_bitset.h.
 * Used only in LRC builds — RC reuses the per-rank version map for the same
 * purpose (set membership = nonzero entry). */

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

/*--- per-cache pending_rw queue (Vyukov MPSC) ---------------------------
 *
 * The per-cache RW waiter chain.  Replaces the prior Harris-style
 * marked-list.
 *
 * Algorithm: the same lock-free MPSC enqueue / single-consumer drain used by
 * the route-table OoO list, specialized to a typed waiter
 * (arts_db_rw_waiter_s) instead of a generic (data, next) node.
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
 * about-to-be-freed pointer. */

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
