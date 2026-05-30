/* SPDX-License-Identifier: Apache-2.0
 *
 * Home metadata helper implementations.  See coherence_home.h.
 */

#include "arts/memory/coherence_home.h"

#include <sched.h>
#include <stdlib.h>
#ifdef ARTS_MEMORY_MODEL_LRC
#include "arts/memory/coherence_readers.h"
#endif

/*--- pending_rw home FIFO (Vyukov MPSC) ---------------------------------
 *
 * Algorithm mirrors coherence_pending_rw.c (cache-side) but stores only
 * an unsigned int rank instead of edt_guid+slot.  The embedded stub
 * sentinel in arts_home_lockreq_queue_s is the permanent queue sentinel;
 * it is never malloc'd or free'd separately.
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

/*--- pending_ro_forwards queue (Vyukov MPSC, LRC only) ------------------
 *
 * Algorithm mirrors arts_home_lockreq_queue_* above but carries both a
 * requester_rank and an opaque waiter_addr.  The embedded stub sentinel in
 * arts_home_pending_ro_queue_s is permanent; it is never malloc'd or free'd
 * separately.
 *
 * Memory ordering: identical discipline to arts_home_lockreq_queue_*:
 *   push  — acq_rel on tail exchange, release on prev->next store.
 *   pop   — acquire on head load, acquire on head->next load.
 */

void arts_home_pending_ro_queue_init(struct arts_home_pending_ro_queue_s *q) {
#ifdef ARTS_MEMORY_MODEL_LRC
  atomic_store_explicit(&q->stub.next, (struct arts_home_ro_node_s *)NULL,
                        memory_order_relaxed);
  q->stub.requester_rank = 0;
  q->stub.waiter_addr = NULL;
  atomic_store_explicit(&q->tail, &q->stub, memory_order_relaxed);
  atomic_store_explicit(&q->head, &q->stub, memory_order_relaxed);
#else
  q->_reserved = NULL;
#endif
}

void arts_home_pending_ro_queue_destroy(
    struct arts_home_pending_ro_queue_s *q) {
#ifdef ARTS_MEMORY_MODEL_LRC
  struct arts_home_ro_node_s *cur =
      atomic_load_explicit(&q->head, memory_order_relaxed);
  while (cur != NULL) {
    struct arts_home_ro_node_s *nxt =
        atomic_load_explicit(&cur->next, memory_order_relaxed);
    if (cur != &q->stub) {
      free(cur);
    }
    cur = nxt;
  }
  atomic_store_explicit(&q->tail, (struct arts_home_ro_node_s *)NULL,
                        memory_order_relaxed);
  atomic_store_explicit(&q->head, (struct arts_home_ro_node_s *)NULL,
                        memory_order_relaxed);
#else
  (void)q;
#endif
}

#ifdef ARTS_MEMORY_MODEL_LRC
void arts_home_pending_ro_queue_push(struct arts_home_pending_ro_queue_s *q,
                                     unsigned int requester_rank,
                                     void *waiter_addr) {
  struct arts_home_ro_node_s *n =
      (struct arts_home_ro_node_s *)malloc(sizeof(*n));
  n->requester_rank = requester_rank;
  n->waiter_addr = waiter_addr;
  atomic_store_explicit(&n->next, (struct arts_home_ro_node_s *)NULL,
                        memory_order_relaxed);
  struct arts_home_ro_node_s *prev =
      atomic_exchange_explicit(&q->tail, n, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, n, memory_order_release);
}

bool arts_home_pending_ro_queue_pop(struct arts_home_pending_ro_queue_s *q,
                                    unsigned int *out_rank,
                                    void **out_waiter_addr) {
  for (;;) {
    struct arts_home_ro_node_s *head =
        atomic_load_explicit(&q->head, memory_order_acquire);
    struct arts_home_ro_node_s *next =
        atomic_load_explicit(&head->next, memory_order_acquire);

    if (next == NULL) {
      struct arts_home_ro_node_s *tail =
          atomic_load_explicit(&q->tail, memory_order_acquire);
      if (head == tail) {
        return false; /* truly empty */
      }
      /* Producer mid-link: tight retry until the in-flight
       * store_release(prev->next) lands (single consumer, ns window).
       * Lock-free — no scheduler yield. */
      continue;
    }

    *out_rank = next->requester_rank;
    *out_waiter_addr = next->waiter_addr;
    atomic_store_explicit(&q->head, next, memory_order_release);
    if (head != &q->stub) {
      free(head);
    }
    return true;
  }
}
#endif /* ARTS_MEMORY_MODEL_LRC */

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

/*--- arts_db_home_s lifecycle -------------------------------------------*/

void arts_db_home_init(struct arts_db_home_s *home, unsigned int rw_holder,
                       unsigned int nranks) {
  /* Embedded by value in the cache: the caller zeroed it via calloc.
   * Common fields: destroy baton + ack counter. */
  atomic_store_explicit(&home->destroy_in_flight, 0, memory_order_relaxed);
  atomic_store_explicit(&home->destroy_ack_outstanding, 0,
                        memory_order_relaxed);
#if defined(ARTS_MEMORY_MODEL_LRC)
  atomic_store_explicit(&home->rw_holder, rw_holder, memory_order_relaxed);
  arts_home_lockreq_queue_init(&home->pending_rw);
  atomic_store_explicit(&home->invalidate_in_flight, 0, memory_order_relaxed);
  arts_home_pending_ro_queue_init(&home->pending_ro_forwards);
  arts_readers_bits_init(&home->readers, nranks);
  home->pending_install_owner = 0;
#elif defined(ARTS_MEMORY_MODEL_LC)
  /* LC: only last_sent_version.  rw_holder param is unused in LC —
   * DB has no exclusive owner. */
  (void)rw_holder;
  home->last_sent_version = arts_rank_u64_map_create(nranks);
#else
  /* RC */
  atomic_store_explicit(&home->rw_holder, rw_holder, memory_order_relaxed);
  arts_home_lockreq_queue_init(&home->pending_rw);
  atomic_store_explicit(&home->invalidate_in_flight, 0, memory_order_relaxed);
  /* RO forward queue: Vyukov MPSC in LRC builds; no-op stub in RC. */
  arts_home_pending_ro_queue_init(&home->pending_ro_forwards);
  home->last_sent_version = arts_rank_u64_map_create(nranks);
#endif
}

void arts_db_home_teardown(struct arts_db_home_s *home) {
  if (home == NULL) {
    return;
  }
#if defined(ARTS_MEMORY_MODEL_LRC)
  arts_home_lockreq_queue_destroy(&home->pending_rw);
  arts_home_pending_ro_queue_destroy(&home->pending_ro_forwards);
  arts_readers_bits_destroy(&home->readers);
#elif defined(ARTS_MEMORY_MODEL_LC)
  arts_rank_u64_map_destroy(home->last_sent_version);
#else
  /* RC */
  arts_home_lockreq_queue_destroy(&home->pending_rw);
  arts_home_pending_ro_queue_destroy(&home->pending_ro_forwards);
  arts_rank_u64_map_destroy(home->last_sent_version);
#endif
  /* No free: the home block is embedded by value in the cache. */
}

#ifdef ARTS_MEMORY_MODEL_LRC
/*--- last_sent_version map serialization / deserialization ---------------*/

#include "arts/transport/protocol.h"
#include <string.h>

size_t arts_rank_u64_map_serialize(const struct arts_rank_to_u64_map_s *m,
                                   void *out) {
  uint32_t *count_field = (uint32_t *)out;
  struct arts_remote_rank_version_pair_s *entries =
      (struct arts_remote_rank_version_pair_s *)((char *)out +
                                                 sizeof(uint32_t) * 2);
  uint32_t n = 0;
  for (unsigned int r = 0; r < m->nranks; r++) {
    uint64_t v = atomic_load_explicit(&m->slots[r], memory_order_acquire);
    if (v == 0) {
      continue;
    }
    entries[n].rank = (uint32_t)r;
    entries[n].pad = 0;
    entries[n].version = v;
    n++;
  }
  count_field[0] = n;
  count_field[1] = 0; /* alignment pad */
  return sizeof(uint32_t) * 2 + (size_t)n * sizeof(*entries);
}

struct arts_rank_to_u64_map_s *
arts_rank_u64_map_deserialize(const void *in, size_t size,
                              unsigned int nranks) {
  (void)size; /* used by debug assertions; production ignores it */
  struct arts_rank_to_u64_map_s *m = arts_rank_u64_map_create(nranks);
  const uint32_t *count_field = (const uint32_t *)in;
  uint32_t n = count_field[0];
  const struct arts_remote_rank_version_pair_s *entries =
      (const struct arts_remote_rank_version_pair_s *)((const char *)in +
                                                       sizeof(uint32_t) * 2);
  for (uint32_t i = 0; i < n; i++) {
    arts_rank_u64_map_set(m, (unsigned int)entries[i].rank, entries[i].version);
  }
  return m;
}
#endif /* ARTS_MEMORY_MODEL_LRC */
