/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol shared home-side infrastructure.
 *
 * Defines: arts_home_lockreq_queue_{init,push,pop,peek,empty,destroy},
 *          arts_rank_bitset_{init,set,for_each,destroy},
 *          arts_db_home_init, arts_db_home_teardown.
 *
 * arts_handler_db_destroy lives in the timing TU (eager.c): it fans out over
 * the wire, and this TU must stay linkable standalone (pure queue/bitset/
 * lifecycle) for the whitebox unit tests.
 *
 * This TU carries the pieces shared by any future timing variant: the home
 * request FIFO, the rank bitset, the home-directory lifecycle, and the
 * DB-destroy fan-out handler.  The arbiters and all handler/sender bodies
 * live in the timing TU (eager.c).
 *
 * The lockreq queue and rank_bitset bodies are verbatim from rcu/home.c
 * (same Vyukov MPSC + bit-packed bitset shapes; MSI reuses the same struct
 * shapes from msi/types.h).
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=MSI.
 */

/* msi/types.h must precede home.h: it defines arts_home_lockreq_node_s and
 * arts_home_lockreq_queue_s for the MSI build. */
#include "arts/coherence/msi/types.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "arts/coherence/coherence.h"
#include "arts/coherence/home.h"
#include "arts/utils/malloc.h"

/* ===== Home REQUEST FIFO (Vyukov MPSC) ================================== */

void arts_home_lockreq_queue_init(struct arts_home_lockreq_queue_s *q) {
  atomic_store_explicit(&q->stub.next, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  q->stub.rank = 0;
  atomic_store_explicit(&q->tail, &q->stub, memory_order_relaxed);
  atomic_store_explicit(&q->head, &q->stub, memory_order_relaxed);
}

void arts_home_lockreq_queue_push(struct arts_home_lockreq_queue_s *q,
                                  unsigned int rank,
                                  const struct arts_rdzv_landing_s *rdzv) {
  struct arts_home_lockreq_node_s *n =
      (struct arts_home_lockreq_node_s *)malloc(sizeof(*n));
  n->rank = rank;
  if (rdzv != NULL) {
    n->rdzv = *rdzv;
  } else {
    n->rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  }
  atomic_store_explicit(&n->next, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  struct arts_home_lockreq_node_s *prev =
      atomic_exchange_explicit(&q->tail, n, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, n, memory_order_release);
}

bool arts_home_lockreq_queue_pop(struct arts_home_lockreq_queue_s *q,
                                 unsigned int *out_rank,
                                 struct arts_rdzv_landing_s *out_rdzv) {
  for (;;) {
    struct arts_home_lockreq_node_s *head =
        atomic_load_explicit(&q->head, memory_order_acquire);
    struct arts_home_lockreq_node_s *next =
        atomic_load_explicit(&head->next, memory_order_acquire);
    if (next == NULL) {
      struct arts_home_lockreq_node_s *tail =
          atomic_load_explicit(&q->tail, memory_order_acquire);
      if (head == tail) {
        return false;
      }
      continue;
    }
    *out_rank = next->rank;
    if (out_rdzv != NULL) {
      *out_rdzv = next->rdzv;
    }
    atomic_store_explicit(&q->head, next, memory_order_release);
    if (head != &q->stub) {
      free(head);
    }
    return true;
  }
}

bool arts_home_lockreq_queue_peek(const struct arts_home_lockreq_queue_s *q,
                                  unsigned int *out_rank,
                                  struct arts_rdzv_landing_s *out_rdzv) {
  struct arts_home_lockreq_node_s *head = atomic_load_explicit(
      (_Atomic(struct arts_home_lockreq_node_s *) *)&q->head,
      memory_order_acquire);
  struct arts_home_lockreq_node_s *next =
      atomic_load_explicit(&head->next, memory_order_acquire);
  if (next == NULL) {
    return false;
  }
  *out_rank = next->rank;
  if (out_rdzv != NULL) {
    *out_rdzv = next->rdzv;
  }
  return true;
}

bool arts_home_lockreq_queue_empty(const struct arts_home_lockreq_queue_s *q) {
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
  atomic_store_explicit(&q->tail, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  atomic_store_explicit(&q->head, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
}

/* ===== rank bit-set ===================================================== */

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
    uint64_t snap = atomic_load_explicit(&r->words[w], memory_order_acquire);
    while (snap) {
      unsigned int b = (unsigned int)__builtin_ctzll(snap);
      cb(w * 64 + b, ctx);
      snap &= snap - 1;
    }
  }
}

/* ===== Home-directory lifecycle ========================================= */

void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks) {
#ifdef ARTS_TIMING_LAZY
  /* The creator boots as the owner holding the canonical copy in its own cache
   * word; the directory's w counts only ranks waiting for ownership, so it
   * starts at 0. */
  atomic_store_explicit(&db->dir_state,
                        MSI_LAZY_DIR_MAKE(0u, 0u, 0u, (uint32_t)rw_holder, 0u),
                        memory_order_relaxed);
  atomic_store_explicit(&db->cur_round, (struct arts_db_msi_round_req_s *)NULL,
                        memory_order_relaxed);
  atomic_store_explicit(&db->round_pending, 0u, memory_order_relaxed);
  arts_mpsc_init(&db->round_q);
#else
  /* Seed the creator's write tenure: arts_db_create defaults to an RW
   * acquire (OCR contract), so the directory starts with one writer (the
   * creator rank) owning the tenure — its matching final writeback drives
   * w back to 0 and returns ownership.  NO_ACQUIRE resets this to the idle
   * state in arts_db_create. */
  atomic_store_explicit(
      &db->dir_state, MSI_DIR_MAKE(0u, 0u, (uint32_t)rw_holder, 1u),
      memory_order_relaxed);
  atomic_store_explicit(&db->hver, 1u, memory_order_relaxed);
  atomic_store_explicit(&db->opening_pending, false, memory_order_relaxed);
  arts_lf_stack_init(&db->wb_queue);
#endif
  arts_home_lockreq_queue_init(&db->rw_waiters);
  arts_rank_bitset_init(&db->roster, nranks);
  arts_rank_bitset_init(&db->cached_ranks, nranks);
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_lockreq_queue_destroy(&db->rw_waiters);
  /* Drain + free anything still queued (single-threaded at teardown — the
   * route slot is already absent, no concurrent push). */
#ifdef ARTS_TIMING_LAZY
  for (;;) {
    arts_lf_link_t *node = arts_mpsc_pop(&db->round_q);
    if (node == NULL) {
      break;
    }
    arts_free(ARTS_CONTAINER_OF(node, struct arts_db_msi_round_req_s, link));
  }
  struct arts_db_msi_round_req_s *open_req = atomic_exchange_explicit(
      &db->cur_round, (struct arts_db_msi_round_req_s *)NULL,
      memory_order_acq_rel);
  arts_free(open_req);
#else
  arts_lf_link_t *node = arts_lf_stack_drain(&db->wb_queue);
  while (node != NULL) {
    arts_lf_link_t *nx = atomic_load_explicit(&node->next, memory_order_relaxed);
    arts_free(ARTS_CONTAINER_OF(node, struct arts_db_msi_wb_s, link));
    node = nx;
  }
#endif
  arts_rank_bitset_destroy(&db->roster);
  arts_rank_bitset_destroy(&db->cached_ranks);
}
