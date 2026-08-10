/* SPDX-License-Identifier: Apache-2.0
 *
 * RWLOCK protocol shared home-side infrastructure.
 *
 * Defines: arts_home_grantreq_queue_{init,push,pop,peek,empty,destroy},
 *          arts_rank_bitset_{init,set,for_each,destroy},
 *          arts_db_home_init, arts_db_home_teardown,
 *          arts_handler_db_destroy.
 *
 * This TU contains only the shared pieces that both the HOME and OWNER placement
 * variants need: the home ownership-request FIFO, the rank bitset, the
 * home-directory lifecycle, and the DB-destroy fan-out handler.  Timing-
 * specific arbiters (excl_compute_next, cache_compute_next) and all
 * acquire/release handler/sender bodies live in the per-placement TU (home.c
 * or owner.c).
 *
 * The grantreq queue and rank_bitset bodies are verbatim from rcu/home.c
 * (same Vyukov MPSC + bit-packed bitset shapes; RWLOCK reuses the same struct
 * definitions from lock/types.h).
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=RWLOCK.
 */

/* lock/types.h must precede home.h: it defines arts_home_grantreq_node_s and
 * arts_home_grantreq_queue_s for the RWLOCK build (home.h declares functions
 * that take these types by pointer, and coherence.h embeds them in
 * arts_db_s). */
#include "arts/coherence/excl/types.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/directory.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/system/identity.h"
#include "arts/transport/net.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/counter/Preamble.h"

/* ===== Home OWNERSHIP-REQUEST FIFO (Vyukov MPSC) =======================
 *
 * Standard Vyukov MPSC FIFO storing an unsigned int requester rank.  The
 * embedded stub sentinel in arts_home_grantreq_queue_s is the permanent queue
 * sentinel; it is never malloc'd or free'd separately.
 */

void arts_home_grantreq_queue_init(struct arts_home_grantreq_queue_s *q) {
  atomic_store_explicit(&q->stub.next, (struct arts_home_grantreq_node_s *)NULL,
                        memory_order_relaxed);
  q->stub.rank = 0;
  atomic_store_explicit(&q->tail, &q->stub, memory_order_relaxed);
  atomic_store_explicit(&q->head, &q->stub, memory_order_relaxed);
}

void arts_home_grantreq_queue_push(struct arts_home_grantreq_queue_s *q,
                                  unsigned int rank,
                                  const struct arts_rdzv_landing_s *rdzv) {
  /* Reached only when the turn could not be granted on arrival. */
  INCREMENT_NUM_EXCL_QUEUE_WAIT_BY(1);
  struct arts_home_grantreq_node_s *n =
      (struct arts_home_grantreq_node_s *)malloc(sizeof(*n));
  n->rank = rank;
  if (rdzv != NULL) {
    n->rdzv = *rdzv;
  } else {
    n->rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  }
  atomic_store_explicit(&n->next, (struct arts_home_grantreq_node_s *)NULL,
                        memory_order_relaxed);
  struct arts_home_grantreq_node_s *prev =
      atomic_exchange_explicit(&q->tail, n, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, n, memory_order_release);
}

bool arts_home_grantreq_queue_pop(struct arts_home_grantreq_queue_s *q,
                                 unsigned int *out_rank,
                                 struct arts_rdzv_landing_s *out_rdzv) {
  for (;;) {
    struct arts_home_grantreq_node_s *head =
        atomic_load_explicit(&q->head, memory_order_acquire);
    struct arts_home_grantreq_node_s *next =
        atomic_load_explicit(&head->next, memory_order_acquire);
    if (next == NULL) {
      struct arts_home_grantreq_node_s *tail =
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

bool arts_home_grantreq_queue_peek(const struct arts_home_grantreq_queue_s *q,
                                  unsigned int *out_rank,
                                  struct arts_rdzv_landing_s *out_rdzv) {
  struct arts_home_grantreq_node_s *head = atomic_load_explicit(
      (_Atomic(struct arts_home_grantreq_node_s *) *)&q->head,
      memory_order_acquire);
  struct arts_home_grantreq_node_s *next =
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

bool arts_home_grantreq_queue_empty(const struct arts_home_grantreq_queue_s *q) {
  struct arts_home_grantreq_node_s *head = atomic_load_explicit(
      (_Atomic(struct arts_home_grantreq_node_s *) *)&q->head,
      memory_order_acquire);
  struct arts_home_grantreq_node_s *next =
      atomic_load_explicit(&head->next, memory_order_acquire);
  if (next != NULL) {
    return false;
  }
  struct arts_home_grantreq_node_s *tail = atomic_load_explicit(
      (_Atomic(struct arts_home_grantreq_node_s *) *)&q->tail,
      memory_order_acquire);
  return head == tail;
}

void arts_home_grantreq_queue_destroy(struct arts_home_grantreq_queue_s *q) {
  struct arts_home_grantreq_node_s *cur =
      atomic_load_explicit(&q->head, memory_order_relaxed);
  while (cur != NULL) {
    struct arts_home_grantreq_node_s *nxt =
        atomic_load_explicit(&cur->next, memory_order_relaxed);
    if (cur != &q->stub) {
      free(cur);
    }
    cur = nxt;
  }
  atomic_store_explicit(&q->tail, (struct arts_home_grantreq_node_s *)NULL,
                        memory_order_relaxed);
  atomic_store_explicit(&q->head, (struct arts_home_grantreq_node_s *)NULL,
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

/* ===== RO waiter node (rank-granular RO queue entry) ===================
 * Placed before teardown and lock_home_grant which both use ARTS_CONTAINER_OF
 * on this type. */
struct arts_lock_ro_node_s {
  arts_lf_link_t link; /* FIRST */
  unsigned int rank;
};

/* ===== Home-directory lifecycle ========================================= */

void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks) {
  /* Seed the creator's RW hold: arts_db_create defaults to an RW acquire (OCR
   * contract), so the home arbiter starts with one writer (the creator rank)
   * holding the lock — its matching release drives the count back to 0.
   * NO_ACQUIRE resets this to the idle state in arts_db_create. */
#ifdef ARTS_RELEASE_PURGE
  (void)rw_holder; /* HOME tracks mode via lock_state phase bit, not a
                    * separate rw_holder field */
  atomic_store_explicit(&db->lock_state, LOCK_MAKE_STATE(0u, 1u, 0u),
                        memory_order_relaxed);
#else /* ARTS_RELEASE_RETAIN */
  /* OWNER: the home is a pure directory.  The creator is the first DATA owner
   * (cache-side: owner-bit + RW grant), but its RW hold/release are entirely
   * local (sticky — no REQUEST/RELEASE to home), so the home directory must NOT
   * count it.  Seed phase=IDLE, w=0, r=0, owner=creator: the lock is idle (no
   * home-tracked participant) and `owner` names the rank a future migrate/serve
   * FORWARD routes to.  This differs from HOME, where the creator's RELEASE
   * drives a seeded w=1 back to 0. */
  atomic_store_explicit(&db->lock_state,
                        LOCK_MAKE(LOCK_PHASE_IDLE, rw_holder, 0u, 0u),
                        memory_order_relaxed);
#endif
  arts_home_grantreq_queue_init(&db->rw_waiters);
  arts_lf_stack_init(&db->ro_waiters);
  arts_rank_bitset_init(&db->cached_ranks, nranks);
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_grantreq_queue_destroy(&db->rw_waiters);
  /* Drain + free any rank-nodes still on ro_waiters (single-threaded at
   * teardown — the route slot is already absent, no concurrent push). */
  arts_lf_link_t *node = arts_lf_stack_drain(&db->ro_waiters);
  while (node != NULL) {
    arts_lf_link_t *nx =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    arts_free(ARTS_CONTAINER_OF(node, struct arts_lock_ro_node_s, link));
    node = nx;
  }
  arts_rank_bitset_destroy(&db->cached_ranks);
}

/* ===== arts_handler_db_destroy =========================================
 * OOO_DB_DESTROY Cat-B body for the RWLOCK protocol.  Fan-out DESTROY_NOTIFY
 * to every rank in cached_ranks (the destroy roster), wake parked waiters,
 * then detach the route-table slot. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  /* Fan-out DESTROY_NOTIFY to every rank that ever acquired this DB.
   * Inline the bitset word-walk (arts_rank_bitset_for_each requires a
   * callback; C has no local functions, so we open-code the loop). */
  {
    const struct arts_rank_bitset_s *bs = &db->cached_ranks;
    for (unsigned int w = 0; w < bs->nwords; w++) {
      uint64_t snap = atomic_load_explicit(&bs->words[w], memory_order_acquire);
      while (snap) {
        unsigned int b = (unsigned int)__builtin_ctzll(snap);
        unsigned int rank = w * 64 + b;
        if (rank != self) {
          arts_send_db_cache_destroy(rank, a->db_guid);
        }
        snap &= snap - 1;
      }
    }
  }
  /* No rw_waiters drain here: the request queue is single-consumer (the CONFIRM
   * that pops the migrated requester is its only consumer, serialized by the
   * lock phase machine).  A legitimately destroyed DB has every RW acquire
   * released, so the queue is empty; and every rank that ever requested RW is
   * in cached_ranks (recorded at request time) and was just notified above.
   * Draining here would be a second, unsynchronized consumer of a
   * single-consumer queue — a use-after-free against a concurrent CONFIRM. */
  (void)arts_route_table_set_destroyed(a->db_guid);
}
