/* SPDX-License-Identifier: Apache-2.0
 *
 * last_sent_version dense map — owner-side per-rank dedup of the version each
 * rank last received.  Protocol-agnostic: every build links it (the ownership
 * protocols use it for DATA_RESPONSE dedup; MRMW uses it to skip redundant
 * sends), so it lives in its own TU rather than the ownership-only home.c (the
 * home OWNERSHIP_REQUEST FIFO, which MRMW does not link).
 *
 * Concurrency: each rank slot is an independent _Atomic(uint64_t) accessed via
 * atomic load/store and a CAS-loop monotonic-max for advance.  No cross-slot
 * invariant; concurrent advances on the same slot are safe (a higher value
 * wins the CAS, losers are harmless).
 */

#include "arts/coherence/home.h"

#include <stdatomic.h>
#include <stdlib.h>

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
