/* SPDX-License-Identifier: Apache-2.0
 *
 * cached_version dense map — owner-side per-rank dedup of the version each
 * rank last received.  Protocol-agnostic: every build links it (the ownership
 * protocols use it for SNAPSHOT_RESPONSE dedup; WRF_VAL uses it to skip redundant
 * sends), so it lives in its own TU rather than the ownership-only val/directory.c (the
 * home GRANT_REQUEST FIFO, which WRF_VAL does not link).
 *
 * Concurrency: each rank slot is an independent _Atomic(uint64_t) accessed via
 * atomic load/store and a CAS-loop monotonic-max for advance.  No cross-slot
 * invariant; concurrent advances on the same slot are safe (a higher value
 * wins the CAS, losers are harmless).
 */

#include "arts/coherence/directory.h"
#include "arts/transport/protocol.h" /* arts_msg_rank_version_pair_s */

#include <stdatomic.h>
#include <stdint.h>
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

/* ===== serialize / deserialize (ownership-transfer wire payload) =====
 * Shared by both VAL write policies (the owner→owner GRANT_RESPONSE carries the
 * owner-side map).  Layout: count(u32) + pad(u32) + count pairs. */
size_t arts_rank_u64_map_serialize(const struct arts_rank_to_u64_map_s *m,
                                   void *out) {
  uint32_t *count_field = (uint32_t *)out;
  struct arts_msg_rank_version_pair_s *entries =
      (struct arts_msg_rank_version_pair_s *)((char *)out +
                                              (sizeof(uint32_t) * 2));
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
  return (sizeof(uint32_t) * 2) + ((size_t)n * sizeof(*entries));
}

struct arts_rank_to_u64_map_s *
arts_rank_u64_map_deserialize(const void *in, size_t size,
                              unsigned int nranks) {
  (void)size; /* used by debug assertions; production ignores it */
  struct arts_rank_to_u64_map_s *m = arts_rank_u64_map_create(nranks);
  const uint32_t *count_field = (const uint32_t *)in;
  uint32_t n = count_field[0];
  const struct arts_msg_rank_version_pair_s *entries =
      (const struct arts_msg_rank_version_pair_s *)((const char *)in +
                                                    (sizeof(uint32_t) * 2));
  for (uint32_t i = 0; i < n; i++) {
    arts_rank_u64_map_set(m, (unsigned int)entries[i].rank, entries[i].version);
  }
  return m;
}
