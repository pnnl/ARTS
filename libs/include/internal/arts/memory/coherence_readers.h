/* SPDX-License-Identifier: Apache-2.0
 *
 * Bit-packed atomic readers set, sized to the cluster's rank count.
 * Used only in LRC builds — RC reuses the per-rank version map for
 * the same purpose (set membership = nonzero entry).
 *
 * Each word covers 64 ranks.  At ARTS_GUID_RANK_BITS = 14 (max 16384
 * ranks), max size is (16384 + 63) / 64 = 256 words = 2 KiB per DB.
 * At experiment scale (<=32 ranks), 1 word = 8 B. */

#ifndef ARTS_MEMORY_COHERENCE_READERS_H
#define ARTS_MEMORY_COHERENCE_READERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

struct arts_readers_bits_s {
  _Atomic(uint64_t) *words;
  unsigned int nranks;
  unsigned int nwords;
};

void arts_readers_bits_init(struct arts_readers_bits_s *r, unsigned int nranks);
void arts_readers_bits_destroy(struct arts_readers_bits_s *r);
/* Set bit for rank; returns true if the bit was previously clear (first-time
 * set), false if already set.  Safe for concurrent callers. */
bool arts_readers_bits_set(struct arts_readers_bits_s *r, unsigned int rank);
/* Iterate over all set bits, invoking cb(rank, ctx) for each.  The snapshot
 * is acquired per-word; callers must ensure no concurrent set() during
 * iteration (destroy fan-out traversal holds the destroy_in_flight baton). */
void arts_readers_bits_for_each(const struct arts_readers_bits_s *r,
                                void (*cb)(unsigned int rank, void *ctx),
                                void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_READERS_H */
