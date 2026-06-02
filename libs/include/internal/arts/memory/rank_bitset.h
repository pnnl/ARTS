/* SPDX-License-Identifier: Apache-2.0
 *
 * Bit-packed atomic rank bit-set, sized to the cluster's rank count.
 * Used only in LRC builds — RC reuses the per-rank version map for
 * the same purpose (set membership = nonzero entry).
 *
 * Each word covers 64 ranks.  At ARTS_GUID_RANK_BITS = 14 (max 16384
 * ranks), max size is (16384 + 63) / 64 = 256 words = 2 KiB per DB.
 * At experiment scale (<=32 ranks), 1 word = 8 B.
 *
 * This header carries ONLY the struct layout: it is included directly by
 * the header that lays out the embedding DB descriptor (which embeds
 * arts_rank_bitset_s by value) and therefore must not depend on any other
 * coherence header.  The rank-bitset FUNCTION declarations live in
 * coherence_home.h.
 *
 * Because the embedding descriptor is reached by the C++/nvcc layout-only
 * TUs (which cannot parse C11 _Atomic), the `words` pointer takes the same
 * C/C++ split the other layout structs use: a plain pointer for the C++
 * layout view (identical size/alignment to a pointer-to-_Atomic) and the
 * real atomic element type under C. */

#ifndef ARTS_MEMORY_RANK_BITSET_H
#define ARTS_MEMORY_RANK_BITSET_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#ifndef __cplusplus
#include <stdatomic.h>
#endif

struct arts_rank_bitset_s {
#ifdef __cplusplus
  uint64_t *words; /* layout view — real element type is _Atomic under C */
#else
  _Atomic(uint64_t) *words;
#endif
  unsigned int nranks;
  unsigned int nwords;
};

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_RANK_BITSET_H */
