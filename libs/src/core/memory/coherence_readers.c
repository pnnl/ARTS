/* SPDX-License-Identifier: Apache-2.0
 *
 * Bit-packed atomic readers set implementation.  See coherence_readers.h. */

#include "arts/memory/coherence_readers.h"

#include <stdlib.h>

void arts_readers_bits_init(struct arts_readers_bits_s *r,
                            unsigned int nranks) {
  r->nranks = nranks;
  r->nwords = (nranks + 63) / 64;
  r->words = (_Atomic(uint64_t) *)calloc(r->nwords, sizeof(_Atomic(uint64_t)));
}

void arts_readers_bits_destroy(struct arts_readers_bits_s *r) {
  free(r->words);
  r->words = NULL;
  r->nwords = 0;
}

bool arts_readers_bits_set(struct arts_readers_bits_s *r, unsigned int rank) {
  if (rank >= r->nranks) {
    return false;
  }
  unsigned int word_idx = rank / 64;
  uint64_t bit = (uint64_t)1 << (rank % 64);
  uint64_t prev = atomic_fetch_or_explicit(&r->words[word_idx], bit,
                                           memory_order_acq_rel);
  return (prev & bit) == 0;
}

void arts_readers_bits_for_each(const struct arts_readers_bits_s *r,
                                void (*cb)(unsigned int rank, void *ctx),
                                void *ctx) {
  for (unsigned int w = 0; w < r->nwords; w++) {
    uint64_t snap = atomic_load_explicit(
        (_Atomic(uint64_t) *)&r->words[w], memory_order_acquire);
    while (snap) {
      unsigned int b = (unsigned int)__builtin_ctzll(snap);
      cb(w * 64 + b, ctx);
      snap &= snap - 1;
    }
  }
}
