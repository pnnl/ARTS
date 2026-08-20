/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ARTS_UTILS_VECTOR_H
#define ARTS_UTILS_VECTOR_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* A contiguous growable array with unordered removal.
 *
 * It exists beside arts_array_list because the two answer different questions,
 * and picking the wrong one costs asymptotics rather than constants:
 *
 *   arts_array_list  chunks linked together, never moved, never shrunk.  An
 *                    element's address is stable for the list's life and there
 *                    is no removal — the right shape for something appended and
 *                    then only iterated.
 *   arts_vector      one block, doubled on growth, with swap-remove.  Addresses
 *                    are invalidated by a push, and in exchange the length
 *                    tracks what is LIVE rather than what was ever added.
 *
 * That last property is the whole point.  A collection that is SEARCHED on
 * every removal must shrink, or the search walks entries that were removed long
 * ago: marking a slot dead instead of removing it turns d removals into
 * d + (d-1) + ... over the entries ever added, not over the ones still there.
 *
 * Removal is unordered on purpose: the last element is moved into the hole.  A
 * caller that needs insertion order must not use this.
 *
 * Single-threaded, like the per-EDT bookkeeping it is written for.  No locking,
 * no atomics.
 */
typedef struct {
  void *data;
  size_t element_size;
  uint64_t count;
  uint64_t capacity;
} arts_vector_t;

/* Initialise empty; the first push allocates.  `initial` is a capacity hint
 * (0 = 8).  Must not be called on a vector that still owns a block — init
 * does not free, it forgets. */
void arts_vector_init(arts_vector_t *v, size_t element_size, uint64_t initial);

/* Release the block.  KEEPS element_size and capacity: element_size doubles
 * as callers' "initialised" sentinel, so a freed vector stays armed for
 * reuse and the next push reallocates at the retained capacity.  Safe on an
 * already-released or never-pushed vector. */
void arts_vector_free(arts_vector_t *v);

/* Append a copy of `element`.  Invalidates every pointer previously returned
 * by arts_vector_at. */
void arts_vector_push(arts_vector_t *v, const void *element);

/* Address of element `index`, or NULL if out of range.  Valid until the next
 * push. */
void *arts_vector_at(const arts_vector_t *v, uint64_t index);

/* Remove element `index` by moving the last element into its place.  Order is
 * not preserved.  The block is not shrunk — capacity is kept for reuse. */
void arts_vector_swap_remove(arts_vector_t *v, uint64_t index);

/* Length: the number of LIVE elements. */
static inline uint64_t arts_vector_count(const arts_vector_t *v) {
  return v->count;
}

/* Drop every element, keeping the block for reuse. */
static inline void arts_vector_clear(arts_vector_t *v) { v->count = 0; }

#ifdef __cplusplus
}
#endif
#endif /* ARTS_UTILS_VECTOR_H */
