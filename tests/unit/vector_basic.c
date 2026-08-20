/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_vector contract (libs/src/core/utils/vector.c).  Pure unit test —
 * single-owner container, libc shims back arts_malloc/arts_free.
 *
 * Pins the contracts the header states and callers lean on:
 *   1. Lazy allocation: init sets no block; the first push allocates.
 *   2. Growth: pushing past the capacity doubles it and preserves every
 *      element (crossing at least two doublings from the minimum).
 *   3. at(): exact addressing in bounds, NULL past the count.
 *   4. swap_remove: the last element fills the hole (order NOT preserved,
 *      by contract), the count drops, removing the last element moves
 *      nothing, and an out-of-range index is a no-op.
 *   5. clear: count -> 0, block retained (same data pointer on next push).
 *   6. free: block released, count zeroed, element_size and capacity KEPT —
 *      element_size doubles as callers' "initialised" sentinel and the next
 *      push reallocates at the retained capacity.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void *arts_malloc(size_t s) { return malloc(s); }
void arts_free(void *p) { free(p); }

#include "../../libs/src/core/utils/vector.c"

static int fails;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL vector_basic: %s\n", msg);                                  \
      fails++;                                                                 \
    }                                                                          \
  } while (0)

int main(void) {
  arts_vector_t v;

  /* (1) lazy allocation */
  arts_vector_init(&v, sizeof(uint64_t), 0);
  CHECK(v.data == NULL && arts_vector_count(&v) == 0, "init allocated");
  CHECK(v.capacity == 8, "default initial capacity");

  /* (2) growth across doublings, all elements preserved */
  enum { N = 40 }; /* 8 -> 16 -> 32 -> 64: three doublings */
  for (uint64_t i = 0; i < N; i++) {
    arts_vector_push(&v, &i);
  }
  CHECK(arts_vector_count(&v) == N, "count after pushes");
  CHECK(v.capacity == 64, "capacity doubled to fit");
  int intact = 1;
  for (uint64_t i = 0; i < N; i++) {
    uint64_t *p = (uint64_t *)arts_vector_at(&v, i);
    if (p == NULL || *p != i) {
      intact = 0;
    }
  }
  CHECK(intact, "elements preserved across growth");

  /* (3) at() bounds */
  CHECK(arts_vector_at(&v, N) == NULL, "at(count) not NULL");

  /* (4) swap_remove: last fills the hole; removing the last moves nothing;
   * out-of-range is a no-op */
  arts_vector_swap_remove(&v, 3); /* value N-1 lands at index 3 */
  CHECK(arts_vector_count(&v) == N - 1, "count after swap_remove");
  CHECK(*(uint64_t *)arts_vector_at(&v, 3) == N - 1, "last filled the hole");
  uint64_t before_last = *(uint64_t *)arts_vector_at(&v, N - 3);
  arts_vector_swap_remove(&v, N - 2); /* remove the (new) last */
  CHECK(arts_vector_count(&v) == N - 2, "count after removing last");
  CHECK(*(uint64_t *)arts_vector_at(&v, N - 3) == before_last,
        "removing the last moved something");
  arts_vector_swap_remove(&v, 12345);
  CHECK(arts_vector_count(&v) == N - 2, "out-of-range remove not a no-op");

  /* (5) clear keeps the block */
  void *block = v.data;
  arts_vector_clear(&v);
  CHECK(arts_vector_count(&v) == 0 && v.data == block, "clear dropped block");
  uint64_t x = 7;
  arts_vector_push(&v, &x);
  CHECK(v.data == block, "push after clear reallocated");

  /* (6) free keeps element_size + capacity, releases the block */
  uint64_t cap_before = v.capacity;
  arts_vector_free(&v);
  CHECK(v.data == NULL && arts_vector_count(&v) == 0, "free left state");
  CHECK(v.element_size == sizeof(uint64_t) && v.capacity == cap_before,
        "free forgot the sentinel/capacity");
  arts_vector_push(&v, &x); /* re-arm: push must reallocate */
  CHECK(v.data != NULL && arts_vector_count(&v) == 1 &&
            *(uint64_t *)arts_vector_at(&v, 0) == 7,
        "push after free");
  arts_vector_free(&v);
  arts_vector_free(&v); /* idempotent */

  if (fails) {
    return 1;
  }
  printf("PASS vector_basic\n");
  return 0;
}
