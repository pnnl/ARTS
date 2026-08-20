/* SPDX-License-Identifier: Apache-2.0
 *
 * T150 — array_list reset/push-after-reset semantics
 * (libs/src/core/utils/array_list.c).  Pure unit test: the array_list is
 * single-owner / not thread-safe by contract, so no runtime is needed (libc
 * shims back arts_malloc/arts_free).
 *
 * The container's remaining users are append-then-iterate lists that reuse
 * one allocation across rounds (the GPU stream's per-cycle new-EDT list is
 * the live reset caller; the counter capture lists append only), and they
 * depend on these exact semantics:
 *
 *   1. arts_reset_array_list sets length -> 0 WHILE KEEPING the backing
 *      storage (head segment pointer unchanged) — the per-round reset is
 *      O(1) and never reallocates.
 *
 *   2. A PUSH after a reset restarts the indices at 0,1,2,... and the values
 *      read back exactly — the segments are reused, not appended to stale
 *      data.
 *
 *   3. Reset is idempotent and a freshly-created list is already length 0.
 *
 * These are pinned with the same element type the context lists use
 * (sizeof(uint64_t), the GUID width) and a capacity that forces multi-segment
 * growth so reset-keeps-ALL-segments (not just the head) is covered.
 *
 * exposes_runtime_bug = false.
 */
#include "arts/utils/array_list.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int fail(const char *msg) {
  (void)fprintf(stderr, "FAIL array_list_ctx_semantics: %s\n", msg);
  return 1;
}

int main(void) {
  /* (3) fresh list is empty. */
  const size_t ALEN = 4; /* small -> multi-segment after a handful of pushes */
  arts_array_list_t *l = arts_new_array_list(sizeof(uint64_t), ALEN);
  if (arts_length_array_list(l) != 0) {
    return fail("fresh list not length 0");
  }

  /* Fill past several segment boundaries. */
  const uint64_t N = 13; /* spans ~4 segments at ALEN=4 */
  for (uint64_t i = 0; i < N; i++) {
    uint64_t v = 0x9000 + i;
    uint64_t idx = arts_push_to_array_list(l, &v);
    if (idx != i) {
      return fail("initial push index wrong");
    }
  }
  if (arts_length_array_list(l) != N) {
    return fail("length after fill wrong");
  }
  arts_array_list_element_t *head_before = l->head;

  /* (1) reset keeps storage: length -> 0, head pointer unchanged. */
  arts_reset_array_list(l);
  if (arts_length_array_list(l) != 0) {
    return fail("reset did not zero length");
  }
  if (l->head != head_before) {
    return fail("reset reallocated head segment (storage not kept)");
  }
  if (l->element_size != sizeof(uint64_t) || l->array_length != ALEN) {
    return fail("reset perturbed element_size/array_length");
  }

  /* reset idempotent. */
  arts_reset_array_list(l);
  if (arts_length_array_list(l) != 0 || l->head != head_before) {
    return fail("second reset not idempotent");
  }

  /* (2) push-after-reset restarts indices at 0 and reuses the SAME segments
   * across boundaries; values read back exactly. */
  const uint64_t M = 11;
  for (uint64_t i = 0; i < M; i++) {
    uint64_t v = 0xAB00 + i;
    uint64_t idx = arts_push_to_array_list(l, &v);
    if (idx != i) {
      return fail("post-reset push index did not restart at 0");
    }
  }
  if (l->head != head_before) {
    return fail("post-reset push reallocated head (segments not reused)");
  }
  for (uint64_t i = 0; i < M; i++) {
    uint64_t *p = (uint64_t *)arts_get_from_array_list(l, i);
    if (!p || *p != 0xAB00 + i) {
      (void)fprintf(stderr, "post-reset get[%" PRIu64 "]=%" PRIu64 "\n", i,
                    p ? *p : 0);
      return fail("post-reset value/order wrong");
    }
  }

  arts_delete_array_list(l);
  printf("PASS array_list_ctx_semantics: reset keeps multi-segment storage; "
         "push-after-reset restarts indices and reuses segments\n");
  return 0;
}

/* libc-backed shims so the test links array_list.c without the runtime. */
void *arts_malloc(size_t size) { return malloc(size); }
void arts_free(void *ptr) { free(ptr); }
