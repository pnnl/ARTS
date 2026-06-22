/* SPDX-License-Identifier: Apache-2.0
 *
 * T023 — array_list functional coverage + empty/stale-cache get bug pin
 * (libs/src/core/utils/array_list.c).  Single-thread (the structure is
 * single-owner / not thread-safe by contract).
 *
 * Properties exercised:
 *   1. push returns the assigned index 0,1,2,...; length tracks the count.
 *   2. get returns the exact bytes pushed, across the FAST path (single
 *      segment, index<array_length) AND the SLOW path (index spanning
 *      multiple segments after a grow), AND the cached fastest path
 *      (re-get of the last requested index).
 *   3. cross-segment indexing: with a small array_length, pushing past the
 *      segment boundary creates new segments at start = k*array_length; get
 *      walks them correctly.
 *   4. get(out-of-range) returns NULL for index >= count (when not the
 *      cached index).
 *   5. reset clears logically (length->0) but reuses storage; push-after-reset
 *      re-fills the SAME segments and get returns the new values.
 *   6. snapshot iterator: iter_init snapshots the count; has_next/next walk
 *      exactly [0,count) in order across segment boundaries; pushes after
 *      init are NOT seen (snapshot semantics).
 *
 * Suspected-bug pin (B053 / B1, array_list.c:113-115) — record, do NOT fix:
 *   The "fastest path" `if (index == a_list->lastRequest) return
 * lastRequestPtr` bypasses the bounds check.  After arts_new_array_list (and
 * after arts_reset_array_list), lastRequest==0 and lastRequestPtr==head->array
 *   while the count (index) is 0.  Therefore get(list, 0) on an EMPTY/just-
 *   reset list returns head->array (a pointer into uninitialized segment
 *   memory) instead of NULL.  The CORRECT behavior is NULL (index 0 >= count
 *   0 is out of range).  We assert the BUGGY observed behavior (non-NULL) so
 *   the test PASSES today while documenting the defect; if the runtime is
 *   ever fixed to return NULL, this assertion flips and the test must be
 *   updated to expect NULL.  exposes_runtime_bug=true.
 */

#include "arts/utils/array_list.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *msg) {
  (void)fprintf(stderr, "FAIL array_list_basic: %s\n", msg);
  return 1;
}

int main(void) {
  /* ---- 1+2+3: push / length / get across fast + slow + cached paths. ---- */
  {
    const size_t ALEN = 4; /* small so we cross segment boundaries */
    arts_array_list_t *l = arts_new_array_list(sizeof(uint64_t), ALEN);
    const uint64_t N = 30; /* spans ~8 segments */
    for (uint64_t i = 0; i < N; i++) {
      uint64_t v = 0xA000 + i;
      uint64_t idx = arts_push_to_array_list(l, &v);
      if (idx != i)
        return fail("push returned wrong index");
    }
    if (arts_length_array_list(l) != N)
      return fail("length wrong");

    /* forward get (mostly slow path: count>ALEN so multi-segment). */
    for (uint64_t i = 0; i < N; i++) {
      uint64_t *p = (uint64_t *)arts_get_from_array_list(l, i);
      if (!p)
        return fail("get returned NULL for in-range index");
      if (*p != 0xA000 + i) {
        (void)fprintf(stderr, "get[%" PRIu64 "]=%" PRIu64 " want %" PRIu64 "\n",
                      i, *p, (uint64_t)(0xA000 + i));
        return fail("get value wrong");
      }
    }
    /* cached fastest path: re-get the same index twice. */
    uint64_t *c1 = (uint64_t *)arts_get_from_array_list(l, 7);
    uint64_t *c2 = (uint64_t *)arts_get_from_array_list(l, 7);
    if (c1 != c2 || !c1 || *c1 != 0xA000 + 7)
      return fail("cached path wrong");

    /* out-of-range (not the cached index) -> NULL. */
    if (arts_get_from_array_list(l, N) != NULL) {
      return fail("get(count) should be NULL");
    }
    if (arts_get_from_array_list(l, N + 100) != NULL) {
      return fail("get(count+100) should be NULL");
    }
    arts_delete_array_list(l);
  }

  /* ---- 2b: single-segment "faster path" (count < array_length). ---- */
  {
    arts_array_list_t *l = arts_new_array_list(sizeof(uint64_t), 16);
    for (uint64_t i = 0; i < 5; i++) { /* 5 < 16 -> all in head segment */
      uint64_t v = 0xB000 + i;
      arts_push_to_array_list(l, &v);
    }
    for (uint64_t i = 0; i < 5; i++) {
      uint64_t *p = (uint64_t *)arts_get_from_array_list(l, i);
      if (!p || *p != 0xB000 + i)
        return fail("single-segment get wrong");
    }
    arts_delete_array_list(l);
  }

  /* ---- 5: reset reuses storage; push-after-reset re-fills. ---- */
  {
    const size_t ALEN = 4;
    arts_array_list_t *l = arts_new_array_list(sizeof(uint64_t), ALEN);
    for (uint64_t i = 0; i < 20; i++) {
      uint64_t v = i;
      arts_push_to_array_list(l, &v);
    }
    arts_array_list_element_t *head_before = l->head;
    arts_reset_array_list(l);
    if (arts_length_array_list(l) != 0)
      return fail("reset: length != 0");
    if (l->head != head_before)
      return fail("reset: head reallocated");
    /* push again — should reuse the existing segments. */
    for (uint64_t i = 0; i < 12; i++) {
      uint64_t v = 0xC000 + i;
      uint64_t idx = arts_push_to_array_list(l, &v);
      if (idx != i)
        return fail("reset: re-push index wrong");
    }
    for (uint64_t i = 0; i < 12; i++) {
      uint64_t *p = (uint64_t *)arts_get_from_array_list(l, i);
      if (!p || *p != 0xC000 + i)
        return fail("reset: re-get value wrong");
    }
    arts_delete_array_list(l);
  }

  /* ---- 6: snapshot iterator across segments; ignores later pushes. ---- */
  {
    const size_t ALEN = 4;
    arts_array_list_t *l = arts_new_array_list(sizeof(uint64_t), ALEN);
    const uint64_t N = 18;
    for (uint64_t i = 0; i < N; i++) {
      uint64_t v = 0xD000 + i;
      arts_push_to_array_list(l, &v);
    }
    arts_array_list_iterator_t it;
    arts_array_list_iter_init(&it, l);
    /* push MORE after snapshot — must not be visited. */
    for (uint64_t i = N; i < N + 5; i++) {
      uint64_t v = 0xD000 + i;
      arts_push_to_array_list(l, &v);
    }
    uint64_t count = 0;
    while (arts_array_list_has_next(&it)) {
      uint64_t *p = (uint64_t *)arts_array_list_next(&it);
      if (!p)
        return fail("iter: next returned NULL within range");
      if (*p != 0xD000 + count) {
        (void)fprintf(stderr, "iter[%" PRIu64 "]=%" PRIu64 "\n", count, *p);
        return fail("iter: value/order wrong");
      }
      count++;
    }
    if (count != N)
      return fail("iter: visited wrong number (saw post-snapshot push?)");
    if (arts_array_list_next(&it) != NULL)
      return fail("iter: next past end");
    arts_delete_array_list(l);
  }

  /* ---- B053/B1 pin: empty-list get(0) hits the cache fastest path and
   *      returns a NON-NULL pointer into uninitialized memory instead of
   *      NULL.  Asserting the BUGGY behavior so the test passes while
   *      documenting the defect.  (Correct behavior would be NULL.) ---- */
  {
    arts_array_list_t *l = arts_new_array_list(sizeof(uint64_t), 8);
    if (arts_length_array_list(l) != 0)
      return fail("B053: fresh list not empty");
    void *p = arts_get_from_array_list(l, 0); /* index 0 == lastRequest 0 */
    if (p == NULL) {
      /* If this ever happens, the bug was FIXED — update the test. */
      return fail("B053 pin: get(0) on EMPTY list returned NULL — bug appears "
                  "FIXED; update this test to expect NULL");
    }
    /* p == head->array (uninitialized) — the documented out-of-bounds cache
     * read.  We do NOT dereference it (it is uninitialized). */
    if (p != l->head->array) {
      return fail("B053 pin: get(0) empty returned an unexpected pointer");
    }

    /* Same after reset: reset re-arms lastRequest=0 + lastRequestPtr=head. */
    uint64_t v = 42;
    arts_push_to_array_list(l, &v);
    arts_reset_array_list(l);
    void *p2 = arts_get_from_array_list(l, 0);
    if (p2 == NULL) {
      return fail("B053 pin: get(0) after reset returned NULL — bug appears "
                  "FIXED; update this test to expect NULL");
    }
    if (p2 != l->head->array) {
      return fail("B053 pin: get(0) after reset returned unexpected pointer");
    }
    arts_delete_array_list(l);
  }

  printf("PASS array_list_basic: push/get(fast+slow+cached)/reset/iterator OK; "
         "B053 empty-get(0)-returns-non-NULL pinned (suspected bug)\n");
  return 0;
}

/* ── libc-backed shims so the test links array_list.c without the runtime ── */
void *arts_malloc(size_t size) { return malloc(size); }
void arts_free(void *ptr) { free(ptr); }
