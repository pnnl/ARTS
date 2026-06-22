/* SPDX-License-Identifier: Apache-2.0
 *
 * T016 — arts_link_list lifecycle: new_item(0) leak + group_new/get/delete
 * (link_list.c).  Census 30.md §B.6 / SUSPECTED-BUG B118.
 *
 * B118 (MEDIUM): arts_link_list_new_item(0) calls
 * arts_calloc(1, sizeof(header)+0) and then, because size==0, returns NULL
 * WITHOUT freeing the just-allocated header — the allocation is leaked and
 * unreachable.  Callers in transport always pass a non-zero size, so it is
 * latent, but it is a clear resource leak on the size==0 path.  A pure-unit
 * test under ASan/LSan flags it.
 *
 * This test EXERCISES new_item(0) and asserts the API contract (returns NULL).
 * Whether the internal header is leaked is observed by LeakSanitizer: if the
 * runtime leaks it (B118 confirmed), ASan/LSan reports a leak on exit and the
 * test process exits non-zero — we leave the test correct-and-failing and
 * record exposes_runtime_bug=true rather than masking the leak.
 *
 * It also pins the rest of the lifecycle (which must be leak-clean):
 *   - group_new(n) builds n initialized lists; get(arr,pos) is arr+pos;
 *     push/pop a few items per list; delete drains + frees the array.
 *   - new_item(k)/push/pop/delete_item round-trip for k>0.
 */

#include "arts/utils/link_list.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }

#define GROUP_N 4
#define PER_LIST 5

static int lifecycle_clean(void) {
  /* group_new builds GROUP_N initialized lists. */
  struct arts_link_list_s *grp = arts_link_list_group_new(GROUP_N);
  if (!grp) {
    (void)fprintf(stderr, "FAIL link_list_lifecycle: group_new NULL\n");
    return 1;
  }
  for (unsigned i = 0; i < GROUP_N; i++) {
    struct arts_link_list_s *l = arts_link_list_get(grp, i);
    if (l != &grp[i]) {
      (void)fprintf(stderr, "FAIL link_list_lifecycle: get(%u) != &grp[%u]\n",
                    i, i);
      return 1;
    }
    if (arts_link_list_is_empty(l) != 1) {
      (void)fprintf(
          stderr, "FAIL link_list_lifecycle: list %u not empty after new\n", i);
      return 1;
    }
    for (int k = 0; k < PER_LIST; k++) {
      uint64_t *v = (uint64_t *)arts_link_list_new_item(sizeof(uint64_t));
      *v = ((uint64_t)i << 32) | (uint64_t)k;
      arts_link_list_push_back(l, v);
    }
  }
  /* Pop a couple from list 0 and verify FIFO + payload round-trip. */
  struct arts_link_list_s *l0 = arts_link_list_get(grp, 0);
  for (int k = 0; k < 2; k++) {
    uint64_t *v = (uint64_t *)arts_link_list_pop_front(l0, NULL);
    if (!v || *v != (uint64_t)k) {
      (void)fprintf(stderr, "FAIL link_list_lifecycle: list0 pop %d got %llu\n",
                    k, v ? (unsigned long long)*v : 0ull);
      return 1;
    }
    arts_link_list_delete_item(v);
  }
  /* NOTE: arts_link_list_delete drains only the FIRST list of a group (it casts
   * its arg to a single arts_link_list_s* and pop_front's only that), then
   * frees the whole array — calling it on a multi-list group LEAKS items in
   * lists 1..n-1.  The runtime (transport/outbox.c) does NOT use delete for
   * groups; it drains each list manually via get(i)+pop_front then arts_free's
   * the array.  We mirror that correct teardown here. */
  for (unsigned i = 0; i < GROUP_N; i++) {
    struct arts_link_list_s *l = arts_link_list_get(grp, i);
    void *out;
    while ((out = arts_link_list_pop_front(l, NULL)) != NULL) {
      arts_link_list_delete_item(out);
    }
  }
  arts_free(grp);

  /* arts_link_list_delete is correct for a SINGLE-list group: drain + free in
   * one shot.  Exercise that supported contract too. */
  struct arts_link_list_s *single = arts_link_list_group_new(1);
  for (int k = 0; k < PER_LIST; k++) {
    uint64_t *v = (uint64_t *)arts_link_list_new_item(sizeof(uint64_t));
    *v = (uint64_t)k;
    arts_link_list_push_back(&single[0], v);
  }
  arts_link_list_delete(single); /* drains the single list + frees array */

  /* new_item(k>0) round-trip stand-alone. */
  struct arts_link_list_s l;
  arts_link_list_new(&l);
  char *blob = (char *)arts_link_list_new_item(32);
  if (!blob) {
    (void)fprintf(stderr, "FAIL link_list_lifecycle: new_item(32) NULL\n");
    return 1;
  }
  memset(blob, 0xAB, 32);
  arts_link_list_push_back(&l, blob);
  char *out = (char *)arts_link_list_pop_front(&l, NULL);
  if (out != blob) {
    (void)fprintf(stderr,
                  "FAIL link_list_lifecycle: data ptr not round-tripped\n");
    return 1;
  }
  for (int i = 0; i < 32; i++) {
    if ((unsigned char)out[i] != 0xAB) {
      (void)fprintf(stderr, "FAIL link_list_lifecycle: payload corrupt\n");
      return 1;
    }
  }
  arts_link_list_delete_item(out);
  return 0;
}

int main(void) {
  if (lifecycle_clean() != 0) {
    return 1;
  }

  /* B118: new_item(0) must return NULL (API contract).  Internally it callocs
   * the header first and does NOT free it on the size==0 path -> the header is
   * leaked + unreachable.  LeakSanitizer will report it at exit; we
   * deliberately do NOT try to recover/free it (we cannot — new_item returned
   * NULL, so the pointer is unreachable, which is exactly the bug). */
  void *zero = arts_link_list_new_item(0);
  if (zero != NULL) {
    (void)fprintf(stderr,
                  "FAIL link_list_lifecycle: new_item(0) returned non-NULL\n");
    return 1;
  }

  /* The PASS line documents that the API contract held; the leak (if any) is
   * surfaced by LSan as a separate, intentional finding (B118). */
  printf("PASS link_list_lifecycle: group/get/delete + new_item(k) round-trip "
         "clean; new_item(0)==NULL (header leak per B118 surfaced by LSan)\n");
  return 0;
}
