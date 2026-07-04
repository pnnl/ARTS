/* SPDX-License-Identifier: Apache-2.0 — registered slab pool confinement guard
 *
 * Whitebox unit test: no ARTS runtime, no ports, no config.  Drives the two
 * growth paths and asserts the confinement invariant on each:
 *   1. Allocating past one slab forces a grow; a second slab must appear in the
 *      lookup table (>= 2 distinct registered ranges observed).
 *   2. An oversize request (> slab) must be served from a direct slab (or a
 *      grow) — never as an unregistered pointer.
 *   3. A freed direct (oversize) slab must be live-reclaimed, not merely
 *      leaked until process exit: repeating an oversize alloc/free cycle
 *      must not leave the freed pointer resolvable, and must not exhaust the
 *      pool's fixed-capacity slab table. */
#include "arts/memory/regpool.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SLAB_BYTES ((size_t)64 * 1024 * 1024)
#define ALIGN ((size_t)64)
#define BLOCK ((size_t)1 * 1024 * 1024) /* 1 MiB blocks, < slab/2 -> arena path */
#define N_BLOCKS 160                    /* 160 MiB total: forces multiple grows  */
#define OVERSIZE ((size_t)128 * 1024 * 1024) /* > slab -> direct-slab path        */
#define RECLAIM_SIZE ((size_t)100 * 1024 * 1024) /* > slab/2 -> direct-slab path  */
#define RECLAIM_ITERS 8

/* Track distinct slab base addresses seen through lookups. */
#define MAX_DISTINCT 64
static const void *distinct[MAX_DISTINCT];
static int n_distinct;

static void record_slab(const arts_regpool_mr_t *mr) {
  for (int i = 0; i < n_distinct; i++)
    if (distinct[i] == mr->base)
      return;
  assert(n_distinct < MAX_DISTINCT);
  distinct[n_distinct++] = mr->base;
}

int main(void) {
  bool ok = arts_regpool_init(NULL, SLAB_BYTES, 1);
  assert(ok && "regpool_init must succeed");

  /* Part 1: exceed one slab through the arena path and force grows. */
  static void *blocks[N_BLOCKS];
  for (int i = 0; i < N_BLOCKS; i++) {
    void *p = arts_regpool_alloc_aligned(BLOCK, ALIGN);
    assert(p != NULL && "arena allocation must not fail (grow on demand)");
    const arts_regpool_mr_t *mr = arts_regpool_lookup(p);
    assert(mr != NULL && "every returned pointer must resolve to a slab");
    record_slab(mr);
    blocks[i] = p;
  }
  assert(n_distinct >= 2 &&
         "allocating past one slab must add a second slab to the table");

  for (int i = 0; i < N_BLOCKS; i++)
    arts_regpool_free(blocks[i]);

  /* Part 2: oversize request must never escape the registered slabs. */
  void *big = arts_regpool_alloc_aligned(OVERSIZE, ALIGN);
  assert(big != NULL && "oversize allocation must succeed");
  assert(((uintptr_t)big & (ALIGN - 1)) == 0);
  const arts_regpool_mr_t *bmr = arts_regpool_lookup(big);
  assert(bmr != NULL && "oversize pointer must resolve to a registered slab");
  assert((const char *)big >= (const char *)bmr->base);
  assert((const char *)big + OVERSIZE <= (const char *)bmr->base + bmr->len);
  /* touch both ends of the oversize allocation */
  ((volatile unsigned char *)big)[0] = 0xC3;
  ((volatile unsigned char *)big)[OVERSIZE - 1] = 0x3C;
  arts_regpool_free(big);

  /* Part 3: repeated direct-slab alloc/free must reclaim live, not leak.
   * Track distinct bases purely as reporting evidence of slot reuse (the
   * kernel is not contractually obligated to hand back the same address
   * for a same-size mmap after munmap, so it is not asserted on) -- the
   * hard gate is the lookup-miss check below, which is exactly the
   * observable contract a live reclaim must satisfy and a leak-until-
   * cleanup implementation cannot: before the fix, arts_regpool_free was a
   * no-op for direct slabs, so the freed pointer stayed resolvable and
   * every iteration burned a fresh, never-recycled slot in the pool's
   * fixed-capacity table. */
  int reclaim_distinct = 0;
  const void *reclaim_bases[RECLAIM_ITERS];
  for (int i = 0; i < RECLAIM_ITERS; i++) {
    void *p = arts_regpool_alloc_aligned(RECLAIM_SIZE, ALIGN);
    assert(p != NULL && "reclaim-loop allocation must not fail");
    assert(((uintptr_t)p & (ALIGN - 1)) == 0);
    const arts_regpool_mr_t *mr = arts_regpool_lookup(p);
    assert(mr != NULL && "freshly allocated direct slab must resolve");
    ((volatile unsigned char *)p)[0] = 0xC3;
    ((volatile unsigned char *)p)[RECLAIM_SIZE - 1] = 0x3C;

    bool seen = false;
    for (int j = 0; j < reclaim_distinct; j++)
      if (reclaim_bases[j] == p)
        seen = true;
    if (!seen)
      reclaim_bases[reclaim_distinct++] = p;

    arts_regpool_free(p);
    assert(arts_regpool_lookup(p) == NULL &&
           "a freed direct slab must not remain resolvable -- live reclaim, "
           "not leak-until-cleanup");
  }

  arts_regpool_cleanup();
  printf("REGPOOL_GUARD_OK distinct_slabs=%d oversize=%zuMiB "
         "reclaim_iters=%d reclaim_distinct_bases=%d\n",
         n_distinct, OVERSIZE / (1024 * 1024), RECLAIM_ITERS,
         reclaim_distinct);
  return 0;
}
