/* SPDX-License-Identifier: Apache-2.0 — registered slab pool basic alloc/lookup
 *
 * Whitebox unit test: no ARTS runtime, no ports, no config.  Exercises the
 * registered slab pool with a NULL fabric domain (registration skipped, arenas
 * carved identically).  Verifies the confinement invariant — every returned
 * pointer resolves through arts_regpool_lookup — across many mixed-size,
 * 64-byte-aligned allocations, then a free-all + re-allocate cycle. */
#include "arts/memory/regpool.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_ALLOC 1000
#define SLAB_BYTES ((size_t)64 * 1024 * 1024)
#define MAX_SIZE ((size_t)1 * 1024 * 1024)
#define ALIGN ((size_t)64)

/* 64-byte-aligned size in [64, MAX_SIZE] driven by a deterministic PRNG so the
 * run is reproducible. */
static size_t pick_size(unsigned *seed) {
  size_t s = (size_t)64 + (size_t)(rand_r(seed) % (int)(MAX_SIZE - 63));
  s &= ~(ALIGN - 1);
  if (s < ALIGN)
    s = ALIGN;
  return s;
}

/* Touch the first and last byte so a mis-sized / escaped mapping would fault
 * under ASan. */
static void touch(void *p, size_t size) {
  volatile unsigned char *b = (volatile unsigned char *)p;
  b[0] = 0xA5;
  b[size - 1] = 0x5A;
}

int main(void) {
  bool ok = arts_regpool_init(NULL, SLAB_BYTES, 1);
  assert(ok && "regpool_init(NULL domain, 64 MiB, 1 node) must succeed");

  static void *ptrs[N_ALLOC];
  static size_t sizes[N_ALLOC];
  unsigned seed = 0x1234u;

  for (int i = 0; i < N_ALLOC; i++) {
    size_t s = pick_size(&seed);
    sizes[i] = s;
    void *p = arts_regpool_alloc_aligned(s, ALIGN);
    assert(p != NULL && "allocation must not fail");
    assert(((uintptr_t)p & (ALIGN - 1)) == 0 && "pointer must be 64-aligned");
    const arts_regpool_mr_t *mr = arts_regpool_lookup(p);
    assert(mr != NULL && "every returned pointer must resolve to a slab");
    /* the whole allocation must fall inside the resolved registered range */
    assert((const char *)p >= (const char *)mr->base);
    assert((const char *)p + s <= (const char *)mr->base + mr->len);
    assert(mr->mr == NULL && "NULL-domain pool leaves mr unregistered");
    assert(mr->rkey == 0);
    touch(p, s);
    ptrs[i] = p;
  }

  for (int i = 0; i < N_ALLOC; i++)
    arts_regpool_free(ptrs[i]);

  /* Re-allocate after free-all: the arenas must still serve requests. */
  for (int i = 0; i < N_ALLOC / 2; i++) {
    size_t s = pick_size(&seed);
    void *p = arts_regpool_alloc_aligned(s, ALIGN);
    assert(p != NULL && "re-allocation after free-all must succeed");
    assert(arts_regpool_lookup(p) != NULL);
    touch(p, s);
    arts_regpool_free(p);
  }

  arts_regpool_cleanup();
  printf("REGPOOL_BASIC_OK allocs=%d slab=%zuMiB\n", N_ALLOC,
         SLAB_BYTES / (1024 * 1024));
  return 0;
}
