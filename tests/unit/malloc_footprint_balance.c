/* SPDX-License-Identifier: Apache-2.0
 *
 * T245 — footprint-counter balance of the arts_malloc/arts_calloc/arts_realloc/
 * arts_free family in libs/src/core/utils/malloc.c.
 *
 * The allocator stores no per-allocation header: it bumps
 * BYTES_MEMORY_FOOTPRINT by the allocator's USABLE size (queried back from the
 * allocator, which already tracks every block) on each alloc and decrements by
 * the same usable size on free.  Usable size is >= the requested size
 * (size-class rounding) and is stable for a given live pointer, so matched
 * alloc/free is exactly net zero. This test pins:
 *
 *   1. Matched alloc/free leaves the footprint at its starting value (net 0),
 *      and the bump equals the allocator's usable size (>= requested).
 *   2. size==0 semantics: arts_malloc(0)==NULL (no counter bump),
 *      arts_calloc(0,_)==arts_calloc(_,0)==NULL, arts_realloc(p,0) frees and
 *      returns NULL, arts_realloc(NULL,0)==NULL.
 *   3. calloc zero-initialises and balances.
 *   4. realloc (grow and shrink) re-bases the footprint to the new block's
 *      usable size and stays balanced across the eventual free.
 *   5. Concurrent consistency: N threads each malloc+free M blocks; the
 *      footprint counter (an atomic add/sub) returns to its start with no torn
 *      updates.
 *
 * STANDALONE STRATEGY: malloc.c reaches the footprint macro and ARTS_ERROR via
 * the heavy "arts/system/print.h" -> runtime_state.h -> counter chain.  To
 * unit-test it without the runtime we pre-define the include guards of those
 * headers and supply our own minimal ARTS_ERROR / footprint macros, then
 * #include the malloc.c translation unit directly.  Our footprint macros target
 * a test-local atomic counter so we can observe balance exactly.  malloc.c
 * exposes ARTS_SYS_USABLE (mi_usable_size or ARTS_SYS_USABLE for whichever
 * allocator the build selected), which we reuse for the exact expected bumps so
 * the test is path-agnostic.  (Precedent: tests build edt_gpu.cu by
 * #include'ing edt.c.)
 */

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- test-local footprint counter the included malloc.c will drive ---- */
static atomic_uint_least64_t g_footprint;

/* ---- stub the runtime macros malloc.c expects, and block the heavy headers
 *      by pre-asserting their include guards. ---- */
#define ARTS_SYSTEM_PRINT_H 1 /* skip arts/system/print.h body */
#define ARTS_DEFS_H 1         /* skip arts/defs.h body */

/* print.h would have provided ARTS_ERROR (which aborts).  Keep the abort
 * semantics so invalid-param paths terminate as the real runtime does. */
#define ARTS_ERROR(...)                                                        \
  do {                                                                         \
    (void)fprintf(stderr, "ARTS_ERROR: " __VA_ARGS__);                         \
    (void)fprintf(stderr, "\n");                                               \
    abort();                                                                   \
  } while (0)

/* counter.h footprint macros (normally generated into Preamble.h). */
#define INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(v)                                 \
  atomic_fetch_add_explicit(&g_footprint, (uint64_t)(v), memory_order_relaxed)
#define DECREMENT_BYTES_MEMORY_FOOTPRINT_BY(v)                                 \
  atomic_fetch_sub_explicit(&g_footprint, (uint64_t)(v), memory_order_relaxed)

/* Now pull in the unit under test (also makes ARTS_SYS_USABLE visible). */
#include "../../libs/src/core/utils/malloc.c"

static uint64_t footprint(void) {
  return atomic_load_explicit(&g_footprint, memory_order_relaxed);
}

/* ---- concurrent balance ---- */
#define C_THREADS 10
#define C_ITERS 20000
static atomic_int g_gate;

static void *churn(void *arg) {
  (void)arg;
  while (atomic_load_explicit(&g_gate, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < C_ITERS; i++) {
    size_t sz = (size_t)((i % 257) + 1); /* 1..257 bytes */
    void *p = arts_malloc(sz);
    /* touch the memory so ASan would catch under-allocation */
    ((volatile char *)p)[0] = (char)i;
    ((volatile char *)p)[sz - 1] = (char)i;
    arts_free(p);
  }
  return NULL;
}

int main(void) {
  int rc = 0;
  atomic_init(&g_footprint, 0);

  /* ===== Part 1: matched alloc/free is net zero; bump == usable size. ===== */
  {
    uint64_t base = footprint();
    void *p = arts_malloc(123);
    uint64_t bumped = footprint() - base;
    size_t usable = ARTS_SYS_USABLE(p);
    if (bumped != usable || bumped < 123) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: malloc(123) bumped %" PRIu64
                    ", expected usable=%zu (>=123)\n",
                    bumped, usable);
      rc = 1;
    }
    arts_free(p);
    if (footprint() != base) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: free did not restore "
                    "footprint (%" PRIu64 " != %" PRIu64 ")\n",
                    footprint(), base);
      rc = 1;
    }
  }

  /* ===== Part 2: size==0 / NULL semantics, no counter bump. ===== */
  {
    uint64_t base = footprint();
    if (arts_malloc(0) != NULL) {
      (void)fprintf(stderr, "FAIL malloc_footprint_balance: malloc(0)!=NULL\n");
      rc = 1;
    }
    if (arts_calloc(0, 16) != NULL || arts_calloc(16, 0) != NULL) {
      (void)fprintf(stderr, "FAIL malloc_footprint_balance: calloc 0 path\n");
      rc = 1;
    }
    if (arts_realloc(NULL, 0) != NULL) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: realloc(NULL,0)!=NULL\n");
      rc = 1;
    }
    /* realloc(p,0) frees and returns NULL, restoring footprint */
    void *p = arts_malloc(64);
    if (arts_realloc(p, 0) != NULL) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: realloc(p,0)!=NULL\n");
      rc = 1;
    }
    if (footprint() != base) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: zero-size paths perturbed "
                    "footprint (%" PRIu64 " != %" PRIu64 ")\n",
                    footprint(), base);
      rc = 1;
    }
  }

  /* ===== Part 3: calloc zero-initialises and balances. ===== */
  {
    uint64_t base = footprint();
    size_t n = 32, sz = 8;
    unsigned char *p = (unsigned char *)arts_calloc(n, sz);
    for (size_t i = 0; i < n * sz; i++) {
      if (p[i] != 0) {
        (void)fprintf(stderr,
                      "FAIL malloc_footprint_balance: calloc not zeroed\n");
        rc = 1;
        break;
      }
    }
    if (footprint() != base + ARTS_SYS_USABLE(p)) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: calloc footprint wrong\n");
      rc = 1;
    }
    arts_free(p);
    if (footprint() != base) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: calloc/free not net 0\n");
      rc = 1;
    }
  }

  /* ===== Part 4: realloc shrink re-bases the footprint and stays balanced. */
  {
    uint64_t base = footprint();
    void *p = arts_malloc(1000);
    void *q =
        arts_realloc(p, 100); /* shrink: footprint follows the new block */
    if (footprint() != base + ARTS_SYS_USABLE(q)) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: shrink footprint %" PRIu64
                    " != base+usable %" PRIu64 "\n",
                    footprint(), base + ARTS_SYS_USABLE(q));
      rc = 1;
    }
    arts_free(q);
    if (footprint() != base) {
      (void)fprintf(
          stderr,
          "FAIL malloc_footprint_balance: realloc shrink+free not net "
          "0 (got %" PRIu64 ", expected %" PRIu64 ")\n",
          footprint(), base);
      rc = 1;
    }
  }

  /* ===== Part 5: realloc grow preserves bytes, re-bases, and balances. =====
   */
  {
    uint64_t base = footprint();
    unsigned char *p = (unsigned char *)arts_malloc(16);
    for (int i = 0; i < 16; i++) {
      p[i] = (unsigned char)(i + 1);
    }
    unsigned char *q = (unsigned char *)arts_realloc(p, 64); /* grow */
    for (int i = 0; i < 16; i++) {
      if (q[i] != (unsigned char)(i + 1)) {
        (void)fprintf(stderr,
                      "FAIL malloc_footprint_balance: grow lost byte %d\n", i);
        rc = 1;
        break;
      }
    }
    if (footprint() != base + ARTS_SYS_USABLE(q)) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: grow footprint %" PRIu64
                    " != base+usable %" PRIu64 "\n",
                    footprint(), base + ARTS_SYS_USABLE(q));
      rc = 1;
    }
    arts_free(q);
    if (footprint() != base) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: grow/free not net 0\n");
      rc = 1;
    }
  }

  /* ===== Part 6: concurrent malloc/free returns footprint to start. ===== */
  {
    uint64_t base = footprint();
    atomic_init(&g_gate, 0);
    pthread_t th[C_THREADS];
    for (int i = 0; i < C_THREADS; i++) {
      pthread_create(&th[i], NULL, churn, NULL);
    }
    atomic_store_explicit(&g_gate, 1, memory_order_release);
    for (int i = 0; i < C_THREADS; i++) {
      pthread_join(th[i], NULL);
    }
    if (footprint() != base) {
      (void)fprintf(stderr,
                    "FAIL malloc_footprint_balance: concurrent net %" PRIu64
                    " != %" PRIu64 " (torn footprint updates)\n",
                    footprint(), base);
      rc = 1;
    }
  }

  if (rc) {
    return 1;
  }
  printf("PASS malloc_footprint_balance: net-zero matched alloc/free, bump=="
         "usable size, size==0 NULL paths, calloc zeroing, realloc grow/shrink "
         "re-base + net-correct, %d-thread concurrent balance\n",
         C_THREADS);
  return 0;
}
