/* SPDX-License-Identifier: Apache-2.0
 *
 * T246 — alignment + realloc correctness of arts_malloc_aligned /
 * arts_calloc_aligned / arts_realloc in libs/src/core/utils/malloc.c.
 *
 * Properties:
 *   A. arts_malloc_aligned(n, a) returns an a-aligned pointer for every valid
 *      power-of-two a >= 16, and arts_free frees it directly (ASan must stay
 *      clean — no heap corruption, no leak, no double-free).  The allocator's
 *      native aligned entry point returns a directly-freeable pointer; there is
 *      no manual over-allocate-and-offset and no stored base.
 *   B. arts_calloc_aligned(n, sz, a) is a-aligned AND zero-initialised.
 *   C. arts_realloc on an aligned block preserves the first min(old,new) bytes.
 *      It yields only the BASE allocator alignment — the original
 * over-alignment is intentionally NOT preserved (realloc has no alignment
 * argument).  This is the documented contract: an over-aligned block must not
 * be grown via realloc; the test pins that the data survives and the result is
 * at least base-aligned. D. arts_realloc shrink preserves the data (the
 * returned pointer may or may not equal the original — native realloc decides).
 *   E. size==0 / NULL-return paths (no abort): arts_malloc_aligned(0,a)==NULL,
 *      arts_realloc(NULL, n) == arts_malloc(n), arts_realloc(p, 0) frees and
 *      returns NULL.
 *
 * INVALID ALIGNMENT ABORTS: arts_malloc_aligned with align<16 hits the explicit
 * minimum-alignment guard -> ARTS_ERROR -> abort; a zero / non-power-of-two
 * alignment instead fails inside the allocator (NULL return), which the wrapper
 * also turns into ARTS_ERROR -> abort.  We verify the align<16 path aborts by
 * forking a child and checking it died via SIGABRT.
 *
 * STANDALONE STRATEGY identical to malloc_footprint_balance.c: pre-define the
 * heavy header include guards and supply minimal ARTS_ERROR / footprint macros,
 * then #include the malloc.c TU directly.
 */

#include <inttypes.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static unsigned long long g_footprint;

#define ARTS_SYSTEM_PRINT_H 1
#define ARTS_DEFS_H 1
#define ARTS_ERROR(...)                                                        \
  do {                                                                         \
    (void)fprintf(stderr, "ARTS_ERROR: " __VA_ARGS__);                         \
    (void)fprintf(stderr, "\n");                                               \
    abort();                                                                   \
  } while (0)
#define INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(v)                                 \
  (g_footprint += (unsigned long long)(v))
#define DECREMENT_BYTES_MEMORY_FOOTPRINT_BY(v)                                 \
  (g_footprint -= (unsigned long long)(v))

#include "../../libs/src/core/utils/malloc.c"

static int g_fail;

static void check_aligned(void *p, size_t a, const char *what) {
  if (p == NULL) {
    (void)fprintf(stderr, "FAIL malloc_alignment: %s returned NULL\n", what);
    g_fail = 1;
    return;
  }
  if (((uintptr_t)p & (a - 1)) != 0) {
    (void)fprintf(stderr, "FAIL malloc_alignment: %s ptr %p not %zu-aligned\n",
                  what, p, a);
    g_fail = 1;
  }
}

int main(void) {
  /* ===== A: a-aligned allocations across the valid alignment classes. ===== */
  {
    size_t aligns[] = {16, 32, 64, 128, 256, 512, 1024, 4096};
    for (size_t i = 0; i < sizeof(aligns) / sizeof(aligns[0]); i++) {
      size_t a = aligns[i];
      for (size_t sz = 1; sz <= 300; sz += 37) {
        void *p = arts_malloc_aligned(sz, a);
        check_aligned(p, a, "malloc_align");
        /* touch full payload so ASan catches under-allocation. */
        memset(p, 0xAB, sz);
        arts_free(p);
      }
    }
  }

  /* ===== B: calloc_align is aligned AND zeroed. ===== */
  {
    size_t a = 128;
    unsigned char *p = (unsigned char *)arts_calloc_aligned(40, 3, a);
    check_aligned(p, a, "calloc_align");
    for (size_t i = 0; i < 40 * 3; i++) {
      if (p[i] != 0) {
        (void)fprintf(stderr, "FAIL malloc_alignment: calloc_align not 0\n");
        g_fail = 1;
        break;
      }
    }
    arts_free(p);
  }

  /* ===== C: realloc grow preserves bytes; result is at least base-aligned
   *          (over-alignment intentionally not preserved). ===== */
  {
    size_t a = 256;
    unsigned char *p = (unsigned char *)arts_malloc_aligned(50, a);
    check_aligned(p, a, "pre-grow malloc_align");
    for (int i = 0; i < 50; i++) {
      p[i] = (unsigned char)(i + 1);
    }
    unsigned char *q = (unsigned char *)arts_realloc(p, 500); /* grow */
    check_aligned(q, ALIGNMENT, "grown realloc"); /* base alignment only */
    for (int i = 0; i < 50; i++) {
      if (q[i] != (unsigned char)(i + 1)) {
        (void)fprintf(stderr, "FAIL malloc_alignment: grow lost byte %d\n", i);
        g_fail = 1;
        break;
      }
    }
    memset(q, 0xCD, 500); /* full new payload usable */
    arts_free(q);
  }

  /* ===== D: realloc shrink preserves the data. ===== */
  {
    size_t a = 64;
    unsigned char *p = (unsigned char *)arts_malloc_aligned(400, a);
    for (int i = 0; i < 100; i++) {
      p[i] = (unsigned char)(i + 5);
    }
    unsigned char *q = (unsigned char *)arts_realloc(p, 100); /* shrink */
    check_aligned(q, ALIGNMENT, "shrunk realloc");
    for (int i = 0; i < 100; i++) {
      if (q[i] != (unsigned char)(i + 5)) {
        (void)fprintf(stderr, "FAIL malloc_alignment: shrink lost byte %d\n",
                      i);
        g_fail = 1;
        break;
      }
    }
    arts_free(q);
  }

  /* ===== E: realloc NULL/zero paths. ===== */
  {
    void *p = arts_realloc(NULL, 80); /* == arts_malloc(80) */
    if (p == NULL) {
      (void)fprintf(stderr, "FAIL malloc_alignment: realloc(NULL,80)\n");
      g_fail = 1;
    } else {
      memset(p, 1, 80);
      if (arts_realloc(p, 0) != NULL) {
        (void)fprintf(stderr, "FAIL malloc_alignment: realloc(p,0)!=NULL\n");
        g_fail = 1;
      }
    }
  }

  /* ===== Invalid-param abort path (align<16 must abort, not return). ===== */
  {
    pid_t pid = fork();
    if (pid == 0) {
      /* child: silence stderr noise, then trigger the abort path */
      (void)freopen("/dev/null", "w", stderr);
      void *bad = arts_malloc_aligned(64, 8); /* align 8 < 16 -> ARTS_ERROR */
      /* should never reach here */
      (void)bad;
      _exit(0);
    } else if (pid > 0) {
      int status = 0;
      (void)waitpid(pid, &status, 0);
      if (!(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT)) {
        (void)fprintf(stderr,
                      "FAIL malloc_alignment: malloc_align(align<16) did not "
                      "abort (status=%d)\n",
                      status);
        g_fail = 1;
      }
    } else {
      (void)fprintf(stderr, "FAIL malloc_alignment: fork failed\n");
      g_fail = 1;
    }
  }

  if (g_fail) {
    return 1;
  }
  printf("PASS malloc_alignment: a-aligned alloc (16..4096), calloc_align "
         "zeroed, realloc grow/shrink preserve bytes (base-aligned), NULL/zero "
         "paths, align<16 aborts\n");
  return 0;
}
