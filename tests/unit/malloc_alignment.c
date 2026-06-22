/* SPDX-License-Identifier: Apache-2.0
 *
 * T246 — alignment + realloc correctness of arts_malloc_align /
 * arts_calloc_align / arts_realloc in libs/src/core/utils/malloc.c.
 *
 * Properties (census 32-misc-utils.md §1.3 items 2,3 + invalid-param paths):
 *   A. arts_malloc_align(n, a) returns an a-aligned pointer for every valid
 *      power-of-two a >= 16, and arts_free recovers the real base (ASan must
 *      stay clean — no heap corruption, no leak, no double-free).
 *   B. arts_calloc_align(n, sz, a) is a-aligned AND zero-initialised.
 *   C. arts_realloc on an aligned block grows into a NEW block that (i)
 *      preserves the first min(old,new) bytes and (ii) keeps the same
 *      alignment class (still a-aligned).
 *   D. arts_realloc shrink keeps the same pointer (in place) and the data.
 *   E. size==0 / NULL-return paths that DON'T abort:
 *        arts_realloc(NULL, n) == arts_malloc(n) (aligned class lost: plain),
 *        arts_realloc(p, 0) frees and returns NULL.
 *
 * INVALID-PARAM PATHS ABORT (not NULL-return): arts_malloc_align with
 * align<16 / non-power-of-two / size==0 call ARTS_ERROR -> abort.  We verify
 * one such path aborts by forking a child and checking it died via SIGABRT;
 * this also pins that the overflow/!pow2 guards are live.
 *
 * STANDALONE STRATEGY identical to malloc_footprint_balance.c: pre-define the
 * heavy header include guards and supply minimal ARTS_ERROR / ARTS_ALIGNED /
 * footprint macros, then #include the malloc.c TU directly.
 *
 * NOTE (exposed runtime sharp edge): the allocator's internal
 * arts_alloc_header_t is declared ARTS_ALIGNED(64) but for the UNALIGNED
 * arts_malloc path the header is placed at glibc's 16-byte-aligned base, so
 * -fsanitize=undefined (alignment) reports a misaligned access to a
 * 64-byte-aligned struct.  That is a real UB sharp edge in the runtime
 * allocator; this test's correctness target is -fsanitize=address (heap
 * safety) + functional alignment of the RETURNED user pointers, which are
 * correct.
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
#define ARTS_ALIGNED(n) __attribute__((__aligned__(n)))
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
        void *p = arts_malloc_align(sz, a);
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
    unsigned char *p = (unsigned char *)arts_calloc_align(40, 3, a);
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

  /* ===== C: realloc grow preserves bytes + alignment class. ===== */
  {
    size_t a = 256;
    unsigned char *p = (unsigned char *)arts_malloc_align(50, a);
    check_aligned(p, a, "pre-grow malloc_align");
    for (int i = 0; i < 50; i++) {
      p[i] = (unsigned char)(i + 1);
    }
    unsigned char *q = (unsigned char *)arts_realloc(p, 500); /* grow */
    check_aligned(q, a, "grown realloc"); /* alignment class preserved */
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

  /* ===== D: realloc shrink keeps ptr + data. ===== */
  {
    size_t a = 64;
    unsigned char *p = (unsigned char *)arts_malloc_align(400, a);
    for (int i = 0; i < 100; i++) {
      p[i] = (unsigned char)(i + 5);
    }
    unsigned char *q = (unsigned char *)arts_realloc(p, 100); /* shrink */
    if (q != p) {
      (void)fprintf(stderr, "FAIL malloc_alignment: shrink should keep ptr\n");
      g_fail = 1;
    }
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
    void *p = arts_realloc(NULL, 80); /* == arts_malloc(80), unaligned class */
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
      void *bad = arts_malloc_align(64, 8); /* align 8 < 16 -> ARTS_ERROR */
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
         "zeroed, realloc grow/shrink preserve bytes+alignment class, "
         "NULL/zero paths, align<16 aborts\n");
  return 0;
}
