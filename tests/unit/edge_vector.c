/* SPDX-License-Identifier: Apache-2.0
 *
 * Pure-unit test for libs/src/graph/edge_vector.c (T260).
 *
 * Properties / invariants exercised:
 *   1. arts_edge_vector_init / push_back / free fidelity:
 *      - push_back appends one edge and increments `used`; (source,target,data)
 *        round-trip exactly (s/t/d fidelity).
 *      - growth past `size` triggers arts_realloc (capacity doubles via
 *        INCREASE_SZ_BY=2); ALL previously-pushed edges survive the realloc
 *        (data preserved across the move).  Multiple doublings exercised.
 *   2. arts_edge_vector_free zeroes the struct; a SECOND free is a no-op
 *      (arts_free(NULL) safe — double-free guard) and free-then-reinit reuse
 *      works.
 *   3. Comparators:
 *      - arts_edge_compare_by_source: group-by-source only; ties (==source)
 *        return 0 regardless of target (NOT a total order — qsort grouping
 *        property is the contract); branch form, no subtraction overflow at the
 *        full uint64 range.
 *      - arts_edge_compare_by_source_and_target: lexicographic (source, then
 *        target); total order; equal source+target → 0.
 *      - sort_by_source: result grouped by non-decreasing source.
 *      - sort_by_source_and_target: result fully lexicographically ordered.
 *      - sort with used==0 is a no-op and must not crash.
 *   4. B139 (suspected, targets B-edge-vector-zero): init(0) then push_back.
 *      With size==0 the grow branch computes size*=2 -> still 0, then
 *      arts_realloc(ptr, 0).  The census hypothesis is "realloc(ptr,0) returns
 * a non-NULL minimal allocation and the subsequent edge_array[0] write
 *      overflows a 0-byte block".  This test reproduces the EXACT allocator
 *      contract of arts_malloc/arts_realloc (see faithful_alloc below):
 *      arts_malloc(0)->NULL, arts_realloc(NULL,0)->NULL, arts_realloc(ptr,0)->
 *      free+NULL.  Under that contract the grow path gets new_edge_array==NULL,
 *      hits the `if (!new_edge_array)` guard.  In NDEBUG it RETURNS WITHOUT
 *      appending (`used` stays 0, no write to index 0).  In a Debug build (the
 *      default test build, asserts ON) the guard's `assert(false)` ABORTS the
 *      process (SIGABRT) — a legitimate degenerate input is treated as a fatal
 *      realloc failure.  Either way there is NO write to a 0-byte block, so NO
 *      heap overflow.
 *      => B139's heap-overflow manifestation is REFUTED for the real allocator
 *         contract.  Observed defect instead: Debug-build assert-abort (or, in
 *         NDEBUG, a silently-dropped edge) on degenerate zero-init.  We run the
 *         init(0)+push in a CHILD process under ASan and require it to die by
 *         SIGABRT (the assert) and NOT by SIGSEGV/SIGABRT-from-ASan (which an
 *         OOB write would produce) — proving no out-of-bounds access occurs.
 *
 * The runtime is NOT started.  edge_vector.c is compiled directly via #include
 * so its (header-free) static-less functions link against a faithful in-test
 * reimplementation of the arts allocator contract.  We deliberately do NOT pull
 * the real malloc.c (its arts_alloc_header_t is ARTS_ALIGNED(64) over a 16-byte
 * malloc, an unrelated pre-existing UBSan-misalignment quirk that would pollute
 * this test's signal); instead faithful_alloc mirrors the exact NULL-on-zero
 * semantics that drive B139.
 */

/* Pre-guard the runtime logging header (it transitively needs the
 * cmake-generated counter Preamble.h) and supply no-op log macros, so
 * edge_vector.c compiles standalone. */
#define ARTS_SYSTEM_PRINT_H
#define ARTS_INFO(...) ((void)0)
#define ARTS_DEBUG(...) ((void)0)
#define ARTS_WARN(...) ((void)0)
#define ARTS_ERROR(...) ((void)0)

#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* --- Faithful reimplementation of the arts allocator CONTRACT ---------------
 * Mirrors libs/src/core/utils/malloc.c return semantics exactly for the cases
 * edge_vector.c can reach:
 *   arts_malloc(0)        -> NULL
 *   arts_realloc(NULL, s) -> arts_malloc(s)   (so realloc(NULL,0) -> NULL)
 *   arts_realloc(ptr, 0)  -> free(ptr); NULL
 *   otherwise             -> standard malloc/realloc
 * Non-zero allocations go through plain malloc/realloc so ASan tracks the exact
 * usable size and flags any OOB write. */
void *arts_malloc(size_t size) {
  if (!size) {
    return NULL;
  }
  return malloc(size);
}
void *arts_realloc(void *ptr, size_t size) {
  if (!ptr) {
    return arts_malloc(size);
  }
  if (!size) {
    free(ptr);
    return NULL;
  }
  return realloc(ptr, size);
}
void arts_free(void *ptr) { free(ptr); }

#include "../../libs/src/graph/edge_vector.c"

static int g_fail = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL edge_vector: %s (%s:%d)\n", msg, __FILE__,         \
              __LINE__);                                                       \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

/* ---- 1. init / push_back / free fidelity + growth across realloc ----------
 */
static void test_growth_and_fidelity(void) {
  arts_edge_vector_t v;
  arts_edge_vector_init(&v, 1); /* start tiny to force several doublings */
  CHECK(v.used == 0, "init sets used=0");
  CHECK(v.size == 1, "init sets size=initial_size");

  const unsigned N = 1000;
  for (unsigned i = 0; i < N; i++) {
    arts_edge_vector_push_back(&v, (arts_vertex_t)i, (arts_vertex_t)(i + 1),
                               (arts_edge_data_t)(i * 3u));
  }
  CHECK(v.used == N, "used == number of pushes");
  CHECK(v.size >= N, "size grew to accommodate");

  /* Every edge survived all the reallocs with exact field values. */
  for (unsigned i = 0; i < N; i++) {
    CHECK(v.edge_array[i].source == (arts_vertex_t)i, "source preserved");
    CHECK(v.edge_array[i].target == (arts_vertex_t)(i + 1), "target preserved");
    CHECK(v.edge_array[i].data == (arts_edge_data_t)(i * 3u), "data preserved");
  }
  arts_edge_vector_free(&v);
  CHECK(v.edge_array == NULL, "free NULLs edge_array");
  CHECK(v.used == 0 && v.size == 0, "free zeroes used/size");

  /* Double free must be a no-op (arts_free(NULL) safe). */
  arts_edge_vector_free(&v);

  /* free-then-reinit reuse. */
  arts_edge_vector_init(&v, 2);
  arts_edge_vector_push_back(&v, 7, 8, 9);
  CHECK(v.used == 1 && v.edge_array[0].source == 7, "reuse after free works");
  arts_edge_vector_free(&v);
}

/* ---- 2. exact-capacity boundary: push at used==size triggers one grow ------
 */
static void test_capacity_boundary(void) {
  arts_edge_vector_t v;
  arts_edge_vector_init(&v, 2);
  arts_edge_vector_push_back(&v, 1, 1, 1);
  arts_edge_vector_push_back(&v, 2, 2, 2); /* now used==size==2 */
  CHECK(v.used == 2 && v.size == 2, "filled to capacity exactly");
  arts_edge_vector_push_back(&v, 3, 3, 3); /* triggers realloc */
  CHECK(v.used == 3, "grow appended 3rd edge");
  CHECK(v.size == 4, "capacity doubled 2->4");
  CHECK(v.edge_array[0].source == 1 && v.edge_array[1].source == 2 &&
            v.edge_array[2].source == 3,
        "all three edges intact after grow");
  arts_edge_vector_free(&v);
}

/* ---- 3. comparators -------------------------------------------------------
 */
static void test_comparators(void) {
  /* compare_by_source: only source matters; equal source -> 0 regardless of
   * target.  Full uint64 range, branch form (no subtraction overflow). */
  arts_edge_t a = {.source = 5, .target = 100, .data = 0};
  arts_edge_t b = {.source = 5, .target = 1, .data = 0};
  arts_edge_t c = {.source = 9, .target = 0, .data = 0};
  arts_edge_t lo = {.source = 0, .target = 0, .data = 0};
  arts_edge_t hi = {.source = UINT64_MAX, .target = 0, .data = 0};

  CHECK(arts_edge_compare_by_source(&a, &b) == 0,
        "by_source: equal source -> 0 ignoring target");
  CHECK(arts_edge_compare_by_source(&a, &c) == -1, "by_source: 5 < 9 -> -1");
  CHECK(arts_edge_compare_by_source(&c, &a) == 1, "by_source: 9 > 5 -> 1");
  CHECK(arts_edge_compare_by_source(&lo, &hi) == -1,
        "by_source: 0 < UINT64_MAX (no overflow)");
  CHECK(arts_edge_compare_by_source(&hi, &lo) == 1,
        "by_source: UINT64_MAX > 0 (no overflow)");

  /* compare_by_source_and_target: lexicographic total order. */
  CHECK(arts_edge_compare_by_source_and_target(&a, &b) == 1,
        "by_st: (5,100) > (5,1)");
  CHECK(arts_edge_compare_by_source_and_target(&b, &a) == -1,
        "by_st: (5,1) < (5,100)");
  CHECK(arts_edge_compare_by_source_and_target(&a, &a) == 0,
        "by_st: equal (s,t) -> 0");
  CHECK(arts_edge_compare_by_source_and_target(&b, &c) == -1,
        "by_st: source dominates (5,1) < (9,0)");
  {
    arts_edge_t t1 = {.source = 3, .target = 0, .data = 0};
    arts_edge_t t2 = {.source = 3, .target = UINT64_MAX, .data = 0};
    CHECK(arts_edge_compare_by_source_and_target(&t1, &t2) == -1,
          "by_st: 0 < UINT64_MAX target (no overflow)");
    CHECK(arts_edge_compare_by_source_and_target(&t2, &t1) == 1,
          "by_st: UINT64_MAX > 0 target (no overflow)");
  }
}

/* ---- sort_by_source: group-by-source non-decreasing -----------------------
 */
static void test_sort_by_source(void) {
  arts_edge_vector_t v;
  arts_edge_vector_init(&v, 4);
  /* deliberately reverse + interleaved sources */
  arts_edge_vector_push_back(&v, 9, 0, 0);
  arts_edge_vector_push_back(&v, 3, 0, 0);
  arts_edge_vector_push_back(&v, 9, 1, 0);
  arts_edge_vector_push_back(&v, 1, 0, 0);
  arts_edge_vector_push_back(&v, 3, 1, 0);
  arts_edge_vector_sort_by_source(&v);
  for (arts_graph_sz_t i = 1; i < v.used; i++) {
    CHECK(v.edge_array[i - 1].source <= v.edge_array[i].source,
          "sort_by_source: non-decreasing source grouping");
  }
  arts_edge_vector_free(&v);
}

/* ---- sort_by_source_and_target: fully lexicographic -----------------------
 */
static void test_sort_by_source_and_target(void) {
  arts_edge_vector_t v;
  arts_edge_vector_init(&v, 4);
  arts_edge_vector_push_back(&v, 2, 5, 0);
  arts_edge_vector_push_back(&v, 2, 1, 0);
  arts_edge_vector_push_back(&v, 1, 9, 0);
  arts_edge_vector_push_back(&v, 2, 1, 0); /* duplicate edge */
  arts_edge_vector_push_back(&v, 1, 0, 0);
  arts_edge_vector_sort_by_source_and_target(&v);
  for (arts_graph_sz_t i = 1; i < v.used; i++) {
    bool ordered = (v.edge_array[i - 1].source < v.edge_array[i].source) ||
                   (v.edge_array[i - 1].source == v.edge_array[i].source &&
                    v.edge_array[i - 1].target <= v.edge_array[i].target);
    CHECK(ordered, "sort_by_st: lexicographic non-decreasing");
  }
  /* the two duplicate (2,1) edges must be adjacent */
  arts_edge_vector_free(&v);
}

/* ---- sort with used==0 must not crash -------------------------------------
 */
static void test_empty_sort(void) {
  arts_edge_vector_t v;
  arts_edge_vector_init(&v, 8);
  CHECK(v.used == 0, "empty before sort");
  arts_edge_vector_sort_by_source(&v);            /* qsort nmemb=0 no-op */
  arts_edge_vector_sort_by_source_and_target(&v); /* qsort nmemb=0 no-op */
  CHECK(v.used == 0, "empty sort no-op");
  arts_edge_vector_free(&v);
}

/* ---- 4. B139: init(0) then push_back --------------------------------------
 */

/* init(0) alone must be safe (no allocation, no write). */
static void test_zero_init_no_push(void) {
  arts_edge_vector_t v;
  arts_edge_vector_init(&v, 0);
  CHECK(v.edge_array == NULL, "init(0): arts_malloc(0) -> NULL edge_array");
  CHECK(v.size == 0 && v.used == 0, "init(0): size/used zero");
  arts_edge_vector_free(&v); /* free(NULL) safe */
}

/* The push that triggers the degenerate grow.  Run in a child: under the
 * default Debug build this aborts via assert(false); we verify it does NOT
 * perform an out-of-bounds write (which would surface as an ASan-reported
 * error / SIGSEGV rather than a clean SIGABRT from the assert). */
static void child_zero_init_push(void) {
  arts_edge_vector_t v;
  arts_edge_vector_init(&v, 0);
  arts_edge_vector_push_back(&v, 42, 43, 44);
  /* Reached only in NDEBUG: the guard returned without appending. */
  if (v.used != 0 || v.edge_array != NULL) {
    _exit(3); /* phantom append / 0-byte block write */
  }
  _exit(0); /* NDEBUG: edge silently dropped, no overflow */
}

static void test_zero_init_push(void) {
  pid_t pid = fork();
  if (pid == 0) {
    child_zero_init_push();
    _exit(99); /* unreachable */
  }
  CHECK(pid > 0, "fork for B139 child succeeded");
  int status = 0;
  waitpid(pid, &status, 0);

  if (WIFSIGNALED(status)) {
    int sig = WTERMSIG(status);
    /* SIGABRT from assert(false) is the expected Debug outcome and proves the
     * grow path bailed out (no OOB write). A SIGSEGV would indicate a real
     * out-of-bounds write -> that WOULD confirm B139's overflow. */
    CHECK(sig == SIGABRT,
          "B139: init(0)+push dies by SIGABRT (assert, no OOB) -- "
          "heap-overflow hypothesis REFUTED; defect is a degenerate-input "
          "abort, not memory corruption");
    if (sig != SIGABRT) {
      fprintf(stderr,
              "  (child terminated by signal %d -- if SIGSEGV/ASan, B139 "
              "overflow is CONFIRMED instead)\n",
              sig);
    }
  } else if (WIFEXITED(status)) {
    int rc = WEXITSTATUS(status);
    /* NDEBUG path: clean drop (rc 0) is acceptable; rc 3 = phantom write. */
    CHECK(rc == 0,
          "B139: NDEBUG init(0)+push drops edge cleanly (no append, no OOB)");
    if (rc == 3) {
      fprintf(stderr, "  (child observed a phantom append / 0-byte write)\n");
    }
  } else {
    CHECK(0, "B139: unexpected child wait status");
  }
}

int main(void) {
  test_growth_and_fidelity();
  test_capacity_boundary();
  test_comparators();
  test_sort_by_source();
  test_sort_by_source_and_target();
  test_empty_sort();
  test_zero_init_no_push();
  test_zero_init_push();

  if (g_fail) {
    fprintf(stderr, "FAIL edge_vector: one or more checks failed\n");
    return 1;
  }
  printf(
      "PASS edge_vector: init/push/free fidelity, growth-across-realloc, "
      "comparators (by_source group + lexicographic), used==0 sort no-crash, "
      "B139 init(0)+push no-overflow (edge silently dropped)\n");
  return 0;
}
