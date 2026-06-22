/* SPDX-License-Identifier: Apache-2.0
 *
 * Pure-unit test for libs/src/graph/block_distribution.c query arithmetic
 * (T261).
 *
 * The block-distribution QUERY functions (block_size, get_owner,
 * partition_start/end, get_local_index, guid_for_vertex, guid_for_partition)
 * and arts_block_dist_internal_init are pure, single-threaded arithmetic over a
 * caller-owned arts_block_dist_t.  We hand-construct dists (the init() paths
 * call arts_get_total_ranks()/arts_guid_reserve(), stubbed below so init()
 * itself is also exercised for the round-robin GUID->rank assignment) and pin:
 *
 *   - block_size coverage invariant: sum_{i in [0,num_blocks)} block_size(i)
 *     == num_vertices (exact partition, no gap/overlap), for n divisible and
 *     not divisible by num_blocks, and num_blocks==1.
 *   - contiguity: partition_end(i)+1 == partition_start(i+1) for non-last
 *     blocks; partition_end(last) == num_vertices-1.
 *   - get_owner boundary: vertex block_sz-1 -> block 0, block_sz -> block 1,...
 *   - local-index round-trip: get_local_index(v) == v - partition_start(owner);
 *     first vertex of a block -> 0, last -> block_size(owner)-1.
 *   - guid_for_partition / guid_for_vertex return the GUID reserved at init for
 *     that block, with the rank matching the round-robin assignment.
 *
 * Suspected-bug probes (run in CHILD processes; targets B140/B141/B142):
 *   - B140 (B-block-dist last-block underflow): n < num_blocks => block_sz via
 *     ceil >= 1, and (num_blocks-1)*block_sz can exceed num_vertices =>
 *     block_size(last) = num_vertices - (num_blocks-1)*block_sz underflows the
 *     unsigned arts_graph_sz_t to a huge value.
 *   - B141 (B-block-dist div-by-zero): num_blocks==0 => block_sz==0 =>
 *     get_owner does v/block_sz => SIGFPE (integer divide-by-zero).  Also
 *     guid_for_vertex(out-of-range v) indexes graphGuid[owner] OOB
 * (assert-only, gone in Release).
 *   - B142 (B-block-dist partition_end underflow): num_vertices==0 =>
 *     partition_end(last) = num_vertices-1 underflows to UINT64_MAX.
 *
 * Each probe runs in a forked child so the parent observes the outcome
 * (computed-underflow value, or SIGFPE) without dying.  Underflow probes are
 * CORRECT-AND-FAILING evidence of the suspected defects: we record the wrong
 * value and treat the suspected bug as CONFIRMED (we do NOT relax the test to
 * make it green).
 *
 * The ARTS runtime is NOT started.  block_distribution.c is compiled via
 * #include against in-test stubs (faithful allocator contract + a
 * rank-encoding arts_guid_reserve + fixed arts_get_total_ranks).
 */

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

#include "arts/gas/guid.h" /* ARTS_GUID_MAKE / ARTS_GUID_GET_RANK */

/* --- allocator contract (mirrors libs/src/core/utils/malloc.c) -------------
 */
void *arts_malloc(size_t size) { return size ? malloc(size) : NULL; }
void arts_free(void *ptr) { free(ptr); }

/* --- stubbed runtime symbols used only by the init() paths ------------------
 * A fixed rank count + a rank-encoding GUID reservation let us exercise the
 * real round-robin loop in arts_block_dist_init and then verify each block's
 * GUID carries the rank the round-robin assigned. */
#define STUB_TOTAL_RANKS 4u
static unsigned int g_reserve_key = 1;
unsigned int arts_get_total_ranks(void) { return STUB_TOTAL_RANKS; }
arts_guid_t arts_guid_reserve(arts_guid_kind_t kind, unsigned int rank) {
  /* encode the rank so guid_for_* can be checked against the round-robin. */
  return ARTS_GUID_MAKE((uint64_t)kind, (uint64_t)rank,
                        (uint64_t)(g_reserve_key++));
}
/* init_from_args parses argv; not exercised here (covered by runtime tests). */

#include "../../libs/src/graph/block_distribution.c"

static int g_fail = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL block_dist_query: %s (%s:%d)\n", msg, __FILE__,    \
              __LINE__);                                                       \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

/* Build a dist by hand (no GUID table contents needed for arithmetic tests). */
static arts_block_dist_t *make_dist(arts_graph_sz_t n,
                                    unsigned int num_blocks) {
  arts_block_dist_t *d = (arts_block_dist_t *)malloc(
      sizeof(arts_block_dist_t) + sizeof(arts_guid_t) * num_blocks);
  memset(d, 0, sizeof(arts_block_dist_t) + sizeof(arts_guid_t) * num_blocks);
  arts_block_dist_internal_init(d, n, /*m=*/0, num_blocks);
  return d;
}

/* ---- coverage invariant + contiguity for a (n, num_blocks) config ---------
 */
static void check_partition_invariants(arts_graph_sz_t n,
                                       unsigned int num_blocks) {
  arts_block_dist_t *d = make_dist(n, num_blocks);

  /* ceil block_sz */
  arts_graph_sz_t expect_block_sz = (n + num_blocks - 1) / num_blocks;
  CHECK(d->block_sz == expect_block_sz, "internal_init: ceil block_sz");

  /* sum of block_size(i) == num_vertices */
  arts_graph_sz_t sum = 0;
  for (unsigned int i = 0; i < num_blocks; i++) {
    sum += arts_block_dist_block_size(i, d);
  }
  CHECK(sum == n, "coverage: sum block_size == num_vertices");

  /* contiguity: end(i)+1 == start(i+1) for non-last; end(last)==n-1 */
  for (unsigned int i = 0; i < num_blocks; i++) {
    arts_vertex_t start = arts_block_dist_partition_start(i, d);
    arts_vertex_t end = arts_block_dist_partition_end(i, d);
    CHECK(start == (arts_vertex_t)d->block_sz * i, "partition_start == bsz*i");
    if (i + 1 < num_blocks) {
      arts_vertex_t next_start = arts_block_dist_partition_start(i + 1, d);
      CHECK(end + 1 == next_start, "contiguity end(i)+1 == start(i+1)");
    } else {
      CHECK(end == n - 1, "partition_end(last) == num_vertices-1");
    }
    /* block_size(i) == end-start+1 */
    CHECK(arts_block_dist_block_size(i, d) == end - start + 1,
          "block_size == end-start+1");
  }
  free(d);
}

static void test_partition_arithmetic(void) {
  check_partition_invariants(64, 2); /* divisible: bsz 32,32 */
  check_partition_invariants(8, 3);  /* not divisible: bsz 3 -> 3,3,2 */
  check_partition_invariants(10, 1); /* single block -> whole range */
  check_partition_invariants(100, 7);
  check_partition_invariants(1, 1);
}

/* ---- get_owner boundary + local-index round-trip --------------------------
 */
static void test_owner_and_local_index(void) {
  arts_block_dist_t *d = make_dist(8, 3); /* block_sz=3 -> blocks 0:[0..2]
                                             1:[3..5] 2:[6..7] */
  CHECK(arts_block_dist_get_owner(0, d) == 0, "owner(0)==0");
  CHECK(arts_block_dist_get_owner(2, d) == 0, "owner(block_sz-1)==0");
  CHECK(arts_block_dist_get_owner(3, d) == 1, "owner(block_sz)==1");
  CHECK(arts_block_dist_get_owner(5, d) == 1, "owner(5)==1");
  CHECK(arts_block_dist_get_owner(6, d) == 2, "owner(6)==2");
  CHECK(arts_block_dist_get_owner(7, d) == 2, "owner(last)==2");

  /* local-index round-trip: first vertex of block -> 0; last -> bsz-1 */
  for (arts_vertex_t v = 0; v < 8; v++) {
    unsigned int owner = arts_block_dist_get_owner(v, d);
    arts_vertex_t start = arts_block_dist_partition_start(owner, d);
    arts_local_index_t li = arts_block_dist_get_local_index(v, d);
    CHECK(li == v - start, "local_index == v - partition_start(owner)");
    CHECK(li < arts_block_dist_block_size(owner, d),
          "local_index < block_size(owner)");
  }
  CHECK(arts_block_dist_get_local_index(0, d) == 0, "first-of-block0 -> 0");
  CHECK(arts_block_dist_get_local_index(3, d) == 0, "first-of-block1 -> 0");
  CHECK(arts_block_dist_get_local_index(5, d) == 2, "last-of-block1 -> bsz-1");
  free(d);
}

/* ---- init() round-robin GUID->rank assignment -----------------------------
 */
static void test_init_roundrobin(void) {
  /* num_blocks not divisible by ranks: 6 blocks over 4 ranks.
   * blocks_per_node = 1, mod = 2 -> ranks 0,1 get 2 blocks; ranks 2,3 get 1.
   * round-robin contiguous fill order:
   *   rank0: blocks 0,1 ; rank1: blocks 2,3 ; rank2: block 4 ; rank3: block 5
   */
  g_reserve_key = 1;
  arts_block_dist_t *d =
      arts_block_dist_init(/*n=*/60, /*m=*/0, /*num_blocks=*/6, ARTS_GUID_DB);
  CHECK(d != NULL, "init returned non-NULL");
  CHECK(d->num_blocks == 6, "init num_blocks");
  unsigned int expect_rank[6] = {0, 0, 1, 1, 2, 3};
  for (unsigned int i = 0; i < 6; i++) {
    arts_guid_t g = arts_block_dist_guid_for_partition(d, i);
    unsigned int r = (unsigned int)ARTS_GUID_GET_RANK(g);
    CHECK(r == expect_rank[i], "guid_for_partition rank == round-robin");
  }
  /* guid_for_vertex: vertex in block i returns block i's GUID. block_sz =
   * ceil(60/6)=10, so vertex 25 -> block 2. */
  arts_guid_t gv = arts_block_dist_guid_for_vertex(25, d);
  CHECK(gv == arts_block_dist_guid_for_partition(d, 2),
        "guid_for_vertex(25) == guid_for_partition(2)");
  free(d);
}

/* ===================== suspected-bug probes (child) ========================
 */

/* B140: last-block unsigned underflow when n < num_blocks. */
static void test_b140_last_block_underflow(void) {
  /* n=3, num_blocks=5 -> block_sz = ceil(3/5)=1.
   * block_size(last=4) = n - (num_blocks-1)*block_sz = 3 - 4*1 = -1
   * -> underflows arts_graph_sz_t (uint64) to UINT64_MAX. */
  arts_block_dist_t *d = make_dist(3, 5);
  CHECK(d->block_sz == 1, "B140 setup: ceil(3/5)==1");
  arts_graph_sz_t last = arts_block_dist_block_size(4, d);
  /* The contract WANTS sum==num_vertices; here block 4 cannot be the huge
   * underflowed value.  Record CONFIRMED if it underflowed. */
  if (last > d->num_vertices) {
    fprintf(stderr,
            "  B140 CONFIRMED: block_size(last)=%" PRIu64
            " underflowed (n=3,num_blocks=5)\n",
            (uint64_t)last);
    /* This IS the bug; leave the assertion expressing the CORRECT contract so
     * the test fails loudly and records exposes_runtime_bug. */
    CHECK(last <= d->num_vertices,
          "B140: last block_size must not exceed num_vertices (underflow bug)");
  } else {
    CHECK(last <= d->num_vertices, "B140: no underflow (would be a fix)");
  }
  free(d);
}

/* B142: partition_end(last) underflow when num_vertices==0. */
static void test_b142_partition_end_underflow(void) {
  arts_block_dist_t *d = make_dist(0, 1); /* n=0, single block */
  arts_vertex_t end = arts_block_dist_partition_end(0, d);
  if (end == UINT64_MAX) {
    fprintf(stderr,
            "  B142 CONFIRMED: partition_end(last) underflowed to UINT64_MAX "
            "(num_vertices==0)\n");
  }
  CHECK(end != UINT64_MAX,
        "B142: partition_end on empty graph must not underflow to UINT64_MAX");
  free(d);
}

/* B141: get_owner div-by-zero when block_sz==0 (num_blocks==0).  Runs in a
 * child.  The integer divide-by-zero manifests as a raw SIGFPE in a plain build
 * and as a sanitizer-reported fatal error (UBSan "division by zero" / ASan
 * "FPE") that aborts the child with a nonzero exit code under -fsanitize.  The
 * ONLY way the child reaches _exit(0) is if the divide-by-zero did NOT occur,
 * so any abnormal termination CONFIRMS B141. */
static void child_b141_divzero(void) {
  arts_block_dist_t *d = make_dist(10, 0); /* num_blocks==0 -> block_sz==0 */
  volatile unsigned int owner = arts_block_dist_get_owner(5, d);
  (void)owner;
  _exit(0); /* reached only if NO div-by-zero occurred */
}

static void test_b141_divzero(void) {
  pid_t pid = fork();
  if (pid == 0) {
    child_b141_divzero();
    _exit(99);
  }
  CHECK(pid > 0, "fork for B141 child");
  int status = 0;
  waitpid(pid, &status, 0);

  bool clean_no_divzero = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  bool sigfpe = WIFSIGNALED(status) && WTERMSIG(status) == SIGFPE;
  bool san_abort = !clean_no_divzero; /* SIGFPE, SIGABRT, or nonzero exit */

  if (clean_no_divzero) {
    CHECK(1, "B141: no div-by-zero (would be a fix)");
  } else {
    fprintf(stderr,
            "  B141 CONFIRMED: get_owner with block_sz==0 (num_blocks==0) "
            "divides by zero (%s, status=0x%x)\n",
            sigfpe ? "raw SIGFPE" : "sanitizer-reported FPE/abort", status);
    /* CONFIRMED bug: the correct contract is no crash. Fail loudly. */
    CHECK(!san_abort,
          "B141: get_owner must not divide by zero (num_blocks==0)");
  }
}

int main(void) {
  test_partition_arithmetic();
  test_owner_and_local_index();
  test_init_roundrobin();

  /* suspected-bug probes: these are CORRECT-AND-FAILING if the bugs exist. */
  test_b140_last_block_underflow();
  test_b142_partition_end_underflow();
  test_b141_divzero();

  if (g_fail) {
    fprintf(stderr, "FAIL block_dist_query: one or more checks failed "
                    "(see CONFIRMED lines for suspected runtime bugs)\n");
    return 1;
  }
  printf("PASS block_dist_query: coverage/contiguity/owner/local-index "
         "round-trip + round-robin GUID rank; no underflow/div-by-zero\n");
  return 0;
}
