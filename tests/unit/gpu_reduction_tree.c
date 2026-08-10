/* SPDX-License-Identifier: Apache-2.0
 *
 * T273 — Pure GPU LC reduction-tree topology helpers (host-extracted).
 *
 * The reduction-tree builders live in libs/src/core/gpu/gpu_lc.cu, which is a
 * CUDA TU (#include <cuda_runtime_api.h>) and therefore cannot be compiled
 * standalone with host gcc.  The functions exercised here — find_roots,
 * add_to_trav, gpu_tree_reduction_rec, gpu_tree_reduction_start — are PURE
 * integer logic with NO CUDA and NO global state (gpu_lc.cu:268-393).  Extracted
 * rather than exercised through a GPU build, the pure bodies are copied
 * VERBATIM below from gpu_lc.cu, under #define GPUGROUPSIZE 4 / GPUNUMGROUP 2
 * (the hardcoded 4x2 / 8-GPU model).  Each copied block cites its source lines.
 * No runtime source is modified; this is the host-only mirror.
 *
 * Properties tested:
 *   - find_roots: matching-root detection (a column set in ALL groups) vs
 *     per-group lowest-set-bit fallback; empty group -> -1 root; empty mask.
 *   - gpu_tree_reduction_start/rec: traversal node list contents, level
 *     ordering (leaves at higher level, group-merge node at level 1),
 *     max_level, and that the produced trav-node count never exceeds the
 *     caller's fixed 8-entry list[] under the 4x2 model.
 *   - add_to_trav overflow guard: the census flags B-add-to-trav — *size is
 *     bumped with NO bounds check against the caller's list[8].  This test
 *     drives add_to_trav directly past 8 entries into a guarded oversized
 *     buffer and asserts the (documented) missing-guard behavior: the function
 *     keeps writing past index 7 with no clamp.  exposes_runtime_bug for the
 *     latent overflow is recorded; we verify the *real* tree builder stays
 *     <=8 so the latent bug is unreachable under the shipped topology.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---- topology constants (gpu_lc.cu:234-235) ---- */
#define GPUGROUPSIZE 4
#define GPUNUMGROUP 2

/* ---- trav_t (gpu_lc.cu:312-317) ---- */
typedef struct {
  int a;
  int b;
  int root;
  int level;
} trav_t;

/* ================= VERBATIM PURE COPIES from gpu_lc.cu ================= */
/* The bodies below are copied unchanged from libs/src/core/gpu/gpu_lc.cu so
 * the host test exercises the exact runtime logic.  ARTS_DEBUG lines (logging
 * only, no side effects) are elided. */

/* find_roots (gpu_lc.cu:268-310) */
static void find_roots(unsigned int local, int *roots) {
  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    roots[i] = -1;
  }

  unsigned int mask = 0;
  for (unsigned int j = 0; j < GPUGROUPSIZE; j++) {
    unsigned int bit = 1 << j;
    mask |= bit;
  }

  unsigned int local_roots = (unsigned int)-1;
  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    unsigned int temp_local = local >> (i * GPUGROUPSIZE);
    unsigned int temp = mask & temp_local;
    local_roots &= temp;
  }

  for (int i = 0; i < GPUGROUPSIZE; i++) {
    if (local_roots & (1 << i)) {
      for (unsigned int j = 0; j < GPUNUMGROUP; j++) {
        roots[j] = (int)(i + (j * GPUGROUPSIZE));
      }
      return;
    }
  }

  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    for (unsigned int j = 0; j < GPUGROUPSIZE; j++) {
      unsigned int bit = (i * GPUGROUPSIZE) + j;
      if (local & (1 << bit)) {
        roots[i] = (int)bit;
        break;
      }
    }
  }
}

/* add_to_trav (gpu_lc.cu:319-333) — NOTE: no bounds check on *size
 * (B-add-to-trav) */
static void add_to_trav(int root, int a, int b, unsigned int level,
                        unsigned int *size, trav_t *ds,
                        unsigned int *max_level) {
  if (a < 0 || b < 0) {
    return;
  }

  unsigned int index = (*size);
  *size = *size + 1;
  ds[index].a = a;
  ds[index].b = b;
  ds[index].root = root;
  ds[index].level = (int)level;

  *max_level = (*max_level < level) ? level : *max_level;
}

/* gpu_tree_reduction_rec (gpu_lc.cu:335-380) */
static int gpu_tree_reduction_rec(int root, unsigned int start,
                                  unsigned int stop, unsigned int mask,
                                  unsigned int level, unsigned int *list_size,
                                  trav_t *list, unsigned int *max_level) {
  int local_root = -1;
  int gpu_id[2] = {(int)start, (int)stop};

  if (stop - start > 1) {
    unsigned int middle = (1 + stop - start) / 2;
    gpu_id[0] = gpu_tree_reduction_rec(root, start, start + middle - 1, mask,
                                       level + 1, list_size, list, max_level);
    gpu_id[1] = gpu_tree_reduction_rec(root, start + middle, stop, mask,
                                       level + 1, list_size, list, max_level);
  }

  bool start_found = (gpu_id[0] >= 0) && ((mask & (1 << gpu_id[0])) != 0);
  bool stop_found = (gpu_id[1] >= 0) && ((mask & (1 << gpu_id[1])) != 0);

  if (start_found && stop_found) {
    if (root == gpu_id[0] || root == gpu_id[1]) {
      local_root = root;
    } else {
      local_root = gpu_id[0];
    }
  } else if (start_found && !stop_found) {
    gpu_id[1] = -1;
    local_root = gpu_id[0];
  } else if (!start_found && stop_found) {
    gpu_id[0] = -1;
    local_root = gpu_id[1];
  } else {
    gpu_id[1] = -1;
    gpu_id[0] = -1;
  }

  add_to_trav(local_root, gpu_id[0], gpu_id[1], level, list_size, list,
              max_level);
  return local_root;
}

/* gpu_tree_reduction_start (gpu_lc.cu:382-393) */
static void gpu_tree_reduction_start(unsigned int mask, unsigned int *list_size,
                                     trav_t *list, unsigned int *max_level) {
  int root[GPUNUMGROUP];
  find_roots(mask, root);
  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    gpu_tree_reduction_rec(root[i], i * GPUGROUPSIZE,
                           ((i + 1) * GPUGROUPSIZE) - 1, mask, 2, list_size,
                           list, max_level);
  }
  add_to_trav(root[0], root[0], root[1], 1, list_size, list, max_level);
}

/* ===================== test helpers ===================== */

static int g_fail = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL gpu_reduction_tree: " __VA_ARGS__);                \
      fprintf(stderr, "  (at %s:%d)\n", __FILE__, __LINE__);                   \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

/* -------- find_roots tests -------- */

static void test_find_roots_matching(void) {
  /* Bit 1 set in BOTH groups (gpu 1 in group0, gpu 5 in group1) -> matching
   * root column 1 -> roots = {1, 5}. */
  unsigned int local = (1u << 1) | (1u << 5);
  int roots[GPUNUMGROUP];
  find_roots(local, roots);
  CHECK(roots[0] == 1, "matching: roots[0]=%d expected 1\n", roots[0]);
  CHECK(roots[1] == 5, "matching: roots[1]=%d expected 5\n", roots[1]);
}

static void test_find_roots_matching_lowest(void) {
  /* Columns 0 and 2 both common (gpu {0,2} in group0, {4,6} in group1).
   * find_roots picks the LOWEST common column (0) -> roots {0,4}. */
  unsigned int local = (1u << 0) | (1u << 2) | (1u << 4) | (1u << 6);
  int roots[GPUNUMGROUP];
  find_roots(local, roots);
  CHECK(roots[0] == 0, "matching-lowest: roots[0]=%d expected 0\n", roots[0]);
  CHECK(roots[1] == 4, "matching-lowest: roots[1]=%d expected 4\n", roots[1]);
}

static void test_find_roots_no_match_fallback(void) {
  /* No common column: gpu 0 (group0 col0) and gpu 5 (group1 col1).
   * No matching root -> per-group lowest-set-bit fallback: roots {0,5}. */
  unsigned int local = (1u << 0) | (1u << 5);
  int roots[GPUNUMGROUP];
  find_roots(local, roots);
  CHECK(roots[0] == 0, "fallback: roots[0]=%d expected 0\n", roots[0]);
  CHECK(roots[1] == 5, "fallback: roots[1]=%d expected 5\n", roots[1]);
}

static void test_find_roots_empty_group(void) {
  /* group1 entirely empty (only gpu 2 present, in group0).
   * No matching root (group1 contributes 0 -> local_roots &= 0 -> 0).
   * Fallback: roots[0]=2 (lowest in group0), roots[1]=-1 (empty group). */
  unsigned int local = (1u << 2);
  int roots[GPUNUMGROUP];
  find_roots(local, roots);
  CHECK(roots[0] == 2, "empty-group: roots[0]=%d expected 2\n", roots[0]);
  CHECK(roots[1] == -1, "empty-group: roots[1]=%d expected -1\n", roots[1]);
}

static void test_find_roots_empty_mask(void) {
  /* No GPU present at all -> both roots -1. */
  int roots[GPUNUMGROUP];
  find_roots(0u, roots);
  CHECK(roots[0] == -1, "empty-mask: roots[0]=%d expected -1\n", roots[0]);
  CHECK(roots[1] == -1, "empty-mask: roots[1]=%d expected -1\n", roots[1]);
}

/* -------- gpu_tree_reduction_start tests -------- */

/* Helper: count trav nodes whose level == L. */
static unsigned int count_level(const trav_t *list, unsigned int n, int level) {
  unsigned int c = 0;
  for (unsigned int i = 0; i < n; i++)
    if (list[i].level == level)
      c++;
  return c;
}

static void test_tree_full_8gpu(void) {
  /* All 8 GPUs present.  Matching root col 0 -> roots {0,4}.
   * The two group subtrees build over [0..3] and [4..7] at level 2/3, then a
   * final level-1 merge node {root0=0, root1=4}.  Verify:
   *   - list_size <= 8 (the caller's fixed list[8]) — B-add-to-trav latent
   *     overflow stays UNREACHED under the shipped 4x2 model.
   *   - exactly one level-1 (group merge) node, and it merges roots 0 and 4.
   *   - max_level >= 2 (leaves built deeper than the merge). */
  unsigned int mask = 0xFFu; /* gpus 0..7 */
  unsigned int list_size = 0, max_level = 0;
  trav_t list[GPUNUMGROUP * GPUGROUPSIZE]; /* exactly 8, as in gpu_lc.cu:401 */
  memset(list, 0, sizeof(list));

  gpu_tree_reduction_start(mask, &list_size, list, &max_level);

  CHECK(list_size <= 8, "full8: list_size=%u exceeds list[8]!\n", list_size);
  CHECK(max_level >= 2, "full8: max_level=%u expected >=2\n", max_level);

  unsigned int merges = count_level(list, list_size, 1);
  CHECK(merges == 1, "full8: level-1 merge count=%u expected 1\n", merges);

  /* find the level-1 node and verify it merges the two group roots {0,4}. */
  int found = 0;
  for (unsigned int i = 0; i < list_size; i++) {
    if (list[i].level == 1) {
      found = 1;
      CHECK((list[i].a == 0 && list[i].b == 4),
            "full8: merge node a=%d b=%d expected {0,4}\n", list[i].a,
            list[i].b);
      CHECK(list[i].root == 0, "full8: merge root=%d expected 0\n",
            list[i].root);
    }
  }
  CHECK(found, "full8: no level-1 merge node found\n");
}

static void test_tree_sparse(void) {
  /* Only gpus 0 and 4 present (one per group, the roots themselves).
   * Each group subtree contributes intermediate trav nodes; the final merge
   * is {0,4} at level 1.  list_size must stay <= 8. */
  unsigned int mask = (1u << 0) | (1u << 4);
  unsigned int list_size = 0, max_level = 0;
  trav_t list[GPUNUMGROUP * GPUGROUPSIZE];
  memset(list, 0, sizeof(list));

  gpu_tree_reduction_start(mask, &list_size, list, &max_level);

  CHECK(list_size <= 8, "sparse: list_size=%u exceeds list[8]!\n", list_size);
  unsigned int merges = count_level(list, list_size, 1);
  CHECK(merges == 1, "sparse: level-1 merge count=%u expected 1\n", merges);
  for (unsigned int i = 0; i < list_size; i++) {
    if (list[i].level == 1) {
      CHECK(list[i].a == 0 && list[i].b == 4,
            "sparse: merge a=%d b=%d expected {0,4}\n", list[i].a, list[i].b);
    }
  }
}

static void test_tree_single_group(void) {
  /* Only group0 populated (gpus 0,1).  root1 == -1 so the final level-1
   * add_to_trav has b<0 and is skipped (early return a<0||b<0).  No merge
   * node should appear. */
  unsigned int mask = (1u << 0) | (1u << 1);
  unsigned int list_size = 0, max_level = 0;
  trav_t list[GPUNUMGROUP * GPUGROUPSIZE];
  memset(list, 0, sizeof(list));

  gpu_tree_reduction_start(mask, &list_size, list, &max_level);

  CHECK(list_size <= 8, "single-group: list_size=%u exceeds list[8]!\n",
        list_size);
  unsigned int merges = count_level(list, list_size, 1);
  CHECK(merges == 0,
        "single-group: level-1 merge count=%u expected 0 (root1=-1)\n", merges);
}

/* -------- add_to_trav overflow-guard documentation (B-add-to-trav) -------- */

static void test_add_to_trav_no_overflow_guard(void) {
  /* The census flags that add_to_trav increments *size with NO bound check.
   * Demonstrate the latent defect deterministically against an OVERSIZED
   * guarded buffer: 10 valid appends all succeed and write past index 7.
   * (We use a 16-entry buffer so the test itself never overflows; a real
   * caller passing list[8] would be silently corrupted.)  This is recorded
   * as exposes_runtime_bug for the missing guard.  The shipped tree builder
   * (tested above) never produces >8 nodes, so the bug is currently latent. */
  trav_t big[16];
  memset(big, 0, sizeof(big));
  unsigned int size = 0, max_level = 0;
  for (int k = 0; k < 10; k++) {
    add_to_trav(k, k, k + 1, 2, &size, big, &max_level);
  }
  /* No clamp exists: size advances to 10, writing indices 8 and 9 that a
   * list[8] caller would NOT own.  This asserts the (buggy) unguarded
   * behavior so a future guard addition flips this test RED on purpose. */
  CHECK(size == 10,
        "add_to_trav: size=%u — a guard would clamp at 8 (B-add-to-trav: "
        "currently UNGUARDED, latent overflow)\n",
        size);
}

int main(void) {
  test_find_roots_matching();
  test_find_roots_matching_lowest();
  test_find_roots_no_match_fallback();
  test_find_roots_empty_group();
  test_find_roots_empty_mask();
  test_tree_full_8gpu();
  test_tree_sparse();
  test_tree_single_group();
  test_add_to_trav_no_overflow_guard();

  if (g_fail) {
    fprintf(stderr, "FAIL gpu_reduction_tree\n");
    return 1;
  }
  printf("PASS gpu_reduction_tree (find_roots/tree-reduction/add_to_trav, "
         "4x2 model)\n");
  return 0;
}
