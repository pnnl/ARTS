/* SPDX-License-Identifier: Apache-2.0
 *
 * T274 — Pure GPU fit-scheme mask logic (host-extracted).
 *
 * first_fit / round_robin_fit / best_fit / worst_fit live in
 * libs/src/core/gpu/gpu_placement.cu (lines 108-191), a CUDA TU that cannot be
 * compiled standalone.  The integer "which GPU index is a candidate" logic is
 * pure; its only externals are arts_node_info.gpu (GPU count), a per-GPU
 * avail_global_mem table, try_reserve(), jrand48(), and arts_atomic_fetch_add.
 * The bodies
 * are copied VERBATIM below with those externals satisfied by host stubs that
 * RECORD which indices each scheme actually offered to try_reserve.  No runtime
 * source is modified.
 *
 * Target bug B-fit-mask-and (census Part 4 #2, severity HIGH), now FIXED:
 *   the candidacy test was `if (mask && check_mask)` — LOGICAL-AND of two
 *   nonzero values — instead of bitwise `mask & check_mask`.  With `&&`,
 *   whenever mask != 0 the per-GPU bit `check_mask` was never actually
 *   consulted, so EVERY GPU index became a candidate and the locality mask
 *   was ignored.  The runtime now uses bitwise `mask & check_mask`; the
 *   verbatim copies below are kept in sync with that fix.
 *
 * This test feeds a mask selecting only a SUBSET of GPUs and asserts the
 * CORRECT contract: only masked-in indices are offered to try_reserve.  With
 * the fixed bitwise `&`, only the masked-in indices are candidates, so the
 * assertion PASSES.  (Pre-fix `&&` offered ALL indices and this FAILED.)
 *
 * A second sub-test (round-robin rotation) verifies the rotating start index
 * advances across calls — that part is independent of the mask bug and passes.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---------- host stubs for the externals the fit bodies reference ----------
 */

/* Minimal mirror of the fields the fit bodies read. */
struct stub_node_info {
  unsigned int gpu; /* GPU count */
};
static struct stub_node_info arts_node_info;

struct stub_gpu {
  uint64_t avail_global_mem;
};
#define MAX_GPU 8
static struct stub_gpu arts_gpus[MAX_GPU];

/* jrand48 over a fixed buffer — the fit bodies only use it as a random start
 * offset.  We force it to 0 so the scan order is deterministic (index == i),
 * making "which indices were offered" reproducible. */
static unsigned short g_drand[3];
static long stub_jrand48(unsigned short xsubi[3]) {
  (void)xsubi;
  return 0; /* deterministic start offset */
}
#define jrand48(buf) stub_jrand48(buf)

/* arts_thread_info.drand_buf — only passed to jrand48 (which ignores it). */
static struct {
  unsigned short *drand_buf;
} arts_thread_info = {g_drand};

/* arts_atomic_fetch_add — single-threaded here; plain add is fine. */
static unsigned int g_rr_next; /* mirrors round_robin_fit's static next */
static unsigned int stub_fetch_add(volatile unsigned int *p, unsigned int v) {
  unsigned int old = *p;
  *p = old + v;
  return old;
}
#define arts_atomic_fetch_add(p, v) stub_fetch_add(p, v)
#define arts_atomic_add_u64(p, v) (*(p) += (v))

/* ---- recording try_reserve stub: logs every index it was offered ---- */
static int g_offered[MAX_GPU];
static unsigned int g_offered_n;
static int g_reserve_succeeds_at; /* index that "succeeds"; -1 = all fail */

static bool try_reserve(int gpu, uint64_t size, unsigned int threads) {
  (void)size;
  (void)threads;
  if (g_offered_n < MAX_GPU)
    g_offered[g_offered_n++] = gpu;
  if (g_reserve_succeeds_at < 0)
    return false; /* keep scanning -> records the full candidate set */
  return gpu == g_reserve_succeeds_at;
}

static void reset_log(void) {
  g_offered_n = 0;
  memset(g_offered, 0, sizeof(g_offered));
}

/* ===================== VERBATIM PURE COPIES (gpu_placement.cu) =============
 * Bodies copied unchanged from libs/src/core/gpu/gpu_placement.cu; ARTS_DEBUG
 * logging lines elided.  The candidacy test is the fixed bitwise
 * `if (mask & check_mask)` form (B-fit-mask-and resolved). */

/* first_fit (gpu_placement.cu:108-121) */
static int first_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int random = (int)jrand48(arts_thread_info.drand_buf);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + (unsigned int)random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask & check_mask) {
      if (try_reserve(index, size, total_threads)) {
        return index;
      }
    }
  }
  return -1;
}

/* round_robin_fit (gpu_placement.cu:123-137) — `static next` lifted to global
 * stub g_rr_next so the test can observe rotation. */
static int round_robin_fit(uint64_t mask, uint64_t size,
                           unsigned int total_threads) {
  unsigned int start = arts_atomic_fetch_add(&g_rr_next, 1U);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + start) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask & check_mask) {
      if (try_reserve(index, size, total_threads)) {
        return index;
      }
    }
  }
  return -1;
}

/* best_fit (gpu_placement.cu:139-164) */
static int best_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int selected_gpu = -1;
  uint64_t selected_gpu_avail_size = 0;
  int random = (int)jrand48(arts_thread_info.drand_buf);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + (unsigned int)random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask & check_mask) {
      if (selected_gpu != -1) {
        if (arts_gpus[index].avail_global_mem - size >
            selected_gpu_avail_size) {
          continue;
        }
      }
      if (try_reserve(index, size, total_threads)) {
        if (selected_gpu != -1) {
          arts_atomic_add_u64(&arts_gpus[selected_gpu].avail_global_mem, size);
        }
        selected_gpu = index;
        selected_gpu_avail_size = arts_gpus[index].avail_global_mem;
      }
    }
  }
  return selected_gpu;
}

/* worst_fit (gpu_placement.cu:166-191) */
static int worst_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int selected_gpu = -1;
  uint64_t selected_gpu_avail_size = 0;
  int random = (int)jrand48(arts_thread_info.drand_buf);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + (unsigned int)random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask & check_mask) {
      if (selected_gpu != -1) {
        if (arts_gpus[index].avail_global_mem - size <
            selected_gpu_avail_size) {
          continue;
        }
      }
      if (try_reserve(index, size, total_threads)) {
        if (selected_gpu != -1) {
          arts_atomic_add_u64(&arts_gpus[selected_gpu].avail_global_mem, size);
        }
        selected_gpu = index;
        selected_gpu_avail_size = arts_gpus[index].avail_global_mem;
      }
    }
  }
  return selected_gpu;
}

/* ===================== tests ===================== */

static int g_fail = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL gpu_fit_schemes: " __VA_ARGS__);                   \
      fprintf(stderr, "  (at %s:%d)\n", __FILE__, __LINE__);                   \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

static bool offered_contains(int idx) {
  for (unsigned int i = 0; i < g_offered_n; i++)
    if (g_offered[i] == idx)
      return true;
  return false;
}

/* The CORRECT contract: with a subset mask, only masked-in GPUs are candidates.
 * Returns true if the scheme respected the mask (no unmasked index offered). */
static bool scheme_respects_mask(int (*fit)(uint64_t, uint64_t, unsigned int),
                                 uint64_t mask) {
  reset_log();
  g_reserve_succeeds_at = -1; /* all fail -> scheme scans full candidate set */
  fit(mask, 1024, 1);
  for (unsigned int i = 0; i < g_offered_n; i++) {
    if (((mask >> g_offered[i]) & 1u) == 0) {
      return false; /* offered an UNMASKED GPU -> mask ignored */
    }
  }
  return true;
}

static void test_mask_respected_all_schemes(void) {
  arts_node_info.gpu = 4;
  for (int g = 0; g < MAX_GPU; g++)
    arts_gpus[g].avail_global_mem = 1u << 20;

  /* mask selects ONLY gpu 2.  A correct scheme offers ONLY index 2. */
  uint64_t mask = (uint64_t)1 << 2;

  struct {
    const char *name;
    int (*fn)(uint64_t, uint64_t, unsigned int);
  } schemes[] = {
      {"first_fit", first_fit},
      {"round_robin_fit", round_robin_fit},
      {"best_fit", best_fit},
      {"worst_fit", worst_fit},
  };

  for (unsigned int s = 0; s < 4; s++) {
    bool ok = scheme_respects_mask(schemes[s].fn, mask);
    /* CORRECT EXPECTATION: ok == true.  Fixed code uses bitwise `&` so only
     * the masked-in index is offered -> ok == true -> this PASSES.  (Pre-fix
     * `&&` offered indices 0,1,3 too -> ok == false -> FAILED.) */
    CHECK(ok,
          "%s ignored mask 0x%llx: offered %u indices incl. unmasked ones "
          "(B-fit-mask-and: `&&` should be `&`)\n",
          schemes[s].name, (unsigned long long)mask, g_offered_n);
  }
}

/* round-robin rotation: independent of the mask bug.  With a full mask, the
 * starting index advances by one each call (mod gpu count). */
static void test_round_robin_rotation(void) {
  arts_node_info.gpu = 4;
  g_rr_next = 0;
  uint64_t full = 0xFu; /* all 4 gpus */

  int first[4];
  for (int call = 0; call < 4; call++) {
    reset_log();
    g_reserve_succeeds_at = -1; /* fail all -> record scan order */
    round_robin_fit(full, 1024, 1);
    CHECK(g_offered_n == 4, "rr rotation: call %d offered %u expected 4\n",
          call, g_offered_n);
    first[call] = g_offered[0];
  }
  /* successive first-offered indices rotate: 0,1,2,3 */
  for (int call = 0; call < 4; call++) {
    CHECK(first[call] == call,
          "rr rotation: call %d first index=%d expected %d\n", call,
          first[call], call);
  }
}

/* sanity: with a full mask, first_fit returns a valid index (reservation
 * succeeds at the first scanned slot). */
static void test_first_fit_full_mask_picks(void) {
  arts_node_info.gpu = 4;
  reset_log();
  g_reserve_succeeds_at = 0;
  int r = first_fit(0xFu, 1024, 1);
  CHECK(r == 0, "first_fit full mask returned %d expected 0\n", r);
}

int main(void) {
  test_mask_respected_all_schemes();
  test_round_robin_rotation();
  test_first_fit_full_mask_picks();

  if (g_fail) {
    fprintf(stderr, "FAIL gpu_fit_schemes (B-fit-mask-and regressed: fit "
                    "schemes ignore the locality mask — `mask & check_mask`?)\n");
    return 1;
  }
  printf("PASS gpu_fit_schemes\n");
  return 0;
}
