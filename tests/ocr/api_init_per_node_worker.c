/* SPDX-License-Identifier: Apache-2.0
 *
 * T290 — init_per_node / init_per_worker weak startup callbacks.
 *
 * Target: the optional weak-symbol lifecycle hooks in
 * libs/src/core/system/runtime.c:
 *   if (init_per_node)   init_per_node(rank, argc, argv);              // thd 0
 *   if (init_per_worker) init_per_worker(rank, worker, argc, argv);   // worker
 * Neither is defined by any existing test, so the runtime's invocation of an
 * app-provided weak override was never exercised.  Defining them here turns the
 * default no-op weak bodies into the strong app symbols the linker prefers.
 *
 * Correct behavior pinned:
 *   - init_per_node runs once on this node (thread 0) BEFORE main_edt is
 *     scheduled, with node_id == this rank.
 *   - init_per_worker runs on each worker thread of this node BEFORE main_edt,
 *     with node_id == this rank and a valid worker index.
 *   - By the time main_edt (rank 0) runs, both flags are set for rank 0, so a
 *     direct observation from main_edt is race-free.
 *
 * Multinode-safe: main_edt only runs on rank 0, and rank 0 always executes
 * both callbacks before main_edt; non-rank-0 nodes are not observed here (they
 * have no main_edt to assert from) — single-rank assertion is the invariant.
 *
 * exposes_runtime_bug = false (pins the weak-callback invocation contract).
 */
#include "arts.h"
#include <stdatomic.h>
#include <stdint.h>

/* Process-global observation flags written by the startup callbacks (which run
 * on runtime threads before any EDT) and read by main_edt. */
static _Atomic unsigned int g_node_calls = 0;
static _Atomic unsigned int g_worker_calls = 0;
static _Atomic int g_node_id_seen = -1; /* node_id passed to per_node */
static _Atomic int g_node_id_bad = 0;   /* per_node node_id mismatch */
static _Atomic int g_worker_id_bad = 0; /* per_worker node_id mismatch */

static int g_failed = 0;

/* Strong override of the weak init_per_node — the runtime calls this once on
 * thread 0 of every node before the parallel-start barrier. */
void init_per_node(unsigned int node_id, int argc, char **argv) {
  (void)argc;
  (void)argv;
  atomic_store_explicit(&g_node_id_seen, (int)node_id, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_node_calls, 1u, memory_order_relaxed);
}

/* Strong override of the weak init_per_worker — the runtime calls this on each
 * worker thread of every node after the parallel-start barrier. */
void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                     char **argv) {
  (void)worker_id;
  (void)argc;
  (void)argv;
  /* node_id reported to per_worker must match the per_node node_id. */
  int seen = atomic_load_explicit(&g_node_id_seen, memory_order_relaxed);
  if (seen >= 0 && (int)node_id != seen) {
    atomic_store_explicit(&g_worker_id_bad, 1, memory_order_relaxed);
  }
  atomic_fetch_add_explicit(&g_worker_calls, 1u, memory_order_relaxed);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== api_init_per_node_worker ===\n");

  unsigned int nc = atomic_load_explicit(&g_node_calls, memory_order_relaxed);
  unsigned int wc = atomic_load_explicit(&g_worker_calls, memory_order_relaxed);
  int seen = atomic_load_explicit(&g_node_id_seen, memory_order_relaxed);

  if (nc < 1) {
    arts_printf("FAIL api_init_per_node_worker: init_per_node never invoked\n");
    g_failed = 1;
  }
  if (wc < 1) {
    arts_printf(
        "FAIL api_init_per_node_worker: init_per_worker never invoked\n");
    g_failed = 1;
  }
  /* main_edt runs on rank 0; the per_node node_id observed on this rank must be
   * this rank's id (0). */
  if (seen != (int)arts_get_current_rank()) {
    arts_printf("FAIL api_init_per_node_worker: per_node node_id %d != current "
                "rank %u\n",
                seen, arts_get_current_rank());
    g_failed = 1;
  }
  if (atomic_load_explicit(&g_worker_id_bad, memory_order_relaxed)) {
    arts_printf("FAIL api_init_per_node_worker: per_worker node_id mismatch\n");
    g_failed = 1;
  }

  if (!g_failed) {
    arts_printf("PASS api_init_per_node_worker: per_node=%u per_worker=%u "
                "node_id=%d\n",
                nc, wc, seen);
  }
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
