/* SPDX-License-Identifier: Apache-2.0
 *
 * scheduler_loop_variants — verifies the scheduler_loop[] dispatch table layout
 * and that the configured loop is the one the runtime actually installs
 * (census 17 §1 scheduler_loop[] + §4 LOW "scheduler dispatch index drift under
 * CXL").
 *
 * scheduler_loop[] (scheduler.c) is a function-pointer dispatch table indexed
 * by config->scheduler.  Its LAYOUT is build-config dependent and the
 * index→loop mapping is positional and undocumented at the config layer:
 *   - non-GPU, non-CXL: {default, network_before_steal, network_first}
 *   - non-GPU,  +CXL  : {default, CXL, network_before_steal, network_first}
 *                        ^ inserting CXL at index 1 SHIFTS the network loops:
 *                          network_before_steal 1→2, network_first 2→3.
 *   - GPU build       : {default, network_before_steal, network_first, gpu,
 *                        gpu_backoff, gpu_demand}.
 * A numeric `scheduler=N` in a config written for one build silently selects a
 * different loop under another (the documented foot-gun).
 *
 * Build-stable invariants this test asserts at RUNTIME (a test TU cannot see
 * the private per-library ARTS_USE_GPU / ARTS_USE_CXL macros, so it must not
 * hardcode the drifting indices — it checks only what is invariant across every
 * build):
 *   1. scheduler_loop[0] is ALWAYS arts_default_scheduler_loop (index 0 never
 *      shifts — CXL is inserted at 1, GPU loops appended after the CPU loops).
 *   2. The three CPU loop entry points (default / network_before_steal /
 *      network_first) are three DISTINCT function pointers — the table is a
 * real dispatch table, not aliased entries.
 *   3. The runtime installs the configured loop: arts_node_info.scheduler ==
 *      scheduler_loop[0] under the default config (configs/local/1n.cfg sets no
 *      `scheduler` key → it defaults to 0).  This proves index 0 selects the
 *      default loop in the running build, i.e. the positional contract holds
 * for the only index that does not drift.
 *
 * config_specific: runs meaningfully in EVERY build (no protocol/loop skip);
 * the checks are layout invariants, not behavior that varies by coherence
 * protocol. No spin, no watchdog — verification is synchronous in main_edt; a
 * hang is not a failure mode here (any assertion failure aborts immediately).
 */

#include <stdint.h>
#include <stdio.h>

#include "arts.h"
#include "arts/runtime.h"       /* scheduler_loop[], loop fn externs */
#include "arts/runtime_state.h" /* arts_node_info.scheduler */

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== scheduler_loop_variants ===\n");

  scheduler_t s_default = (scheduler_t)arts_default_scheduler_loop;
  scheduler_t s_nfirst = (scheduler_t)arts_network_first_scheduler_loop;
  scheduler_t s_nbefore = (scheduler_t)arts_network_before_steal_scheduler_loop;

  /* Invariant 1: index 0 is the default loop in every build. */
  if (scheduler_loop[0] != s_default) {
    (void)fprintf(
        stderr,
        "FAIL: scheduler_loop[0]=%p is not arts_default_scheduler_loop"
        " (%p) — index-0 invariant broken\n",
        (void *)scheduler_loop[0], (void *)s_default);
    arts_abort(1);
  }

  /* Invariant 2: the three CPU loops are distinct dispatch targets. */
  if (s_default == s_nfirst || s_default == s_nbefore ||
      s_nfirst == s_nbefore) {
    (void)fprintf(stderr,
                  "FAIL: CPU scheduler loops are not distinct "
                  "(default=%p network_first=%p network_before_steal=%p)\n",
                  (void *)s_default, (void *)s_nfirst, (void *)s_nbefore);
    arts_abort(1);
  }

  /* Invariant 3: the runtime installed the configured (default) loop.  The
   * stock config selects no scheduler index → it defaults to 0 → the active
   * loop must be scheduler_loop[0].  This proves the positional contract holds
   * for index 0 (the only index immune to the CXL/GPU drift). */
  if (arts_node_info.scheduler != scheduler_loop[0]) {
    (void)fprintf(stderr,
                  "FAIL: active scheduler %p != scheduler_loop[0] %p — "
                  "configured loop not installed\n",
                  (void *)arts_node_info.scheduler, (void *)scheduler_loop[0]);
    arts_abort(1);
  }
  if (arts_node_info.scheduler != s_default) {
    (void)fprintf(stderr,
                  "FAIL: active scheduler %p != arts_default_scheduler_loop %p "
                  "under default config\n",
                  (void *)arts_node_info.scheduler, (void *)s_default);
    arts_abort(1);
  }

  printf("scheduler_loop_variants: index-0=default, 3 distinct CPU loops, "
         "active loop = configured (index 0) — PASS\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
