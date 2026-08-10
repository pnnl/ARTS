/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file runtime_barrier_counts.c
/// @brief Asserts the five startup/teardown barrier counters
///        (ready_to_push / ready_to_parallel_start / ready_to_inspect /
///         ready_to_execute / ready_to_clean) were each decremented exactly
///        total_thread_count times during bring-up.
///
/// Each barrier counter is initialised to total_thread_count in
/// arts_runtime_node_init and every thread (thread-0 in
/// arts_thread_zero_node_start, non-zero threads in arts_runtime_private_init)
/// decrements it exactly once then spins until it reaches zero.  An off-by-one
/// (double-decrement or a skipped decrement) would either hang startup or leave
/// a barrier counter non-zero.  Once main_edt runs, all four startup barriers
/// (push/parallel_start/inspect/execute) have necessarily reached zero
/// (main_edt is scheduled only after the inspect+execute rendezvous), so this
/// test asserts they read zero exactly.  ready_to_clean is the teardown barrier
/// and is still at its initial value (total_thread_count) at main_edt time,
/// which we also assert to pin the init-count invariant.
///
/// White-box: reads arts_node_info via runtime_state.h, mirroring
/// tests/unit/route_table_install_race.c's internal-state inspection idiom.
///
/// runtime_single, all configs (protocol-agnostic).

#include "arts.h"
#include "arts/runtime_state.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== runtime_barrier_counts ===\n");

  unsigned int tc = arts_node_info.total_thread_count;
  unsigned int push = arts_node_info.ready_to_push;
  unsigned int par = arts_node_info.ready_to_parallel_start;
  unsigned int insp = arts_node_info.ready_to_inspect;
  unsigned int exec = arts_node_info.ready_to_execute;
  unsigned int clean = arts_node_info.ready_to_clean;

  bool ok = true;

  /* All four startup barriers must have drained to zero by the time main_edt
   * runs (it is scheduled only after the execute rendezvous). */
  if (push != 0) {
    arts_printf("  FAIL: ready_to_push=%u (expected 0)\n", push);
    ok = false;
  }
  if (par != 0) {
    arts_printf("  FAIL: ready_to_parallel_start=%u (expected 0)\n", par);
    ok = false;
  }
  if (insp != 0) {
    arts_printf("  FAIL: ready_to_inspect=%u (expected 0)\n", insp);
    ok = false;
  }
  if (exec != 0) {
    arts_printf("  FAIL: ready_to_execute=%u (expected 0)\n", exec);
    ok = false;
  }

  /* ready_to_clean is the teardown barrier; at main_edt time no thread has
   * reached private_cleanup yet, so it must still equal its init value tc.
   * This pins the "init = total_thread_count" invariant for the count. */
  if (tc == 0 || clean != tc) {
    arts_printf("  FAIL: ready_to_clean=%u (expected tc=%u)\n", clean, tc);
    ok = false;
  }

  if (ok) {
    arts_printf("PASS runtime_barrier_counts: tc=%u all startup barriers 0, "
                "clean=%u\n",
                tc, clean);
  }

  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
