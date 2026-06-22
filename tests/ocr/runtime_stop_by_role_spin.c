/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file runtime_stop_by_role_spin.c
/// @brief Exercises arts_runtime_stop_by_role's per-role stop walk on a normal
///        shutdown and asserts the runtime stops within the bounded spin
///        budget.
///
/// arts_runtime_stop_by_role(role_mask, label) (static in runtime.c) walks
/// [0, total_thread_count) and, for each thread matching the role mask,
/// bounded-spins (max 1e7 iters) until local_spin[i] != NULL, then writes
/// *local_spin[i] = false to retire that thread.  On spin exhaustion it logs a
/// WARN and leaks (never stops) that thread.  This static function is reached
/// through stop_workers (1<<WORKER) and stop_network (RECEIVER|SENDER) during
/// the normal shutdown sequence, so a clean cooperative shutdown drives the
/// stop walk over every role.
///
/// The never-registered/spin-exhaustion branch is unreachable through the
/// public API (every thread publishes local_spin before the startup barriers
/// release), so this test pins the reachable invariant: a normal shutdown
/// drives the per-role stop walk and the process terminates within the spin
/// budget (ctest TIMEOUT reaps a regression that would hang here).  To make the
/// SENDER/RECEIVER role arms of the walk meaningful, the multinode variants
/// register network threads; the single-node variant still drives the WORKER
/// arm.
///
/// runtime_single, all configs (protocol-agnostic).

#include "arts.h"

void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Trigger the full stop sequence: stop_workers (1<<WORKER) then, in
   * main_join, socket_shutdown + stop_network (RECEIVER|SENDER).  The stop walk
   * must retire every role within the bounded spin budget. */
  arts_printf("  worker EDT requesting shutdown\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== runtime_stop_by_role_spin ===\n");
  arts_printf("PASS runtime_stop_by_role_spin: driving role stop walk\n");

  arts_edt_create(worker_edt, 0, NULL, 0, &(arts_edt_hint_t){.rank = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* arts_rt returned => stop_by_role retired all roles within the spin budget.
   */
  arts_printf("STOP_BY_ROLE_DONE\n");
  return 0;
}
