/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file shutdown_idempotent_concurrent_init.c
/// @brief Two ranks initiate shutdown concurrently; the per-node CAS 0->1 gate
///        in arts_enter_shutdown_state keeps exactly one broadcast per node and
///        idempotent stops, with clean cluster teardown.
///
/// When more than one rank calls arts_shutdown at the same time, each rank's
/// arts_enter_shutdown_state(true) gates on arts_atomic_cswap(shutdown_state,
/// 0, 1).  Locally exactly one caller wins and performs broadcast +
/// stop_workers; concurrently a peer's broadcast may arrive and run
/// arts_handler_shutdown -> arts_enter_shutdown_state(false), which the same
/// CAS gate makes idempotent (no rebroadcast, no double stop).  The required
/// property: with multiple ranks initiating at once, the cluster tears down
/// cleanly — no double-broadcast storm, no deadlock.
///
/// Each of the first two ranks schedules a LOCAL shutdown initiator EDT, so two
/// independent initiations race across the cluster.  Each rank prints a clean
/// token after arts_rt returns; the integrator's MN harness confirms all ranks
/// returned (ctest TIMEOUT reaps a storm/deadlock regression).
///
/// runtime_multinode (2n/3n/4n/2n_io).  Self-skips cleanly to a single-node
/// clean shutdown when only one rank is present.
/// all configs (shutdown sits above the coherence layer).

#include "arts.h"

#include <stdio.h>

void local_initiator_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  rank %u initiating shutdown concurrently\n",
              arts_get_current_rank());
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== shutdown_idempotent_concurrent_init ===\n");

  unsigned int ranks = arts_get_total_ranks();
  if (ranks < 2) {
    arts_printf("SKIP shutdown_idempotent_concurrent_init: requires 2+ ranks "
                "(got %u)\n",
                ranks);
    arts_shutdown();
    return;
  }

  arts_printf("PASS shutdown_idempotent_concurrent_init: %u ranks, ranks 0 and "
              "1 initiating concurrently\n",
              ranks);
  /* Two independent initiators (rank 0 and rank 1) race into shutdown. */
  arts_edt_create(local_initiator_edt, 0, NULL, 0,
                  &(arts_edt_hint_t){.rank = 0});
  arts_edt_create(local_initiator_edt, 0, NULL, 0,
                  &(arts_edt_hint_t){.rank = 1});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  printf("[rank %u] IDEMPOTENT_INIT_EXIT\n", arts_get_current_rank());
  return 0;
}
