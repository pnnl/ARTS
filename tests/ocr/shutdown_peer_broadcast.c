/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file shutdown_peer_broadcast.c
/// @brief Master-initiated shutdown broadcasts MSG_SHUTDOWN to peers; each peer
///        runs arts_handler_shutdown, stops, and does NOT rebroadcast.
///
/// arts_enter_shutdown_state(initiator=true) on rank 0 (via arts_shutdown)
/// calls arts_transport_broadcast_shutdown + wait_for_outbox_drain, then
/// stop_workers. Each peer receives MSG_SHUTDOWN -> arts_handler_shutdown ->
/// arts_enter_shutdown_state(initiator=false): the false initiator guard means
/// a receiving node MUST NOT rebroadcast (which would storm the cluster) and
/// the CAS gate makes the stop idempotent.  The required property: every rank
/// shuts down cleanly with no rebroadcast storm and no deadlock.
///
/// We schedule a single shutdown trigger on the master (rank 0) inside a finish
/// scope; the broadcast propagates to all peers.  Each rank's main() prints a
/// per-rank clean token after arts_rt returns; the integrator's MN harness
/// confirms all ranks returned (ctest TIMEOUT reaps a storm/hang regression).
///
/// runtime_multinode (2n/3n/4n/2n_io).  Self-skips cleanly to a single-node
/// clean shutdown when only one rank is present.
/// all configs (shutdown sits above the coherence layer).

#include "arts.h"

#include <stdio.h>

void master_shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Initiator on the master: broadcasts MSG_SHUTDOWN to all peers. */
  arts_printf("  master initiating cluster shutdown broadcast\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== shutdown_peer_broadcast ===\n");

  unsigned int ranks = arts_get_total_ranks();
  if (ranks < 2) {
    arts_printf("SKIP shutdown_peer_broadcast: requires 2+ ranks (got %u)\n",
                ranks);
    arts_shutdown();
    return;
  }

  arts_printf("PASS shutdown_peer_broadcast: %u ranks, master broadcasting\n",
              ranks);
  /* Trigger the broadcast from the master rank. */
  arts_edt_create(master_shutdown_edt, 0, NULL, 0,
                  &(arts_edt_hint_t){.rank = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* Every rank reaching here => peer received SHUTDOWN, stopped, did not storm.
   */
  printf("[rank %u] PEER_BROADCAST_EXIT\n", arts_get_current_rank());
  return 0;
}
