/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
#include <stdlib.h>

#include "arts.h"

/* Test: rank 1 prints a sentinel via arts_printf. Non-master stdout is
 * unconditionally forwarded to master's stdout under the local/SSH
 * launcher, so the CTest-captured output should contain:
 *   [1] MULTINODE_STDOUT_FORWARD_SENTINEL 42 */

void remote_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_printf("MULTINODE_STDOUT_FORWARD_SENTINEL %lu\n", paramv[0]);
  arts_shutdown();
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                     char **argv) {
  (void)argc;
  (void)argv;
  /* Only spawn from rank 0 worker 0 to avoid 2*2 spawns in 2-node. */
  if (node_id != 0 || worker_id != 0) {
    return;
  }
  /* Spawn the sentinel EDT on rank 1 if nodes >= 2, else rank 0. */
  unsigned int target = (arts_get_total_ranks() > 1) ? 1u : 0u;
  uint64_t paramv[1] = {42};
  arts_edt_create(remote_edt, 1, paramv, 0, &(arts_edt_hint_t){.rank = target});
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
