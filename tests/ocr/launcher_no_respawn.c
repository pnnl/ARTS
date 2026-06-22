/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/

/// @file launcher_no_respawn.c
/// @brief Validates the ARTS_RANK recursive-spawn guard of the launcher.
///
/// The local launcher spawns exactly `table_length - 1` non-master ranks, and
/// every spawned child receives ARTS_RANK in its environment.  That env var is
/// what `arts_transport_setup` keys off to short-circuit rank discovery and,
/// crucially, prevent a spawned child from itself running the launcher and
/// re-spawning the whole cluster (which would be an exponential fork storm).
///
/// If the guard were broken, the cluster would contain many MORE ranks than the
/// config requests (each child re-launching N more children).  This test pins
/// down the contract by asserting:
///   1. arts_get_total_ranks() == the configured rank count (no extra ranks).
///   2. Every non-master rank reports ARTS_RANK set and equal to its own rank,
///      and a single ancestor master (no recursive launcher invocation).
///
/// A finish event per rank gates main_edt so the test completes
/// deterministically; a stranded check is caught by the ctest TIMEOUT (no
/// in-test spin/watchdog). Protocol-agnostic: the launcher carries no coherence
/// state, runs in all configs.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/// Per-rank probe: verify this rank's identity against ARTS_RANK and confirm it
/// did not itself launch a cluster.  Runs in the target rank's own process.
void rank_probe_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int expected_rank = (unsigned int)paramv[0];
  unsigned int total = (unsigned int)paramv[1];

  unsigned int me = arts_get_current_rank();
  unsigned int seen_total = arts_get_total_ranks();

  if (me != expected_rank) {
    arts_printf("FAIL: rank probe ran on %u, expected %u\n", me, expected_rank);
    arts_abort(1);
    return;
  }
  if (seen_total != total) {
    arts_printf("FAIL: rank %u sees total_ranks %u, expected %u (respawn?)\n",
                me, seen_total, total);
    arts_abort(1);
    return;
  }

  /* The master (rank 0) has no ARTS_RANK in its env; every spawned child must
   * have it set to its own rank.  A missing/wrong value on a non-master rank
   * means the recursive-spawn guard was not applied. */
  const char *rank_env = getenv("ARTS_RANK");
  if (me != 0) {
    if (rank_env == NULL) {
      arts_printf("FAIL: rank %u has no ARTS_RANK in env\n", me);
      arts_abort(1);
      return;
    }
    unsigned int env_rank = (unsigned int)strtoul(rank_env, NULL, 10);
    if (env_rank != me) {
      arts_printf("FAIL: rank %u has ARTS_RANK=%s (mismatch)\n", me, rank_env);
      arts_abort(1);
      return;
    }
  }

  arts_printf("  PASS: rank %u identity ok (total=%u, ARTS_RANK=%s)\n", me,
              seen_total, rank_env ? rank_env : "(unset/master)");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== launcher_no_respawn ===\n");

  unsigned int total = arts_get_total_ranks();

  /* Fan a probe onto every rank (including the master) and wait on each so the
   * test ends deterministically once all ranks have reported. */
  for (unsigned int r = 0; r < total; r++) {
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t params[2] = {(uint64_t)r, (uint64_t)total};
    arts_edt_create(rank_probe_edt, 2, params, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_event_wait(fe);
  }

  arts_printf("PASS: launcher_no_respawn (%u ranks, no recursive spawn)\n",
              total);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
