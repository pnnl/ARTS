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

/// @file system_info_per_rank.c
/// @brief Multinode check of the runtime-introspection API in
///        libs/src/core/utils/system_info.c.  An EDT pinned to each rank N
///        asserts:
///          - arts_get_current_rank()  == N            (the rank it ran on)
///          - arts_get_total_ranks()   == total        (same on every rank)
///          - arts_get_total_workers() == workers_per_rank * total_ranks
///        The last identity is exactly what arts_get_total_workers() computes,
///        and it relies on the homogeneous-worker-count assumption that the
///        local configs satisfy.
///
/// Each per-rank verifier joins one finish event owned by rank 0; rank 0 waits
/// on it and shuts the runtime down.  Requires node_count > 1; on a single
/// node it prints SKIP and exits cleanly (so it is harmless when the runner
/// lands a 1n config on the binary).

#include <stdint.h>

#include "arts.h"

/// Runs on a specific rank: paramv[0] = expected rank, paramv[1] = total ranks.
void rank_verifier(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int expect_rank = (unsigned int)paramv[0];
  unsigned int expect_total = (unsigned int)paramv[1];

  unsigned int rank = arts_get_current_rank();
  unsigned int total = arts_get_total_ranks();
  unsigned int per_rank = arts_get_workers_per_rank();
  unsigned int total_workers = arts_get_total_workers();

  bool ok = true;
  if (rank != expect_rank) {
    arts_printf("  FAIL: rank %u ran on rank %u\n", expect_rank, rank);
    ok = false;
  }
  if (total != expect_total) {
    arts_printf("  FAIL: rank %u sees total_ranks=%u (expected %u)\n",
                expect_rank, total, expect_total);
    ok = false;
  }
  if (total_workers != per_rank * total) {
    arts_printf("  FAIL: rank %u total_workers=%u != per_rank(%u)*ranks(%u)\n",
                expect_rank, total_workers, per_rank, total);
    ok = false;
  }
  if (ok) {
    arts_printf("  PASS: rank %u: total_ranks=%u workers_per_rank=%u "
                "total_workers=%u\n",
                rank, total, per_rank, total_workers);
  }
}

void finalize_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS system_info_per_rank\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== system_info_per_rank ===\n");

  unsigned int total = arts_get_total_ranks();
  if (total < 2) {
    arts_printf("SKIP system_info_per_rank: needs node_count > 1 (have %u)\n",
                total);
    arts_shutdown();
    return;
  }

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (unsigned int r = 0; r < total; r++) {
    uint64_t pv[2] = {(uint64_t)r, (uint64_t)total};
    arts_edt_create(rank_verifier, 2, pv, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = fe});
  }

  /* All verifiers done -> announce success and shut down. */
  arts_event_wait(fe);
  arts_edt_create(finalize_edt, 0, NULL, 0, NULL);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
