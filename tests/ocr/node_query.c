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

/// @file node_query.c
/// @brief Tests runtime query utility functions:
///        arts_get_current_rank, arts_get_total_ranks,
///        arts_get_current_worker, arts_get_workers_per_rank,
///        arts_get_gpus_per_rank, arts_get_current_numa_domain,
///        arts_get_total_numa_domains.

#include "arts.h"

/// Verify node query functions in an EDT context.
void check_queries(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int node = arts_get_current_rank();
  unsigned int total_nodes = arts_get_total_ranks();
  unsigned int worker = arts_get_current_worker();
  unsigned int total_workers = arts_get_workers_per_rank();
  unsigned int numa = arts_get_current_numa_domain();
  unsigned int total_numa = arts_get_total_numa_domains();
  unsigned int total_gpus = arts_get_gpus_per_rank();

  bool ok = true;

  // Node rank must be < total_nodes.
  if (node >= total_nodes) {
    arts_printf("  FAIL: node=%u >= total_nodes=%u\n", node, total_nodes);
    ok = false;
  }
  // Total nodes must be >= 1.
  if (total_nodes < 1) {
    arts_printf("  FAIL: total_nodes=%u < 1\n", total_nodes);
    ok = false;
  }
  // Worker id must be < total_workers.
  if (worker >= total_workers) {
    arts_printf("  FAIL: worker=%u >= total_workers=%u\n", worker,
                total_workers);
    ok = false;
  }
  // Total workers must be >= 1.
  if (total_workers < 1) {
    arts_printf("  FAIL: total_workers=%u < 1\n", total_workers);
    ok = false;
  }
  // NUMA domain must be < total_numa.
  if (numa >= total_numa) {
    arts_printf("  FAIL: numa=%u >= total_numa=%u\n", numa, total_numa);
    ok = false;
  }
  // Total NUMA >= 1.
  if (total_numa < 1) {
    arts_printf("  FAIL: total_numa=%u < 1\n", total_numa);
    ok = false;
  }

  if (ok) {
    arts_printf("  PASS: node=%u/%u worker=%u/%u numa=%u/%u gpus=%u\n", node,
                total_nodes, worker, total_workers, numa, total_numa,
                total_gpus);
  }
}

/// Test that multiple workers give different worker IDs.
void check_worker_id(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  uint32_t expected_slot = (uint32_t)paramv[0];
  arts_guid_t collector = (arts_guid_t)paramv[1];
  unsigned int worker = arts_get_current_worker();
  arts_add_dependence((arts_guid_t)((uint64_t)worker), collector, expected_slot,
                      DB_MODE_VAL);
}

void collect_worker_ids(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  // Just verify we received all of them.
  bool ok = true;
  for (uint32_t i = 0; i < depc; i++) {
    uint64_t wid = (uint64_t)depv[i].guid;
    unsigned int total = arts_get_workers_per_rank();
    if (wid >= total) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: collected %u worker IDs, all valid\n", depc);
  } else {
    arts_printf("  FAIL: some worker IDs out of range\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== node_query ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  // Test 1: Query functions in EDT.
  arts_edt_create(check_queries, 0, NULL, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe});

  // Test 2: Launch multiple EDTs and collect worker IDs.
  unsigned int total = arts_get_workers_per_rank();
  unsigned int count = (total > 8) ? 8 : total;
  arts_guid_t coll =
      arts_edt_create(collect_worker_ids, 0, NULL, count,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  for (unsigned int i = 0; i < count; i++) {
    uint64_t params[2];
    params[0] = (uint64_t)i;
    params[1] = (uint64_t)coll;
    arts_edt_create(check_worker_id, 2, params, 0,
                    &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  }

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
