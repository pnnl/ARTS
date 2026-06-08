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

/// @file finish_event_mn_termination.c
/// @brief Tests finish-event termination detection across nodes with result
///        verification: each task reports its rank, collector verifies
///        all tasks ran on the correct nodes.
///        Requires multi-node (node_count > 1).

#include "arts.h"

#define TASKS_PER_NODE 10
#define MAX_NODES 64

/// Task on each node: reports rank to collector.
void node_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t collector = (arts_guid_t)paramv[0];
  uint32_t slot = (uint32_t)paramv[1];
  unsigned int my_rank = arts_get_current_rank();
  arts_add_dependence((arts_guid_t)((uint64_t)my_rank), collector, slot,
                      DB_MODE_VAL);
}

/// Collector: verify that each rank appears TASKS_PER_NODE times.
void check_results(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  unsigned int total_nodes = (unsigned int)paramv[0];
  unsigned int counts[MAX_NODES] = {0};
  bool ok = true;

  for (uint32_t i = 0; i < depc; i++) {
    unsigned int rank = (unsigned int)(uint64_t)depv[i].guid;
    if (rank >= total_nodes) {
      arts_printf("  FAIL: task %u reported invalid rank %u\n", i, rank);
      ok = false;
    } else {
      counts[rank]++;
    }
  }

  for (unsigned int r = 0; r < total_nodes && ok; r++) {
    if (counts[r] != TASKS_PER_NODE) {
      arts_printf("  FAIL: rank %u ran %u tasks, expected %u\n", r, counts[r],
                  (unsigned int)TASKS_PER_NODE);
      ok = false;
    }
  }

  if (ok) {
    arts_printf("  PASS: all %u tasks ran across %u nodes (%u per node)\n",
                depc, total_nodes, (unsigned int)TASKS_PER_NODE);
  }
}

/// Finish EDT: finish scope completed across all nodes.
void scope_finish(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: multi-node finish scope completed\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== finish_event_mn_termination ===\n");

  unsigned int total = arts_get_total_ranks();

  // Create finish event + completion EDT.
  arts_guid_t fin =
      arts_edt_create(scope_finish, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, fin, 0, DB_MODE_NULL);

  // Create collector EDT that receives one signal per task.
  unsigned int total_tasks = TASKS_PER_NODE * total;
  uint64_t total_param = (uint64_t)total;
  arts_guid_t collector =
      arts_edt_create(check_results, 1, &total_param, total_tasks,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});

  // Launch TASKS_PER_NODE tasks on each node.
  uint32_t slot = 0;
  for (unsigned int r = 0; r < total; r++) {
    for (unsigned int i = 0; i < TASKS_PER_NODE; i++) {
      uint64_t params[2];
      params[0] = (uint64_t)collector;
      params[1] = (uint64_t)slot;
      arts_edt_create(node_task, 2, params, 0,
                      &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      slot++;
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
