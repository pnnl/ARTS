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

/// @file multinode_edt.c
/// @brief Tests EDT creation and signaling across nodes.
///        Requires multi-node (node_count > 1).

#include "arts.h"

/// EDT running on remote node, signals master on slot 0.
void remote_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t collector = (arts_guid_t)paramv[0];
  unsigned int my_rank = arts_get_current_node();
  arts_signal_edt_value(collector, 0, (uint64_t)my_rank);
}

/// Collector on node 0: verify the value came from the remote node.
void check_remote_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t remote_rank = (uint64_t)depv[0].guid;
  unsigned int expected = (unsigned int)paramv[0];
  bool ok = ((unsigned int)remote_rank == expected);
  if (ok) {
    arts_printf("  PASS: remote EDT ran on rank %u\n",
                (unsigned int)remote_rank);
  } else {
    arts_printf("  FAIL: expected rank %u got %lu\n", expected,
                (unsigned long)remote_rank);
  }
}

/// EDT on all nodes: each signals the collector with its rank.
void all_nodes_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t collector = (arts_guid_t)paramv[0];
  uint32_t slot = (uint32_t)paramv[1];
  unsigned int my_rank = arts_get_current_node();
  arts_signal_edt_value(collector, slot, (uint64_t)my_rank);
}

void check_all_nodes(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  bool ok = true;
  // Each slot should have a unique rank.
  for (uint32_t i = 0; i < depc; i++) {
    uint64_t rank = (uint64_t)depv[i].guid;
    if (rank != (uint64_t)i) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: EDTs ran on all %u nodes\n", depc);
  } else {
    arts_printf("  FAIL: EDT distribution incorrect\n");
  }
  arts_shutdown();
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== multinode_edt ===\n");

  unsigned int total = arts_get_total_nodes();
  if (total < 2) {
    arts_printf("  SKIP: need node_count >= 2 (have %u)\n", total);
    arts_shutdown();
    return;
  }

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: Create EDT on remote node 1, have it signal back.
  uint64_t expected_param = 1;
  arts_guid_t checker =
      arts_edt_create_with_epoch(check_remote_edt, 1, &expected_param, 1, epoch,
                                 &(arts_hint_t){.route = 0});
  uint64_t coll_param = (uint64_t)checker;
  arts_edt_create_with_epoch(remote_task, 1, &coll_param, 0, epoch,
                             &(arts_hint_t){.route = 1});

  // Test 2: Create one EDT per node, each reports its rank.
  arts_guid_t all_coll = arts_edt_create_with_epoch(
      check_all_nodes, 0, NULL, total, epoch, &(arts_hint_t){.route = 0});
  for (unsigned int r = 0; r < total; r++) {
    uint64_t params[2];
    params[0] = (uint64_t)all_coll;
    params[1] = (uint64_t)r;
    arts_edt_create_with_epoch(all_nodes_task, 2, params, 0, epoch,
                               &(arts_hint_t){.route = r});
  }

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
