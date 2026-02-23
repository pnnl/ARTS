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

/// @file multinode_db.c
/// @brief Tests DB operations across nodes: create on one node, put/get from
///        another. Requires multi-node (node_count > 1).

#include "arts.h"
#include <string.h>

/// Verify data received via cross-node get.
void check_cross_get(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL);
  if (ok) {
    for (int i = 0; i < 8 && ok; i++) {
      if (data[i] != i * 100) {
        ok = false;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: cross-node DB get correct\n");
  } else {
    arts_printf("  FAIL: cross-node DB get data mismatch\n");
  }
}

/// Remote writer task: creates a DB on its node and puts data into it.
void remote_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t db_guid = (arts_guid_t)paramv[0];
  arts_guid_t reader_edt = (arts_guid_t)paramv[1];

  // Write data into the pre-allocated DB.
  int data[8];
  for (int i = 0; i < 8; i++) {
    data[i] = i * 100;
  }
  arts_put_in_db(data, reader_edt, db_guid, 0, 0, 8 * sizeof(int));
}

/// Verify round-robin GUIDs across nodes.
void check_round_robin(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  unsigned int total_nodes = (unsigned int)paramv[0];
  unsigned int count = (unsigned int)paramv[1];
  arts_guid_t *guids = (arts_guid_t *)depv[0].ptr;
  bool ok = (guids != NULL);
  if (ok) {
    for (unsigned int i = 0; i < count && ok; i++) {
      unsigned int expected_rank = i % total_nodes;
      unsigned int actual_rank = arts_guid_get_rank(guids[i]);
      if (actual_rank != expected_rank) {
        arts_printf("  FAIL: round-robin guid[%u] rank=%u expected=%u\n", i,
                    actual_rank, expected_rank);
        ok = false;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: round-robin %u GUIDs distributed correctly\n", count);
  } else if (guids == NULL) {
    arts_printf("  FAIL: round-robin null pointer\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== multinode_db ===\n");

  unsigned int total = arts_get_total_nodes();
  if (total < 2) {
    arts_printf("  SKIP: need node_count >= 2 (have %u)\n", total);
    arts_shutdown();
    return;
  }

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: Create DB on node 0, have node 1 write data, then read on node 0.
  arts_guid_t db = arts_db_create_remote(0, 8 * sizeof(int));
  arts_guid_t reader = arts_edt_create_with_epoch(
      check_cross_get, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  uint64_t params[2];
  params[0] = (uint64_t)db;
  params[1] = (uint64_t)reader;
  arts_edt_create_with_epoch(remote_writer, 2, params, 0, epoch,
                             &(arts_hint_t){.route = 1});

  // Test 2: arts_guid_reserve_round_robin across nodes.
  unsigned int count = total * 4;
  arts_guid_t *rr_guids = arts_guid_reserve_round_robin(count, ARTS_DB);

  // Pass GUIDs via ptr signal.
  uint64_t rr_params[2];
  rr_params[0] = (uint64_t)total;
  rr_params[1] = (uint64_t)count;
  arts_guid_t rr_checker = arts_edt_create_with_epoch(
      check_round_robin, 2, rr_params, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt_ptr(rr_checker, 0, rr_guids, count * sizeof(arts_guid_t));

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
