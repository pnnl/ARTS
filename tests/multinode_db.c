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
  arts_db_put(data, reader_edt, db_guid, 0, 0, 8 * sizeof(int), NULL);
}

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_shutdown();
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
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== multinode_db ===\n");

  unsigned int total = arts_get_total_ranks();
  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), shut, 0);
  arts_epoch_start(epoch);

  // Test 1: Create DB on node 0, have node 1 write data, then read on node 0.
  void *db_ptr;
  arts_guid_t db =
      arts_db_create(&db_ptr, 8 * sizeof(int), ARTS_DB_DEFAULT,
                     ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
  arts_guid_t reader =
      arts_edt_create(check_cross_get, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  uint64_t params[2];
  params[0] = (uint64_t)db;
  params[1] = (uint64_t)reader;
  arts_edt_create(remote_writer, 2, params, 0,
                  &(arts_edt_hint_t){.rank = 1, .epoch = epoch});

  // Test 2: arts_guid_reserve_range with ARTS_HINT_ROUND_ROBIN across
  // nodes.  Stash the GUID array in a DB so the checker EDT receives it
  // via dependency wiring (cross-node ptr delivery is not a thing).
  unsigned int count = total * 4;
  arts_guid_t rr_range =
      arts_guid_reserve_range(ARTS_DB, count, ARTS_HINT_ROUND_ROBIN);

  uint64_t rr_params[2];
  rr_params[0] = (uint64_t)total;
  rr_params[1] = (uint64_t)count;
  arts_guid_t rr_checker =
      arts_edt_create(check_round_robin, 2, rr_params, 1,
                      &(arts_edt_hint_t){.rank = 0, .epoch = epoch});

  arts_guid_t *rr_db_ptr = NULL;
  arts_guid_t rr_db =
      arts_db_create((void **)&rr_db_ptr, count * sizeof(arts_guid_t),
                     ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  for (unsigned int i = 0; i < count; i++) {
    rr_db_ptr[i] = arts_guid_from_index(rr_range, i);
  }
  arts_db_release(rr_db);
  arts_add_dependence(rr_db, rr_checker, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
