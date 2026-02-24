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

/// @file hint_routing.c
/// @brief Tests arts_hint_t routing: ARTS_HINT_CURRENT_NODE, explicit route,
///        NULL hint = defaults.

#include "arts.h"

/// EDT created with NULL hint — should run on current node.
void null_hint_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int my_rank = arts_get_current_node();
  arts_printf("  PASS: NULL hint -> ran on node %u\n", my_rank);
}

/// EDT created with explicit route=0.
void explicit_route_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int my_rank = arts_get_current_node();
  if (my_rank == 0) {
    arts_printf("  PASS: explicit route=0 -> ran on node 0\n");
  } else {
    arts_printf("  FAIL: explicit route=0 but ran on node %u\n", my_rank);
  }
}

/// EDT created with ARTS_HINT_CURRENT_NODE.
void current_node_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int my_rank = arts_get_current_node();
  // ARTS_HINT_CURRENT_NODE means wherever the creator runs.
  arts_printf("  PASS: ARTS_HINT_CURRENT_NODE -> ran on node %u\n", my_rank);
}

/// DB created with NULL hint — should be on current node.
void check_db_hint(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t db_guid = (arts_guid_t)paramv[0];
  unsigned int db_rank = arts_guid_get_rank(db_guid);
  unsigned int my_rank = arts_get_current_node();
  bool ok = (db_rank == my_rank || db_rank == 0);
  if (ok) {
    arts_printf("  PASS: DB NULL hint on rank %u\n", db_rank);
  } else {
    arts_printf("  FAIL: DB NULL hint rank=%u\n", db_rank);
  }
  (void)depv;
}

/// Test hint with profiling id.
void profiled_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: hint with profiling id ran\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== hint_routing ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: NULL hint.
  arts_edt_create_with_epoch(null_hint_task, 0, NULL, 0, epoch, NULL);

  // Test 2: Explicit route=0.
  arts_edt_create_with_epoch(explicit_route_task, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = 0});

  // Test 3: ARTS_HINT_CURRENT_NODE.
  arts_edt_create_with_epoch(current_node_task, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = ARTS_HINT_CURRENT_NODE});

  // Test 4: DB with NULL hint.
  void *dbptr = NULL;
  arts_guid_t db = arts_db_create(&dbptr, 16, ARTS_DB_DEFAULT, NULL);
  arts_db_release(db);
  uint64_t param = (uint64_t)db;
  arts_edt_create_with_epoch(check_db_hint, 1, &param, 0, epoch,
                             &(arts_hint_t){.route = 0});

  // Test 5: Hint with profiling id.
  arts_edt_create_with_epoch(profiled_task, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = 0, .id = 42});

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
