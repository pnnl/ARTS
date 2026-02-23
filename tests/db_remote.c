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

/// @file db_remote.c
/// @brief Tests arts_db_create_remote. Requires multi-node (node_count > 1).
///        Creates a DB on a remote node and then puts/gets data to/from it.

#include "arts.h"
#include <string.h>

#define DATA_SIZE 256

/// EDT on remote node: verify the DB was created there.
void check_remote_db(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t db_guid = (arts_guid_t)paramv[0];
  unsigned int target_rank = (unsigned int)paramv[1];
  unsigned int db_rank = arts_guid_get_rank(db_guid);
  bool ok = (db_rank == target_rank);
  if (ok) {
    arts_printf("  PASS: db_create_remote created DB on rank %u\n", db_rank);
  } else {
    arts_printf("  FAIL: db_create_remote rank=%u expected=%u\n", db_rank,
                target_rank);
  }
}

/// EDT: verify put/get round-trip to remote DB.
void check_remote_put_get(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned char *data = (unsigned char *)depv[0].ptr;
  bool ok = (data != NULL);
  if (ok) {
    for (unsigned int i = 0; i < DATA_SIZE && ok; i++) {
      if (data[i] != (unsigned char)(i & 0xFF)) {
        ok = false;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: remote DB put/get round-trip correct\n");
  } else {
    arts_printf("  FAIL: remote DB put/get data mismatch\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_remote (multi-node) ===\n");

  unsigned int total = arts_get_total_nodes();
  if (total < 2) {
    arts_printf("  SKIP: need node_count >= 2 (have %u)\n", total);
    arts_shutdown();
    return;
  }

  unsigned int target = 1; // Remote node.

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: arts_db_create_remote on node 1.
  void *tmp;
  arts_guid_t remote_db = arts_db_create(&tmp, DATA_SIZE, ARTS_DB_DEFAULT,
                                         &(arts_hint_t){.route = target});

  uint64_t params[2];
  params[0] = (uint64_t)remote_db;
  params[1] = (uint64_t)target;
  arts_edt_create_with_epoch(check_remote_db, 2, params, 0, epoch,
                             &(arts_hint_t){.route = 0});

  // Test 2: Put data to remote DB, then get it back.
  unsigned char send_buf[DATA_SIZE];
  for (unsigned int i = 0; i < DATA_SIZE; i++) {
    send_buf[i] = (unsigned char)(i & 0xFF);
  }

  arts_guid_t read_edt = arts_edt_create_with_epoch(
      check_remote_put_get, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_put_in_db(send_buf, NULL_GUID, remote_db, 0, 0, DATA_SIZE);

  // Get the data back.
  arts_get_from_db(read_edt, remote_db, 0, 0, DATA_SIZE);

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
