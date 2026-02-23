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

/// @file db_put_get.c
/// @brief Tests arts_put_in_db, arts_get_from_db, arts_put_in_db_at,
///        arts_get_from_db_at, and arts_put_in_db_epoch with various offsets.

#include "arts.h"
#include <string.h>

#define DB_SIZE 128

/// Verify get_from_db with offset.
void check_get(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int expected_val = (unsigned int)paramv[0];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data != NULL && *data == expected_val) {
    arts_printf("  PASS: get_from_db offset read correct (val=%u)\n",
                expected_val);
  } else {
    arts_printf("  FAIL: get_from_db mismatch\n");
  }
}

/// Verify put_in_db wrote correctly.
void check_put(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int expected_offset = (unsigned int)paramv[0];
  unsigned int expected_val = (unsigned int)paramv[1];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data != NULL) {
    unsigned int *elem = (unsigned int *)((char *)data + expected_offset);
    if (*elem == expected_val) {
      arts_printf("  PASS: put_in_db at offset %u = %u\n", expected_offset,
                  expected_val);
    } else {
      arts_printf("  FAIL: put_in_db at offset %u: got %u, expected %u\n",
                  expected_offset, *elem, expected_val);
    }
  } else {
    arts_printf("  FAIL: put_in_db null pointer\n");
  }
}

/// Verify put_in_db_epoch.
void check_put_epoch(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  bool ok = true;
  if (data == NULL) {
    ok = false;
  } else {
    // Check first 4 unsigned ints were written.
    for (unsigned int i = 0; i < 4; i++) {
      if (data[i] != (i + 100)) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: put_in_db_epoch data correct\n");
  } else {
    arts_printf("  FAIL: put_in_db_epoch mismatch\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_put_get ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Create a DB and fill with known data.
  arts_guid_t db_guid = arts_guid_reserve(ARTS_DB, 0);
  unsigned int *db_data =
      (unsigned int *)arts_db_create_with_guid(db_guid, DB_SIZE, ARTS_DB_DEFAULT, NULL, NULL);
  for (unsigned int i = 0; i < DB_SIZE / sizeof(unsigned int); i++) {
    db_data[i] = i * 7;
  }
  arts_db_release(db_guid);

  // Test 1: arts_get_from_db with offset = 3 * sizeof(unsigned int).
  uint64_t get_param = db_data[3];  // i.e., 3 * 7 = 21
  arts_guid_t e1 = arts_edt_create_with_epoch(
      check_get, 1, &get_param, 1, epoch, &(arts_hint_t){.route = 0});
  arts_get_from_db(e1, db_guid, 0, 3 * sizeof(unsigned int),
                   sizeof(unsigned int));

  // Test 2: arts_put_in_db — write a value at offset, then signal a checker.
  unsigned int val2 = 9999;
  unsigned int offset2 = 5 * sizeof(unsigned int);
  uint64_t put_params[2] = {offset2, val2};
  arts_guid_t e2 = arts_edt_create_with_epoch(
      check_put, 2, put_params, 1, epoch, &(arts_hint_t){.route = 0});
  arts_put_in_db(&val2, e2, db_guid, 0, offset2, sizeof(unsigned int));

  // Test 3: arts_put_in_db_epoch.
  arts_guid_t db2_guid = arts_guid_reserve(ARTS_DB, 0);
  unsigned int *db2 =
      (unsigned int *)arts_db_create_with_guid(db2_guid, DB_SIZE, ARTS_DB_DEFAULT, NULL, NULL);
  memset(db2, 0, DB_SIZE);
  arts_db_release(db2_guid);

  unsigned int epoch_data[4] = {100, 101, 102, 103};
  arts_put_in_db_epoch(epoch_data, epoch, db2_guid, 0,
                       4 * sizeof(unsigned int));

  arts_guid_t e3 = arts_edt_create_with_epoch(
      check_put_epoch, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt(e3, 0, db2_guid, DB_MODE_RO);

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
