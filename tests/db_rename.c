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

/// @file db_rename.c
/// @brief Tests arts_db_rename and arts_db_rename_with_guid.

#include "arts.h"
#include <string.h>

#define DB_SIZE 64

/// Verify renamed DB contains correct data.
void check_renamed(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  arts_guid_t old_guid = (arts_guid_t)paramv[0];
  bool ok = (data != NULL && depv[0].guid != old_guid);
  if (ok) {
    for (unsigned int i = 0; i < DB_SIZE / sizeof(uint64_t); i++) {
      if (data[i] != (i + 42)) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: db_rename new GUID has correct data\n");
  } else {
    arts_printf("  FAIL: db_rename data or GUID mismatch\n");
  }
}

/// Verify arts_db_rename_with_guid.
void check_renamed_guid(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t expected_new = (arts_guid_t)paramv[0];
  uint64_t *data = (uint64_t *)depv[0].ptr;
  bool ok = (data != NULL && depv[0].guid == expected_new);
  if (ok) {
    for (unsigned int i = 0; i < DB_SIZE / sizeof(uint64_t); i++) {
      if (data[i] != (i * 5)) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: db_rename_with_guid correct\n");
  } else {
    arts_printf("  FAIL: db_rename_with_guid mismatch\n");
  }
  arts_shutdown();
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_rename ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: arts_db_rename.
  void *ptr1 = NULL;
  arts_guid_t db1 = arts_db_create(&ptr1, DB_SIZE, NULL);
  uint64_t *d1 = (uint64_t *)ptr1;
  for (unsigned int i = 0; i < DB_SIZE / sizeof(uint64_t); i++) {
    d1[i] = i + 42;
  }
  arts_db_release(db1);

  arts_guid_t new_guid1 = arts_db_rename(db1);
  arts_printf("  Renamed %lu -> %lu\n", db1, new_guid1);

  uint64_t old_param = (uint64_t)db1;
  arts_guid_t e1 = arts_edt_create_with_epoch(
      check_renamed, 1, &old_param, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt(e1, 0, new_guid1, ARTS_MODE_RO);

  // Test 2: arts_db_rename_with_guid.
  void *ptr2 = NULL;
  arts_guid_t db2 = arts_db_create(&ptr2, DB_SIZE, NULL);
  uint64_t *d2 = (uint64_t *)ptr2;
  for (unsigned int i = 0; i < DB_SIZE / sizeof(uint64_t); i++) {
    d2[i] = i * 5;
  }
  arts_db_release(db2);

  arts_guid_t target = arts_guid_reserve(ARTS_DB, 0);
  bool ok = arts_db_rename_with_guid(target, db2);
  arts_printf("  rename_with_guid returned %s\n", ok ? "true" : "false");

  uint64_t target_param = (uint64_t)target;
  arts_guid_t e2 =
      arts_edt_create_with_epoch(check_renamed_guid, 1, &target_param, 1, epoch,
                                 &(arts_hint_t){.route = 0});
  arts_signal_edt(e2, 0, target, ARTS_MODE_RO);

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
