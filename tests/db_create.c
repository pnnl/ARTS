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

/// @file db_create.c
/// @brief Tests all DataBlock creation APIs: arts_db_create,
///        arts_db_create_with_guid, arts_db_create_with_guid_and_data,
///        arts_db_release, arts_db_destroy.

#include "arts.h"
#include <string.h>

#define DB_SIZE 256
#define DB_ELEMS (DB_SIZE / sizeof(uint64_t))

/// Test 1: arts_db_create returns valid pointer and GUID.
void check_db_create(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  bool ok = (data != NULL);
  for (unsigned int i = 0; i < DB_ELEMS && ok; i++) {
    if (data[i] != i + 1) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: db_create data intact via EW signal\n");
  } else {
    arts_printf("  FAIL: db_create data corrupted\n");
  }
}

/// Test 2: arts_db_create_with_guid returns correct pointer.
void check_db_with_guid(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  arts_guid_t expected = (arts_guid_t)paramv[0];
  bool ok = (data != NULL && depv[0].guid == expected);
  if (ok) {
    for (unsigned int i = 0; i < DB_ELEMS; i++) {
      if (data[i] != (i * 3)) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: db_create_with_guid data & GUID correct\n");
  } else {
    arts_printf("  FAIL: db_create_with_guid mismatch\n");
  }
}

/// Test 3: arts_db_create_with_guid_and_data copies initial data.
void check_db_with_data(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  bool ok = (data != NULL);
  for (unsigned int i = 0; i < DB_ELEMS && ok; i++) {
    if (data[i] != 0xBEEF + i) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: db_create_with_guid_and_data initial data correct\n");
  } else {
    arts_printf("  FAIL: db_create_with_guid_and_data mismatch\n");
  }
}

/// Test 4: arts_db_destroy removes DB.
void post_destroy_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: db_destroy did not crash, post-destroy EDT ran\n");
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_create ===\n");
  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: arts_db_create + arts_db_release.
  void *ptr1 = NULL;
  arts_guid_t db1 = arts_db_create(&ptr1, DB_SIZE, NULL);
  uint64_t *d1 = (uint64_t *)ptr1;
  for (unsigned int i = 0; i < DB_ELEMS; i++) {
    d1[i] = i + 1;
  }
  arts_db_release(db1);
  arts_guid_t e1 = arts_edt_create_with_epoch(
      check_db_create, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt(e1, 0, db1, ARTS_MODE_EW);

  // Test 2: arts_db_create_with_guid.
  arts_guid_t reserved = arts_guid_reserve(ARTS_DB, 0);
  uint64_t *d2 = (uint64_t *)arts_db_create_with_guid(reserved, DB_SIZE, NULL);
  for (unsigned int i = 0; i < DB_ELEMS; i++) {
    d2[i] = i * 3;
  }
  arts_db_release(reserved);
  uint64_t param2 = (uint64_t)reserved;
  arts_guid_t e2 = arts_edt_create_with_epoch(
      check_db_with_guid, 1, &param2, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt(e2, 0, reserved, ARTS_MODE_RO);

  // Test 3: arts_db_create_with_guid_and_data.
  uint64_t init_data[DB_ELEMS];
  for (unsigned int i = 0; i < DB_ELEMS; i++) {
    init_data[i] = 0xBEEF + i;
  }
  arts_guid_t reserved3 = arts_guid_reserve(ARTS_DB, 0);
  arts_db_create_with_guid_and_data(reserved3, init_data, DB_SIZE);
  arts_db_release(reserved3);
  arts_guid_t e3 = arts_edt_create_with_epoch(
      check_db_with_data, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt(e3, 0, reserved3, ARTS_MODE_RO);

  // Test 4: arts_db_destroy.
  void *ptr4 = NULL;
  arts_guid_t db4 = arts_db_create(&ptr4, 64, NULL);
  arts_db_release(db4);
  arts_db_destroy(db4);

  arts_edt_create_with_epoch(post_destroy_edt, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = 0});

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
