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

/// @file record_dep_at.c
/// @brief Tests arts_record_dep and arts_record_dep_at (byte-offset slicing).

#include "arts.h"
#include <string.h>

/// Test 1: Basic arts_record_dep with DB_MODE_RO.
void check_record_dep_ro(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 42 && data[1] == 99);
  if (ok) {
    arts_printf("  PASS: record_dep RO - data read correctly\n");
  } else {
    arts_printf("  FAIL: record_dep RO\n");
  }
}

/// Test 2: arts_record_dep with DB_MODE_EW (exclusive write).
/// After the first writer finishes, the second reader sees modified data.
void writer_ew(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = 1000;
    data[1] = 2000;
  }
  arts_printf("  PASS: record_dep EW - write completed\n");
}

void reader_after_ew(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 1000 && data[1] == 2000);
  if (ok) {
    arts_printf("  PASS: record_dep EW->RO ordering correct\n");
  } else {
    arts_printf("  FAIL: record_dep EW->RO data mismatch\n");
  }
}

/// Test 3: arts_record_dep_at - byte offset slicing.
/// DB layout: [int a, int b, int c, int d] (16 bytes)
/// Slice at offset=8, len=8 gives pointer to c,d.
void check_slice(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  // depv[0].ptr should point to offset 8 within the DB.
  int *slice = (int *)depv[0].ptr;
  bool ok = (slice != NULL && slice[0] == 300 && slice[1] == 400);
  if (ok) {
    arts_printf("  PASS: record_dep_at byte offset slice correct\n");
  } else {
    if (slice) {
      arts_printf("  FAIL: record_dep_at got [%d, %d] expected [300, 400]\n",
                  slice[0], slice[1]);
    } else {
      arts_printf("  FAIL: record_dep_at null pointer\n");
    }
  }
}

/// Test 4: arts_record_dep_at preserves the original DB GUID.
void check_slice_guid(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)depc;
  arts_guid_t expected_guid = (arts_guid_t)paramv[0];
  bool ok = (depv[0].guid == expected_guid && depv[0].ptr != NULL);
  if (ok) {
    arts_printf("  PASS: record_dep_at preserves DB GUID\n");
  } else {
    arts_printf("  FAIL: record_dep_at GUID mismatch\n");
  }
  (void)paramc;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== record_dep_at ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: Basic RO record_dep.
  void *ptr1 = NULL;
  arts_guid_t db1 =
      arts_db_create(&ptr1, 2 * sizeof(int), ARTS_DB_DEFAULT, NULL);
  int *d1 = (int *)ptr1;
  d1[0] = 42;
  d1[1] = 99;
  arts_db_release(db1);

  arts_guid_t e1 = arts_edt_create_with_epoch(
      check_record_dep_ro, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence(db1, e1, 0, DB_MODE_RO);

  // Test 2: EW → RO ordering via record_dep.
  void *ptr2 = NULL;
  arts_guid_t db2 =
      arts_db_create(&ptr2, 2 * sizeof(int), ARTS_DB_DEFAULT, NULL);
  int *d2 = (int *)ptr2;
  d2[0] = 0;
  d2[1] = 0;
  arts_db_release(db2);

  arts_guid_t ew_edt = arts_edt_create_with_epoch(writer_ew, 0, NULL, 1, epoch,
                                                  &(arts_hint_t){.route = 0});
  arts_add_dependence(db2, ew_edt, 0, DB_MODE_EW);

  arts_guid_t ro_edt = arts_edt_create_with_epoch(
      reader_after_ew, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence(db2, ro_edt, 0, DB_MODE_RO);

  // Test 3: record_dep_at with byte offset.
  void *ptr3 = NULL;
  arts_guid_t db3 =
      arts_db_create(&ptr3, 4 * sizeof(int), ARTS_DB_DEFAULT, NULL);
  int *d3 = (int *)ptr3;
  d3[0] = 100;
  d3[1] = 200;
  d3[2] = 300;
  d3[3] = 400;
  arts_db_release(db3);

  arts_guid_t e3 = arts_edt_create_with_epoch(check_slice, 0, NULL, 1, epoch,
                                              &(arts_hint_t){.route = 0});
  arts_add_dependence_at(db3, e3, 0, DB_MODE_RO, 2 * sizeof(int), 2 * sizeof(int));

  // Test 4: record_dep_at preserves DB GUID.
  uint64_t guid_param = (uint64_t)db3;
  arts_guid_t e4 = arts_edt_create_with_epoch(
      check_slice_guid, 1, &guid_param, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence_at(db3, e4, 0, DB_MODE_RO, sizeof(int), sizeof(int));

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
