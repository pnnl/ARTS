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

/// @file edt_signal.c
/// @brief Tests all EDT signaling variants: arts_signal_edt, _value, _ptr,
///        _ptr_with_guid, _null.

#include "arts.h"
#include <string.h>

#define MAGIC 0xCAFEBABE12345678ULL

/// 1) arts_signal_edt: deliver DB with RO mode.
void signal_db_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  if (depc == 1 && data != NULL && *data == MAGIC) {
    arts_printf("  PASS: signal_edt delivers DB with correct data\n");
  } else {
    arts_printf("  FAIL: signal_edt DB data mismatch\n");
  }
}

/// 2) arts_signal_edt_value: deliver raw uint64 value.
void signal_value_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  if (depc == 2 && (uint64_t)depv[0].guid == 42 &&
      (uint64_t)depv[1].guid == 0xDEADULL) {
    arts_printf("  PASS: signal_edt_value delivers correct values\n");
  } else {
    arts_printf("  FAIL: signal_edt_value mismatch: depv[0].guid=%lu "
                "depv[1].guid=%lu\n",
                (uint64_t)depv[0].guid, (uint64_t)depv[1].guid);
  }
}

/// 3) arts_signal_edt_ptr: deliver data copy.
void signal_ptr_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  bool ok = (depc == 1 && data != NULL);
  if (ok) {
    for (unsigned int i = 0; i < 8; i++) {
      if (data[i] != i * 10) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: signal_edt_ptr delivers correct data copy\n");
  } else {
    arts_printf("  FAIL: signal_edt_ptr data mismatch\n");
  }
}

/// 4) arts_signal_edt_ptr_with_guid: delivers ptr + original DB GUID.
void signal_ptr_guid_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)depc;
  arts_guid_t expected_guid = (arts_guid_t)paramv[0];
  unsigned int expected_val = (unsigned int)paramv[1];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (paramc == 2 && depv[0].guid == expected_guid && data != NULL &&
      *data == expected_val) {
    arts_printf("  PASS: signal_edt_ptr_with_guid delivers ptr + GUID\n");
  } else {
    arts_printf("  FAIL: signal_edt_ptr_with_guid mismatch\n");
  }
}

/// 5) arts_signal_edt_null: satisfies slot with no data.
void signal_null_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  // Slot 0 is null, slot 1 has a value.
  bool ok = (depc == 2 && depv[0].ptr == NULL && (uint64_t)depv[1].guid == 77);
  if (ok) {
    arts_printf("  PASS: signal_edt_null satisfies with no data\n");
  } else {
    arts_printf("  FAIL: signal_edt_null mismatch\n");
  }
}

/// Final cleanup EDT.
void done_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depv;
  (void)depc;
  arts_printf("=== edt_signal: all sub-tests executed ===\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_signal ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // 1) signal_edt with DB.
  void *db_ptr = NULL;
  arts_guid_t db = arts_db_create(&db_ptr, sizeof(uint64_t), NULL);
  *(uint64_t *)db_ptr = MAGIC;
  arts_db_release(db);
  arts_guid_t e1 = arts_edt_create_with_epoch(signal_db_edt, 0, NULL, 1, epoch,
                                              &(arts_hint_t){.route = 0});
  arts_signal_edt(e1, 0, db, DB_MODE_RO);

  // 2) signal_edt_value.
  arts_guid_t e2 = arts_edt_create_with_epoch(
      signal_value_edt, 0, NULL, 2, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt_value(e2, 0, 42);
  arts_signal_edt_value(e2, 1, 0xDEADULL);

  // 3) signal_edt_ptr.
  unsigned int buf[8];
  for (unsigned int i = 0; i < 8; i++) {
    buf[i] = i * 10;
  }
  arts_guid_t e3 = arts_edt_create_with_epoch(signal_ptr_edt, 0, NULL, 1, epoch,
                                              &(arts_hint_t){.route = 0});
  arts_signal_edt_ptr(e3, 0, buf, sizeof(buf));

  // 4) signal_edt_ptr_with_guid.
  void *db2_ptr = NULL;
  arts_guid_t db2 = arts_db_create(&db2_ptr, sizeof(unsigned int) * 4, NULL);
  unsigned int *db2_data = (unsigned int *)db2_ptr;
  db2_data[0] = 9999;
  arts_db_release(db2);
  uint64_t args4[2];
  args4[0] = (uint64_t)db2;
  args4[1] = 9999;
  arts_guid_t e4 = arts_edt_create_with_epoch(
      signal_ptr_guid_edt, 2, args4, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt_ptr_with_guid(e4, 0, db2, db2_data, sizeof(unsigned int));

  // 5) signal_edt_null.
  arts_guid_t e5 = arts_edt_create_with_epoch(
      signal_null_edt, 0, NULL, 2, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt_null(e5, 0);
  arts_signal_edt_value(e5, 1, 77);

  arts_wait_on_handle(epoch);
  arts_printf("=== edt_signal complete ===\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
