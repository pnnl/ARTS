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
/// @brief Tests all EDT signaling variants now expressed via
///        arts_add_dependence: DB source (RO/RW), raw value (DB_MODE_VAL),
///        and NULL_GUID source (DB_MODE_NULL).

#include "arts.h"
#include <string.h>

#define MAGIC 0xCAFEBABE12345678ULL

/// 1) arts_add_dependence(db, edt, slot, DB_MODE_RO): deliver DB read-only.
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

/// 2) arts_add_dependence(value, ..., DB_MODE_VAL): deliver raw uint64.
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

/// 3) NULL source via arts_add_dependence(NULL_GUID, ...).
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

  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  // 1) signal_edt with DB.
  void *db_ptr = NULL;
  arts_guid_t db = arts_db_create(&db_ptr, sizeof(uint64_t), ARTS_DB_DEFAULT,
                                  ARTS_DB_PROP_NONE, NULL);
  *(uint64_t *)db_ptr = MAGIC;
  arts_db_release(db);
  arts_guid_t e1 = arts_edt_create(signal_db_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(db, e1, 0, DB_MODE_RO);

  // 2) signal_edt_value.
  arts_guid_t e2 = arts_edt_create(signal_value_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence((arts_guid_t)(42), e2, 0, DB_MODE_VAL);
  arts_add_dependence((arts_guid_t)(0xDEADULL), e2, 1, DB_MODE_VAL);

  // 3) NULL source + raw value via arts_add_dependence.
  arts_guid_t e5 = arts_edt_create(signal_null_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(NULL_GUID, e5, 0, DB_MODE_NULL);
  arts_add_dependence((arts_guid_t)(77), e5, 1, DB_MODE_VAL);

  arts_epoch_wait(epoch);
  arts_printf("=== edt_signal complete ===\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
