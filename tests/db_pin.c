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

/// @file db_pin.c
/// @brief Tests ARTS_DB_PIN (node-pinned) datablocks: RW access on home
///        rank, modification persistence, and arts_db_copy_to_new_type.

#include "arts.h"
#include "arts/memory/db.h" /* arts_db_copy_to_new_type */
#include <string.h>

#define DB_SIZE 128

/// Verify ARTS_DB_PIN with RW mode.
void check_pin_rw(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  bool ok = (data != NULL);
  if (ok) {
    for (unsigned int i = 0; i < DB_SIZE / sizeof(unsigned int); i++) {
      if (data[i] != i) {
        ok = false;
        break;
      }
    }
    if (ok) {
      for (unsigned int i = 0; i < DB_SIZE / sizeof(unsigned int); i++) {
        data[i] *= 2;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: DB_PIN with RW mode read/write OK\n");
  } else {
    arts_printf("  FAIL: DB_PIN with RW mode failed\n");
  }
}

/// Verify after RW modification.
void check_modified(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  bool ok = (data != NULL);
  if (ok) {
    for (unsigned int i = 0; i < DB_SIZE / sizeof(unsigned int); i++) {
      if (data[i] != i * 2) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: DB_PIN modification persisted\n");
  } else {
    arts_printf("  FAIL: DB_PIN modification lost\n");
  }
}

/// Verify db_copy_to_new_type.
void check_copy_type(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t new_guid = (arts_guid_t)paramv[0];
  arts_guid_kind_t new_type = arts_guid_get_kind(new_guid);
  uint64_t *data = (uint64_t *)depv[0].ptr;
  bool ok = (data != NULL && new_type == ARTS_GUID_DB);
  if (ok) {
    for (unsigned int i = 0; i < DB_SIZE / sizeof(uint64_t); i++) {
      if (data[i] != (i + 100)) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: db_copy_to_new_type data preserved, type=%d\n",
                new_type);
  } else {
    arts_printf("  FAIL: db_copy_to_new_type mismatch\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_pin ===\n");

  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  // Test 1: Create ARTS_DB_PIN and use DB_MODE_RW.  PIN home == this
  // rank by default; route hint forces it explicitly here so the test
  // is robust to hint-default changes.
  arts_guid_t pin_guid = arts_guid_reserve(ARTS_GUID_DB, 0);
  unsigned int *pin_data = (unsigned int *)arts_db_create_with_guid(
      pin_guid, DB_SIZE, ARTS_DB_PIN, ARTS_DB_PROP_NONE, NULL);
  for (unsigned int i = 0; i < DB_SIZE / sizeof(unsigned int); i++) {
    pin_data[i] = i;
  }
  arts_db_release(pin_guid);

  // Test 2: Verify RW modifications persisted.
  // Chain: e1 (modify) -> e2 (verify) using RW per-node-exclusive
  // ordering through the DB.  Both EDTs registered on home rank (0).
  arts_guid_t e2 = arts_edt_create(check_modified, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});

  arts_guid_t e1 = arts_edt_create(check_pin_rw, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(pin_guid, e1, 0, DB_MODE_RW);
  arts_add_dependence(pin_guid, e2, 0, DB_MODE_RW);

  // Test 3: arts_db_copy_to_new_type (DIST -> PIN).
  void *src_ptr = NULL;
  arts_guid_t src_db =
      arts_db_create(&src_ptr, DB_SIZE, ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  uint64_t *src = (uint64_t *)src_ptr;
  for (unsigned int i = 0; i < DB_SIZE / sizeof(uint64_t); i++) {
    src[i] = i + 100;
  }
  arts_db_release(src_db);

  arts_guid_t copied = arts_db_copy_to_new_type(src_db, ARTS_DB_PIN);
  uint64_t copy_param = (uint64_t)copied;
  arts_guid_t e3 = arts_edt_create(check_copy_type, 1, &copy_param, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(copied, e3, 0, DB_MODE_RO);

  arts_epoch_wait(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
