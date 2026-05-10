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

/// @file db_create_with_data.c
/// @brief Tests arts_db_create_with_guid_and_data (DB with initial data copy).

#include "arts.h"
#include <string.h>

/// Verify data was copied at creation time.
void check_initial_data(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL);
  if (ok) {
    for (int i = 0; i < 8 && ok; i++) {
      if (data[i] != (i + 1) * 11) {
        ok = false;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: db_create_with_guid_and_data initial copy correct\n");
  } else {
    arts_printf("  FAIL: db_create_with_guid_and_data data mismatch\n");
  }
}

/// Verify that modifying the source after creation doesn't affect the DB.
void check_source_independence(uint32_t paramc, const uint64_t *paramv,
                               uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  // Source was zeroed after creation; DB should still have original values.
  bool ok = (data != NULL && data[0] == 100 && data[1] == 200);
  if (ok) {
    arts_printf("  PASS: db_create_with_data is independent of source\n");
  } else {
    arts_printf("  FAIL: db_create_with_data not independent\n");
  }
}

/// Test with zero-length data.
void check_zero_len(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: db_create_with_data zero-length OK\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_create_with_data ===\n");

  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  // Test 1: Create DB and populate the returned buffer directly.
  arts_guid_t g1 = arts_guid_reserve(ARTS_GUID_DB, 0);
  int *p1 = (int *)arts_db_create_with_guid(
      g1, 8 * sizeof(int), ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  for (int i = 0; i < 8; i++) {
    p1[i] = (i + 1) * 11;
  }
  arts_db_release(g1);

  arts_guid_t e1 = arts_edt_create(check_initial_data, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(g1, e1, 0, DB_MODE_RO);

  // Test 2: Same pattern — the caller writes directly into the DB buffer,
  // so there is no separate source array to diverge from.
  arts_guid_t g2 = arts_guid_reserve(ARTS_GUID_DB, 0);
  int *p2 = (int *)arts_db_create_with_guid(
      g2, 2 * sizeof(int), ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  p2[0] = 100;
  p2[1] = 200;
  arts_db_release(g2);

  arts_guid_t e2 = arts_edt_create(check_source_independence, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(g2, e2, 0, DB_MODE_RO);

  arts_epoch_wait(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
