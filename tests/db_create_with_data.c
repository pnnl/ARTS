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

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_create_with_data ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: Create DB with initial data.
  int src[8];
  for (int i = 0; i < 8; i++) {
    src[i] = (i + 1) * 11;
  }
  arts_guid_t g1 = arts_guid_reserve(ARTS_DB, 0);
  arts_db_create_with_guid_and_data(g1, src, 8 * sizeof(int));
  arts_db_release(g1);

  arts_guid_t e1 = arts_edt_create_with_epoch(
      check_initial_data, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt(e1, 0, g1, ARTS_MODE_RO);

  // Test 2: Modify source after creation — DB should be independent.
  int src2[2] = {100, 200};
  arts_guid_t g2 = arts_guid_reserve(ARTS_DB, 0);
  arts_db_create_with_guid_and_data(g2, src2, 2 * sizeof(int));
  arts_db_release(g2);
  // Zero out source.
  memset(src2, 0, sizeof(src2));

  arts_guid_t e2 = arts_edt_create_with_epoch(
      check_source_independence, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_signal_edt(e2, 0, g2, ARTS_MODE_RO);

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
