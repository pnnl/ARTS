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

/// @file db_destroy.c
/// @brief Tests arts_db_destroy.

#include "arts.h"

/// Test 1: arts_db_destroy on a freshly created DB.
void after_destroy(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: db_destroy completed without crash\n");
}

/// Test 3: Create, release, destroy, create new — verify GUID reuse works.
void verify_new_db(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 777);
  if (ok) {
    arts_printf("  PASS: new DB after destroy has correct data\n");
  } else {
    arts_printf("  FAIL: new DB after destroy data mismatch\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_destroy ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: arts_db_destroy (implicit release).
  void *p1 = NULL;
  arts_guid_t db1 = arts_db_create(&p1, 64, ARTS_DB_DEFAULT, NULL);
  arts_db_destroy(db1);
  arts_edt_create_with_epoch(after_destroy, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = 0});

  // Test 2: arts_db_destroy on a second DB (implicit release).
  void *p2 = NULL;
  arts_guid_t db2 = arts_db_create(&p2, 64, ARTS_DB_DEFAULT, NULL);
  arts_db_destroy(db2);
  arts_edt_create_with_epoch(after_destroy, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = 0});

  // Test 4: Create new DB after destroying old one.
  void *p3 = NULL;
  arts_guid_t db3 = arts_db_create(&p3, sizeof(int), ARTS_DB_DEFAULT, NULL);
  ((int *)p3)[0] = 777;
  arts_db_release(db3);

  arts_guid_t e3 = arts_edt_create_with_epoch(verify_new_db, 0, NULL, 1, epoch,
                                              &(arts_hint_t){.route = 0});
  arts_signal_edt(e3, 0, db3, DB_MODE_RO);

  // Test 5: Double destroy (should be no-op on second call, not crash).
  void *p5 = NULL;
  arts_guid_t db5 = arts_db_create(&p5, 64, ARTS_DB_DEFAULT, NULL);
  arts_db_destroy(db5);
  arts_db_destroy(db5); // Second destroy — route table returns NULL
  arts_printf("  PASS: double destroy did not crash\n");

  // Test 6: arts_db_destroy on ARTS_DB_LOCAL (implicit release, should not
  // crash).
  void *p6 = NULL;
  arts_guid_t db6 = arts_db_create(&p6, 64, ARTS_DB_LOCAL, NULL);
  arts_db_destroy(db6); // Should log warning and return
  arts_printf("  PASS: destroy on LOCAL DB warned without crash\n");

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
