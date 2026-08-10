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

/// @file db_dependence.c
/// @brief Tests DB dependence with immediate satisfy: arts_add_dependence
///        with DB sources in various access modes.

#include "arts.h"
#include <string.h>

/// Test 1: arts_db_add_dependence — fire EDT when DB's internal event triggers.
void check_db_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 77);
  if (ok) {
    arts_printf("  PASS: db_add_dependence delivered DB data\n");
  } else {
    arts_printf("  FAIL: db_add_dependence\n");
  }
}

/// Test 2: arts_db_add_dependence_with_mode — explicit mode override.
void check_db_dep_mode(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 88);
  if (ok) {
    arts_printf("  PASS: db_add_dependence_with_mode OK\n");
  } else {
    arts_printf("  FAIL: db_add_dependence_with_mode\n");
  }
}

/// Test 3: arts_db_add_dependence_with_mode_and_diff.
void check_db_dep_mode_diff(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 33);
  if (ok) {
    arts_printf("  PASS: db_add_dependence_with_mode_and_diff OK\n");
  } else {
    arts_printf("  FAIL: db_add_dependence_with_mode_and_diff\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_dependence ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  // Test 1: arts_db_add_dependence.
  void *p1 = NULL;
  arts_guid_t db1 = arts_db_create(&p1, sizeof(int), ARTS_DB_DEFAULT,
                                   ARTS_DB_PROP_NONE, NULL);
  ((int *)p1)[0] = 77;
  arts_db_release(db1, DB_MODE_RW);

  arts_guid_t e1 =
      arts_edt_create(check_db_dep, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db1, e1, 0, DB_MODE_RW);

  // Test 2: arts_db_add_dependence_with_mode.
  void *p2 = NULL;
  arts_guid_t db2 = arts_db_create(&p2, sizeof(int), ARTS_DB_DEFAULT,
                                   ARTS_DB_PROP_NONE, NULL);
  ((int *)p2)[0] = 88;
  arts_db_release(db2, DB_MODE_RW);

  arts_guid_t e2 =
      arts_edt_create(check_db_dep_mode, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db2, e2, 0, DB_MODE_RO);

  // Test 3: arts_db_add_dependence_with_mode_and_diff.
  void *p4 = NULL;
  arts_guid_t db4 = arts_db_create(&p4, sizeof(int), ARTS_DB_DEFAULT,
                                   ARTS_DB_PROP_NONE, NULL);
  ((int *)p4)[0] = 33;
  arts_db_release(db4, DB_MODE_RW);

  arts_guid_t e4 =
      arts_edt_create(check_db_dep_mode_diff, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db4, e4, 0, DB_MODE_RO);

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
