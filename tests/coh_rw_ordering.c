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

/// @file coh_rw_ordering.c
/// @brief Tests coherence RW (Exclusive Owner) ordering: multiple RW → RO
///        readers should all see the final writer's data.
///        Also tests sequential RW writers with correct ordering.

#include "arts.h"

/// Writer 1: writes 100.
void writer1(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = 100;
  }
  arts_printf("  writer1 wrote 100\n");
}

/// Writer 2: writes 200.
void writer2(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = 200;
  }
  arts_printf("  writer2 wrote 200\n");
}

/// Reader: verifies that it sees the value written by the preceding writer.
/// paramv[0] = expected value.
void reader_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  int *data = (int *)depv[0].ptr;
  int expected = (int)paramv[0];
  bool ok = (data != NULL && data[0] == expected);
  if (ok) {
    arts_printf("  PASS: reader saw %d\n", expected);
  } else {
    arts_printf("  FAIL: reader expected %d got %d\n", expected,
                data ? data[0] : -1);
  }
}

/// Test 2: Multiple RO readers on the same DB.
void concurrent_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  uint32_t reader_id = (uint32_t)paramv[0];
  bool ok = (data != NULL && data[0] == 555);
  if (ok) {
    arts_printf("  PASS: concurrent reader %u saw 555\n", reader_id);
  } else {
    arts_printf("  FAIL: concurrent reader %u\n", reader_id);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coh_rw_ordering ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  // Test 1: Sequential RW ordering: writer1(RW) → writer2(RW) → reader(RO).
  // record_dep with RW ensures writer1 runs before writer2, and writer2
  // before reader.
  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  ((int *)ptr)[0] = 0;
  arts_db_release(db, DB_MODE_RW);

  arts_guid_t w1 = arts_edt_create(writer1, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, w1, 0, DB_MODE_RW);

  arts_guid_t w2 = arts_edt_create(writer2, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, w2, 0, DB_MODE_RW);

  uint64_t exp_param = 200;
  arts_guid_t r1 = arts_edt_create(reader_check, 1, &exp_param, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, r1, 0, DB_MODE_RO);

  // Test 2: Multiple concurrent RO readers.
  void *ptr2 = NULL;
  arts_guid_t db2 = arts_db_create(&ptr2, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  ((int *)ptr2)[0] = 555;
  arts_db_release(db2, DB_MODE_RW);

  for (uint32_t i = 0; i < 4; i++) {
    uint64_t id_param = (uint64_t)i;
    arts_guid_t reader = arts_edt_create(concurrent_reader, 1, &id_param, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db2, reader, 0, DB_MODE_RO);
  }

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
