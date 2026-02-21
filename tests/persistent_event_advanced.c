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

/// @file persistent_event_advanced.c
/// @brief Tests advanced persistent event features:
///        arts_persistent_event_satisfy,
///        arts_add_dependence_to_persistent_event_with_byte_offset,
///        arts_add_dependence_to_persistent_event_with_mode_and_diff.

#include "arts.h"
#include <string.h>

/// Test 1: arts_persistent_event_satisfy with explicit action.
void pe_satisfy_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 111);
  if (ok) {
    arts_printf("  PASS: persistent_event_satisfy delivered data\n");
  } else {
    arts_printf("  FAIL: persistent_event_satisfy\n");
  }
}

/// Test 2: arts_add_dependence_to_persistent_event_with_byte_offset.
/// DB = [int a, int b, int c]. Offset=sizeof(int), len=sizeof(int) → b.
void pe_byte_offset_check(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *slice = (int *)depv[0].ptr;
  arts_guid_t expected_guid = (arts_guid_t)paramv[0];
  bool ok = (slice != NULL && slice[0] == 200);
  bool guid_ok = (depv[0].guid == expected_guid);
  if (ok && guid_ok) {
    arts_printf("  PASS: persistent_event byte_offset slice correct\n");
  } else {
    arts_printf(
        "  FAIL: persistent_event byte_offset (data_ok=%d, guid_ok=%d)\n", ok,
        guid_ok);
  }
}

/// Test 3: arts_add_dependence_to_persistent_event_with_mode_and_diff.
void pe_mode_diff_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 999);
  if (ok) {
    arts_printf("  PASS: persistent_event_with_mode_and_diff OK\n");
  } else {
    arts_printf("  FAIL: persistent_event_with_mode_and_diff\n");
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== persistent_event_advanced ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: arts_persistent_event_satisfy.
  void *p1 = NULL;
  arts_guid_t db1 = arts_db_create(&p1, sizeof(int), NULL);
  ((int *)p1)[0] = 111;
  arts_db_release(db1);

  arts_guid_t pe1 = arts_persistent_event_create(0, 1, db1);
  arts_guid_t e1 = arts_edt_create_with_epoch(
      pe_satisfy_check, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence_to_persistent_event(pe1, e1, 0);
  arts_persistent_event_satisfy(pe1, ARTS_EVENT_LATCH_DECR_SLOT, true);

  // Test 2: byte-offset dependence from persistent event.
  void *p2 = NULL;
  arts_guid_t db2 = arts_db_create(&p2, 3 * sizeof(int), NULL);
  int *d2 = (int *)p2;
  d2[0] = 100;
  d2[1] = 200;
  d2[2] = 300;
  arts_db_release(db2);

  arts_guid_t pe2 = arts_persistent_event_create(0, 1, db2);
  uint64_t guid_param = (uint64_t)db2;
  arts_guid_t e2 =
      arts_edt_create_with_epoch(pe_byte_offset_check, 1, &guid_param, 1, epoch,
                                 &(arts_hint_t){.route = 0});
  arts_add_dependence_to_persistent_event_with_byte_offset(
      pe2, e2, 0, ARTS_MODE_RO, sizeof(int), sizeof(int));
  arts_persistent_event_satisfy(pe2, ARTS_EVENT_LATCH_DECR_SLOT, true);

  // Test 3: mode_and_diff.
  void *p3 = NULL;
  arts_guid_t db3 = arts_db_create(&p3, sizeof(int), NULL);
  ((int *)p3)[0] = 999;
  arts_db_release(db3);

  arts_guid_t pe3 = arts_persistent_event_create(0, 1, db3);
  arts_guid_t e3 = arts_edt_create_with_epoch(
      pe_mode_diff_check, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence_to_persistent_event_with_mode_and_diff(pe3, e3, 0,
                                                             ARTS_MODE_RO);
  arts_persistent_event_satisfy(pe3, ARTS_EVENT_LATCH_DECR_SLOT, true);

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
