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

/// @file event_channel_advanced.c
/// @brief Tests advanced channel event features:
///        arts_event_satisfy_slot (CHANNEL path),
///        arts_event_add_dependence_with_byte_offset,
///        arts_event_add_dependence_with_mode.

#include "arts.h"
#include <string.h>

/// Test 1: arts_event_satisfy_slot on a CHANNEL event.
void pe_satisfy_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 111);
  if (ok) {
    arts_printf("  PASS: channel event satisfy delivered data\n");
  } else {
    arts_printf("  FAIL: channel event satisfy\n");
  }
}

/// Test 2: full-DB delivery via channel event; receiver indexes into
/// payload directly.  DB = [int a, int b, int c]; read element b.
/// (Byte-offset slicing is reserved for arts_db_hint_t.access_offset/_size.)
void pe_byte_offset_check(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *full = (int *)depv[0].ptr;
  arts_guid_t expected_guid = (arts_guid_t)paramv[0];
  bool ok = (full != NULL && full[1] == 200);
  bool guid_ok = (depv[0].guid == expected_guid);
  if (ok && guid_ok) {
    arts_printf("  PASS: channel event full-DB delivery indexed correctly\n");
  } else {
    arts_printf("  FAIL: channel event full-DB delivery (data_ok=%d, "
                "guid_ok=%d)\n",
                ok, guid_ok);
  }
}

/// Test 3: arts_event_add_dependence_with_mode.
void pe_mode_diff_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 999);
  if (ok) {
    arts_printf("  PASS: channel event with_mode OK\n");
  } else {
    arts_printf("  FAIL: channel event with_mode\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== persistent_event_advanced ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  // Channel-equivalent hint: channel=true.  Each satisfy/add_dep pushes
  // into its queue and increments its counter; the drainer pops one of
  // each per fire pair.
  arts_event_hint_t channel_hint = ARTS_EVENT_HINT_CHANNEL;

  // Test 1: arts_event_satisfy_slot on CHANNEL — data delivered via satisfy.
  void *p1 = NULL;
  arts_guid_t db1 = arts_db_create(&p1, sizeof(int), ARTS_DB_DEFAULT,
                                   ARTS_DB_PROP_NONE, NULL);
  ((int *)p1)[0] = 111;
  arts_db_release(db1, DB_MODE_RW);

  arts_guid_t ch1 = arts_event_create(&channel_hint);
  arts_guid_t e1 =
      arts_edt_create(pe_satisfy_check, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(ch1, e1, 0, DB_MODE_RW);
  arts_event_satisfy_slot(ch1, db1, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 2: byte-offset dependence from channel event.
  void *p2 = NULL;
  arts_guid_t db2 = arts_db_create(&p2, 3 * sizeof(int), ARTS_DB_DEFAULT,
                                   ARTS_DB_PROP_NONE, NULL);
  int *d2 = (int *)p2;
  d2[0] = 100;
  d2[1] = 200;
  d2[2] = 300;
  arts_db_release(db2, DB_MODE_RW);

  arts_guid_t ch2 = arts_event_create(&channel_hint);
  uint64_t guid_param = (uint64_t)db2;
  arts_guid_t e2 =
      arts_edt_create(pe_byte_offset_check, 1, &guid_param, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  /* Slicing reserved for arts_db_hint_t; receiver indexes into full DB. */
  arts_add_dependence(ch2, e2, 0, DB_MODE_RO);
  arts_event_satisfy_slot(ch2, db2, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 3: mode dependence.
  void *p3 = NULL;
  arts_guid_t db3 = arts_db_create(&p3, sizeof(int), ARTS_DB_DEFAULT,
                                   ARTS_DB_PROP_NONE, NULL);
  ((int *)p3)[0] = 999;
  arts_db_release(db3, DB_MODE_RW);

  arts_guid_t ch3 = arts_event_create(&channel_hint);
  arts_guid_t e3 =
      arts_edt_create(pe_mode_diff_check, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(ch3, e3, 0, DB_MODE_RO);
  arts_event_satisfy_slot(ch3, db3, ARTS_EVENT_LATCH_DECR_SLOT);

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
