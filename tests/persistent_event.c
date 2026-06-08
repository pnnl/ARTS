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

/// @file persistent_event.c
/// @brief Tests channel event APIs:
///        arts_event_create (ARTS_EVENT_CHANNEL), arts_event_satisfy_slot,
///        arts_add_dependence, arts_event_add_dependence_with_mode.

#include "arts.h"

/// EDT triggered by channel event — counts invocations.
volatile unsigned int pe_fire_count = 0;

void pe_dependent(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int count = __sync_add_and_fetch((unsigned int *)&pe_fire_count, 1);
  arts_printf("  pe_dependent fired (count=%u)\n", count);
}

/// Verify channel event fired dependent with correct data GUID.
void pe_data_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t expected_db = (arts_guid_t)paramv[0];
  if (depv[0].guid == expected_db && depv[0].ptr != NULL) {
    arts_printf("  PASS: channel event delivered data GUID correctly\n");
  } else {
    arts_printf("  FAIL: channel event data mismatch (got guid=%lu)\n",
                (uint64_t)depv[0].guid);
  }
}

/// Final check, then shutdown.
void pe_final(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  pe_fire_count = %u\n", pe_fire_count);
  if (pe_fire_count >= 2) {
    arts_printf("  PASS: channel event fired multiple dependents\n");
  } else {
    arts_printf("  FAIL: expected >= 2 fires, got %u\n", pe_fire_count);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== persistent_event ===\n");

  // Create a DB to associate with the channel event.
  void *db_ptr = NULL;
  arts_guid_t db =
      arts_db_create(&db_ptr, 64, ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  *(uint64_t *)db_ptr = 0xABCDULL;
  arts_db_release(db, DB_MODE_RW);

  /* pe_final must run only after every contributing EDT has finished.
   * Wire it as the finish scope's finish_edt callback (depc=1, slot 0 satisfied
   * by finish scope fire). Without this, pe_final has depc=0 and races with
   * dep1/dep2/dep3 — worker scheduling can run pe_final before all
   * pe_dependent fires land, reading pe_fire_count<2 and triggering
   * arts_shutdown which then strands the remaining dependents. */
  arts_guid_t pe_final_edt =
      arts_edt_create(pe_final, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, pe_final_edt, 0, DB_MODE_NULL);

  // Channel-equivalent hint: channel=true.  Each satisfy/add_dep increments
  // its own counter; the drainer pops one satisfy and one dep per fire pair.
  arts_event_hint_t channel_hint = ARTS_EVENT_HINT_CHANNEL;

  // Test 1: Channel event with 2 dependents — fire twice, each fire drains one.
  arts_guid_t ch1 = arts_event_create(&channel_hint);

  arts_guid_t dep1 =
      arts_edt_create(pe_dependent, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(ch1, dep1, 0, DB_MODE_RW);

  arts_guid_t dep2 =
      arts_edt_create(pe_dependent, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(ch1, dep2, 0, DB_MODE_RW);

  arts_event_satisfy_slot(ch1, db, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_satisfy_slot(ch1, db, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 2: Channel event delivers the data GUID to a dependent.
  arts_guid_t ch2 = arts_event_create(&channel_hint);
  uint64_t db_param = (uint64_t)db;
  arts_guid_t dep3 =
      arts_edt_create(pe_data_check, 1, &db_param, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(ch2, dep3, 0, DB_MODE_RO);
  arts_event_satisfy_slot(ch2, db, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 3: Multiple satisfies on the same channel accumulate in the
  // data_queue (nb_sat increments).  No dependents → only verifies the
  // counter does not crash.
  arts_guid_t ch3 = arts_event_create(&channel_hint);
  arts_event_satisfy_slot(ch3, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_satisfy_slot(ch3, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_printf("  PASS: channel re-arm pattern did not crash\n");

  /* pe_final fires after the finish scope completes (all dep1/dep2/dep3 done)
   * and calls arts_shutdown. main_edt does not wait or shutdown. */
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
