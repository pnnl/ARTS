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

/// @file event_chain.c
/// @brief Tests event-to-event chaining and event-to-EDT dependence wiring.
///        Event A → Event B → EDT: when A fires, B should also fire,
///        which triggers the final EDT.

#include "arts.h"

/// Test 1: Linear chain: Event1 → Event2 → EDT.
void chain_end(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: event chain Event1->Event2->EDT fired\n");
}

/// Test 2: Fan-in event: Two events → one event → EDT.
void fan_in_end(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: event fan-in (2 events -> 1 event -> EDT) fired\n");
}

/// Test 3: Event chain with data propagation.
void chain_data_end(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 12345);
  if (ok) {
    arts_printf("  PASS: event chain propagated data correctly\n");
  } else {
    arts_printf("  FAIL: event chain data mismatch\n");
  }
}

/// Test 4: Already-fired event wired to EDT fires immediately.
void already_fired_end(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: already-fired event triggered EDT immediately\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== event_chain ===\n");

  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  // Test 1: Event1(latch=1) → Event2(latch=1) → EDT.
  arts_guid_t ev1 = arts_event_create(NULL);
  arts_guid_t ev2 = arts_event_create(NULL);
  arts_guid_t edt1 = arts_edt_create(
      chain_end, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});

  // Wire: ev1 fires → satisfies ev2 slot 0 → ev2 fires → satisfies edt1 slot 0.
  arts_add_dependence(ev1, ev2, ARTS_EVENT_LATCH_DECR_SLOT, DB_MODE_RW);
  arts_add_dependence(ev2, edt1, 0, DB_MODE_RW);

  // Fire ev1.
  arts_event_satisfy_slot(ev1, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 2: Fan-in: ev_a + ev_b → ev_c → EDT.  ev_c needs both decrements.
  arts_guid_t ev_a = arts_event_create(NULL);
  arts_guid_t ev_b = arts_event_create(NULL);
  arts_event_hint_t fan_in_hint = ARTS_EVENT_HINT_DEFAULTS;
  fan_in_hint.latch = 2;
  arts_guid_t ev_c = arts_event_create(&fan_in_hint);
  arts_guid_t edt2 = arts_edt_create(
      fan_in_end, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});

  arts_add_dependence(ev_a, ev_c, ARTS_EVENT_LATCH_DECR_SLOT, DB_MODE_RW);
  arts_add_dependence(ev_b, ev_c, ARTS_EVENT_LATCH_DECR_SLOT, DB_MODE_RW);
  arts_add_dependence(ev_c, edt2, 0, DB_MODE_RW);

  arts_event_satisfy_slot(ev_a, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_satisfy_slot(ev_b, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 3: Event chain with DB data propagation.
  void *dbptr = NULL;
  arts_guid_t db = arts_db_create(&dbptr, sizeof(int), ARTS_DB_DEFAULT,
                                  ARTS_DB_PROP_NONE, NULL);
  ((int *)dbptr)[0] = 12345;
  arts_db_release(db, DB_MODE_RW);

  arts_guid_t ev3 = arts_event_create(NULL);
  arts_guid_t edt3 =
      arts_edt_create(chain_data_end, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(ev3, edt3, 0, DB_MODE_RW);
  // Fire with data.
  arts_event_satisfy_slot(ev3, db, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 4: Already-fired IDEM-equivalent event → wire EDT after fire →
  // add_dependence fast path delivers immediate signal.
  arts_event_hint_t idem_hint = ARTS_EVENT_HINT_IDEMPOTENT;
  arts_guid_t ev4 = arts_event_create(&idem_hint);
  arts_event_satisfy_slot(ev4, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  // Now wire after fire — fast path self-signals.
  arts_guid_t edt4 =
      arts_edt_create(already_fired_end, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(ev4, edt4, 0, DB_MODE_RW);
  arts_event_destroy(ev4);

  arts_epoch_wait(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
