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

/// @file event_basic.c
/// @brief Single-node test for latch events: arts_event_create,
///        arts_event_create_with_guid, arts_event_satisfy_slot,
///        arts_add_dependence, arts_is_event_fired, arts_event_destroy,
///        arts_add_local_event_callback.

#include "arts.h"

volatile unsigned int callback_ran = 0;

/// Callback for arts_add_local_event_callback.
void my_callback(arts_edt_dep_t data) {
  (void)data;
  __sync_fetch_and_add((unsigned int *)&callback_ran, 1);
}

/// EDT wired via arts_add_dependence from an event.
void dependent_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: dependent EDT fired from event\n");
}

/// Check arts_is_event_fired.
void check_fired_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  if (arts_is_event_fired(event)) {
    arts_printf("  PASS: is_event_fired returns true after fire\n");
  } else {
    arts_printf("  FAIL: is_event_fired returns false after fire\n");
  }
}

/// EDT to verify event_create_with_guid.
void guid_event_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: event_create_with_guid fires dependent correctly\n");
}

/// EDT that checks the callback ran.
void check_callback_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  if (callback_ran > 0) {
    arts_printf("  PASS: local event callback was invoked (%u times)\n",
                callback_ran);
  } else {
    arts_printf("  FAIL: local event callback was NOT invoked\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== event_basic ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: Basic latch event with initial count=2.
  arts_guid_t ev1 = arts_event_create(0, 2);
  arts_guid_t dep1 = arts_edt_create_with_epoch(
      dependent_edt, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence(ev1, dep1, 0);

  // Decrement twice to fire.
  arts_event_satisfy_slot(ev1, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_satisfy_slot(ev1, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 2: arts_is_event_fired on a pre-fired event.
  arts_guid_t ev2 = arts_event_create(0, 1);
  arts_event_satisfy_slot(ev2, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  uint64_t ev2_param = (uint64_t)ev2;
  arts_edt_create_with_epoch(check_fired_edt, 1, &ev2_param, 0, epoch,
                             &(arts_hint_t){.route = 0});

  // Test 3: Increment then decrement.
  arts_guid_t ev3 = arts_event_create(0, 1);
  arts_guid_t dep3 = arts_edt_create_with_epoch(
      dependent_edt, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence(ev3, dep3, 0);
  // Increment (+1 -> 2), then decrement twice.
  arts_event_satisfy_slot(ev3, NULL_GUID, ARTS_EVENT_LATCH_INCR_SLOT);
  arts_event_satisfy_slot(ev3, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_satisfy_slot(ev3, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 4: arts_event_create_with_guid.
  arts_guid_t reserved_ev = arts_guid_reserve(ARTS_EVENT, 0);
  arts_event_create_with_guid(reserved_ev, 1);
  arts_guid_t dep4 = arts_edt_create_with_epoch(
      guid_event_dep, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence(reserved_ev, dep4, 0);
  arts_event_satisfy_slot(reserved_ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  // Test 5: arts_event_destroy (destroy unfired event).
  arts_guid_t ev5 = arts_event_create(0, 100);
  arts_event_destroy(ev5);
  arts_printf("  PASS: event_destroy did not crash\n");

  // Test 6: arts_add_local_event_callback.
  arts_guid_t ev6 = arts_event_create(0, 1);
  arts_add_local_event_callback(ev6, my_callback);
  arts_event_satisfy_slot(ev6, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  // Schedule a check after a brief delay.
  arts_edt_create_with_epoch(check_callback_edt, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = 0});

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
