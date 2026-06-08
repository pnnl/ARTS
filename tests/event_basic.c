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
/// @brief Single-node test for event hint variants: defaults (ONCE-equivalent),
///        LATCH, IDEM, STICKY, COUNTED, plus pre-reserved GUID (hint->guid) and
///        arts_event_destroy on an unfired event.

#include "arts.h"

#include <stdint.h>

/// EDT wired via arts_add_dependence from an event.
void dependent_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: dependent EDT fired from event\n");
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

/// EDT for ONCE event test.
void once_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: ONCE event fired dependent\n");
}

/// EDT for STICKY late-dep test.
void sticky_late_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: STICKY late dep immediately satisfied\n");
}

/// EDT for IDEM re-satisfy test.
void idem_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: IDEM event fired dependent\n");
}

/// EDT for COUNTED event test.
void counted_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: COUNTED event fired dependent\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== event_basic ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  // Test 1: Basic latch event with initial count=2.
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_DEFAULTS;
    h.latch = 2;
    arts_guid_t ev1 = arts_event_create(&h);
    arts_guid_t dep1 =
        arts_edt_create(dependent_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev1, dep1, 0, DB_MODE_RW);

    // Decrement twice to fire.
    arts_event_satisfy_slot(ev1, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    arts_event_satisfy_slot(ev1, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  }

  // Test 2: (removed) — arts_is_event_fired no longer exists in the new API.
  // The "did the event fire" check is now expressed by chaining a dependent
  // EDT off the event; firing is observable by that EDT running.

  // Test 3: Increment then decrement — exercises ARTS_EVENT_LATCH_INCR_SLOT.
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_DEFAULTS;
    h.latch = 1;
    arts_guid_t ev3 = arts_event_create(&h);
    arts_guid_t dep3 =
        arts_edt_create(dependent_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev3, dep3, 0, DB_MODE_RW);
    // Increment (1 -> 2), then decrement twice.
    arts_event_satisfy_slot(ev3, NULL_GUID, ARTS_EVENT_LATCH_INCR_SLOT);
    arts_event_satisfy_slot(ev3, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    arts_event_satisfy_slot(ev3, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  }

  // Test 4: pre-reserved GUID via hint->guid (defaults = ONCE-equivalent).
  {
    arts_guid_t reserved_ev = arts_guid_reserve(ARTS_GUID_EVENT, 0);
    arts_event_hint_t h4 = ARTS_EVENT_HINT_DEFAULTS;
    h4.guid = reserved_ev;
    arts_event_create(&h4);
    arts_guid_t dep4 =
        arts_edt_create(guid_event_dep, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(reserved_ev, dep4, 0, DB_MODE_RW);
    arts_event_satisfy_slot(reserved_ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  }

  // Test 5: arts_event_destroy on an unfired event.
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_DEFAULTS;
    h.latch = 100;
    arts_guid_t ev5 = arts_event_create(&h);
    arts_event_destroy(ev5);
    arts_printf("  PASS: event_destroy did not crash\n");
  }

  // Test 6: (removed) — the legacy local event callback API was deleted.

  // Test 7: single-fire (defaults: latch=1).  Fires once, then lingers.
  {
    arts_guid_t ev7 = arts_event_create(NULL);
    arts_guid_t dep7 = arts_edt_create(
        once_dep, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev7, dep7, 0, DB_MODE_RW);
    arts_event_satisfy_slot(ev7, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  }

  // Test 8: fire-and-linger late bind — a dep registered after the event
  // fires is satisfied immediately from stored data via the add_dependence
  // fast path, then the event is explicitly destroyed.
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_LATCH(1);
    arts_guid_t ev8 = arts_event_create(&h);
    arts_event_satisfy_slot(ev8, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    arts_guid_t dep8 =
        arts_edt_create(sticky_late_dep, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev8, dep8, 0, DB_MODE_RW);
    arts_event_destroy(ev8);
  }

  // Test 9: over-satisfy tolerance — latch=1, a second satisfy past the
  // fire is silently absorbed; the already-fired event keeps the dependent
  // silent.
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_LATCH(1);
    arts_guid_t ev9 = arts_event_create(&h);
    arts_guid_t dep9 = arts_edt_create(
        idem_dep, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev9, dep9, 0, DB_MODE_RW);
    arts_event_satisfy_slot(ev9, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    // Re-satisfy: with the default hint this just decrements past zero; the
    // already-fired event keeps the dependent silent.
    arts_event_satisfy_slot(ev9, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    arts_event_destroy(ev9);
  }

  // Test 10: multi-decrement LATCH, latch=3 — fire-and-linger keeps the
  // event addressable until arts_event_destroy below.
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_LATCH(3);
    arts_guid_t ev10 = arts_event_create(&h);
    arts_guid_t dep10 = arts_edt_create(
        counted_dep, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev10, dep10, 0, DB_MODE_RW);
    arts_event_satisfy_slot(ev10, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    arts_event_satisfy_slot(ev10, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    arts_event_satisfy_slot(ev10, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    arts_event_destroy(ev10);
  }

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
