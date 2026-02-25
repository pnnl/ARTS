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

/// @file multinode_event_types.c
/// @brief Tests non-LATCH event types across nodes: ONCE, STICKY, IDEM,
///        COUNTED. Each type tested with remote satisfaction from node 1.
///        Requires multi-node (node_count > 1).

#include "arts.h"

/// Generic remote satisfier: receives event GUID in paramv[0].
void remote_satisfy(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_event_satisfy_slot(event, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/// Test 1: ONCE event dependent.
void once_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: cross-node ONCE event fired dependent\n");
}

/// Test 2: STICKY event — trampoline registers late dep after fire.
void sticky_late_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: cross-node STICKY late dep fired\n");
}

void sticky_trampoline(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_guid_t epoch = (arts_guid_t)paramv[1];
  // Event has already fired — register a late dependent.
  arts_guid_t late = arts_edt_create_with_epoch(
      sticky_late_dep, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence(event, late, 0);
  arts_event_destroy(event);
}

/// Test 3: IDEM event dependent.
void idem_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: cross-node IDEM event fired dependent\n");
}

/// Test 3: IDEM re-satisfy from node 0 (should be silent no-op).
void idem_re_satisfy(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_event_satisfy_slot(event, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_destroy(event);
  arts_printf("  PASS: cross-node IDEM re-satisfy did not crash\n");
}

/// Test 4: COUNTED fan-in dependent.
void counted_fan_in_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  unsigned int expected = (unsigned int)paramv[0];
  arts_printf("  PASS: cross-node COUNTED fan-in fired (%u nodes)\n", expected);
}

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== multinode_event_types ===\n");

  unsigned int total = arts_get_total_nodes();
  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t epoch = arts_initialize_and_start_epoch(shut, 0);

  // Test 1: ONCE event — create on node 0, satisfy from node 1.
  {
    arts_guid_t ev = arts_event_create(0, ARTS_EVENT_ONCE, 0, NULL_GUID);
    arts_guid_t dep = arts_edt_create_with_epoch(once_dep, 0, NULL, 1, epoch,
                                                 &(arts_hint_t){.route = 0});
    arts_add_dependence(ev, dep, 0);

    uint64_t ev_param = (uint64_t)ev;
    arts_edt_create_with_epoch(remote_satisfy, 1, &ev_param, 0, epoch,
                               &(arts_hint_t){.route = 1});
  }

  // Test 2: STICKY event — satisfy from node 1, then register late dep.
  // The trampoline EDT depends on the event (fires when event fires),
  // and then registers a SECOND late-dependent on the same (persisted) event.
  {
    arts_guid_t ev = arts_event_create(0, ARTS_EVENT_STICKY, 0, NULL_GUID);

    uint64_t tramp_params[2] = {(uint64_t)ev, (uint64_t)epoch};
    arts_guid_t tramp =
        arts_edt_create_with_epoch(sticky_trampoline, 2, tramp_params, 1, epoch,
                                   &(arts_hint_t){.route = 0});
    arts_add_dependence(ev, tramp, 0);

    uint64_t ev_param = (uint64_t)ev;
    arts_edt_create_with_epoch(remote_satisfy, 1, &ev_param, 0, epoch,
                               &(arts_hint_t){.route = 1});
  }

  // Test 3: IDEM event — satisfy from node 1, re-satisfy from node 0.
  {
    arts_guid_t ev = arts_event_create(0, ARTS_EVENT_IDEM, 0, NULL_GUID);
    arts_guid_t dep = arts_edt_create_with_epoch(idem_dep, 0, NULL, 1, epoch,
                                                 &(arts_hint_t){.route = 0});
    arts_add_dependence(ev, dep, 0);

    // Satisfy from node 1.
    uint64_t ev_param = (uint64_t)ev;
    arts_edt_create_with_epoch(remote_satisfy, 1, &ev_param, 0, epoch,
                               &(arts_hint_t){.route = 1});

    // Re-satisfy from node 0 (trampoline after the event fires via dep).
    uint64_t re_params[1] = {(uint64_t)ev};
    arts_guid_t re = arts_edt_create_with_epoch(
        idem_re_satisfy, 1, re_params, 1, epoch, &(arts_hint_t){.route = 0});
    arts_add_dependence(ev, re, 0);
  }

  // Test 4: COUNTED event — fan-in from all nodes.
  {
    arts_guid_t ev = arts_event_create(0, ARTS_EVENT_COUNTED, total, NULL_GUID);
    uint64_t total_param = (uint64_t)total;
    arts_guid_t dep =
        arts_edt_create_with_epoch(counted_fan_in_dep, 1, &total_param, 1,
                                   epoch, &(arts_hint_t){.route = 0});
    arts_add_dependence(ev, dep, 0);

    for (unsigned int r = 0; r < total; r++) {
      uint64_t ev_param = (uint64_t)ev;
      arts_edt_create_with_epoch(remote_satisfy, 1, &ev_param, 0, epoch,
                                 &(arts_hint_t){.route = r});
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
