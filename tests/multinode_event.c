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

/// @file multinode_event.c
/// @brief Tests cross-node latch events.
///        Requires multi-node (node_count > 1).

#include "arts.h"

/// EDT on remote node: satisfy event on node 0.
void remote_satisfier(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_event_satisfy_slot(event, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/// Final EDT: triggered when cross-node event fires.
void event_done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: cross-node event fired\n");
}

/// Test 2: Multi-node fan-in event.
void node_satisfier(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_event_satisfy_slot(event, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

void fan_in_done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int expected_nodes = (unsigned int)paramv[0];
  arts_printf("  PASS: multi-node fan-in event fired (%u nodes)\n",
              expected_nodes);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== multinode_event ===\n");

  unsigned int total = arts_get_total_nodes();
  if (total < 2) {
    arts_printf("  SKIP: need node_count >= 2 (have %u)\n", total);
    arts_shutdown();
    return;
  }

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: Event on node 0, satisfied from node 1.
  arts_guid_t ev1 = arts_event_create(0, 1);
  arts_guid_t done1 = arts_edt_create_with_epoch(event_done, 0, NULL, 1, epoch,
                                                 &(arts_hint_t){.route = 0});
  arts_add_dependence(ev1, done1, 0);

  uint64_t ev_param = (uint64_t)ev1;
  arts_edt_create_with_epoch(remote_satisfier, 1, &ev_param, 0, epoch,
                             &(arts_hint_t){.route = 1});

  // Test 2: Event with latch = total_nodes, each node satisfies once.
  arts_guid_t ev2 = arts_event_create(0, total);
  uint64_t total_param = (uint64_t)total;
  arts_guid_t done2 = arts_edt_create_with_epoch(
      fan_in_done, 1, &total_param, 1, epoch, &(arts_hint_t){.route = 0});
  arts_add_dependence(ev2, done2, 0);

  for (unsigned int r = 0; r < total; r++) {
    uint64_t param = (uint64_t)ev2;
    arts_edt_create_with_epoch(node_satisfier, 1, &param, 0, epoch,
                               &(arts_hint_t){.route = r});
  }

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
