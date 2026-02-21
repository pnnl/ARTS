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

/// @file epoch_basic.c
/// @brief Tests epoch APIs: arts_initialize_epoch, arts_start_epoch,
///        arts_initialize_and_start_epoch, arts_wait_on_handle,
///        arts_get_current_epoch_guid, arts_add_edt_to_epoch.

#include "arts.h"

volatile unsigned int task_count = 0;

void dummy_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  __sync_fetch_and_add((unsigned int *)&task_count, 1);
}

/// EDT that runs inside an epoch and checks current epoch GUID.
void epoch_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t expected = (arts_guid_t)paramv[0];
  arts_guid_t current = arts_get_current_epoch_guid();
  if (current == expected) {
    arts_printf("  PASS: get_current_epoch_guid matches expected\n");
  } else {
    arts_printf("  FAIL: epoch mismatch: current=%lu expected=%lu\n",
                (uint64_t)current, (uint64_t)expected);
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== epoch_basic ===\n");

  // Test 1: arts_initialize_and_start_epoch + wait.
  task_count = 0;
  arts_guid_t epoch1 = arts_initialize_and_start_epoch(NULL_GUID, 0);
  for (unsigned int i = 0; i < 10; i++) {
    arts_edt_create_with_epoch(dummy_task, 0, NULL, 0, epoch1,
                               &(arts_hint_t){.route = 0});
  }
  arts_wait_on_handle(epoch1);
  arts_printf("  Test 1: epoch completed, task_count=%u\n", task_count);
  if (task_count == 10) {
    arts_printf("  PASS: all 10 tasks ran in epoch\n");
  } else {
    arts_printf("  FAIL: expected 10, got %u\n", task_count);
  }

  // Test 2: arts_get_current_epoch_guid inside epoch.
  arts_guid_t epoch3 = arts_initialize_and_start_epoch(NULL_GUID, 0);
  uint64_t ep3_param = (uint64_t)epoch3;
  arts_edt_create_with_epoch(epoch_check, 1, &ep3_param, 0, epoch3,
                             &(arts_hint_t){.route = 0});
  arts_wait_on_handle(epoch3);

  arts_printf("=== epoch_basic complete ===\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
