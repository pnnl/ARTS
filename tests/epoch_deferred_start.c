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

/// @file epoch_deferred_start.c
/// @brief Tests arts_epoch_create + arts_epoch_start (deferred start).

#include "arts.h"

#define NUM_TASKS 5

/// Simple task within the epoch.
void deferred_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

/// Finish callback.
void deferred_finish(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: deferred start epoch completed\n");
}

/// Test 2: Start epoch, then use arts_epoch_add_edt.
void added_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: EDT added to epoch via arts_epoch_add_edt\n");
}

/// Test 3: arts_epoch_get_current_guid inside an epoch.
void check_current_epoch(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t expected = (arts_guid_t)paramv[0];
  arts_guid_t current = arts_epoch_get_current_guid();
  bool ok = (current == expected);
  if (ok) {
    arts_printf("  PASS: get_current_epoch_guid matches expected\n");
  } else {
    arts_printf("  FAIL: get_current_epoch_guid mismatch\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== epoch_deferred_start ===\n");

  // Test 1: Initialize epoch with finish EDT, create tasks, start, wait.
  arts_guid_t fin1 = arts_edt_create(deferred_finish, 0, NULL, 1,
                                     &(arts_edt_hint_t){.rank = 0});
  arts_guid_t epoch1 = arts_epoch_create(0, fin1, 0);

  // Create tasks using arts_edt_create_with_epoch (proper epoch enrollment).
  for (int i = 0; i < NUM_TASKS; i++) {
    arts_edt_create(deferred_task, 0, NULL, 0,
                    &(arts_edt_hint_t){.rank = 0, .epoch = epoch1});
  }

  // Now start — epoch begins tracking completion.
  arts_epoch_start(epoch1);
  arts_epoch_wait(epoch1);

  // Test 2: get_current_epoch_guid inside an epoch.
  arts_guid_t epoch2 = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch2);
  uint64_t ep_param = (uint64_t)epoch2;
  arts_edt_create(check_current_epoch, 1, &ep_param, 0,
                  &(arts_edt_hint_t){.rank = 0, .epoch = epoch2});
  arts_epoch_wait(epoch2);

  arts_printf("=== epoch_deferred_start complete ===\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
