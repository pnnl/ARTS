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

/// @file edt_fan_out.c
/// @brief Tests fan-out / fan-in EDT patterns.
///        One EDT creates N child EDTs that all signal a single collector.

#include "arts.h"

#define FAN_WIDTH 32

/// Each child signals the collector with its index as a value.
void fan_child(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  // paramv[0] = index, paramv[1] = collector GUID.
  uint32_t index = (uint32_t)paramv[0];
  arts_guid_t collector = (arts_guid_t)paramv[1];
  arts_signal_edt_value(collector, index, (uint64_t)index + 1);
}

/// Collector: receives FAN_WIDTH value-mode deps.
/// Verifies each slot contains its (index + 1) value.
void collector(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  bool ok = (depc == FAN_WIDTH);
  unsigned int sum = 0;
  for (uint32_t i = 0; i < depc && ok; i++) {
    uint64_t val = (uint64_t)depv[i].guid;
    if (val != (uint64_t)i + 1) {
      ok = false;
    }
    sum += (unsigned int)val;
  }
  unsigned int expected = (unsigned int)(FAN_WIDTH * (FAN_WIDTH + 1) / 2);
  if (ok && sum == expected) {
    arts_printf("  PASS: fan-out/fan-in %u children, sum=%u\n", FAN_WIDTH, sum);
  } else {
    arts_printf("  FAIL: fan-out/fan-in sum=%u expected=%u\n", sum, expected);
  }
}

/// Test 2: fan-out with DB mode — each child writes to its own DB,
/// then collector reads all.

void db_fan_child(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  uint32_t index = (uint32_t)paramv[0];
  arts_guid_t coll_guid = (arts_guid_t)paramv[1];
  arts_guid_t db_guid = (arts_guid_t)paramv[2];

  // Write index * 10 into our DB.
  void *db_ptr = arts_db_create_with_guid(db_guid, sizeof(int), ARTS_DB_DEFAULT,
                                          NULL, NULL);
  ((int *)db_ptr)[0] = (int)(index * 10);
  arts_db_release(db_guid);
  arts_signal_edt(coll_guid, index, db_guid, DB_MODE_RO);
}

void db_collector(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  int sum = 0;
  bool ok = true;
  for (uint32_t i = 0; i < depc; i++) {
    int *data = (int *)depv[i].ptr;
    if (!data || data[0] != (int)(i * 10)) {
      ok = false;
      break;
    }
    sum += data[0];
  }
  int expected = 0;
  for (uint32_t i = 0; i < FAN_WIDTH; i++) {
    expected += (int)(i * 10);
  }
  if (ok && sum == expected) {
    arts_printf("  PASS: fan-out DB mode, sum=%d\n", sum);
  } else {
    arts_printf("  FAIL: fan-out DB mode, sum=%d expected=%d\n", sum, expected);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_fan_out ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: Value-mode fan-out/fan-in.
  arts_guid_t coll = arts_edt_create_with_epoch(
      collector, 0, NULL, FAN_WIDTH, epoch, &(arts_hint_t){.route = 0});
  for (uint32_t i = 0; i < FAN_WIDTH; i++) {
    uint64_t params[2];
    params[0] = (uint64_t)i;
    params[1] = (uint64_t)coll;
    arts_edt_create_with_epoch(fan_child, 2, params, 0, epoch,
                               &(arts_hint_t){.route = 0});
  }

  // Test 2: DB-mode fan-out/fan-in.
  arts_guid_t db_coll = arts_edt_create_with_epoch(
      db_collector, 0, NULL, FAN_WIDTH, epoch, &(arts_hint_t){.route = 0});
  arts_guid_t range_start = arts_guid_reserve_range(ARTS_DB, FAN_WIDTH, 0);
  for (uint32_t i = 0; i < FAN_WIDTH; i++) {
    arts_guid_t db_guid = arts_guid_from_index(range_start, i);
    uint64_t params[3];
    params[0] = (uint64_t)i;
    params[1] = (uint64_t)db_coll;
    params[2] = (uint64_t)db_guid;
    arts_edt_create_with_epoch(db_fan_child, 3, params, 0, epoch,
                               &(arts_hint_t){.route = 0});
  }

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
