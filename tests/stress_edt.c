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

/// @file stress_edt.c
/// @brief Stress test: create many EDTs rapidly to stress the scheduler.

#include "arts.h"

#define NUM_EDTS 1000

/// Simple EDT that increments a shared counter via value signal.
void stress_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t collector = (arts_guid_t)paramv[0];
  uint32_t slot = (uint32_t)paramv[1];
  arts_add_dependence((arts_guid_t)(1), collector, slot, DB_MODE_VAL);
}

/// Collector: depc = NUM_EDTS, each slot holds value 1.
void stress_collector(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  unsigned int count = 0;
  for (uint32_t i = 0; i < depc; i++) {
    if ((uint64_t)depv[i].guid == 1) {
      count++;
    }
  }
  if (count == NUM_EDTS) {
    arts_printf("  PASS: stress test %u EDTs completed\n", NUM_EDTS);
  } else {
    arts_printf("  FAIL: stress test got %u/%u completions\n", count, NUM_EDTS);
  }
}

/// Test 2: EDT chain stress — 100-stage pipeline.
#define CHAIN_LEN 100

void chain_stage(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  uint64_t stage = (uint64_t)depv[0].guid;
  arts_guid_t next = (arts_guid_t)paramv[0];
  arts_add_dependence((arts_guid_t)(stage + 1), next, 0, DB_MODE_VAL);
}

void chain_final(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t val = (uint64_t)depv[0].guid;
  if (val == CHAIN_LEN) {
    arts_printf("  PASS: chain stress %u stages completed, val=%lu\n",
                CHAIN_LEN, (unsigned long)val);
  } else {
    arts_printf("  FAIL: chain stress val=%lu expected %u\n",
                (unsigned long)val, CHAIN_LEN);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== stress_edt ===\n");

  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  // Test 1: Fan-out stress with NUM_EDTS.
  arts_guid_t collector = arts_edt_create(stress_collector, 0, NULL, NUM_EDTS, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});

  for (uint32_t i = 0; i < NUM_EDTS; i++) {
    uint64_t params[2];
    params[0] = (uint64_t)collector;
    params[1] = (uint64_t)i;
    arts_edt_create(stress_task, 2, params, 0, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  }

  // Test 2: Chain stress — CHAIN_LEN stages.
  // Build chain in reverse: final ← stage[N-1] ← ... ← stage[0].
  arts_guid_t final_edt = arts_edt_create(chain_final, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_guid_t prev = final_edt;
  for (int i = CHAIN_LEN - 1; i >= 0; i--) {
    uint64_t param = (uint64_t)prev;
    prev = arts_edt_create(chain_stage, 1, &param, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  }
  // Seed first stage.
  arts_add_dependence((arts_guid_t)(0), prev, 0, DB_MODE_VAL);

  arts_epoch_wait(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
