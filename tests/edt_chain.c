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

/// @file edt_chain.c
/// @brief Tests EDT chaining: EDT A signals EDT B which signals EDT C.
///        Validates data flows correctly through a pipeline of EDTs.

#include "arts.h"

#define CHAIN_LEN 10

/// Each stage increments the value by 1 and passes it on.
void chain_stage(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t stage = paramv[0];
  arts_guid_t next_guid = (arts_guid_t)paramv[1];
  uint64_t value = (uint64_t)depv[0].guid;

  arts_printf("  Stage %lu: received value %lu\n", stage, value);
  arts_signal_edt_value(next_guid, 0, value + 1);
}

/// Final stage verifies the accumulated value.
void chain_final(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t value = (uint64_t)depv[0].guid;

  if (value == CHAIN_LEN) {
    arts_printf("  PASS: chain accumulated value %lu == %d\n", value,
                CHAIN_LEN);
  } else {
    arts_printf("  FAIL: chain value %lu != %d\n", value, CHAIN_LEN);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_chain (length=%d) ===\n", CHAIN_LEN);

  // Build the chain backwards: final <- stage[N-1] <- ... <- stage[0].
  arts_guid_t final_edt =
      arts_edt_create(chain_final, 0, NULL, 1, &(arts_hint_t){.route = 0});

  arts_guid_t next = final_edt;
  for (int i = CHAIN_LEN - 1; i >= 0; i--) {
    uint64_t args[2];
    args[0] = (uint64_t)i;
    args[1] = (uint64_t)next;
    next = arts_edt_create(chain_stage, 2, args, 1, &(arts_hint_t){.route = 0});
  }

  // Kick off chain with value 0.
  arts_signal_edt_value(next, 0, 0);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
