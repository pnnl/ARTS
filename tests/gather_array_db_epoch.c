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

/// @file gather_array_db_epoch.c
/// @brief Tests arts_gather_array_db_epoch.

#include "arts.h"
#include "arts/array_db.h"

void gather_epoch_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  bool ok = true;
  for (uint32_t i = 0; i < depc; i++) {
    if (depv[i].ptr == NULL) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: gather_array_db_epoch gathered %u blocks\n", depc);
  } else {
    arts_printf("  FAIL: gather_array_db_epoch missing blocks\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== gather_array_db_epoch ===\n");

  arts_array_db_t *arr = NULL;
  arts_new_array_db(&arr, sizeof(int), 8);

  if (arr == NULL) {
    arts_printf("  FAIL: arts_new_array_db returned NULL\n");
    arts_shutdown();
    return;
  }

  // Put some data.
  for (unsigned int i = 0; i < 8; i++) {
    int val = (int)(i + 1);
    arts_put_in_array_db(&val, NULL_GUID, 0, arr, i);
  }

  // Gather within an epoch (depc=0 means no extra deps beyond num_blocks).
  // The gather callback will call arts_shutdown. Don't wait in main_edt because
  // the array DB blocks are auto-acquired by main_edt; waiting here would
  // deadlock since the gather EDT needs to read them.
  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);
  arts_gather_array_db_epoch(arr, gather_epoch_check, 0, 0, NULL, 0, epoch);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
