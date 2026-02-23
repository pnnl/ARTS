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

/// @file array_db_with_guid.c
/// @brief Tests arts_new_array_db_with_guid (distributed array with
///        pre-reserved GUID).

#include "arts.h"
#include "arts/array_db.h"

void check_array_with_guid(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  bool ok = true;
  for (uint32_t i = 0; i < depc; i++) {
    if (depv[i].ptr == NULL) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: new_array_db_with_guid gathered %u blocks\n", depc);
  } else {
    arts_printf("  FAIL: new_array_db_with_guid missing blocks\n");
  }
}

/// Test 2: get_from_array_db.
void check_get_from_array(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *val = (int *)depv[0].ptr;
  bool ok = (val != NULL && *val == 42);
  if (ok) {
    arts_printf("  PASS: get_from_array_db element value correct\n");
  } else {
    arts_printf("  FAIL: get_from_array_db value mismatch\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== array_db_with_guid ===\n");

  // Test 1: Create distributed array with pre-reserved GUID.
  arts_guid_t arr_guid = arts_guid_reserve(ARTS_DB_LOCAL, 0);
  arts_array_db_t *arr = arts_new_array_db_with_guid(arr_guid, sizeof(int), 16);

  if (arr == NULL) {
    arts_printf("  FAIL: arts_new_array_db_with_guid returned NULL\n");
    arts_shutdown();
    return;
  }

  // Put a value at index 3.
  int val = 42;
  arts_put_in_array_db(&val, NULL_GUID, 0, arr, 3);

  // Test 1a: Gather (depc=0 means no extra deps beyond num_blocks).
  arts_gather_array_db(arr, check_array_with_guid, 0, 0, NULL, 0);

  // Test 2: get_from_array_db. This callback will call arts_shutdown.
  arts_guid_t reader = arts_edt_create(check_get_from_array, 0, NULL, 1, NULL);
  arts_get_from_array_db(reader, 0, arr, 3);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
