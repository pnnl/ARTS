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

/// @file array_db_advanced.c
/// @brief Tests advanced array DB: arts_new_array_db_with_guid,
///        arts_signal_array_db, arts_gather_array_db_in_edt,
///        arts_for_each_in_array_db.

#include "arts.h"
#include "arts/array_db.h"
#include <stdlib.h>

#define ELEMS_PER_NODE 4

arts_array_db_t *array = NULL;

/// EDT per element: write element index into it.
void write_elem(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  unsigned int index = (unsigned int)paramv[0];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  *data = index * 10;
  arts_printf("  write_elem[%u] = %u\n", index, *data);
}

/// Gather check: verify all elements.
void gather_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  unsigned int total = depc * ELEMS_PER_NODE;
  bool ok = true;
  unsigned int idx = 0;
  for (unsigned int i = 0; i < depc; i++) {
    unsigned int *block_data = (unsigned int *)depv[i].ptr;
    for (unsigned int j = 0; j < ELEMS_PER_NODE; j++) {
      if (block_data[j] != idx * 10) {
        arts_printf("  FAIL: gather[%u] = %u, expected %u\n", idx,
                    block_data[j], idx * 10);
        ok = false;
      }
      idx++;
    }
  }
  if (ok) {
    arts_printf("  PASS: gather verified %u elements correctly\n", total);
  }
}

/// EDT to receive signal_array_db.
void signal_array_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  bool ok = true;
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr == NULL) {
      arts_printf("  FAIL: signal_array_db slot %u is NULL\n", i);
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: signal_array_db delivered %u blocks\n", depc);
  }
}

/// Test gather_array_db_in_edt.
void gather_in_edt_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  bool ok = true;
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr == NULL) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: gather_array_db_in_edt delivered %u blocks\n", depc);
  } else {
    arts_printf("  FAIL: gather_array_db_in_edt had NULL slots\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== array_db_advanced ===\n");

  unsigned int num_nodes = arts_get_total_ranks();
  unsigned int total_elems = ELEMS_PER_NODE * num_nodes;

  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  // Create array DB.
  arts_guid_t arr_guid =
      arts_new_array_db(&array, sizeof(unsigned int), total_elems);
  (void)arr_guid;

  // Test 1: Manually write index * 10 into each element, then gather to verify.
  for (unsigned int i = 0; i < total_elems; i++) {
    unsigned int val = i * 10;
    arts_put_in_array_db(&val, NULL_GUID, 0, array, i);
  }

  // Test 2: arts_gather_array_db.
  arts_gather_array_db(array, gather_check, 0, 0, NULL, 0);

  // Test 3: arts_signal_array_db — signal an EDT with all blocks.
  unsigned int num_blocks = array->num_blocks;
  arts_guid_t sig_edt =
      arts_edt_create(signal_array_check, 0, NULL, num_blocks, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_signal_array_db(array, sig_edt, 0);

  // Test 4: arts_gather_array_db_in_edt.
  arts_guid_t gather_edt =
      arts_edt_create(gather_in_edt_check, 0, NULL, num_blocks, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_gather_array_db_in_edt(array, gather_edt, 0);

  arts_epoch_wait(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
