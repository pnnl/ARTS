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

/// @file multinode_array_db.c
/// @brief Tests array DB distributed across nodes: cross-node put/get,
///        cross-node gather. Requires multi-node (node_count > 1).

#include "arts.h"
#include "arts/array_db.h"

#define ELEMS_PER_NODE 4

/// Test 1: Verify cross-node get retrieved correct value.
void check_remote_get(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int expected = (unsigned int)paramv[0];
  bool ok = (data != NULL && *data == expected);
  if (ok) {
    arts_printf("  PASS: cross-node array get element=%u\n", expected);
  } else {
    arts_printf("  FAIL: cross-node array get expected=%u got=%u\n", expected,
                data ? *data : 0xFFFFFFFF);
  }
}

/// Test 2: Gather checker — verify all blocks across nodes.
void mn_gather_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  bool ok = true;
  unsigned int idx = 0;
  for (unsigned int i = 0; i < depc; i++) {
    unsigned int *block_data = (unsigned int *)depv[i].ptr;
    if (!block_data) {
      arts_printf("  FAIL: gather block %u is NULL\n", i);
      ok = false;
      idx += ELEMS_PER_NODE;
      continue;
    }
    for (unsigned int j = 0; j < ELEMS_PER_NODE; j++) {
      if (block_data[j] != idx * 10) {
        arts_printf("  FAIL: gather[%u] = %u expected %u\n", idx, block_data[j],
                    idx * 10);
        ok = false;
      }
      idx++;
    }
  }
  if (ok) {
    arts_printf("  PASS: cross-node gather verified %u elements\n", idx);
  }
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

  arts_printf("=== multinode_array_db ===\n");

  unsigned int total = arts_get_total_ranks();
  unsigned int total_elems = ELEMS_PER_NODE * total;

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), shut, 0);
  arts_epoch_start(epoch);

  // Create distributed array — blocks spread across all nodes.
  arts_array_db_t *array = NULL;
  arts_new_array_db(&array, sizeof(unsigned int), total_elems);

  // Write all elements with value = index * 10.
  for (unsigned int i = 0; i < total_elems; i++) {
    unsigned int val = i * 10;
    arts_put_in_array_db(&val, NULL_GUID, 0, array, i);
  }

  // Test 1: Cross-node get — retrieve an element owned by node 1.
  // Element ELEMS_PER_NODE should be on node 1 (round-robin block placement).
  {
    unsigned int remote_idx = ELEMS_PER_NODE; // first element on node 1
    uint64_t expected_param = (uint64_t)remote_idx * 10;
    arts_guid_t checker =
        arts_edt_create(check_remote_get, 1, &expected_param, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
    arts_get_from_array_db(checker, 0, array, remote_idx);
  }

  // Test 2: Gather all blocks to node 0 and verify data integrity.
  arts_gather_array_db_epoch(array, mn_gather_check, 0, 0, NULL, 0, epoch);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
