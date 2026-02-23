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

/// @file guid_range.c
/// @brief Tests GUID range APIs: reserve_range, from_index, index_from.

#include "arts.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== guid_range ===\n");
  unsigned int my_node = arts_get_current_node();
  bool all_pass = true;

#define RANGE_SIZE 16

  // Test 1: Reserve range and verify GUIDs via arts_guid_from_index.
  arts_guid_t start = arts_guid_reserve_range(ARTS_DB, RANGE_SIZE, my_node);
  if (start == NULL_GUID) {
    arts_printf("  FAIL: guid_reserve_range returned NULL_GUID\n");
    arts_shutdown();
    return;
  }
  arts_printf("  PASS: guid_reserve_range succeeded (size=%d)\n", RANGE_SIZE);

  // Verify all GUIDs are unique and have correct type/rank.
  arts_guid_t gotten[RANGE_SIZE];
  for (unsigned int i = 0; i < RANGE_SIZE; i++) {
    gotten[i] = arts_guid_from_index(start, i);
    if (arts_guid_get_type(gotten[i]) != ARTS_DB) {
      arts_printf("  FAIL: range[%u] type mismatch\n", i);
      all_pass = false;
    }
    if (arts_guid_get_rank(gotten[i]) != my_node) {
      arts_printf("  FAIL: range[%u] rank mismatch\n", i);
      all_pass = false;
    }
  }
  // Check uniqueness.
  for (unsigned int i = 0; i < RANGE_SIZE; i++) {
    for (unsigned int j = i + 1; j < RANGE_SIZE; j++) {
      if (gotten[i] == gotten[j]) {
        arts_printf("  FAIL: range[%u] == range[%u]\n", i, j);
        all_pass = false;
      }
    }
  }
  if (all_pass) {
    arts_printf(
        "  PASS: arts_guid_from_index returns unique, correctly-typed GUIDs\n");
  }

  // Test 2: arts_guid_from_index is consistent (idempotent).
  for (unsigned int i = 0; i < RANGE_SIZE; i++) {
    arts_guid_t g = arts_guid_from_index(start, i);
    if (g != gotten[i]) {
      arts_printf("  FAIL: from_index(%u) inconsistent\n", i);
      all_pass = false;
    }
  }
  if (all_pass) {
    arts_printf("  PASS: arts_guid_from_index is idempotent\n");
  }

  // Test 3: arts_guid_index_from (reverse lookup).
  for (unsigned int i = 0; i < RANGE_SIZE; i++) {
    int idx = arts_guid_index_from(start, gotten[i]);
    if (idx != (int)i) {
      arts_printf("  FAIL: index_from(gotten[%u]) = %d, expected %u\n", i, idx,
                  i);
      all_pass = false;
    }
  }
  // Mismatch: a GUID from a different-type range should return -1.
  arts_guid_t edt_start = arts_guid_reserve_range(ARTS_EDT, 1, my_node);
  arts_guid_t bad = arts_guid_from_index(edt_start, 0);
  if (arts_guid_index_from(start, bad) == -1) {
    arts_printf("  PASS: arts_guid_index_from returns -1 on type mismatch\n");
  } else {
    arts_printf("  FAIL: arts_guid_index_from did not detect type mismatch\n");
    all_pass = false;
  }

  arts_printf("=== guid_range: %s ===\n", all_pass ? "ALL PASSED" : "FAILED");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
