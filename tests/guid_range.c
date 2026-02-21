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
/// @brief Tests GUID range APIs: create, get, next, has_next, reset_iter.

#include "arts.h"
#include <stdlib.h>

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== guid_range ===\n");
  unsigned int my_node = arts_get_current_node();
  bool all_pass = true;

#define RANGE_SIZE 16

  // Test 1: Create range and iterate with arts_guid_range_get.
  arts_guid_range_t *range =
      arts_guid_range_create(ARTS_DB, RANGE_SIZE, my_node);
  if (range == NULL) {
    arts_printf("  FAIL: guid_range_create returned NULL\n");
    arts_shutdown();
    return;
  }
  arts_printf("  PASS: guid_range_create succeeded (size=%d)\n", RANGE_SIZE);

  // Verify all GUIDs via get() are unique and have correct type/rank.
  arts_guid_t gotten[RANGE_SIZE];
  for (unsigned int i = 0; i < RANGE_SIZE; i++) {
    gotten[i] = arts_guid_range_get(range, i);
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
        "  PASS: guid_range_get returns unique, correctly-typed GUIDs\n");
  }

  // Test 2: Iterator: has_next + next.
  arts_guid_range_reset_iter(range);
  unsigned int iter_count = 0;
  while (arts_guid_range_has_next(range)) {
    arts_guid_t g = arts_guid_range_next(range);
    if (g != gotten[iter_count]) {
      arts_printf("  FAIL: next[%u] != get[%u]\n", iter_count, iter_count);
      all_pass = false;
    }
    iter_count++;
  }
  if (iter_count == RANGE_SIZE) {
    arts_printf("  PASS: iterator yielded %u GUIDs\n", iter_count);
  } else {
    arts_printf("  FAIL: iterator yielded %u, expected %d\n", iter_count,
                RANGE_SIZE);
    all_pass = false;
  }

  // Test 3: has_next returns false after exhaustion.
  if (!arts_guid_range_has_next(range)) {
    arts_printf("  PASS: has_next returns false after exhaustion\n");
  } else {
    arts_printf("  FAIL: has_next still true\n");
    all_pass = false;
  }

  // Test 4: reset_iter and re-iterate.
  arts_guid_range_reset_iter(range);
  if (arts_guid_range_has_next(range)) {
    arts_guid_t first = arts_guid_range_next(range);
    if (first == gotten[0]) {
      arts_printf("  PASS: reset_iter restarts from beginning\n");
    } else {
      arts_printf("  FAIL: reset_iter first GUID mismatch\n");
      all_pass = false;
    }
  } else {
    arts_printf("  FAIL: has_next false after reset\n");
    all_pass = false;
  }

  free(range);

  arts_printf("=== guid_range: %s ===\n", all_pass ? "ALL PASSED" : "FAILED");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
