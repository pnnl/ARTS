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

/// @file guid_basic.c
/// @brief Tests GUID management: arts_guid_reserve, arts_guid_is_local,
///        arts_guid_get_rank, arts_guid_get_kind,
///        arts_guid_reserve_range with ARTS_HINT_ROUND_ROBIN.

#include "arts.h"
#include <stdlib.h>

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== guid_basic ===\n");
  unsigned int num_nodes = arts_get_total_ranks();
  unsigned int my_node = arts_get_current_rank();
  bool all_pass = true;

  // Test 1: Reserve local GUID — is_local should be true.
  arts_guid_t g1 = arts_guid_reserve(ARTS_GUID_EDT, my_node);
  if (!arts_guid_is_local(g1)) {
    arts_printf("  FAIL: guid_is_local returned false for local GUID\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: guid_is_local correct for local GUID\n");
  }

  // Test 2: get_rank matches reservation node.
  unsigned int rank = arts_guid_get_rank(g1);
  if (rank != my_node) {
    arts_printf("  FAIL: guid_get_rank = %u, expected %u\n", rank, my_node);
    all_pass = false;
  } else {
    arts_printf("  PASS: guid_get_rank correct\n");
  }

  // Test 3: get_type matches reservation type.
  arts_guid_kind_t type = arts_guid_get_kind(g1);
  if (type != ARTS_GUID_EDT) {
    arts_printf("  FAIL: guid_get_type = %d, expected %d\n", type,
                ARTS_GUID_EDT);
    all_pass = false;
  } else {
    arts_printf("  PASS: guid_get_type correct for ARTS_GUID_EDT\n");
  }

  // Test 4: Reserve GUIDs for multiple types, verify get_type.
  arts_guid_kind_t types[] = {ARTS_GUID_DB, ARTS_GUID_EVENT, ARTS_GUID_EDT};
  const char *names[] = {"ARTS_GUID_DB", "ARTS_GUID_EVENT", "ARTS_GUID_EDT"};
  for (unsigned int i = 0; i < 3; i++) {
    arts_guid_t g = arts_guid_reserve(types[i], my_node);
    arts_guid_kind_t got = arts_guid_get_kind(g);
    if (got != types[i]) {
      arts_printf("  FAIL: type for %s: got %d, expected %d\n", names[i], got,
                  types[i]);
      all_pass = false;
    } else {
      arts_printf("  PASS: type for %s correct\n", names[i]);
    }
  }

// Test 5: Reserve multiple GUIDs, ensure uniqueness.
#define N_GUIDS 100
  arts_guid_t guids[N_GUIDS];
  for (unsigned int i = 0; i < N_GUIDS; i++) {
    guids[i] = arts_guid_reserve(ARTS_GUID_DB, my_node);
  }
  bool unique = true;
  for (unsigned int i = 0; i < N_GUIDS && unique; i++) {
    for (unsigned int j = i + 1; j < N_GUIDS && unique; j++) {
      if (guids[i] == guids[j]) {
        unique = false;
      }
    }
  }
  if (unique) {
    arts_printf("  PASS: %d reserved GUIDs are unique\n", N_GUIDS);
  } else {
    arts_printf("  FAIL: duplicate GUIDs found\n");
    all_pass = false;
  }

  // Test 6: arts_guid_reserve_range with ARTS_HINT_ROUND_ROBIN.
  unsigned int rr_count = num_nodes * 3;
  arts_guid_t rr_range =
      arts_guid_reserve_range(ARTS_GUID_DB, rr_count, ARTS_HINT_ROUND_ROBIN);
  if (rr_range == NULL_GUID) {
    arts_printf("  FAIL: round_robin range returned NULL_GUID\n");
    all_pass = false;
  } else {
    bool rr_ok = true;
    for (unsigned int i = 0; i < rr_count; i++) {
      unsigned int expected_rank = i % num_nodes;
      arts_guid_t guid_i = arts_guid_from_index(rr_range, i);
      unsigned int actual_rank = arts_guid_get_rank(guid_i);
      if (actual_rank != expected_rank) {
        arts_printf("  FAIL: round_robin[%u] rank=%u, expected=%u\n", i,
                    actual_rank, expected_rank);
        rr_ok = false;
        break;
      }
    }
    if (rr_ok) {
      arts_printf("  PASS: round_robin distributes across %u nodes\n",
                  num_nodes);
    }
    if (!rr_ok) {
      all_pass = false;
    }
  }

  arts_printf("=== guid_basic: %s ===\n", all_pass ? "ALL PASSED" : "FAILED");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
