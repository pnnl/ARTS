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

/// @file multinode_edt.c
/// @brief Tests EDT creation and signaling across nodes: remote execution,
///        all-nodes fan-in, multi-hop chains, and paramv delivery.
///        Requires multi-node (node_count > 1).

#include "arts.h"

// ---------------------------------------------------------------------------
// Test 1: Remote EDT signals back to master.
// ---------------------------------------------------------------------------

void remote_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t collector = (arts_guid_t)paramv[0];
  unsigned int my_rank = arts_get_current_rank();
  arts_add_dependence((arts_guid_t)((uint64_t)my_rank), collector, 0,
                      DB_MODE_VAL);
}

void check_remote_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t remote_rank = (uint64_t)depv[0].guid;
  unsigned int expected = (unsigned int)paramv[0];
  bool ok = ((unsigned int)remote_rank == expected);
  if (ok) {
    arts_printf("  PASS: remote EDT ran on rank %u\n",
                (unsigned int)remote_rank);
  } else {
    arts_printf("  FAIL: expected rank %u got %lu\n", expected,
                (unsigned long)remote_rank);
  }
}

// ---------------------------------------------------------------------------
// Test 2: One EDT per node, fan-in to collector.
// ---------------------------------------------------------------------------

void all_nodes_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t collector = (arts_guid_t)paramv[0];
  uint32_t slot = (uint32_t)paramv[1];
  unsigned int my_rank = arts_get_current_rank();
  arts_add_dependence((arts_guid_t)((uint64_t)my_rank), collector, slot,
                      DB_MODE_VAL);
}

void check_all_nodes(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  bool ok = true;
  for (uint32_t i = 0; i < depc; i++) {
    uint64_t rank = (uint64_t)depv[i].guid;
    if (rank != (uint64_t)i) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: EDTs ran on all %u nodes\n", depc);
  } else {
    arts_printf("  FAIL: EDT distribution incorrect\n");
  }
}

// ---------------------------------------------------------------------------
// Test 3: Multi-hop chain: A(node 0) -> B(node 1) -> C(node 0).
// ---------------------------------------------------------------------------

/// Intermediate hop: add 10 to value, signal next.
void chain_hop(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  arts_guid_t next = (arts_guid_t)paramv[0];
  uint64_t value = (uint64_t)depv[0].guid;
  arts_add_dependence((arts_guid_t)(value + 10), next, 0, DB_MODE_VAL);
}

/// Final hop: assert accumulated value.
void chain_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t value = (uint64_t)depv[0].guid;
  uint64_t expected = (uint64_t)paramv[0];
  bool ok = (value == expected);
  if (ok) {
    arts_printf("  PASS: multi-hop chain value=%lu\n", (unsigned long)value);
  } else {
    arts_printf("  FAIL: multi-hop chain value=%lu expected=%lu\n",
                (unsigned long)value, (unsigned long)expected);
  }
}

// ---------------------------------------------------------------------------
// Test 4: Paramv delivery to remote node.
// ---------------------------------------------------------------------------

void check_paramv(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  uint64_t expected[] = {0xDEAD, 0xBEEF, 0xCAFE, 0xF00D};
  bool ok = (paramc == 4);
  for (uint32_t i = 0; i < paramc && ok; i++) {
    if (paramv[i] != expected[i]) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: paramv delivered correctly to remote node\n");
  } else {
    arts_printf("  FAIL: paramv mismatch on remote node\n");
  }
}

// ---------------------------------------------------------------------------

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

  arts_printf("=== multinode_edt ===\n");

  unsigned int total = arts_get_total_ranks();
  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  // Test 1: Create EDT on remote node 1, have it signal back.
  {
    uint64_t expected_param = 1;
    arts_guid_t checker =
        arts_edt_create(check_remote_edt, 1, &expected_param, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    uint64_t coll_param = (uint64_t)checker;
    arts_edt_create(remote_task, 1, &coll_param, 0,
                    &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
  }

  // Test 2: Create one EDT per node, each reports its rank.
  {
    arts_guid_t all_coll =
        arts_edt_create(check_all_nodes, 0, NULL, total,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    for (unsigned int r = 0; r < total; r++) {
      uint64_t params[2];
      params[0] = (uint64_t)all_coll;
      params[1] = (uint64_t)r;
      arts_edt_create(all_nodes_task, 2, params, 0,
                      &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    }
  }

  // Test 3: Multi-hop chain A(node 0) -> B(node 1) -> C(node 0).
  // A receives 100, adds 10, sends to B. B adds 10, sends to C.
  // C asserts value == 120.
  {
    uint64_t exp3 = 120;
    arts_guid_t c =
        arts_edt_create(chain_check, 1, &exp3, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    uint64_t c_param = (uint64_t)c;
    arts_guid_t b =
        arts_edt_create(chain_hop, 1, &c_param, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    uint64_t b_param = (uint64_t)b;
    arts_guid_t a =
        arts_edt_create(chain_hop, 1, &b_param, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence((arts_guid_t)(100), a, 0, DB_MODE_VAL);
  }

  // Test 4: Paramv delivery to remote node.
  {
    uint64_t pv[4] = {0xDEAD, 0xBEEF, 0xCAFE, 0xF00D};
    arts_edt_create(check_paramv, 4, pv, 0,
                    &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
