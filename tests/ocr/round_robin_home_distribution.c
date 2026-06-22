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

/// @file round_robin_home_distribution.c
/// @brief T039 — ARTS_HINT_ROUND_ROBIN + reserve_range: each labeled index's
///        home rank equals idx % nrank, and a DB created at that GUID is
///        actually local to (homed on) that rank.
///
/// reserve_range(ARTS_HINT_ROUND_ROBIN) returns a DISTRIBUTED range whose
/// from_index plants GUID i at home = idx % nrank.  This test (a) asserts the
/// decoded rank of every index is idx % nrank, and (b) drives the stronger
/// runtime claim: the rank that homes an index can create the DB with that GUID
/// and arts_guid_is_local agrees — the rank that does NOT home it sees it as
/// non-local.  This couples the GUID encoding to actual DB home placement.
///
/// Requires multi-node (rank_count >= 2).  Prints SKIP single-node.

#include "arts.h"
#include <stdint.h>

#define RR_PER_RANK 3 /* range size = RR_PER_RANK * nrank */

/// Per-rank checker.  paramv[0] = broadcast range GUID, paramv[1] = nrank.
/// Verifies home == idx % nrank for all indices, and is_local agreement +
/// local DB creation for the indices this rank homes.
static void checker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t range = (arts_guid_t)paramv[0];
  unsigned int nrank = (unsigned int)paramv[1];
  unsigned int me = arts_get_current_rank();
  unsigned int size = RR_PER_RANK * nrank;
  bool ok = true;

  for (unsigned int i = 0; i < size; i++) {
    arts_guid_t g = arts_guid_from_index(range, i);
    unsigned int expect_home = i % nrank;

    /* (a) decoded home matches idx % nrank. */
    if (arts_guid_get_rank(g) != expect_home) {
      arts_printf("  FAIL: rank %u idx %u home=%u expected %u\n", me, i,
                  arts_guid_get_rank(g), expect_home);
      ok = false;
      continue;
    }

    /* (b) is_local agrees with the computed home. */
    bool should_be_local = (expect_home == me);
    if (arts_guid_is_local(g) != should_be_local) {
      arts_printf("  FAIL: rank %u idx %u is_local=%d expected %d\n", me, i,
                  (int)arts_guid_is_local(g), (int)should_be_local);
      ok = false;
      continue;
    }

    /* (c) for the indices this rank homes, the DB created at that GUID lives
     * here — exercising the encoding->home placement coupling. */
    if (should_be_local) {
      uint64_t *ptr = (uint64_t *)arts_db_create_with_guid(
          g, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
      if (ptr == NULL) {
        arts_printf("  FAIL: rank %u idx %u local DB create returned NULL\n",
                    me, i);
        ok = false;
      } else {
        ptr[0] = (uint64_t)i;
        arts_db_release(g, DB_MODE_RW);
      }
    }
  }

  if (ok) {
    arts_printf("  PASS: rank %u home == idx %% nrank and home placement "
                "agrees for all %u indices\n",
                me, size);
  }
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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

  unsigned int nrank = arts_get_total_ranks();
  if (nrank < 2) {
    arts_printf(
        "SKIP: round_robin_home_distribution requires rank_count >= 2\n");
    arts_shutdown();
    return;
  }

  arts_printf("=== round_robin_home_distribution (%u ranks) ===\n", nrank);

  arts_guid_t range = arts_guid_reserve_range(ARTS_GUID_DB, RR_PER_RANK * nrank,
                                              ARTS_HINT_ROUND_ROBIN);
  if (range == NULL_GUID) {
    arts_printf("  FAIL: round_robin reserve_range returned NULL_GUID\n");
    arts_shutdown();
    return;
  }

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  uint64_t params[2] = {(uint64_t)range, (uint64_t)nrank};
  for (unsigned int r = 0; r < nrank; r++) {
    arts_edt_create(checker_edt, 2, params, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = fe});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
