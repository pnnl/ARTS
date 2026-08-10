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

/// @file guid_index_from_mismatch.c
/// @brief T040 — pin every -1 (mismatch) branch of arts_guid_index_from.
///
/// arts_guid_index_from is the inverse of arts_guid_from_index.  It must return
/// -1 (not a bogus index) on every kind of mismatch.  This test exercises each
/// rejection branch deterministically on a single node:
///   1. type mismatch (range type != query type)        — both range flavors
///   2. rank mismatch (non-distributed range)            — non-distributed
///   3. key-before-range-start (non-distributed range)   — non-distributed
///   4. key-before-base (distributed range)              — distributed
///   5. home >= nrank (distributed range)                — distributed
/// plus a positive round-trip control so a blanket "always -1" bug can't pass.

#include "arts.h"
#include "arts/gas/guid.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== guid_index_from_mismatch ===\n");
  unsigned int my_node = arts_get_current_rank();
  unsigned int nrank = arts_get_total_ranks();
  bool all_pass = true;

#define RANGE_SIZE 8

  /* A pinned (non-distributed) range on the local rank. */
  arts_guid_t db_range =
      arts_guid_reserve_range(ARTS_GUID_DB, RANGE_SIZE, my_node);
  /* A different-type range so we can build a type-mismatched query GUID. */
  arts_guid_t edt_range =
      arts_guid_reserve_range(ARTS_GUID_EDT, RANGE_SIZE, my_node);
  if (db_range == NULL_GUID || edt_range == NULL_GUID) {
    arts_printf("  FAIL: reserve_range returned NULL_GUID\n");
    arts_shutdown();
    return;
  }

  /* Positive control: a valid in-range GUID round-trips to its index. */
  arts_guid_t g3 = arts_guid_from_index(db_range, 3);
  if (arts_guid_index_from(db_range, g3) != 3) {
    arts_printf("  FAIL: positive control round-trip != 3\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: positive control round-trips (idx 3)\n");
  }

  /* Branch 1: type mismatch.  A GUID minted from an EDT range queried against a
   * DB range differs in the type field => -1. */
  arts_guid_t edt_g0 = arts_guid_from_index(edt_range, 0);
  if (arts_guid_index_from(db_range, edt_g0) != -1) {
    arts_printf("  FAIL: type mismatch did not return -1\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: type mismatch returns -1\n");
  }

  /* Branch 2 (non-distributed): rank mismatch.  Same type & key but a foreign
   * rank.  Pick a rank != my_node when one exists; otherwise synthesize one
   * directly so the branch is exercised even single-node. */
  {
    uint64_t key = ARTS_GUID_GET_KEY(db_range);
    unsigned int foreign_rank = (my_node == 0) ? 1u : 0u;
    arts_guid_t wrong_rank = ARTS_GUID_MAKE(ARTS_GUID_DB, foreign_rank, key);
    if (arts_guid_index_from(db_range, wrong_rank) != -1) {
      arts_printf("  FAIL: rank mismatch did not return -1\n");
      all_pass = false;
    } else {
      arts_printf("  PASS: rank mismatch returns -1\n");
    }
  }

  /* Branch 3 (non-distributed): key before range start.  Same type & rank but
   * a key strictly below the range's start key => -1. */
  {
    uint64_t start_key = ARTS_GUID_GET_KEY(db_range);
    if (start_key == 0) {
      arts_printf("  SKIP: range start key 0, cannot build a before-start "
                  "key (keys start at 1 by construction)\n");
    } else {
      arts_guid_t before = ARTS_GUID_MAKE(ARTS_GUID_DB, my_node, start_key - 1);
      if (arts_guid_index_from(db_range, before) != -1) {
        arts_printf("  FAIL: key-before-start did not return -1\n");
        all_pass = false;
      } else {
        arts_printf("  PASS: key-before-start returns -1\n");
      }
    }
  }

  /* Distributed branches: reserve a ROUND_ROBIN range. */
  arts_guid_t dist_range =
      arts_guid_reserve_range(ARTS_GUID_DB, RANGE_SIZE, ARTS_HINT_ROUND_ROBIN);
  if (dist_range == NULL_GUID) {
    arts_printf("  FAIL: distributed reserve_range returned NULL_GUID\n");
    all_pass = false;
  } else {
    uint64_t base_key = ARTS_GUID_GET_KEY(dist_range);

    /* Distributed positive control. */
    arts_guid_t d0 = arts_guid_from_index(dist_range, 0);
    if (arts_guid_index_from(dist_range, d0) != 0) {
      arts_printf("  FAIL: distributed positive control != 0\n");
      all_pass = false;
    } else {
      arts_printf("  PASS: distributed positive control round-trips (idx 0)\n");
    }

    /* Branch 4 (distributed): key before base => -1. */
    if (base_key == 0) {
      arts_printf("  SKIP: distributed base key 0, cannot build a "
                  "below-base key\n");
    } else {
      arts_guid_t below = ARTS_GUID_MAKE(ARTS_GUID_DB, 0u, base_key - 1);
      if (arts_guid_index_from(dist_range, below) != -1) {
        arts_printf("  FAIL: distributed key-before-base did not return -1\n");
        all_pass = false;
      } else {
        arts_printf("  PASS: distributed key-before-base returns -1\n");
      }
    }

    /* Branch 5 (distributed): home (decoded rank) >= nrank => -1.  Encode a
     * home one past the valid range; keep type & key valid so only the home
     * check can reject it. */
    {
      unsigned int bad_home = nrank; /* one past last valid home */
      arts_guid_t bad = ARTS_GUID_MAKE(ARTS_GUID_DB, bad_home, base_key);
      /* Guard: if bad_home accidentally equals the DISTRIBUTED sentinel the
       * query path would take the distributed-vs-distributed branch; nrank is
       * far below 0x3FFE so this never happens, but assert intent. */
      if ((unsigned int)ARTS_GUID_GET_RANK(bad) == ARTS_DISTRIBUTED_RANK) {
        arts_printf("  SKIP: synthesized home collided with DISTRIBUTED "
                    "sentinel\n");
      } else if (arts_guid_index_from(dist_range, bad) != -1) {
        arts_printf("  FAIL: distributed home>=nrank did not return -1\n");
        all_pass = false;
      } else {
        arts_printf("  PASS: distributed home>=nrank returns -1\n");
      }
    }
  }

  arts_printf("=== guid_index_from_mismatch: %s ===\n",
              all_pass ? "ALL PASSED" : "FAILED");
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
