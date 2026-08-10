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

/// @file guid_crossrank_determinism.c
/// @brief T037 — a ROUND_ROBIN range GUID reserved on rank 0 and broadcast to
///        every rank yields the IDENTICAL per-index GUID on every rank, even
///        after each rank has perturbed its own per-(rank,kind) auto-counters
///        by a different amount.
///
/// arts_guid_from_index for a distributed range is a pure function of the
/// range's encoded base key and arts_global_rank_count — both rank-independent
/// given the same range GUID — so home = idx % nrank and key = base + idx/nrank
/// must be the same number on every rank.  Determinism is the property under
/// test: each verifier rank perturbs its local counters first (different per
/// rank) and then recomputes the mapping; if from_index leaked any local
/// counter state the homes / round-trips would diverge between ranks.
///
/// Requires multi-node (rank_count >= 2).  Prints SKIP single-node.

#include "arts.h"
#include <stdint.h>

#define RR_PER_RANK 5 /* indices per rank -> range size = RR_PER_RANK * nrank  \
                       */

/// Per-rank verifier.  paramv[0] = broadcast range GUID, paramv[1] = nrank.
/// First perturbs this rank's auto-counters by a rank-dependent amount, then
/// verifies the distributed mapping and round-trip for every index.
static void verifier_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t range = (arts_guid_t)paramv[0];
  unsigned int nrank = (unsigned int)paramv[1];
  unsigned int me = arts_get_current_rank();

  /* Perturb local counter prehistory by a rank-dependent amount.  This must NOT
   * change what from_index(range, idx) returns. */
  for (unsigned int p = 0; p < (me + 1) * 3; p++) {
    (void)arts_guid_reserve(ARTS_GUID_DB, me);
  }

  unsigned int size = RR_PER_RANK * nrank;
  bool ok = true;
  for (unsigned int i = 0; i < size; i++) {
    arts_guid_t g = arts_guid_from_index(range, i);
    unsigned int expect_home = i % nrank;
    if (arts_guid_get_rank(g) != expect_home) {
      arts_printf("  FAIL: rank %u idx %u home=%u expected %u\n", me, i,
                  arts_guid_get_rank(g), expect_home);
      ok = false;
    }
    if (arts_guid_get_kind(g) != ARTS_GUID_DB) {
      arts_printf("  FAIL: rank %u idx %u kind mismatch\n", me, i);
      ok = false;
    }
    if (arts_guid_index_from(range, g) != (int)i) {
      arts_printf("  FAIL: rank %u idx %u did not round-trip\n", me, i);
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: rank %u recovered identical (home, key) mapping for "
                "all %u indices\n",
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
    arts_printf("SKIP: guid_crossrank_determinism requires rank_count >= 2\n");
    arts_shutdown();
    return;
  }

  arts_printf("=== guid_crossrank_determinism (%u ranks) ===\n", nrank);

  /* Reserve the distributed range ONCE on rank 0 and broadcast the start GUID
   * verbatim (via EDT paramv) to a verifier on every rank.  Because the range
   * GUID is identical on every rank, from_index must produce identical results
   * everywhere. */
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
    arts_edt_create(verifier_edt, 2, params, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = fe});
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
