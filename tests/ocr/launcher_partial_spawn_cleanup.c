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

/// @file launcher_partial_spawn_cleanup.c
/// @brief Validates the launcher's child-index <-> rank mapping that cleanup
///        relies on (the i+1 rank mapping; partial-spawn consistency).
///
/// In the local launcher, the spawn loop runs `i = 1 .. table_length-1` and
/// records each successful fork into `child_pids[child_count++]`.  child_count
/// is advanced ONLY on a successful fork, so `child_pids[k]` always corresponds
/// to spawned rank `k+1` and the cleanup ladder's "rank i+1" log/escalation is
/// correct even if some fork failed (a failed fork does not advance the index,
/// keeping the index->rank mapping contiguous and gap-free over the ranks that
/// actually came up).
///
/// The runtime-observable invariant that this consistency guarantees: the set
/// of live ranks is exactly {0, 1, ..., total-1} with NO gaps, and every rank's
/// ARTS_RANK identity equals its runtime rank.  A broken index<->rank mapping
/// would surface as a missing rank (an EDT targeted at it never completes ->
/// caught by TIMEOUT) or a rank whose ARTS_RANK disagrees with its runtime
/// rank.
///
/// This test stamps a per-rank marker into a shared result DB from each rank,
/// then verifies on the home rank that every slot 0..total-1 was filled exactly
/// once with the matching rank id — proving the contiguous, gap-free mapping.
/// Finish events gate completion; a stranded rank is caught by the ctest
/// TIMEOUT. Protocol-agnostic: launcher carries no coherence state, runs in all
/// configs.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MARKER_BASE 0x5A5A0000u

/// Per-rank marker writer: stamps (MARKER_BASE | rank) into result[rank] and
/// cross-checks ARTS_RANK against the runtime rank (the index->rank contract).
void marker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int expected_rank = (unsigned int)paramv[0];
  unsigned int me = arts_get_current_rank();
  uint32_t *result = (uint32_t *)depv[0].ptr;

  if (me != expected_rank) {
    arts_printf("FAIL: marker ran on rank %u, expected %u\n", me,
                expected_rank);
    arts_abort(1);
    return;
  }

  /* On non-master ranks, the ARTS_RANK env (set per spawned child as the loop
   * index i) must equal the runtime rank — this is exactly the index->rank
   * mapping cleanup depends on. */
  if (me != 0) {
    const char *rank_env = getenv("ARTS_RANK");
    if (rank_env == NULL || (unsigned int)strtoul(rank_env, NULL, 10) != me) {
      arts_printf("FAIL: rank %u ARTS_RANK=%s != runtime rank\n", me,
                  rank_env ? rank_env : "(null)");
      arts_abort(1);
      return;
    }
  }

  if (result != NULL) {
    result[me] = MARKER_BASE | me;
  }
  arts_printf("  rank %u stamped marker 0x%x\n", me, MARKER_BASE | me);
}

/// Home-side verifier: every slot 0..total-1 carries its own rank's marker.
void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int total = (unsigned int)paramv[0];
  const uint32_t *result = (const uint32_t *)depv[0].ptr;

  if (result == NULL) {
    arts_printf("FAIL: verifier got NULL result DB\n");
    arts_abort(1);
    return;
  }
  for (unsigned int r = 0; r < total; r++) {
    if (result[r] != (MARKER_BASE | r)) {
      arts_printf("FAIL: slot %u = 0x%x, expected 0x%x (gap/mismatch)\n", r,
                  result[r], MARKER_BASE | r);
      arts_abort(1);
      return;
    }
  }
  arts_printf("PASS: launcher_partial_spawn_cleanup (%u ranks, contiguous "
              "index<->rank mapping)\n",
              total);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== launcher_partial_spawn_cleanup ===\n");

  unsigned int total = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, total * sizeof(uint32_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  for (unsigned int r = 0; r < total; r++) {
    ((uint32_t *)ptr)[r] = 0u;
  }
  arts_db_release(db, DB_MODE_RW);

  /* Each rank stamps its own slot.  Serialize via per-rank finish events so the
   * writes are causally ordered before the verifier's RO read. */
  for (unsigned int r = 0; r < total; r++) {
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t params[1] = {(uint64_t)r};
    arts_guid_t e =
        arts_edt_create(marker_edt, 1, params, 1,
                        &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_add_dependence(db, e, 0, DB_MODE_RW);
    arts_event_wait(fe);
  }

  arts_guid_t vfe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  uint64_t vparams[1] = {(uint64_t)total};
  arts_guid_t v =
      arts_edt_create(verify_edt, 1, vparams, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = vfe});
  arts_add_dependence(db, v, 0, DB_MODE_RO);
  arts_event_wait(vfe);

  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
