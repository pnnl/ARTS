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

/// @file mrsw_eager_invalidate_during_install.c
/// @brief MRSW EAGER stale-INVALIDATE-during-install hazard (config_specific).
///
/// EXPOSES a suspected runtime bug (census 08 §4, "EAGER no-INVALIDATE-during-
/// install assumption"; SUSPECTED-BUGS B-eager-no-invalidate-assumption).
///
/// arts_handler_db_ownership_response (EAGER GRANT install) does add(writer_-
/// count, 2) 0->2 and COMMENTS that no INVALIDATE can target this rank before
/// CONFIRM advances rw_holder, so the count holds >= 2 across the body.  EAGER
/// lacks LAZY's ownership_unconfirmed guard.  If a stale/duplicate INVALIDATE
/// (a prior round's, or a self-send reorder) lands mid-install, sub(writer_-
/// count, 1) drives the count to 1 mid-install and the subsequent add(2)
/// over-counts -> a phantom owned epoch / lost-or-double writer account.
///
/// This is a CORRECT race-free program: a single RW write chain ping-ponging
/// ownership rapidly between ranks (each writer +1), under heavy concurrent
/// ownership churn so the home directory issues back-to-back
/// INVALIDATE/transfer rounds — the interleaving that can deliver an
/// out-of-round INVALIDATE during a fresh install.  The final value MUST equal
/// the writer count.  If the over-count fires, the value is wrong (->
/// arts_abort, ctest FAIL) or a writer is stranded (-> ctest TIMEOUT).  No PASS
/// regex: this test is expected to FAIL until the install boundary is guarded,
/// and MUST NOT be weakened to pass.
///
/// EAGER + MRSW only.  Requires >= 2 ranks (more ranks = more churn); SKIPs on
/// 1n (no transfer, the bug cannot manifest).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW) || !defined(ARTS_TIMING_EAGER)

int main(void) {
  printf("SKIP mrsw_eager_invalidate_during_install: MRSW+EAGER-only\n");
  return 0;
}

#else

#define ROUNDS 60u

static void rw_inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

static unsigned int g_total = 0u;

static void final_check_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v != (uint64_t)g_total) {
    (void)fprintf(
        stderr, "FAIL: final value %llu != %u (stale INVALIDATE over-count)\n",
        (unsigned long long)v, g_total);
    arts_abort(1);
  }
  /* No PASS-regex gate: this test exposes a suspected runtime bug.  Reaching
   * here with the correct value is the desired-but-not-yet-guaranteed outcome.
   */
  arts_printf("mrsw_eager_invalidate_during_install: final=%u\n", g_total);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_eager_invalidate_during_install ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2u) {
    arts_printf(
        "SKIP: mrsw_eager_invalidate_during_install requires >= 2 ranks\n");
    arts_shutdown();
    return;
  }

  g_total = ROUNDS * nranks;

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* final_check reads depv[0].ptr (db) -> db is slot 0 (RO); the writers'
   * finish event fe is slot 1 (NULL).  depc=2 so the passive immediate db
   * satisfy does NOT fire final_check early — it also waits for fe (every RW
   * writer done).  (Both deps at slot 0 with depc=1 fired final_check before
   * any writer ran, masking the install-boundary scenario this test targets.)
   */
  arts_guid_t fin = arts_edt_create(final_check_edt, 0, NULL, 2,
                                    &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(db, fin, 0, DB_MODE_RO);
  arts_add_dependence(fe, fin, 1, DB_MODE_NULL);

  /* Ping-pong ownership rank-by-rank, ROUNDS times, with TWO RW EDTs queued per
   * (round,rank) so the home directory has a back-to-back transfer queued — the
   * second round's INVALIDATE can race the first's GRANT install. */
  for (unsigned int r = 0; r < ROUNDS; r++) {
    for (unsigned int k = 0; k < nranks; k++) {
      arts_guid_t w =
          arts_edt_create(rw_inc_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = k, .finish_event = fe});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW && ARTS_TIMING_EAGER */
