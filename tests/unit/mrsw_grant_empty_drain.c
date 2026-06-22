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

/// @file mrsw_grant_empty_drain.c
/// @brief MRSW GRANT/CONFIRM_ACK orphan-token drop (config_specific).
///
/// arts_db_drain_pending_rw_after_grant pops EXACTLY ONE waiter on a fresh
/// install (EAGER OWNERSHIP_RESPONSE GRANT / LAZY CONFIRM_ACK).  If the queue
/// is EMPTY on arrival — the motivating waiter was already served by an earlier
/// hand-off, or it re-kicked and got served elsewhere — the install's token has
/// no writer to account; the orphan token (writer_count stuck at 2) must be
/// dropped via release_rw_local so the chain does not stall.
///
/// Construction: a remote rank repeatedly RW-acquires the home DB.  Per round
/// we fire TWO RW EDTs on the SAME remote rank ordered by the DB RW chain.  The
/// first acquire triggers the ownership round (home->remote transfer); by the
/// time the GRANT install drains, the second RW EDT may or may not have pushed
/// yet.  When the install drains an empty queue (first waiter already run on a
/// prior round's residual token), the orphan-token drop must keep the second
/// waiter reachable.  Across many rounds every RW EDT must run exactly once;
/// final == 2*ROUNDS.  A dropped/duplicated token -> wrong count or hang
/// (ctest TIMEOUT).
///
/// On 1n there is no ownership transfer (no GRANT) and the test passes
/// trivially via the local token hand-off.  MRSW-only.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW)

int main(void) {
  printf("SKIP mrsw_grant_empty_drain: MRSW-only\n");
  return 0;
}

#else

#define ROUNDS 200u

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

static void final_check_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t v = (d != NULL) ? d[0] : 0u;
  uint64_t expect = 2ull * ROUNDS;
  if (v != expect) {
    (void)fprintf(stderr,
                  "FAIL: final value %llu != %llu (orphan-token drop "
                  "lost/duplicated a waiter)\n",
                  (unsigned long long)v, (unsigned long long)expect);
    arts_abort(1);
  }
  arts_printf("PASS: mrsw_grant_empty_drain final=%llu\n",
              (unsigned long long)expect);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_grant_empty_drain ===\n");

  unsigned int nranks = arts_get_total_ranks();
  unsigned int remote = (nranks > 1) ? 1u : 0u; /* prefer a non-home rank */

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: two RW EDTs per round on the remote rank: the first drives the
   * home->remote ownership round; the second's push races the GRANT install's
   * single-waiter drain, exercising the empty-queue orphan-token drop.  Waiting
   * on the finish event blocks main_edt until every writer has run AND written
   * back. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int i = 0; i < ROUNDS; i++) {
    arts_guid_t a =
        arts_edt_create(rw_inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = remote, .finish_event = fe});
    arts_add_dependence(db, a, 0, DB_MODE_RW);

    arts_guid_t b =
        arts_edt_create(rw_inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = remote, .finish_event = fe});
    arts_add_dependence(db, b, 0, DB_MODE_RW);
  }
  arts_event_wait(fe); /* blocks until ALL writers ran + wrote back */

  /* Phase 2: the final checker, created only now that phase 1 has fully
   * quiesced, so its RO dependency is registered after the committed value is
   * in place and its snapshot observes every increment. */
  arts_guid_t fe2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t fin =
      arts_edt_create(final_check_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe2});
  arts_add_dependence(db, fin, 0, DB_MODE_RO);
  arts_event_wait(fe2);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW */
