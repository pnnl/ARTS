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

/// @file mrsw_owner_transfer_churn.c
/// @brief MRSW cross-rank ownership transfer churn (runtime_multinode).
///
/// A ring of ranks each take RW ownership of a single home DB in turn, with
/// concurrent RO readers fanned in between writes.  This drives the home
/// OWNERSHIP_REQUEST FIFO + the invalidate_in_flight baton + the owner->owner
/// ship (OWNERSHIP_RESPONSE) + CONFIRM(_ACK) round, repeatedly, while RO
/// readers exercise the snapshot/redirect path concurrently.  The
/// stranded-waiter re-request tail in send_ownership_response (a waiter that
/// raced into pending_rw on a rank losing ownership re-kicks OWNERSHIP_REQUEST)
/// must keep every RW reachable.
///
/// Each RW EDT increments the counter by exactly 1, ordered by the DB RW chain
/// (a single happens-before line across the ring).  After ROUNDS*nranks RW
/// increments the final value MUST equal ROUNDS*nranks.  A lost transfer /
/// stranded baton -> wrong count or a hang (ctest TIMEOUT).  RO readers assert
/// the value stays within [0, total] (no torn / out-of-range read).
///
/// Requires >= 2 ranks; SKIPs cleanly on 1n.  MRSW-only (self-skips at compile
/// time under other protocols).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW)

int main(void) {
  printf("SKIP mrsw_owner_transfer_churn: MRSW-only\n");
  return 0;
}

#else

#define ROUNDS 40u
#define N_RO_PER_WRITE 2u

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

static void ro_check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t hi = paramv[0];
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v > hi) {
    (void)fprintf(stderr, "FAIL: RO read %llu > total %llu (torn transfer)\n",
                  (unsigned long long)v, (unsigned long long)hi);
    arts_abort(1);
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
    (void)fprintf(stderr, "FAIL: final value %llu != %u (lost transfer)\n",
                  (unsigned long long)v, g_total);
    arts_abort(1);
  }
  arts_printf("PASS: mrsw_owner_transfer_churn final=%u\n", g_total);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_owner_transfer_churn ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2u) {
    arts_printf("SKIP: mrsw_owner_transfer_churn requires >= 2 ranks\n");
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

  /* Phase 1: ring — round r, step k -> RW on rank k.  The DB RW chain
   * serializes them in creation order, so the home directory walks rank
   * 0,1,..,n-1,0,1,.. issuing an ownership transfer each step; concurrent RO
   * readers (which legitimately race mid-flight values) are fanned in between.
   * Waiting on the finish event blocks main_edt until every writer has run AND
   * written back. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int r = 0; r < ROUNDS; r++) {
    for (unsigned int k = 0; k < nranks; k++) {
      arts_guid_t w =
          arts_edt_create(rw_inc_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = k, .finish_event = fe});
      arts_add_dependence(db, w, 0, DB_MODE_RW);

      /* Concurrent RO fan on assorted ranks between writes. */
      for (unsigned int j = 0; j < N_RO_PER_WRITE; j++) {
        uint64_t hi = (uint64_t)g_total;
        arts_guid_t ro =
            arts_edt_create(ro_check_edt, 1, &hi, 1,
                            &(arts_edt_hint_t){.rank = (k + j + 1u) % nranks,
                                               .finish_event = fe});
        arts_add_dependence(db, ro, 0, DB_MODE_RO);
      }
    }
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
