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

/// @file mrsw_release_dekker_race.c
/// @brief MRSW release-time Dekker reclaim micro-window stress
/// (config_specific).
///
/// arts_db_release_rw_local, on an empty pop, post-decrements writer_count.
/// When the post-value is 1 (idle owner, still holds sentinel) it runs a LOCAL
/// Dekker re-check: a fresh acquirer can push a waiter into pending_rw between
/// the releaser's pop-empty and its sub; that acquirer read the pre-sub
/// positive count, returned "active writer present", and expects US (the
/// releaser) to pop it.  The re-check (!peek_empty -> cswap 1->2 -> run_one)
/// reclaims the token and runs the raced-in waiter.  A lost-wakeup here
/// (peek_empty false-empty during the producer mid-link, or a 3-way interleave
/// with a concurrent idle claim) strands a writer: it never runs, the latch
/// never fires, and the counter ends below the serial sum.
///
/// Construction: many worker threads on a single rank, each RW-acquiring the
/// SAME DB on the home rank with NO ordering between them (a flat fan, not a
/// chain).  This maximizes the rate of "releaser pops empty just as the next
/// acquirer pushes" windows.  Each writer increments the counter exactly once;
/// the final value MUST equal N (serial sum of N increments of 1).  A stranded
/// writer -> final < N -> arts_abort.  A hang -> ctest TIMEOUT.
///
/// MRSW-only (the Dekker re-check is MRSW ownership.c machinery).  config-
/// agnostic for the rank where the DB lives; home pinned to rank 0.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW)

int main(void) {
  printf("SKIP mrsw_release_dekker_race: MRSW-only\n");
  return 0;
}

#else

#define N_WRITERS 256u

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
  if (v != (uint64_t)N_WRITERS) {
    (void)fprintf(
        stderr, "FAIL: final value %llu != %u (stranded writer / lost token)\n",
        (unsigned long long)v, N_WRITERS);
    arts_abort(1);
  }
  arts_printf("PASS: mrsw_release_dekker_race final=%u\n", N_WRITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_release_dekker_race ===\n");

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: a flat fan of N RW writers, all on the home rank, NO inter-writer
   * ordering: the single-writer token funnels them one at a time, hammering the
   * pop-empty/sub-vs-push window in release_rw_local.  Waiting on the finish
   * event blocks main_edt until every writer has run AND written back. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int i = 0; i < N_WRITERS; i++) {
    arts_guid_t w =
        arts_edt_create(rw_inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
  }
  arts_event_wait(fe); /* blocks until ALL writers ran + wrote back */

  /* Phase 2: the final checker, created only now that phase 1 has fully
   * quiesced, so its RO dependency is registered after the committed value is
   * in place and its snapshot observes the full serial sum. */
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
