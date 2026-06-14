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

/// @file coherence_rw_multihop.c
/// @brief Regression: many ranks (>=3) RW-acquire the SAME DataBlock, forming a
///        multi-hop ownership-transfer chain on one DB.  This stresses the
///        home-side ownership-transfer chain (LOCK_REQ -> INVALIDATE -> GRANT
///        -> transfer) when more than two ranks contend for one DB — the path
///        where a chain-continuation decision can race a late LOCK_REQ and
///        strand an ownership-transfer waiter.
///
///        Structure (two phases so the value check is race-free):
///          Phase 1: N concurrent RW incrementers, gated ONLY on the DB, all in
///                   one finish scope.  Concurrency is essential — it produces
///                   multiple simultaneous LOCK_REQs at home, which is what
///                   exercises the chain.  arts_event_wait blocks until every
///                   incrementer has run AND written back; a stranded waiter
///                   therefore shows up as the finish scope never quiescing
///                   (caught by the watchdog).
///          Phase 2: a single RO reader, created only AFTER phase 1 has fully
///                   quiesced, so its snapshot deterministically observes every
///                   increment (sum == rank count).
///        Requires 3+ ranks.

#include "arts.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>

/// 1 = the RO reader saw every increment; -1 = wrong value; 0 = never ran.
static atomic_int g_check_result = 0;
/// Set once both phases finish so the watchdog exits quietly on success.
static atomic_int g_finished = 0;

/// Watchdog: if the multi-hop chain strands a waiter, phase 1's finish scope
/// never quiesces and arts_event_wait blocks forever.  Fail loudly rather than
/// hang.
static void *wd_thread(void *arg) {
  unsigned int nranks = (unsigned int)(uintptr_t)arg;
  for (int slept = 0; slept < 20; slept++) {
    sleep(1);
    if (atomic_load(&g_finished)) {
      return NULL;
    }
  }
  (void)fprintf(
      stderr,
      "HANG: multi-hop RW chain over %u ranks did not complete in 20s "
      "(ownership-transfer waiter stranded)\n",
      nranks);
  (void)fflush(stderr);
  _exit(1);
}

/// RW incrementer: data[0]++ under exclusive (per-node) RW access.
void inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL: inc_edt got NULL ptr\n");
    arts_abort(1);
  }
  data[0] = data[0] + 1;
}

/// Final RO reader: assert every rank's increment landed (sum == rank count).
void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  int expected = (int)paramv[0];
  if (data != NULL && data[0] == expected) {
    atomic_store(&g_check_result, 1);
    arts_printf("PASS: multi-hop RW chain over %d ranks summed to %d\n",
                expected, expected);
  } else {
    atomic_store(&g_check_result, -1);
    (void)fprintf(stderr, "FAIL: multi-hop RW chain expected %d got %d\n",
                  expected, data ? data[0] : -1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_rw_multihop ===\n");

#ifdef ARTS_PROTOCOL_MRMW
  /* The relaxed (DB-DRF) model unifies RW with RO (concurrent replicas,
   * reduce on release) and has no ownership-transfer chain (no
   * LOCK_REQ/INVALIDATE/GRANT).  A plain serial increment is therefore not a
   * meaningful relaxed-model workload — concurrent acquirers race the
   * read-modify-write.  This test targets the OCR-model (eager/lazy)
   * exclusive-RW ownership-transfer chain. */
  arts_printf("SKIP: RELAXED has no exclusive-RW ownership-transfer chain\n");
  atomic_store(&g_check_result, 1);
  arts_shutdown();
  return;
#endif

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 3) {
    arts_printf("SKIP: requires 3+ ranks (got %u)\n", nranks);
    atomic_store(&g_check_result, 1); /* not under test at <3 ranks */
    arts_shutdown();
    return;
  }

  pthread_t wdt;
  pthread_create(&wdt, NULL, wd_thread, (void *)(uintptr_t)nranks);
  pthread_detach(wdt);

  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                                  &(arts_db_hint_t){.rank = 0});
  ((int *)ptr)[0] = 0;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: one RW incrementer per rank, all depending on the SAME db → the
   * coherence layer serializes them into a multi-hop ownership-transfer chain.
   * They are mutually unordered (gated only on the DB), so multiple LOCK_REQs
   * reach home concurrently — exactly the chain-contention the bug needs. */
  arts_guid_t e1 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int r = 0; r < nranks; r++) {
    arts_guid_t w = arts_edt_create(
        inc_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = r, .finish_event = e1});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
  }
  arts_event_wait(e1); /* blocks until ALL incs ran + wrote back */

  /* Phase 2: RO reader, created only now that phase 1 has fully quiesced, so
   * its snapshot deterministically observes every increment. */
  uint64_t expected = (uint64_t)nranks;
  arts_guid_t e2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t chk =
      arts_edt_create(check_edt, 1, &expected, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e2});
  arts_add_dependence(db, chk, 0, DB_MODE_RO);
  arts_event_wait(e2);

  atomic_store(&g_finished, 1);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  if (arts_get_current_rank() == 0 && atomic_load(&g_check_result) != 1) {
    (void)fprintf(stderr,
                  "FAIL: multi-hop RW chain check did not pass (result=%d)\n",
                  atomic_load(&g_check_result));
    return 1;
  }
  return 0;
}
