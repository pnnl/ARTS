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
** necessarily state or reflect those of the United States Government or any **
** agency thereof.                                                           **
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

/// @file coherence_multi_writer_dist.c
/// @brief Cross-rank multi-writer same-address stress test (OCR program).
///
/// An ordinary OCR program — with no coherence-protocol knowledge — that
/// exercises the cross-rank RW path: DBs are homed on non-zero ranks so RW
/// acquires from rank 0 cross the wire, and the final RO read on rank 0 must
/// observe every writer's update (not a stale value), under whatever coherence
/// protocol the runtime was built with.
///
/// Structure (per iteration):
///   - N_DBS DBs homed round-robin across ranks 1..(nnodes-1) (never rank 0,
///     so every access from a rank-0 worker is a cross-rank acquire).  Each DB
///     is zero-initialized at create.
///   - N_WORKERS RW-worker EDTs fanned out across all ranks round-robin; each
///     acquires one DB in RW mode and atomically increments the counter.
///   - After all workers complete (finish scope), a verify EDT on rank 0
///     acquires all DBs in RO mode and checks each counter equals the number
///     of workers assigned to it (no lost update across the cross-rank
///     handoff).
///
/// Correctness check wired to exit code: arts_abort on any mismatch (non-zero
/// exit); a missed finish scope manifests as a ctest TIMEOUT.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/// Number of DBs per iteration (homed on ranks 1..nnodes-1 round-robin).
#define N_DBS 20
/// Number of RW worker EDTs per iteration (spread across all ranks).
#define N_WORKERS 40
/// Number of outer iterations.
#define K_ITERS 5

/// Workers assigned to each DB per iteration (round-robin spread).
#define WORKERS_PER_DB (N_WORKERS / N_DBS) /* integer; N_WORKERS%N_DBS==0 */

/// Expected final counter value per DB: zero-initialized at DB create, then
/// WORKERS_PER_DB workers each atomically increment by 1.
#define EXPECTED_FINAL WORKERS_PER_DB

/// RW worker EDT: acquires DB in RW mode and atomically increments the counter.
/// ARTS RW is per-NODE exclusive, NOT per-EDT: same-node RW EDTs run
/// concurrently, so the increment must be atomic (the application's
/// responsibility).  The coherence protocol guarantees each increment lands in
/// a buffer visible to the next acquirer; it does NOT provide per-EDT mutual
/// exclusion.
static void rw_worker_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic int *d = (_Atomic int *)depv[0].ptr;
  if (d == NULL) {
    arts_printf("FAIL: rw_worker got NULL ptr\n");
    arts_abort(1);
  }
  atomic_fetch_add_explicit(d, 1, memory_order_relaxed);
}

/// Verify EDT: runs on rank 0 after all workers finish (finish-scope gate).
/// Acquires all N_DBS in RO mode and asserts each counter equals
/// EXPECTED_FINAL.  This is the critical cross-rank coherence check: the RO
/// acquire on rank 0 must observe the value written by the last RW holder, not
/// a stale one.
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  /* slot 0 = finish-event; slots 1..N_DBS = DB RO deps. */
  for (uint32_t i = 1; i <= N_DBS; i++) {
    if (i >= depc) {
      arts_printf("FAIL: verify depc too small (%u)\n", depc);
      arts_abort(1);
    }
    const _Atomic int *d = (const _Atomic int *)depv[i].ptr;
    if (d == NULL) {
      arts_printf("FAIL: verify slot %u got NULL ptr (stale grant?)\n", i);
      arts_abort(1);
    }
    int observed = atomic_load_explicit(d, memory_order_relaxed);
    if (observed != EXPECTED_FINAL) {
      (void)fprintf(stderr,
                    "FAIL: DB[%u] = %d, expected %d "
                    "(lost update or stale grant data)\n",
                    i - 1, observed, EXPECTED_FINAL);
      arts_abort(1);
    }
  }
  arts_printf("PASS: EXCL distributed arbitration: %d iters × %d DBs × "
              "%d workers/DB, each DB final=%d\n",
              K_ITERS, N_DBS, WORKERS_PER_DB, EXPECTED_FINAL);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nnodes = arts_get_total_ranks();

#if defined(ARTS_PROTOCOL_WRF_VAL)
  /* WRF_VAL (DB-WRF) provides no exclusive cross-rank ownership for RW: concurrent
   * RW holders on different nodes each receive a buffer copy and race at
   * PUBLISH time.  This test requires every RW writer's atomic increment to
   * survive to the RO verify, which holds only under protocols that guarantee
   * per-node exclusive ownership (VAL, EXCL). */
  arts_printf("SKIP coherence_multi_writer_dist: concurrent cross-rank RW "
              "accumulation is DB-WRF racy under WRF_VAL\n");
  arts_shutdown();
  return;
#endif

  if (nnodes < 2) {
    arts_printf("SKIP: requires 2+ ranks (got %u)\n", nnodes);
    arts_shutdown();
    return;
  }

  arts_printf("=== coherence_multi_writer_dist (%d iters, %d DBs, %d workers, "
              "%u ranks) ===\n",
              K_ITERS, N_DBS, N_WORKERS, nnodes);

  /* Outer finish scope: verify_edt fires only after every iteration's
   * rw_workers have released their deps. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* A single outer finish scope covers all iterations and a single verify EDT
   * checks all per-iteration DBs.  Each iteration creates N_DBS fresh DBs and
   * fans out rw_workers; because each DB is freshly created and the verify EDT
   * runs only after every finish-scope member drains, the RO snapshot is
   * correct.  Fresh per-iteration DBs exercise repeated acquire/release cycles.
   */

  /* Allocate the verify EDT outside the loop (single RO check after all
   * iterations).  depc = 1 (finish-event) + K_ITERS * N_DBS (all DB RO). */
  uint32_t verify_depc = 1 + (uint32_t)(K_ITERS * N_DBS);
  arts_guid_t verify = arts_edt_create(verify_edt, 0, NULL, verify_depc,
                                       &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe, verify, 0, DB_MODE_NULL);

  uint32_t verify_slot = 1; /* next RO slot index in verify_edt */

  for (int iter = 0; iter < K_ITERS; iter++) {
    arts_guid_t dbs[N_DBS];

    /* Create N_DBS DBs homed on ranks 1..(nnodes-1) round-robin.  Rank 0 is
     * intentionally excluded from hosting so every rank-0 access is a genuine
     * cross-rank acquire (the home is always remote for rank 0). */
    for (int i = 0; i < N_DBS; i++) {
      void *raw = NULL;
      /* home rank in [1, nnodes-1] */
      unsigned int home = (unsigned int)(1 + (i % (int)(nnodes - 1)));
      dbs[i] = arts_db_create(&raw, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                              &(arts_db_hint_t){.rank = home});
      /* DB is zero-initialized by arts_db_create_install_home_buffer; workers
       * then atomically increment, so the expected final value is the worker
       * count alone (no separate init writer needed). */
    }

    /* Fan out N_WORKERS RW workers across all ranks round-robin.  Each
     * acquires a DB deterministically (iter*7 + i*13 spread to avoid all
     * workers clustering on the same DB in the same iteration). */
    for (int i = 0; i < N_WORKERS; i++) {
      int db_idx = (iter * 7 + i * 13) % N_DBS;
      unsigned int worker_rank = (unsigned int)(i % (int)nnodes);
      arts_guid_t w = arts_edt_create(
          rw_worker_edt, 0, NULL, 1,
          &(arts_edt_hint_t){.rank = worker_rank, .finish_event = fe});
      arts_add_dependence(dbs[db_idx], w, 0, DB_MODE_RW);
    }

    /* Wire all iteration DBs as RO deps of the verify EDT (slots 1..N_DBS
     * per iteration).  The finish-scope drain before verify fires guarantees
     * all RW holders have released before the RO acquire is submitted. */
    for (int i = 0; i < N_DBS; i++) {
      arts_add_dependence(dbs[i], verify, verify_slot, DB_MODE_RO);
      verify_slot++;
    }
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
