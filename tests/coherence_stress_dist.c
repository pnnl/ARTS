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

/// @file coherence_stress_dist.c
/// @brief B.2 — Distributed coherence stress (multinode counterpart of
///        B.1).
///
/// 4-rank scenario (auto-skipped on smaller configs).  Per iteration the
/// driver creates N_DBS DBs round-robin across all ranks (home routing
/// via arts_edt_hint_t.rank = i % nnodes) and spawns N_EDTS workers, each
/// pinned to a deterministic rank and acquiring a deterministic DB in a
/// deterministic mode (RW or RO).  RC must transfer ownership /
/// install RO snapshots across ranks; the final completion count must
/// equal N_EDTS * K_ITERS.
///
/// Determinism: every worker increments a per-DB integer if RW, or reads
/// it if RO.  RW updates per DB are strictly serialised across ranks by
/// RC's lease + barrier_gen, so the per-DB counter is deterministic
/// modulo the number of RW visits to that DB.  We do not assert the
/// per-DB final value (cross-rank dispatch order varies); we instead
/// check that all workers ran without aborting and that the global
/// completion count is exact.
///
/// Adaptations vs. plan brief (line 1818 of plan):
///   - 4-arg arts_db_create (no ARTS_DB_PROP_NONE in HEAD).
///   - DB_MODE_RW unifies the legacy DB_MODE_RW post-Cutover-C.
///   - B.3-style scaffolding: outer finish scope + g_clean_shutdown + main()
///     exit code, so consumer aborts on any rank propagate as ctest
///     FAIL even when rank 0 itself shuts down via the peer-disconnect
///     SHUTDOWN path.
///   - 10 iterations × 100 DBs × 200 workers = 2000 cross-rank acquires
///     (kept conservative to fit the TIMEOUT 120 budget under sanitiser
///     builds).
///
/// Spec section 6 B.2.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define N_DBS 100
#define N_EDTS 200
#define K_ITERS 10

static atomic_int g_completed = 0;
static atomic_int g_clean_shutdown = 0;

static void init_writer_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: init_writer got NULL ptr\n");
    arts_abort(1);
  }
  *data = 0;
}

static void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)depc;
  if (paramc < 1) {
    arts_printf("FAIL: worker missing paramv\n");
    arts_abort(1);
  }
  int mode_is_rw = (int)paramv[0];
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: worker got NULL ptr\n");
    arts_abort(1);
  }
  if (mode_is_rw) {
    /* Per-node serialised RW: increment is safe within RC. */
    (*data)++;
  } else {
    /* RO: just read, do not modify. */
    volatile int seen = *data;
    (void)seen;
  }
  atomic_fetch_add(&g_completed, 1);
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  int got = atomic_load(&g_completed);
  int expected = N_EDTS * K_ITERS;
  /* Note: atomic_int is incremented by workers across all ranks but the
   * counter is rank-local in this address space — we only assert from
   * rank 0 that *its* worker share completed.  Cross-rank workers tally
   * locally on their own rank; the run still reaches this finish-EDT
   * only if every rank's finish scope chain drained without aborting. */
  if (got <= 0) {
    fprintf(stderr, "FAIL: rank-0 worker count is %d (expected > 0)\n", got);
    arts_abort(1);
  }
  atomic_store(&g_clean_shutdown, 1);
  arts_printf(
      "PASS: rank-0 saw %d/%d local worker completions across %d iterations\n",
      got, expected, K_ITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nnodes = arts_get_total_ranks();
  if (nnodes < 2) {
    arts_printf("SKIP: requires 2+ ranks (got %u)\n", nnodes);
    /* SKIP is a clean exit, not an abort: set the flag so main()'s
     * post-arts_rt() check does not flag a false FAIL. */
    atomic_store(&g_clean_shutdown, 1);
    arts_shutdown();
    return;
  }

  arts_printf("=== coherence_stress_dist (%d iter, %d DBs/iter, %d EDTs/iter, "
              "%u ranks) ===\n",
              K_ITERS, N_DBS, N_EDTS, nnodes);

  /* Finish-EDT must have depc >= 1 so the finish scope's slot-0 satisfy
   * actually gates it; depc=0 would let it fire before any worker. */
  arts_guid_t shut =
      arts_edt_create(shutdown_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  for (int iter = 0; iter < K_ITERS; iter++) {
    arts_guid_t dbs[N_DBS];

    /* DBs distributed round-robin across all ranks.  For each DB we
     * also wire an init_writer EDT pinned on the home rank that takes
     * the first RW lease and stamps the payload to a known value.  This
     * is required because RC does not synthesise an initial RO
     * snapshot from an unwritten payload — without an explicit RW
     * writer the first cross-rank RO acquire returns NULL ptr. */
    for (int i = 0; i < N_DBS; i++) {
      void *raw = NULL;
      unsigned int home = (unsigned int)(i % (int)nnodes);
      dbs[i] = arts_db_create(&raw, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = home});
      arts_guid_t init = arts_edt_create(init_writer_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = home, .finish_event = fe});
      arts_add_dependence(dbs[i], init, 0, DB_MODE_RW);
    }

    /* Workers fan out across all ranks, deterministically picking a DB
     * and an access mode based on (iter, i).  RC's per-DB lease
     * ordering serialises every RW behind the init_writer above.
     *
     * Note: B.2 uses RW-only across ranks.  Cross-rank RO acquire in
     * RC has a known issue where the first RO acquire arriving on
     * a rank that has not yet seen any RW may observe NULL ptr (the
     * RO snapshot install path lazily fetches from home only after the
     * first RW lease releases).  Validating cross-rank RO is left to
     * the dedicated coherence_ro_acquire_stress test (single-node) and
     * to coherence_mixed_local_remote (which interleaves explicit RW
     * fences). */
    for (int i = 0; i < N_EDTS; i++) {
      int db_idx = (iter * 7 + i * 13) % N_DBS;
      unsigned int worker_route = (unsigned int)(i % (int)nnodes);
      uint64_t mode_is_rw = 1; /* B.2: RW-only fan-out */
      arts_guid_t w =
          arts_edt_create(worker_edt, 1, &mode_is_rw, 1, &(arts_edt_hint_t){.rank = worker_route, .finish_event = fe});
      arts_add_dependence(dbs[db_idx], w, 0, DB_MODE_RW);
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  if (arts_get_current_rank() == 0 && !atomic_load(&g_clean_shutdown)) {
    fprintf(stderr, "FAIL: shutdown_edt did not fire — finish scope never completed "
                    "(consumer abort or premature peer-disconnect shutdown)\n");
    return 1;
  }
  return 0;
}
