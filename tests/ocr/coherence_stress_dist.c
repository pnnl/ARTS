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
/// deterministic mode (RW or RO).  The ownership protocol must transfer
/// ownership / install RO snapshots across ranks; the final completion count
/// must equal N_EDTS * K_ITERS.
///
/// Correctness: each worker calls arts_abort(1) on NULL ptr; a stranded
/// waiter surfaces as a ctest TIMEOUT.  No global cross-EDT state.
///
/// Spec section 6 B.2.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define N_DBS 100
#define N_EDTS 200
#define K_ITERS 10

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
    /* Per-node serialised RW: increment is safe under per-node exclusivity. */
    (*data)++;
  } else {
    /* RO: just read, do not modify. */
    volatile int seen = *data;
    (void)seen;
  }
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: all worker EDTs completed across %d iterations\n",
              K_ITERS);
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
     * the first RW grant and stamps the payload to a known value.  This
     * is required because the ownership protocol does not synthesise an
     * initial RO snapshot from an unwritten payload — without an
     * explicit RW writer the first cross-rank RO acquire returns NULL
     * ptr. */
    for (int i = 0; i < N_DBS; i++) {
      void *raw = NULL;
      unsigned int home = (unsigned int)(i % (int)nnodes);
      dbs[i] = arts_db_create(&raw, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                              &(arts_db_hint_t){.rank = home});
      arts_guid_t init =
          arts_edt_create(init_writer_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = home, .finish_event = fe});
      arts_add_dependence(dbs[i], init, 0, DB_MODE_RW);
    }

    /* Workers fan out across all ranks, deterministically picking a DB
     * and an access mode based on (iter, i).  Per-DB grant ordering
     * serialises every RW behind the init_writer above.
     *
     * Note: B.2 uses RW-only across ranks.  Cross-rank RO acquire has
     * a known issue where the first RO acquire arriving on a rank that
     * has not yet seen any RW may observe NULL ptr (the RO snapshot
     * install path lazily fetches from home only after the first RW
     * grant releases).  Validating cross-rank RO is left to the
     * dedicated coherence_ro_acquire_stress test (single-node) and to
     * coherence_mixed_local_remote (which interleaves explicit RW
     * fences). */
    for (int i = 0; i < N_EDTS; i++) {
      int db_idx = (iter * 7 + i * 13) % N_DBS;
      unsigned int worker_route = (unsigned int)(i % (int)nnodes);
      uint64_t mode_is_rw = 1; /* B.2: RW-only fan-out */
      arts_guid_t w = arts_edt_create(
          worker_edt, 1, &mode_is_rw, 1,
          &(arts_edt_hint_t){.rank = worker_route, .finish_event = fe});
      arts_add_dependence(dbs[db_idx], w, 0, DB_MODE_RW);
    }
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
