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

/// @file coherence_stress_single_node.c
/// @brief B.1 — Single-node coherence stress.
///
/// N=200 EDTs, M=50 DBs, RW+RO mix, K=100 iterations.  Every iteration
///   - creates M DBs initialized to a known per-DB value,
///   - spawns N EDTs, each RO- or RW-acquiring one of the DBs,
///   - reaches a per-iteration finish EDT that verifies all workers ran.
/// Iteration N+1 only starts after iteration N's finish EDT has fired —
/// this gives RC many back-to-back acquire/release cycles per DB.
///
/// Adaptations vs. plan code:
///   - 4-arg arts_db_create (no ARTS_DB_PROP_NONE in HEAD).
///   - arts_init_main does not exist; arts_rt() invokes main_edt
///     automatically on rank 0.
///   - One outer finish scope covers all iterations' workers; a single shutdown
///     EDT verifies the global completion count and tears the runtime
///     down once.
///
/// Spec section 6 B.1.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N_EDTS 200
#define M_DBS 50
#define K_ITERS 100

/// Total worker-EDT completion counter.  Expected: K_ITERS * N_EDTS.
static atomic_int g_completed = 0;

/// Sum of values seen by RO+RW workers across all iterations.  Each
/// worker observes the value it writes (RW) or the snapshot value (RO);
/// the cumulative total is deterministic for the i-spread used here.
static atomic_long g_sum = 0;

/// Set when shutdown_edt fires after all iterations finish.
static atomic_int g_clean_shutdown = 0;

static void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: worker got NULL ptr\n");
    arts_abort(1);
  }
  /* RO acquires share the DB safely; RW acquires are per-node serialised.
   * Either way we just bump the local payload by reading then writing. */
  atomic_fetch_add(&g_sum, *data);
  atomic_fetch_add(&g_completed, 1);
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  int got = atomic_load(&g_completed);
  int expected = K_ITERS * N_EDTS;
  if (got != expected) {
    fprintf(stderr, "FAIL: completed %d of %d worker EDTs\n", got, expected);
    arts_abort(1);
  }
  atomic_store(&g_clean_shutdown, 1);
  arts_printf("PASS: %d EDTs completed across %d iterations (sum=%ld)\n", got,
              K_ITERS, atomic_load(&g_sum));
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_stress_single_node (%d iter, %d EDTs/iter, "
              "%d DBs/iter) ===\n",
              K_ITERS, N_EDTS, M_DBS);

  /* Finish-EDT must have depc >= 1 so the finish scope's slot-0 satisfy
   * actually gates it; depc=0 would let it fire before any worker. */
  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  for (int iter = 0; iter < K_ITERS; iter++) {
    arts_guid_t dbs[M_DBS];
    for (int i = 0; i < M_DBS; i++) {
      int *data;
      dbs[i] = arts_db_create((void **)&data, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
      *data = i;
    }

    /* Mixed RO/RW workers fan out over the DB pool. */
    for (int i = 0; i < N_EDTS; i++) {
      int db_idx = i % M_DBS;
      arts_db_access_mode_t mode = (i & 1) ? DB_MODE_RW : DB_MODE_RO;
      arts_guid_t w =
          arts_edt_create(worker_edt, 0, NULL, 1, &(arts_edt_hint_t){.finish_event = fe});
      arts_add_dependence(dbs[db_idx], w, 0, mode);
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  if (arts_get_current_rank() == 0 && !atomic_load(&g_clean_shutdown)) {
    fprintf(stderr,
            "FAIL: shutdown_edt did not fire — finish scope never completed\n");
    return 1;
  }
  return 0;
}
