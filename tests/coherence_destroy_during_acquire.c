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

/// @file coherence_destroy_during_acquire.c
/// @brief B.4 — In-flight acquire safety during destroy.
///
/// While an EDT group is normally progressing through acquire/release on a
/// DB, another EDT calls arts_db_destroy on the same GUID.  Verifies that
/// cache_s's own ref counting (writer_count + pending_count +
/// buffer.ref_count) protects in-flight users — they either complete
/// normally with a valid pointer or observe NULL cleanly without
/// segfaulting.
///
/// Adaptations vs. plan code:
///   - 4-arg arts_db_create (no ARTS_DB_PROP_NONE in HEAD).
///   - DB_MODE_RW unifies the legacy DB_MODE_RW post-Cutover-C.
///   - arts_init_main does not exist; arts_rt() invokes main_edt
///     automatically on rank 0.
///   - Outer finish scope + finish-EDT shuts down only after all iterations'
///     workers + destroyer have completed — keeps arts_rt() alive across
///     all 20 iterations.
///
/// Spec section 6 B.4.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define N_EDTS 50
#define N_ITERATIONS 20

/// Per-iteration completion counter.  Reset by main_edt before each iter.
/// Workers that observe NULL ptr (DB destroyed before they ran) skip the
/// data update but still bump the counter so the iteration finish-EDT
/// fires.
static atomic_int g_completed = 0;

/// Total successful (non-NULL) data updates across all iterations.  Used
/// only as a sanity counter — exact value is non-deterministic because of
/// the race with the destroyer.
static atomic_int g_data_writes = 0;

/// Set by the outer finish-EDT.  If arts_rt() returns without this being
/// set the run was aborted early — main() reports FAIL.
static atomic_int g_clean_shutdown = 0;

static void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data != NULL) {
    /* Brief use — non-NULL ptr means cache_s ref counting kept the
     * payload alive across the destroyer's call. */
    int local = *data;
    *data = local + 1;
    atomic_fetch_add(&g_data_writes, 1);
  }
  /* NULL is acceptable here: destroyer ran first and the cache_s was
   * fully torn down before this worker acquired.  Test passes as long as
   * we did not SIGSEGV on access. */
  atomic_fetch_add(&g_completed, 1);
}

static void destroyer_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t db_guid = (arts_guid_t)paramv[0];
  arts_db_destroy(db_guid);
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  atomic_store(&g_clean_shutdown, 1);
  arts_printf("PASS: %d iterations, %d non-NULL writes (race-dependent)\n",
              N_ITERATIONS, atomic_load(&g_data_writes));
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_destroy_during_acquire (%d iter, %d EDTs/iter) "
              "===\n",
              N_ITERATIONS, N_EDTS);

  /* Outer finish scope covers all workers + destroyers across all iterations.
   * The finish-EDT must have depc >= 1 so the finish scope's slot-0 satisfy
   * actually gates it; depc=0 would let it fire before any worker. */
  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  for (int iter = 0; iter < N_ITERATIONS; iter++) {
    int *data;
    arts_guid_t db_guid =
        arts_db_create((void **)&data, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
    *data = 0;

    /* Spawn N workers all RW-acquiring the same DB. */
    for (int i = 0; i < N_EDTS; i++) {
      arts_guid_t w =
          arts_edt_create(worker_edt, 0, NULL, 1, &(arts_edt_hint_t){.finish_event = fe});
      arts_add_dependence(db_guid, w, 0, DB_MODE_RW);
    }

    /* Destroyer runs in parallel — no dependency wiring forces ordering
     * with respect to the workers, so the destroy may land before, during,
     * or after some workers' acquires.  cache_s's writer_count +
     * pending_count + buffer.ref_count are the safety net under test. */
    uint64_t prm = (uint64_t)db_guid;
    arts_edt_create(destroyer_edt, 1, &prm, 0, &(arts_edt_hint_t){.finish_event = fe});
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
