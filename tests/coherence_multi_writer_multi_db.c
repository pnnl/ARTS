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

/// @file coherence_multi_writer_multi_db.c
/// @brief Multi-writer, multi-DB RW determinism + liveness check.
///
/// Generalizes coherence_multi_writer_same_addr (one shared DB) to a SET of
/// M shared RW DBs that each of N writer EDTs acquires.  Every writer takes
/// all M DBs RW (the runtime acquires an EDT's RW deps one at a time in a
/// single global order), increments each, then drops a fan-in LATCH.  A final
/// RO verify EDT checks every DB reached N and shuts down.
///
/// Why this exists separately from the single-DB test: an EDT that holds an
/// already-acquired RW DB while it is still acquiring its next RW dep is the
/// case that exercises an exclusive-ownership protocol's hand-off chain under
/// genuine multi-DB contention.  The single-DB test (one RW dep per EDT) never
/// holds one DB while waiting for another, so it cannot surface a hand-off
/// liveness defect that only appears when many EDTs each need the same SET of
/// single-writer DBs.  All writers acquire the M DBs in the same global order,
/// so a correct ownership protocol serializes them with no deadlock regardless
/// of release order.
///
/// PASS: every DB observes exactly N increments; all N writers + verify finish
/// (no stranded EDT).

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define N_WRITERS 8
#define M_DBS 4

static atomic_int g_clean_shutdown = 0;

/// writer_edt -- atomic-increment each of the M RW DB deps, then LATCH decr.
/// paramv[0] = LATCH event GUID (count N_WRITERS).
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  for (uint32_t i = 0; i < depc; i++) {
    _Atomic int *data = (_Atomic int *)depv[i].ptr;
    if (data == NULL) {
      (void)fprintf(stderr, "FAIL: writer_edt got NULL ptr on slot %u\n", i);
      arts_abort(1);
    }
    /* Per-node-concurrent RW: serialize the RMW with an atomic (the protocol's
     * job is buffer visibility + single-writer hand-off, not per-EDT mutex). */
    atomic_fetch_add_explicit(data, 1, memory_order_relaxed);
  }
  arts_guid_t evt = (arts_guid_t)paramv[0];
  arts_event_satisfy_slot(evt, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/// verify_edt -- slot 0..M-1 = the M DBs (RO), slot M = LATCH (fires after all
/// writers drop it).  Checks every DB == N_WRITERS, then shuts down.
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  for (uint32_t i = 0; i < M_DBS; i++) {
    _Atomic int *data = (_Atomic int *)depv[i].ptr;
    if (data == NULL) {
      (void)fprintf(stderr, "FAIL: verify_edt got NULL ptr on slot %u\n", i);
      arts_abort(1);
    }
    int observed = atomic_load_explicit(data, memory_order_relaxed);
    if (observed != N_WRITERS) {
      (void)fprintf(stderr, "FAIL: DB %u expected %d, got %d\n", i, N_WRITERS,
                    observed);
      arts_abort(1);
    }
  }
  (void)depc;
  atomic_store(&g_clean_shutdown, 1);
  arts_printf("PASS: %d writers x %d DBs each reached %d\n", N_WRITERS, M_DBS,
              N_WRITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_multi_writer_multi_db (N=%d, M=%d) ===\n",
              N_WRITERS, M_DBS);

  arts_guid_t dbs[M_DBS];
  for (int j = 0; j < M_DBS; j++) {
    int *data;
    dbs[j] = arts_db_create((void **)&data, sizeof(int), ARTS_DB,
                            ARTS_DB_PROP_NONE, NULL);
    *data = 0;
  }

  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(N_WRITERS);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);

  /* N writers, each acquiring ALL M DBs RW (slots 0..M-1).  The runtime walks
   * the RW deps in one global (GUID-sorted) order, holding each acquired DB
   * while it acquires the next. */
  uint64_t wpv[1] = {(uint64_t)latch};
  for (int i = 0; i < N_WRITERS; i++) {
    arts_guid_t w = arts_edt_create(writer_edt, /*paramc=*/1, wpv,
                                    /*depc=*/M_DBS, NULL);
    for (int j = 0; j < M_DBS; j++) {
      arts_add_dependence(dbs[j], w, /*slot=*/(uint32_t)j, DB_MODE_RW);
    }
  }

  /* verify: M DBs RO + the LATCH (fires after every writer drops it). */
  arts_guid_t v = arts_edt_create(verify_edt, /*paramc=*/0, NULL,
                                  /*depc=*/M_DBS + 1, NULL);
  for (int j = 0; j < M_DBS; j++) {
    arts_add_dependence(dbs[j], v, /*slot=*/(uint32_t)j, DB_MODE_RO);
  }
  arts_add_dependence(latch, v, /*slot=*/M_DBS, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  if (arts_get_current_rank() == 0 && !atomic_load(&g_clean_shutdown)) {
    (void)fprintf(stderr,
                  "FAIL: verify_edt did not fire cleanly — abort or premature "
                  "shutdown\n");
    return 1;
  }
  return 0;
}
