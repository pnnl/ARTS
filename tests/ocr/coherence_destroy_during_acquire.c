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
/// @brief Verify clean destroy AFTER all DB users have quiesced.
///
/// A DB is destroyed only after every EDT that held a dependence on it has
/// completed: workers are collected under a dedicated per-iteration finish
/// event (few), and the destroyer depends on few via DB_MODE_NULL so it
/// cannot run until all workers have released the DB.  Destroyer is pinned
/// to the DB home rank (rank 0) to satisfy the OWNER placement's requirement
/// that destroy originates at the owner.  Exercising destroy-in-use is OCR
/// undefined behaviour; this test verifies the legal, quiesced path.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define N_EDTS 50
#define N_ITERATIONS 20

static void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data != NULL) {
    int local = *data;
    *data = local + 1;
  }
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
  arts_printf("PASS: coherence_destroy_during_acquire %d iterations\n",
              N_ITERATIONS);
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
    arts_guid_t db_guid = arts_db_create((void **)&data, sizeof(int), ARTS_DB,
                                         ARTS_DB_PROP_NONE, NULL);
    *data = 0;
    arts_db_release(db_guid, DB_MODE_RW);

    /* Dedicated finish scope covering all workers for this iteration.
     * The destroyer depends on few so it only runs after every worker has
     * released its RW hold — destroy-in-use is OCR undefined behaviour. */
    arts_guid_t few = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    for (int i = 0; i < N_EDTS; i++) {
      arts_guid_t w = arts_edt_create(worker_edt, 0, NULL, 1,
                                      &(arts_edt_hint_t){.finish_event = few});
      arts_add_dependence(db_guid, w, 0, DB_MODE_RW);
    }

    /* Destroyer on the DB home rank (rank 0), gated after all workers. */
    uint64_t prm = (uint64_t)db_guid;
    arts_guid_t dz =
        arts_edt_create(destroyer_edt, 1, &prm, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(few, dz, 0, DB_MODE_NULL);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
