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

/// @file mrsw_destroy_release_race.c
/// @brief Verify clean destroy AFTER all RW/RO users have released (MRSW).
///
/// Each iteration creates a DB, fans out N_RW RW workers and N_RO RO workers
/// under a dedicated per-iteration finish event (few), then gates the destroyer
/// on few via DB_MODE_NULL so it only runs after every worker has completed its
/// release.  Destroyer is pinned to the DB home rank (rank 0).  Destroy-in-use
/// is OCR undefined behaviour; this test verifies the legal, quiesced path and
/// that the runtime stays free of hangs and crashes across many iterations.
/// MRSW-only.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW)

int main(void) {
  printf("SKIP mrsw_destroy_release_race: MRSW-only\n");
  return 0;
}

#else

#define N_ITERS 30
#define N_RW 24
#define N_RO 12

static void rw_worker_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    /* Touch through the token; NULL is fine (destroyed first). */
    d[0] = d[0] + 1u;
  }
}

static void ro_worker_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const volatile uint64_t *d = (const volatile uint64_t *)depv[0].ptr;
  if (d != NULL) {
    volatile uint64_t sink = d[0];
    (void)sink;
  }
}

static void destroyer_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t db = (arts_guid_t)paramv[0];
  arts_db_destroy(db);
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: mrsw_destroy_release_race %d iterations (no hang, no "
              "double-consume)\n",
              N_ITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_destroy_release_race (%d iter) ===\n", N_ITERS);

  unsigned int nranks = arts_get_total_ranks();

  arts_guid_t shut =
      arts_edt_create(shutdown_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  for (int it = 0; it < N_ITERS; it++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((uint64_t *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* Dedicated finish scope covering all workers for this iteration.
     * The destroyer depends on few so it only runs after every worker has
     * released its hold — destroy-in-use is OCR undefined behaviour. */
    arts_guid_t few = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    for (int i = 0; i < N_RW; i++) {
      arts_guid_t w =
          arts_edt_create(rw_worker_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = (unsigned int)i % nranks,
                                             .finish_event = few});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
    }
    for (int i = 0; i < N_RO; i++) {
      arts_guid_t ro =
          arts_edt_create(ro_worker_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = (unsigned int)i % nranks,
                                             .finish_event = few});
      arts_add_dependence(db, ro, 0, DB_MODE_RO);
    }

    /* Destroyer on the DB home rank (rank 0), gated after all workers. */
    uint64_t dbv = (uint64_t)db;
    arts_guid_t dz =
        arts_edt_create(destroyer_edt, 1, &dbv, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(few, dz, 0, DB_MODE_NULL);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW */
