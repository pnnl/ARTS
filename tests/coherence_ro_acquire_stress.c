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

/// @file coherence_ro_acquire_stress.c
/// @brief B.6 — RO concurrent acquire stress (cache install correctness).
///
/// Single DB initialized to a known sentinel value (42).  N RO-acquiring
/// EDTs are spawned concurrently; each verifies that *data == 42.
/// Exercises eager-protocol cases 1/3/7 (concurrent local RO + RO snapshot
/// install + cached RO version pull).  If any reader sees a corrupted value
/// the cache install path itself is wrong — that is a real coherence bug.
///
/// Adaptations vs. plan description (line 1882 of plan):
///   - 4-arg arts_db_create (no ARTS_DB_PROP_NONE in HEAD).
///   - arts_init_main does not exist; arts_rt() invokes main_edt
///     automatically on rank 0.
///   - g_clean_shutdown / non-zero exit propagation so ctest sees FAIL on
///     any reader abort.
///
/// Spec section 6 B.6.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define N_READERS 100
#define SENTINEL 42

static atomic_int g_completed = 0;
static atomic_int g_clean_shutdown = 0;

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const int *data = (const int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: reader got NULL ptr\n");
    arts_abort(1);
  }
  if (*data != SENTINEL) {
    (void)fprintf(stderr,
                  "FAIL: reader saw %d (expected %d) — bad cache install\n",
                  *data, SENTINEL);
    arts_abort(1);
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
  if (got != N_READERS) {
    (void)fprintf(stderr, "FAIL: only %d of %d readers completed\n", got,
                  N_READERS);
    arts_abort(1);
  }
  atomic_store(&g_clean_shutdown, 1);
  arts_printf("PASS: %d concurrent RO readers all observed sentinel %d\n", got,
              SENTINEL);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_ro_acquire_stress (N=%d, sentinel=%d) ===\n",
              N_READERS, SENTINEL);

  int *data;
  arts_guid_t db = arts_db_create((void **)&data, sizeof(int), ARTS_DB,
                                  ARTS_DB_PROP_NONE, NULL);
  *data = SENTINEL;

  /* Outer finish scope ensures shutdown_edt runs only after every reader has
   * finished — a peer-disconnect SHUTDOWN_MSG would otherwise let the
   * runtime exit while readers are still in flight.  The finish-EDT
   * must have depc >= 1 so the finish scope's slot-0 satisfy actually gates
   * it; depc=0 would let it fire before any reader. */
  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  for (int i = 0; i < N_READERS; i++) {
    arts_guid_t r = arts_edt_create(reader_edt, 0, NULL, 1,
                                    &(arts_edt_hint_t){.finish_event = fe});
    arts_add_dependence(db, r, 0, DB_MODE_RO);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  if (arts_get_current_rank() == 0 && !atomic_load(&g_clean_shutdown)) {
    (void)fprintf(stderr,
                  "FAIL: shutdown_edt did not fire — reader abort or premature "
                  "shutdown\n");
    return 1;
  }
  return 0;
}
