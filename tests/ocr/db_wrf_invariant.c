/******************************************************************************
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

/// @file db_wrf_invariant.c
/// @brief DB-WRF invariant: RW writer followed by RO reader
///        on the same rank
///        (home rank).  The writer increments a counter N times; the reader
///        verifies the expected value.
///
///        Uses a two-level finish scope pattern:
///          outer finish scope  → shutdown_edt (finish-EDT, depc=1: outer VAL)
///          inner finish scope  → reader_edt  (finish-EDT, depc=2: DB RO +
///          inner VAL) writer_edt in inner finish scope
///
///        Under DRF the inner finish scope guarantees reader_edt executes only
///        after writer_edt has released the DB.  When writer and reader are
///        both on the home rank no WRITEBACK round-trip is needed — the
///        home buffer is updated in-place — so the DRF guarantee holds.
///        Registered as a single-node ctest; the single-rank path exercises
///        the finish scope ordering invariant without depending on multi-node
///        writeback delivery.

#include <stdint.h>

#include "arts.h"

#define N_ITERS 16

static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *db = (uint64_t *)depv[0].ptr;
  if (db) {
    for (int i = 0; i < N_ITERS; i++) {
      (*db)++;
    }
  }
}

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  /* depv[0] = DB RO; depv[1] = inner finish scope VAL */
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *db = (uint64_t *)depv[0].ptr;
  uint64_t got = db ? *db : (uint64_t)-1;
  if (got == (uint64_t)N_ITERS) {
    arts_printf("DB_WRF: PASS value=%llu\n", (unsigned long long)got);
  } else {
    arts_printf("DB_WRF: FAIL got=%llu expected=%d\n", (unsigned long long)got,
                N_ITERS);
  }
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int total = arts_get_total_ranks();
  unsigned int writer_rank = (total > 1) ? 1 : 0;

  void *addr = NULL;
  arts_guid_t db =
      arts_db_create(&addr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  *(uint64_t *)addr = 0;
  arts_db_release(db, DB_MODE_RW);

  /* Outer finish scope → shutdown_edt (depc=1, slot 0 = outer finish scope
   * VAL). */
  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t outer = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(outer, shut, 0, DB_MODE_NULL);

  /* reader_edt is the finish-EDT of the inner finish scope.
   * depc=2: slot 0 = DB RO dep, slot 1 = inner finish scope VAL.
   * It lives in the outer finish scope so shutdown waits for it. */
  arts_guid_t rdr =
      arts_edt_create(reader_edt, 0, NULL, 2,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = outer});
  arts_add_dependence(db, rdr, 0, DB_MODE_RO);

  /* Inner finish scope: writer runs inside it, reader is the finish-EDT. */
  arts_guid_t inner = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(inner, rdr, 1, DB_MODE_NULL);

  arts_guid_t wtr = arts_edt_create(
      writer_edt, 0, NULL, 1,
      &(arts_edt_hint_t){.rank = writer_rank, .finish_event = inner});
  arts_add_dependence(db, wtr, 0, DB_MODE_RW);
  (void)wtr;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
