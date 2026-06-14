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

/// @file db_drf_multi_writer_concurrent.c
/// @brief DB-DRF multi-writer test: two ranks each acquire RW and
///        write their rank-id into a shared DB.  Under the DB-DRF
///        contract the DB accepts concurrent RW requests from different nodes;
///        the last writer's value wins.  The verifier (run after both writes
///        complete via a finish scope) accepts any value in {0, 1} as long as
///        the buffer is internally consistent.
///
///        Manual test — not added to CTest because the non-deterministic
///        "last writer wins" outcome is intentional and cannot be expressed
///        as a PASS_REGULAR_EXPRESSION.  Run with a 2-node config.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define DB_SIZE 4096

static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint8_t my_rank = (uint8_t)paramv[0];
  uint8_t *p = (uint8_t *)depv[0].ptr;
  if (p) {
    for (size_t i = 0; i < DB_SIZE; i++) {
      p[i] = my_rank;
    }
  }
}

static void verifier_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint8_t *p = (uint8_t *)depv[0].ptr;
  uint8_t v = p ? p[0] : 255;
  int consistent = 1;
  if (p) {
    for (size_t i = 1; i < DB_SIZE; i++) {
      if (p[i] != v) {
        consistent = 0;
        break;
      }
    }
  }
  /* Under the DB-DRF contract the final value is one of the writer rank-ids. */
  if (consistent && (v == 0 || v == 1)) {
    arts_printf("DB_DRF_MULTI_WRITER: PASS final=%u\n", (unsigned)v);
  } else {
    arts_printf("DB_DRF_MULTI_WRITER: FAIL value=%u consistent=%d\n",
                (unsigned)v, consistent);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("DB_DRF_MULTI_WRITER: SKIP requires 2 ranks\n");
    arts_shutdown();
    return;
  }

  void *addr = NULL;
  arts_guid_t db = arts_db_create(&addr, DB_SIZE, ARTS_DB, ARTS_DB_PROP_NONE,
                                  &(arts_db_hint_t){.rank = 0});
  memset(addr, 0xff, DB_SIZE);
  arts_db_release(db, DB_MODE_RW);

  /* verifier_edt is the finish-EDT of a finish scope that contains the two
   * concurrent writers.  The finish scope guarantees the verifier runs only
   * after both writers have released the DB. */
  arts_guid_t ver = arts_edt_create(verifier_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, ver, 1, DB_MODE_NULL);

  for (unsigned int r = 0; r < 2; r++) {
    uint64_t param = (uint64_t)r;
    arts_guid_t edt =
        arts_edt_create(writer_edt, 1, &param, 1,
                        &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_add_dependence(db, edt, 0, DB_MODE_RW);
    (void)edt;
  }
  arts_add_dependence(db, ver, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
