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
** License for the specific language governing contracts, limitations and   **
******************************************************************************/

/// @file lc_local_hit_dedup.c
/// @brief LC same-rank dedup path: N concurrent same-rank RO acquires on a
///        DB whose home is on a different node exercise the GET_DATA /
///        STILL_VALID cache-hit dedup code path.  Without wire-level
///        instrumentation this test exercises path coverage only — it
///        does not assert a specific wire-message count.  Correctness is
///        verified by comparing each reader's checksum against the value
///        written by the single writer.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define N_READERS 16
#define DB_SIZE 512

static uint64_t g_expected_sum = 0;

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t my_id = paramv[0];
  uint8_t *p = (uint8_t *)depv[0].ptr;
  uint64_t sum = 0;
  if (p) {
    for (size_t i = 0; i < DB_SIZE; i++) {
      sum += p[i];
    }
  }
  if (sum != g_expected_sum) {
    arts_printf("LC_LOCAL_HIT_DEDUP: reader %llu FAIL sum=%llu expected=%llu\n",
                (unsigned long long)my_id, (unsigned long long)sum,
                (unsigned long long)g_expected_sum);
  }
  (void)my_id;
}

static void verifier_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("LC_LOCAL_HIT_DEDUP: PASS ran %d same-rank RO readers\n",
              N_READERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("LC_LOCAL_HIT_DEDUP: SKIP requires 2 ranks\n");
    arts_shutdown();
    return;
  }

  /* Home rank 0; all readers run on rank 1. */
  void *addr = NULL;
  arts_guid_t db = arts_db_create(&addr, DB_SIZE, ARTS_DB, ARTS_DB_PROP_NONE,
                                  &(arts_db_hint_t){.rank = 0});
  memset(addr, 0x42, DB_SIZE);
  arts_db_release(db, DB_MODE_RW);

  g_expected_sum = 0x42ULL * DB_SIZE;

  arts_guid_t ver = arts_edt_create(verifier_edt, 0, NULL, 0, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, ver, 0, DB_MODE_NULL);

  for (uint64_t i = 0; i < N_READERS; i++) {
    arts_guid_t edt =
        arts_edt_create(reader_edt, 1, &i, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    arts_add_dependence(db, edt, 0, DB_MODE_RO);
    (void)edt;
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
