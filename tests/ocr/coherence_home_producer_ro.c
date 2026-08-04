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

/// @file coherence_home_producer_ro.c
/// @brief Regression: a creator-owner producer running ON the DB's home rank
///        that never transfers ownership, followed by a FOREIGN rank acquiring
///        the DB read-only.
///
///        This exercises the path where the home owner has no per-rank dedup
///        map yet (only an ownership transfer ever lazily creates one), so the
///        foreign RO request must still be served the producer's real bytes via
///        a lazily-created map rather than an empty (no-data) response.  A
///        prior coherence simplification regressed this into serving version-N
///        with no payload, so the foreign reader observed a NULL / stale
///        pointer.
///
///        Model-agnostic: under every model (HOME/OWNER/WRF_VAL) the same
///        producer→foreign-RO chain must deliver the written value.  Failure
///        aborts the consumer, which propagates as a non-zero exit and a ctest
///        FAIL.  Requires 2+ ranks.

#include "arts.h"

#include <stdio.h>

#define PRODUCER_VALUE 0x5A5A5A

/// Producer EDT — runs on the DB's home rank, RW.  Writes the sentinel value.
void producer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL: producer on home got NULL ptr\n");
    arts_abort(1);
  }
  data[0] = PRODUCER_VALUE;
}

/// Foreign reader EDT — runs on a non-home rank, RO.  Asserts the producer's
/// value is delivered (not a no-data NULL).
void foreign_reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL || data[0] != PRODUCER_VALUE) {
    (void)fprintf(stderr, "FAIL: foreign RO expected %d got %d (ptr=%p)\n",
                  PRODUCER_VALUE, data ? data[0] : -1, (void *)data);
    arts_abort(1);
  }
  arts_printf("PASS: foreign RO read %d from a home-producer DB\n",
              PRODUCER_VALUE);
}

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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

  arts_printf("=== coherence_home_producer_ro ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    arts_printf("SKIP: requires 2+ ranks (got %u)\n", nranks);
    arts_shutdown();
    return;
  }

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  /* DB homed on rank 0; the producer also runs on rank 0, so the home rank is
   * the sole creator-owner and never ships ownership to anyone. */
  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                                  &(arts_db_hint_t){.rank = 0});
  ((int *)ptr)[0] = 0;
  arts_db_release(db, DB_MODE_RW);

  /* Producer on home (rank 0), RW. */
  arts_guid_t prod =
      arts_edt_create(producer_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, prod, 0, DB_MODE_RW);

  /* Foreign reader (rank 1), RO — must see the producer's value. */
  arts_guid_t rdr =
      arts_edt_create(foreign_reader_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
  arts_add_dependence(db, rdr, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
