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

/// @file coh_basic.c
/// @brief Basic coherence smoke test: a chain of writers feeds a reader
///        that verifies the final value.  Uses arts_add_dependence + an
///        epoch finish-EDT for ordering — no deprecated arts_signal_edt.

#include "arts.h"

#define CHAIN_LEN 8

void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int idx = (unsigned int)paramv[0];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  data[0] = idx;
}

void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data && data[0] == CHAIN_LEN - 1) {
    arts_printf("  PASS: reader saw final write %u\n", data[0]);
  } else {
    arts_printf("  FAIL: reader expected %u got %d\n", CHAIN_LEN - 1,
                data ? (int)data[0] : -1);
  }
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

  arts_printf("=== coh_basic ===\n");

  /* Create a single DIST DB on rank 0; chain writes ordered by RW
   * per-node-exclusive (each writer sees previous writer's value);
   * reader as inner-epoch finish-EDT runs only after every writer has
   * released its RW slot. */
  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB_DIST,
                                  ARTS_DB_PROP_NONE,
                                  &(arts_hint_t){.route = 0});
  ((unsigned int *)ptr)[0] = 0;
  arts_db_release(db);

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t outer = arts_initialize_and_start_epoch(shut, 0);

  arts_guid_t reader = arts_edt_create_with_epoch(
      reader_edt, 0, NULL, 1, outer, &(arts_hint_t){.route = 0});
  arts_add_dependence(db, reader, 0, DB_MODE_RO);

  arts_guid_t inner = arts_initialize_and_start_epoch(reader, 1);
  for (unsigned int i = 0; i < CHAIN_LEN; i++) {
    uint64_t param = (uint64_t)i;
    arts_guid_t w = arts_edt_create_with_epoch(writer_edt, 1, &param, 1, inner,
                                               &(arts_hint_t){.route = 0});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
