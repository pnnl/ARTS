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

/// @file multinode_labeled_guid.c
/// @brief Tests that a DB created with a pre-reserved GUID on rank 0 can be
///        read by an EDT on rank 1.  Requires multi-node (rank_count >= 2).
///        Prints SKIP if running single-node.

#include "arts.h"
#include <stdint.h>
#include <stdio.h>

#define SENTINEL 0xFEEDFACEULL

/// Creator EDT: runs on rank 0 inside the inner finish scope.
/// Creates the DB with the reserved GUID and writes the sentinel value.
static void creator_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t reserved = (arts_guid_t)paramv[0];
  uint64_t *ptr = (uint64_t *)arts_db_create_with_guid(
      reserved, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (ptr) {
    ptr[0] = SENTINEL;
  }
  arts_db_release(reserved, DB_MODE_RW);
}

/// Reader EDT: runs on rank 1, receives the DB via slot 0 (RO dep).
/// Slot 1 carries the inner-finish scope completion signal ensuring ordering.
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *data = (const uint64_t *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == SENTINEL);
  arts_printf("  %s: cross-rank labeled-GUID DB read (got 0x%lx)\n",
              ok ? "PASS" : "FAIL", data ? (unsigned long)data[0] : 0UL);
}

/// Shutdown EDT: fires after the outer finish scope completes.
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

  unsigned int ranks = arts_get_total_ranks();
  if (ranks < 2) {
    arts_printf("SKIP: multinode_labeled_guid requires node_count >= 2\n");
    arts_shutdown();
    return;
  }

  arts_printf("=== multinode_labeled_guid (%u ranks) ===\n", ranks);

  /* Reserve the DB GUID on rank 0 (home = 0). */
  arts_guid_t reserved = arts_guid_reserve(ARTS_GUID_DB, 0);

  /* Outer finish scope: fires shutdown_edt when all work completes. */
  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t outer = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(outer, shut, 0, DB_MODE_NULL);

  /* Reader EDT on rank 1: depc=2.
   *   slot 0 — DB RO dependence (delivers data pointer)
   *   slot 1 — inner finish scope completion signal (ensures DB is created
   * first) */
  uint64_t rparam = (uint64_t)reserved;
  arts_guid_t reader =
      arts_edt_create(reader_edt, 1, &rparam, 2,
                      &(arts_edt_hint_t){.rank = 1, .finish_event = outer});
  arts_add_dependence(reserved, reader, 0, DB_MODE_RO);

  /* Inner finish scope: creator EDT runs here; reader_edt slot 1 is the finish
   * slot. When all EDTs in the inner finish scope complete, slot 1 of reader
   * fires. */
  arts_guid_t inner = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(inner, reader, 1, DB_MODE_NULL);

  /* Creator EDT on rank 0 inside the inner finish scope. */
  arts_edt_create(creator_edt, 1, &rparam, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = inner});
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
