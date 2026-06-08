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

/// @file coherence_zero_size_xfer.c
/// @brief Regression: a zero-size (sentinel db_size==0) DataBlock driven
/// through
///        a cross-rank RW ownership transfer.
///
///        A sentinel DB never installs a payload buffer, so when ownership is
///        transferred the shedding owner ships a transfer message with a NULL
///        buffer.  Such an empty transfer must still carry the per-rank dedup
///        map count-header (count=0); omitting it underflows the receiver's
///        reconstructed payload length and corrupts the install (heap crash).
///        Driving the size-0 DB across ranks exercises exactly that path.
///
///        Completion is the assertion: if the empty transfer crashes any rank,
///        the finish scope never drains and shutdown never fires, surfacing as a ctest
///        FAIL.  Model-agnostic (RC/LC simply move ownership without a map).
///        Requires 2+ ranks.

#include "arts.h"

#include <stdatomic.h>
#include <stdio.h>

static atomic_int g_clean_shutdown = 0;

/// RW holder on a sentinel DB.  The dep pointer is NULL (no payload); acquiring
/// it RW is what makes this rank the owner and forces a transfer when the next
/// rank's RW lease arrives.
void rw_holder_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Sentinel DB: depv[0].ptr is expected NULL — nothing to read or write.
   * Holding the RW lease is the whole point. */
}

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  atomic_store(&g_clean_shutdown, 1);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_zero_size_xfer ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    arts_printf("SKIP: requires 2+ ranks (got %u)\n", nranks);
    atomic_store(&g_clean_shutdown, 1);
    arts_shutdown();
    return;
  }

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  /* Zero-size sentinel DB homed on rank 0.  No buffer is ever installed. */
  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, 0, ARTS_DB, ARTS_DB_PROP_NONE,
                                  &(arts_db_hint_t){.rank = 0});
  arts_db_release(db, DB_MODE_RW);

  /* One RW holder on home (rank 0, the initial owner) and one on a foreign
   * rank (rank 1): the second lease forces a single owner→foreign transfer of
   * a NULL-buffer DB, which is the empty-transfer path under test.  A single
   * hop is sufficient to exercise it; the coherence layer serializes the two
   * RW leases. */
  arts_guid_t w_home = arts_edt_create(
      rw_holder_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, w_home, 0, DB_MODE_RW);

  arts_guid_t w_foreign = arts_edt_create(
      rw_holder_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
  arts_add_dependence(db, w_foreign, 0, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  if (arts_get_current_rank() == 0 && !atomic_load(&g_clean_shutdown)) {
    fprintf(stderr, "FAIL: shutdown_edt did not fire — finish scope never completed "
                    "(empty-transfer crash or premature shutdown)\n");
    return 1;
  }
  return 0;
}
