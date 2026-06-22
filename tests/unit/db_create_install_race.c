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

/// @file db_create_install_race.c
/// @brief Two concurrent creator-remote installs of the SAME labeled GUID drive
///        the install_if_absent loser path in arts_db_create (remote branch):
///        the loser frees its creator_stub, adopts the existing cache, and
///        (non-LOCK) bumps writer_count += 2 + auto_acquire(creator_stub).
///        Targets the suspected use-after-free where auto_acquire reads
///        creator_stub->cache.db_guid AFTER arts_db_free(creator_stub).
///
/// Setup: rank 0 reserves a labeled GUID range whose first child is homed on
/// rank 0, broadcasts the range GUID to two DIFFERENT remote ranks, and both
/// remote ranks call arts_db_create_with_guid on the identical child GUID.
/// Because the home (rank 0) is remote to both creators, each takes the remote
/// creator-stub path; whichever loses route_table_install_if_absent runs the
/// adoption arm.  Both creators write a sentinel; an RO reader on home then
/// verifies a consistent (non-corrupt) value — a UAF / refcount underflow on
/// the loser path tends to surface as a crash under sanitizers or a stuck DB
/// (caught by the ctest TIMEOUT).
///
/// non-LOCK only: the writer_count += 2 adoption arithmetic is #if !LOCK.  The
/// race is irrelevant under LOCK (no writer_count), so this self-skips there.
/// Needs >= 3 ranks for two distinct remote creators; SKIPs cleanly otherwise.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define SENTINEL 0xC0FFEEu

/// remote_creator: runs on a non-home rank.  Re-derives the shared child GUID
/// from the broadcast range GUID and creates the coherent DB at it.  Two of
/// these race on the same GUID with a remote home (rank 0).
void remote_creator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t range = (arts_guid_t)paramv[0];
  arts_guid_t child = arts_guid_from_index(range, 0);
  void *p = arts_db_create_with_guid(child, sizeof(unsigned int), ARTS_DB,
                                     ARTS_DB_PROP_NONE, NULL);
  if (p != NULL) {
    ((unsigned int *)p)[0] = SENTINEL;
  }
  arts_db_release(child, DB_MODE_RW);
}

/// reader: RO on home; the DB must resolve to a consistent SENTINEL (both
/// concurrent creators wrote the same value, so a clean install yields it).
void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d == NULL || d[0] != SENTINEL) {
    (void)fprintf(stderr,
                  "FAIL: db_create_install_race reader got 0x%x want 0x%x\n",
                  d ? d[0] : 0u, SENTINEL);
    arts_abort(1);
    return;
  }
  arts_printf("PASS: db_create_install_race\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_create_install_race ===\n");

#if defined(ARTS_PROTOCOL_LOCK)
  arts_printf("SKIP db_create_install_race: non-LOCK only (writer_count "
              "adoption arm)\n");
  arts_shutdown();
  return;
#else
  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 3) {
    arts_printf("SKIP db_create_install_race: needs >= 3 ranks (have %u)\n",
                nranks);
    arts_shutdown();
    return;
  }

  /* Reserve a one-element labeled range homed on rank 0; child idx 0 is homed
   * on rank 0, which is REMOTE to both creator ranks (1 and 2). */
  arts_guid_t range = arts_guid_reserve_range(ARTS_GUID_DB, 1, 0);
  uint64_t rparam = (uint64_t)range;

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Two DISTINCT remote ranks create the same labeled GUID concurrently. */
  arts_edt_create(remote_creator, 1, &rparam, 0,
                  &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
  arts_edt_create(remote_creator, 1, &rparam, 0,
                  &(arts_edt_hint_t){.rank = 2, .finish_event = fe});

  /* Wait for both concurrent creates (and their releases) to drain. */
  arts_event_wait(fe);

  /* Read back on home; verify a consistent, non-corrupt SENTINEL. */
  arts_guid_t child = arts_guid_from_index(range, 0);
  arts_guid_t e_rd = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t r =
      arts_edt_create(reader_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_rd});
  arts_add_dependence(child, r, 0, DB_MODE_RO);
  arts_event_wait(e_rd);

  arts_shutdown();
#endif
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
