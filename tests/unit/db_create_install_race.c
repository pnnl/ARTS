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
/// @brief Two creators on ONE rank race the same labeled GUID, so one loses
///        arts_route_table_install_if_absent and takes the adopt path: it
///        frees its own creator_stub, adopts the cache already installed, and
///        folds its hold into that cache's word.  Targets the use-after-free
///        where the auto-acquire registration read the FREED stub instead of
///        the descriptor that survived the race.
///
/// Same-rank on purpose.  Racing creators that take the creator's implicit
/// hold are supported only within one rank: they share one cache and one
/// coherence word, so the race has a winner and the loser's hold is folded
/// into the winner's state.  Across ranks each creator would stamp itself the
/// holder of a block only one of them can hold, which the programming model
/// does not admit — that pattern is diagnosed at the home, not exercised here.
/// (Cross-rank racing creators that take NO hold are legal and are covered by
/// grant_purge_no_acquire_creator; not duplicated here.)
///
/// Setup: rank 0 reserves a labeled GUID range homed on rank 0 and sends TWO
/// creator EDTs to the SAME remote rank, which both create the identical child
/// GUID.  The home being remote is what puts them on the creator-stub path
/// where the adoption lives.  A reader on the home then verifies a consistent
/// value — a UAF or refcount underflow on the loser path tends to surface as a
/// crash under sanitizers or a stuck DB (caught by the ctest TIMEOUT).
///
/// non-EXCL only: the adoption arithmetic is #if !EXCL.  Needs >= 2 ranks so
/// the creators' home is remote to them; SKIPs cleanly otherwise.

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
  /* slot 0 gates on both creators finishing; slot 1 is the block itself. */
  unsigned int *d = (unsigned int *)depv[1].ptr;
  if (d == NULL || d[0] != SENTINEL) {
    (void)fprintf(stderr,
                  "FAIL: db_create_install_race reader got 0x%x want 0x%x\n",
                  d ? d[0] : 0u, SENTINEL);
    arts_abort(1);
    return;
  }
  arts_printf("PASS: db_create_install_race\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_create_install_race ===\n");

#if defined(ARTS_PROTOCOL_EXCL)
  arts_printf("SKIP db_create_install_race: non-EXCL only (writer_count "
              "adoption arm)\n");
  arts_shutdown();
  return;
#else
  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    arts_printf("SKIP db_create_install_race: needs >= 2 ranks (have %u)\n",
                nranks);
    arts_shutdown();
    return;
  }

  /* Reserve a one-element labeled range homed on rank 0; child idx 0 is homed
   * on rank 0, which is REMOTE to the creators (both on rank 1). */
  arts_guid_t range = arts_guid_reserve_range(ARTS_GUID_DB, 1, 0);
  uint64_t rparam = (uint64_t)range;

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* BOTH creators on the SAME rank: they race that rank's route table, which
   * is the only install race a coherent create is allowed to have. */
  arts_edt_create(remote_creator, 1, &rparam, 0,
                  &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
  arts_edt_create(remote_creator, 1, &rparam, 0,
                  &(arts_edt_hint_t){.rank = 1, .finish_event = fe});

  arts_guid_t r =
      arts_edt_create(reader_edt, 1, &rparam, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe, r, 0, DB_MODE_NULL);
  arts_add_dependence(arts_guid_from_index(range, 0), r, 1, DB_MODE_RO);
#endif
}

int main(int argc, char **argv) {
  /* The checks abort the process, so their verdict is already the exit
     status; this only adds the runtime's own view of the ranks it spawned. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
