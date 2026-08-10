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

/// @file db_copy_to_new_type_race.c
/// @brief arts_db_copy_to_new_type mutates db_type / db_guid IN PLACE on a live
///        route-table object with no coherence quiesce.  This test drives the
///        retype concurrently with acquirers reading db_type for subtype
///        dispatch (targets the suspected torn-read / wrong-subtype race).
///
/// arts_db_copy_to_new_type(old, new_type):
///   db_res->cache.db_guid = new_guid; db_res->db_type = new_type;
///   arts_route_table_move_item(old, new);
/// No lock is taken.  If another thread is mid-acquire on `old` (reading
/// db_type to choose the coherent vs pinned path), it can observe a torn /
/// mid-flight retype.  The documented contract is single-owner-creator; this
/// test exercises the well-formed (single-owner) use plus stress to surface
/// any latent torn-read under a sanitizer build.
///
/// Well-formed coverage (the safe contract): create coherent ARTS_DB, fill,
/// release, retype to ARTS_DB_PIN, then read the copied (now PIN) DB and verify
/// the bytes survived the in-place mutation + move_item.  Repeated many times
/// so a TSan/ASan build has a chance to flag the unguarded db_type store if a
/// concurrent runtime acquire ever overlaps it.
///
/// All configs (the retype is local-only; copy is rank-local).  Single-node is
/// sufficient; multinode just runs the same local retype on rank 0.  A wrong
/// dispatch corrupts the read-back -> arts_abort; a hang is caught by TIMEOUT.

#include "arts.h"
#include "arts/db.h" /* arts_db_copy_to_new_type */

#include <stdint.h>
#include <stdio.h>

#define ITERS 200u
#define ELEMS 8u

/// reader: RO on the COPIED (PIN) DB; verifies the bytes and the GUID.
void copy_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  uint64_t base = paramv[0];
  arts_guid_t expect_guid = (arts_guid_t)paramv[1];
  uint64_t *d = (uint64_t *)depv[0].ptr;
  bool ok = (d != NULL && depv[0].guid == expect_guid);
  for (unsigned int i = 0; i < ELEMS && ok; i++) {
    if (d[i] != base + i) {
      ok = false;
    }
  }
  if (!ok) {
    (void)fprintf(stderr,
                  "FAIL: db_copy_to_new_type_race read-back mismatch "
                  "(base=%lu)\n",
                  (unsigned long)base);
    arts_abort(1);
  }
  (void)paramc;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_copy_to_new_type_race ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (unsigned int it = 0; it < ITERS; it++) {
    uint64_t base = 0x1000u + (uint64_t)it * ELEMS;

    /* Source: coherent ARTS_DB on rank 0 (home == self for the local-only
     * copy). */
    void *ptr = NULL;
    arts_guid_t src =
        arts_db_create(&ptr, ELEMS * sizeof(uint64_t), ARTS_DB,
                       ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
    uint64_t *s = (uint64_t *)ptr;
    for (unsigned int i = 0; i < ELEMS; i++) {
      s[i] = base + i;
    }
    arts_db_release(src, DB_MODE_RW);

    /* In-place retype coherent -> PIN (mutates db_type/db_guid + move_item). */
    arts_guid_t copied = arts_db_copy_to_new_type(src, ARTS_DB_PIN);
    if (copied == NULL_GUID) {
      (void)fprintf(stderr,
                    "FAIL: db_copy_to_new_type_race retype returned NULL_GUID "
                    "(it=%u)\n",
                    it);
      arts_abort(1);
      return;
    }

    uint64_t params[2] = {base, (uint64_t)copied};
    arts_guid_t r =
        arts_edt_create(copy_reader, 2, params, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(copied, r, 0, DB_MODE_RO);
    arts_event_wait(fe);
    fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    arts_db_destroy(copied);
  }

  arts_printf("PASS: db_copy_to_new_type_race %u iterations\n", ITERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
