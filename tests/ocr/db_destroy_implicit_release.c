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

/// @file db_destroy_implicit_release.c
/// @brief arts_db_destroy implicit release: Path-1 (created_db_list) fires for
/// a
///        DB the EDT created, even when the same GUID is also a dep slot — the
///        RO dep slot is then left to the epilogue.  Benign-coverage of the
///        documented Path-1-wins / Path-2-RO-left interaction.
///
/// arts_db_destroy(guid) unconditionally calls arts_db_release(guid, RW) first
/// (OCR ocrDbDestroy semantics).  arts_db_release scans created_db_list (Path
/// 1) before depv (Path 2); on a match in Path 1 it releases the created hold
/// (RW) and returns early.  So when an EDT BOTH created a DB AND holds it as a
/// dep slot, the implicit release fires Path 1 (the created RW hold), and the
/// dep slot is left for the epilogue release_dbs to clean.
///
/// This test exercises that exact shape: an EDT creates a DB, also receives it
/// as a RO dep, then destroys it mid-body.  It must not crash, double-release,
/// or leak, and a follow-up EDT must run cleanly after the DB is gone.  A
/// double-free / underflow tends to crash under sanitizers; a leaked hold that
/// blocks teardown manifests as a hang caught by the ctest TIMEOUT.
///
/// All configs.  Home the DB on rank 0 so create + dep + destroy are co-located
/// on the running EDT's worker (the created_db_list is thread-local).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define SENTINEL 0xDED0Du

/// creator_and_holder: receives a DB as a RO dep (slot 0), then RE-CREATES the
/// same labeled GUID in-body so that GUID lands on this EDT's created_db_list
/// (Path-1 hold).  Now the GUID is BOTH a dep slot (RO, old generation, already
/// resolved) AND a created hold.  Destroying it must fire Path 1 (the created
/// RW hold) and return early, leaving the RO dep slot to the epilogue.
void creator_and_holder(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t db = (arts_guid_t)paramv[0];
  /* depv[0] is the DB in RO mode (old generation); reading it is valid here. */
  unsigned int *ro = (unsigned int *)depv[0].ptr;
  if (ro == NULL) {
    (void)fprintf(stderr,
                  "FAIL: db_destroy_implicit_release RO dep NULL ptr\n");
    arts_abort(1);
    return;
  }
  /* Re-create at the SAME labeled GUID: this EDT becomes the creator, so the
   * GUID is recorded on created_db_list (Path-1 hold).  Default props =>
   * auto-acquire RW. */
  void *p = (void *)arts_db_create_with_guid(db, sizeof(unsigned int), ARTS_DB,
                                             ARTS_DB_PROP_NONE, NULL);
  if (p != NULL) {
    ((unsigned int *)p)[0] = SENTINEL + 1u;
  }
  /* Destroy: implicit RW release fires Path 1 (the created hold) and returns
   * early; the RO dep slot is left to the epilogue release_dbs. */
  arts_db_destroy(db);
}

/// after_edt: runs after the destroying EDT's finish scope drains; proves the
/// runtime stayed healthy (no crash / no stuck teardown).
void after_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: db_destroy_implicit_release\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_destroy_implicit_release ===\n");

  /* Create on rank 0, fill, release so the DB carries data, then create an EDT
   * that BOTH re-creates(no) — instead we create the DB here and pass its GUID
   * to an EDT that will also depend on it.  To make the EDT the CREATOR of the
   * DB on its created_db_list, the EDT must itself call arts_db_create; but it
   * also needs the DB as a dep.  We satisfy both: the EDT creates a fresh DB
   * (Path-1 created hold) AND depends on a pre-existing DB at the same GUID via
   * labeled reuse.  Simpler and faithful to the contract: have the EDT create
   * the DB at a labeled GUID it already holds as a RO dep. */
  arts_guid_t db = arts_guid_reserve(ARTS_GUID_DB, 0);

  /* Pre-create + fill so the RO dep has data when the EDT runs. */
  void *p = (void *)arts_db_create_with_guid(db, sizeof(unsigned int), ARTS_DB,
                                             ARTS_DB_PROP_NONE, NULL);
  if (p != NULL) {
    ((unsigned int *)p)[0] = SENTINEL;
  }
  arts_db_release(db, DB_MODE_RW);

  uint64_t param = (uint64_t)db;
  arts_guid_t e_c = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t c =
      arts_edt_create(creator_and_holder, 1, &param, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_c});
  arts_add_dependence(db, c, 0, DB_MODE_RO);
  arts_event_wait(e_c);

  /* Health check after the DB is destroyed. */
  arts_guid_t e_a = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_edt_create(after_edt, 0, NULL, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = e_a});
  arts_event_wait(e_a);

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
