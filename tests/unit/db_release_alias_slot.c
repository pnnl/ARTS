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

/// @file db_release_alias_slot.c
/// @brief Mid-EDT arts_db_release of an ALIAS slot (the SAME DB bound to two RW
///        slots) drives the suspected writer_count underflow.
///
/// EXPOSES SUSPECTED BUG (db.c arts_db_release Path 2 alias underflow):
/// arts_db_release Path 2 always passes alias_only=false to release_one_dep.
/// When a DB is bound to two RW slots of one EDT, only the smallest-index slot
/// took the real coherence acquire (writer_count hold); the later slot is an
/// alias (buffer ref only).  release_dbs at the epilogue correctly computes
/// alias_only and decrements writer_count exactly once.  But a mid-EDT
/// arts_db_release naming the DB matches the FIRST slot in depv order — and
/// after that slot is nulled, the epilogue then treats the remaining (alias)
/// slot as the real acquirer.  Releasing one of two alias RW slots mid-EDT and
/// leaving the other to the epilogue therefore drops writer_count for both an
/// alias and the real slot -> a double decrement -> underflow / premature
/// ownership transfer.
///
/// Scenario: an EDT acquires DB in slot 0 (RW) and slot 1 (RW), writes a value,
/// mid-EDT releases the GUID once (Path 2 hits slot 0, alias_only=false), then
/// the epilogue releases slot 1 (also alias_only=false, since slot 0 is now
/// nulled).  Two writer_count decrements for a single real hold -> underflow.
/// A subsequent RW writer then either is granted ownership prematurely
/// (corruption -> reader mismatch -> arts_abort) or the underflow wraps the
/// counter and the writer never gets ownership (hang -> ctest TIMEOUT).
///
/// This test is designed to FAIL while the bug exists (it must stay
/// correct-and-failing); it documents the contract that mid-EDT release of an
/// alias slot must not double-decrement.  ownership/LOCK only (MRMW does not
/// serialize RW so there is no alias dedup; it self-skips).
///
/// No in-test watchdog: a hang is reaped by the ctest TIMEOUT.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define V0 0xAB0u
#define V1 0xAB1u

/// alias_releaser: same DB in slots 0 and 1 (both RW).  Writes V0, then mid-EDT
/// releases the DB once (Path 2 hits the first matching slot).  The second
/// (alias) slot is left to the epilogue.
void alias_releaser(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  if (depc < 2 || depv[0].ptr == NULL) {
    (void)fprintf(stderr, "FAIL: db_release_alias_slot bad deps\n");
    arts_abort(1);
    return;
  }
  unsigned int *d = (unsigned int *)depv[0].ptr;
  d[0] = V0;
  /* Mid-EDT release of the aliased DB.  Path 2 finds the first depv slot with
   * this GUID and runs release_one_dep(alias_only=false). */
  arts_db_release(depv[0].guid, DB_MODE_RW);
  /* Slot 1 (the alias) is still held; the epilogue release_dbs will release it,
   * also with alias_only=false (slot 0 was nulled), double-decrementing
   * writer_count. */
}

/// followup_writer: a plain RW writer.  Only schedulable if writer_count is in
/// a sane (grantable, non-wrapped) state after the aliased EDT released.
void followup_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = V1;
  }
}

/// checker: RO reader; MUST observe the follow-up writer's value (no
/// corruption from a premature ownership transfer caused by the underflow).
void checker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d == NULL || d[0] != V1) {
    (void)fprintf(stderr,
                  "FAIL: db_release_alias_slot follow-up mismatch got 0x%x "
                  "(writer_count underflow / premature transfer)\n",
                  d ? d[0] : 0u);
    arts_abort(1);
    return;
  }
  arts_printf("PASS: db_release_alias_slot\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_release_alias_slot ===\n");

#if defined(ARTS_PROTOCOL_MRMW)
  arts_printf("SKIP db_release_alias_slot: MRMW does not serialize RW (no "
              "alias dedup)\n");
  arts_shutdown();
  return;
#else
  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: aliased EDT releases one alias slot mid-EDT, the other at
   * epilogue. */
  arts_guid_t e_a = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t a =
      arts_edt_create(alias_releaser, 0, NULL, 2,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_a});
  arts_add_dependence(db, a, 0, DB_MODE_RW);
  arts_add_dependence(db, a, 1, DB_MODE_RW);
  arts_event_wait(e_a);

  /* Phase 2: follow-up RW writer — needs a clean writer_count. */
  arts_guid_t e_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t w =
      arts_edt_create(followup_writer, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_w});
  arts_add_dependence(db, w, 0, DB_MODE_RW);
  arts_event_wait(e_w);

  /* Phase 3: RO reader verifies the follow-up value. */
  arts_guid_t e_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t r =
      arts_edt_create(checker_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_r});
  arts_add_dependence(db, r, 0, DB_MODE_RO);
  arts_event_wait(e_r);

  arts_shutdown();
#endif
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
