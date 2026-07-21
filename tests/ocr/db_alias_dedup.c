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

/// @file db_alias_dedup.c
/// @brief Same DB GUID bound to TWO RW dep slots of one EDT: the acquire-side
///        alias dedup (rw_fire_from_cursor reentrant branch) must exactly match
///        the release-side dedup (release_dbs alias_only), with no writer_count
///        under/over-count.
///
/// This is the single hardest acquire/release accounting path in db.c.  An EDT
/// that names the same coherent ARTS_DB in slot 0 (RW) and slot 1 (RW):
///   - rw_fire_from_cursor secures slot 0 via a real ownership round, then on
///     slot 1 detects a serialized GUID-equal predecessor (reentrant=true) and
///     resolves it as a LOCAL HIT (a fresh per-slot buffer ref, NOT a second
///     ownership round — a second round would self-deadlock behind the EDT's
///     own unreleased hold under a single-writer protocol).
///   - release_dbs mirrors this: slot 1 is alias_only (drop buffer ref only,
///     no writer_count decrement); slot 0 is the real release.
///
/// If the two sides disagree, writer_count under/over-counts: a subsequent
/// writer either never gets ownership (stuck — caught by ctest TIMEOUT) or
/// gets it while a phantom hold remains (data corruption — caught by the
/// follow-up reader's value check -> arts_abort).
///
/// Both slots point at the SAME buffer, so a write through depv[0].ptr must be
/// visible through depv[1].ptr (same DB).  We verify that, then a follow-up RW
/// writer + RO reader chain proves writer_count returned cleanly to a grantable
/// state after the aliased EDT released.
///
/// Config: ownership protocols (RCU) serialize RW so the reentrant
/// branch is live.  Under RWLOCK RW is also serialized (alias dedup applies).
/// Under WRF_RCU RW is NOT serialized — the reentrant branch is a no-op but the
/// test is still a valid two-slot-same-DB correctness check, so it runs in all
/// configs.  Config-agnostic across 1n..4n.  A stranded EDT is caught by the
/// ctest TIMEOUT (no in-test watchdog).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define MAGIC 0xA11A5u

/// aliased: receives the SAME DB in slot 0 (RW) and slot 1 (RW).  Both ptrs
/// must be non-NULL and must reference the same payload.  Write MAGIC through
/// slot 0; read it back through slot 1.
void aliased_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  if (depc < 2) {
    (void)fprintf(stderr, "FAIL: db_alias_dedup expected depc>=2 got %u\n",
                  depc);
    arts_abort(1);
    return;
  }
  unsigned int *a = (unsigned int *)depv[0].ptr;
  unsigned int *b = (unsigned int *)depv[1].ptr;
  if (a == NULL || b == NULL) {
    (void)fprintf(stderr, "FAIL: db_alias_dedup NULL alias ptr (%p,%p)\n",
                  (void *)a, (void *)b);
    arts_abort(1);
    return;
  }
  a[0] = MAGIC;
  /* Same DB: the write through slot 0 must be visible through slot 1. */
  if (b[0] != MAGIC) {
    (void)fprintf(stderr,
                  "FAIL: db_alias_dedup alias slots not same DB (b=0x%x)\n",
                  b[0]);
    arts_abort(1);
  }
}

/// followup: a plain RW writer on the same DB — proves ownership is grantable
/// again after the aliased EDT released (writer_count returned to a clean
/// state).  Bumps the value.
void followup_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = MAGIC + 1u;
  }
}

/// checker: RO reader, MUST observe the follow-up writer's value.
void checker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d == NULL || d[0] != MAGIC + 1u) {
    (void)fprintf(stderr, "FAIL: db_alias_dedup follow-up mismatch got 0x%x\n",
                  d ? d[0] : 0u);
    arts_abort(1);
    return;
  }
  arts_printf("PASS: db_alias_dedup\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_alias_dedup ===\n");

  /* Home the DB on rank 0 so the whole chain runs locally; the accounting
   * path (reentrant dedup) is identical regardless of node count. */
  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: the aliased EDT (same DB in two RW slots). */
  arts_guid_t e_alias = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t a =
      arts_edt_create(aliased_edt, 0, NULL, 2,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_alias});
  arts_add_dependence(db, a, 0, DB_MODE_RW);
  arts_add_dependence(db, a, 1, DB_MODE_RW);
  arts_event_wait(e_alias);

  /* Phase 2: a follow-up RW writer — only schedulable if writer_count is back
   * to a grantable state (no phantom hold left by the aliased EDT). */
  arts_guid_t e_wr = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t w =
      arts_edt_create(followup_writer, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_wr});
  arts_add_dependence(db, w, 0, DB_MODE_RW);
  arts_event_wait(e_wr);

  /* Phase 3: RO reader verifies the follow-up value (no corruption from an
   * over-count that would have transferred ownership prematurely). */
  arts_guid_t e_rd = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t r =
      arts_edt_create(checker_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_rd});
  arts_add_dependence(db, r, 0, DB_MODE_RO);
  arts_event_wait(e_rd);

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
