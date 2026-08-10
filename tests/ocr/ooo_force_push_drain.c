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

/// @file ooo_force_push_drain.c
/// @brief Asserts the OoO "force-push then drain" eventual-replay invariant:
///        a deferred operation pushed onto a route slot whose target object is
///        not yet installed must be replayed by a later create-handler drain
///        and NOT stranded (ooo.c arts_ooo_push_guid / arts_ooo_drain_guid).
///
/// The force-push entry (arts_ooo_push_guid) is internal to the GPU-LC
/// invalidation path and has no public API.  This test drives the SAME engine
/// invariant through the public API's before-install defer path: an operation
/// that targets a GUID is issued while that GUID's object is still absent from
/// the route table, so dispatch_or_defer MISSes and pushes onto the slot's
/// ooo_list; the object is then installed and its create handler drains the
/// chain.  The deferred op must fire.
///
/// Construction (single-node deterministic, runs on every config):
///   1. main reserves a DB GUID `db` but does NOT create it yet.
///   2. main creates a consumer EDT `c` (with a reserved GUID) that depends on
///      `db` in slot 0 (RW).  The dependency is registered against a route
///      slot that has no DB installed → the satisfy/acquire is deferred.
///   3. main creates the DB at `db` and releases it RW.  The DB-create handler
///      installs the object and drains the slot, replaying the deferred dep so
///      `c` becomes runnable.
///   4. `c` must run; it signals its finish event.  A stranded (never-drained)
///      deferred op would leave the finish scope open → ctest TIMEOUT reaps it.
///
/// No in-test watchdog/spin: completion is proven by arts_event_wait returning
/// and the PASS line printing; a strand manifests as a TIMEOUT failure.

#include "arts.h"

#include <stdint.h>

#define SENTINEL 0xABCD1234u

/// Consumer: fires only after the deferred db dependency is replayed.  The
/// invariant under test is that the replay DELIVERS the DB buffer (d != NULL) —
/// i.e. the deferred op was not stranded.
///
/// The buffer CONTENT (== SENTINEL) is deliberately NOT asserted: under the OCR
/// intra-node multi-writer RW model the creator's RW hold does not serialize
/// this consumer — the replayed RW acquire JOINS the held RW phase (it does not
/// wait for the creator's release), so on a multi-worker run the consumer can
/// read the buffer concurrently with the creator's stamp.  Pinning the content
/// would need an explicit producer->event->consumer happens-before, which is
/// orthogonal to the OoO-replay invariant this test exists to check (and would
/// make the test racy, as a bare DB RW dep does not order intra-node writers).
void fpd_consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const unsigned int *d = (const unsigned int *)depv[0].ptr;
  if (d != NULL) {
    arts_printf("PASS: ooo_force_push_drain deferred dep replayed "
                "(DB delivered, val 0x%x)\n",
                d[0]);
  } else {
    arts_printf("FAIL: ooo_force_push_drain consumer got NULL ptr "
                "(deferred dep not delivered)\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== ooo_force_push_drain ===\n");

  /* Reserve the DB GUID up front so the dependency can be wired before the DB
   * object exists in the route table (forces the OoO defer path). */
  arts_guid_t db = arts_guid_reserve(ARTS_GUID_DB, 0);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Consumer EDT depends on the not-yet-created DB.  The dependency lands on a
   * route slot with value==NULL → the acquire/satisfy is deferred. */
  arts_guid_t c =
      arts_edt_create(fpd_consumer, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, c, 0, DB_MODE_RW);

  /* Now create the DB at the reserved GUID.  The create handler installs the
   * object and drains the slot, replaying the deferred dep.  The creator
   * auto-acquires RW: stamp the sentinel, then release so the consumer's RW
   * acquire (replayed by the drain) can take ownership. */
  unsigned int *dptr = (unsigned int *)arts_db_create_with_guid(
      db, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
      &(arts_db_hint_t){.rank = 0});
  dptr[0] = SENTINEL;
  arts_db_release(db, DB_MODE_RW);

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
