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

/// @file db_acquire_replay_local.c
/// @brief An EDT acquires a LOCAL DB whose DB_CREATE has not run yet
///        (home == self): acquire_one_dep finds the DB absent locally and OoO
///        defers it; arts_db_acquire_replay_dep replays the ONE deferred dep
///        when the DB installs, delivering a valid buffer to the consumer.
///
/// Flow (no busy-wait):
///   - reserve a labeled GUID homed on rank 0 (== self when run single-node).
///   - create a consumer EDT with an RO dep on that GUID *before* the DB
///   exists.
///     The DB->EDT dep immediately satisfies the consumer's dependence count
///     (passive DB model), so the consumer enters its acquire phase while the
///     route entry is still absent -> acquire_one_dep OoO-defers the dep.
///   - create the DB at that exact GUID.  The install fires the OoO drain,
///   which
///     replays the deferred dep -> resolves depv[0].ptr to the
///     freshly-installed buffer and schedules the consumer.
///
/// CONTRACT (race-free): the consumer must run with a valid (non-NULL) buffer
/// pointer.  A missed replay -> the consumer never becomes ready -> the finish
/// event never fires -> arts_event_wait hangs -> ctest TIMEOUT.
///
/// It deliberately does NOT check the buffer CONTENTS.  The consumer's RO
/// acquire resolves at DB-install (the replay, inside arts_db_create), which
/// happens-before any value the creator writes through the pointer returned
/// AFTER arts_db_create returns.  ARTS RO hands back the live buffer (not a
/// snapshot) and same-rank RW<->RO is deliberately not serialized, so observing
/// a creator's post-create write without explicit synchronization is a
/// by-design data race -- not a deterministic signal.  Replay DELIVERY
/// (non-NULL ptr, no hang) is the property under test.
///
/// All configs.  Single-node exercises the home==self defer-replay directly;
/// multinode runs the same with the DB homed on rank 0.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

/// consumer: RO on a DB that did not exist when this EDT was created.  Its DB
/// dep is satisfied only by the OoO replay at install time.  The replay must
/// deliver a valid (non-NULL) buffer pointer; the buffer CONTENTS are a
/// by-design data race against the creator's post-create write and are not
/// checked (see file header).
void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  if (depv[0].ptr == NULL) {
    (void)fprintf(stderr, "FAIL: db_acquire_replay_local got NULL ptr "
                          "(deferred dep not replayed)\n");
    arts_abort(1);
    return;
  }
  arts_printf("PASS: db_acquire_replay_local\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_acquire_replay_local ===\n");

  /* Reserve a labeled GUID homed on rank 0 BEFORE the DB exists. */
  arts_guid_t db = arts_guid_reserve(ARTS_GUID_DB, 0);

  /* Create the consumer EDT with a dependency on the not-yet-created DB.  Its
   * acquire (when it becomes ready) will find the route entry absent and OoO
   * defer the dep, to be replayed once the DB installs. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t c =
      arts_edt_create(consumer_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, c, 0, DB_MODE_RO);

  /* Now create the DB at that exact GUID.  The install fires the OoO drain,
   * replaying the consumer's deferred dep and resolving its depv[0].ptr to the
   * freshly-installed buffer.  No value is written through `ptr`: the
   * consumer's replay-resolve already happened-before this point (inside
   * arts_db_create), so any write here would race the consumer's read (by
   * design — see header). */
  (void)arts_db_create_with_guid(db, sizeof(unsigned int), ARTS_DB,
                                 ARTS_DB_PROP_NONE, NULL);
  arts_db_release(db, DB_MODE_RW);

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
