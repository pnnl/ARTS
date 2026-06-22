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

/// @file create_coalesce.c
/// @brief T113 — DB_CREATE coalesce vs lazy_install winner; home_initialized
///        double-init (B015/B016).
///
/// On the home rank a DB's route-table entry can appear via two paths that
/// race: (a) the owning DB_CREATE handler, and (b) a foreign rank's acquire
/// that lazy-installs a cache stub at the home before CREATE arrives.  Whoever
/// loses must coalesce / promote-in-place rather than re-initialize:
///   - `home_initialized` must transition exactly once (B015 — a non-atomic
///     read-test-set across the two coalesce sites would double `home_init`,
///     corrupting the MPSC head / double-allocating the map);
///   - the `buf_absent` TOCTOU must not leave a version regression or leak a
///     redundant buffer (B016).
///
/// Black-box driver: a reserved (labeled) DB GUID with home=0 is acquired RO by
/// a remote rank in the SAME generation the home creates it — the remote
/// acquire lazy-installs the stub, racing the home's CREATE.  The reader
/// carries only its RO dependence (no control gate) so it dispatches
/// immediately and its acquire genuinely races CREATE; the DB's RW->RO
/// coherence then orders the read after the creator's release, so it MUST
/// observe the creator's sentinel (no lost write / version regression).  Both
/// the reader and the creator join a single finish scope owned and waited by
/// the driver — the only finish event — so no per-generation finish token is
/// held across the wait.  Repeated over many generations to widen the race
/// window.  A coalesce that hangs (double-init corrupting the directory) is
/// caught by the ctest TIMEOUT; a version regression is caught by the in-EDT
/// assertion.
///
/// Harness: runtime_multinode — the lazy_install-vs-create race only exists
/// with a remote acquirer, so this needs 2+ ranks (clean SKIP single-node).
/// Config-agnostic (the create/coalesce path is protocol-shared).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ITERS 200u
#define SENTINEL_BASE 0x50000000u

/// creator: owns the labeled GUID; creates + writes the per-iter sentinel.
static void creator_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  arts_guid_t reserved = (arts_guid_t)paramv[0];
  unsigned int sentinel = (unsigned int)paramv[1];
  (void)paramc;
  unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(
      reserved, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (ptr != NULL) {
    ptr[0] = sentinel;
  }
  arts_db_release(reserved, DB_MODE_RW);
}

/// reader: remote RO acquire that races the home CREATE's coalesce; its
/// acquire LOCK_REQ / SNAPSHOT_REQUEST may arrive before or after CREATE.  The
/// DB's RW->RO coherence orders it after the creator's RW release, so it MUST
/// observe the creator's sentinel (no lost write / version regression).
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const unsigned int *data = (const unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (data == NULL || data[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL: coalesce lost write — expected 0x%x got 0x%x\n",
                  expect, data ? data[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int ranks = arts_get_total_ranks();
  if (ranks < 2) {
    arts_printf("SKIP: create_coalesce requires node_count >= 2\n");
    arts_shutdown();
    return;
  }

  arts_printf("=== create_coalesce (%u ranks) ===\n", ranks);

  arts_guid_t outer = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (unsigned int it = 0; it < ITERS; it++) {
    unsigned int sentinel = SENTINEL_BASE + it;
    arts_guid_t reserved = arts_guid_reserve(ARTS_GUID_DB, 0);

    /* Reader on a remote rank with a single RO dependence and no control gate,
     * so it dispatches immediately: its RO acquire lazy-installs the home stub
     * and genuinely races the home's own CREATE coalesce.  When the acquire
     * lands before CREATE it defers (OoO) until the DB exists; the DB's RW->RO
     * coherence then orders it after the creator's RW release, so it observes
     * the published sentinel (no manual finish gate needed for the value). */
    uint64_t rparam = (uint64_t)sentinel;
    arts_guid_t reader =
        arts_edt_create(reader_edt, 1, &rparam, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = outer});
    arts_add_dependence(reserved, reader, 0, DB_MODE_RO);

    /* Creator on home rank 0, joined to the same finish scope. */
    uint64_t cparams[2] = {(uint64_t)reserved, (uint64_t)sentinel};
    arts_edt_create(creator_edt, 2, cparams, 0,
                    &(arts_edt_hint_t){.rank = 0, .finish_event = outer});
  }

  /* Drain every generation, then end. */
  arts_event_wait(outer);
  arts_printf("PASS: create_coalesce %u iters x %u ranks\n", ITERS, ranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
