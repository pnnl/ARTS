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

/// @file destroy_before_create_defer.c
/// @brief T114 — request/writeback before create → Cat-B OoO defer + replay on
///        DB_CREATE drain (B019).
///
/// A coherence request that reaches the home before the owning DB_CREATE
/// installs the home directory is a Cat-B deferrable op: the OoO engine defers
/// it on the GUID slot and replays it when DB_CREATE drains.  DB_CREATE itself
/// is NON-deferrable — it is the drain trigger.  Two families exercise this:
///   - SNAPSHOT_REQUEST (RO acquire) — non-LOCK protocols;
///   - WRITEBACK (RW release) — EAGER / MRMW only.
/// Critically, the LAZY WRITEBACK sender must never be reached with home==self
/// (B019): the self-send fast path is `#if !ARTS_TIMING_LAZY`-excluded, so
/// under LAZY a stray writeback would fall to the async send → self_send_check
/// drops it → silent lost writeback hang.  This test self-skips LOCK and only
/// drives the RW-writeback leg under EAGER/MRMW; the RO-snapshot leg runs on
/// all non-LOCK.
///
/// Black-box driver: a reserved (labeled) DB GUID home=0 with a remote acquirer
/// in the SAME generation as the home create.  The remote's request can land
/// before CREATE installs (wire reorder, widest under 2n_io); the defer/replay
/// path must deliver the value correctly with no lost request.  A stranded
/// deferred op (never drained) is caught by the ctest TIMEOUT; a lost write by
/// the in-EDT assertion.
///
/// Harness: runtime_multinode (the before-create reorder needs a remote
/// sender); clean SKIP single-node.  Config gate: self-skip LOCK.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if defined(ARTS_PROTOCOL_LOCK)
int main(void) {
  printf(
      "SKIP destroy_before_create_defer: non-LOCK (snapshot/writeback) only\n");
  return 0;
}
#else

#define ITERS 200u
#define SENTINEL_BASE 0x60000000u

/// creator: owns the labeled GUID; creates + writes the per-iter sentinel.
static void creator_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t reserved = (arts_guid_t)paramv[0];
  unsigned int sentinel = (unsigned int)paramv[1];
  unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(
      reserved, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (ptr != NULL) {
    ptr[0] = sentinel;
  }
  arts_db_release(reserved, DB_MODE_RW);
}

/// remote RW writer: forces a WRITEBACK round on release whose message can
/// reach the home before CREATE installs (EAGER/MRMW only — see gate below).
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/// remote RO reader: SNAPSHOT_REQUEST can reach the home before CREATE
/// installs; after defer+replay it MUST observe the expected value.
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const unsigned int *data = (const unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (data == NULL || data[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL: lost deferred request — expected 0x%x got 0x%x\n",
                  expect, data ? data[0] : 0u);
    arts_abort(1);
  }
}

/// launcher: owns ONE generation's inner finish scope.  A finish event's
/// creator-token is released when the EDT that created it completes; the driver
/// blocks on the outer wait and so cannot release per-generation tokens, so
/// each generation's inner scope is owned by this short-lived EDT that returns
/// immediately — its completion releases the inner token, letting inner fire
/// once the creator (and, under EAGER/MRMW, the writer) finish.
static void launcher_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t outer = (arts_guid_t)paramv[0];
  unsigned int sentinel = (unsigned int)paramv[1];

  /* The value the reader must ultimately observe.  Under EAGER/MRMW a remote
   * RW writer runs after the creator and overwrites it; the reader then expects
   * the writer's value.  Under LAZY there is no writeback leg, so the reader
   * expects the creator's sentinel. */
#if !defined(ARTS_TIMING_LAZY)
  unsigned int expect = sentinel ^ 1u;
#else
  unsigned int expect = sentinel;
#endif

  arts_guid_t reserved = arts_guid_reserve(ARTS_GUID_DB, 0);

  /* Reader (remote RO), gated on this generation's inner finish scope (slot 1)
   * so all writes are causally complete before it reads; its SNAPSHOT_REQUEST
   * can still race CREATE at the home (before-create reorder -> Cat-B defer).
   */
  uint64_t rparam = (uint64_t)expect;
  arts_guid_t reader =
      arts_edt_create(reader_edt, 1, &rparam, 2,
                      &(arts_edt_hint_t){.rank = 1, .finish_event = outer});
  arts_add_dependence(reserved, reader, 0, DB_MODE_RO);

  arts_guid_t inner = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(inner, reader, 1, DB_MODE_NULL);

  /* Creator on home rank 0 inside this generation's inner finish scope:
   * installs the DB and stamps the sentinel.  It is the first RW holder. */
  uint64_t cparams[2] = {(uint64_t)reserved, (uint64_t)sentinel};
  arts_guid_t creator =
      arts_edt_create(creator_edt, 2, cparams, 0,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = inner});
  (void)creator;

#if !defined(ARTS_TIMING_LAZY)
  /* RW-writeback leg (EAGER/MRMW only): a remote RW writer ordered strictly
   * AFTER the creator by the DB's per-node exclusive RW lease (serialized), so
   * the write->read chain is unambiguous.  Its release sheds ownership with a
   * synchronous WRITEBACK whose message can reach the home before / around
   * CREATE — exercising Cat-B defer of WRITEBACK.  Statically excluded under
   * LAZY, which has no synchronous writeback (B019). */
  uint64_t wparam = (uint64_t)expect;
  arts_guid_t writer =
      arts_edt_create(writer_edt, 1, &wparam, 1,
                      &(arts_edt_hint_t){.rank = 1, .finish_event = inner});
  arts_add_dependence(reserved, writer, 0, DB_MODE_RW);
  (void)writer;
#endif
  /* Return: completion cleanup releases inner's creator-token. */
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int ranks = arts_get_total_ranks();
  if (ranks < 2) {
    arts_printf("SKIP: destroy_before_create_defer requires node_count >= 2\n");
    arts_shutdown();
    return;
  }

  arts_printf("=== destroy_before_create_defer (%u ranks) ===\n", ranks);

  arts_guid_t outer = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Each generation runs under its own launcher EDT (home rank) joined to
   * outer; the launcher owns that generation's inner finish scope.  The driver
   * owns and waits ONLY outer, holding no per-generation finish token across
   * the wait. */
  for (unsigned int it = 0; it < ITERS; it++) {
    unsigned int sentinel = SENTINEL_BASE + it;
    uint64_t lparams[2] = {(uint64_t)outer, (uint64_t)sentinel};
    arts_edt_create(launcher_edt, 2, lparams, 0,
                    &(arts_edt_hint_t){.rank = 0, .finish_event = outer});
  }

  arts_event_wait(outer);
  arts_printf("PASS: destroy_before_create_defer %u iters x %u ranks\n", ITERS,
              ranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_LOCK */
