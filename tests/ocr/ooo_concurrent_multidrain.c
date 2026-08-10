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

/// @file ooo_concurrent_multidrain.c
/// @brief Exercises the OoO engine's concurrent-multidrain path: many deferred
///        operations accumulate on a freshly-reserved route slot and are then
///        replayed by a create-handler drain RACING the producers' own
///        TOCTOU-rescue drains (ooo.c arts_ooo_drain).  The invariant under
///        test: every deferred op lands in EXACTLY one drain snapshot (none
///        stranded, none replayed twice), even though FIFO order is not
///        globally preserved across two concurrent disjoint snapshots.
///
/// Drive (public-API only, since the OoO engine has no public entry):
///   A single home DB (rank 0).  A wide fan-out of RW writer EDTs, spread
///   across every rank.  Each writer's RW acquire sends an ownership/lock
///   request toward the home; many of these requests arrive while the home is
///   busy installing/transferring, so they queue on the slot's ooo_list and
///   are replayed by the create/transfer drain.  We do NOT rely on the answer
///   being any particular writer's value (cross-snapshot order is not FIFO);
///   we rely on COMPLETION — every writer must run exactly once and the finish
///   scope must close.  A stranded (lost-wakeup) deferred op would hang the
///   finish scope and be reaped by the ctest TIMEOUT.
///
/// A per-writer atomic counter in a separate node-pinned bookkeeping DB proves
/// each writer ran exactly once (no double-replay, no drop).  Config-agnostic:
/// all protocols route a remote RW acquire through the OoO defer path.  On 1n
/// the deferral is local but the drain logic is identical.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>

#define NUM_WRITERS 24u

/// Bookkeeping DB: one atomic ran-count per writer plus a global total.
typedef struct {
  _Atomic unsigned int ran[NUM_WRITERS];
  _Atomic unsigned int total;
} book_t;

/// Writer: bumps its own ran-count exactly once and writes its id into the
/// shared payload DB (depv[0]=payload RW, depv[1]=book RW).
void mw_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int id = (unsigned int)paramv[0];
  unsigned int *payload = (unsigned int *)depv[0].ptr;
  book_t *book = (book_t *)depv[1].ptr;
  if (payload != NULL) {
    payload[0] = id; /* last writer wins; value not asserted */
  }
  if (book != NULL) {
    atomic_fetch_add_explicit(&book->ran[id], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&book->total, 1u, memory_order_relaxed);
  }
}

/// Verifier: reads the book RO and asserts each writer ran exactly once.
void mw_verify(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  book_t *book = (book_t *)depv[0].ptr;
  bool ok = (book != NULL);
  if (ok) {
    unsigned int total =
        atomic_load_explicit(&book->total, memory_order_relaxed);
    ok = (total == NUM_WRITERS);
    for (unsigned int i = 0; i < NUM_WRITERS; i++) {
      unsigned int n =
          atomic_load_explicit(&book->ran[i], memory_order_relaxed);
      if (n != 1u) {
        ok = false;
        arts_printf("  FAIL: writer %u ran %u times (expected 1)\n", i, n);
      }
    }
    if (ok) {
      arts_printf("  total writers replayed: %u\n", total);
    }
  }
  if (ok) {
    arts_printf("PASS: ooo_concurrent_multidrain all %u writers ran once\n",
                NUM_WRITERS);
  } else {
    arts_printf("FAIL: ooo_concurrent_multidrain\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== ooo_concurrent_multidrain ===\n");

  unsigned int nranks = arts_get_total_ranks();

  /* Shared payload DB, home rank 0. */
  void *pptr = NULL;
  arts_guid_t payload =
      arts_db_create(&pptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)pptr)[0] = 0u;
  arts_db_release(payload, DB_MODE_RW);

  /* Bookkeeping DB, home rank 0. */
  void *bptr = NULL;
  arts_guid_t book_db =
      arts_db_create(&bptr, sizeof(book_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  book_t *book = (book_t *)bptr;
  for (unsigned int i = 0; i < NUM_WRITERS; i++) {
    atomic_init(&book->ran[i], 0u);
  }
  atomic_init(&book->total, 0u);
  arts_db_release(book_db, DB_MODE_RW);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Wide RW fan-out across every rank: each writer acquires payload RW.  The
   * concurrent remote acquires force many requests onto the home slot's
   * ooo_list, replayed by concurrent create/transfer drains. */
  for (unsigned int i = 0; i < NUM_WRITERS; i++) {
    uint64_t pv = (uint64_t)i;
    unsigned int target = nranks > 1 ? (i % nranks) : 0u;
    arts_guid_t w =
        arts_edt_create(mw_writer, 1, &pv, 2,
                        &(arts_edt_hint_t){.rank = target, .finish_event = fe});
    arts_add_dependence(payload, w, 0, DB_MODE_RW);
    arts_add_dependence(book_db, w, 1, DB_MODE_RW);
  }

  /* Verifier runs after every writer (finish-scope ordering via wait below). */
  arts_event_wait(fe);

  arts_guid_t ve = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t v = arts_edt_create(
      mw_verify, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = ve});
  arts_add_dependence(book_db, v, 0, DB_MODE_RO);
  arts_event_wait(ve);

  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
