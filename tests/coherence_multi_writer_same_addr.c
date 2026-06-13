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

/// @file coherence_multi_writer_same_addr.c
/// @brief B.5 -- Multi-writer same-address eager-protocol determinism check.
///
/// N RW EDTs increment the same int in the same DB, then a single RO EDT
/// verifies the final count.  Tests that the ownership protocol delivers a
/// coherent view of the buffer to every RW acquirer regardless of cross-rank
/// GRANT timing.
///
/// IMPORTANT: ARTS RW is per-NODE exclusive (cross-rank LOCK_REQ chain),
/// not per-EDT exclusive.  Multiple RW EDTs on the same rank can run
/// concurrently.  The increment must therefore be atomic; a plain
/// `(*data)++` would lose updates under multi-EDT same-node concurrency
/// (TSan reproduces this on stress).  This is per the ARTS RW contract
/// documented in CLAUDE.md / memory feedback-rw-mode-semantics.md:
/// "Multi-EDT same-node RW concurrency is permitted; the application is
/// responsible for ordering same-address writes via atomics or explicit
/// EDT-graph happens-before."
///
/// What this test verifies (post-atomic-increment):
///   1. The ownership protocol routes every RW acquirer to a buffer that
///      becomes visible to subsequent acquirers (cache->buffer + writer_count
///      handoff).
///   2. The final RO acquirer observes the stable post-all-RW value.
///   3. No EDTs are stranded (all N + 1 finish; outer finish scope shutdown
///      fires cleanly).
///
/// Spec section 6 B.5.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define N 100

static atomic_int g_clean_shutdown = 0;

/// inc_edt -- atomic increment + LATCH decrement.
/// paramv[0] = LATCH event GUID with initial count N.  Decrementing
/// produces a happens-before from this body to verify_edt's slot 1
/// (which depends on the event firing).
static void inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  /* ARTS RW = per-node exclusive (cross-rank LOCK_REQ chain) but NOT
   * per-EDT exclusive on the same node.  Use atomic_fetch_add so
   * concurrent same-node EDTs serialize the read-modify-write
   * themselves; the ownership protocol's job is buffer visibility, not
   * per-EDT mutual exclusion. */
  _Atomic int *data = (_Atomic int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: inc_edt got NULL ptr\n");
    arts_abort(1);
  }
  atomic_fetch_add_explicit(data, 1, memory_order_relaxed);
  /* Drop the latch.  When the Nth inc_edt drops it to 0, the event
   * fires and verify_edt's slot 1 gets satisfied. */
  arts_guid_t evt = (arts_guid_t)paramv[0];
  arts_event_satisfy_slot(evt, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic int *data = (_Atomic int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: verify_edt got NULL ptr\n");
    arts_abort(1);
  }
  int observed = atomic_load_explicit(data, memory_order_relaxed);
  if (observed != N) {
    (void)fprintf(stderr,
                  "FAIL: expected %d, got %d (ownership visibility bug)\n", N,
                  observed);
    arts_abort(1);
  }
  atomic_store(&g_clean_shutdown, 1);
  arts_printf("PASS: %d atomic RW increments produced %d\n", N, observed);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_multi_writer_same_addr (N=%d) ===\n", N);

  int *data;
  arts_guid_t db = arts_db_create((void **)&data, sizeof(int), ARTS_DB,
                                  ARTS_DB_PROP_NONE, NULL);
  *data = 0;

  /* LATCH event with count = N: each inc_edt decrements once, so the
   * Nth decrement fires the event and unblocks verify_edt's slot 1.
   * This is the canonical "fan-in barrier" producer -> event ->
   * consumer happens-before that ARTS RW does NOT enforce on its own
   * (RW is per-NODE exclusive, not per-EDT).
   *
   * life_count=INT32_MAX makes the event persistent so verify_edt's
   * add_dep can race with the Nth satisfy: even after the event fires the
   * late-binder fast-path delivers the stored data (fire-and-linger). */
  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(N);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);

  uint64_t inc_paramv[1] = {(uint64_t)latch};
  for (int i = 0; i < N; i++) {
    arts_guid_t w = arts_edt_create(inc_edt, /*paramc=*/1, inc_paramv,
                                    /*depc=*/1, NULL);
    arts_add_dependence(db, w, /*slot=*/0, DB_MODE_RW);
  }

  /* verify_edt: slot 0 = DB (RO snapshot of final state); slot 1 =
   * LATCH event (fires after all N RW EDTs drop the latch).  The
   * event chain guarantees verify_edt runs strictly AFTER the last
   * inc_edt body, so the RO snapshot includes every increment. */
  arts_guid_t v = arts_edt_create(verify_edt, /*paramc=*/0, NULL,
                                  /*depc=*/2, NULL);
  arts_add_dependence(db, v, /*slot=*/0, DB_MODE_RO);
  arts_add_dependence(latch, v, /*slot=*/1, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  if (arts_get_current_rank() == 0 && !atomic_load(&g_clean_shutdown)) {
    (void)fprintf(stderr,
                  "FAIL: verify_edt did not fire cleanly — abort or premature "
                  "shutdown\n");
    return 1;
  }
  return 0;
}
