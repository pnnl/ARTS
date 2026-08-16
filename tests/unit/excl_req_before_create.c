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

/// @file coherence_lock_req_before_create.c
/// @brief B.3 — LOCK_REQ-before-DB_CREATE race regression test.
///
/// 3-rank scenario: creator(0), home(1), consumer(2).
///   Creator  -> home:     DB_CREATE / DB_SEND    (rank 0 -> 1)
///   Creator  -> writer:   EDT spawn (init data)  (rank 0 -> 1)
///   Creator  -> consumer: EDT spawn              (rank 0 -> 2)
///   Consumer -> home:     cross-node EW request  (rank 2 -> 1)
///
/// The race is between rank 0->1 (DB metadata) and rank 2->1 (the consumer's
/// cross-node EW request).  Different sources, no FIFO ordering across the
/// pair.
///
/// Legacy: home-side LOCK_REQ may arrive
/// before the cache_s is installed -> NULL cache -> DESTROY_NOTIFY ->
/// consumer sees NULL ptr -> SIGSEGV.
/// Post-LOCK_REQ deferred vian OoO list -> DB_CREATE arrives ->
/// fire_oo re-issues LOCK_REQ -> consumer sees correct data.
///
/// Adaptations vs. plan code:
///   - ARTS_DB (final) is not defined in HEAD; legacy ARTS_DB_DEFAULT
///     names the same enum value.
///   - arts_db_create has a 4-arg signature (no ARTS_DB_PROP_NONE).
///   - DB_MODE_RW is documented as LOCAL-DB-only in arts.h; the cross-node
///     ordered-write mode in HEAD is DB_MODE_RW.
///   - arts_db_create on a remote route returns *addr = NULL, so the data
///     initialization is performed by a writer EDT pinned on the home rank
///     (acquires DB in EW mode, writes the sentinel, releases).
///   - arts_init_main does not exist; the runtime invokes the global
///     main_edt symbol on rank 0 automatically from arts_rt().
///
/// Spec section 6 B.3.

#include "arts.h"

#include <stdio.h>

/// Number of iterations of the race scenario per test run.
#define N_ITERATIONS 100

/// Sentinel value the writer stamps into the DB.
#define SENTINEL 42

/// Writer EDT -- pinned on the DB home rank (rank 1).  Writes the sentinel
/// into the DB, then satisfies a LATCH event so the consumer's slot 1 is
/// satisfied AFTER the writer body completes.  This event chain enforces
/// the writer -> consumer happens-before that data-race-free OCR semantics
/// require: the runtime's RW serialization at home only guarantees that
/// concurrent RW acquires take turns -- it does NOT guarantee that the
/// turn order matches user wiring order.  Without an explicit event chain
/// the test was racy on the data value (consumer could read 0 if it won
/// the LOCK_REQ arrival race at home).
///
/// paramv[0] = LATCH event GUID for the iteration.
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: writer got NULL ptr\n");
    arts_abort(1);
  }
  *data = SENTINEL;
  /* Fire the LATCH event so the consumer's parked slot 1 is satisfied.
   * Writer's release_rw runs after this body returns, so the consumer's
   * subsequent acquire is guaranteed to land at home AFTER writer's
   * WB_AND_TRANSFER.  This is the OCR-canonical producer -> event ->
   * consumer happens-before pattern. */
  arts_guid_t evt = (arts_guid_t)paramv[0];
  arts_event_satisfy_slot(evt, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/// Consumer EDT -- pinned on rank 2.  Two deps: slot 0 = DB (RW), slot 1 =
/// LATCH event satisfied by the writer.  The event chain ensures consumer
/// runs AFTER writer's body completes, so the RW acquire at home arrives
/// after writer's WB_AND_TRANSFER and the consumer reads SENTINEL
/// deterministically.
static void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: consumer got NULL ptr\n");
    arts_abort(1);
  }
  if (*data != SENTINEL) {
    arts_printf("FAIL: data corrupted (%d != %d)\n", *data, SENTINEL);
    arts_abort(1);
  }
}

/// Finish-EDT for the outer finish scope — runs on rank 0 after every
/// iteration's EDTs have completed and released their DB references.  Clean
/// shutdown.
static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: %d iterations completed\n", N_ITERATIONS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int rank_count = arts_get_total_ranks();
  if (rank_count < 3) {
    arts_printf("SKIP: requires 3+ ranks (got %u)\n", rank_count);
    arts_shutdown();
    return;
  }

#if defined(ARTS_PROTOCOL_WRF_VAL)
  /* Relies on OCR RW serialization: the consumer's RW acquire observes the
   * writer's published value only because a coherent protocol blocks the
   * acquire until the writer releases (publishes).  WRF_VAL is DB-WRF — the RW
   * acquire is an unserialized GET_DATA snapshot, and the writer satisfies the
   * event before its release (satisfy-before-release), so the consumer can read
   * home before the publish publishes and observe a stale value.  Not a
   * defined program under DB-WRF. */
  arts_printf(
      "SKIP excl_req_before_create: relies on RW serialization (WRF_VAL is "
      "DB-WRF)\n");
  arts_shutdown();
  return;
#endif

  arts_printf("=== coherence_lock_req_before_create (%d iterations, ranks=%u)"
              " ===\n",
              N_ITERATIONS, rank_count);

  /* Outer finish scope: shutdown_edt fires only after every iteration's writer
   * + consumer have run AND released their DB refs.  Without this fence we
   * would race shutdown against the in-flight cross-node EW transfers. */
  arts_guid_t shut =
      arts_edt_create(shutdown_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  for (int iter = 0; iter < N_ITERATIONS; iter++) {
    /* Step 1: rank 0 creates DB on rank 1 (home).  Wire message
     * DB_CREATE_COHERENT travels rank 0 -> 1.  arts_db_create returns
     * *addr = NULL on remote create (data init is via writer EDT). */
    void *raw = NULL;
    arts_guid_t db =
        arts_db_create(&raw, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 1});

    /* Step 2: ONCE-equivalent (defaults: latch=1, auto_destroy=true) --
     * writer satisfies, consumer waits.  Hosted on rank 0 so the
     * decrement-on-satisfy round-trip is short. */
    arts_event_hint_t evt_hint = ARTS_EVENT_HINT_DEFAULTS;
    evt_hint.rank = 0;
    arts_guid_t evt = arts_event_create(&evt_hint);

    /* Step 3: writer on rank 1.  Slot 0 = DB (RW); paramv[0] = event
     * GUID so writer can satisfy after writing. */
    uint64_t writer_paramv[1] = {(uint64_t)evt};
    arts_guid_t writer =
        arts_edt_create(writer_edt, /*paramc=*/1, writer_paramv, /*depc=*/1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    arts_add_dependence(db, writer, /*slot=*/0, DB_MODE_RW);

    /* Step 4: consumer on rank 2.  Slot 0 = DB (RW), slot 1 = LATCH
     * event.  The event chain (writer -> evt -> consumer) is the
     * canonical OCR happens-before that races below the RW serialization
     * at home -- without it, consumer's LOCK_REQ could arrive at home
     * BEFORE writer's, and home's strict-arrival FIFO grants ownership
     * to consumer first.  This is the LOCK_REQ-before-DB_CREATE race
     * regression test (OoO defer must keep the consumer from
     * SIGSEGV'ing on a NULL ptr in that ordering); the data check below
     * verifies that the OCR happens-before chain is honored end-to-end. */
    arts_guid_t consumer =
        arts_edt_create(consumer_edt, /*paramc=*/0, NULL, /*depc=*/2,
                        &(arts_edt_hint_t){.rank = 2, .finish_event = fe});
    arts_add_dependence(db, consumer, /*slot=*/0, DB_MODE_RW);
    arts_add_dependence(evt, consumer, /*slot=*/1, DB_MODE_NULL);
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
