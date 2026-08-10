/* SPDX-License-Identifier: Apache-2.0
 *
 * excl_destroy_during_acquire — EXCL-config: verify clean destroy AFTER all
 * RW acquirers have quiesced (EXCL protocol).
 *
 * Each iteration fans out N RW workers under a dedicated per-iteration finish
 * event (few), then gates the destroyer on few via DB_MODE_NULL so it only
 * runs after every worker has completed its release.  Destroyer is pinned to
 * the DB home rank (rank 0).  Destroy-in-use is OCR undefined behaviour; this
 * test verifies the runtime stays free of hangs and crashes on the legal,
 * quiesced path across many iterations.
 *
 * Correctness is structural: every iteration's finish event MUST fire (the
 * outer LATCH counts them), which can only happen if no EDT is stranded and
 * the home counter remains balanced.  A stranded waiter or counter imbalance
 * stalls the finish scope → outer scope never completes → ctest TIMEOUT.
 * No SIGSEGV is tolerated.
 *
 * Self-skips on any non-EXCL build at compile time.
 */

#if !defined(ARTS_PROTOCOL_EXCL)
#include <stdio.h>
int main(void) {
  printf("SKIP excl_destroy_during_acquire: EXCL-only\n");
  return 0;
}
#else

#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#define N_EDTS 32
#define N_ITERATIONS 40

/* RW acquirer: non-NULL ptr means it JOINed/was-granted; NULL means the
 * destroy tore the cache down first — both are acceptable, we must not crash.
 */
static void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  volatile int *data = (volatile int *)depv[0].ptr;
  if (data != NULL) {
    int v = *data;
    *data = v + 1;
  }
}

static void destroyer_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t db_guid = (arts_guid_t)paramv[0];
  arts_db_destroy(db_guid);
}

/* relay_edt — bound to one iteration's finish event (slot 0, DB_MODE_NULL).
 * Fires after every member of that finish scope completed; drops the global
 * LATCH so the all-iterations count advances.  paramv: [latch]. */
static void relay_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* shutdown_edt — bound to the all-iterations LATCH (slot 0).  Reaching here
 * means every per-iteration finish scope completed → no stranded waiter, home
 * counter balanced across every destroy-vs-acquire race. */
static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  printf("excl_destroy_during_acquire: %d iters x %d acquirers — PASS\n",
         N_ITERATIONS, N_EDTS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== excl_destroy_during_acquire (%d iter, %d acquirers) ===\n",
              N_ITERATIONS, N_EDTS);

  /* One LATCH counting each iteration's finish event.  shutdown_edt fires when
   * all N_ITERATIONS finish scopes have drained. */
  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(N_ITERATIONS);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  arts_guid_t shut =
      arts_edt_create(shutdown_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(latch, shut, 0, DB_MODE_NULL);

  for (int iter = 0; iter < N_ITERATIONS; iter++) {
    /* Per-iteration finish scope covering the destroyer (dz joins fe so the
     * outer LATCH can count iterations). */
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    int *data = NULL;
    arts_guid_t db = arts_db_create((void **)&data, sizeof(int), ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    if (data != NULL) {
      *data = 0;
    }
    /* Drop the creator hold so the acquirer cohort can proceed. */
    arts_db_release(db, DB_MODE_RW);

    /* Dedicated finish scope for all workers; destroyer depends on it so it
     * only runs after every RW hold has been released. */
    arts_guid_t few = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    for (int i = 0; i < N_EDTS; i++) {
      arts_guid_t w = arts_edt_create(worker_edt, 0, NULL, 1,
                                      &(arts_edt_hint_t){.finish_event = few});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
    }

    /* Destroyer on the DB home rank (rank 0), gated after all workers. */
    uint64_t prm = (uint64_t)db;
    arts_guid_t dz =
        arts_edt_create(destroyer_edt, 1, &prm, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(few, dz, 0, DB_MODE_NULL);

    /* Relay: fires when the iteration's outer finish scope completes (destroyer
     * done); drops the LATCH. */
    uint64_t relay_pv[1] = {(uint64_t)latch};
    arts_guid_t relay = arts_edt_create(relay_edt, 1, relay_pv, 1,
                                        &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(fe, relay, 0, DB_MODE_NULL);
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif /* ARTS_PROTOCOL_EXCL */
