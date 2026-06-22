/* SPDX-License-Identifier: Apache-2.0
 *
 * lock_request_coalesce — LOCK-config-specific test for the request_in_flight
 * coalesce set-bit logic in arts_handler_db_acquire (lock/acquire.c §not-held
 * path; census 11-lock §2 acquire.c + table row request_in_flight).
 *
 * The cache-side coalesce rules under test (observed via correctness, not
 * white-box state):
 *   - RW not-held: set LOCK_REQ_RW unless already set — an in-flight RO does
 *     NOT suppress the RW REQUEST (RW ⊇ RO needs the writer).  So even when an
 *     RO REQUEST is already in flight on this rank, a concurrent same-rank RW
 *     acquire must still drive a writer GRANT and the writer's store must be
 *     observable.
 *   - RO not-held: set LOCK_REQ_RO only if NEITHER bit set — an in-flight RW
 *     already covers RO locally, so a co-pending RO must coalesce and be served
 *     by the RW phase's drain (RW grant drains ro_pending too).
 *
 * Construction: many same-rank cohorts each hold a mix of RW and RO acquirers
 * on the SAME per-iteration DB, launched depc=0 so they all hit the not-held
 * path concurrently and contend on request_in_flight.  Every RW writer stamps
 * a per-iteration sentinel into the DB; the verify step requires the
 * accumulated write count to equal the writer count (no RW GRANT suppressed by
 * an in-flight RO) and every RO acquirer to have observed non-NULL data (RO
 * coalesced onto the covering phase rather than being lost).  A suppressed RW
 * REQUEST or a lost RO coalesce leaves an EDT parked → LATCH never fires →
 * ctest TIMEOUT (no in-test watchdog, no spin).
 *
 * Self-skips on any non-LOCK build at compile time (the coalesce bits are a
 * LOCK-cache concept).
 */

#if !defined(ARTS_PROTOCOL_LOCK)
#include <stdio.h>
int main(void) {
  printf("SKIP lock_request_coalesce: LOCK-only\n");
  return 0;
}
#else

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define M_ITERS 48
#define N_RW 4
#define N_RO 4
#define PER_ITER (N_RW + N_RO)
#define TOTAL (M_ITERS * PER_ITER)

/* per-iteration DB layout (uint64_t): [0] = RW write accumulator. */

/* State DB layout (uint64_t):
 *   [0]            — global RW write count (atomic)
 *   [1]            — global RO non-NULL-observation count (atomic) */
#define STATE_RWWRITES 0
#define STATE_ROSEEN 1
#define STATE_NELEMS 2

/* RW acquirer: bump the per-iteration DB accumulator + the global RW count,
 * drop latch.  paramv: [state_db, latch]. */
static void rw_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t state_db = (arts_guid_t)paramv[0];
  arts_guid_t latch = (arts_guid_t)paramv[1];
  (void)state_db;
  _Atomic uint64_t *cell = (_Atomic uint64_t *)depv[0].ptr;
  if (cell != NULL) {
    atomic_fetch_add_explicit(cell, 1u, memory_order_acq_rel);
  }
  uint64_t *state = (uint64_t *)depv[1].ptr;
  if (state != NULL) {
    atomic_fetch_add_explicit((_Atomic uint64_t *)&state[STATE_RWWRITES], 1u,
                              memory_order_acq_rel);
  }
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* RO acquirer: read the per-iteration DB (must be non-NULL → it coalesced onto
 * a covering phase), record the observation, drop latch. paramv: [latch]. */
static void ro_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  const void *cell = depv[0].ptr;
  uint64_t *state = (uint64_t *)depv[1].ptr;
  if (cell != NULL && state != NULL) {
    atomic_fetch_add_explicit((_Atomic uint64_t *)&state[STATE_ROSEEN], 1u,
                              memory_order_acq_rel);
  }
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* verify_edt — bound to all-done LATCH (slot 0) + state DB RO (slot 1). */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *state = (uint64_t *)depv[1].ptr;
  uint64_t rww = atomic_load_explicit(
      (_Atomic uint64_t *)&state[STATE_RWWRITES], memory_order_acquire);
  uint64_t ros = atomic_load_explicit((_Atomic uint64_t *)&state[STATE_ROSEEN],
                                      memory_order_acquire);
  if (rww != (uint64_t)(M_ITERS * N_RW)) {
    (void)fprintf(stderr, "FAIL: RW writes=%llu (want %d) — RW REQUEST lost\n",
                  (unsigned long long)rww, M_ITERS * N_RW);
    arts_abort(1);
  }
  if (ros != (uint64_t)(M_ITERS * N_RO)) {
    (void)fprintf(stderr,
                  "FAIL: RO non-NULL=%llu (want %d) — RO coalesce lost\n",
                  (unsigned long long)ros, M_ITERS * N_RO);
    arts_abort(1);
  }
  printf("lock_request_coalesce: %d RW + %d RO over %d iters — PASS\n",
         M_ITERS * N_RW, M_ITERS * N_RO, M_ITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf(
      "=== lock_request_coalesce (%d RW + %d RO / iter, %d iters) ===\n", N_RW,
      N_RO, M_ITERS);

  void *state_raw = NULL;
  arts_guid_t state_db =
      arts_db_create(&state_raw, STATE_NELEMS * sizeof(uint64_t), ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);
  if (state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: state_db create NULL_GUID\n");
    arts_abort(1);
  }
  memset(state_raw, 0, STATE_NELEMS * sizeof(uint64_t));
  arts_db_release(state_db, DB_MODE_RW);

  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(TOTAL);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: LATCH create NULL_GUID\n");
    arts_abort(1);
  }

  for (int it = 0; it < M_ITERS; it++) {
    void *dbp = NULL;
    arts_guid_t db = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    if (db == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: db create NULL_GUID\n", it);
      arts_abort(1);
    }
    *(uint64_t *)dbp = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* Interleave RW and RO issuance so that, on the same rank, an RO REQUEST is
     * frequently in flight when an RW acquire arrives (RW must still request)
     * and vice-versa (RO must coalesce).  depc=0 → all hit the not-held path
     * concurrently. */
    uint64_t rw_pv[2] = {(uint64_t)state_db, (uint64_t)latch};
    uint64_t ro_pv[1] = {(uint64_t)latch};
    for (int k = 0; k < PER_ITER; k++) {
      if (k % 2 == 0 && (k / 2) < N_RW) {
        arts_guid_t e = arts_edt_create(rw_edt, 2, rw_pv, 2, NULL);
        arts_add_dependence(db, e, 0, DB_MODE_RW);
        arts_add_dependence(state_db, e, 1, DB_MODE_RW);
      } else {
        arts_guid_t e = arts_edt_create(ro_edt, 1, ro_pv, 2, NULL);
        arts_add_dependence(db, e, 0, DB_MODE_RO);
        arts_add_dependence(state_db, e, 1, DB_MODE_RW);
      }
    }
  }

  arts_guid_t v =
      arts_edt_create(verify_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);
  arts_add_dependence(state_db, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_LOCK */
