/* SPDX-License-Identifier: Apache-2.0
 *
 * lock_samerank_rw_grant_race — LOCK-config-specific reproducer for the
 * confirmed intra-node concurrent-RW grant-loss class (census 11-lock §4
 * bugs A/B; memory: lock-cache-race-event-storm).
 *
 * SCENARIO
 * --------
 * N consumer EDTs concurrently RW-acquire the SAME data DB on the SAME rank.
 * Under the LOCK protocol the home grants one rank-granular GRANT and the
 * grant handler's drain wakes the whole same-rank RW cohort.  The interplay
 * among `local_count` guard up/down, `held_mode` publish, `request_in_flight`
 * clear and the Treiber-stack drain snapshot has a window in which a
 * late-arriving acquire pushes a waiter that is neither drained by the current
 * GRANT nor re-requested for the next round — exactly one wakeup is lost.  The
 * symptom is: the last consumer's EDT never becomes ready, the storm-wide
 * LATCH never fires, verify_edt never runs, and the runtime hangs (caught by
 * the ctest TIMEOUT — there is NO in-test watchdog and NO spin).
 *
 * The test is CORRECT-AND-FAILING by design under LOCK while the bug is
 * present: the pass token is only emitted when ALL N*ITERS deliveries land and
 * the per-DB visited-count equals N exactly.  It must NOT be weakened to pass.
 *
 * Each consumer's RW acquire of the per-iteration DB increments depv[0]'s
 * counter word (RW is exclusive on this rank, so the increments are serialized
 * by the lock phase — but we still use atomics defensively because the same
 * rank's RW cohort is drained to run concurrently on the worker pool).  The
 * final counter for each iteration DB must equal N_CONSUMERS; if a grant is
 * lost the cohort never fully runs and the LATCH stays armed.
 *
 * Self-skips on any non-LOCK build at compile time.
 */

#if !defined(ARTS_PROTOCOL_LOCK)
#include <stdio.h>
int main(void) {
  printf("SKIP lock_samerank_rw_grant_race: LOCK-only\n");
  return 0;
}
#else

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define M_ITERS 64
#define N_CONSUMERS 8
#define TOTAL_DELIVERIES (M_ITERS * N_CONSUMERS)

/* State DB layout (uint64_t elements):
 *   [0]                          — global delivery count (atomic)
 *   [1 .. M_ITERS]               — per-iteration visited count (atomic) */
#define STATE_COUNT_OFF 0
#define STATE_PERITER_OFF 1
#define STATE_NELEMS (STATE_PERITER_OFF + M_ITERS)

/* Consumer EDT — RW-acquires the per-iteration DB (slot 0) and the shared
 * state DB (slot 1).  paramv: [state_db, latch, it]. */
static void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[1];
  uint64_t it = paramv[2];

  /* depv[0] = per-iteration DB (RW): bump its own embedded counter so a lost
   * grant is observable both as a stuck LATCH and as a short per-iter count. */
  _Atomic uint64_t *cell = (_Atomic uint64_t *)depv[0].ptr;
  if (cell != NULL) {
    atomic_fetch_add_explicit(cell, 1u, memory_order_acq_rel);
  }

  /* depv[1] = state DB (RW): record per-iteration + global delivery counts. */
  uint64_t *state = (uint64_t *)depv[1].ptr;
  if (state != NULL && it < M_ITERS) {
    _Atomic uint64_t *peri = (_Atomic uint64_t *)&state[STATE_PERITER_OFF + it];
    atomic_fetch_add_explicit(peri, 1u, memory_order_acq_rel);
    _Atomic uint64_t *gc = (_Atomic uint64_t *)&state[STATE_COUNT_OFF];
    atomic_fetch_add_explicit(gc, 1u, memory_order_acq_rel);
  }

  /* Drop one latch.  The TOTAL_DELIVERIES-th drop fires verify_edt.  If even
   * one consumer's grant is lost, this drop never happens and the LATCH stays
   * armed → ctest TIMEOUT. */
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* verify_edt — bound to the storm-wide LATCH (slot 0, DB_MODE_NULL) + state
 * DB (slot 1, RO).  Runs strictly after the last consumer dropped the latch. */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *state = (uint64_t *)depv[1].ptr;

  _Atomic uint64_t *gc = (_Atomic uint64_t *)&state[STATE_COUNT_OFF];
  uint64_t got = atomic_load_explicit(gc, memory_order_acquire);
  if (got != (uint64_t)TOTAL_DELIVERIES) {
    (void)fprintf(stderr, "FAIL: delivery count=%llu (want %d)\n",
                  (unsigned long long)got, TOTAL_DELIVERIES);
    arts_abort(1);
  }
  for (int it = 0; it < M_ITERS; it++) {
    _Atomic uint64_t *peri = (_Atomic uint64_t *)&state[STATE_PERITER_OFF + it];
    uint64_t v = atomic_load_explicit(peri, memory_order_acquire);
    if (v != (uint64_t)N_CONSUMERS) {
      (void)fprintf(stderr, "FAIL [iter=%d]: visited=%llu (want %d)\n", it,
                    (unsigned long long)v, N_CONSUMERS);
      arts_abort(1);
    }
  }
  printf("lock_samerank_rw_grant_race: %d iters x %d RW = %d grants — PASS\n",
         M_ITERS, N_CONSUMERS, TOTAL_DELIVERIES);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== lock_samerank_rw_grant_race (%d iters x %d RW) ===\n",
              M_ITERS, N_CONSUMERS);

  void *state_raw = NULL;
  arts_guid_t state_db =
      arts_db_create(&state_raw, STATE_NELEMS * sizeof(uint64_t), ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);
  if (state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: state_db create returned NULL_GUID\n");
    arts_abort(1);
  }
  memset(state_raw, 0, STATE_NELEMS * sizeof(uint64_t));
  arts_db_release(state_db, DB_MODE_RW);

  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(TOTAL_DELIVERIES);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: LATCH create returned NULL_GUID\n");
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
    /* Drop the creator's RW hold so the consumer cohort can be granted. */
    arts_db_release(db, DB_MODE_RW);

    /* N depc=0 consumers (ready immediately) all RW-acquire the SAME db on the
     * SAME rank → they coalesce onto one REQUEST and must all be drained by the
     * one GRANT.  This is the grant-loss window. */
    for (int i = 0; i < N_CONSUMERS; i++) {
      uint64_t pv[3] = {(uint64_t)state_db, (uint64_t)latch, (uint64_t)it};
      arts_guid_t c = arts_edt_create(consumer_edt, 3, pv, 2, NULL);
      arts_add_dependence(db, c, 0, DB_MODE_RW);
      arts_add_dependence(state_db, c, 1, DB_MODE_RW);
    }
  }

  /* verify_edt fires when the whole storm drains the latch. */
  arts_guid_t v = arts_edt_create(verify_edt, 0, NULL, 2, NULL);
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);
  arts_add_dependence(state_db, v, 1, DB_MODE_RO);

  /* main_edt terminates — no creator-hold remains on any iteration DB. */
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_LOCK */
