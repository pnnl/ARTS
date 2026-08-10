/* SPDX-License-Identifier: Apache-2.0
 *
 * scheduler_async_nonowner_push — runtime_single stress for RW
 * ownership-transfer chain scheduling (census 17 §1 arts_schedule_ready_edt).
 *
 * Background (scheduler.c): a fully DB-acquired EDT is pushed onto a
 * work-stealing deque.  Each runtime thread (worker, sender, receiver) owns its
 * own deque, so a coherence-wake / OoO-drain that resumes an RW acquire walk
 * pushes onto the resuming thread's own deque — a single-producer (owner-only)
 * Chase-Lev push, the contract arts_deque_push_front requires.  (The
 * my_deque == NULL fallback to deque[0] is for non-runtime threads only — e.g.
 * GPU callbacks, which take the new_edts path — and is not exercised here.)
 *
 * What this exercises: the RW ownership-transfer chain drives a long sequence
 * of asynchronous acquire resumes + deque pushes + work-stealing pops, plus the
 * runnable-phase EDT lifetime path, all under heavy concurrent worker activity.
 *
 * Construction: a long single-rank chain of RW acquirers on ONE shared DB; each
 * link, when it runs, releases the DB (transferring ownership to the next
 * acquirer) and explicitly wakes its successor via a NO_SLOT control dep.  A
 * second wide fan of independent RW acquirers on per-element DBs runs
 * concurrently to keep all workers busy stealing/popping while the chain
 * resumes pushes.
 *
 * Correctness (black-box): every chain link and every fan acquirer must run
 * EXACTLY ONCE.  Each link/fan EDT decrements the all-done LATCH (sized to
 * TOTAL_EDTS); a lost or stalled EDT leaves the LATCH unfired → ctest TIMEOUT,
 * a duplicated EDT is caught by the per-EDT run-once bitset and the global
 * run-count over-count.  Completion is a LATCH wait; hang is reaped by the
 * ctest TIMEOUT.
 *
 * Runs in every coherence configuration (single-rank, ARTS_DB).
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

/* Length of the serialized RW ownership-transfer chain. */
#define CHAIN_LEN 64
/* Width of the concurrent independent-RW fan (each on its own DB). */
#define FAN_WIDTH 64
#define TOTAL_EDTS (CHAIN_LEN + FAN_WIDTH)

/* State DB layout (uint64_t):
 *   [0]               — global run count (atomic, must end == TOTAL_EDTS)
 *   [1]               — global double-run count (atomic, must end == 0)
 *   [2 .. 2+TOTAL-1]  — per-EDT run bitset cells (each set exactly once) */
#define ST_RUNS 0
#define ST_DOUBLE 1
#define ST_BITS 2
#define ST_NELEMS (ST_BITS + TOTAL_EDTS)

/* Mark EDT index `idx` as having run; bump double-run if it ran before. */
static void mark_run(uint64_t *state, unsigned idx) {
  uint64_t prev = atomic_fetch_add_explicit(
      (_Atomic uint64_t *)&state[ST_BITS + idx], 1u, memory_order_acq_rel);
  if (prev != 0u) {
    atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_DOUBLE], 1u,
                              memory_order_acq_rel);
  }
  atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_RUNS], 1u,
                            memory_order_acq_rel);
}

/* Chain link: acquires the shared chain DB RW (ownership transferred from the
 * previous link), stamps its index, records the run, drops the all-done LATCH,
 * then signals the next link.
 *   paramv: [state_db, chain_db, my_index, next_guid_or_0, latch]
 *   depv[0] = chain_db (RW), depv[1] = state_db (RW); third dep is the NO_SLOT
 *   control wake from the predecessor (depc == 3, no depv entry). */
static void chain_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned idx = (unsigned)paramv[2];
  arts_guid_t next = (arts_guid_t)paramv[3];
  arts_guid_t latch = (arts_guid_t)paramv[4];
  uint64_t *chain = (uint64_t *)depv[0].ptr;
  uint64_t *state = (uint64_t *)depv[1].ptr;
  if (chain != NULL) {
    chain[0] = (uint64_t)idx; /* serialized RW write — establishes the order */
  }
  mark_run(state, idx);
  /* Every link decrements the all-done LATCH (sized to TOTAL_EDTS): the chain
   * contributes one DECR per link, the fan one per acquirer, so the count is
   * balanced only if EVERY EDT — not just the last link — signals here. */
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  /* Hand the chain forward: wake the successor with a NO_SLOT control
   * dependence (counted in the successor's depc).  The successor's chain_db RW
   * acquire is in any case resolved by the ownership transfer this link's
   * release drives; the explicit wake makes the serialization order the test
   * asserts independent of acquire-registration order. */
  if (next != NULL_GUID) {
    arts_add_dependence((arts_guid_t)0, next, (uint32_t)-1, DB_MODE_VAL);
  }
}

/* Independent fan acquirer: RW on its own per-element DB (no contention with
 * the chain) — keeps workers churning their deques (push/steal/pop) while the
 * chain resumes pushes.  paramv: [state_db, my_index, latch]
 *   depv[0] = element DB (RW), depv[1] = state_db (RW) */
static void fan_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned idx = (unsigned)paramv[1];
  arts_guid_t latch = (arts_guid_t)paramv[2];
  uint64_t *cell = (uint64_t *)depv[0].ptr;
  uint64_t *state = (uint64_t *)depv[1].ptr;
  if (cell != NULL) {
    cell[0] = (uint64_t)idx;
  }
  mark_run(state, idx);
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* verify_edt — bound to the all-done LATCH (slot 0) + state DB RO (slot 1). */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *state = (uint64_t *)depv[1].ptr;
  uint64_t runs = atomic_load_explicit((_Atomic uint64_t *)&state[ST_RUNS],
                                       memory_order_acquire);
  uint64_t dbl = atomic_load_explicit((_Atomic uint64_t *)&state[ST_DOUBLE],
                                      memory_order_acquire);
  if (dbl != 0u) {
    (void)fprintf(stderr,
                  "FAIL: %llu EDT(s) ran more than once — duplicate dispatch\n",
                  (unsigned long long)dbl);
    arts_abort(1);
  }
  if (runs != (uint64_t)TOTAL_EDTS) {
    (void)fprintf(stderr, "FAIL: %llu EDTs ran (want %d) — lost EDT\n",
                  (unsigned long long)runs, TOTAL_EDTS);
    arts_abort(1);
  }
  /* Every per-EDT bitset cell set exactly once. */
  for (unsigned i = 0; i < TOTAL_EDTS; i++) {
    uint64_t c = atomic_load_explicit((_Atomic uint64_t *)&state[ST_BITS + i],
                                      memory_order_acquire);
    if (c != 1u) {
      (void)fprintf(stderr, "FAIL: EDT %u ran %llu times (want 1)\n", i,
                    (unsigned long long)c);
      arts_abort(1);
    }
  }
  printf("scheduler_async_nonowner_push: %d chain + %d fan EDTs each ran once "
         "— PASS\n",
         CHAIN_LEN, FAN_WIDTH);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== scheduler_async_nonowner_push (chain=%d, fan=%d) ===\n",
              CHAIN_LEN, FAN_WIDTH);

  void *state_raw = NULL;
  arts_guid_t state_db =
      arts_db_create(&state_raw, ST_NELEMS * sizeof(uint64_t), ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);
  if (state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: state_db create NULL_GUID\n");
    arts_abort(1);
  }
  memset(state_raw, 0, ST_NELEMS * sizeof(uint64_t));
  arts_db_release(state_db, DB_MODE_RW);

  /* Shared DB the whole chain acquires RW in turn (ownership transfer). */
  void *chain_raw = NULL;
  arts_guid_t chain_db = arts_db_create(&chain_raw, sizeof(uint64_t), ARTS_DB,
                                        ARTS_DB_PROP_NONE, NULL);
  if (chain_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: chain_db create NULL_GUID\n");
    arts_abort(1);
  }
  *(uint64_t *)chain_raw = 0u;
  arts_db_release(chain_db, DB_MODE_RW);

  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(TOTAL_EDTS);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: LATCH create NULL_GUID\n");
    arts_abort(1);
  }

  /* Pre-reserve the chain link GUIDs so each link can name its successor. */
  arts_guid_t link[CHAIN_LEN];
  for (int i = 0; i < CHAIN_LEN; i++) {
    link[i] = arts_guid_reserve(ARTS_GUID_EDT, 0);
  }
  for (int i = 0; i < CHAIN_LEN; i++) {
    arts_guid_t next = (i + 1 < CHAIN_LEN) ? link[i + 1] : NULL_GUID;
    uint64_t pv[5] = {(uint64_t)state_db, (uint64_t)chain_db, (uint64_t)i,
                      (uint64_t)next, (uint64_t)latch};
    /* depc = 3: chain_db RW (slot 0) + state_db RW (slot 1) + the predecessor's
     * NO_SLOT control wake.  The wake is a real readiness decrement, so it must
     * be counted in depc; a depc sized to the two data slots only would let a
     * third decrement underflow depc_needed and re-fire an already-run EDT. */
    arts_edt_create(chain_edt, 5, pv, 3, &(arts_edt_hint_t){.guid = link[i]});
    arts_add_dependence(chain_db, link[i], 0, DB_MODE_RW);
    arts_add_dependence(state_db, link[i], 1, DB_MODE_RW);
  }
  /* Kick the head of the chain. */
  arts_add_dependence((arts_guid_t)0, link[0], (uint32_t)-1, DB_MODE_VAL);

  /* Independent fan: each acquirer indexed CHAIN_LEN..TOTAL-1. */
  for (int j = 0; j < FAN_WIDTH; j++) {
    void *ep = NULL;
    arts_guid_t el =
        arts_db_create(&ep, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
    if (el == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [fan=%d]: element db NULL_GUID\n", j);
      arts_abort(1);
    }
    *(uint64_t *)ep = 0u;
    arts_db_release(el, DB_MODE_RW);
    unsigned idx = (unsigned)(CHAIN_LEN + j);
    uint64_t pv[3] = {(uint64_t)state_db, (uint64_t)idx, (uint64_t)latch};
    arts_guid_t e = arts_edt_create(fan_edt, 3, pv, 2, NULL);
    arts_add_dependence(el, e, 0, DB_MODE_RW);
    arts_add_dependence(state_db, e, 1, DB_MODE_RW);
  }

  arts_guid_t v =
      arts_edt_create(verify_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);
  arts_add_dependence(state_db, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
