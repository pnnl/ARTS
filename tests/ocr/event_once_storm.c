/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_once_storm — Phase: event-redesign, Task §6.1.
 *
 * Stress the ONCE event satisfy↔addDep race window (spec §4.1 R1-R7).
 *
 * For each of M=64 iterations:
 *   - Allocate one fresh ONCE/IDEM event + a unique data DB.
 *   - Spawn N_CONSUMERS=8 consumer EDTs and N_SATISFIERS=8 satisfier
 *     EDTs.  Each runs on a worker thread (the runtime's worker pool
 *     provides the natural concurrency without pthread bookkeeping that
 *     would bypass arts_thread_info / current_edt setup).  Consumer EDT
 *     bodies create a counter_edt and call arts_add_dependence on the
 *     event.  Satisfier EDT bodies call arts_event_satisfy with the
 *     iteration's data DB.
 *   - Each counter_edt records the data GUID it received and drops one
 *     latch.  A single LATCH event sized M_ITERS*N_CONSUMERS fans the
 *     whole storm into one verify_edt that checks, per iteration, (a) the
 *     count matches, (b) every consumer observed the same data GUID, (c)
 *     no double-fire.
 *
 * No EDT busy-waits: an EDT may only wait via events/dependencies, never
 * by spinning on an atomic (a spinning EDT occupies a worker and, while it
 * also creator-holds the iteration's RW DB, blocks the RW consumers from
 * ever acquiring under strict single-writer coherence).  main_edt sets up
 * all iterations and TERMINATES, releasing every creator-hold so the RW
 * counter EDTs can proceed; the verify_edt bound to the latch performs all
 * checks once the storm has drained.
 *
 * PASS criterion: 64 × 8 = 512 deliveries, all carrying the same data
 * GUID per iteration.  Stresses spec §4.1 R1 (S→A immediate deliver),
 * R3 (A push before S CAS), R4 (A push after S CAS), R6 (concurrent
 * late binders), R7 (concurrent pre-fire pushes).
 * Any mismatch calls arts_abort(1) for a non-zero exit code.
 * A stranded waiter is caught by the ctest TIMEOUT (no in-test watchdog).
 *
 * State DB layout (uint64_t elements):
 *   [0]                              — signaled_count (atomic uint32 in low 32
 * bits) [1 .. M_ITERS*N_CONSUMERS]      — consumer_data[it*N_CONSUMERS + idx]
 *   [M_ITERS*N_CONSUMERS+1 ..
 *    M_ITERS*N_CONSUMERS+M_ITERS]   — iter_db_expected[it]
 *
 * All writes from counter EDTs are atomic (RW is per-node exclusive, but
 * concurrent same-node EDTs share the node's RW window — slots are distinct
 * words so no aliased read-modify-write occurs; signaled_count uses
 * atomic_fetch_add to accumulate across concurrent increments safely).
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define M_ITERS 64
#define N_CONSUMERS 8
#define N_SATISFIERS 8

/* Offsets into the state DB (all uint64_t elements). */
#define STATE_SIGNALED_OFF 0
#define STATE_CDATA_OFF 1
#define STATE_ITERDB_OFF (STATE_CDATA_OFF + (M_ITERS) * (N_CONSUMERS))
#define STATE_NELEMS (STATE_ITERDB_OFF + (M_ITERS))

/* Counter EDT — invoked when its event-source dep fires.
 * paramv: [state_db_guid, latch_guid, it, idx, iter_db_expected_guid]. */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  arts_guid_t state_db = (arts_guid_t)paramv[0];
  arts_guid_t latch = (arts_guid_t)paramv[1];
  uint64_t it = paramv[2];
  uint64_t idx = paramv[3];
  uint64_t expected = paramv[4]; /* iter_db guid for this iteration */
  (void)state_db;                /* dep slot index given below */

  /* depv[0] = the event-data dep (DB_MODE_RW from event satisfy).
   * depv[1] = state DB (DB_MODE_RW). */
  uint64_t *state = (uint64_t *)depv[1].ptr;

  if (it < M_ITERS && idx < N_CONSUMERS) {
    /* Store the received data GUID into this consumer's slot. */
    _Atomic uint64_t *cdata_slot =
        (_Atomic uint64_t *)&state[STATE_CDATA_OFF + it * N_CONSUMERS + idx];
    atomic_store_explicit(cdata_slot, (uint64_t)depv[0].guid,
                          memory_order_release);
    /* Record the expected iter_db guid for this iteration (idempotent:
     * all consumers of the same iteration write the same value). */
    _Atomic uint64_t *iterdb_slot =
        (_Atomic uint64_t *)&state[STATE_ITERDB_OFF + it];
    atomic_store_explicit(iterdb_slot, expected, memory_order_release);
  }
  _Atomic uint64_t *count = (_Atomic uint64_t *)&state[STATE_SIGNALED_OFF];
  atomic_fetch_add_explicit(count, 1u, memory_order_acq_rel);

  /* Drop one latch.  When the last of the M_ITERS*N_CONSUMERS counter EDTs
   * drops it, the LATCH fires and verify_edt runs. */
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* Consumer worker EDT: create counter_edt + addDep on event.
 * paramv: [event_guid, state_db_guid, latch_guid, it, idx,
 *          iter_db_expected_guid] */
static void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_guid_t state_db = (arts_guid_t)paramv[1];
  /* counter_edt paramv: [state_db_guid, latch_guid, it, idx,
   *                      iter_db_expected_guid] */
  uint64_t counter_pv[5] = {paramv[1], paramv[2], paramv[3], paramv[4],
                            paramv[5]};
  /* counter_edt has 2 deps: [0]=event-data, [1]=state DB */
  arts_guid_t edt = arts_edt_create(counter_edt, 5, counter_pv, 2, NULL);
  arts_add_dependence(event, edt, 0, DB_MODE_RW);
  arts_add_dependence(state_db, edt, 1, DB_MODE_RW);
}

/* Satisfier worker EDT: call arts_event_satisfy.
 * paramv: [event_guid, data_db_guid] */
static void satisfier_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_guid_t data = (arts_guid_t)paramv[1];
  arts_event_satisfy(event, data);
}

/* verify_edt — bound to the storm-wide LATCH (slot 0, DB_MODE_NULL).
 * Slot 1 is the state DB (DB_MODE_RW).
 * Runs strictly after the last counter_edt dropped the latch, so every
 * store into the state DB is visible (LATCH fire is the happens-before).
 * Performs the original per-iteration assertions, then shuts the runtime
 * down. */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;

  /* depv[0] = latch event dep (DB_MODE_NULL, ptr==NULL by spec).
   * depv[1] = state DB (DB_MODE_RW). */
  uint64_t *state = (uint64_t *)depv[1].ptr;

  _Atomic uint64_t *count_p = (_Atomic uint64_t *)&state[STATE_SIGNALED_OFF];
  uint64_t got = atomic_load_explicit(count_p, memory_order_acquire);
  if (got != (unsigned int)(M_ITERS * N_CONSUMERS)) {
    (void)fprintf(stderr, "FAIL: signaled_count=%llu (want %u)\n",
                  (unsigned long long)got,
                  (unsigned int)(M_ITERS * N_CONSUMERS));
    arts_abort(1);
  }

  for (int it = 0; it < M_ITERS; it++) {
    /* All consumers must observe the same data GUID — the unique winner
     * of the satisfy race wrote simple.data, every late binder reads from
     * that slot via the fired==true short-circuit. */
    _Atomic uint64_t *slot0 =
        (_Atomic uint64_t *)&state[STATE_CDATA_OFF + it * N_CONSUMERS + 0];
    arts_guid_t first =
        (arts_guid_t)atomic_load_explicit(slot0, memory_order_acquire);
    for (int i = 1; i < N_CONSUMERS; i++) {
      _Atomic uint64_t *sloti =
          (_Atomic uint64_t *)&state[STATE_CDATA_OFF + it * N_CONSUMERS + i];
      arts_guid_t cd =
          (arts_guid_t)atomic_load_explicit(sloti, memory_order_acquire);
      if (cd != first) {
        (void)fprintf(
            stderr,
            "FAIL [iter=%d]: consumer %d got data=%lu, consumer 0 got %lu\n",
            it, i, (uint64_t)cd, (uint64_t)first);
        arts_abort(1);
      }
    }
    _Atomic uint64_t *iter_expected_p =
        (_Atomic uint64_t *)&state[STATE_ITERDB_OFF + it];
    arts_guid_t expected = (arts_guid_t)atomic_load_explicit(
        iter_expected_p, memory_order_acquire);
    if (first != expected) {
      (void)fprintf(stderr,
                    "FAIL [iter=%d]: consumer data=%lu != satisfier db=%lu\n",
                    it, (uint64_t)first, (uint64_t)expected);
      arts_abort(1);
    }
  }

  printf("event_once_storm: %d iters x %d consumers = %d deliveries — PASS\n",
         M_ITERS, N_CONSUMERS, M_ITERS * N_CONSUMERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* Allocate the state DB used by all counter EDTs and verify_edt. */
  void *state_raw = NULL;
  arts_guid_t state_db =
      arts_db_create(&state_raw, STATE_NELEMS * sizeof(uint64_t), ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);
  if (state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: arts_db_create state_db returned NULL_GUID\n");
    arts_abort(1);
  }
  memset(state_raw, 0, STATE_NELEMS * sizeof(uint64_t));
  arts_db_release(state_db, DB_MODE_RW);

  /* One LATCH event for the whole storm: each counter_edt drops it once,
   * so the M_ITERS*N_CONSUMERS-th drop fires the event and unblocks
   * verify_edt's slot 0.  fire-and-linger means a late add_dependence
   * after the final drop still delivers, but here verify_edt is wired
   * before any consumer runs (main_edt is the creator-hold owner of every
   * iteration DB and must terminate first), so no race is needed. */
  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(M_ITERS * N_CONSUMERS);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: arts_event_create LATCH returned NULL_GUID\n");
    arts_abort(1);
  }

  for (int it = 0; it < M_ITERS; it++) {
    /* Use IDEMPOTENT instead of ONCE.  Under the new latch+life_count
     * invariant, ONCE auto-destroys on fire, so a late add_dependence
     * after the destroy is user-error per OCR §1.4.3.  The single-fire
     * satisfy↔addDep race rescue path (spec §4.1 R1-R7) is identical for
     * IDEM, but the event persists so the test's "all consumers must
     * observe the same data" assertion is well-defined regardless of race
     * ordering. */
    arts_event_hint_t h = ARTS_EVENT_HINT_IDEMPOTENT;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(
          stderr, "FAIL [iter=%d]: arts_event_create returned NULL_GUID\n", it);
      arts_abort(1);
    }

    void *dbp = NULL;
    arts_guid_t db = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    if (db == NULL_GUID) {
      (void)fprintf(stderr,
                    "FAIL [iter=%d]: arts_db_create returned NULL_GUID\n", it);
      arts_abort(1);
    }

    /* Spawn N consumer EDTs + N satisfier EDTs.  Each is depc=0 (ready
     * immediately), so the worker pool dispatches them in parallel and the
     * satisfy↔addDep race window opens. */
    uint64_t satisfier_pv[2] = {(uint64_t)ev, (uint64_t)db};
    for (int i = 0; i < N_CONSUMERS; i++) {
      /* paramv: [event_guid, state_db_guid, latch_guid, it, idx,
       *          iter_db_expected_guid] */
      uint64_t consumer_pv[6] = {(uint64_t)ev,    (uint64_t)state_db,
                                 (uint64_t)latch, (uint64_t)it,
                                 (uint64_t)i,     (uint64_t)db};
      arts_edt_create(consumer_edt, 6, consumer_pv, 0, NULL);
    }
    for (int i = 0; i < N_SATISFIERS; i++) {
      arts_edt_create(satisfier_edt, 2, satisfier_pv, 0, NULL);
    }
  }

  /* verify_edt fires after the whole storm drains the latch.
   * dep[0] = latch (DB_MODE_NULL), dep[1] = state DB (DB_MODE_RW). */
  arts_guid_t v = arts_edt_create(verify_edt, 0, NULL, 2, NULL);
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);
  arts_add_dependence(state_db, v, 1, DB_MODE_RO);

  /* main_edt terminates here, releasing the creator-hold RW on every
   * iteration DB so the RW counter EDTs can acquire. */
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
