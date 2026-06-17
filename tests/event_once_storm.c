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
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"

#define M_ITERS 64
#define N_CONSUMERS 8
#define N_SATISFIERS 8

static atomic_uint signaled_count = 0;
/* Per-iteration slots — index [it][consumer].  A counter_edt stores the
 * data GUID it received; verify_edt compares all consumers of one
 * iteration after the storm has fully drained (the LATCH fire is the
 * happens-before that makes every store visible to verify_edt). */
static atomic_ulong consumer_data[M_ITERS][N_CONSUMERS];
/* iter_db[it] = the data DB satisfied into iteration `it`'s event. */
static arts_guid_t iter_db[M_ITERS];
static atomic_int g_clean_shutdown = 0;

/* Counter EDT — invoked when its event-source dep fires.
 * paramv: [latch_guid, it, idx]. */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  uint64_t it = paramv[1];
  uint64_t idx = paramv[2];
  if (it < M_ITERS && idx < N_CONSUMERS) {
    atomic_store_explicit(&consumer_data[it][idx], (uint64_t)depv[0].guid,
                          memory_order_release);
  }
  atomic_fetch_add_explicit(&signaled_count, 1u, memory_order_acq_rel);
  /* Drop one latch.  When the last of the M_ITERS*N_CONSUMERS counter EDTs
   * drops it, the LATCH fires and verify_edt runs. */
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* Consumer worker EDT: create counter_edt + addDep on event.
 * paramv: [event_guid, latch_guid, it, idx] */
static void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  uint64_t counter_pv[3] = {paramv[1], paramv[2], paramv[3]};
  arts_guid_t edt = arts_edt_create(counter_edt, 3, counter_pv, 1, NULL);
  arts_add_dependence(event, edt, 0, DB_MODE_RW);
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
 * Runs strictly after the last counter_edt dropped the latch, so every
 * consumer_data store is visible.  Performs the original per-iteration
 * assertions, then shuts the runtime down. */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int got =
      atomic_load_explicit(&signaled_count, memory_order_acquire);
  if (got != (unsigned int)(M_ITERS * N_CONSUMERS)) {
    (void)fprintf(stderr, "FAIL: signaled_count=%u (want %u)\n", got,
                  (unsigned int)(M_ITERS * N_CONSUMERS));
    arts_abort(1);
  }

  for (int it = 0; it < M_ITERS; it++) {
    /* All consumers must observe the same data GUID — the unique winner
     * of the satisfy race wrote simple.data, every late binder reads from
     * that slot via the fired==true short-circuit. */
    arts_guid_t first = (arts_guid_t)atomic_load_explicit(&consumer_data[it][0],
                                                          memory_order_acquire);
    for (int i = 1; i < N_CONSUMERS; i++) {
      arts_guid_t cd = (arts_guid_t)atomic_load_explicit(&consumer_data[it][i],
                                                         memory_order_acquire);
      if (cd != first) {
        (void)fprintf(
            stderr,
            "FAIL [iter=%d]: consumer %d got data=%lu, consumer 0 got %lu\n",
            it, i, (uint64_t)cd, (uint64_t)first);
        arts_abort(1);
      }
    }
    if (first != iter_db[it]) {
      (void)fprintf(stderr,
                    "FAIL [iter=%d]: consumer data=%lu != satisfier db=%lu\n",
                    it, (uint64_t)first, (uint64_t)iter_db[it]);
      arts_abort(1);
    }
  }

  atomic_store(&g_clean_shutdown, 1);
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
    iter_db[it] = db;

    /* Spawn N consumer EDTs + N satisfier EDTs.  Each is depc=0 (ready
     * immediately), so the worker pool dispatches them in parallel and the
     * satisfy↔addDep race window opens. */
    uint64_t satisfier_pv[2] = {(uint64_t)ev, (uint64_t)db};
    for (int i = 0; i < N_CONSUMERS; i++) {
      uint64_t consumer_pv[4] = {(uint64_t)ev, (uint64_t)latch, (uint64_t)it,
                                 (uint64_t)i};
      arts_edt_create(consumer_edt, 4, consumer_pv, 0, NULL);
    }
    for (int i = 0; i < N_SATISFIERS; i++) {
      arts_edt_create(satisfier_edt, 2, satisfier_pv, 0, NULL);
    }
  }

  /* verify_edt fires after the whole storm drains the latch. */
  arts_guid_t v = arts_edt_create(verify_edt, 0, NULL, 1, NULL);
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);

  /* main_edt terminates here, releasing the creator-hold RW on every
   * iteration DB so the RW counter EDTs can acquire. */
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
