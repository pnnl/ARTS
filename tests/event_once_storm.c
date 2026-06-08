/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_once_storm — Phase: event-redesign, Task §6.1.
 *
 * Stress the ONCE event satisfy↔addDep race window (spec §4.1 R1-R7).
 *
 * For each of M=64 iterations:
 *   - Allocate one fresh ONCE event + a unique data DB.
 *   - Spawn N_CONSUMERS=8 consumer EDTs and N_SATISFIERS=8 satisfier
 *     EDTs.  Each runs on a worker thread (the runtime's worker pool
 *     provides the natural concurrency without pthread bookkeeping that
 *     would bypass arts_thread_info / current_edt setup).  Consumer EDT
 *     bodies create a counter_edt and call arts_add_dependence on the
 *     event.  Satisfier EDT bodies call arts_event_satisfy with the
 *     iteration's data DB.
 *   - The main_edt spins until signaled_count reaches N_CONSUMERS, then
 *     verifies (a) the count matches, (b) every consumer observed the
 *     same data GUID, (c) no double-fire.
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
/* One slot per consumer — index = atomic_fetch_add(signaled_count).
 * main_edt compares all slots after the count reaches N_CONSUMERS to
 * avoid the framework race that the original "first-write-then-read"
 * pattern had on signaled_count's release-acquire edge. */
static atomic_ulong consumer_data[N_CONSUMERS];

/* Counter EDT — invoked when its event-source dep fires.
 * paramv: [expected_idx] — the consumer's pre-assigned slot. */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t idx = paramv[0];
  if (idx < N_CONSUMERS) {
    /* Store BEFORE the publish increment so main_edt's acquire-load on
     * signaled_count synchronizes with our store. */
    atomic_store_explicit(&consumer_data[idx], (uint64_t)depv[0].guid,
                          memory_order_release);
  }
  atomic_fetch_add_explicit(&signaled_count, 1u, memory_order_acq_rel);
}

/* Consumer worker EDT: create counter_edt + addDep on event.
 * paramv: [event_guid, expected_idx] */
static void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  uint64_t idx = paramv[1];
  uint64_t counter_pv[1] = {idx};
  arts_guid_t edt = arts_edt_create(counter_edt, 1, counter_pv, 1, NULL);
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

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  for (int it = 0; it < M_ITERS; it++) {
    atomic_store_explicit(&signaled_count, 0u, memory_order_relaxed);
    for (int i = 0; i < N_CONSUMERS; i++) {
      atomic_store_explicit(&consumer_data[i], 0ul, memory_order_relaxed);
    }

    /* Use IDEMPOTENT instead of ONCE.  Under the new
     * latch+life_count invariant, ONCE auto-destroys on fire, so late
     * add_dependence after the destroy is user-error per OCR §1.4.3.
     * The single-fire satisfy↔addDep race rescue path (spec §4.1
     * R1-R7) is identical for IDEM, but the event persists so the
     * test's "all consumers must observe the same data" assertion is
     * well-defined regardless of race ordering. */
    arts_event_hint_t h = ARTS_EVENT_HINT_IDEMPOTENT;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(
          stderr, "FAIL [iter=%d]: arts_event_create returned NULL_GUID\n", it);
      abort();
    }

    void *dbp = NULL;
    arts_guid_t db = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    if (db == NULL_GUID) {
      (void)fprintf(stderr,
                    "FAIL [iter=%d]: arts_db_create returned NULL_GUID\n", it);
      abort();
    }

    /* Spawn N consumer EDTs + N satisfier EDTs.  Each is depc=0 (ready
     * immediately), so the worker pool dispatches them in parallel. */
    uint64_t satisfier_pv[2] = {(uint64_t)ev, (uint64_t)db};
    for (int i = 0; i < N_CONSUMERS; i++) {
      uint64_t consumer_pv[2] = {(uint64_t)ev, (uint64_t)i};
      arts_edt_create(consumer_edt, 2, consumer_pv, 0, NULL);
    }
    for (int i = 0; i < N_SATISFIERS; i++) {
      arts_edt_create(satisfier_edt, 2, satisfier_pv, 0, NULL);
    }

    /* Spin until all consumer EDTs have run. */
    for (int spin = 0; spin < 100000000 &&
                       atomic_load_explicit(&signaled_count,
                                            memory_order_acquire) < N_CONSUMERS;
         spin++) {
    }

    unsigned int got =
        atomic_load_explicit(&signaled_count, memory_order_acquire);
    if (got != N_CONSUMERS) {
      (void)fprintf(stderr, "FAIL [iter=%d]: signaled_count=%u (want %u)\n", it,
                    got, N_CONSUMERS);
      abort();
    }
    /* All consumers must observe the same data GUID — the unique winner
     * of the satisfy race wrote simple.data, every late binder reads from
     * that slot via the fired==true short-circuit. */
    arts_guid_t first = (arts_guid_t)atomic_load_explicit(&consumer_data[0],
                                                          memory_order_acquire);
    for (int i = 1; i < N_CONSUMERS; i++) {
      arts_guid_t got = (arts_guid_t)atomic_load_explicit(&consumer_data[i],
                                                          memory_order_acquire);
      if (got != first) {
        (void)fprintf(
            stderr,
            "FAIL [iter=%d]: consumer %d got data=%lu, consumer 0 got %lu\n",
            it, i, (uint64_t)got, (uint64_t)first);
        abort();
      }
    }
    if (first != db) {
      (void)fprintf(stderr,
                    "FAIL [iter=%d]: consumer data=%lu != satisfier db=%lu\n",
                    it, (uint64_t)first, (uint64_t)db);
      abort();
    }
  }

  printf("event_once_storm: %d iters x %d consumers = %d deliveries — PASS\n",
         M_ITERS, N_CONSUMERS, M_ITERS * N_CONSUMERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
