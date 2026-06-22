/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_sticky_late_bind — Phase: event-redesign, Task §6.2.
 *
 * STICKY persist + concurrent late-bind delivery.  Validates that
 * STICKY events correctly serve cached data to addDep callers that
 * arrive after the satisfy.
 *
 * For each iteration:
 *   - Create one STICKY event.
 *   - Pre-satisfy with a unique data DB.
 *   - Spawn N_CONSUMERS consumer EDTs that each addDep the event.
 *   - Verify every consumer received the same data GUID.
 *
 * No EDT busy-waits and no global cross-EDT state.  All coordination data
 * lives in two DBs passed via deps:
 *   ref_db     — reference GUIDs (one per iteration) written by main_edt.
 *   results_db — received GUIDs written by counter_edts via RW dep;
 *                read by verify_edt (RO dep) after finish scope drains.
 *
 * verify_edt deps: slot 0 = finish event (DB_MODE_NULL),
 *                  slot 1 = ref_db (RO), slot 2 = results_db (RO).
 * counter_edt deps: slot 0 = STICKY data DB (RW),
 *                   slot 1 = results_db (RW).
 */

#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#define M_ITERS 32
#define N_CONSUMERS 16
#define TOTAL (M_ITERS * N_CONSUMERS)

/* Counter EDT — paramv: [it, idx].
 * depv[0] = data DB delivered by STICKY event (RW).
 * depv[1] = results_db (RW): write received GUID into slot
 * [it*N_CONSUMERS+idx]. */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t it = paramv[0];
  uint64_t idx = paramv[1];
  arts_guid_t received = depv[0].guid;
  arts_guid_t *rdb = (arts_guid_t *)depv[1].ptr;
  if (rdb == NULL) {
    arts_printf("FAIL: counter_edt results_db ptr NULL\n");
    arts_abort(1);
  }
  rdb[(int)(it * N_CONSUMERS + idx)] = received;
}

/* Consumer worker EDT: create counter_edt + addDep on event.
 * paramv: [event_guid, it, idx, results_db_guid] */
static void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_guid_t results_db = (arts_guid_t)paramv[3];
  uint64_t counter_pv[2] = {paramv[1], paramv[2]};
  arts_guid_t edt = arts_edt_create(counter_edt, 2, counter_pv, 2, NULL);
  arts_add_dependence(event, edt, 0, DB_MODE_RW);
  arts_add_dependence(results_db, edt, 1, DB_MODE_RW);
}

/* verify_edt — fires after finish scope drains.
 * depv[0] = finish event (DB_MODE_NULL).
 * depv[1] = ref_db (RO): one reference GUID per iteration.
 * depv[2] = results_db (RO): N_CONSUMERS received GUIDs per iteration. */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;

  const arts_guid_t *ref = (const arts_guid_t *)depv[1].ptr;
  const arts_guid_t *got = (const arts_guid_t *)depv[2].ptr;
  if (ref == NULL || got == NULL) {
    (void)fprintf(stderr, "FAIL: verify_edt got NULL DB ptr\n");
    arts_abort(1);
  }

  for (int it = 0; it < M_ITERS; it++) {
    for (int i = 0; i < N_CONSUMERS; i++) {
      arts_guid_t cd = got[it * N_CONSUMERS + i];
      if (cd != ref[it]) {
        (void)fprintf(stderr,
                      "FAIL [iter=%d]: consumer %d got %lu (want %lu)\n", it, i,
                      (uint64_t)cd, (uint64_t)ref[it]);
        arts_abort(1);
      }
    }
  }

  arts_printf("event_sticky_late_bind: %d iters x %d consumers = %d "
              "deliveries — PASS\n",
              M_ITERS, N_CONSUMERS, TOTAL);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* ref_db: M_ITERS arts_guid_t slots — one reference GUID per iteration,
   * written here and read by verify_edt (RO dep). */
  arts_guid_t *ref = NULL;
  arts_guid_t ref_db =
      arts_db_create((void **)&ref, sizeof(arts_guid_t) * M_ITERS, ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);

  /* results_db: TOTAL arts_guid_t slots — written by counter_edts (RW dep,
   * serialised), read by verify_edt (RO dep). */
  arts_guid_t *res = NULL;
  arts_guid_t results_db =
      arts_db_create((void **)&res, sizeof(arts_guid_t) * TOTAL, ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);
  for (int i = 0; i < TOTAL; i++) {
    res[i] = NULL_GUID;
  }
  arts_db_release(results_db, DB_MODE_RW);

  /* finish event: all consumer_edts (and their counter_edts) join it;
   * verify_edt deps on it. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (int it = 0; it < M_ITERS; it++) {
    arts_event_hint_t h = ARTS_EVENT_HINT_STICKY;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: arts_event_create\n", it);
      arts_abort(1);
    }

    void *dbp = NULL;
    arts_guid_t db = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    if (db == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: arts_db_create\n", it);
      arts_abort(1);
    }
    ref[it] = db;

    /* Pre-satisfy.  STICKY persists, so all later addDeps deliver immediately
     * from event->simple.data. */
    arts_event_satisfy(ev, db);

    /* Spawn consumer EDTs that addDep on the (already-fired) STICKY.
     * consumer_edt itself creates the counter_edt and wires its deps. */
    for (int i = 0; i < N_CONSUMERS; i++) {
      uint64_t consumer_pv[4] = {(uint64_t)ev, (uint64_t)it, (uint64_t)i,
                                 (uint64_t)results_db};
      arts_edt_create(consumer_edt, 4, consumer_pv, 0,
                      &(arts_edt_hint_t){.finish_event = fe});
    }
  }

  /* Release ref_db creator hold so verify_edt can acquire it RO. */
  arts_db_release(ref_db, DB_MODE_RW);

  /* verify_edt: slot 0 = fe (NULL), slot 1 = ref_db (RO),
   *             slot 2 = results_db (RO). */
  arts_guid_t v = arts_edt_create(verify_edt, 0, NULL, 3, NULL);
  arts_add_dependence(fe, v, 0, DB_MODE_NULL);
  arts_add_dependence(ref_db, v, 1, DB_MODE_RO);
  arts_add_dependence(results_db, v, 2, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
