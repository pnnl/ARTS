/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_channel_fifo — Phase: event-redesign, Task §6.4.
 *
 * CHANNEL FIFO ordering test.  A single producer EDT pushes K satisfies
 * with strictly-increasing payloads (sequence numbers boxed in DB GUIDs);
 * a single consumer EDT chain registers K addDeps in order.  Every
 * dispatched counter EDT records the received GUID into its slot of a shared
 * results DB via a RW dep; verify_edt compares the recovered sequence against
 * the producer order once the finish scope drains.
 *
 * Producer-before-consumer satisfy/addDep stresses the buffering path
 * (mpsc data_queue accumulating up to K entries before any matching
 * dep arrives).
 *
 * No EDT busy-waits and no global cross-EDT state.  All coordination data
 * lives in two DBs passed via deps:
 *   ref_db     — reference GUIDs written by main_edt (RO dep of verify_edt).
 *   results_db — recovered GUIDs written by counter_edts via RW dep;
 *                RO dep of verify_edt after the finish scope drains.
 *
 * verify_edt deps: slot 0 = finish event (DB_MODE_NULL), slot 1 = ref_db (RO),
 *                  slot 2 = results_db (RO).
 * counter_edt deps: slot 0 = CHANNEL data DB (RW), slot 1 = results_db (RW).
 */

#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#define M_ITERS 16
#define K_GENS 256
#define TOTAL (M_ITERS * K_GENS)

/* Counter EDT — paramv: [it, g].
 * depv[0] = data DB delivered by CHANNEL satisfy (RW).
 * depv[1] = results_db (RW): write received GUID into slot [it*K_GENS+g]. */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t it = paramv[0];
  uint64_t g = paramv[1];
  arts_guid_t received = depv[0].guid;
  arts_guid_t *rdb = (arts_guid_t *)depv[1].ptr;
  if (rdb == NULL) {
    arts_printf("FAIL: counter_edt results_db ptr NULL\n");
    arts_abort(1);
  }
  rdb[(int)(it * K_GENS + g)] = received;
}

/* verify_edt — fires after finish scope drains.
 * depv[0] = finish event (DB_MODE_NULL).
 * depv[1] = ref_db (RO): reference GUIDs written by main_edt.
 * depv[2] = results_db (RO): recovered GUIDs written by counter_edts. */
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

  /* Verify FIFO: recovered[it*K_GENS+g] must equal reference[it*K_GENS+g]. */
  for (int it = 0; it < M_ITERS; it++) {
    for (int g = 0; g < K_GENS; g++) {
      int idx = it * K_GENS + g;
      if (got[idx] != ref[idx]) {
        (void)fprintf(
            stderr,
            "FAIL [iter=%d]: gen %d recovered=%lu, expected=%lu (FIFO "
            "broken)\n",
            it, g, (uint64_t)got[idx], (uint64_t)ref[idx]);
        arts_abort(1);
      }
    }
  }

  arts_printf("event_channel_fifo: %d iters x %d gens = %d deliveries (FIFO "
              "ordered) — PASS\n",
              M_ITERS, K_GENS, TOTAL);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* ref_db: TOTAL arts_guid_t slots — reference GUIDs set here and read by
   * verify_edt. Released RW by main_edt before any counter_edt can acquire. */
  arts_guid_t *ref = NULL;
  arts_guid_t ref_db =
      arts_db_create((void **)&ref, sizeof(arts_guid_t) * TOTAL, ARTS_DB,
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

  /* finish event: all counter_edts join it; verify_edt deps on it. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (int it = 0; it < M_ITERS; it++) {
    arts_event_hint_t h = ARTS_EVENT_HINT_CHANNEL;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: arts_event_create CHANNEL\n", it);
      arts_abort(1);
    }

    /* producer pushes K satisfies with unique data DBs; record reference. */
    for (int g = 0; g < K_GENS; g++) {
      void *dbp = NULL;
      arts_guid_t dg = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                      ARTS_DB_PROP_NONE, NULL);
      ref[it * K_GENS + g] = dg;
      arts_event_satisfy(ev, dg);
    }

    /* consumer registers K addDeps in order; each counter_edt also gets
     * results_db as slot 1 (RW) to record its received GUID. */
    for (int g = 0; g < K_GENS; g++) {
      uint64_t pv[2] = {(uint64_t)it, (uint64_t)g};
      arts_guid_t edt = arts_edt_create(counter_edt, 2, pv, 2,
                                        &(arts_edt_hint_t){.finish_event = fe});
      arts_add_dependence(ev, edt, 0, DB_MODE_RW);
      arts_add_dependence(results_db, edt, 1, DB_MODE_RW);
    }
  }

  /* Release ref_db creator hold so verify_edt can acquire it RO. */
  arts_db_release(ref_db, DB_MODE_RW);

  /* verify_edt: slot 0 = fe, slot 1 = ref_db (RO), slot 2 = results_db (RO).
   * The finish event guarantees all counter_edts have released their RW dep
   * on results_db before verify_edt acquires it RO. */
  arts_guid_t v = arts_edt_create(verify_edt, 0, NULL, 3, NULL);
  arts_add_dependence(fe, v, 0, DB_MODE_NULL);
  arts_add_dependence(ref_db, v, 1, DB_MODE_RO);
  arts_add_dependence(results_db, v, 2, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
