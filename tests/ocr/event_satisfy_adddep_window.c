/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_satisfy_adddep_window — C15 / T155.
 *
 * Target: the simple-event satisfy <-> addDep race window
 *   - handler_event_satisfy_slot: prev==1 is the UNIQUE fire trigger — only
 *     that thread writes simple.data and CAS-sets fired.
 *   - arts_add_dependence (event source): push onto the Treiber stack, then
 *     re-load fired (race rescue) and drain — but it MUST NOT CAS fired and
 *     MUST NOT write data.
 *
 * Beyond event_once_storm (which checks "all consumers see the same data"),
 * this test pins WHICH data every consumer must see: the data written by the
 * single prev==1 satisfy.  A second, racing over-satisfier carries a DISTINCT
 * data DB.  Because the latch starts at 1, exactly one DECR satisfy (prev==1)
 * fires and writes data; every later DECR is prev<=0 and silently absorbed
 * WITHOUT touching simple.data.  Therefore every delivered GUID must equal
 * exactly one of the two satisfy payloads, and ALL consumers in one iteration
 * must agree (the data slot is written once).  If a future regression let
 * addDep CAS fired or an over-satisfy rewrite data, consumers within an
 * iteration would disagree.
 *
 * To make the window real, for each iteration the unique satisfier, the
 * over-satisfier, and N late add-deppers all run as depc=0 EDTs so they race.
 *
 * PASS criterion: M*N deliveries, per iteration all consumers observe the SAME
 * data GUID, and that GUID is one of the two satisfy payloads.  A stranded
 * consumer => ctest TIMEOUT.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define M_ITERS 48
#define N_DEPS 8
#define TOTAL (M_ITERS * N_DEPS)

/* State DB (uint64_t):
 *   [0]                          — delivered_count (atomic)
 *   [1 .. TOTAL]                 — cdata[it*N_DEPS+idx]
 *   [TOTAL+1 .. TOTAL+M_ITERS]   — payload_a[it]
 *   [TOTAL+M_ITERS+1 ..
 *        TOTAL+2*M_ITERS]        — payload_b[it]
 */
#define ST_DELIVERED 0
#define ST_CDATA 1
#define ST_PA (ST_CDATA + TOTAL)
#define ST_PB (ST_PA + M_ITERS)
#define ST_NELEMS (ST_PB + M_ITERS)

/* counter_edt — paramv: [state_db, latch, it, idx].
 * depv[0] = event data (RW), depv[1] = state DB (RW). */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[1];
  uint64_t it = paramv[2];
  uint64_t idx = paramv[3];
  uint64_t *state = (uint64_t *)depv[1].ptr;

  atomic_store_explicit(
      (_Atomic uint64_t *)&state[ST_CDATA + it * N_DEPS + idx],
      (uint64_t)depv[0].guid, memory_order_release);
  atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_DELIVERED], 1u,
                            memory_order_acq_rel);
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* satisfier_edt — paramv: [event, data]. */
static void satisfier_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_event_satisfy((arts_guid_t)paramv[0], (arts_guid_t)paramv[1]);
}

/* depper_edt — register one counter onto the event.
 * paramv: [event, state_db, latch, it, idx]. */
static void depper_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  uint64_t cpv[4] = {paramv[1], paramv[2], paramv[3], paramv[4]};
  arts_guid_t edt = arts_edt_create(counter_edt, 4, cpv, 2, NULL);
  arts_add_dependence(event, edt, 0, DB_MODE_RW);
  arts_add_dependence((arts_guid_t)paramv[1], edt, 1, DB_MODE_RW);
}

static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *state = (uint64_t *)depv[1].ptr;

  uint64_t got = atomic_load_explicit((_Atomic uint64_t *)&state[ST_DELIVERED],
                                      memory_order_acquire);
  if (got != (uint64_t)TOTAL) {
    (void)fprintf(stderr, "FAIL: delivered=%llu want %d\n",
                  (unsigned long long)got, TOTAL);
    arts_abort(1);
  }
  for (int it = 0; it < M_ITERS; it++) {
    arts_guid_t pa = (arts_guid_t)atomic_load_explicit(
        (_Atomic uint64_t *)&state[ST_PA + it], memory_order_acquire);
    arts_guid_t pb = (arts_guid_t)atomic_load_explicit(
        (_Atomic uint64_t *)&state[ST_PB + it], memory_order_acquire);
    arts_guid_t first = (arts_guid_t)atomic_load_explicit(
        (_Atomic uint64_t *)&state[ST_CDATA + it * N_DEPS + 0],
        memory_order_acquire);
    if (first != pa && first != pb) {
      (void)fprintf(stderr,
                    "FAIL [it=%d]: delivered data=%lu is neither satisfy "
                    "payload (a=%lu b=%lu)\n",
                    it, (uint64_t)first, (uint64_t)pa, (uint64_t)pb);
      arts_abort(1);
    }
    for (int i = 1; i < N_DEPS; i++) {
      arts_guid_t ci = (arts_guid_t)atomic_load_explicit(
          (_Atomic uint64_t *)&state[ST_CDATA + it * N_DEPS + i],
          memory_order_acquire);
      if (ci != first) {
        (void)fprintf(stderr,
                      "FAIL [it=%d]: consumer %d got %lu, consumer 0 got %lu "
                      "(data written more than once)\n",
                      it, i, (uint64_t)ci, (uint64_t)first);
        arts_abort(1);
      }
    }
  }
  printf("event_satisfy_adddep_window: %d iters x %d deps = %d deliveries, "
         "data written once by the unique fire — PASS\n",
         M_ITERS, N_DEPS, TOTAL);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  uint64_t *state = NULL;
  arts_guid_t state_db =
      arts_db_create((void **)&state, ST_NELEMS * sizeof(uint64_t), ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);
  memset(state, 0, ST_NELEMS * sizeof(uint64_t));

  arts_event_hint_t lh = ARTS_EVENT_HINT_LATCH(TOTAL);
  lh.rank = 0;
  arts_guid_t latch = arts_event_create(&lh);
  if (latch == NULL_GUID || state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: create NULL_GUID\n");
    arts_abort(1);
  }

  for (int it = 0; it < M_ITERS; it++) {
    /* IDEM: latch=1, over-satisfy silently absorbed, event lingers so late
     * add-deppers still deliver from the fire data. */
    arts_event_hint_t h = ARTS_EVENT_HINT_IDEMPOTENT;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [it=%d]: event create\n", it);
      arts_abort(1);
    }

    void *ap = NULL;
    void *bp = NULL;
    arts_guid_t da =
        arts_db_create(&ap, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
    arts_guid_t db =
        arts_db_create(&bp, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
    state[ST_PA + it] = (uint64_t)da;
    state[ST_PB + it] = (uint64_t)db;

    /* Two satisfiers (one wins the prev==1 fire) + N late add-deppers, all
     * depc=0 so they race the satisfy<->addDep window. */
    uint64_t spa[2] = {(uint64_t)ev, (uint64_t)da};
    uint64_t spb[2] = {(uint64_t)ev, (uint64_t)db};
    arts_edt_create(satisfier_edt, 2, spa, 0, NULL);
    arts_edt_create(satisfier_edt, 2, spb, 0, NULL);
    for (int i = 0; i < N_DEPS; i++) {
      uint64_t dpv[5] = {(uint64_t)ev, (uint64_t)state_db, (uint64_t)latch,
                         (uint64_t)it, (uint64_t)i};
      arts_edt_create(depper_edt, 5, dpv, 0, NULL);
    }
  }

  arts_db_release(state_db, DB_MODE_RW);

  arts_guid_t v = arts_edt_create(verify_edt, 0, NULL, 2, NULL);
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);
  arts_add_dependence(state_db, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
