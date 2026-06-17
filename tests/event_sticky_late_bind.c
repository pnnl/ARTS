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
 *   - Verify every consumer received the same data GUID via the
 *     fired==true immediate-deliver path (spec §3.1 R1).
 *
 * No EDT busy-waits: an EDT may only wait via events/dependencies, never
 * by spinning on an atomic (a spinning EDT occupies a worker and, while it
 * also creator-holds the iteration's RW DB, blocks the RW consumers from
 * ever acquiring under strict single-writer coherence).  main_edt sets up
 * all iterations and TERMINATES, releasing every creator-hold so the RW
 * counter EDTs can proceed; one LATCH event sized M_ITERS*N_CONSUMERS fans
 * the storm into a verify_edt that performs the checks once it drains.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#define M_ITERS 32
#define N_CONSUMERS 16

static atomic_uint signaled_count = 0;
/* Per-iteration slots — index [it][consumer]. */
static atomic_ulong consumer_data[M_ITERS][N_CONSUMERS];
/* iter_db[it] = the data DB satisfied into iteration `it`'s event. */
static arts_guid_t iter_db[M_ITERS];
static atomic_int g_clean_shutdown = 0;

/* Counter EDT — paramv: [latch_guid, it, idx]. */
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

/* verify_edt — bound to the storm-wide LATCH (slot 0, DB_MODE_NULL). */
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
    for (int i = 0; i < N_CONSUMERS; i++) {
      arts_guid_t cd = (arts_guid_t)atomic_load_explicit(&consumer_data[it][i],
                                                         memory_order_acquire);
      if (cd != iter_db[it]) {
        (void)fprintf(stderr,
                      "FAIL [iter=%d]: consumer %d got %lu (want %lu)\n", it, i,
                      (uint64_t)cd, (uint64_t)iter_db[it]);
        arts_abort(1);
      }
    }
  }

  atomic_store(&g_clean_shutdown, 1);
  printf("event_sticky_late_bind: %d iters x %d consumers = %d deliveries — "
         "PASS\n",
         M_ITERS, N_CONSUMERS, M_ITERS * N_CONSUMERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(M_ITERS * N_CONSUMERS);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: arts_event_create LATCH returned NULL_GUID\n");
    arts_abort(1);
  }

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
    iter_db[it] = db;

    /* Pre-satisfy.  STICKY persists, so all later addDeps must immediately
     * deliver from event->simple.data. */
    arts_event_satisfy(ev, db);

    /* Spawn consumer EDTs that addDep on the (already-fired) STICKY. */
    for (int i = 0; i < N_CONSUMERS; i++) {
      uint64_t consumer_pv[4] = {(uint64_t)ev, (uint64_t)latch, (uint64_t)it,
                                 (uint64_t)i};
      arts_edt_create(consumer_edt, 4, consumer_pv, 0, NULL);
    }
  }

  /* verify_edt fires after the whole storm drains the latch. */
  arts_guid_t v = arts_edt_create(verify_edt, 0, NULL, 1, NULL);
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);

  /* main_edt terminates, releasing the creator-hold RW on every iteration
   * DB so the RW counter EDTs can acquire. */
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
