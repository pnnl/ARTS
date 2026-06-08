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
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arts.h"

#define M_ITERS 32
#define N_CONSUMERS 16

static atomic_uint signaled_count = 0;
static atomic_ulong consumer_data[N_CONSUMERS];

static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t idx = paramv[0];
  if (idx < N_CONSUMERS) {
    atomic_store_explicit(&consumer_data[idx], (uint64_t)depv[0].guid,
                          memory_order_release);
  }
  atomic_fetch_add_explicit(&signaled_count, 1u, memory_order_acq_rel);
}

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

    arts_event_hint_t h = ARTS_EVENT_HINT_STICKY;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: arts_event_create\n", it);
      abort();
    }

    void *dbp = NULL;
    arts_guid_t db = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    if (db == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: arts_db_create\n", it);
      abort();
    }

    /* Pre-satisfy.  STICKY persists, so all later addDeps must immediately
     * deliver from event->simple.data. */
    arts_event_satisfy(ev, db);

    /* Spawn consumer EDTs that addDep on the (already-fired) STICKY. */
    for (int i = 0; i < N_CONSUMERS; i++) {
      uint64_t consumer_pv[2] = {(uint64_t)ev, (uint64_t)i};
      arts_edt_create(consumer_edt, 2, consumer_pv, 0, NULL);
    }

    /* Spin until all consumer EDTs ran. */
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
    for (int i = 0; i < N_CONSUMERS; i++) {
      arts_guid_t cd = (arts_guid_t)atomic_load_explicit(&consumer_data[i],
                                                         memory_order_acquire);
      if (cd != db) {
        (void)fprintf(stderr,
                      "FAIL [iter=%d]: consumer %d got %lu (want %lu)\n", it, i,
                      (uint64_t)cd, (uint64_t)db);
        abort();
      }
    }

    /* STICKY does not auto-destroy; explicit destroy. */
    arts_event_destroy(ev);
  }

  printf("event_sticky_late_bind: %d iters x %d consumers = %d deliveries — "
         "PASS\n",
         M_ITERS, N_CONSUMERS, M_ITERS * N_CONSUMERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
