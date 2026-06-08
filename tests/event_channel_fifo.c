/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_channel_fifo — Phase: event-redesign, Task §6.4.
 *
 * CHANNEL FIFO ordering test.  A single producer EDT pushes K satisfies
 * with strictly-increasing payloads (sequence numbers boxed in DB GUIDs);
 * a single consumer EDT chain registers K addDeps in order.  Every
 * dispatched counter EDT records the received sequence number; main_edt
 * compares the recovered sequence against the producer order.
 *
 * Producer-before-consumer satisfy/addDep stresses the buffering path
 * (mpsc data_queue accumulating up to K entries before any matching
 * dep arrives).  This is the exact pattern that broke under the legacy
 * Treiber-stack-with-drop-on-empty-drain implementation.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arts.h"

#define M_ITERS 16
#define K_GENS 256

/* recovered_seq[g] = data DB GUID delivered to the addDep registered g-th.
 * CHANNEL FIFO guarantees the g-th dep gets the g-th satisfy's data, so
 * we test that exact relationship — independent of counter_edt
 * execution order on the worker pool. */
static atomic_uint received_count = 0;
static atomic_ulong recovered_seq[K_GENS];

static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t expected_idx = paramv[0];
  if (expected_idx < K_GENS) {
    atomic_store_explicit(&recovered_seq[expected_idx], (uint64_t)depv[0].guid,
                          memory_order_release);
  }
  atomic_fetch_add_explicit(&received_count, 1u, memory_order_acq_rel);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  for (int it = 0; it < M_ITERS; it++) {
    atomic_store_explicit(&received_count, 0u, memory_order_relaxed);
    for (int i = 0; i < K_GENS; i++) {
      atomic_store_explicit(&recovered_seq[i], 0ul, memory_order_relaxed);
    }

    arts_event_hint_t h = ARTS_EVENT_HINT_CHANNEL;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: arts_event_create CHANNEL\n", it);
      abort();
    }

    /* producer pushes K satisfies with unique data DBs. */
    arts_guid_t data_dbs[K_GENS];
    for (int g = 0; g < K_GENS; g++) {
      void *dbp = NULL;
      data_dbs[g] = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                   ARTS_DB_PROP_NONE, NULL);
      arts_event_satisfy(ev, data_dbs[g]);
    }

    /* consumer registers K addDeps in order.  CHANNEL FIFO
     * matches data_dbs[g] with addDep[g]; each counter EDT is told its
     * own g via paramv so the comparison is independent of execution
     * order on the worker pool. */
    for (int g = 0; g < K_GENS; g++) {
      uint64_t pv[1] = {(uint64_t)g};
      arts_guid_t edt = arts_edt_create(counter_edt, 1, pv, 1, NULL);
      arts_add_dependence(ev, edt, 0, DB_MODE_RW);
    }

    /* Spin until all K consumers fired. */
    for (int spin = 0;
         spin < 200000000 &&
         atomic_load_explicit(&received_count, memory_order_acquire) < K_GENS;
         spin++) {
    }

    unsigned int got =
        atomic_load_explicit(&received_count, memory_order_acquire);
    if (got != K_GENS) {
      (void)fprintf(stderr, "FAIL [iter=%d]: received_count=%u (want %u)\n", it,
                    got, K_GENS);
      abort();
    }

    /* Verify FIFO: recovered_seq[i] must equal data_dbs[i] for all i. */
    for (int i = 0; i < K_GENS; i++) {
      arts_guid_t r = (arts_guid_t)atomic_load_explicit(&recovered_seq[i],
                                                        memory_order_acquire);
      if (r != data_dbs[i]) {
        (void)fprintf(
            stderr,
            "FAIL [iter=%d]: gen %d recovered=%lu, expected=%lu (FIFO "
            "broken)\n",
            it, i, (uint64_t)r, (uint64_t)data_dbs[i]);
        abort();
      }
    }

    arts_event_destroy(ev);
  }

  printf("event_channel_fifo: %d iters x %d gens = %d deliveries (FIFO "
         "ordered) — PASS\n",
         M_ITERS, K_GENS, M_ITERS * K_GENS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
