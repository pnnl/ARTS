/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_channel_fifo — Phase: event-redesign, Task §6.4.
 *
 * CHANNEL FIFO ordering test.  A single producer EDT pushes K satisfies
 * with strictly-increasing payloads (sequence numbers boxed in DB GUIDs);
 * a single consumer EDT chain registers K addDeps in order.  Every
 * dispatched counter EDT records the received sequence number; verify_edt
 * compares the recovered sequence against the producer order.
 *
 * Producer-before-consumer satisfy/addDep stresses the buffering path
 * (mpsc data_queue accumulating up to K entries before any matching
 * dep arrives).  This is the exact pattern that broke under the legacy
 * Treiber-stack-with-drop-on-empty-drain implementation.
 *
 * No EDT busy-waits: an EDT may only wait via events/dependencies, never
 * by spinning on an atomic (a spinning EDT occupies a worker and, while it
 * also creator-holds the iteration's RW DBs, blocks the RW counter EDTs
 * from ever acquiring under strict single-writer coherence).  main_edt
 * sets up all iterations and TERMINATES, releasing every creator-hold so
 * the RW counter EDTs can proceed; one LATCH event sized M_ITERS*K_GENS
 * fans the storm into a verify_edt that checks FIFO once it drains.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#define M_ITERS 16
#define K_GENS 256

/* recovered_seq[it][g] = data DB GUID delivered to the addDep registered
 * g-th within iteration `it`.  CHANNEL FIFO guarantees the g-th dep gets
 * the g-th satisfy's data, so we test that exact relationship —
 * independent of counter_edt execution order on the worker pool. */
static atomic_uint received_count = 0;
static atomic_ulong recovered_seq[M_ITERS][K_GENS];
/* data_dbs[it][g] = the g-th data DB pushed into iteration `it`'s CHANNEL. */
static arts_guid_t data_dbs[M_ITERS][K_GENS];
static atomic_int g_clean_shutdown = 0;

/* Counter EDT — paramv: [latch_guid, it, g]. */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  uint64_t it = paramv[1];
  uint64_t g = paramv[2];
  if (it < M_ITERS && g < K_GENS) {
    atomic_store_explicit(&recovered_seq[it][g], (uint64_t)depv[0].guid,
                          memory_order_release);
  }
  atomic_fetch_add_explicit(&received_count, 1u, memory_order_acq_rel);
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* verify_edt — bound to the storm-wide LATCH (slot 0, DB_MODE_NULL). */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int got =
      atomic_load_explicit(&received_count, memory_order_acquire);
  if (got != (unsigned int)(M_ITERS * K_GENS)) {
    (void)fprintf(stderr, "FAIL: received_count=%u (want %u)\n", got,
                  (unsigned int)(M_ITERS * K_GENS));
    arts_abort(1);
  }

  /* Verify FIFO: recovered_seq[it][i] must equal data_dbs[it][i]. */
  for (int it = 0; it < M_ITERS; it++) {
    for (int i = 0; i < K_GENS; i++) {
      arts_guid_t r = (arts_guid_t)atomic_load_explicit(&recovered_seq[it][i],
                                                        memory_order_acquire);
      if (r != data_dbs[it][i]) {
        (void)fprintf(
            stderr,
            "FAIL [iter=%d]: gen %d recovered=%lu, expected=%lu (FIFO "
            "broken)\n",
            it, i, (uint64_t)r, (uint64_t)data_dbs[it][i]);
        arts_abort(1);
      }
    }
  }

  atomic_store(&g_clean_shutdown, 1);
  printf("event_channel_fifo: %d iters x %d gens = %d deliveries (FIFO "
         "ordered) — PASS\n",
         M_ITERS, K_GENS, M_ITERS * K_GENS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(M_ITERS * K_GENS);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: arts_event_create LATCH returned NULL_GUID\n");
    arts_abort(1);
  }

  for (int it = 0; it < M_ITERS; it++) {
    arts_event_hint_t h = ARTS_EVENT_HINT_CHANNEL;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: arts_event_create CHANNEL\n", it);
      arts_abort(1);
    }

    /* producer pushes K satisfies with unique data DBs. */
    for (int g = 0; g < K_GENS; g++) {
      void *dbp = NULL;
      data_dbs[it][g] = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                       ARTS_DB_PROP_NONE, NULL);
      arts_event_satisfy(ev, data_dbs[it][g]);
    }

    /* consumer registers K addDeps in order.  CHANNEL FIFO matches
     * data_dbs[it][g] with addDep[g]; each counter EDT is told its own g
     * via paramv so the comparison is independent of execution order on
     * the worker pool. */
    for (int g = 0; g < K_GENS; g++) {
      uint64_t pv[3] = {(uint64_t)latch, (uint64_t)it, (uint64_t)g};
      arts_guid_t edt = arts_edt_create(counter_edt, 3, pv, 1, NULL);
      arts_add_dependence(ev, edt, 0, DB_MODE_RW);
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
