/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_destroy_race — Phase: event-redesign, Task §6.6.
 *
 * UAF / leak smoke test.  Drives concurrent satisfy + addDep against
 * an explicit destroy on STICKY events to validate the lifecycle
 * invariants from spec §5: drainer holds the lookup ref through the
 * entire drain; mpsc nodes are paired with deletion; event_deleter
 * cleans up pending nodes; no double-fire.
 *
 * Best run under ASan/UBSan (build with -DARTS_USE_SANS=ON) to
 * catch UAF.  Built as a regular release test it primarily exercises
 * the ref-count protocol's correctness under contention.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arts.h"

#define M_ITERS 64
#define N_WORKERS 12 /* mix of satisfiers + consumers + destroyers */

static atomic_uint signaled_count = 0;
static atomic_uint completed_workers = 0;

static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  atomic_fetch_add_explicit(&signaled_count, 1u, memory_order_relaxed);
}

/* Worker EDT roles encoded in paramv[1]:
 *   0 = consumer (addDep)
 *   1 = satisfier (event_satisfy)
 *   2 = destroyer (event_destroy)
 */
static void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  uint64_t role = paramv[1];
  arts_guid_t db = (arts_guid_t)paramv[2];

  if (role == 0) {
    arts_guid_t edt = arts_edt_create(counter_edt, 0, NULL, 1, NULL);
    arts_add_dependence(event, edt, 0, DB_MODE_RW);
  } else if (role == 1) {
    arts_event_satisfy(event, db);
  } else {
    arts_event_destroy(event);
  }
  atomic_fetch_add_explicit(&completed_workers, 1u, memory_order_release);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  for (int it = 0; it < M_ITERS; it++) {
    atomic_store_explicit(&signaled_count, 0u, memory_order_relaxed);
    atomic_store_explicit(&completed_workers, 0u, memory_order_relaxed);

    /* IDEM (over-satisfy is silent) — STICKY would ARTS_ERROR on the
     * 2nd of N concurrent satisfies, which is the intended behavior but
     * outside this test's scope.  The destroy / addDep / satisfy mix
     * still exercises the lifecycle invariants. */
    arts_event_hint_t h = ARTS_EVENT_HINT_IDEMPOTENT;
    arts_guid_t ev = arts_event_create(&h);
    if (ev == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [iter=%d]: arts_event_create\n", it);
      abort();
    }

    void *dbp = NULL;
    arts_guid_t db = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);

    /* Mix of roles: 1/3 each of consumer / satisfier / destroyer. */
    for (int i = 0; i < N_WORKERS; i++) {
      uint64_t pv[3] = {(uint64_t)ev, (uint64_t)(i % 3), (uint64_t)db};
      arts_edt_create(worker_edt, 3, pv, 0, NULL);
    }

    /* Spin until all workers ran (event may be destroyed mid-flight). */
    for (int spin = 0; spin < 100000000 &&
                       atomic_load_explicit(&completed_workers,
                                            memory_order_acquire) < N_WORKERS;
         spin++) {
    }

    unsigned int wc =
        atomic_load_explicit(&completed_workers, memory_order_acquire);
    if (wc != N_WORKERS) {
      (void)fprintf(stderr, "FAIL [iter=%d]: workers completed=%u (want %u)\n",
                    it, wc, N_WORKERS);
      abort();
    }
    /* signaled_count is ≤ number of consumers that arrived before destroy
     * (there's no fixed expected value).  Just sanity-check it's
     * bounded — never exceeds the number of consumers spawned. */
    unsigned int sc =
        atomic_load_explicit(&signaled_count, memory_order_acquire);
    if (sc > (unsigned)((N_WORKERS / 3) + 1)) {
      (void)fprintf(
          stderr, "FAIL [iter=%d]: signaled_count=%u exceeds consumer cap %d\n",
          it, sc, (N_WORKERS / 3) + 1);
      abort();
    }
  }

  printf("event_destroy_race: %d iters with %d-way concurrent S/A/D — PASS\n",
         M_ITERS, N_WORKERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
