/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_destroy_race — Phase: event-redesign, Task §6.6.
 *
 * UAF / leak smoke test.  Drives concurrent satisfy + addDep against an
 * explicit destroy on IDEM events to validate the lifecycle invariants from
 * spec §5: drainer holds the lookup ref through the entire drain; mpsc nodes
 * are paired with deletion; event_deleter cleans up pending nodes; no
 * double-fire.
 *
 * Best run under ASan/UBSan (build with -DARTS_USE_SANS=ON) to catch UAF.
 * Built as a regular release test it primarily exercises the ref-count
 * protocol's correctness under contention.
 *
 * Per iteration: N_WORKERS worker EDTs (1/3 each consumer/satisfier/destroyer)
 * run inside a finish event; an iter_verify_edt gated on that finish event
 * checks every worker completed, then chains the next iteration.
 *
 * The marker EDTs that consumers addDep onto the event are created OUTSIDE the
 * finish scope, by start_iter (which is itself finish-free).  A destroyer may
 * leave a marker forever pending; a pending EDT *inside* the finish scope would
 * deadlock the verify EDT (the finish event waits on its transitive children,
 * so it would never fire).  Keeping the markers finish-free makes that pending
 * harmless.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#define M_ITERS 64
#define N_WORKERS 12

/* Counter DB layout: one _Atomic counter for completed workers. */
typedef struct {
  _Atomic unsigned int completed_workers;
} iter_counters_t;

/* counter_edt: a marker a consumer addDep's onto the IDEM event.  Created
 * finish-free by start_iter, owns no DB — reaching here just means the event
 * fired it before a destroyer tore it down. */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

/* Worker roles in paramv[2]: 0=consumer(addDep), 1=satisfier, 2=destroyer.
 *   paramv[0] = event guid
 *   paramv[1] = db guid (payload for satisfy)
 *   paramv[2] = role
 *   paramv[3] = pre-created marker guid (consumer only; NULL_GUID otherwise)
 *
 * depv[0] = iter_counters DB (RW) — worker increments completed_workers.
 */
static void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t event = (arts_guid_t)paramv[0];
  arts_guid_t db = (arts_guid_t)paramv[1];
  uint64_t role = paramv[2];

  if (role == 0) {
    /* consumer: addDep the pre-created (finish-free) marker onto the event. */
    arts_add_dependence(event, (arts_guid_t)paramv[3], 0, DB_MODE_RW);
  } else if (role == 1) {
    arts_event_satisfy(event, db);
  } else {
    arts_event_destroy(event);
  }

  iter_counters_t *c = (iter_counters_t *)depv[0].ptr;
  atomic_fetch_add_explicit(&c->completed_workers, 1u, memory_order_release);
}

static void iter_verify_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]);

/* start_iter_edt: sets up one iteration.  Finish-free (created with NULL hint),
 * so the marker EDTs it creates are also finish-free. */
static void start_iter_edt(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int it = (int)paramv[0];

  void *cptr = NULL;
  arts_guid_t ctr_db = arts_db_create(&cptr, sizeof(iter_counters_t), ARTS_DB,
                                      ARTS_DB_PROP_NONE, NULL);
  atomic_init(&((iter_counters_t *)cptr)->completed_workers, 0u);
  arts_db_release(ctr_db, DB_MODE_RW);

  /* IDEM event (over-satisfy is silent). */
  arts_event_hint_t h = ARTS_EVENT_HINT_IDEMPOTENT;
  arts_guid_t ev = arts_event_create(&h);
  if (ev == NULL_GUID) {
    (void)fprintf(stderr, "FAIL [iter=%d]: arts_event_create\n", it);
    arts_abort(1);
  }

  void *dbp = NULL;
  arts_guid_t db =
      arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);

  /* Finish event over all N_WORKERS worker EDTs. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Workers (RW on ctr_db) FIRST, so their exclusive RW holds queue ahead of
   * the verify EDT's RO read.  verify is gated on the workers' finish event;
   * registering its RO before the workers' RW would deadlock (verify waits on
   * the finish, the workers' RW waits on verify's RO). */
  for (int i = 0; i < N_WORKERS; i++) {
    /* The consumer's marker is created here, finish-free; non-consumers carry
     * NULL_GUID. */
    arts_guid_t marker = (i % 3 == 0)
                             ? arts_edt_create(counter_edt, 0, NULL, 1, NULL)
                             : NULL_GUID;
    uint64_t pv[4] = {(uint64_t)ev, (uint64_t)db, (uint64_t)(i % 3),
                      (uint64_t)marker};
    arts_guid_t w = arts_edt_create(worker_edt, 4, pv, 1,
                                    &(arts_edt_hint_t){.finish_event = fe});
    arts_add_dependence(ctr_db, w, 0, DB_MODE_RW);
  }

  /* verify EDT: gated on the finish event (all workers done) + counter DB (RO,
   * registered after the workers' RW so it drains last). */
  uint64_t vpv[1] = {(uint64_t)it};
  arts_guid_t verify = arts_edt_create(iter_verify_edt, 1, vpv, 2, NULL);
  arts_add_dependence(fe, verify, 0, DB_MODE_NULL);
  arts_add_dependence(ctr_db, verify, 1, DB_MODE_RO);
}

static void iter_verify_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int it = (int)paramv[0];
  iter_counters_t *c = (iter_counters_t *)depv[1].ptr;

  unsigned int wc =
      atomic_load_explicit(&c->completed_workers, memory_order_acquire);
  if (wc != N_WORKERS) {
    (void)fprintf(stderr, "FAIL [iter=%d]: workers completed=%u (want %u)\n",
                  it, wc, N_WORKERS);
    arts_abort(1);
  }

  /* Chain next iteration or print final PASS. */
  if (it + 1 < M_ITERS) {
    uint64_t npv[1] = {(uint64_t)(it + 1)};
    arts_edt_create(start_iter_edt, 1, npv, 0, NULL);
  } else {
    printf("event_destroy_race: %d iters with %d-way concurrent S/A/D — PASS\n",
           M_ITERS, N_WORKERS);
    arts_shutdown();
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  uint64_t pv[1] = {0};
  arts_edt_create(start_iter_edt, 1, pv, 0, NULL);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
