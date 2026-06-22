/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_finish_latch_rearm — C15 / T157.
 *
 * Target: simple.latch INCR (Mechanism A auto-chain) racing the child DECR —
 * the re-arm-after-fire edge ordering on a finish latch (event.c).
 *
 * A finish event is a simple latch=1 (the creator-token).  Mechanism A: when a
 * child finish event F_child is created under an ambient finish F1, the runtime
 * INCRs F1 and wires F_child->DECR F1.  F1 must not drain until every INCR has
 * a matching DECR.  The race of interest: many child finish scopes are created
 * (INCR F1) concurrently while earlier children complete (DECR F1).  If an
 * INCR were ever ordered after the matching child population so the latch
 * transiently hit 0, F1 would fire-and-not-re-fire and the successor would run
 * before all work completed (or never).
 *
 * Design: ORCHS orchestrator EDTs run under outer finish F1.  Each, while
 * running under F1, creates its own child finish F_child (auto-chained: INCR
 * F1) and spawns LEAVES leaf EDTs joined to F_child; each leaf marks a unique
 * slot. F1's creator-token is released when main_edt completes.  Because every
 * child INCR happens-before the child's own leaves can complete (the
 * orchestrator creates the child and its leaves before returning, and the
 * child's DECR fires only after its leaves finish), F1 drains exactly once
 * after ALL leaves ran.
 *
 * The successor (dep on F1) asserts: (a) every leaf slot is set (no premature
 * drain), and (b) it ran exactly once (re-arm would either deadlock — caught by
 * ctest TIMEOUT — or double-fire — caught by the fire-count slot).
 *
 * PASS criterion: all ORCHS*LEAVES slots set, successor fired once.  No EDT
 * busy-waits; a stranded F1 is reaped by the ctest TIMEOUT.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#define ORCHS 8
#define LEAVES 8
#define N_SLOTS (ORCHS * LEAVES)

/* leaf_edt: mark slot. depv[0] = counter DB (RW). paramv[0] = slot. */
static void leaf_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  _Atomic int *c = (_Atomic int *)depv[0].ptr;
  if (c == NULL) {
    (void)fprintf(stderr, "FAIL: leaf NULL counter DB\n");
    arts_abort(1);
  }
  atomic_store_explicit(&c[(int)paramv[0]], 1, memory_order_release);
}

/* orchestrator: runs under F1.  Creates a child finish (auto-chain INCR F1) and
 * LEAVES leaves joined to it.  depv[0] = counter DB (RW). paramv[0] = base. */
static void orchestrator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t cdb = depv[0].guid;
  uint64_t base = paramv[0];

  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t fc = arts_event_create(&fh); /* INCRs ambient F1 */
  if (fc == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: child finish create\n");
    arts_abort(1);
  }
  arts_edt_hint_t ih = ARTS_EDT_HINT_DEFAULTS;
  ih.finish_event = fc;
  for (int i = 0; i < LEAVES; i++) {
    uint64_t slot = base + (uint64_t)i;
    arts_guid_t l = arts_edt_create(leaf_edt, 1, &slot, 1, &ih);
    arts_add_dependence(cdb, l, 0, DB_MODE_RW);
  }
  /* No wait: orchestrator completion releases fc's creator-token; fc drains
   * when its leaves finish; fc's fire DECRs F1. */
}

/* successor: dep[0] = F1 (NULL), dep[1] = counter DB (RO), dep[2] = fire DB
 * (RW, counts successor fires). */
static void successor(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic int *c = (_Atomic int *)depv[1].ptr;
  _Atomic int *fire = (_Atomic int *)depv[2].ptr;
  if (c == NULL || fire == NULL) {
    (void)fprintf(stderr, "FAIL: successor NULL DB\n");
    arts_abort(1);
  }
  int prevfire = atomic_fetch_add_explicit(fire, 1, memory_order_acq_rel);
  if (prevfire != 0) {
    (void)fprintf(stderr, "FAIL: successor fired %d times (latch re-armed)\n",
                  prevfire + 1);
    arts_abort(1);
  }
  int set = 0;
  for (int i = 0; i < N_SLOTS; i++) {
    set += atomic_load_explicit(&c[i], memory_order_acquire);
  }
  if (set != N_SLOTS) {
    (void)fprintf(stderr,
                  "FAIL: %d/%d leaf slots set when F1 drained (premature "
                  "drain / re-arm)\n",
                  set, N_SLOTS);
    arts_abort(1);
  }
  printf("event_finish_latch_rearm: %d orchestrators x %d leaves all drained "
         "before single F1 successor — PASS\n",
         ORCHS, LEAVES);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  _Atomic int *c = NULL;
  arts_guid_t cdb = arts_db_create((void **)&c, sizeof(_Atomic int) * N_SLOTS,
                                   ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  for (int i = 0; i < N_SLOTS; i++) {
    atomic_init(&c[i], 0);
  }
  arts_db_release(cdb, DB_MODE_RW);

  _Atomic int *fire = NULL;
  arts_guid_t fdb = arts_db_create((void **)&fire, sizeof(_Atomic int), ARTS_DB,
                                   ARTS_DB_PROP_NONE, NULL);
  atomic_init(fire, 0);
  arts_db_release(fdb, DB_MODE_RW);

  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t f1 = arts_event_create(&fh);

  arts_edt_hint_t oh = ARTS_EDT_HINT_DEFAULTS;
  oh.finish_event = f1;
  for (int o = 0; o < ORCHS; o++) {
    uint64_t base = (uint64_t)(o * LEAVES);
    arts_guid_t orch = arts_edt_create(orchestrator, 1, &base, 1, &oh);
    arts_add_dependence(cdb, orch, 0, DB_MODE_RW);
  }

  arts_guid_t s = arts_edt_create(successor, 0, NULL, 3, NULL);
  arts_add_dependence(f1, s, 0, DB_MODE_NULL);
  arts_add_dependence(cdb, s, 1, DB_MODE_RO);
  arts_add_dependence(fdb, s, 2, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
