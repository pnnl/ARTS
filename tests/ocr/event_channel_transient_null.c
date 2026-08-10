/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_channel_transient_null — C15 / T153.
 *
 * Target: try_drain_channel's MPSC pop spin-on-transient-NULL (event.c).
 *
 * The CHANNEL drainer pops one node from each of data_queue and dep_queue per
 * fire.  A producer first links its node (mpsc_push) and only THEN bumps the
 * matching counter (nb_sat / nb_deps).  A concurrent producer can leave the
 * popped node's `next` momentarily NULL (the link store is in flight).  The
 * drainer MUST spin-retry on that transient NULL — never drop the
 * already-popped partner — or the counter/queue pair desyncs and a consumer is
 * stranded forever (lost-wakeup regression).
 *
 * Strategy: deliberately interleave MANY satisfiers and MANY add-deppers on the
 * SAME channel event so the two MPSC queues are simultaneously contended and
 * the transient-NULL window opens repeatedly.  Each satisfy carries a unique
 * data DB; each add-dep wires a counter_edt that records the GUID it received
 * and decrements a storm-wide LATCH.  CHANNEL pairs satisfies with deps in
 * arrival order (FIFO), so when exactly K satisfies meet K deps, EXACTLY K
 * counter_edts must fire — none dropped — and every delivered GUID must be one
 * of the K satisfied GUIDs (a matched pair, never NULL/garbage).
 *
 * PASS criterion: across M iters x K pairs, all M*K deliveries land (the
 * storm-wide LATCH fires => no node dropped, no consumer stranded) AND every
 * delivered GUID is a real satisfied data GUID.  A dropped node strands the
 * LATCH and the ctest TIMEOUT reaps it (no in-test spin).
 *
 * No EDT busy-waits.  main_edt sets everything up and terminates, releasing the
 * creator-hold on the shared state DB so the RW counter_edts can acquire.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define M_ITERS 24
#define K_PAIRS 64
#define TOTAL (M_ITERS * K_PAIRS)

/* State DB (uint64_t elements):
 *   [0]                  — delivered_count (atomic)
 *   [1]                  — bad_guid_count  (atomic; delivered GUID not in set)
 *   [2 .. 1+TOTAL]       — satisfied[] : the TOTAL satisfied data GUIDs
 */
#define ST_DELIVERED 0
#define ST_BAD 1
#define ST_SAT_OFF 2
#define ST_NELEMS (ST_SAT_OFF + TOTAL)

/* counter_edt — fired by one CHANNEL pairing.
 * paramv: [state_db, latch, it].
 * depv[0] = channel data (RW), depv[1] = state DB (RW). */
static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[1];
  uint64_t it = paramv[2];
  uint64_t *state = (uint64_t *)depv[1].ptr;
  arts_guid_t received = depv[0].guid;

  /* The received GUID must be one of the K satisfied GUIDs for this iter. */
  int found = 0;
  for (int g = 0; g < K_PAIRS; g++) {
    _Atomic uint64_t *s =
        (_Atomic uint64_t *)&state[ST_SAT_OFF + it * K_PAIRS + g];
    if ((arts_guid_t)atomic_load_explicit(s, memory_order_acquire) ==
        received) {
      found = 1;
      break;
    }
  }
  if (!found) {
    _Atomic uint64_t *bad = (_Atomic uint64_t *)&state[ST_BAD];
    atomic_fetch_add_explicit(bad, 1u, memory_order_acq_rel);
  }
  _Atomic uint64_t *cnt = (_Atomic uint64_t *)&state[ST_DELIVERED];
  atomic_fetch_add_explicit(cnt, 1u, memory_order_acq_rel);

  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* satisfier_edt — push one satisfy onto the channel.
 * paramv: [event, data]. */
static void satisfier_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_event_satisfy((arts_guid_t)paramv[0], (arts_guid_t)paramv[1]);
}

/* depper_edt — register one addDep (counter_edt) onto the channel.
 * paramv: [event, state_db, latch, it]. */
static void depper_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t event = (arts_guid_t)paramv[0];
  uint64_t cpv[3] = {paramv[1], paramv[2], paramv[3]};
  arts_guid_t edt = arts_edt_create(counter_edt, 3, cpv, 2, NULL);
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
  uint64_t bad = atomic_load_explicit((_Atomic uint64_t *)&state[ST_BAD],
                                      memory_order_acquire);
  if (got != (uint64_t)TOTAL) {
    (void)fprintf(stderr, "FAIL: delivered=%llu want %d (dropped node)\n",
                  (unsigned long long)got, TOTAL);
    arts_abort(1);
  }
  if (bad != 0) {
    (void)fprintf(stderr,
                  "FAIL: %llu deliveries carried a non-satisfied GUID\n",
                  (unsigned long long)bad);
    arts_abort(1);
  }
  printf("event_channel_transient_null: %d iters x %d pairs = %d matched "
         "deliveries, no drop — PASS\n",
         M_ITERS, K_PAIRS, TOTAL);
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
  if (state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: state_db NULL_GUID\n");
    arts_abort(1);
  }
  memset(state, 0, ST_NELEMS * sizeof(uint64_t));

  arts_event_hint_t lh = ARTS_EVENT_HINT_LATCH(TOTAL);
  lh.rank = 0;
  arts_guid_t latch = arts_event_create(&lh);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: latch NULL_GUID\n");
    arts_abort(1);
  }

  for (int it = 0; it < M_ITERS; it++) {
    arts_event_hint_t ch = ARTS_EVENT_HINT_CHANNEL;
    arts_guid_t ev = arts_event_create(&ch);
    if (ev == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [it=%d]: CHANNEL create\n", it);
      arts_abort(1);
    }

    /* Pre-record the K data GUIDs this iteration will satisfy, then spawn K
     * satisfier EDTs and K depper EDTs all depc=0 so the worker pool runs
     * them concurrently — opening the transient-NULL window on both queues. */
    for (int g = 0; g < K_PAIRS; g++) {
      void *dbp = NULL;
      arts_guid_t dg = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                      ARTS_DB_PROP_NONE, NULL);
      state[ST_SAT_OFF + it * K_PAIRS + g] = (uint64_t)dg;
    }
    for (int g = 0; g < K_PAIRS; g++) {
      uint64_t spv[2] = {(uint64_t)ev, state[ST_SAT_OFF + it * K_PAIRS + g]};
      arts_edt_create(satisfier_edt, 2, spv, 0, NULL);
      uint64_t dpv[4] = {(uint64_t)ev, (uint64_t)state_db, (uint64_t)latch,
                         (uint64_t)it};
      arts_edt_create(depper_edt, 4, dpv, 0, NULL);
    }
  }

  /* Publish the state DB to the counter EDTs by releasing the creator hold. */
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
