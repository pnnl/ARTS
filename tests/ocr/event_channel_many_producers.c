/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_channel_many_producers — C15 / T154.
 *
 * Target: try_drain_channel + nb_sat / nb_deps counters under MANY concurrent
 * producers on BOTH MPSC queues simultaneously (event.c).
 *
 * Distinct from event_channel_fifo (single producer / single consumer) and from
 * event_channel_transient_null (balanced K:K pairs): here a SINGLE channel
 * event is hammered by a large pool of satisfier EDTs and add-dep EDTs that
 * each fire MULTIPLE pushes in a tight burst, so nb_sat and nb_deps are bumped
 * from many threads at once and the drainer must pair every satisfy with
 * exactly one dep.  The point is the matched-pair counter discipline: a counter
 * `--` without a matching `++` would wrap nb_* to UINT32_MAX and spin the
 * drainer endlessly; a dropped node leaves a consumer stranded.
 *
 * The total number of satisfies and the total number of deps are made EQUAL
 * (P_SAT*B == P_DEP*B), so all TOTAL pairings must resolve and all TOTAL
 * counter_edts must fire.  No payload-ordering assertion is made (FIFO is
 * covered by event_channel_fifo); the invariant here is "no dropped node, no
 * counter underflow" — verified by a storm-wide LATCH that only fires when
 * every pairing delivered.
 *
 * PASS criterion: all TOTAL counter_edts fire (LATCH drains) — stranded
 * consumer => ctest TIMEOUT.  No EDT busy-waits.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define P_SAT 8               /* satisfier EDTs */
#define P_DEP 8               /* depper EDTs    */
#define BURST 64              /* pushes per producer EDT */
#define TOTAL (P_SAT * BURST) /* == P_DEP * BURST */

#define ST_DELIVERED 0
#define ST_NELEMS 1

static void counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  uint64_t *state = (uint64_t *)depv[1].ptr;
  atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_DELIVERED], 1u,
                            memory_order_acq_rel);
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* satisfier_edt: BURST satisfies in a tight burst.
 * paramv: [event, data]. */
static void satisfier_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t ev = (arts_guid_t)paramv[0];
  arts_guid_t data = (arts_guid_t)paramv[1];
  for (int i = 0; i < BURST; i++) {
    arts_event_satisfy(ev, data);
  }
}

/* depper_edt: BURST addDeps in a tight burst.
 * paramv: [event, state_db, latch]. */
static void depper_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t ev = (arts_guid_t)paramv[0];
  arts_guid_t state_db = (arts_guid_t)paramv[1];
  arts_guid_t latch = (arts_guid_t)paramv[2];
  for (int i = 0; i < BURST; i++) {
    uint64_t cpv[1] = {(uint64_t)latch};
    arts_guid_t edt = arts_edt_create(counter_edt, 1, cpv, 2, NULL);
    arts_add_dependence(ev, edt, 0, DB_MODE_RW);
    arts_add_dependence(state_db, edt, 1, DB_MODE_RW);
  }
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
  printf("event_channel_many_producers: %d satisfies paired with %d deps = %d "
         "deliveries, no drop/underflow — PASS\n",
         TOTAL, TOTAL, TOTAL);
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

  arts_event_hint_t ch = ARTS_EVENT_HINT_CHANNEL;
  arts_guid_t ev = arts_event_create(&ch);
  if (ev == NULL_GUID || latch == NULL_GUID || state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: create returned NULL_GUID\n");
    arts_abort(1);
  }

  /* Each satisfier reuses one data DB for its whole burst (payload identity
   * doesn't matter here — only the pairing count does). */
  for (int p = 0; p < P_SAT; p++) {
    void *dbp = NULL;
    arts_guid_t dg = arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    uint64_t spv[2] = {(uint64_t)ev, (uint64_t)dg};
    arts_edt_create(satisfier_edt, 2, spv, 0, NULL);
  }
  for (int p = 0; p < P_DEP; p++) {
    uint64_t dpv[3] = {(uint64_t)ev, (uint64_t)state_db, (uint64_t)latch};
    arts_edt_create(depper_edt, 3, dpv, 0, NULL);
  }

  arts_db_release(state_db, DB_MODE_RW);

  arts_guid_t v = arts_edt_create(verify_edt, 0, NULL, 2, NULL);
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);
  arts_add_dependence(state_db, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
