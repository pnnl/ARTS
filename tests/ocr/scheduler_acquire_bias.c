/* SPDX-License-Identifier: Apache-2.0
 *
 * scheduler_acquire_bias — runtime_single stress for the +1 acquire bias in
 * arts_handle_ready_edt and the release-before-satisfy / outstanding-decrement
 * ordering in arts_run_edt (census 17 §1 arts_handle_ready_edt + arts_run_edt).
 *
 * Property 1 — the +1 bias (arts_handle_ready_edt: rw_cursor=0,
 * acquire_remaining=1, then arts_db_acquire_all adds the real dep count).  The
 * bias prevents a premature schedule when every per-dep decrement races to
 * completion BEFORE the seed loop finishes.  An off-by-one would either
 * double-schedule (UAF, EDT body runs twice) or never-schedule (EDT parked
 * forever → hang).  We stress it with many EDTs each carrying MANY DB deps,
 * all of whose owners resolve concurrently, and require each EDT body to run
 * EXACTLY ONCE.
 *
 * Property 2 — release-before-satisfy (arts_run_edt step 5 before step 6).
 * A producer EDT writes a sentinel into an output DB, then arts_run_edt
 * releases the DB and only AFTER that satisfies the producer's output_event.
 * A consumer wired on that output_event + the output DB (RO) must observe the
 * written sentinel, never the pre-write value.  A satisfy issued before the
 * release would let the consumer acquire stale data → mismatch.
 *
 * Construction (single rank, ARTS_DB):
 *   - MANY multi-dep EDTs: each acquires DEPS_PER distinct RW DBs + the shared
 *     state DB RW; depc=0-fired so all per-dep resolutions race the seed.  Each
 *     records a run in a per-EDT bitset.
 *   - MANY producer/consumer pairs: producer has an output_event; its body
 *     stamps SENTINEL into its output DB and registers the DB as the result;
 *     the consumer (wired output_event slot 0 + output DB RO slot 1) checks the
 *     value equals SENTINEL.  Each consumer records its result.
 *
 * Correctness (black-box): run count must equal the EDT count, double-run count
 * must be 0, every per-EDT bitset cell exactly 1, every consumer must see
 * SENTINEL.  A lost/never-scheduled EDT stalls the all-done LATCH → ctest
 * TIMEOUT (no in-test watchdog, no spin); a double-schedule trips the
 * double-run check; an ordering violation trips the SENTINEL check.
 *
 * Runs in every coherence configuration.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define MULTI_EDTS 48 /* EDTs exercising the multi-dep +1 bias race */
#define DEPS_PER 6    /* RW DBs each multi-dep EDT acquires (+ state) */
#define PAIRS 32      /* producer/consumer release-before-satisfy pairs */
#define SENTINEL 0xC0FFEEull
#define TOTAL_RUNNERS (MULTI_EDTS) /* EDTs tracked in the run bitset */
/* LATCH counts: every multi EDT + every consumer drops the latch once. */
#define LATCH_TOTAL (MULTI_EDTS + PAIRS)

/* State DB layout (uint64_t):
 *   [0]                     — run count (atomic; must == MULTI_EDTS)
 *   [1]                     — double-run count (atomic; must == 0)
 *   [2]                     — consumer-mismatch count (atomic; must == 0)
 *   [3]                     — consumers-seen count (atomic; must == PAIRS)
 *   [4 .. 4+MULTI_EDTS-1]   — per-multi-EDT run bitset */
#define ST_RUNS 0
#define ST_DOUBLE 1
#define ST_MISMATCH 2
#define ST_CONSUMED 3
#define ST_BITS 4
#define ST_NELEMS (ST_BITS + MULTI_EDTS)

/* Multi-dep EDT: depv[0..DEPS_PER-1] = distinct RW DBs, depv[DEPS_PER] = state
 * DB (RW).  paramv: [state_db, my_index, latch]. */
static void multi_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned idx = (unsigned)paramv[1];
  arts_guid_t latch = (arts_guid_t)paramv[2];
  uint64_t *state = (uint64_t *)depv[DEPS_PER].ptr;
  /* Touch every acquired dep so a not-yet-resolved (NULL) dep would surface. */
  for (unsigned d = 0; d < DEPS_PER; d++) {
    uint64_t *cell = (uint64_t *)depv[d].ptr;
    if (cell != NULL) {
      cell[0] = (uint64_t)idx;
    }
  }
  if (state != NULL) {
    uint64_t prev = atomic_fetch_add_explicit(
        (_Atomic uint64_t *)&state[ST_BITS + idx], 1u, memory_order_acq_rel);
    if (prev != 0u) {
      atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_DOUBLE], 1u,
                                memory_order_acq_rel);
    }
    atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_RUNS], 1u,
                              memory_order_acq_rel);
  }
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* Producer EDT: stamp SENTINEL into its output DB (depv[0] RW) and register it
 * as the result.  arts_run_edt releases the DB BEFORE satisfying output_event.
 * paramv: [out_db]. */
static void producer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t out_db = (arts_guid_t)paramv[0];
  uint64_t *out = (uint64_t *)depv[0].ptr;
  if (out != NULL) {
    out[0] = SENTINEL;
  }
  arts_edt_set_result(out_db);
}

/* Consumer EDT: wired on the producer's output_event (slot 0) + output DB RO
 * (slot 1).  Must observe SENTINEL (release happened before satisfy).
 * paramv: [state_db, latch]. */
static void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[1];
  uint64_t *state = (uint64_t *)depv[1].ptr; /* depv[1] is state_db (RW) */
  const uint64_t *out = (const uint64_t *)depv[2].ptr; /* output DB (RO) */
  if (state != NULL) {
    if (out == NULL || out[0] != SENTINEL) {
      atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_MISMATCH], 1u,
                                memory_order_acq_rel);
    }
    atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_CONSUMED], 1u,
                              memory_order_acq_rel);
  }
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* verify_edt — bound to all-done LATCH (slot 0) + state DB RO (slot 1). */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *state = (uint64_t *)depv[1].ptr;
  uint64_t runs = atomic_load_explicit((_Atomic uint64_t *)&state[ST_RUNS],
                                       memory_order_acquire);
  uint64_t dbl = atomic_load_explicit((_Atomic uint64_t *)&state[ST_DOUBLE],
                                      memory_order_acquire);
  uint64_t mism = atomic_load_explicit((_Atomic uint64_t *)&state[ST_MISMATCH],
                                       memory_order_acquire);
  uint64_t cons = atomic_load_explicit((_Atomic uint64_t *)&state[ST_CONSUMED],
                                       memory_order_acquire);
  if (dbl != 0u) {
    (void)fprintf(stderr,
                  "FAIL: %llu EDT(s) double-scheduled — +1 bias off-by-one\n",
                  (unsigned long long)dbl);
    arts_abort(1);
  }
  if (runs != (uint64_t)MULTI_EDTS) {
    (void)fprintf(stderr,
                  "FAIL: %llu multi-EDTs ran (want %d) — never-scheduled\n",
                  (unsigned long long)runs, MULTI_EDTS);
    arts_abort(1);
  }
  for (unsigned i = 0; i < MULTI_EDTS; i++) {
    uint64_t c = atomic_load_explicit((_Atomic uint64_t *)&state[ST_BITS + i],
                                      memory_order_acquire);
    if (c != 1u) {
      (void)fprintf(stderr, "FAIL: multi-EDT %u ran %llu times (want 1)\n", i,
                    (unsigned long long)c);
      arts_abort(1);
    }
  }
  if (cons != (uint64_t)PAIRS) {
    (void)fprintf(stderr, "FAIL: %llu consumers ran (want %d)\n",
                  (unsigned long long)cons, PAIRS);
    arts_abort(1);
  }
  if (mism != 0u) {
    (void)fprintf(stderr,
                  "FAIL: %llu consumer(s) saw stale data — satisfy preceded "
                  "release\n",
                  (unsigned long long)mism);
    arts_abort(1);
  }
  printf("scheduler_acquire_bias: %d multi-dep EDTs (run-once) + %d "
         "release-before-satisfy pairs — PASS\n",
         MULTI_EDTS, PAIRS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== scheduler_acquire_bias (multi=%d x %d deps, pairs=%d) ===\n",
              MULTI_EDTS, DEPS_PER, PAIRS);

  void *state_raw = NULL;
  arts_guid_t state_db =
      arts_db_create(&state_raw, ST_NELEMS * sizeof(uint64_t), ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);
  if (state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: state_db create NULL_GUID\n");
    arts_abort(1);
  }
  memset(state_raw, 0, ST_NELEMS * sizeof(uint64_t));
  arts_db_release(state_db, DB_MODE_RW);

  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(LATCH_TOTAL);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: LATCH create NULL_GUID\n");
    arts_abort(1);
  }

  /* Multi-dep EDTs: many DB deps each, all owners resolve concurrently so the
   * per-dep decrements race the +1-biased seed loop. */
  for (int i = 0; i < MULTI_EDTS; i++) {
    uint64_t pv[3] = {(uint64_t)state_db, (uint64_t)i, (uint64_t)latch};
    arts_guid_t e = arts_edt_create(multi_edt, 3, pv, DEPS_PER + 1, NULL);
    for (int d = 0; d < DEPS_PER; d++) {
      void *dp = NULL;
      arts_guid_t db = arts_db_create(&dp, sizeof(uint64_t), ARTS_DB,
                                      ARTS_DB_PROP_NONE, NULL);
      if (db == NULL_GUID) {
        (void)fprintf(stderr, "FAIL [multi=%d dep=%d]: db NULL_GUID\n", i, d);
        arts_abort(1);
      }
      *(uint64_t *)dp = 0u;
      arts_db_release(db, DB_MODE_RW);
      arts_add_dependence(db, e, (uint32_t)d, DB_MODE_RW);
    }
    arts_add_dependence(state_db, e, (uint32_t)DEPS_PER, DB_MODE_RW);
  }

  /* Producer/consumer pairs exercising release-before-satisfy ordering. */
  for (int p = 0; p < PAIRS; p++) {
    void *op = NULL;
    arts_guid_t out_db =
        arts_db_create(&op, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
    if (out_db == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [pair=%d]: out_db NULL_GUID\n", p);
      arts_abort(1);
    }
    *(uint64_t *)op = 0u; /* pre-write value — must NOT be seen by consumer */
    arts_db_release(out_db, DB_MODE_RW);

    arts_event_hint_t oe_hint = ARTS_EVENT_HINT_ONCE;
    arts_guid_t oe = arts_event_create(&oe_hint);
    if (oe == NULL_GUID) {
      (void)fprintf(stderr, "FAIL [pair=%d]: output_event NULL_GUID\n", p);
      arts_abort(1);
    }

    uint64_t prod_pv[1] = {(uint64_t)out_db};
    arts_guid_t prod = arts_edt_create(producer_edt, 1, prod_pv, 1,
                                       &(arts_edt_hint_t){.output_event = oe});
    arts_add_dependence(out_db, prod, 0, DB_MODE_RW);

    uint64_t cons_pv[2] = {(uint64_t)state_db, (uint64_t)latch};
    arts_guid_t cons = arts_edt_create(consumer_edt, 2, cons_pv, 3, NULL);
    arts_add_dependence(oe, cons, 0, DB_MODE_NULL);     /* wait for satisfy */
    arts_add_dependence(state_db, cons, 1, DB_MODE_RW); /* state */
    arts_add_dependence(out_db, cons, 2, DB_MODE_RO);   /* read written value */
  }

  arts_guid_t v =
      arts_edt_create(verify_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);
  arts_add_dependence(state_db, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
