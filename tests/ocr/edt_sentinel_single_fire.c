/* SPDX-License-Identifier: Apache-2.0
 *
 * T130 — EDT pre-reserved-GUID sentinel single-fire protocol.
 *
 * Property under test
 * -------------------
 * arts_edt_create_core's pre-reserved-GUID path sets depc_needed = depc + 1
 * (the sentinel) BEFORE installing the EDT, replays any out-of-order satisfies
 * queued against the still-RESERVED GUID, then removes the sentinel (-1).  The
 * invariant is: exactly ONE party observes the depc_needed==0 transition and
 * fires the EDT — never zero (lost wakeup) and never two (double dispatch).
 *
 * Three scenarios per run, each driving a distinct branch of the protocol:
 *
 *   A) depc==0, fresh GUID: immediate fire exactly once.
 *
 *   B) pre-reserved GUID, all deps satisfied BEFORE create: the satisfies are
 *      queued on the RESERVED slot (OoO).  The sentinel keeps depc_needed >= 1
 *      across the install+replay, so the sentinel-removal at the end of create
 *      observes 0 and fires.  Must fire exactly once.
 *
 *   C) pre-reserved GUID, last real satisfy lands AFTER create: create's
 *      sentinel removal sees depc_needed > 0 (does not fire); the trailing
 *      satisfy drives it to 0 and fires.  Must fire exactly once.
 *
 * Each fired EDT increments its own per-scenario atomic counter.  A collector
 * EDT (gated on a finish event that all scenarios join) verifies every counter
 * is EXACTLY 1.  A stranded EDT (counter 0) or a double fire (counter 2) is a
 * FAIL; a deadlock is caught by the ctest TIMEOUT.
 *
 * Config-agnostic: pure single-rank EDT lifecycle, no coherence transfer.
 */

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

/* Counter DB: one atomic per scenario fire-count, plus a fail flag. */
typedef struct {
  _Atomic unsigned int fired_a;
  _Atomic unsigned int fired_b;
  _Atomic unsigned int fired_c;
} counters_t;

#define PV_CTR 0 /* paramv[0] = counters DB guid */

static void bump(arts_guid_t ctr_guid, _Atomic unsigned int *which) {
  (void)ctr_guid;
  atomic_fetch_add_explicit(which, 1u, memory_order_relaxed);
}

/* Scenario A body: fired by depc==0 immediate path. depv[0]=counters(RW). */
void fire_a(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  counters_t *c = (counters_t *)depv[0].ptr;
  bump((arts_guid_t)paramv[PV_CTR], &c->fired_a);
}

/* Scenario B body: pre-reserved, deps queued before create. depv[0]=counters
 * (RW), depv[1..2] are the pre-queued VAL slots that drove readiness. */
void fire_b(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  counters_t *c = (counters_t *)depv[0].ptr;
  bump((arts_guid_t)paramv[PV_CTR], &c->fired_b);
}

/* Scenario C body: pre-reserved, last satisfy after create. depv[0]=counters
 * (RW), depv[1] is the trailing VAL slot. */
void fire_c(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  counters_t *c = (counters_t *)depv[0].ptr;
  bump((arts_guid_t)paramv[PV_CTR], &c->fired_c);
}

/* Collector: reads the counters RO after all scenarios drained. */
void collector(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  /* depv[0] = finish event (NULL payload), depv[1] = counters DB (RO). */
  counters_t *c = (counters_t *)depv[1].ptr;
  unsigned int a = atomic_load_explicit(&c->fired_a, memory_order_relaxed);
  unsigned int b = atomic_load_explicit(&c->fired_b, memory_order_relaxed);
  unsigned int cc = atomic_load_explicit(&c->fired_c, memory_order_relaxed);
  arts_printf("edt_sentinel_single_fire: A=%u B=%u C=%u\n", a, b, cc);
  if (a == 1u && b == 1u && cc == 1u) {
    arts_printf("PASS edt_sentinel_single_fire\n");
  } else {
    arts_printf("FAIL edt_sentinel_single_fire: each must fire exactly once\n");
    arts_abort(1);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_sentinel_single_fire ===\n");

  void *cptr = NULL;
  arts_guid_t ctr =
      arts_db_create(&cptr, sizeof(counters_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  counters_t *c = (counters_t *)cptr;
  atomic_init(&c->fired_a, 0u);
  atomic_init(&c->fired_b, 0u);
  atomic_init(&c->fired_c, 0u);
  arts_db_release(ctr, DB_MODE_RW);

  uint64_t pv[1] = {(uint64_t)ctr};

  /* Finish scope: collector waits until A/B/C all drain (their finish-scope
   * INCRs balance their DECRs at completion). */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* --- Scenario A: depc==0 immediate fire (fresh GUID). The counters dep is
   * its only dependency. --- */
  {
    arts_guid_t a = arts_edt_create(
        fire_a, 1, pv, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ctr, a, 0, DB_MODE_RW);
  }

  /* --- Scenario B: pre-reserved GUID with all real deps satisfied BEFORE the
   * EDT is created, so they queue on the RESERVED slot (OoO). depc = 3 (the
   * counters RW + two VAL slots). We satisfy the two VAL slots and the RW dep
   * first, then create the EDT — its sentinel-removal must be the sole firer.
   * --- */
  {
    arts_guid_t b = arts_guid_reserve(ARTS_GUID_EDT, 0);
    /* Pre-queue the two VAL satisfies and the RW dep against the RESERVED b. */
    arts_edt_satisfy_slot(b, 1, NULL_GUID, DB_MODE_VAL);
    arts_edt_satisfy_slot(b, 2, NULL_GUID, DB_MODE_VAL);
    arts_add_dependence(ctr, b, 0, DB_MODE_RW);
    /* Now create with the pre-reserved GUID; depc=3. */
    arts_edt_create(fire_b, 1, pv, 3,
                    &(arts_edt_hint_t){.guid = b, .finish_event = fe});
  }

  /* --- Scenario C: pre-reserved GUID where the LAST real satisfy lands after
   * create. We pre-queue only the RW dep, create the EDT (depc=2), then deliver
   * the trailing VAL satisfy. Create's sentinel removal must see depc_needed>0;
   * the trailing satisfy fires it. --- */
  {
    arts_guid_t cg = arts_guid_reserve(ARTS_GUID_EDT, 0);
    arts_add_dependence(ctr, cg, 0, DB_MODE_RW);
    arts_edt_create(fire_c, 1, pv, 2,
                    &(arts_edt_hint_t){.guid = cg, .finish_event = fe});
    /* Trailing satisfy that drives readiness to 0 after the sentinel removal.
     */
    arts_edt_satisfy_slot(cg, 1, NULL_GUID, DB_MODE_VAL);
  }

  /* Collector gated on the finish scope: created last, depends on fe firing. */
  arts_guid_t coll =
      arts_edt_create(collector, 1, pv, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe, coll, 0, DB_MODE_NULL);
  arts_add_dependence(ctr, coll, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
