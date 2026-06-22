/* SPDX-License-Identifier: Apache-2.0
 *
 * T138 — arts_signal_edt deprecated-alias equivalence.
 *
 * Property under test (arts_signal_edt inline alias)
 * --------------------------------------------------
 * arts_signal_edt is a static-inline backward-compat alias that forwards its
 * arguments verbatim to arts_edt_satisfy_slot (same signature, same argument
 * order).  This test pins that it routes identically: an EDT satisfied entirely
 * through arts_signal_edt receives the same dep data (a DB delivered RO and a
 * raw VAL) and fires exactly once, just as if arts_edt_satisfy_slot were used.
 *
 * Scenario
 * --------
 * Create a 2-dep EDT.  Satisfy slot 0 with a DB (RO) via arts_signal_edt, and
 * slot 1 with a DB_MODE_VAL via arts_signal_edt.  The EDT verifies it received
 * the DB pointer/guid on slot 0 and the value on slot 1, then passes.  A
 * finish-event collector confirms a single fire and shuts down.
 *
 * Config-agnostic single-rank EDT lifecycle.
 */

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define DB_MAGIC 0x1234ABCDu
#define VAL_MAGIC 0x9999u

typedef struct {
  _Atomic unsigned int fired;
  _Atomic unsigned int slot0_ok;
  _Atomic unsigned int slot1_ok;
} tally_t;

#define PV_TALLY 0

/* Target: depv[0] = data DB (RO) via signal, depv[1] = VAL via signal. */
void target(uint32_t pc, const uint64_t *pv, uint32_t dc, arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  tally_t *t = (tally_t *)(void *)pv[1]; /* tally raw pointer via paramv[1] */
  atomic_fetch_add_explicit(&t->fired, 1u, memory_order_relaxed);
  unsigned int *d = (unsigned int *)dv[0].ptr;
  if (dv[0].guid != NULL_GUID && d != NULL && d[0] == DB_MAGIC) {
    atomic_fetch_add_explicit(&t->slot0_ok, 1u, memory_order_relaxed);
  }
  if ((unsigned int)(uintptr_t)dv[1].guid == VAL_MAGIC) {
    atomic_fetch_add_explicit(&t->slot1_ok, 1u, memory_order_relaxed);
  }
}

void collector(uint32_t pc, const uint64_t *pv, uint32_t dc,
               arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  (void)dv;
  tally_t *t = (tally_t *)(void *)pv[1];
  unsigned int f = atomic_load_explicit(&t->fired, memory_order_relaxed);
  unsigned int s0 = atomic_load_explicit(&t->slot0_ok, memory_order_relaxed);
  unsigned int s1 = atomic_load_explicit(&t->slot1_ok, memory_order_relaxed);
  arts_printf("edt_signal_alias: fired=%u slot0_ok=%u slot1_ok=%u\n", f, s0,
              s1);
  if (f == 1u && s0 == 1u && s1 == 1u) {
    arts_printf("PASS edt_signal_alias\n");
  } else {
    arts_printf("FAIL edt_signal_alias\n");
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

  arts_printf("=== edt_signal_alias ===\n");

  /* Tally lives in a node-pinned DB so its raw pointer is stable to pass via
   * paramv (no coherence migration). */
  void *tp = NULL;
  arts_guid_t tally =
      arts_db_create(&tp, sizeof(tally_t), ARTS_DB_PIN, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  tally_t *t = (tally_t *)tp;
  atomic_init(&t->fired, 0u);
  atomic_init(&t->slot0_ok, 0u);
  atomic_init(&t->slot1_ok, 0u);

  /* Data DB delivered on slot 0 via the alias. */
  void *dp = NULL;
  arts_guid_t ddb =
      arts_db_create(&dp, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)dp)[0] = DB_MAGIC;
  arts_db_release(ddb, DB_MODE_RW);

  uint64_t pv[2] = {(uint64_t)tally, (uint64_t)(uintptr_t)t};

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  arts_guid_t e = arts_edt_create(
      target, 2, pv, 2, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});

  /* Route both deps THROUGH the deprecated alias to prove equivalence. */
  arts_signal_edt(e, 0, ddb, DB_MODE_RO, NULL, 0);
  arts_signal_edt(e, 1, (arts_guid_t)VAL_MAGIC, DB_MODE_VAL, NULL, 0);

  arts_guid_t coll =
      arts_edt_create(collector, 2, pv, 1, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe, coll, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
