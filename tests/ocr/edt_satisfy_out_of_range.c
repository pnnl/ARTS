/* SPDX-License-Identifier: Apache-2.0
 *
 * T131 — EDT out-of-range slot satisfy drives a premature fire (exposes B074).
 *
 * Bug under test (B074, edt_apply_satisfy)
 * ----------------------------------------
 * edt_apply_satisfy guards only the depv WRITE with `slot < edt->depc`; the
 * `arts_atomic_sub(&depc_needed, 1)` and the fire decision run UNCONDITIONALLY
 * afterward.  A satisfy delivered to an out-of-range slot therefore still
 * counts toward readiness and can fire the EDT before a real, in-range slot has
 * been written — a silent premature fire.
 *
 * Scenario
 * --------
 * Create an EDT `doomed` with depc == 2:
 *   slot 0 — a real DB (db0) delivered RW,
 *   slot 1 — a real DB (db1) delivered RW, satisfied LAST.
 * Sequence: satisfy slot 0, then deliver an out-of-range satisfy to slot 5
 * (>= depc), then satisfy slot 1.
 *   - Buggy runtime: the out-of-range satisfy's decrement (plus slot 0's)
 *     drives depc_needed to 0 and fires `doomed` BEFORE slot 1 is satisfied —
 *     the body sees slot 1 unwritten (guid == NULL_GUID / ptr == NULL) and
 *     prints FAIL + abort(1).
 *   - Correct runtime: the out-of-range satisfy is ignored (it is not one of
 *     the EDT's dependences), so `doomed` fires only once BOTH real slots are
 *     satisfied — the body sees both written and prints PASS + shutdown.
 * Either way `doomed` eventually fires, so the test terminates promptly (no
 * deadlock); the PASS/FAIL token discriminates fixed vs buggy.
 *
 * Config-agnostic single-rank EDT lifecycle.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

/* depv[0]=db0(RW), depv[1]=db1(RW).  Fires prematurely (slot 1 unwritten) only
 * if an out-of-range satisfy was miscounted toward readiness. */
void doomed(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  if (depv[1].guid == NULL_GUID || depv[1].ptr == NULL) {
    arts_printf(
        "FAIL edt_satisfy_out_of_range: premature fire — slot 1 "
        "unwritten (B074: out-of-range slot decremented depc_needed)\n");
    arts_abort(1);
  }
  arts_printf("PASS edt_satisfy_out_of_range: out-of-range satisfy ignored, "
              "fired only after both real slots written\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_satisfy_out_of_range ===\n");

  /* Two real DBs for slots 0 and 1. */
  void *p0 = NULL;
  arts_guid_t db0 =
      arts_db_create(&p0, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)p0)[0] = 0xABCD1234u;
  arts_db_release(db0, DB_MODE_RW);

  void *p1 = NULL;
  arts_guid_t db1 =
      arts_db_create(&p1, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)p1)[0] = 0x5678u;
  arts_db_release(db1, DB_MODE_RW);

  arts_guid_t d =
      arts_edt_create(doomed, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(db0, d, 0, DB_MODE_RW); /* slot 0 satisfies now */
  arts_add_dependence(db1, d, 1, DB_MODE_RW); /* slot 1 satisfies last */

  /* Out-of-range satisfy: slot 5 >= depc (2).  A correct runtime ignores it; a
   * buggy one decrements depc_needed and, with slot 0 already in, fires
   * `doomed` before slot 1 (db1) lands. */
  arts_edt_satisfy_slot(d, 5, NULL_GUID, DB_MODE_VAL, NULL, 0);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
