/* SPDX-License-Identifier: Apache-2.0
 *
 * T101 — WRF_RCU arts_db_acquire_is_serialized: returns false for EVERY mode.
 *
 * WRF_RCU (DB-WRF) has no ownership round, so nothing is GUID-serialized by the
 * acquire-all engine.  Unlike the ownership protocols (RCU serialize RW,
 * RWLOCK serializes both RW and RO), the WRF_RCU predicate must answer false for ALL
 * dep modes — including the placeholder DB_MODE_NULL / raw-value DB_MODE_VAL.
 *
 * This is the WRF_RCU-focused sibling of T068 (acquire_is_serialized.c, which
 * checks the answer across every protocol).  It is compiled once per build but
 * only asserts under the WRF_RCU build; under any other protocol it self-skips
 * cleanly (the predicate's answer differs, so asserting it here would be
 * wrong).
 *
 * pure_unit: links the real symbol out of the per-config static libarts (the
 * defining TU wrf_rcu/wrf_rcu.c is a heavyweight TU, so we link rather than #include)
 * and only calls the pure query function — the runtime is never started.
 */

#include "arts.h" /* arts_db_access_mode_t, DB_MODE_* */

#include <stdbool.h>
#include <stdio.h>

/* Declared in coherence.h; defined per-protocol in the coherence TUs linked
 * from libarts.  Re-declared here to keep the test header-light. */
bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode);

#if !defined(ARTS_PROTOCOL_WRF_RCU)
int main(void) {
  printf("SKIP wrf_rcu_is_serialized: WRF_RCU-only\n");
  return 0;
}
#else
int main(void) {
  /* WRF_RCU: nothing is serialized — false for every mode, including the
   * placeholder / raw-value modes. */
  const arts_db_access_mode_t modes[] = {DB_MODE_NULL, DB_MODE_RO, DB_MODE_RW,
                                         DB_MODE_VAL};
  const char *names[] = {"NULL", "RO", "RW", "VAL"};
  int rc = 0;
  for (unsigned i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
    bool s = arts_db_acquire_is_serialized(modes[i]);
    if (s) {
      (void)fprintf(
          stderr, "FAIL wrf_rcu_is_serialized: mode %s serialized (want false)\n",
          names[i]);
      rc = 1;
    }
  }
  if (rc != 0) {
    return 1;
  }
  printf("PASS wrf_rcu_is_serialized: all modes false (DB-WRF, no ownership)\n");
  return 0;
}
#endif
