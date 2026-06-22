/* SPDX-License-Identifier: Apache-2.0
 *
 * T101 — MRMW arts_db_acquire_is_serialized: returns false for EVERY mode.
 *
 * MRMW (DB-DRF) has no ownership round, so nothing is GUID-serialized by the
 * acquire-all engine.  Unlike the ownership protocols (MRNEW/MRSW serialize RW,
 * LOCK serializes both RW and RO), the MRMW predicate must answer false for ALL
 * dep modes — including the placeholder DB_MODE_NULL / raw-value DB_MODE_VAL.
 *
 * This is the MRMW-focused sibling of T068 (acquire_is_serialized.c, which
 * checks the answer across every protocol).  It is compiled once per build but
 * only asserts under the MRMW build; under any other protocol it self-skips
 * cleanly (the predicate's answer differs, so asserting it here would be
 * wrong).
 *
 * pure_unit: links the real symbol out of the per-config static libarts (the
 * defining TU mrmw/mrmw.c is a heavyweight TU, so we link rather than #include)
 * and only calls the pure query function — the runtime is never started.
 */

#include "arts.h" /* arts_db_access_mode_t, DB_MODE_* */

#include <stdbool.h>
#include <stdio.h>

/* Declared in coherence.h; defined per-protocol in the coherence TUs linked
 * from libarts.  Re-declared here to keep the test header-light. */
bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode);

#if !defined(ARTS_PROTOCOL_MRMW)
int main(void) {
  printf("SKIP mrmw_is_serialized: MRMW-only\n");
  return 0;
}
#else
int main(void) {
  /* MRMW: nothing is serialized — false for every mode, including the
   * placeholder / raw-value modes. */
  const arts_db_access_mode_t modes[] = {DB_MODE_NULL, DB_MODE_RO, DB_MODE_RW,
                                         DB_MODE_VAL};
  const char *names[] = {"NULL", "RO", "RW", "VAL"};
  int rc = 0;
  for (unsigned i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
    bool s = arts_db_acquire_is_serialized(modes[i]);
    if (s) {
      (void)fprintf(
          stderr, "FAIL mrmw_is_serialized: mode %s serialized (want false)\n",
          names[i]);
      rc = 1;
    }
  }
  if (rc != 0) {
    return 1;
  }
  printf("PASS mrmw_is_serialized: all modes false (DB-DRF, no ownership)\n");
  return 0;
}
#endif
