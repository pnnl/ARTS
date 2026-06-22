/* SPDX-License-Identifier: Apache-2.0
 *
 * T068 — arts_db_acquire_is_serialized, the per-protocol predicate that tells
 *        the acquire-all engine which dep modes must be acquired in a global
 *        GUID order (so the engine can avoid acquire-cycle deadlock).
 *
 * The function has a different ANSWER per coherence protocol; this single file
 * is compiled once per build config and asserts the answer for whichever
 * protocol macro is defined:
 *   - MRNEW / MRSW (ownership lease, single inter-node writer): RW is
 *     serialized, RO is NOT (RO is a snapshot read, no ordering needed).
 *   - LOCK (blocking RW *and* RO locks): BOTH RW and RO are serialized.
 *   - MRMW (DB-DRF, no ownership round): NOTHING is serialized → false for all.
 *
 * The defining TU per protocol (mrnew/{eager,lazy}.c, mrsw/{eager,lazy}.c,
 * mrmw/mrmw.c, lock/acquire.c) drags in the broader runtime, so this test links
 * the real symbol out of the per-config static libarts (needs_full_build)
 * rather than #including a heavyweight TU — it never starts the runtime, it
 * only calls the pure query function.
 *
 * DB_MODE_NULL / DB_MODE_VAL are placeholder/raw-value modes (never a real DB
 * acquire); the predicate must return false for them under every protocol.
 */

#include "arts.h" /* arts_db_access_mode_t, DB_MODE_* */

#include <stdbool.h>
#include <stdio.h>

/* Declared in coherence.h; defined per-protocol in the coherence TUs linked
 * from libarts.  Re-declared here to keep the test header-light. */
bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode);

int main(void) {
  bool rw = arts_db_acquire_is_serialized(DB_MODE_RW);
  bool ro = arts_db_acquire_is_serialized(DB_MODE_RO);
  bool null_mode = arts_db_acquire_is_serialized(DB_MODE_NULL);
  bool val_mode = arts_db_acquire_is_serialized(DB_MODE_VAL);

  bool exp_rw, exp_ro;
  const char *proto;
#if defined(ARTS_PROTOCOL_LOCK)
  proto = "LOCK";
  exp_rw = true;
  exp_ro = true; /* blocking locks: both serialized */
#elif defined(ARTS_PROTOCOL_MRMW)
  proto = "MRMW";
  exp_rw = false;
  exp_ro = false; /* DB-DRF: nothing serialized */
#elif defined(ARTS_PROTOCOL_MRNEW)
  proto = "MRNEW";
  exp_rw = true;
  exp_ro = false; /* RW serialized, RO snapshot */
#elif defined(ARTS_PROTOCOL_MRSW)
  proto = "MRSW";
  exp_rw = true;
  exp_ro = false; /* RW serialized, RO snapshot */
#else
#error "no ARTS_PROTOCOL_* defined"
#endif

  int rc = 0;
  if (rw != exp_rw) {
    (void)fprintf(stderr, "FAIL acquire_is_serialized[%s]: RW got %d want %d\n",
                  proto, rw, exp_rw);
    rc = 1;
  }
  if (ro != exp_ro) {
    (void)fprintf(stderr, "FAIL acquire_is_serialized[%s]: RO got %d want %d\n",
                  proto, ro, exp_ro);
    rc = 1;
  }
  /* Placeholder / raw-value modes are never serialized under any protocol. */
  if (null_mode) {
    (void)fprintf(stderr,
                  "FAIL acquire_is_serialized[%s]: NULL mode serialized\n",
                  proto);
    rc = 1;
  }
  if (val_mode) {
    (void)fprintf(
        stderr, "FAIL acquire_is_serialized[%s]: VAL mode serialized\n", proto);
    rc = 1;
  }
  if (rc != 0) {
    return 1;
  }
  printf("PASS acquire_is_serialized[%s]: RW=%d RO=%d (NULL/VAL=false)\n",
         proto, rw, ro);
  return 0;
}
