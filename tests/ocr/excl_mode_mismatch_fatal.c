/* SPDX-License-Identifier: Apache-2.0
 *
 * excl_mode_mismatch_fatal — cross-protocol binary mismatch detection driver
 * for the EXCL protocol (census 11-lock; dispatcher.c EXCL fatal arms at
 * lines ~224/255/288/373/393/531 — snapshot / ownership / publish wire
 * messages are rejected under EXCL, and EXCL's own REQUEST/GRANT/RELEASE wire
 * is unknown to non-EXCL builds → the receiver's default arm stops the
 * runtime).
 *
 * Like coherence_mode_mismatch.c, this is a STANDALONE driver (not a plain
 * ctest): run_mode_mismatch.sh launches a EXCL-built binary as one rank and a
 * non-EXCL-built binary (VAL HOME/OWNER, WRF_VAL) as the other.  Across
 * the protocol boundary the wire tags diverge:
 *   - The non-EXCL rank receives MSG_DB_EXCL_REQUEST / _GRANT / _RELEASE
 *     (absent from its enum dispatch) → hits the dispatcher default arm →
 *     arts_shutdown + arts_runtime_stop → non-zero exit.
 *   - The EXCL rank receives MSG_DB_SNAPSHOT_* / _OWNERSHIP_* / _PUBLISH*
 *     → its fatal arms log and drop them → the coherence handshake never
 *     completes → the rank stalls and is reaped by the harness timeout (-k →
 *     SIGKILL → non-zero exit).
 * Either way each side exits non-zero within the harness timeout, which is the
 * mismatch-detected signal.
 *
 * To exercise BOTH the RO-snapshot/LOCK_REQUEST divergence AND the
 * RW-ownership/publish/LOCK_RELEASE divergence, rank 0 creates a DB and wires
 * a remote RW writer plus a remote RO reader on rank 1.  Whichever the build's
 * protocol emits first crosses the boundary and triggers the mismatch.
 *
 * When both ranks are the SAME protocol (not a mismatch pairing) the program
 * completes normally and exits 0 — the harness only declares PASS when a
 * MISMATCHED pair yields non-zero, so this file is correct standalone too.
 *
 * No compile-time protocol skip: the source is built in every config dir; the
 * mismatch is created by PAIRING different-config binaries at run time.
 */

#include <stdint.h>
#include <stdio.h>

#include "arts.h"

/* Remote RW writer (rank 1): forces the build's RW-ownership/publish (VAL)
 * or LOCK_REQUEST(RW)/LOCK_RELEASE (EXCL) wire across the rank boundary. */
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  volatile uint64_t *p = (volatile uint64_t *)depv[0].ptr;
  if (p != NULL) {
    *p = *p + 1u;
  }
}

/* Remote RO reader (rank 1): forces the build's RO-snapshot (VAL) or
 * LOCK_REQUEST(RO) (EXCL) wire across the rank boundary, then shuts down. */
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  volatile uint64_t *p = (volatile uint64_t *)depv[0].ptr;
  if (p != NULL) {
    printf("READER: %lu\n", (unsigned long)*p);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    /* The mismatch can only manifest across a rank boundary. */
    arts_printf("SKIP excl_mode_mismatch_fatal: requires 2+ ranks (got %u)\n",
                nranks);
    arts_shutdown();
    return;
  }

  void *addr = NULL;
  arts_guid_t db =
      arts_db_create(&addr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (addr != NULL) {
    *(uint64_t *)addr = 42u;
  }

  /* Remote RW writer on rank 1: RW path (ownership/publish vs EXCL rel). */
  arts_edt_hint_t wh = ARTS_EDT_HINT_DEFAULTS;
  wh.rank = 1;
  arts_guid_t w = arts_edt_create(writer_edt, 0, NULL, 1, &wh);
  arts_add_dependence(db, w, 0, DB_MODE_RW);

  /* Remote RO reader on rank 1: RO path (snapshot vs EXCL req RO). */
  arts_edt_hint_t rh = ARTS_EDT_HINT_DEFAULTS;
  rh.rank = 1;
  arts_guid_t r = arts_edt_create(reader_edt, 0, NULL, 1, &rh);
  arts_add_dependence(db, r, 0, DB_MODE_RO);

  /* Drop the creator RW hold so the remote acquires drive cross-node wire. */
  arts_db_release(db, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
