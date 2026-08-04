/* SPDX-License-Identifier: Apache-2.0
 *
 * dispatcher_mode_mismatch_matrix — complete cross-protocol binary-mismatch
 * matrix driver (census 19-dispatcher §5/§7; dispatcher.c fatal arms at
 * ~224/255/288/373/393/531 and the `default` arm at ~631).
 *
 * The dispatcher rejects, with a fatal ARTS_ERROR, any coherence wire message
 * that does not exist in this build's protocol — the diagnostic for a cluster
 * accidentally assembled from binaries built with mismatched
 * ARTS_COHERENCE_PROTOCOL / ARTS_WRITE_POLICY / ARTS_RELEASE_POLICY.  The existing
 * coherence_mode_mismatch.c only paired RCU_HOME / RCU_OWNER / WRF_VAL, and
 * excl_mode_mismatch_fatal.c added EXCL.  This driver is
 * protocol-agnostic and, built once per config dir, completes the matrix:
 * pairing any two DIFFERENT-config binaries (RCU_HOME, RCU_OWNER,
 * WRF_VAL, EXCL) makes the first cross-protocol message hit
 * a fatal arm.
 *
 * What each protocol emits across the rank boundary (so every fatal arm is
 * reached by SOME pairing):
 *   - RW remote acquire → the build's RW-ownership wire:
 *       VAL HOME : OWNERSHIP_REQUEST/RESPONSE + PUBLISH/PUBLISH_ACK
 *       VAL OWNER  : OWNERSHIP_REQUEST/RESPONSE + CONFIRM/CONFIRM_ACK
 *       WRF_VAL             : PUBLISH/PUBLISH_ACK (no ownership)
 *       EXCL             : LOCK_REQUEST/GRANT/RELEASE/RELEASE_ACK
 *   - RO remote acquire → the build's RO wire:
 *       non-EXCL         : SNAPSHOT_REQUEST/RESPONSE (+ OWNER SNAPSHOT_REDIRECT)
 *       EXCL             : LOCK_REQUEST(RO)
 * A receiver compiled for a different protocol either fatals on the foreign
 * coherence tag (the #if fatal arms) or — for EXCL's own REQUEST/GRANT/RELEASE
 * tags, which non-EXCL builds do not even compile a case for — falls into the
 * `default` arm (arts_shutdown + arts_runtime_stop).  Either path makes the
 * receiving rank exit non-zero (or stall until the harness -k SIGKILL), which
 * is the mismatch-detected signal.
 *
 * Like coherence_mode_mismatch.c / excl_mode_mismatch_fatal.c this is a
 * STANDALONE pairing driver, NOT a plain ctest: the mismatch only exists when
 * the integrator launches two DIFFERENT-config builds of this binary as the two
 * ranks.  When both ranks are the SAME protocol the program completes normally
 * and exits 0 — so the source is also correct as a single-protocol run.
 *
 * No compile-time protocol skip: the source is built unchanged in every config
 * dir; the fatal is produced by PAIRING different-config binaries at run time.
 * A single rank cannot manifest a cross-rank mismatch, so on <2 ranks it SKIPs.
 */

#include <stdint.h>
#include <stdio.h>

#include "arts.h"

/* Remote RW writer (rank 1): forces the build's RW-ownership/publish (or
 * LOCK_REQUEST(RW)/LOCK_RELEASE) wire across the rank boundary. */
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

/* Remote RO reader (rank 1): forces the build's RO-snapshot (or
 * LOCK_REQUEST(RO)) wire across the rank boundary, then shuts down so a
 * SAME-protocol pairing (the no-mismatch case) terminates cleanly with exit 0.
 */
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
    /* A cross-protocol mismatch can only manifest across a rank boundary. */
    arts_printf(
        "SKIP dispatcher_mode_mismatch_matrix: requires 2+ ranks (got %u)\n",
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

  /* Remote RW writer on rank 1: drives the RW-ownership / publish / lock
   * wire (whatever this build emits) over the rank boundary. */
  arts_edt_hint_t wh = ARTS_EDT_HINT_DEFAULTS;
  wh.rank = 1;
  arts_guid_t w = arts_edt_create(writer_edt, 0, NULL, 1, &wh);
  arts_add_dependence(db, w, 0, DB_MODE_RW);

  /* Remote RO reader on rank 1: drives the RO-snapshot / lock-RO wire. */
  arts_edt_hint_t rh = ARTS_EDT_HINT_DEFAULTS;
  rh.rank = 1;
  arts_guid_t r = arts_edt_create(reader_edt, 0, NULL, 1, &rh);
  arts_add_dependence(db, r, 0, DB_MODE_RO);

  /* Drop the creator RW hold so the remote acquires drive the cross-node wire.
   */
  arts_db_release(db, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
