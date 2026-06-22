/* SPDX-License-Identifier: Apache-2.0
 *
 * outbox_partial_backpressure — runtime_multinode, all configs.
 *
 * Target: the sender partial-send re-park path
 * (libs/src/core/transport/outbox.c arts_transport_pump_outbound +
 *  arts_outbox_partial_store, census 21 §arts_transport_pump_outbound /
 *  §arts_outbox_partial_store).
 *
 * When the kernel socket buffer fills, arts_actual_send returns a non-zero
 * length_remaining (EAGAIN partial); the sender must NOT free the node — it
 * calls arts_outbox_partial_store to advance the header/payload cursors and
 * re-parks the node in arts_outbox_resend[i], retrying on the next pump pass.
 * The cursor arithmetic across the four partial-store cases (header-only
 * partial, header-done/payload-partial, header-still-partial) must reassemble
 * the message byte-exactly, or the peer receives a corrupted packet.
 *
 * Black-box exercise: force LARGE payload transfers that overflow the socket
 * send buffer, so the partial-send/re-park path is taken many times per
 * transfer.  Each remote-home RW DB is sized well beyond a typical socket
 * SO_SNDBUF (256 KiB here) and carries a verifiable byte pattern.  A remote EDT
 * RW-acquires it (large inbound payload transfer), rewrites the pattern with a
 * known transform, releases it (large writeback), and a home RO reader verifies
 * EVERY element.  A single mis-stitched partial send corrupts an element ->
 * caught.  A lost partial (freed instead of re-parked) stalls the transfer ->
 * finish scope never fires -> ctest TIMEOUT.
 *
 * Note on flush-vs-pump divergent error semantics (census suspected-bug
 * MEDIUM): the steady-state sender path (pump_outbound) is what handles
 * EAGAIN partials here; the shutdown flush path's 5s-spin error handling is not
 * separately drivable from app code without a dead peer, so this test covers
 * the pump re-park correctness and documents the divergence
 * (exposes_runtime_bug = the divergence is recorded, not asserted-failing
 * here).
 *
 * Config-agnostic.  On 1n the transfer is local (no wire partial) -> trivial
 * pass.  No in-test watchdog; hang reaped by ctest TIMEOUT.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"

/* Payload large enough to overflow the socket send buffer and force multiple
 * EAGAIN partial sends per transfer.  64K * 8B = 512 KiB per DB. */
#define NELEMS (64u * 1024u)
/* Several big DBs in flight concurrently to keep the sender churning partials.
 */
#define NDBS 6

/* writer: RW-acquire (large inbound payload transfer); transform every element
 * by a known function of the DB index, then release (large writeback). */
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  uint64_t tag = paramv[0];
  if (d != NULL) {
    for (uint64_t i = 0; i < NELEMS; i++) {
      d[i] = tag ^ (i * 0x9e3779b97f4a7c15ULL);
    }
  }
}

/* reader: RO on home after the writer; verify every element survived the
 * partial-send reassembly byte-exactly. */
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  uint64_t tag = paramv[0];
  if (d == NULL) {
    (void)fprintf(stderr, "FAIL: partial_backpressure reader got NULL ptr\n");
    arts_abort(1);
  }
  for (uint64_t i = 0; i < NELEMS; i++) {
    uint64_t want = tag ^ (i * 0x9e3779b97f4a7c15ULL);
    if (d[i] != want) {
      (void)fprintf(stderr,
                    "FAIL: partial_backpressure corruption at elem %llu — "
                    "want 0x%llx got 0x%llx (mis-stitched partial send)\n",
                    (unsigned long long)i, (unsigned long long)want,
                    (unsigned long long)d[i]);
      arts_abort(1);
    }
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W =
      (nranks > 1) ? 1u : 0u; /* writer rank (remote when possible) */
  arts_printf(
      "=== outbox_partial_backpressure (ranks=%u, dbs=%d, %u elems each) ===\n",
      nranks, NDBS, NELEMS);

  arts_guid_t dbs[NDBS];

  /* Ordering: a sibling RO reader on a DB is not ordered after the sibling RW
   * writer by the dependence wiring alone, so split the writers and readers
   * into two finish scopes (every large RW write + writeback happens-before any
   * RO acquire). */

  /* Phase 1: large RW writers — each forces many EAGAIN partial sends. */
  arts_guid_t fe_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (int k = 0; k < NDBS; k++) {
    uint64_t tag = 0xC0FFEE0000000000ULL + (uint64_t)k;

    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, NELEMS * sizeof(uint64_t), ARTS_DB,
                       ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
    if (db == NULL_GUID) {
      (void)fprintf(stderr, "FAIL: big db create NULL_GUID\n");
      arts_abort(1);
    }
    memset(ptr, 0, NELEMS * sizeof(uint64_t));
    arts_db_release(db, DB_MODE_RW);
    dbs[k] = db;

    /* Writer on remote rank W: large inbound payload transfer + writeback. */
    arts_guid_t w =
        arts_edt_create(writer_edt, 1, &tag, 1,
                        &(arts_edt_hint_t){.rank = W, .finish_event = fe_w});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
  }
  arts_event_wait(fe_w);

  /* Phase 2: home RO readers verify the reassembled bytes. */
  arts_guid_t fe_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (int k = 0; k < NDBS; k++) {
    uint64_t tag = 0xC0FFEE0000000000ULL + (uint64_t)k;
    arts_guid_t rd =
        arts_edt_create(reader_edt, 1, &tag, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe_r});
    arts_add_dependence(dbs[k], rd, 0, DB_MODE_RO);
  }

  arts_event_wait(fe_r);

  printf("PASS outbox_partial_backpressure: %d big DBs reassembled byte-exact "
         "across partial sends\n",
         NDBS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
