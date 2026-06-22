/* SPDX-License-Identifier: Apache-2.0
 *
 * outbox_pending_zero_at_shutdown — runtime_multinode, all configs.
 *
 * Target: the cross-file outbox_pending +1/-1 accounting invariant
 * (libs/src/core/transport/outbox.c arts_outbox_insert_node +1 ;
 *  libs/src/core/transport/socket.c arts_actual_send / send_payload -1 ;
 *  census 20 §arts_actual_send, census 21 §arts_outbox_insert_node, and
 *  SUSPECTED-BUG "B-outbox-pending-parity").
 *
 * The invariant: exactly one +1 per logical message (insert_node), one -1 per
 * arts_actual_send call, with arts_transport_send_payload pre-incrementing
 * before its second actual_send so a header+payload split stays balanced.  If
 * any send path adds/removes an actual_send without a matching +1, the
 * shutdown-protocol outbox drain (wait_for_outbox_drain) either HANGS (count
 * stuck > 0) or shuts down PREMATURELY (count underflows/wraps below 0).
 *
 * Black-box exercise: drive a known, large volume of cross-rank traffic of
 * BOTH shapes, then shut down cleanly.
 *   - HEADER-only sends: every RW ownership transfer's control messages
 *     (LOCK_REQ, GRANT, INVALIDATE, release) ride arts_transport_send_async
 *     (payload==NULL).
 *   - HEADER+PAYLOAD sends: an RW DB whose home is a remote rank carries its
 *     bytes to the acquirer via arts_transport_send_payload_async — the path
 *     with the pre-increment dance.
 * A wide grid of remote-home RW DBs, each acquired RW by a remote EDT then read
 * back RO on home, generates thousands of both message shapes.  The finish
 * scope guarantees every transfer completed before main_edt proceeds to
 * shutdown; the runtime's shutdown drain then must observe outbox_pending == 0.
 *
 * Failure modes and how they surface:
 *   - count stuck high  -> shutdown drain never completes -> ctest TIMEOUT.
 *   - count wrapped low -> premature shutdown drops an in-flight transfer ->
 *     a reader sees a stale/zero value -> arts_abort(1) (visible FAIL, nonzero
 *     exit).
 * No in-test watchdog, no spin; hang reaped by ctest TIMEOUT.
 *
 * Config-agnostic: the transport accounting is identical under all protocols.
 * On 1n there are no cross-rank sends (every transfer is a self-send or local)
 * so the test trivially passes — still a valid smoke of the shutdown drain.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"

/* Per-rank grid of remote-home RW DBs.  Total messages scale ~ NDBS * ranks. */
#define NDBS 48
#define SENTINEL_BASE 0x5a5a0000u

/* writer: RW-acquires a DB whose home is a different rank (payload transfer
 * inbound) and stamps a per-DB sentinel.  paramv[0] = expected sentinel. */
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/* reader: RO on home after the writer; MUST observe the writer's sentinel.  A
 * premature shutdown (underflowed outbox_pending) that dropped the transfer
 * back to home would leave a stale value here. */
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL: outbox_pending_zero_at_shutdown stale read — "
                  "expected 0x%x got 0x%x (dropped transfer?)\n",
                  expect, d ? d[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nranks = arts_get_total_ranks();
  arts_printf(
      "=== outbox_pending_zero_at_shutdown (ranks=%u, dbs/rank=%d) ===\n",
      nranks, NDBS);

  /* Build a grid: for each remote rank, NDBS DBs homed on rank 0.  Each is
   * written RW by an EDT on that remote rank (forces an inbound payload
   * transfer + a writeback) then read RO on rank 0 (forces a return transfer).
   *
   * Ordering: sibling EDTs sharing one DB are NOT mutually ordered by the
   * dependence wiring alone (the runtime may grant the RO reader before the RW
   * writer's ownership transfer completes, reading the stale initial value).
   * To get a true writer-before-reader happens-before, the writers ride one
   * finish scope and the readers a second scope wired only after the first has
   * quiesced (every RW write is written back before any RO acquire). */
  unsigned int total = nranks * NDBS;
  arts_guid_t *dbs = (arts_guid_t *)malloc(sizeof(arts_guid_t) * total);
  if (dbs == NULL) {
    (void)fprintf(stderr, "FAIL: alloc db array\n");
    arts_abort(1);
  }

  /* Phase 1: all RW writers under finish scope fe_w. */
  arts_guid_t fe_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int r = 0; r < nranks; r++) {
    for (int k = 0; k < NDBS; k++) {
      unsigned int idx = r * NDBS + (unsigned int)k;
      unsigned int sentinel = SENTINEL_BASE + idx;

      void *ptr = NULL;
      arts_guid_t db =
          arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                         &(arts_db_hint_t){.rank = 0});
      if (db == NULL_GUID) {
        (void)fprintf(stderr, "FAIL: db create NULL_GUID\n");
        arts_abort(1);
      }
      ((unsigned int *)ptr)[0] = 0u;
      arts_db_release(db, DB_MODE_RW);
      dbs[idx] = db;

      uint64_t pv = (uint64_t)sentinel;
      /* Writer on the remote rank (rank 0 when nranks==1). */
      arts_guid_t w =
          arts_edt_create(writer_edt, 1, &pv, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe_w});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
    }
  }
  arts_event_wait(fe_w); /* all RW writes complete + written back to home */

  /* Phase 2: all RO readers under finish scope fe_r — each MUST see its
   * writer's sentinel now that the write has happened-before. */
  arts_guid_t fe_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int r = 0; r < nranks; r++) {
    for (int k = 0; k < NDBS; k++) {
      unsigned int idx = r * NDBS + (unsigned int)k;
      uint64_t pv = (uint64_t)(SENTINEL_BASE + idx);
      arts_guid_t rd =
          arts_edt_create(reader_edt, 1, &pv, 1,
                          &(arts_edt_hint_t){.rank = 0, .finish_event = fe_r});
      arts_add_dependence(dbs[idx], rd, 0, DB_MODE_RO);
    }
  }

  /* Wait for every transfer to complete, then shut down.  The runtime's
   * shutdown-protocol outbox drain must see outbox_pending == 0; a parity bug
   * makes this hang (count high) — reaped by ctest TIMEOUT — and a premature
   * shutdown is caught by a stale reader above. */
  arts_event_wait(fe_r);
  free(dbs);

  printf("PASS outbox_pending_zero_at_shutdown: %u ranks x %d DBs drained "
         "clean\n",
         nranks, NDBS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
