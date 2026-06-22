/* SPDX-License-Identifier: Apache-2.0
 *
 * outbox_send_payload_leak — runtime_multinode, all configs.
 *
 * Target: arts_transport_send_payload header-sent / payload-hard-errors path
 * (libs/src/core/transport/socket.c arts_transport_send_payload, census 20
 *  §arts_transport_send_payload; census 21 free-site table).
 *
 * Documented hazard (census, LOW): when the header fully sends but the second
 * actual_send (payload) hits a HARD error (errno != EAGAIN, e.g. a peer that
 * died mid-transfer), send_payload returns -1.  The outbox_pending counter is
 * kept BALANCED by the pre-increment dance (header-done pre-increments, the
 * failing actual_send decrements once -> net zero), so the shutdown drain does
 * NOT hang or wrap.  But the payload buffer may LEAK until the timeout cleanup
 * path frees it: pump's -1 arm frees a freeable payload, yet the
 * send_payload_async (borrowed, free_method==NULL) payload is caller-owned and
 * not freed by the transport at all on -1.
 *
 * What this test CAN assert (the half that is reachable without killing a live
 * peer): the BALANCED, normal header+payload path.  It drives a large volume of
 * header+payload sends (remote-home RW transfers, the send_payload code path
 * with the pre-increment) and verifies:
 *   - every transferred value is correct (the pre-increment never
 *     double-counts / drops a balanced transfer), and
 *   - the runtime shuts down cleanly (outbox_pending returns to exactly 0; a
 *     broken pre-increment would hang the shutdown drain -> ctest TIMEOUT, or
 *     wrap it -> premature shutdown -> stale read -> arts_abort).
 *
 * The hard-error leak window itself fires only on genuine peer death mid-
 * transfer, which cannot be orchestrated from application code without breaking
 * the runtime for the rest of the test suite — so it is DOCUMENTED here (this
 * comment + the metadata status_note), not asserted-failing.  See census 20/21
 * for the leak-until-cleanup detail.
 *
 * Config-agnostic.  On 1n no wire payload send occurs -> trivial pass.  No in-
 * test watchdog; hang reaped by ctest TIMEOUT.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"

/* Many small-but-nonzero payloads so the send_payload header+payload split path
 * (with the pre-increment) is taken thousands of times. */
#define NDBS 64
#define NELEMS 8u
#define TAG_BASE 0xA11CE000u

static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  uint64_t tag = paramv[0];
  if (d != NULL) {
    for (unsigned i = 0; i < NELEMS; i++) {
      d[i] = tag + i;
    }
  }
}

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  uint64_t tag = paramv[0];
  if (d == NULL) {
    (void)fprintf(stderr, "FAIL: send_payload_leak reader got NULL ptr\n");
    arts_abort(1);
  }
  for (unsigned i = 0; i < NELEMS; i++) {
    if (d[i] != tag + i) {
      (void)fprintf(stderr,
                    "FAIL: send_payload_leak value at elem %u — want %llu got "
                    "%llu (pre-increment mis-accounted transfer)\n",
                    i, (unsigned long long)(tag + i), (unsigned long long)d[i]);
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
  arts_printf("=== outbox_send_payload_leak (ranks=%u, dbs=%d) ===\n", nranks,
              NDBS);

  /* Ordering: sibling EDTs sharing one DB are NOT mutually ordered by the
   * dependence wiring alone (the RO reader may be granted before the RW writer
   * completes, reading the stale initial value).  Split writers and readers
   * into two finish scopes so every RW write happens-before every RO acquire.
   */
  unsigned int total = nranks * NDBS;
  arts_guid_t *dbs = (arts_guid_t *)malloc(sizeof(arts_guid_t) * total);
  if (dbs == NULL) {
    (void)fprintf(stderr, "FAIL: alloc db array\n");
    arts_abort(1);
  }

  /* Phase 1: RW writers (header+payload transfers with the pre-increment). */
  arts_guid_t fe_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int r = 0; r < nranks; r++) {
    for (int k = 0; k < NDBS; k++) {
      unsigned int idx = r * NDBS + (unsigned int)k;
      uint64_t tag = TAG_BASE + (uint64_t)idx * 16u;

      void *ptr = NULL;
      arts_guid_t db =
          arts_db_create(&ptr, NELEMS * sizeof(uint64_t), ARTS_DB,
                         ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
      if (db == NULL_GUID) {
        (void)fprintf(stderr, "FAIL: db create NULL_GUID\n");
        arts_abort(1);
      }
      memset(ptr, 0, NELEMS * sizeof(uint64_t));
      arts_db_release(db, DB_MODE_RW);
      dbs[idx] = db;

      arts_guid_t w =
          arts_edt_create(writer_edt, 1, &tag, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe_w});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
    }
  }
  arts_event_wait(fe_w);

  /* Phase 2: RO readers verify the balanced transfers. */
  arts_guid_t fe_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int r = 0; r < nranks; r++) {
    for (int k = 0; k < NDBS; k++) {
      unsigned int idx = r * NDBS + (unsigned int)k;
      uint64_t tag = TAG_BASE + (uint64_t)idx * 16u;
      arts_guid_t rd =
          arts_edt_create(reader_edt, 1, &tag, 1,
                          &(arts_edt_hint_t){.rank = 0, .finish_event = fe_r});
      arts_add_dependence(dbs[idx], rd, 0, DB_MODE_RO);
    }
  }

  arts_event_wait(fe_r);
  free(dbs);

  printf(
      "PASS outbox_send_payload_leak: %u ranks x %d header+payload transfers "
      "balanced, clean shutdown\n",
      nranks, NDBS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
