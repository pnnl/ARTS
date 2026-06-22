/* SPDX-License-Identifier: Apache-2.0
 *
 * T103 — MRMW GET_DATA dedup watermark stress: hammer one requester rank with
 *        many concurrent SNAPSHOT_REQUESTs and confirm the dedup never serves
 *        stale NO_DATA.
 *
 * Targets update_last_sent_max (census 10, hazard #2).  Under MRMW every
 * non-home acquire (RO and RW) parks on a SNAPSHOT_REQUEST.  The home serves
 * the canonical buffer the first time, records last_sent_version[requester],
 * and replies NO_DATA (deduped) for any later request whose master version is
 * not newer — the parked waiter then resumes against the already-installed
 * cache buffer.  The watermark advance is a get+set (not a monotonic CAS); its
 * correctness relies on the single-network-thread-per-rank dispatch model.  If
 * a NO_DATA were ever served for a version the requester had not yet installed,
 * the reader would resume against missing/stale data and the checksum fails.
 *
 * Structure (2+ ranks; self-skips otherwise):
 *   home = rank 0 writes a known fill; all N readers run on rank 1, acquired
 *   RO concurrently inside one finish scope.  Each reader recomputes the
 *   expected checksum from compile-time constants (no cross-EDT global) and
 *   arts_abort()s on mismatch, so a stale NO_DATA resume is caught.  Repeated
 *   over several rounds to widen the concurrent-serve window.
 *
 * config_specific: MRMW only — the SNAPSHOT_REQUEST/NO_DATA dedup path is the
 * MRMW acquire model; ownership protocols use a different acquire round.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N_READERS 24
#define DB_SIZE 1024
#define DB_FILL 0x37
#define ROUNDS 8u

#if !defined(ARTS_PROTOCOL_MRMW)
void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("SKIP mrmw_getdata_dedup: MRMW-only\n");
  arts_shutdown();
}
#else

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t my_id = paramv[0];
  uint8_t *p = (uint8_t *)depv[0].ptr;
  uint64_t expected = (uint64_t)DB_FILL * DB_SIZE;
  uint64_t sum = 0;
  if (p == NULL) {
    arts_printf(
        "mrmw_getdata_dedup: reader %llu FAIL got NULL (stale NO_DATA)\n",
        (unsigned long long)my_id);
    arts_abort(1);
  }
  for (size_t i = 0; i < DB_SIZE; i++) {
    sum += p[i];
  }
  if (sum != expected) {
    arts_printf("mrmw_getdata_dedup: reader %llu FAIL sum=%llu expected=%llu\n",
                (unsigned long long)my_id, (unsigned long long)sum,
                (unsigned long long)expected);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("SKIP mrmw_getdata_dedup: requires 2+ ranks\n");
    arts_shutdown();
    return;
  }

  for (unsigned int rnd = 0; rnd < ROUNDS; rnd++) {
    void *addr = NULL;
    arts_guid_t db = arts_db_create(&addr, DB_SIZE, ARTS_DB, ARTS_DB_PROP_NONE,
                                    &(arts_db_hint_t){.rank = 0});
    memset(addr, DB_FILL, DB_SIZE);
    arts_db_release(db, DB_MODE_RW);

    /* Finish scope over all N concurrent readers; main waits per round so the
     * DB stays alive for that round's serves. */
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    for (uint64_t i = 0; i < N_READERS; i++) {
      arts_guid_t edt =
          arts_edt_create(reader_edt, 1, &i, 1,
                          &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
      arts_add_dependence(db, edt, 0, DB_MODE_RO);
      (void)edt;
    }

    arts_event_wait(fe);
    arts_db_destroy(db);
  }

  arts_printf(
      "PASS mrmw_getdata_dedup: %u rounds x %d readers, no stale NO_DATA\n",
      ROUNDS, N_READERS);
  arts_shutdown();
}
#endif

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
