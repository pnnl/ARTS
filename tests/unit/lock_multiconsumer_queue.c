/* SPDX-License-Identifier: Apache-2.0
 *
 * lock_multiconsumer_queue — LOCK-config-specific stress for the home-side
 * rw_waiters Vyukov MPSC queue's single-consumer assumption (census 11-lock
 * §3 + §4 bug A; arts_home_lockreq_queue_pop in lock/home.c).
 *
 * The home rank's rw_waiters queue is a strict single-consumer Vyukov FIFO.
 * It is push'd by concurrent REQUEST handlers (multi-producer) and pop'd by
 * lock_home_grant — which is invoked from BOTH the REQUEST handler and the
 * RELEASE handler (the rw→rw D6 chain), and also drained by DESTROY.  If those
 * home bodies can overlap for the same GUID the pop becomes multi-consumer and
 * a writer's rank is double-popped or lost, losing a GRANT.
 *
 * This stress drives a high volume of RW acquire/release cycles against one
 * home DB, interleaved with RO acquires that flip/extend phases at the home
 * arbiter (multiplying the number of lock_home_grant pop calls).  Each RW
 * writer increments the DB's accumulator; the post-condition checks that EVERY
 * scheduled writer ran exactly once (no lost GRANT) by comparing the
 * accumulator to the writer launch count.  A lost rank/GRANT leaves a writer's
 * EDT permanently parked → it never completes → the finish scope never drains →
 * ctest TIMEOUT (no in-test watchdog, no spin).
 *
 * The W/R EDTs all join one finish scope; verify_edt is gated on that scope
 * firing.  The finish DECR is emitted at EDT completion — AFTER release_dbs —
 * so verify observes every writer's increment only once its RW lease has been
 * released (committed).  Satisfying a plain latch from the writer body instead
 * would fire verify before the last writer's release, letting its RO read a
 * pre-commit (stale) accumulator (a satisfy-before-release / OCR §"release
 * before signal" violation).
 *
 * Runs under MRNEW/MRSW/LOCK — protocols that serialize same-DB RW, so the
 * accumulator deterministically reaches N_WRITERS.  SKIP under MRMW: DB-DRF
 * leaves unordered concurrent RW writers racy (read-modify-write increments
 * lose updates by design), so the acc==N_WRITERS invariant does not hold there.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#if defined(ARTS_PROTOCOL_MRMW)
int main(void) {
  printf("SKIP lock_multiconsumer_queue: serialized-RW invariant (acc=="
         "N_WRITERS) is undefined under MRMW (DB-DRF — unordered concurrent RW "
         "writers race)\n");
  return 0;
}
#else

#define N_WRITERS 256
#define N_READERS 64
#define N_TOTAL (N_WRITERS + N_READERS)

/* DB layout (uint64_t): [0] = writer accumulator. */

/* Writer EDT: RW-acquire the home DB (slot 0), bump accumulator.  Completion
 * (post-release) DECRs the finish scope it joined; no in-body signal. */
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic uint64_t *acc = (_Atomic uint64_t *)depv[0].ptr;
  if (acc != NULL) {
    atomic_fetch_add_explicit(acc, 1u, memory_order_acq_rel);
  }
}

/* Reader EDT: RO-acquire the home DB (slot 0).  Pure read; interleaves RO
 * phases against the RW churn to exercise rw→ro flips and the D7 RO-extend
 * path.  Completion DECRs the finish scope; no in-body signal. */
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

/* verify_edt — bound to the all-done LATCH (slot 0, DB_MODE_NULL) + DB RO
 * (slot 1). */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic uint64_t *acc = (_Atomic uint64_t *)depv[1].ptr;
  uint64_t got = atomic_load_explicit(acc, memory_order_acquire);
  if (got != (uint64_t)N_WRITERS) {
    (void)fprintf(stderr,
                  "FAIL: writer accumulator=%llu (want %d) — lost GRANT\n",
                  (unsigned long long)got, N_WRITERS);
    arts_abort(1);
  }
  printf("lock_multiconsumer_queue: %d writers + %d readers, acc=%llu — PASS\n",
         N_WRITERS, N_READERS, (unsigned long long)got);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== lock_multiconsumer_queue (%d W + %d R) ===\n", N_WRITERS,
              N_READERS);

  unsigned int nranks = arts_get_total_ranks();
  unsigned int home = 0u;

  void *dbp = NULL;
  arts_guid_t db =
      arts_db_create(&dbp, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = home});
  if (db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: db create NULL_GUID\n");
    arts_abort(1);
  }
  if (dbp != NULL) {
    *(uint64_t *)dbp = 0u;
  }
  arts_db_release(db, DB_MODE_RW);

  /* Finish scope joined by every participant; its DECR is emitted at each EDT's
   * completion (after release_dbs), so it fires only once every writer's RW
   * lease has been released — the post-condition then reads committed data. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  if (fe == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: finish-event create NULL_GUID\n");
    arts_abort(1);
  }

  /* Spread participants across ranks (when multinode) so concurrent remote
   * REQUEST + local RELEASE (rw→rw chain) home handlers can overlap — the
   * multi-consumer hazard.  depc deps are satisfied immediately on the data
   * DB, so they contend on the lock.  Interleave W and R issuance. */
  for (int i = 0; i < N_TOTAL; i++) {
    unsigned int r = (nranks > 1) ? (unsigned int)(i % nranks) : 0u;
    if (i % 5 == 4) {
      /* every 5th is a reader. */
      arts_guid_t rd =
          arts_edt_create(reader_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      arts_add_dependence(db, rd, 0, DB_MODE_RO);
    } else {
      arts_guid_t w =
          arts_edt_create(writer_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
    }
  }

  /* verify_edt fires when the finish scope drains (every participant has
   * completed AND released).  The driver returns without waiting, releasing the
   * scope's creator-token so it can fire. */
  arts_guid_t v =
      arts_edt_create(verify_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe, v, 0, DB_MODE_NULL);
  arts_add_dependence(db, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRMW */
