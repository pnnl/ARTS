/* SPDX-License-Identifier: Apache-2.0
 *
 * dispatcher_miss_drop_refpin — the dispatcher's Cat-C lookup-acquire-or-drop
 * routing lens (census 19-dispatcher §2 per-case table, §6.4, §7 bucket C).
 *
 * The dispatcher uniquely owns three local correctness obligations for the
 * Cat-C coherence responses that arrive for a torn-down DB:
 *
 *   (a) NO HANG — a coherence message that would have resumed a parked waiter
 *       may arrive AFTER the DB's route slot was NULLed by a concurrent
 *       destroy.  The dispatcher drops it (the lookup MISSes), but the destroy
 *       fan-out / DESTROY_NOTIFY must already have woken every parked waiter,
 * so no EDT is stranded.  The Cat-C handlers covered: non-EXCL : INVALIDATE,
 * SNAPSHOT_RESPONSE, CACHE_DESTROY, CONFIRM, CONFIRM_ACK (WB). EXCL     :
 * EXCL_RELEASE (Cat-B defer), CACHE_DESTROY.
 *
 *   (b) NO UAF — each Cat-C case ref-pins the db_s (arts_route_table_lookup_db
 *       → arts_shared_get) BEFORE running the handler body and releases the pin
 *       AFTER (arts_shared_release); a concurrent destroy on another receiver
 *       thread therefore cannot free the db_s mid-handler.  A torn-down slot is
 *       a clean NULL get, not a dangling pointer deref.
 *
 *   (c) WAKE-ON-MISS — PUBLISH_ACK (WT/WRF_VAL) and PUBLISH_CTS (EXCL)
 *       post the releaser's stack-local sem by pointer identity, INDEPENDENT of
 *       the route lookup (the body is called even on db==NULL, or the sem_post
 *       is inline).  A torn-down home cache must NOT swallow the ACK or the
 *       blocked releaser hangs.
 *
 * Black-box driver: per generation a home (rank 0) DB is created, RW/RO
 * dependents are fanned out across every rank (each foreign rank installs a
 * sharer cache and may park a waiter on the cache / pending_snapshot reorder
 * buffer / RW FIFO), the finish scope is awaited, then the DB is destroyed and
 * the now-stale GUID destroyed AGAIN.  The next generation re-creates while the
 * previous destroy can still be draining, so a coherence response (INVALIDATE /
 * SNAPSHOT_RESPONSE / CONFIRM / CONFIRM_ACK / PUBLISH_ACK / PUBLISH_CTS)
 * can race a torn-down home cache and exercise the dispatcher MISS branch.  If
 * any waiter is stranded or any ACK dropped, the finish scope never drains and
 * the ctest TIMEOUT reaps it as a FAIL (no in-test spin/watchdog).  The second
 * destroy on a stale GUID exercises the idempotent CACHE_DESTROY MISS (no
 * double-free / cb refcount underflow).
 *
 * Config-agnostic: every protocol routes RW/RO acquires + destroy through the
 * dispatcher Cat-C cases, so this runs unchanged under VAL (WT/WB),
 * WRF_VAL, and EXCL.  Cross-rank handoff (nranks>1, the
 * 2n/3n/4n/2n_io registrations) is what generates the real wire responses; on a
 * single rank the self-loopback Cat-C path still exercises the unconditional
 * ACK post.
 */

#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#define ITERS 200u

/* Tolerant RW holder: a NULL dep means the DB was destroyed out from under it
 * (woken with DB_DESTROYED).  The contract under test is that it RUNS at all —
 * i.e. it was woken, not stranded — and that its release ACK is not dropped. */
static void rw_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

/* Tolerant RO reader: parking RO on the sharer cache (then being woken by the
 * destroy fan-out / SNAPSHOT_RESPONSE / REDIRECT-miss DESTROY_NOTIFY) is the
 * point; a NULL dep is acceptable (DB destroyed first). */
static void ro_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== dispatcher_miss_drop_refpin ===\n");

  unsigned int nranks = arts_get_total_ranks();

  for (unsigned int it = 0; it < ITERS; it++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* Fan RW + RO dependents across every rank: each remote rank installs a
     * sharer cache and may park a waiter the dispatcher's Cat-C response would
     * resume.  The RW chain drives ownership/publish/lock ACK rounds; the RO
     * fan drives SNAPSHOT_RESPONSE / REDIRECT. */
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int r = 0; r < nranks; r++) {
      arts_guid_t w =
          arts_edt_create(rw_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      arts_add_dependence(db, w, 0, DB_MODE_RW);

      arts_guid_t ro =
          arts_edt_create(ro_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      arts_add_dependence(db, ro, 0, DB_MODE_RO);
    }
    /* If a coherence response was dropped on a MISS without the destroy fan-out
     * waking the corresponding waiter, this wait never returns → ctest TIMEOUT.
     */
    arts_event_wait(fe);

    /* Destroy the home: the DESTROY_NOTIFY fan-out drives any next-generation
     * Cat-C response into the torn-down-home MISS branch.  Idempotent second
     * destroy on the stale GUID exercises the CACHE_DESTROY MISS (no
     * double-free / cb underflow / crash). */
    arts_db_destroy(db);
    arts_db_destroy(db);
  }

  arts_printf("PASS: dispatcher_miss_drop_refpin %u iters x %u ranks\n", ITERS,
              nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
