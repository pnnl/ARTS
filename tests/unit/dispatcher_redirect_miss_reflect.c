/* SPDX-License-Identifier: Apache-2.0
 *
 * dispatcher_redirect_miss_reflect — the OWNER-only MSG_DB_SNAPSHOT_REDIRECT
 * miss-reflection arm of the dispatcher (census 19-dispatcher §2 per-case table
 * row MSG_DB_SNAPSHOT_REDIRECT; dispatcher.c ~479).
 *
 * Under OWNER, a remote RO acquire is served by the CURRENT owner, not the home:
 * the home forwards the reader's request to the owner as
 * MSG_DB_SNAPSHOT_REDIRECT (carrying the requester rank + parked edt/slot). The
 * dispatcher's REDIRECT case is Cat-C-or-DESTROY_NOTIFY: HIT  (owner cache
 * present) → arts_handler_db_snapshot_redirect serves a DATA_RESPONSE back to
 * the requester (resumes the parked RO waiter); MISS (owner cache destroyed /
 * not yet installed on this rank) → arts_send_db_cache_destroy(requester,
 * db_guid) so the requester's PARKED RO waiter wakes and observes DB_DESTROYED
 * instead of hanging forever waiting for a DATA_RESPONSE that will never come.
 *
 * The MISS-reflection is the contract under test: without it, an RO reader
 * whose REDIRECT raced a destroy at the owner would be stranded (no in-test
 * spin — the ctest TIMEOUT is the failure detector).
 *
 * Black-box driver: per generation a home(0) DB is RW-written by an owner on a
 * REMOTE rank (forcing ownership to migrate off home so a subsequent RO must be
 * REDIRECTed to that owner), then RO readers are fanned across ranks, and the
 * DB is destroyed and re-created across generations so a REDIRECT can land at
 * an owner whose cache is mid-teardown → the MISS branch.  Every RO reader must
 * run (resumed by DATA_RESPONSE on HIT, or woken by the reflected
 * DESTROY_NOTIFY on MISS); the finish scope draining proves none was stranded.
 *
 * Config gate: MSG_DB_SNAPSHOT_REDIRECT exists ONLY in the OWNER-placement builds
 * (HOME serves RO from home directly; the HOME dispatcher fatals on REDIRECT;
 * EXCL/WRF_VAL have no snapshot protocol at all).  Compile-time self-skip on
 * everything that is not OWNER.
 */

#include <stdint.h>
#include <stdio.h>

#include "arts.h"

#if !defined(ARTS_WRITE_POLICY_WB)
int main(void) {
  printf("SKIP dispatcher_redirect_miss_reflect: OWNER-only "
         "(MSG_DB_SNAPSHOT_REDIRECT)\n");
  return 0;
}
#else

#define ITERS 200u
#define WRITE_BASE 0x40000000u

/* Remote RW owner: acquire RW (migrating ownership off home) and stamp a
 * per-iter value, so a later RO must be REDIRECTed from home to this owner. */
static void owner_write_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/* Tolerant RO reader: a NULL dep means the REDIRECT MISSed and the reader was
 * woken by the reflected DESTROY_NOTIFY (DB_DESTROYED) — acceptable.  The
 * contract under test is that the reader RUNS at all (woken, never stranded).
 */
static void ro_reader_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
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

  arts_printf("=== dispatcher_redirect_miss_reflect ===\n");

  unsigned int nranks = arts_get_total_ranks();
  /* The REDIRECT (home → remote owner) only crosses the wire when the owner is
   * a different rank from the home; on a single rank RO is served locally and
   * no REDIRECT is emitted.  Still correct (and fast) at 1n — it degenerates to
   * a local RO/destroy churn that drains trivially. */
  unsigned int owner = (nranks > 1) ? 1u : 0u;

  for (unsigned int it = 0; it < ITERS; it++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    /* Migrate ownership off home to the remote owner. */
    uint64_t v = (uint64_t)(WRITE_BASE + it);
    arts_guid_t wr =
        arts_edt_create(owner_write_edt, 1, &v, 1,
                        &(arts_edt_hint_t){.rank = owner, .finish_event = fe});
    arts_add_dependence(db, wr, 0, DB_MODE_RW);

    /* RO readers across every rank: each remote RO at home is REDIRECTed to the
     * owner.  When the owner cache is mid-teardown the REDIRECT MISSes and must
     * reflect a DESTROY_NOTIFY back so the parked RO waiter wakes. */
    for (unsigned int r = 0; r < nranks; r++) {
      arts_guid_t rd =
          arts_edt_create(ro_reader_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      arts_add_dependence(db, rd, 0, DB_MODE_RO);
    }

    /* A REDIRECT MISS that failed to reflect DESTROY_NOTIFY would strand a
     * reader and this wait would never return → ctest TIMEOUT FAIL. */
    arts_event_wait(fe);

    /* Destroy on home: the owner's cache teardown is what opens the
     * REDIRECT-MISS window for the next generation's RO. */
    arts_db_destroy(db);
  }

  arts_printf("PASS: dispatcher_redirect_miss_reflect %u iters x %u ranks\n",
              ITERS, nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif /* ARTS_WRITE_POLICY_WB */
