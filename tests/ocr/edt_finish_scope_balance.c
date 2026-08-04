/* SPDX-License-Identifier: Apache-2.0
 *
 * T135 — EDT finish-scope INCR/DECR balance (local + remote-create proxy).
 *
 * Property under test (create_core join INCR + unset DECR; remote proxy)
 * --------------------------------------------------------------------
 * Every EDT that joins a finish scope emits exactly one INCR at create
 * (arts_edt_create_core) and exactly one DECR at completion
 * (arts_unset_thread_local_edt_info).  The scope (a LATCH(1) finish event with
 * a creator-token) fires when its counter returns to 0 — i.e. when the creator
 * releases its token (arts_event_wait) AND every joined EDT has completed.
 *
 * For a REMOTE EDT the INCR is emitted on the source rank inside create_core
 * (against the parent finish event there), and the RX side (arts_handler_edt_
 * create) installs a local proxy LATCH(1) wired to forward a DECR to the remote
 * parent when the proxy drains.  So a remote member still contributes a clean
 * +1/-1 to the parent scope.  If any INCR or DECR is dropped (or doubled) the
 * scope never drains and arts_event_wait hangs forever (caught by ctest
 * TIMEOUT); an over-DECR would fire the scope early before members ran, which
 * the per-member tally would expose.
 *
 * Scenario
 * --------
 * main_edt creates a finish event, then M member EDTs joined to it spread
 * round-robin across all ranks (local + remote when nranks>1), each bumping a
 * shared counter.  main_edt then arts_event_wait(fe): this releases the
 * creator-token and blocks until every member completes.  After the wait
 * returns, every member must have run exactly once — proving the scope drained
 * precisely when (and only when) all members finished.  Then shut down.
 *
 * Single-node: all members local (pure INCR/DECR balance).  Multinode: members
 * on remote ranks exercise the proxy-LATCH forward path.
 */

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define MEMBERS 24

typedef struct {
  _Atomic unsigned int ran; /* number of members that executed */
} ctr_t;

/* Member body: bump the shared counter.  depv[0] = counter DB (RW). */
void member(uint32_t pc, const uint64_t *pv, uint32_t dc, arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  ctr_t *c = (ctr_t *)dv[0].ptr;
  if (c) {
    atomic_fetch_add_explicit(&c->ran, 1u, memory_order_relaxed);
  }
}

/* Verifier body: RO-acquire the counter DB AFTER every member completed, so
 * coherence delivers the final tally (the creator cannot read it through its
 * raw create-time pointer — remote RW members migrate/replace the home buffer,
 * leaving that pointer stale/freed).  depv[0] = counter DB (RO). */
void verifier(uint32_t pc, const uint64_t *pv, uint32_t dc,
              arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  const ctr_t *c = (const ctr_t *)dv[0].ptr;
  unsigned int ran =
      c ? atomic_load_explicit(&c->ran, memory_order_relaxed) : 0u;
  if (ran == (unsigned int)MEMBERS) {
    arts_printf("PASS edt_finish_scope_balance: %d members, scope balanced\n",
                MEMBERS);
  } else {
    arts_printf("FAIL edt_finish_scope_balance: scope drained with ran=%u "
                "(want %d) — INCR/DECR imbalance\n",
                ran, MEMBERS);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_finish_scope_balance ===\n");

  unsigned int nranks = arts_get_total_ranks();

#if defined(ARTS_PROTOCOL_WRF_VAL)
  /* WRF_VAL (DB-WRF) provides no exclusive cross-rank ownership for RW: concurrent
   * RW holders on different nodes each receive a buffer copy and race at
   * PUBLISH time (version-monotonic CAS, last writer wins).  Members on
   * different ranks atomically increment their local copy of cdb, but only one
   * copy survives to the verifier — the other increments are silently lost. */
  if (nranks > 1) {
    arts_printf("SKIP edt_finish_scope_balance: concurrent cross-rank RW "
                "accumulation is DB-WRF racy under WRF_VAL\n");
    arts_shutdown();
    return;
  }
#endif

  void *cp = NULL;
  arts_guid_t cdb =
      arts_db_create(&cp, sizeof(ctr_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ctr_t *c = (ctr_t *)cp;
  atomic_init(&c->ran, 0u);
  arts_db_release(cdb, DB_MODE_RW);

  uint64_t pv[1] = {(uint64_t)cdb};

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* M members joined to the finish scope, round-robin across all ranks.  Each
   * acquires the counter RW (serialized), so the writes are ordered and the
   * final tally is exact. */
  for (int i = 0; i < MEMBERS; i++) {
    unsigned int rank = (unsigned int)i % nranks;
    arts_guid_t m = arts_edt_create(
        member, 1, pv, 1, &(arts_edt_hint_t){.rank = rank, .finish_event = fe});
    arts_add_dependence(cdb, m, 0, DB_MODE_RW);
  }

  /* Release the creator-token and block until the scope drains (all members
   * completed).  If any INCR/DECR is dropped the scope never reaches 0 and this
   * hangs — caught by the ctest TIMEOUT. */
  arts_event_wait(fe);

  /* The wait returned ⇒ the scope drained ⇒ every member's DECR landed, which
   * happens only after each member body ran.  Read the tally back through a
   * verifier EDT (RO acquire) rather than the stale create-time pointer, then
   * shut down once that second scope drains. */
  arts_guid_t fe2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t v = arts_edt_create(
      verifier, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe2});
  arts_add_dependence(cdb, v, 0, DB_MODE_RO);
  arts_event_wait(fe2);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
