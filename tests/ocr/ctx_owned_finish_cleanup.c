/* SPDX-License-Identifier: Apache-2.0
 *
 * T147 — owned-finish creator-token balance: exactly-one-DECR per unconsumed
 * finish event, zero-extra-DECR for consumed ones.
 *
 * Target: arts_owned_finish_cleanup / arts_owned_finish_consume
 * (libs/src/core/edt_context.c), driven through the public finish-event API:
 *   - arts_event_create(FINISH) registers a creator-token on the worker
 *     (Mechanism B) + biases the latch by one (the create-time INCR).
 *   - arts_event_wait CONSUMES the token (zeroes the list slot) and issues the
 *     paired DECR itself.
 *   - EDT-epilogue arts_owned_finish_cleanup DECRs every UN-consumed token
 *     exactly once.
 *
 * Balance contract this pins (the owned_finish_list is file-static and cannot
 * be read; we verify the contract through OBSERVABLE latch firing):
 *
 *   - An UNCONSUMED finish event (orchestrator returns without waiting) must
 *     fire EXACTLY once: its successor runs exactly once AND only after its
 *     leaf completed.  A missing cleanup DECR -> never fires -> ctest TIMEOUT.
 *     A spurious extra DECR -> premature fire (successor sees leaf slot 0).
 *
 *   - A CONSUMED finish event (orchestrator arts_event_wait's it) must ALSO
 *     fire exactly once.  If cleanup wrongly DECR'd a consumed token (the
 *     zero-for-consumed property failing), the latch would underflow: either a
 *     premature/duplicate fire (successor count != 1 or leaf slot 0).
 *
 * Mix per orchestrator: K finish scopes, the EVEN-indexed ones are waited
 * (consumed), the ODD-indexed ones are returned-without-wait (unconsumed via
 * cleanup).  Each scope has one leaf (writes its slot) and one successor
 * (increments its own success-count slot).  All successors join an OUTER finish
 * scope F; when F drains a checker EDT verifies every leaf slot == 1 and every
 * successor count == EXACTLY 1 (no double fire, no missing fire).
 *
 * The first-match semantics of arts_owned_finish_consume are exercised by
 * mixing consumed/unconsumed distinct GUIDs: consume must zero only the
 * matching entry, leaving the others for cleanup.  (The GUID-reuse first-match
 * HAZARD documented for B-owned-finish-consume cannot be forced within one
 * EDT-run cycle — unique GUID minting prevents intra-list aliasing — so this
 * test pins the CORRECT first-match-on-distinct-GUIDs behavior rather than the
 * latent mis-fire; exposes_runtime_bug = false.)
 */
#include "arts.h"

#include <stdint.h>

#define K 6 /* finish scopes per orchestrator (3 consumed, 3 unconsumed) */

/* Counter DB layout: K leaf slots [0..K), then K successor-count slots
 * [K..2K). */
#define LEAF_SLOT(i) (i)
#define SUCC_SLOT(i) (K + (i))
#define N_SLOTS (2 * K)

static int g_failed = 0;

/* leaf: RW dep on counter DB; paramv[0] = scope index. Marks its leaf slot. */
void leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *c = (int *)depv[0].ptr;
  if (c) {
    c[LEAF_SLOT((int)paramv[0])] = 1;
  }
}

/* successor: RW dep on counter DB (slot 0), depends on a finish event (slot 1,
 * NULL). paramv[0] = scope index. Increments its success-count slot and checks
 * its leaf already ran (no premature fire). */
void succ(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int idx = (int)paramv[0];
  int *c = (int *)depv[0].ptr;
  if (c == NULL) {
    arts_printf("FAIL ctx_owned_finish_cleanup: succ %d NULL counter DB\n",
                idx);
    g_failed = 1;
    return;
  }
  if (c[LEAF_SLOT(idx)] != 1) {
    arts_printf("FAIL ctx_owned_finish_cleanup: scope %d fired BEFORE its leaf "
                "(premature DECR / latch underflow)\n",
                idx);
    g_failed = 1;
  }
  c[SUCC_SLOT(idx)] += 1; /* must end at exactly 1 */
}

/* checker: fires when OUTER scope F drains. RO dep on counter DB. */
void checker(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *c = (int *)depv[1].ptr;
  if (c == NULL) {
    arts_printf("FAIL ctx_owned_finish_cleanup: checker NULL counter DB\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }
  for (int i = 0; i < K; i++) {
    if (c[LEAF_SLOT(i)] != 1) {
      arts_printf("FAIL ctx_owned_finish_cleanup: scope %d leaf never ran\n",
                  i);
      g_failed = 1;
    }
    if (c[SUCC_SLOT(i)] != 1) {
      arts_printf("FAIL ctx_owned_finish_cleanup: scope %d successor fired %d "
                  "times (expected exactly 1 — token DECR imbalance)\n",
                  i, c[SUCC_SLOT(i)]);
      g_failed = 1;
    }
  }
  if (!g_failed) {
    arts_printf("PASS ctx_owned_finish_cleanup: %d scopes (3 consumed/3 "
                "unconsumed) each fired exactly once\n",
                K);
  }
  arts_shutdown();
}

void orchestrator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  /* cdb passed by VALUE (its GUID): the orchestrator only wires children to it
   * and never reads/writes the buffer.  It must NOT hold a DB grant here —
   * holding cdb RW across the arts_event_wait below self-deadlocks under a
   * sequential single-writer protocol: the awaited leaf needs cdb RW but cannot
   * get the writer token while this parked EDT holds it.  (Protocols that allow
   * intra-node concurrent RW would mask the hazard.) */
  arts_guid_t cdb = (arts_guid_t)paramv[0];

  /* Outer finish scope F: gates the checker until ALL successors complete. */
  arts_event_hint_t Fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t F = arts_event_create(&Fh);

  for (int i = 0; i < K; i++) {
    arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
    arts_guid_t fe = arts_event_create(&fh); /* registers creator-token i */

    /* leaf under fe. */
    arts_edt_hint_t lh = ARTS_EDT_HINT_DEFAULTS;
    lh.finish_event = fe;
    uint64_t pi = (uint64_t)i;
    arts_guid_t l = arts_edt_create(leaf, 1, &pi, 1, &lh);
    arts_add_dependence(cdb, l, 0, DB_MODE_RW);

    /* successor on fe, joined to F so the checker waits for it. */
    arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
    sh.finish_event = F;
    arts_guid_t s = arts_edt_create(succ, 1, &pi, 2, &sh);
    arts_add_dependence(cdb, s, 0, DB_MODE_RW);
    arts_add_dependence(fe, s, 1, DB_MODE_NULL);

    if ((i % 2) == 0) {
      /* CONSUMED: wait closes scope fe; cleanup must skip its token. */
      arts_event_wait(fe);
    }
    /* ODD: leave fe UNCONSUMED -> orchestrator-epilogue cleanup DECRs it. */
  }

  /* checker depends on F (drains when every successor done) + RO counter DB. */
  arts_edt_hint_t ch = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t chk = arts_edt_create(checker, 0, NULL, 2, &ch);
  arts_add_dependence(F, chk, 0, DB_MODE_NULL);
  arts_add_dependence(cdb, chk, 1, DB_MODE_RO);
  /* orchestrator returns: epilogue cleanup DECRs F's token AND every
   * unconsumed odd fe token. */
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  int *c = NULL;
  arts_guid_t cdb = arts_db_create((void **)&c, sizeof(int) * N_SLOTS, ARTS_DB,
                                   ARTS_DB_PROP_NONE, NULL);
  for (int i = 0; i < N_SLOTS; i++) {
    c[i] = 0;
  }
  arts_db_release(cdb, DB_MODE_RW);

  arts_edt_hint_t oh = ARTS_EDT_HINT_DEFAULTS;
  uint64_t op =
      (uint64_t)cdb; /* pass cdb by value; orchestrator holds no grant */
  arts_guid_t o = arts_edt_create(orchestrator, 1, &op, 0, &oh);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
