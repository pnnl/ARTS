/* SPDX-License-Identifier: Apache-2.0
 *
 * T152 — arts_cleanup_edt_tls leak detector: owned_finish_list must be drained
 * (empty) in normal flow before thread teardown.
 *
 * Target: arts_cleanup_edt_tls (libs/src/core/edt_context.c, runtime.c:587 at
 * per-worker shutdown).  Hazard (census f14 suspected-bug #2, LOW): if the
 * owned_finish_list still holds live (non-NULL) creator-tokens at teardown,
 * cleanup_edt_tls FREES them WITHOUT issuing the DECR — the finish event would
 * never fire / be leaked.  In normal flow the per-EDT epilogue
 * arts_owned_finish_cleanup + the init-time cleanups drain every token, so the
 * list is empty at teardown.
 *
 * The owned_finish_list is file-static and not directly observable, so this is
 * an OBSERVABLE leak detector: every finish-event creator-token registered in
 * normal flow MUST be DECR'd (by wait-consume or by epilogue cleanup) so its
 * scope fires.  A token that escaped both — and would be silently dropped by
 * arts_cleanup_edt_tls — manifests as a finish scope that never drains.
 *
 * Stress: WAVES sequential producer EDTs, each creating SCOPES finish events
 * and returning WITHOUT waiting (so every token must be drained by the EDT
 * epilogue cleanup, never surviving to teardown).  Each scope has one leaf and
 * a successor; the successor increments a per-scope counter.  An OUTER finish
 * scope F gates a checker that verifies EVERY scope fired EXACTLY once.  Any
 * leaked (undrained) token -> that scope never fires -> checker FAIL or ctest
 * TIMEOUT.  The producers run back-to-back so their owned_finish_lists are the
 * same reused thread-local — exercising repeated register/cleanup cycles that
 * must each leave the list empty.
 *
 * exposes_runtime_bug = false (pins the normal-flow no-leak invariant).
 */
#include "arts.h"

#include <stdint.h>

#define WAVES 4
#define SCOPES 5
#define TOTAL (WAVES * SCOPES)

/* Counter DB: TOTAL leaf slots [0..TOTAL), TOTAL successor-count slots
 * [TOTAL..2*TOTAL). */
#define LEAF_SLOT(g) (g)
#define SUCC_SLOT(g) (TOTAL + (g))
#define N_SLOTS (2 * TOTAL)

static int g_failed = 0;

void leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *c = (int *)depv[0].ptr;
  if (c) {
    c[LEAF_SLOT((int)paramv[0])] = 1;
  }
}

void succ(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int g = (int)paramv[0];
  int *c = (int *)depv[0].ptr;
  if (c == NULL) {
    arts_printf("FAIL ctx_cleanup_tls_leak: succ %d NULL DB\n", g);
    g_failed = 1;
    return;
  }
  if (c[LEAF_SLOT(g)] != 1) {
    arts_printf("FAIL ctx_cleanup_tls_leak: scope %d fired before its leaf\n",
                g);
    g_failed = 1;
  }
  c[SUCC_SLOT(g)] += 1;
}

/* Producer wave w: creates SCOPES unwaited finish scopes; their creator-tokens
 * are drained by THIS EDT's epilogue cleanup (never survive to teardown). */
void producer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int w = (int)paramv[0];
  arts_guid_t F = (arts_guid_t)paramv[1];
  arts_guid_t cdb = depv[0].guid;

  for (int j = 0; j < SCOPES; j++) {
    int g = w * SCOPES + j;
    arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
    arts_guid_t fe = arts_event_create(&fh); /* registers a creator-token */

    arts_edt_hint_t lh = ARTS_EDT_HINT_DEFAULTS;
    lh.finish_event = fe;
    uint64_t pg = (uint64_t)g;
    arts_guid_t l = arts_edt_create(leaf, 1, &pg, 1, &lh);
    arts_add_dependence(cdb, l, 0, DB_MODE_RW);

    arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
    sh.finish_event = F; /* successor joins outer F so checker waits for it */
    arts_guid_t s = arts_edt_create(succ, 1, &pg, 2, &sh);
    arts_add_dependence(cdb, s, 0, DB_MODE_RW);
    arts_add_dependence(fe, s, 1, DB_MODE_NULL);
    /* No wait: fe's token is drained by this EDT's epilogue cleanup. */
  }
}

void checker(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *c = (int *)depv[1].ptr;
  if (c == NULL) {
    arts_printf("FAIL ctx_cleanup_tls_leak: checker NULL DB\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }
  for (int g = 0; g < TOTAL; g++) {
    if (c[LEAF_SLOT(g)] != 1) {
      arts_printf("FAIL ctx_cleanup_tls_leak: scope %d leaf never ran\n", g);
      g_failed = 1;
    }
    if (c[SUCC_SLOT(g)] != 1) {
      arts_printf(
          "FAIL ctx_cleanup_tls_leak: scope %d successor fired %d times "
          "(expected 1 — token leaked or double-DECR)\n",
          g, c[SUCC_SLOT(g)]);
      g_failed = 1;
    }
  }
  if (!g_failed) {
    arts_printf("PASS ctx_cleanup_tls_leak: all %d unwaited finish tokens "
                "drained before teardown (no leak)\n",
                TOTAL);
  }
  arts_shutdown();
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

  /* Outer finish scope F: gates the checker until every successor completes. */
  arts_event_hint_t Fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t F = arts_event_create(&Fh);

  for (int w = 0; w < WAVES; w++) {
    arts_edt_hint_t ph = ARTS_EDT_HINT_DEFAULTS;
    ph.finish_event =
        F; /* producers also members of F (their successors too) */
    uint64_t pv[2] = {(uint64_t)w, (uint64_t)F};
    arts_guid_t p = arts_edt_create(producer, 2, pv, 1, &ph);
    arts_add_dependence(cdb, p, 0, DB_MODE_RW);
  }

  arts_edt_hint_t ch = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t chk = arts_edt_create(checker, 0, NULL, 2, &ch);
  arts_add_dependence(F, chk, 0, DB_MODE_NULL);
  arts_add_dependence(cdb, chk, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
