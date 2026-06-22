/* SPDX-License-Identifier: Apache-2.0
 *
 * T149 — cross-rank proxy finish DECR emitted by arts_owned_finish_cleanup.
 *
 * Target: the only externally-visible side effect of edt_context.c —
 * arts_owned_finish_cleanup issues arts_event_satisfy_slot(..., DECR) for every
 * un-consumed owned finish token, and that satisfy may route to a REMOTE home
 * (a cross-rank wire send).  This test forces the cleanup-driven DECR to cross
 * a rank boundary.
 *
 * Structure (needs node_count >= 2; SKIPs cleanly otherwise):
 *   - rank 0 main_edt creates an OUTER finish scope F (homed on rank 0).
 *   - an ORCHESTRATOR EDT is placed on rank 1 and JOINED to F.  Because F is
 *     remote to rank 1, the orchestrator's membership DECR (emitted from
 *     arts_unset_thread_local_edt_info on completion) crosses rank 1 -> 0.
 *   - the orchestrator, running on rank 1, creates an INNER finish scope f
 *     (homed rank 1).  The runtime auto-chains f under the ambient F: it INCRs
 *     F (rank 1 -> 0 wire) and registers f -> F DECR.  f's creator-token is
 *     registered in the orchestrator's owned_finish_list.
 *   - the orchestrator spawns leaves under f (rank 1) and RETURNS WITHOUT
 *     WAITING.  Its epilogue arts_owned_finish_cleanup DECRs f's creator-token;
 *     once f's leaves also DECR, f fires and its auto-chain DECRs F on rank 0
 *     (another cross-rank wire).  This is the cleanup -> cross-rank-proxy DECR
 *     path under test.
 *   - a CHECKER on rank 0 depends on F and verifies all leaves ran (via a
 *     counter DB), then shuts down.  A lost cross-rank DECR -> F never drains
 *     -> ctest TIMEOUT.
 *
 * exposes_runtime_bug = false (pins correct cross-rank cleanup DECR routing).
 */
#include "arts.h"

#include <stdint.h>

#define INNER_LEAVES 4
#define N_SLOTS INNER_LEAVES

static int g_failed = 0;

/* leaf on rank 1: RW dep on counter DB, paramv[0] = slot. */
void leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *c = (int *)depv[0].ptr;
  if (c) {
    c[(int)paramv[0]] = 1;
  }
}

/* orchestrator on rank 1: member of remote F; creates inner finish f and
 * returns without waiting -> cleanup DECRs f's token (proxy-chains F on rank
 * 0).
 */
void orchestrator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  arts_guid_t cdb = depv[0].guid;

  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t f = arts_event_create(&fh); /* auto-chains under ambient F */

  arts_edt_hint_t lh = ARTS_EDT_HINT_DEFAULTS;
  lh.finish_event = f;
  lh.rank = 1; /* keep leaves local to rank 1 */
  for (int i = 0; i < INNER_LEAVES; i++) {
    uint64_t slot = (uint64_t)i;
    arts_guid_t l = arts_edt_create(leaf, 1, &slot, 1, &lh);
    arts_add_dependence(cdb, l, 0, DB_MODE_RW);
  }
  /* Return WITHOUT arts_event_wait(f): epilogue cleanup DECRs f's token. */
}

/* checker on rank 0: depends on F + RO counter DB. */
void checker(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *c = (int *)depv[1].ptr;
  if (c == NULL) {
    arts_printf("FAIL ctx_crossrank_proxy_finish: checker NULL counter DB\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }
  int sum = 0;
  for (int i = 0; i < INNER_LEAVES; i++) {
    sum += c[i];
  }
  if (sum != INNER_LEAVES) {
    arts_printf("FAIL ctx_crossrank_proxy_finish: only %d/%d inner leaves ran "
                "before F drained (cross-rank cleanup DECR lost?)\n",
                sum, INNER_LEAVES);
    g_failed = 1;
  } else {
    arts_printf("PASS ctx_crossrank_proxy_finish: %d inner leaves drained via "
                "cross-rank cleanup proxy DECR before F fired\n",
                INNER_LEAVES);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("SKIP ctx_crossrank_proxy_finish: needs node_count >= 2\n");
    arts_shutdown();
    return;
  }

  /* Counter DB homed on rank 1 so the rank-1 leaves write locally. */
  int *c = NULL;
  arts_guid_t cdb =
      arts_db_create((void **)&c, sizeof(int) * N_SLOTS, ARTS_DB,
                     ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 1});
  for (int i = 0; i < N_SLOTS; i++) {
    c[i] = 0;
  }
  arts_db_release(cdb, DB_MODE_RW);

  /* Outer finish scope F homed on rank 0 (current rank). */
  arts_event_hint_t Fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t F = arts_event_create(&Fh);

  /* Orchestrator on rank 1, joined to remote F. */
  arts_edt_hint_t oh = ARTS_EDT_HINT_DEFAULTS;
  oh.rank = 1;
  oh.finish_event = F;
  arts_guid_t o = arts_edt_create(orchestrator, 0, NULL, 1, &oh);
  arts_add_dependence(cdb, o, 0, DB_MODE_RW);

  /* Checker on rank 0 depends on F draining. */
  arts_edt_hint_t ch = ARTS_EDT_HINT_DEFAULTS;
  ch.rank = 0;
  arts_guid_t chk = arts_edt_create(checker, 0, NULL, 2, &ch);
  arts_add_dependence(F, chk, 0, DB_MODE_NULL);
  arts_add_dependence(cdb, chk, 1, DB_MODE_RO);
  /* main_edt returns: epilogue cleanup DECRs F's creator-token on rank 0. */
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
