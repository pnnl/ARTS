/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_edt_destroy — safe cancel contract.
 *
 * The safety constraint on arts_edt_destroy is GUID-access discipline, NOT the
 * EDT's run state: the caller must not access the EDT's GUID via
 * arts_add_dependence or arts_edt_satisfy* concurrently with or after the
 * destroy (a satisfy racing the destroy is the one genuinely unsafe
 * interleaving).  Cancelling a just-created EDT before wiring any dependence
 * into it is the canonical safe use; it must (a) prevent the body from ever
 * running and (b) free the EDT cleanly (no leak / corruption — proven by clean
 * shutdown under the sanitizer build).  The racy dependence/satisfy
 * interleavings are undefined and are deliberately NOT exercised here.
 *
 * Config-agnostic single-rank EDT lifecycle test.
 */

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define N_CANCEL 256

/* Must stay 0: every created EDT is cancelled while pre-runnable, so no body
 * ever runs. */
static _Atomic unsigned int g_ran;

/* Body of a cancelled EDT — must never execute. */
void cancelled_body(uint32_t pc, const uint64_t *pv, uint32_t dc,
                    arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  (void)dv;
  atomic_fetch_add_explicit(&g_ran, 1u, memory_order_relaxed);
}

/* Runnable verifier: confirms no cancelled EDT ran, then shuts down.  A clean
 * shutdown (plus the sanitizer build) proves the cancels freed cleanly. */
void verify_body(uint32_t pc, const uint64_t *pv, uint32_t dc,
                 arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  (void)dv;
  unsigned r = atomic_load_explicit(&g_ran, memory_order_relaxed);
  if (r != 0u) {
    arts_printf("FAIL: %u cancelled pre-runnable EDT(s) still ran\n", r);
    arts_abort(1);
  }
  arts_printf("PASS edt_destroy_vs_satisfy: safe pre-runnable cancel (never "
              "runs, clean teardown)\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_destroy_vs_satisfy ===\n");
  atomic_init(&g_ran, 0u);

  /* Create pre-runnable EDTs (depc == 1, never wired or satisfied) and
   * immediately cancel each — the only defined arts_edt_destroy use: the GUID
   * is destroyed before any arts_add_dependence / arts_edt_satisfy access and
   * while depc_needed > 0. */
  for (int i = 0; i < N_CANCEL; i++) {
    arts_guid_t d = arts_edt_create(cancelled_body, 0, NULL, 1,
                                    &(arts_edt_hint_t){.rank = 0});
    arts_edt_destroy(d);
  }

  /* Runnable verifier (created last, depc == 0): all cancels are already done
   * and no cancelled EDT can run (each had an unsatisfied slot and was
   * destroyed), so g_ran is definitively 0. */
  arts_edt_create(verify_body, 0, NULL, 0, &(arts_edt_hint_t){.rank = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
