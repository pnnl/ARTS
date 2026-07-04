/* SPDX-License-Identifier: Apache-2.0
 *
 * T141 (C13) — cross-rank arts_edt_destroy: cancel-before-fire.
 *
 * arts_edt_destroy forwards to the EDT's home rank when the target is remote
 * (MSG_EDT_DESTROY), where the handler runs the OCR contract guard: an EDT
 * with depc_needed==0 (already runnable) must NOT be destroyed (UB, ignored);
 * an EDT still waiting on deps may be cancelled.  This test exercises that
 * cancel-before-fire path across ranks: rank 0 creates an EDT homed on rank 1
 * with an un-satisfied dep, then issues arts_edt_destroy from rank 0.  The
 * destroy crosses 0->1 and cancels the never-satisfied EDT.  If the EDT
 * wrongly fired anyway it would bump a counter; the checker verifies it did
 * NOT run.
 *
 * Out of scope by design: a destroy racing a satisfy that completes the same
 * EDT (destroy-vs-satisfy).  Destroy is the caller's responsibility, to be
 * issued only once the caller has already established that the target's
 * dependences will never be satisfied (or, symmetrically, will never race a
 * concurrent destroy).  Calling destroy back-to-back with a satisfy that may
 * complete the same EDT is an unordered use with no guaranteed outcome
 * (including a hang); that is not a runtime defect.
 *
 * Needs node_count >= 2; SKIPs cleanly otherwise.
 * runtime_multinode, all configs.
 */
#include "arts.h"

#include <stdint.h>
#include <stdio.h>

static int g_failed = 0;

/* Counter DB homed on rank 1: [0]=cancelled-EDT-ran (must stay 0). */
#define C_CANCELLED 0
#define C_SLOTS 1

/* The EDT that must be cancelled (never run). depv[0]=counter RW. */
void victim_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *c = (int *)depv[0].ptr;
  if (c) {
    c[C_CANCELLED] = 1; /* should never happen */
  }
}

/* Checker on rank 0: depv[0]=RO counter homed on rank 1. */
void checker(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const int *c = (const int *)depv[0].ptr;
  if (c == NULL) {
    arts_printf("FAIL edt_destroy_crossrank: checker NULL counter\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }
  if (c[C_CANCELLED] != 0) {
    arts_printf("FAIL edt_destroy_crossrank: cancelled EDT ran (cross-rank "
                "destroy did not cancel)\n");
    g_failed = 1;
  } else {
    arts_printf("PASS edt_destroy_crossrank: cross-rank cancel-before-fire "
                "verified\n");
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
    arts_printf("SKIP edt_destroy_crossrank: needs node_count >= 2\n");
    arts_shutdown();
    return;
  }

  arts_printf("=== edt_destroy_crossrank ===\n");

  /* Counter DB homed on rank 1 (where the victim would run). */
  int *c = NULL;
  arts_guid_t cdb =
      arts_db_create((void **)&c, sizeof(int) * C_SLOTS, ARTS_DB,
                     ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 1});
  for (int i = 0; i < C_SLOTS; i++) {
    c[i] = 0;
  }
  arts_db_release(cdb, DB_MODE_RW);

  /* Victim EDT homed on rank 1 with TWO deps; we satisfy only slot 0
   * (counter) and leave slot 1 un-satisfied so it stays cancellable.  Then
   * destroy it from rank 0 (cross-rank). */
  arts_edt_hint_t vh = ARTS_EDT_HINT_DEFAULTS;
  vh.rank = 1;
  arts_guid_t victim = arts_edt_create(victim_edt, 0, NULL, 2, &vh);
  arts_add_dependence(cdb, victim, 0, DB_MODE_RW);
  /* slot 1 deliberately never satisfied -> victim is waiting -> destroyable. */
  arts_edt_destroy(victim); /* 0 -> 1 wire MSG_EDT_DESTROY */

  /* Checker on rank 0: RO counter snapshot.  No ordering barrier is needed:
   * the victim never acquires the counter's RW lease (acquire_dbs only runs
   * once ALL deps resolve, and slot 1 never does), so its pending destroy
   * cannot race the checker's RO read of the counter. */
  arts_edt_hint_t ch = ARTS_EDT_HINT_DEFAULTS;
  ch.rank = 0;
  arts_guid_t chk = arts_edt_create(checker, 0, NULL, 1, &ch);
  arts_add_dependence(cdb, chk, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
