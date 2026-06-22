/* SPDX-License-Identifier: Apache-2.0
 *
 * T141 (C13) — cross-rank arts_edt_destroy + destroy-vs-satisfy.
 *
 * arts_edt_destroy forwards to the EDT's home rank when the target is remote
 * (MSG_EDT_DESTROY), where the handler runs the OCR contract guard: an EDT with
 * depc_needed==0 (already runnable) must NOT be destroyed (UB, ignored); an EDT
 * still waiting on deps may be cancelled.  Today only the cancel-before-fire
 * case is covered locally.  This test exercises the CROSS-RANK paths:
 *
 *   (A) cancel-before-fire across ranks: rank 0 creates an EDT homed on rank 1
 *       with one un-satisfied dep, then issues arts_edt_destroy from rank 0.
 *       The destroy crosses 0->1 and cancels the never-satisfied EDT.  If the
 *       EDT wrongly fired anyway it would bump a counter; the checker verifies
 *       it did NOT run.
 *
 *   (B) destroy-vs-satisfy (a satisfy that completes the EDT racing the
 *       destroy): rank 0 creates a second EDT homed on rank 1 with one dep,
 *       SATISFIES it (so it fires + bumps the counter), then issues a destroy.
 *       The destroy must be a safe no-op against an already-runnable/completed
 *       EDT (depc_needed==0 UB-guard) — no crash, no double-free (ASan), and
 *       the EDT's single run is preserved.
 *
 * Needs node_count >= 2; SKIPs cleanly otherwise.
 * runtime_multinode, all configs.
 */
#include "arts.h"

#include <stdint.h>
#include <stdio.h>

static int g_failed = 0;

/* Counter DB homed on rank 1: [0]=cancelled-EDT-ran (must stay 0),
 *                              [1]=satisfied-EDT-ran (must become 1),
 *                              [2]=done flag for the checker. */
#define C_CANCELLED 0
#define C_SATISFIED 1
#define C_SLOTS 3

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

/* The EDT that IS satisfied and must run exactly once. depv[0]=counter RW. */
void survivor_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *c = (int *)depv[0].ptr;
  if (c) {
    c[C_SATISFIED] += 1;
  }
}

/* Barrier on rank 1: takes the counter RW to serialize after the survivor on
 * rank 1's home; does not modify the counter.  Member of finish scope F. */
void barrier_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

/* Checker on rank 0: depv[0]=finish-scope barrier (NULL data),
 *                     depv[1]=RO counter. */
void checker(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const int *c = (const int *)depv[1].ptr;
  if (c == NULL) {
    arts_printf("FAIL edt_destroy_crossrank: checker NULL counter\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }
  bool ok = true;
  if (c[C_CANCELLED] != 0) {
    arts_printf("FAIL edt_destroy_crossrank: cancelled EDT ran (cross-rank "
                "destroy did not cancel)\n");
    ok = false;
  }
  if (c[C_SATISFIED] != 1) {
    arts_printf("FAIL edt_destroy_crossrank: satisfied EDT ran %d times "
                "(expected exactly 1; destroy-vs-satisfy mishandled)\n",
                c[C_SATISFIED]);
    ok = false;
  }
  if (ok) {
    arts_printf("PASS edt_destroy_crossrank: cross-rank cancel + "
                "destroy-vs-satisfy no-op verified\n");
  } else {
    g_failed = 1;
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

  /* Counter DB homed on rank 1 (where the victim/survivor run). */
  int *c = NULL;
  arts_guid_t cdb =
      arts_db_create((void **)&c, sizeof(int) * C_SLOTS, ARTS_DB,
                     ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 1});
  for (int i = 0; i < C_SLOTS; i++) {
    c[i] = 0;
  }
  arts_db_release(cdb, DB_MODE_RW);

  /* (A) Victim EDT homed on rank 1 with TWO deps; we satisfy only slot 0
   * (counter) and leave slot 1 un-satisfied so it stays cancellable.  Then
   * destroy it from rank 0 (cross-rank). */
  arts_edt_hint_t vh = ARTS_EDT_HINT_DEFAULTS;
  vh.rank = 1;
  arts_guid_t victim = arts_edt_create(victim_edt, 0, NULL, 2, &vh);
  arts_add_dependence(cdb, victim, 0, DB_MODE_RW);
  /* slot 1 deliberately never satisfied -> victim is waiting -> destroyable. */
  arts_edt_destroy(victim); /* 0 -> 1 wire MSG_EDT_DESTROY */

  /* (B) Survivor EDT homed on rank 1 with ONE dep; satisfy it (fires), then
   * destroy it (must be a safe no-op against the runnable/completed EDT). */
  arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
  sh.rank = 1;
  arts_guid_t survivor = arts_edt_create(survivor_edt, 0, NULL, 1, &sh);
  arts_add_dependence(cdb, survivor, 0, DB_MODE_RW); /* fires it */
  arts_edt_destroy(survivor); /* destroy-vs-satisfy: no-op expected */

  /* Settle barrier: a finish scope F whose member is a barrier EDT on rank 1
   * that takes the counter RW.  Because RW is per-node exclusive, the barrier
   * is ordered AFTER the survivor's run on rank 1's home, so when F drains the
   * checker reads a stable counter.  The barrier does not modify the counter.
   */
  arts_event_hint_t Fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t F = arts_event_create(&Fh);
  arts_edt_hint_t bh = ARTS_EDT_HINT_DEFAULTS;
  bh.rank = 1;
  bh.finish_event = F;
  arts_guid_t barrier = arts_edt_create(barrier_edt, 0, NULL, 1, &bh);
  arts_add_dependence(cdb, barrier, 0, DB_MODE_RW);

  /* Checker on rank 0: waits on F (barrier scope drained) then reads RO. */
  arts_edt_hint_t ch = ARTS_EDT_HINT_DEFAULTS;
  ch.rank = 0;
  arts_guid_t chk = arts_edt_create(checker, 0, NULL, 2, &ch);
  arts_add_dependence(F, chk, 0, DB_MODE_NULL); /* wait for barrier scope */
  arts_add_dependence(cdb, chk, 1, DB_MODE_RO); /* RO counter snapshot */
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
