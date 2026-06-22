/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/// @file mark_edt_ready_idempotent.c
/// @brief T059 — duplicate wake of the same (edt, slot) must NOT double-count
///        the dependency down (targets suspected bug B-mark-double-dec).
///
/// White-box runtime test.  `mark_edt_ready_by_guid(edt_guid, slot)`
/// (coherence.c) resolves one parked dep slot: it stamps depv[slot].ptr from
/// the now-installed buffer and then unconditionally calls
/// `arts_db_acquire_account(edt)`, which decrements `acquire_remaining`.  Two
/// distinct delivery paths can wake the SAME (edt, slot) — e.g. a
/// snapshot_response case-2 drain plus a redundant direct response, or a
/// drain racing a response.  Each such wake decrements `acquire_remaining`
/// again; there is no per-slot "already counted" guard (unlike the secured
/// path).  A duplicate wake therefore drives `acquire_remaining` one step too
/// far per duplicate — early-scheduling / lost dep accounting.
///
/// This test builds a parked EDT (gated on an unsatisfied event so it never
/// enters the acquire phase and is never freed), points its slot 0 at a live
/// DB, inflates `acquire_remaining` to a known value so neither wake can reach
/// 0 (the EDT must stay parked — no scheduling, no free, no UAF during the
/// observation), then calls `mark_edt_ready_by_guid` TWICE for slot 0.
///
/// CORRECTNESS CONTRACT (what this test asserts): a duplicate wake of one
/// logical slot must decrement `acquire_remaining` by exactly 1, because the
/// slot resolves once.  mark_edt_ready_by_guid CAS-claims depv[slot].ptr
/// (NULL->data) so only the first wake keeps its buffer ref and accounts; a
/// duplicate drops its extra ref and skips the account.  It must NOT be
/// weakened to accept delta == 2.
///
/// runtime_single, config-agnostic: `mark_edt_ready_by_guid` is shared and
/// builds in all 6 configs.  Object lifetime is held by the route-table cb
/// handle + the never-firing gate event for the whole observation.

#include "arts.h"

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h"
#include "arts/runtime_types.h"
#include "arts/utils/shared.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

/* The parked EDT never runs; this body must never execute. */
static void never_runs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  (void)fprintf(stderr, "FAIL: parked EDT was scheduled (should never run)\n");
  arts_abort(1);
}

/* Bias so neither wake can drive acquire_remaining to 0 → the EDT stays parked
 * (never scheduled/freed) across both wakes and the read-back. */
#define BIAS 100u

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mark_edt_ready_idempotent ===\n");

  /* A live coherent DB whose buffer mark_edt_ready_by_guid will re-derive. */
  void *p = NULL;
  arts_guid_t db = arts_db_create(&p, sizeof(uint64_t), ARTS_DB_DEFAULT,
                                  ARTS_DB_PROP_NONE, NULL);
  ((uint64_t *)p)[0] = 0xABCDu;
  arts_db_release(db, DB_MODE_RW);

  /* Build the parked EDT with TWO dep slots:
   *   slot 0 = the DB dependency (RO),
   *   slot 1 = an event that is NEVER satisfied.
   * The never-fired event keeps depc_needed > 0 forever, so the EDT never
   * reaches the acquire phase: the runtime never seeds/decrements
   * acquire_remaining and never schedules or frees the EDT.  We then drive the
   * slot-0 wakeup by hand to isolate mark_edt_ready_by_guid's accounting.
   *
   * NOTE: the DB dep at slot 0 immediately satisfies (passive DB model) and
   * decrements depc_needed 2->1; the gate event holds it at 1.  Because the EDT
   * never becomes ready, the runtime leaves depv[0].ptr / acquire_remaining for
   * us to control. */
  arts_guid_t gate = arts_event_create(NULL);
  arts_guid_t eg = arts_edt_create(never_runs, 0, NULL, 2, NULL);
  arts_add_dependence(db, eg, 0, DB_MODE_RO);
  arts_add_dependence(gate, eg, 1, DB_MODE_NULL);

  /* Pin the EDT and overwrite acquire_remaining with the bias so a wake cannot
   * schedule it.  The EDT is parked on its event gate; we own its lifetime via
   * the route-table cb handle held below. */
  arts_shared_ptr_t eh = arts_route_table_lookup_edt(eg);
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_shared_get(eh);
  if (edt == NULL) {
    (void)fprintf(stderr, "FAIL: parked EDT not found in route table\n");
    arts_abort(1);
  }

  /* Ensure slot 0 names the live DB (so mark_edt_ready_by_guid resolves it). */
  arts_edt_dep_t *dv = (arts_edt_dep_t *)arts_get_depv(edt);
  dv[0].guid = db;
  dv[0].mode = DB_MODE_RO;

  atomic_store_explicit((_Atomic unsigned int *)&edt->acquire_remaining, BIAS,
                        memory_order_release);

  unsigned int before = atomic_load_explicit(
      (_Atomic unsigned int *)&edt->acquire_remaining, memory_order_acquire);

  /* Two wakes of the SAME (edt, slot) — a duplicate delivery of one slot. */
  mark_edt_ready_by_guid(eg, 0);
  mark_edt_ready_by_guid(eg, 0);

  unsigned int after = atomic_load_explicit(
      (_Atomic unsigned int *)&edt->acquire_remaining, memory_order_acquire);

  unsigned int delta = before - after;
  arts_printf("acquire_remaining: before=%u after=%u delta=%u (want 1)\n",
              before, after, delta);

  /* The slot resolves once; a duplicate wake must not count it twice. */
  if (delta != 1u) {
    (void)fprintf(stderr,
                  "FAIL: duplicate wake of one slot decremented "
                  "acquire_remaining by %u (want 1) — double-count\n",
                  delta);
    arts_shared_release(&eh);
    /* keep `gate` referenced so the EDT stays parked; quit non-zero. */
    (void)gate;
    arts_abort(1);
  }

  arts_printf("PASS: mark_edt_ready_idempotent (no double-count)\n");

  /* Tear down the deliberately-parked state: the EDT never runs, so it never
   * releases the buffer ref the wake installed at slot 0.  Drop that ref by
   * hand (while the EDT is still pinned by `eh`), then destroy the DB + gate so
   * the observation leaks nothing.  The idempotency contract is already
   * asserted above; this is pure cleanup. */
  struct arts_db_buffer_s *b = arts_db_buf_from_data(dv[0].ptr);
  if (b != NULL) {
    arts_shared_ptr_t cb = b->cb;
    arts_db_buf_release(&cb);
  }
  dv[0].ptr = NULL;
  arts_shared_release(&eh);
  arts_db_destroy(db);
  arts_event_destroy(gate);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
