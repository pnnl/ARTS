/* SPDX-License-Identifier: Apache-2.0
 *
 * event_counted_reclaim — COUNTED events deliver to every declared consumer AND
 * are reclaimed; ONCE events deliver and are NOT.
 *
 * The decider's table is covered by event_compute_next.  What this covers is the
 * wiring: that the declared count actually moves when a dependence registers,
 * that the destroy edge fires from the same transition, and — the case a review
 * of the design flagged as the one that fails silently — that a consumer
 * binding AFTER the satisfy is still delivered to before the event goes away.
 * Getting that wrong drops the last consumer of a COUNTED(1) event and its slot
 * is never satisfied, which no value oracle elsewhere would notice.
 *
 * Whitebox: reclamation is observed through the route table, since a portable
 * OCR program has no way to ask whether an event still exists.
 */
#include "arts.h"
#include "arts/gas/route_table.h"
#include "../test_failure_status.h"

#include <sched.h>
#include <stdint.h>
#include <stdio.h>

#define NCONS 4u

static _Atomic(uint32_t) g_ran;

static void consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;
  atomic_fetch_add_explicit(&g_ran, 1u, memory_order_acq_rel);
}

static bool present(arts_guid_t g) {
  arts_shared_ptr_t h = arts_route_table_lookup_event(g);
  bool p = (arts_shared_get(h) != NULL);
  arts_shared_release(&h);
  return p;
}

static void wait_ran(uint32_t want) {
  /* The consumers are ordinary EDTs, and this runs INSIDE one — so it waits
   * for the other workers to pick them up and must not try to run them here.
   * Driving the scheduler from inside a task re-enters the per-thread
   * execution bracket the runtime opens around a task body, and a bracket
   * that is not re-entrant reports the second entry rather than nesting.
   * Yielding is the whole cooperation this needs; the spin is bounded, so a
   * run with nobody else to pick them up reports a missing delivery instead
   * of hanging. */
  for (int spin = 0; spin < 200000; spin++) {
    if (atomic_load_explicit(&g_ran, memory_order_acquire) >= want) {
      return;
    }
    sched_yield();
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;

  /* (1) COUNTED, every consumer bound BEFORE the satisfy. */
  atomic_store(&g_ran, 0u);
  arts_event_hint_t ch = ARTS_EVENT_HINT_COUNTED(NCONS);
  arts_guid_t ev = arts_event_create(&ch);
  for (uint32_t i = 0; i < NCONS; i++) {
    arts_guid_t c = arts_edt_create(consumer, 0, NULL, 1, NULL);
    arts_add_dependence(ev, c, 0, DB_MODE_NULL);
  }
  arts_event_satisfy(ev, NULL_GUID);
  wait_ran(NCONS);
  if (atomic_load(&g_ran) != NCONS) {
    arts_printf("FAIL event_counted_reclaim: bound-before-satisfy delivered "
                "%u of %u\n", atomic_load(&g_ran), NCONS);
    arts_test_fail();
  } else if (present(ev)) {
    arts_printf("FAIL event_counted_reclaim: COUNTED event survived after all "
                "%u consumers bound and it fired\n", NCONS);
    arts_test_fail();
  }

  /* (2) COUNTED(1) whose only consumer binds AFTER the satisfy — the case that
   * fails silently if destroy replaces the delivery instead of following it. */
  atomic_store(&g_ran, 0u);
  arts_event_hint_t lh = ARTS_EVENT_HINT_COUNTED(1);
  arts_guid_t late = arts_event_create(&lh);
  arts_event_satisfy(late, NULL_GUID);
  arts_guid_t lc = arts_edt_create(consumer, 0, NULL, 1, NULL);
  arts_add_dependence(late, lc, 0, DB_MODE_NULL);
  wait_ran(1u);
  if (atomic_load(&g_ran) != 1u) {
    arts_printf("FAIL event_counted_reclaim: consumer binding after the "
                "satisfy was never delivered to\n");
    arts_test_fail();
  } else if (present(late)) {
    arts_printf("FAIL event_counted_reclaim: COUNTED(1) survived its late "
                "bind\n");
    arts_test_fail();
  }

  /* (3) An undeclared single-fire event must still linger — reclaiming it is
   * exactly what ARTS cannot do without a count. */
  atomic_store(&g_ran, 0u);
  arts_event_hint_t oh = ARTS_EVENT_HINT_ONCE;
  arts_guid_t once = arts_event_create(&oh);
  arts_guid_t oc = arts_edt_create(consumer, 0, NULL, 1, NULL);
  arts_add_dependence(once, oc, 0, DB_MODE_NULL);
  arts_event_satisfy(once, NULL_GUID);
  wait_ran(1u);
  if (atomic_load(&g_ran) != 1u) {
    arts_printf("FAIL event_counted_reclaim: ONCE consumer not delivered to\n");
    arts_test_fail();
  } else if (!present(once)) {
    arts_printf("FAIL event_counted_reclaim: ONCE event was reclaimed; "
                "fire-and-linger is what an undeclared event must do\n");
    arts_test_fail();
  }

  if (!arts_test_status()) {
    arts_printf("PASS event_counted_reclaim\n");
  }
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Merge the EDT-side verdict with the runtime's: checks run on worker
     threads, so a test that only printed would stay green through the very
     defect it exists to catch. */
  int rc = arts_rt(argc, argv);
  return rc != 0 ? 1 : arts_test_status();
}
