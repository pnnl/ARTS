/* SPDX-License-Identifier: Apache-2.0
 *
 * T292 — ARTS_EVENT_LATCH_INCR_SLOT + explicit latch=N accounting.
 *
 * Target: the latch-counter arithmetic behind arts_event_satisfy_slot's
 * ARTS_EVENT_LATCH_INCR_SLOT (increment) vs ARTS_EVENT_LATCH_DECR_SLOT
 * (decrement), with a non-default initial latch.  The event fires when
 * curr_latch reaches <= 0.  INCR and explicit latch=N are only lightly tested
 * (print-only in event_basic), with no assertion that firing happens EXACTLY
 * once and only after the matching number of decrements.
 *
 * Correct behavior pinned:
 *   - latch = 3, then one INCR (3 -> 4) and four DECR (4 -> 0) -> the wired
 *     dependent fires EXACTLY once.  Fewer than four DECR would leave
 *     curr_latch > 0 and the dependent would not run.
 *   - fire-and-linger: a SECOND dependent added AFTER the event has fired is
 *     satisfied immediately (curr_latch already <= 0), also firing exactly
 * once.
 *   - the total fire count observed across both dependents is exactly 2.
 *
 * An atomic fire-counter DB records each dependent invocation; a finalizer
 * gated on a finish scope reads it back and asserts the exact count, so a
 * premature/duplicate/missed fire is caught as a wrong count (not a hang).
 *
 * Config-agnostic single-node public-API check.
 * exposes_runtime_bug = false (pins INCR/DECR latch accounting + linger).
 */
#include "arts.h"
#include <stdatomic.h>
#include <stdint.h>

static int g_failed = 0;

/* Each wired dependent increments the shared fire counter (depv[0] = ctr RW).
 */
void fire_counter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic unsigned int *ctr = (_Atomic unsigned int *)depv[0].ptr;
  if (ctr) {
    atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
  }
}

/* Finalizer gated on the finish scope: both dependents have run by now. */
void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic unsigned int *ctr = (_Atomic unsigned int *)depv[0].ptr;
  unsigned int n =
      ctr ? atomic_load_explicit(ctr, memory_order_relaxed) : 0xFFFFu;
  if (n != 2u) {
    arts_printf("FAIL api_latch_incr_slot: dependents fired %u times, expected "
                "2 (1 multi-step latch + 1 linger)\n",
                n);
    g_failed = 1;
  } else {
    arts_printf(
        "PASS api_latch_incr_slot: INCR/DECR latch=3 fired once, linger "
        "re-bind fired once (total 2)\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== api_latch_incr_slot ===\n");

  /* Shared atomic fire counter. */
  void *cp = NULL;
  arts_guid_t ctr_db = arts_db_create(&cp, sizeof(_Atomic unsigned int),
                                      ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  atomic_init((_Atomic unsigned int *)cp, 0u);
  arts_db_release(ctr_db, DB_MODE_RW);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Event with explicit initial latch = 3.  Use a labeled GUID so we can both
   * wire dependents and satisfy it, and keep it addressable for the linger
   * re-bind. */
  arts_event_hint_t eh = ARTS_EVENT_HINT_DEFAULTS;
  eh.latch = 3;
  arts_guid_t ev = arts_event_create(&eh);

  /* First dependent: fires when curr_latch reaches <= 0. */
  arts_edt_hint_t dh1 = ARTS_EDT_HINT_DEFAULTS;
  dh1.finish_event = fe;
  arts_guid_t dep1 = arts_edt_create(fire_counter_edt, 0, NULL, 1, &dh1);
  arts_add_dependence(ctr_db, dep1, 0, DB_MODE_RW);
  arts_add_dependence(ev, dep1, 1, DB_MODE_NULL);

  /* One INCR (3 -> 4) then four DECR (4 -> 0) -> fire exactly once. */
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_INCR_SLOT);
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  /* Second dependent registered AFTER the fire: fire-and-linger satisfies it
   * immediately from stored fire state -> fires once more. */
  arts_edt_hint_t dh2 = ARTS_EDT_HINT_DEFAULTS;
  dh2.finish_event = fe;
  arts_guid_t dep2 = arts_edt_create(fire_counter_edt, 0, NULL, 1, &dh2);
  arts_add_dependence(ctr_db, dep2, 0, DB_MODE_RW);
  arts_add_dependence(ev, dep2, 1, DB_MODE_NULL);

  arts_event_destroy(ev);

  /* Finalizer: gated on both dependents draining the finish scope, then reads
   * the fire counter. */
  arts_edt_hint_t ch = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t chk = arts_edt_create(check_edt, 0, NULL, 2, &ch);
  arts_add_dependence(ctr_db, chk, 0, DB_MODE_RO);
  arts_add_dependence(fe, chk, 1, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
