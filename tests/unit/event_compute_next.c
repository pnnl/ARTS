/* SPDX-License-Identifier: Apache-2.0
 *
 * event_compute_next — whitebox table test for the simple-arm event decider.
 *
 * The decider is a pure function of the packed state word, so the state
 * machine can be driven directly instead of only through a live runtime.  The
 * cases below are the ones an adversarial review of the design named as its
 * failure modes; each is here because getting it wrong is silent:
 *
 *   ONCE self-destroy      a predicate keyed on "no consumers expected" is
 *                          true at birth for every non-counted flavour
 *   re-fire on re-arm      an INCR that re-arms a fired latch must not run the
 *                          fire path again (a finish latch's successor asserts
 *                          it ran exactly once)
 *   forgotten debt         a DECR that overtakes its INCR must stay negative;
 *                          clamping makes a legal program never fire
 *   dropped last consumer  destroy must MODIFY the deliver, never replace it —
 *                          the late final bind is what counted events are for
 *   unpublished payload    a binder seeing the fire CLAIMED but not published
 *                          must park, or it delivers a stale/NULL GUID
 *
 * Standalone: includes the decider directly, links nothing.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "../../libs/src/core/event_arbiter.c"
static int fails;
static void chk(const char *what, int got, int want) {
  if (got != want) { printf("FAIL %s: got %d want %d\n", what, got, want); fails++; }
}
int main(void) {
  uint32_t a; uint64_t s;
  /* ONCE (not counted, not auto): fires once, never destroys */
  s = EV_MAKE(0,EV_FIRE_IDLE,0,1);
  s = event_compute_next(s, EV_OP_SAT_DECR, false, false, &a);
  chk("once/fire", a & EV_ACT_BASE_MASK, EV_ACT_FIRE);
  s = event_compute_next(s, EV_OP_PUBLISH, false, false, &a);
  chk("once/drain", a & EV_ACT_BASE_MASK, EV_ACT_DRAIN);
  chk("once/no-destroy", (a & EV_ACT_DESTROY) != 0, 0);
  /* re-arm after fire must NOT re-fire */
  s = event_compute_next(s, EV_OP_SAT_INCR, false, false, &a);
  chk("rearm/no-act", a, EV_ACT_NONE);
  s = event_compute_next(s, EV_OP_SAT_DECR, false, false, &a);
  chk("rearm/no-refire", a, EV_ACT_NONE);
  /* DECR before INCR keeps the debt: 0,-1,0,1,0 must fire exactly at the end */
  s = EV_MAKE(0,EV_FIRE_IDLE,0,0);
  s = event_compute_next(s, EV_OP_SAT_DECR, false, false, &a);
  chk("debt/latch-neg", EV_LATCH(s), -1); chk("debt/no-fire", a, EV_ACT_NONE);
  s = event_compute_next(s, EV_OP_SAT_INCR, false, false, &a);
  s = event_compute_next(s, EV_OP_SAT_INCR, false, false, &a);
  chk("debt/latch1", EV_LATCH(s), 1);
  s = event_compute_next(s, EV_OP_SAT_DECR, false, false, &a);
  chk("debt/fires", a & EV_ACT_BASE_MASK, EV_ACT_FIRE);
  /* COUNTED(1): last consumer binds AFTER the satisfy -> DELIVER *and* destroy */
  s = EV_MAKE(0,EV_FIRE_IDLE,1,1);
  s = event_compute_next(s, EV_OP_SAT_DECR, true, false, &a);
  s = event_compute_next(s, EV_OP_PUBLISH,  true, false, &a);
  chk("counted/no-early-destroy", (a & EV_ACT_DESTROY) != 0, 0);
  s = event_compute_next(s, EV_OP_ADD_DEP, true, false, &a);
  chk("counted/deliver-kept", a & EV_ACT_BASE_MASK, EV_ACT_DELIVER);
  chk("counted/destroy-too", (a & EV_ACT_DESTROY) != 0, 1);
  /* COUNTED(1): consumer binds BEFORE the satisfy -> parks, destroy at publish */
  s = EV_MAKE(0,EV_FIRE_IDLE,1,1);
  s = event_compute_next(s, EV_OP_ADD_DEP, true, false, &a);
  chk("counted/park", a & EV_ACT_BASE_MASK, EV_ACT_PARK);
  chk("counted/park-no-destroy", (a & EV_ACT_DESTROY) != 0, 0);
  s = event_compute_next(s, EV_OP_SAT_DECR, true, false, &a);
  s = event_compute_next(s, EV_OP_PUBLISH,  true, false, &a);
  chk("counted/drain+destroy", a, EV_ACT_DRAIN | EV_ACT_DESTROY);
  /* destroy is emitted at most once */
  s = event_compute_next(s, EV_OP_ADD_DEP, true, false, &a);
  chk("counted/destroy-once", (a & EV_ACT_DESTROY) != 0, 0);
  /* FIRING is not FIRED: a binder must park, never read an unpublished payload */
  s = EV_MAKE(0,EV_FIRE_IDLE,0,1);
  s = event_compute_next(s, EV_OP_SAT_DECR, false, false, &a);
  s = event_compute_next(s, EV_OP_ADD_DEP, false, false, &a);
  chk("firing/parks", a & EV_ACT_BASE_MASK, EV_ACT_PARK);
  printf(fails ? "FAILURES: %d\n" : "all decider cases pass\n", fails);
  return fails != 0;
}
