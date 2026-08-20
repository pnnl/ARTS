/* SPDX-License-Identifier: Apache-2.0
 *
 * event_compute_next — the simple-arm event state machine, as a pure function.
 *
 * The whole destroy predicate lives in ONE word so that moving an axis and
 * observing the other commit in the same atom.  Kept apart the two axes are a
 * Dekker pair: each participant writes one and reads the other, and without a
 * StoreLoad fence on both sides they can miss each other and the event is
 * never reclaimed — the exact failure the reclamation exists to prevent.  This
 * is the same discipline as the coherence home/cache arbiters, and it is what
 * makes the decision single-flight by construction rather than by a downstream
 * backstop.
 *
 * Word layout (MSB->LSB):
 *
 *   [ destroyed:1 (63) | fire_st:2 (62..61) | nb_deps:29 (60..32) | latch:32 ]
 *
 *   latch      two's complement, and DELIBERATELY allowed to go negative while
 *              the event has not fired.  A DECR that overtakes its paired INCR
 *              is a debt, not an error: the two are independent sends to the
 *              same home, per-peer order is not a contract, and with two
 *              progress threads they can be dispatched concurrently.  Clamping
 *              at zero forgets the debt and a legal program then never fires.
 *   nb_deps    consumers still expected; meaningful only when `counted`.
 *              Saturates at 0 so an exhausted count cannot wrap and re-arm
 *              the destroy edge.  An over-binding program's extra consumer
 *              arrives after the reclaim and waits forever — the contract
 *              arts_event_hint_t::nb_deps documents.
 *   fire_st    IDLE -> FIRING -> FIRED.  Two phases because claiming the fire
 *              and publishing the data are not one atom: the winner claims
 *              FIRING, writes the payload GUID, then publishes FIRED.  A binder
 *              that sees anything short of FIRED parks, so it can never read a
 *              payload that has not landed.
 *   destroyed  set by the same transition that emits the destroy modifier, so
 *              exactly one committer can ever ask for the object's teardown.
 *
 * `counted` and `auto_destroy` are NOT in the word: both are immutable after
 * creation, so they race with nothing and enter as arguments.  Keeping the
 * flavour out of the counter is load-bearing — a predicate keyed on
 * "nb_deps == 0" is true at birth for every non-counted event, which would
 * destroy ONCE/IDEM/STICKY/LATCH on their first satisfy.
 *
 * ABA on this word is harmless, which is the opposite of the rule for the
 * Treiber stacks next door: the word can return to a previous value (an INCR
 * after a DECR restores `latch`), but the decider is a pure function of the
 * whole state and holds no cached side pointer, so a returned value really is
 * the same state.
 *
 * event.c #includes this file; the whitebox test
 * tests/unit/event_compute_next.c includes it directly to drive the table
 * without a live runtime.  NOT a standalone TU in CMakeLists.txt.
 */

#include "arts/event.h"

#include <stdbool.h>
#include <stdint.h>

/* ===== field access ===================================================== */

#define EV_LATCH_BITS 32
#define EV_NBDEPS_BITS 29
#define EV_NBDEPS_SHIFT 32
#define EV_FIRE_SHIFT 61
#define EV_DESTROYED_SHIFT 63

#define EV_NBDEPS_MAX ((uint32_t)((1u << EV_NBDEPS_BITS) - 1u))

#define EV_LATCH(s) ((int32_t)(uint32_t)((s) & 0xffffffffULL))
#define EV_NBDEPS(s)                                                           \
  ((uint32_t)(((s) >> EV_NBDEPS_SHIFT) & (uint64_t)EV_NBDEPS_MAX))
#define EV_FIRE(s) ((uint32_t)(((s) >> EV_FIRE_SHIFT) & 0x3ULL))
#define EV_DESTROYED(s) ((uint32_t)(((s) >> EV_DESTROYED_SHIFT) & 0x1ULL))

#define EV_MAKE(destroyed, fire, nb_deps, latch)                               \
  (((uint64_t)((destroyed) & 0x1u) << EV_DESTROYED_SHIFT) |                    \
   ((uint64_t)((fire) & 0x3u) << EV_FIRE_SHIFT) |                              \
   (((uint64_t)(nb_deps) & (uint64_t)EV_NBDEPS_MAX) << EV_NBDEPS_SHIFT) |      \
   ((uint64_t)(uint32_t)(latch)))

/* fire_st */
#define EV_FIRE_IDLE 0u
#define EV_FIRE_FIRING 1u
#define EV_FIRE_FIRED 2u

/* ops */
#define EV_OP_SAT_DECR 0
#define EV_OP_SAT_INCR 1
#define EV_OP_PUBLISH 2 /* the FIRING winner has written simple.data */
#define EV_OP_ADD_DEP 3

/* base actions — what the committer must do AFTER the CAS lands */
#define EV_ACT_NONE 0u
#define EV_ACT_FIRE 1u    /* claimed FIRING: publish data, then EV_OP_PUBLISH */
#define EV_ACT_DRAIN 2u   /* payload is readable: drain the parked consumers */
#define EV_ACT_PARK 3u    /* push this dependence onto deps_stack */
#define EV_ACT_DELIVER 4u /* satisfy this dependence now from simple.data */
#define EV_ACT_BASE_MASK 0x7u

/* destroy is a MODIFIER, never a base action.  Making it a base would let it
 * replace PARK or DELIVER, and the case it would replace them in is precisely
 * the one counted events exist for: the last consumer binding after the
 * satisfy.  That consumer would be dropped and its slot never satisfied. */
#define EV_ACT_DESTROY 0x8u

uint64_t event_compute_next(uint64_t cur, int op, bool counted, bool
                            auto_destroy, uint32_t *out_action);

uint64_t event_compute_next(uint64_t cur, int op, bool counted,
                            bool auto_destroy, uint32_t *out_action) {
  int32_t latch = EV_LATCH(cur);
  uint32_t nb = EV_NBDEPS(cur);
  uint32_t fire = EV_FIRE(cur);
  uint32_t destroyed = EV_DESTROYED(cur);
  uint32_t act = EV_ACT_NONE;

  switch (op) {
  case EV_OP_SAT_DECR:
    if (fire == EV_FIRE_FIRED) {
      break; /* absorbed: the debt cannot matter once the event has fired */
    }
    latch -= 1;
    if (latch == 0 && fire == EV_FIRE_IDLE) {
      fire = EV_FIRE_FIRING; /* claim; the winner publishes the payload next */
      act = EV_ACT_FIRE;
    }
    break;

  case EV_OP_SAT_INCR:
    if (fire == EV_FIRE_FIRED) {
      break; /* a fired event never re-arms: firing is once per event, which
              * is what a finish latch's successor asserts of itself */
    }
    latch += 1;
    break;

  case EV_OP_PUBLISH:
    if (fire != EV_FIRE_FIRING) {
      break; /* not ours to publish */
    }
    fire = EV_FIRE_FIRED;
    act = EV_ACT_DRAIN;
    break;

  case EV_OP_ADD_DEP:
    if (counted && nb > 0u) {
      nb -= 1u; /* saturating: an exhausted count never wraps or re-arms */
    }
    /* FIRING is not FIRED: the payload GUID is not readable yet, so park and
     * let the publisher's drain collect this node. */
    act = (fire == EV_FIRE_FIRED) ? EV_ACT_DELIVER : EV_ACT_PARK;
    break;

  default:
    break;
  }

  /* The destroy edge, evaluated once for every transition that could have
   * completed it.  `auto_destroy` keeps its own meaning — destroy on fire,
   * consumers ignored — and is the promise that every consumer had already
   * bound; `counted` is the promise that exactly nb_deps of them will.  They
   * are contradictory, which is why setting both is refused at creation. */
  if (destroyed == 0u && fire == EV_FIRE_FIRED) {
    if (auto_destroy || (counted && nb == 0u)) {
      destroyed = 1u;
      act |= EV_ACT_DESTROY;
    }
  }

  *out_action = act;
  return EV_MAKE(destroyed, fire, nb, latch);
}
