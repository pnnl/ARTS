/* SPDX-License-Identifier: Apache-2.0
 *
 * RWLOCK protocol pure state-transition arbiters (HOME placement only).
 *
 * Defines: excl_compute_next, cache_compute_next.
 *
 * These arbiters use the HOME single-word layouts (lock_state with
 * [state_bit|w|r] and cache_state with [rw_st|ro_st|wc|rc]).  They are
 * HOME-only and compiled only when ARTS_RELEASE_PURGE is defined.  The OWNER
 * arbiters (lock_owner_compute_next, cache_owner_compute_next) use a different
 * word layout and live in owner.c.
 *
 * home.c #includes this file to get its private copy of the arbiters.
 * The whitebox unit test (tests/unit/excl_compute_next.c) also #includes
 * this file directly for standalone testing of the pure functions.
 *
 * NOT listed in CMakeLists.txt as a standalone TU.
 */
#ifdef ARTS_RELEASE_PURGE

#include "arts/coherence/excl/types.h"

#include <stdint.h>

/* ===== excl_compute_next ================================================
 *
 * Home-side lock-state arbiter.  Pure function of the current packed
 * lock_state word and an op code; returns the next state word and sets
 * *out_grant to one of the LOCK_GRANT_* codes.
 *
 * op codes: LOCK_OP_RW_ACQ / LOCK_OP_RO_ACQ / LOCK_OP_RW_REL / LOCK_OP_RO_REL
 * grant codes: LOCK_GRANT_NONE / LOCK_GRANT_ONE_RW / LOCK_GRANT_ALL_RO
 */
uint64_t excl_compute_next(uint64_t cur, int op, uint32_t *out_grant) {
  uint32_t w = LOCK_STATE_W(cur);
  uint32_t r = LOCK_STATE_R(cur);
  uint32_t bit = LOCK_STATE_BIT(cur);
  uint32_t grant = LOCK_GRANT_NONE;
  switch (op) {
  case LOCK_OP_RW_ACQ:
    /* w+1.  none->rw (w==0 && r==0) grants one RW.  A NEW RW participant
     * (w:0->1) arriving while readers hold the lock (w==0 && r>0) means the RO
     * phase is held and this RW parks → state_bit=RO.  If this rank is ALREADY
     * an RW participant (w>0, the RW phase is in progress, rw->rw) the held
     * writer serves it: state_bit MUST be left unchanged — flipping it to RO
     * here would corrupt the live RW phase into an RO phase.  Hence the guard
     * is `w == 0 && r > 0`, NOT `r > 0`. */
    if (w == 0 && r == 0) {
      grant = LOCK_GRANT_ONE_RW; /* none -> rw */
    } else if (w == 0 && r > 0) {
      bit = LOCK_PHASE_BIT_RO; /* RO held (w was 0), RW waits */
    }
    /* else w>0 (rw->rw): bit unchanged, no grant — the held writer serves it.
     */
    w += 1;
    break;
  case LOCK_OP_RO_ACQ:
    /* r+1.  none->ro / ro->ro drains all RO (D7: new RO grants immediately even
     * with RW waiting); if w>0 (RW held) the RO parks (state_bit=RW). */
    if (w == 0) {
      grant = LOCK_GRANT_ALL_RO; /* none->ro or ro->ro */
      bit = LOCK_PHASE_BIT_RO;
    } else if (bit == LOCK_PHASE_BIT_RO && r > 0) {
      grant = LOCK_GRANT_ALL_RO; /* RO phase extends (D7) */
    } else {
      bit = LOCK_PHASE_BIT_RW; /* RW held, RO waits */
    }
    r += 1;
    break;
  case LOCK_OP_RW_REL:
    /* w-1.  w-1>0 grants next writer (D6); w-1==0 && r>0 flips to RO + drains;
     * w-1==0 && r==0 -> none. */
    w -= 1;
    if (w > 0) {
      grant = LOCK_GRANT_ONE_RW; /* rw->rw */
    } else if (r > 0) {
      grant = LOCK_GRANT_ALL_RO; /* rw->ro */
      bit = LOCK_PHASE_BIT_RO;
    }
    break;
  case LOCK_OP_RO_REL:
    /* r-1.  r-1==0 && w>0 flips to RW + grants one; else nothing. */
    r -= 1;
    if (r == 0 && w > 0) {
      grant = LOCK_GRANT_ONE_RW; /* ro->rw */
      bit = LOCK_PHASE_BIT_RW;
    }
    break;
  default:
    break;
  }
  /* Normalize state_bit when one counter hit 0 (state_bit only meaningful with
   * both > 0). */
  if (w == 0 || r == 0) {
    bit = 0;
  }
  *out_grant = grant;
  return LOCK_MAKE_STATE(bit, w, r);
}

/* ===== cache_compute_next ==============================================
 * Cache-side analogue of excl_compute_next: pure function of the current
 * cache_state word, run inside a CAS-retry loop.  Returns the next word and
 * (via out_action) what the caller must do AFTER the CAS commits.
 *
 * The ACQ_* ops carry the count++ INSIDE the CAS, together with the
 * request/join decision.  That single-atom shape is load-bearing twice over:
 *   - No stale decision: the decision is computed in the same atom that
 *     counts the acquire, so it can never be derived from a word state that
 *     postdates the acquire's own service and release.  (A decision made
 *     after a separately-counted acquire was already served and released
 *     would read post-round IDLE and open a request for a dead acquire —
 *     whose grant no release edge would ever return.)
 *   - Exact grant populations: a phase's counts return to zero at every
 *     0-edge, and an arrival under a held grant joins WITHOUT parking
 *     (SELF_SERVE), so the counts a GRANT_* transition observes are exactly
 *     the parked-waiter population its committer must serve — making the
 *     grant committer the sole drainer, with nothing to serve stale.
 * The REL_* ops carry the count-- inside the CAS, since the decrement and
 * the 0-edge state change must be atomic together. */
uint64_t cache_compute_next(uint64_t cur, int op, uint32_t *out_action) {
  uint32_t rws = CACHE_RW_ST(cur);
  uint32_t ros = CACHE_RO_ST(cur);
  uint32_t wc = CACHE_RW_CNT(cur);
  uint32_t rc = CACHE_RO_CNT(cur);
  uint32_t act = CACHE_ACT_NONE;
  switch (op) {
  case CACHE_OP_ACQ_RW:
    wc += 1;
    if (rws == CACHE_ST_GRANT) {
      act = CACHE_ACT_SELF_SERVE; /* join the held RW phase directly */
    } else if (rws == CACHE_ST_IDLE) {
      rws = CACHE_ST_REQ;
      act = CACHE_ACT_SEND_RW; /* first writer of the round: request home */
    } else {                   /* REQ */
      act = CACHE_ACT_PARK;    /* coalesce onto the in-flight RW request */
    }
    break;
  case CACHE_OP_ACQ_RO:
    rc += 1;
    if (rws == CACHE_ST_GRANT || ros == CACHE_ST_GRANT) {
      act = CACHE_ACT_SELF_SERVE; /* RW⊇RO local join, or RO phase held */
    } else if (rws == CACHE_ST_IDLE && ros == CACHE_ST_IDLE) {
      ros = CACHE_ST_REQ;
      act = CACHE_ACT_SEND_RO;
    } else {
      act = CACHE_ACT_PARK; /* covered by an in-flight RW/RO request */
    }
    break;
  case CACHE_OP_GRANT_RW:
    /* precond: rws==REQ (home grants only what was requested) and wc>=1 (the
     * request opener is counted and cannot have been served before this
     * grant) — so an RW grant always finds its cohort. */
    rws = CACHE_ST_GRANT;
    act = CACHE_ACT_DRAIN_BOTH; /* RW grant serves this rank's RW + RO cohort */
    break;
  case CACHE_OP_GRANT_RO: /* precond: ros==REQ, rws!=GRANT */
    if (rc > 0) {
      ros = CACHE_ST_GRANT;
      act = CACHE_ACT_DRAIN_RO;
    } else {
      /* phantom: the RO waiters were already served by an RW grant (RW⊇RO);
       * this grant has nothing to serve — return it to home at once. */
      ros = CACHE_ST_IDLE;
      act = CACHE_ACT_REL_RO;
    }
    break;
  case CACHE_OP_REL_RW: /* precond: rws==GRANT */
    wc -= 1;
    if (wc == 0 && rc == 0) {
      rws = CACHE_ST_IDLE;
      act = CACHE_ACT_REL_RW; /* last holder of the RW grant → publish */
    }
    break;
  case CACHE_OP_REL_RO:
    rc -= 1;
    if (rws == CACHE_ST_GRANT && wc == 0 && rc == 0) {
      rws = CACHE_ST_IDLE;
      act = CACHE_ACT_REL_RW; /* last RO joiner under an RW grant → publish */
    } else if (ros == CACHE_ST_GRANT && rc == 0) {
      ros = CACHE_ST_IDLE;
      act = CACHE_ACT_REL_RO; /* last holder of the RO grant → notify */
    }
    break;
  default:
    break;
  }
  *out_action = act;
  return CACHE_MAKE(rws, ros, wc, rc);
}
#endif /* ARTS_RELEASE_PURGE */
