/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol pure state-transition arbiters (EAGER timing only).
 *
 * Defines: msi_cache_compute_next, msi_dir_compute_next.
 *
 * Both are pure functions of one packed 64-bit word, run inside CAS-retry
 * loops.  eager.c #includes this file to get its private copy; the whitebox
 * unit test (tests/unit/msi_compute_next.c) also #includes it directly.
 * NOT listed in CMakeLists.txt as a standalone TU.
 */
#ifdef ARTS_PROTOCOL_MSI

#include "arts/coherence/msi/types.h"

#include <stdint.h>

/* ===== msi_cache_compute_next ==========================================
 *
 * Cache-side arbiter.  The single-atom shape is load-bearing three ways:
 *   - {count++, decision} in one CAS: a decision can never be derived from
 *     a word state that postdates the acquire's own service (no stale
 *     decision, no dead request).
 *   - Parking IS the decision CAS: self_idx is linked as the new chain head
 *     in the same atom that decides to park, so a parked node can never
 *     miss the fetch it joined — the caller writes node->next = the head it
 *     read from `cur` before every CAS attempt.
 *   - The install-publishing ops take the whole chain in the same atom
 *     (head -> 0): the committer owns exactly its fetch's cohort and serves
 *     it as the continuation of its own transition, however late that runs;
 *     a foreign node would need a purge first, and a purge needs the very
 *     publish this CAS is.
 *
 * A valid-copy read acquire changes nothing (there is no reader count in
 * the word): the arbiter returns `cur` unchanged with SELF_SERVE and the
 * caller skips the CAS entirely — the wait-free read fast path.
 */
uint64_t msi_cache_compute_next(uint64_t cur, int op, uint32_t self_idx,
                                uint32_t *out_action) {
  uint32_t rw = MSI_CACHE_RW(cur);
  uint32_t ro = MSI_CACHE_RO(cur);
  uint32_t wc = MSI_CACHE_WC(cur);
  uint32_t hrw = MSI_CACHE_HEAD_RW(cur);
  uint32_t hro = MSI_CACHE_HEAD_RO(cur);
  uint32_t act = MSI_CACHE_ACT_NONE;
  switch (op) {
  case MSI_CACHE_OP_ACQ_RW:
    wc += 1;
    if (rw == MSI_RW_GRANT) {
      act = MSI_CACHE_ACT_SELF_SERVE; /* join the held write tenure */
    } else if (rw == MSI_RW_IDLE) {
      rw = MSI_RW_REQ;
      hrw = self_idx;
      act = MSI_CACHE_ACT_SEND_RW; /* opened the fetch: park + send */
    } else {                       /* REQ */
      hrw = self_idx;
      act = MSI_CACHE_ACT_PARK; /* chain onto the in-flight fetch */
    }
    break;
  case MSI_CACHE_OP_ACQ_RO:
    if (rw == MSI_RW_GRANT || ro == MSI_RO_VALID) {
      act = MSI_CACHE_ACT_SELF_SERVE; /* pure loads; word unchanged */
    } else if (ro == MSI_RO_IDLE && rw == MSI_RW_IDLE) {
      ro = MSI_RO_REQ;
      hro = self_idx;
      act = MSI_CACHE_ACT_SEND_RO;
    } else {
      /* Mode semantics (universal): an in-flight RW fetch subsumes any RO
       * need (its grant is also a valid copy); an in-flight RO fetch
       * (REQ/REQ_KILL) covers later readers. */
      hro = self_idx;
      act = MSI_CACHE_ACT_PARK;
    }
    break;
  case MSI_CACHE_OP_DELIVER:
    if (ro == MSI_RO_REQ) {
      ro = MSI_RO_VALID;
      hro = 0; /* the committer grabbed HEAD_RO(cur) */
      act = MSI_CACHE_ACT_PUBLISH;
    } else if (ro == MSI_RO_REQ_KILL) {
      /* Reserved invalidate (kill mark): publish + grab exactly like a
       * normal DELIVER; the caller serves the chain, then executes the
       * purge (KILL_PURGE) and the owed ack, in that order. */
      ro = MSI_RO_VALID;
      hro = 0;
      act = MSI_CACHE_ACT_PUBLISH_KILL;
    } else {
      act = MSI_CACHE_ACT_DROP; /* superseded (absorbed by a grant / stale) */
    }
    break;
  case MSI_CACHE_OP_GRANT:
    /* precond (caller-asserted): rw == REQ — home grants only what was
     * requested, and the request opener is chained, so a grant always
     * finds its cohort. */
    rw = MSI_RW_GRANT;
    ro = MSI_RO_VALID;
    hrw = 0; /* both chains grabbed in this one atom */
    hro = 0;
    act = MSI_CACHE_ACT_GRANT_PUBLISH;
    break;
  case MSI_CACHE_OP_REL_RW:
    /* precond (caller-asserted): rw == GRANT && wc > 0. */
    wc -= 1;
    if (wc == 0) {
      /* final = the LAST writer's release, regardless of lingering local
       * readers: the grant demotes to a plain valid copy (the ex-owner
       * keeps the newest data) and ownership returns with this writeback.
       * Readers never delay the return. */
      rw = MSI_RW_IDLE;
      ro = MSI_RO_VALID;
      act = MSI_CACHE_ACT_WB_FINAL;
    } else {
      act = MSI_CACHE_ACT_WB;
    }
    break;
  case MSI_CACHE_OP_INVALIDATE:
    /* precond (caller-asserted): rw != GRANT (a round self-excludes the
     * tenure owner, and rounds serialize — an invalidate can never hit an
     * active write grant). */
    if (ro == MSI_RO_VALID) {
      ro = MSI_RO_IDLE;
      act = MSI_CACHE_ACT_PURGE_ACK;
    } else if (ro == MSI_RO_REQ) {
      ro = MSI_RO_REQ_KILL;
      act = MSI_CACHE_ACT_KILL_MARKED; /* ack owed — the doomed DELIVER
                                          fires it after its serve-once */
    } else { /* IDLE or REQ_KILL: idempotent (roster over-approximation) */
      act = MSI_CACHE_ACT_NOOP_ACK;
    }
    break;
  case MSI_CACHE_OP_KILL_PURGE:
    /* precond (caller-asserted): ro == VALID — between the kill-publish
     * and this purge nothing can retire the copy (the owed ack blocks the
     * round chain, and local activity never leaves VALID). */
    ro = MSI_RO_IDLE;
    act = MSI_CACHE_ACT_NONE; /* the caller sends the owed ack LAST */
    break;
  default:
    break;
  }
  *out_action = act;
  return MSI_CACHE_MAKE(rw, ro, wc, hrw, hro);
}

/* ===== msi_dir_compute_next ============================================
 *
 * Home-side directory arbiter.  Round mutual exclusion, the write-tenure
 * grant claim, and ack accounting all live in this one word:
 *   - the grant claim {w 0-edge && owner NOBODY -> GRANTING} makes the
 *     claiming CAS's committer the sole popper of the request queue
 *     (no empty pop, no double grant);
 *   - ROUND_CLOSE folds {final w--, owner clear, round_open drop, and the
 *     immediate re-claim when writers remain} into one CAS, so no window
 *     exists where a stale actor can double-grant or lose the chain.
 * arg: REQ_RW/CLAIM/ACK_DEC ignore it; ACKS_ARM = the target count;
 * ROUND_CLOSE = the final releaser's rank (MSI_OWNER_NOBODY = no final in
 * the batch).
 */
uint64_t msi_dir_compute_next(uint64_t cur, int op, unsigned int arg,
                              uint32_t *out_action) {
  uint32_t open = MSI_DIR_ROUND_OPEN(cur);
  uint32_t acks = MSI_DIR_ACKS(cur);
  uint32_t owner = MSI_DIR_OWNER(cur);
  uint32_t w = MSI_DIR_W(cur);
  uint32_t act = MSI_DIR_ACT_NONE;
  switch (op) {
  case MSI_DIR_OP_REQ_RW:
    if (w == 0 && owner == MSI_OWNER_NOBODY) {
      owner = MSI_OWNER_GRANTING;
      act = MSI_DIR_ACT_GRANT_CLAIM;
    }
    w += 1;
    break;
  case MSI_DIR_OP_ROUND_CLAIM:
    if (open == 0u) {
      open = 1u;
      act = MSI_DIR_ACT_CLAIMED;
    } /* else: claim failed — another round is in flight (act NONE) */
    break;
  case MSI_DIR_OP_ACKS_ARM:
    /* Under the claim, before the multicast: arm the outstanding-ack count
     * for this round's snapshot (a CAS, not a store — w keeps moving). */
    acks = arg;
    break;
  case MSI_DIR_OP_ACK_DEC:
    /* precond (caller-asserted): acks > 0. */
    acks -= 1;
    if (acks == 0) {
      act = MSI_DIR_ACT_CLOSE; /* this committer closes the round */
    }
    break;
  case MSI_DIR_OP_OWNER_PUBLISH:
    /* precond (caller-asserted): owner == GRANTING — the claim taken by the
     * committer that is now naming the popped requester. */
    owner = arg & (unsigned int)MSI_DIR_OWNER_MASK;
    break;
  case MSI_DIR_OP_ROUND_CLOSE:
    if (arg != (unsigned int)MSI_OWNER_NOBODY && owner == arg) {
      /* the batch's (<=1) final: its release returns the tenure.  The
       * owner == arg guard is load-bearing: a legitimate final always comes
       * from the published owner (a grant is only issued at the w 0-edge or
       * inside a close-fold, so no earlier tenure's final can still be
       * pending when ownership moves).  A final whose releaser is NOT the
       * owner is a surplus creation hold — concurrent creates of the same
       * GUID each seed a local tenure but the directory counts only the one
       * it installed — and must be unit-neutral here, or w underflows and a
       * phantom grant claim fires on an empty request queue. */
      w -= 1;
      owner = MSI_OWNER_NOBODY;
    }
    open = 0u;
    if (owner == MSI_OWNER_NOBODY && w > 0) {
      owner = MSI_OWNER_GRANTING;
      act = MSI_DIR_ACT_GRANT_CLAIM; /* chain: this committer grants next */
    }
    break;
  default:
    break;
  }
  *out_action = act;
  return MSI_DIR_MAKE(open, acks, owner, w);
}

#endif /* ARTS_PROTOCOL_MSI */
