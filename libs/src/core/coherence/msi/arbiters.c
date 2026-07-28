/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol pure state-transition arbiters.
 *
 * Defines, per timing arm: msi_cache_compute_next / msi_dir_compute_next
 * (EAGER) or msi_lazy_cache_compute_next / msi_lazy_dir_compute_next (LAZY).
 *
 * All are pure functions of one packed 64-bit word, run inside CAS-retry
 * loops.  The timing TU #includes this file to get its private copy; the
 * whitebox unit tests also #include it directly.  NOT listed in
 * CMakeLists.txt as a standalone TU.
 */
#ifdef ARTS_PROTOCOL_MSI

#include "arts/coherence/msi/types.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef ARTS_TIMING_LAZY

/* The ownership-drop half of a hand-off, shared by the release 0-edge, by an
 * order that lands on an already-idle owner, and by the retry that runs when
 * the confirm gate opens.  ONE atom drops ownership, consumes the pending
 * target and retains a readable copy: the retained copy is what still answers a
 * redirect issued before the directory flip, which is what bounds a bounce
 * retry to the number of migrations.  Writers queued behind the door keep their
 * chain; only ownership leaves, and it is re-requested. */
static bool msi_lazy_migrate_fold(uint32_t *rw, uint32_t *ro, uint32_t *mtp,
                                  uint32_t unc, uint32_t wc, uint32_t wq) {
  if (*rw != MSI_RW_GRANT || *mtp == 0u || wc != 0u || unc != 0u) {
    return false;
  }
  *mtp = 0u;
  *rw = (wq > 0u) ? MSI_RW_REQ : MSI_RW_IDLE;
  *ro = MSI_RO_VALID;
  return true;
}

/* ===== msi_lazy_cache_compute_next =====================================
 *
 * Cache-side arbiter.  Same single-atom discipline as the EAGER arm — the
 * count/decision fold, parking as the decision CAS, whole-chain grabs — over
 * the owner-canonical ownership axis:
 *   - `rw` doubles as the ownership bit and outlives any individual writer:
 *     only the migrate fold clears it;
 *   - an acquire consumes no consistency action at all.  A local write acquire
 *     on an owned copy with an open door is one increment and nothing else;
 *   - a release is pure counting plus that fold — every consistency action of
 *     a release happened before this CAS ran (the caller already collected the
 *     round);
 *   - a read acquire inspects the ownership plane ONLY to notice that owning
 *     the copy implies holding it.  No other read branch reads `rw`, and no
 *     read branch anywhere compares a version: a copy is valid until an
 *     invalidate arrives, full stop.
 *
 * The serve decision is a pure read: a redirected reader is never queued, so
 * it returns `cur` unchanged and only names whether this rank holds bytes.
 * Field widths are contract limits — the caller asserts them; this masks.
 */
uint64_t msi_lazy_cache_compute_next(uint64_t cur, int op, uint32_t arg,
                                     uint32_t *out_action) {
  uint32_t rw = MSI_LAZY_CACHE_RW(cur);
  uint32_t ro = MSI_LAZY_CACHE_RO(cur);
  uint32_t unc = MSI_LAZY_CACHE_UNC(cur);
  uint32_t mtp = MSI_LAZY_CACHE_MTP(cur);
  uint32_t inflight = MSI_LAZY_CACHE_INFLIGHT(cur);
  uint32_t wc = MSI_LAZY_CACHE_WC(cur);
  uint32_t wq = MSI_LAZY_CACHE_WQ(cur);
  uint32_t hro = MSI_LAZY_CACHE_HEAD_RO(cur);
  uint32_t hrw = MSI_LAZY_CACHE_HEAD_RW(cur);
  uint32_t act = MSI_LAZY_CACHE_ACT_NONE;
  switch (op) {
  case MSI_LAZY_CACHE_OP_ACQ_RO:
    if (rw == MSI_RW_GRANT || ro == MSI_RO_VALID) {
      act = MSI_LAZY_CACHE_ACT_SELF_SERVE; /* pure loads; word unchanged */
    } else if (ro == MSI_RO_IDLE && inflight == 0u) {
      /* Opening the fetch and parking self are ONE atom. */
      ro = MSI_RO_REQ;
      inflight = 1u;
      hro = arg;
      act = MSI_LAZY_CACHE_ACT_SEND_RO;
    } else {
      /* A reply is already on the wire (or an open fetch will produce one):
       * coalesce onto its chain.  This is the ONLY thing a reader ever waits
       * for — its own rank's outstanding read.  An ownership request in flight
       * here is deliberately not consulted: a reader must never end up behind
       * a writer, not even a local one. */
      hro = arg;
      act = MSI_LAZY_CACHE_ACT_PARK;
    }
    break;
  case MSI_LAZY_CACHE_OP_ACQ_RW:
    if (rw == MSI_RW_GRANT && mtp == 0u && unc == 0u) {
      wc += 1u;
      act = MSI_LAZY_CACHE_ACT_SELF_SERVE; /* run immediately: no round */
    } else if (rw == MSI_RW_GRANT) {
      /* The door is shut (a migration is pending, or the install is not yet
       * confirmed): joining now would starve the hand-off / write before the
       * directory names this rank. */
      wq += 1u;
      hrw = arg;
      act = MSI_LAZY_CACHE_ACT_PARK;
    } else if (rw == MSI_RW_IDLE) {
      wq += 1u;
      rw = MSI_RW_REQ;
      hrw = arg;
      act = MSI_LAZY_CACHE_ACT_SEND_RW; /* opened the request: park + send */
    } else { /* REQ: one ownership request per rank; coalesce */
      wq += 1u;
      hrw = arg;
      act = MSI_LAZY_CACHE_ACT_PARK;
    }
    break;
  case MSI_LAZY_CACHE_OP_REL_RW:
    /* precond (caller-asserted): rw == GRANT && wc > 0, and both the buffer's
     * version fetch_add and this release's invalidation round already
     * completed.  That order is load-bearing: this CAS can hand the buffer to
     * the next owner, which must therefore find every version this release
     * publishes already stamped on it. */
    wc -= 1u;
    if (msi_lazy_migrate_fold(&rw, &ro, &mtp, unc, wc, wq)) {
      act = MSI_LAZY_CACHE_ACT_MIGRATE;
    }
    break;
  case MSI_LAZY_CACHE_OP_DELIVER:
    /* precond (caller-asserted): inflight — this reply consumes the rank's one
     * open fetch, whatever the state does with the bytes. */
    inflight = 0u;
    if (ro == MSI_RO_REQ) {
      /* A durable copy, ALWAYS — whether or not a writer is running at the
       * server.  The whole chain is grabbed in this same atom and served from
       * the slot's current contents. */
      ro = MSI_RO_VALID;
      hro = 0u;
      act = MSI_LAZY_CACHE_ACT_PUBLISH;
    } else if (ro == MSI_RO_REQ_KILL) {
      /* An invalidate reached this fetch first: serve the chain it collected
       * (nothing published since can be visible to them — the round that
       * marked this fetch is still open on its unpaid ack) and leave no copy.
       * The ack is paid AFTER the serve. */
      ro = MSI_RO_IDLE;
      hro = 0u;
      act = MSI_LAZY_CACHE_ACT_PUBLISH_KILL;
    } else if (ro == MSI_RO_IDLE && hro != 0u) {
      /* Orphaned by an ownership install that has since migrated on, with
       * readers already chained behind it: the same atom opens their fetch so
       * the chain cannot strand. */
      ro = MSI_RO_REQ;
      inflight = 1u;
      act = MSI_LAZY_CACHE_ACT_DROP_REFETCH;
    } else { /* VALID / IDLE: an install or a purge already retired the fetch */
      act = MSI_LAZY_CACHE_ACT_DROP;
    }
    break;
  case MSI_LAZY_CACHE_OP_DELIVER_RW:
    /* precond (caller-asserted): rw == REQ, and the buffer swap precedes this
     * CAS.  The read chains are grabbed and served IMMEDIATELY — a reader must
     * not wait for the directory flip — while the writers stay queued behind
     * the confirm gate.  Taking ro to VALID here is what stops an orphan read
     * reply from putting older bytes over the canonical copy; `inflight` is
     * deliberately untouched, since that reply is still coming. */
    rw = MSI_RW_GRANT;
    ro = MSI_RO_VALID;
    unc = 1u;
    hro = 0u;
    act = MSI_LAZY_CACHE_ACT_INSTALL;
    break;
  case MSI_LAZY_CACHE_OP_CONFIRM_ACK:
    /* Clearing the gate IS this rank's first-store permission, and the same
     * atom admits every writer that queued behind it. */
    if (unc != 0u) {
      unc = 0u;
      wc += wq;
      wq = 0u;
      hrw = 0u;
      act = MSI_LAZY_CACHE_ACT_UNGATE;
    }
    break;
  case MSI_LAZY_CACHE_OP_INV:
    if (rw == MSI_RW_GRANT) {
      /* Ownership supersedes the invalidation: the roster bit that produced
       * it was earned by this rank's reader era.  Purging here would take the
       * canonical bytes with it. */
      act = MSI_LAZY_CACHE_ACT_NOOP_ACK;
    } else if (ro == MSI_RO_VALID) {
      ro = MSI_RO_IDLE;
      act = MSI_LAZY_CACHE_ACT_PURGE_ACK;
    } else if (ro == MSI_RO_REQ) {
      /* Mark the open fetch and OWE the ack.  Withholding it is what pins the
       * round open until the doomed reply has landed and served, so nothing
       * newer can complete while bytes older than it are still being handed
       * out.  It is also why a second invalidate can never find this state. */
      ro = MSI_RO_REQ_KILL;
      act = MSI_LAZY_CACHE_ACT_KILL_OWED;
    } else { /* IDLE: idempotent (the roster over-approximates) */
      act = MSI_LAZY_CACHE_ACT_NOOP_ACK;
    }
    break;
  case MSI_LAZY_CACHE_OP_FWDM:
    /* precond (caller-asserted): the target and its landing are published
     * off-word before this CAS.  Arming and shipping are one atom, so an
     * order that finds the owner already idle needs no second event. */
    mtp = 1u;
    if (msi_lazy_migrate_fold(&rw, &ro, &mtp, unc, wc, wq)) {
      act = MSI_LAZY_CACHE_ACT_MIGRATE;
    }
    break;
  case MSI_LAZY_CACHE_OP_MIGRATE:
    /* A standalone hand-off attempt: the word moves only if the fold fires,
     * so a caller that finds ACT_NONE skips its CAS entirely. */
    if (msi_lazy_migrate_fold(&rw, &ro, &mtp, unc, wc, wq)) {
      act = MSI_LAZY_CACHE_ACT_MIGRATE;
    }
    break;
  case MSI_LAZY_CACHE_OP_SERVE_DECIDE:
    /* Pure decision — the word does not move and no CAS is needed.  Two
     * branches only: this rank either holds bytes (own them or hold a valid
     * copy) and serves them live, or it holds none and bounces the request
     * back through the home.  A bounce is a version-free retry, not a park:
     * nothing is queued or slept on, and each pass re-resolves against a later
     * ownership generation.  Whether a writer is running here is NOT consulted
     * — withholding the copy would push the correctness onto the read side. */
    act = (rw == MSI_RW_GRANT || ro == MSI_RO_VALID)
              ? MSI_LAZY_CACHE_ACT_SERVE
              : MSI_LAZY_CACHE_ACT_BOUNCE;
    break;
  default:
    break;
  }
  *out_action = act;
  return MSI_LAZY_CACHE_MAKE(rw, ro, unc, mtp, inflight, wc, wq, hro, hrw);
}

/* ===== msi_lazy_dir_compute_next =======================================
 *
 * Home-side directory arbiter.  The directory holds no data: it arbitrates
 * ownership migration and invalidation rounds.
 *   - the migration claim folds {w++, claim} into one CAS, so the claiming
 *     committer is the sole forwarder and its decision always accounts for
 *     the requester that made it.  Split in two, the same waiter can be
 *     forwarded by a claim it is not counted in AND by its own request;
 *   - the ownership flip folds {owner, moving drop, w--, re-claim} likewise,
 *     so no window exists in which a queued writer is neither counted nor
 *     forwarded.  The request FIFO is popped BEFORE this CAS: the pop cannot
 *     join the atom, and publishing the flip first lets a concurrent claim
 *     forward the entry that is about to be popped — to itself.
 * arg: CONFIRM_FLIP = the new owner rank; ACKS_ARM = the round's target
 * count; the rest ignore it.
 */
uint64_t msi_lazy_dir_compute_next(uint64_t cur, int op, unsigned int arg,
                                   uint32_t *out_action) {
  uint32_t open = MSI_LAZY_DIR_ROUND_OPEN(cur);
  uint32_t moving = MSI_LAZY_DIR_MOVING(cur);
  uint32_t acks = MSI_LAZY_DIR_ACKS(cur);
  uint32_t owner = MSI_LAZY_DIR_OWNER(cur);
  uint32_t w = MSI_LAZY_DIR_W(cur);
  uint32_t act = MSI_LAZY_DIR_ACT_NONE;
  switch (op) {
  case MSI_LAZY_DIR_OP_REQ_RW:
    /* precond (caller-asserted): the requester is already in the FIFO. */
    w += 1u;
    if (moving == 0u) {
      moving = 1u;
      act = MSI_LAZY_DIR_ACT_MIGRATE_CLAIM;
    }
    break;
  case MSI_LAZY_DIR_OP_CONFIRM_FLIP:
    /* precond (caller-asserted): w > 0 && moving, and this rank's entry is
     * already popped — the confirming rank's own claim is still counted. */
    owner = (uint32_t)arg & (uint32_t)MSI_LAZY_DIR_OWNER_MASK;
    w -= 1u;
    if (w > 0u) {
      moving = 1u; /* chain: this committer forwards the next migration */
      act = MSI_LAZY_DIR_ACT_MIGRATE_CLAIM;
    } else {
      moving = 0u;
    }
    break;
  case MSI_LAZY_DIR_OP_ROUND_CLAIM:
    if (open == 0u) {
      open = 1u;
      act = MSI_LAZY_DIR_ACT_CLAIMED;
    } /* else: claim failed — another round is in flight (act NONE) */
    break;
  case MSI_LAZY_DIR_OP_ACKS_ARM:
    /* Under the claim, before the multicast: arm the outstanding-ack count
     * for this round's snapshot (a CAS, not a store — w keeps moving). */
    acks = (uint32_t)arg;
    break;
  case MSI_LAZY_DIR_OP_ACK_DEC:
    /* precond (caller-asserted): acks > 0. */
    acks -= 1u;
    if (acks == 0u) {
      act = MSI_LAZY_DIR_ACT_CLOSE; /* this committer closes the round */
    }
    break;
  case MSI_LAZY_DIR_OP_ROUND_CLOSE:
    /* precond (caller-asserted): the round's target was captured before this
     * CAS — dropping the claim republishes the slot to the next round. */
    open = 0u;
    break;
  default:
    break;
  }
  *out_action = act;
  return MSI_LAZY_DIR_MAKE(open, moving, acks, owner, w);
}

#else /* EAGER */

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

#endif /* ARTS_TIMING_LAZY */

#endif /* ARTS_PROTOCOL_MSI */
