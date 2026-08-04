/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI pure state-transition arbiters: inv_cache_compute_next (reader plane)
 * and inv_dir_compute_next (invalidation round).
 *
 * Two pure functions of one packed 64-bit word each, both run inside a
 * CAS-retry loop by their callers.  Purity is the point: every decision a
 * transition implies is derived from (current word, op) alone, so the
 * committing CAS and the decision it acts on can never disagree — the classic
 * split-decision race (decide on one population, act on another) is
 * unrepresentable.
 *
 * Both are identical under either placement.  Neither knows about ownership:
 * write ownership is the migrating sentinel grant (coherence/grant.c), which
 * lives in plain counters rather than a packed word, so neither arbiter
 * carries rw state or a writer count.
 *
 * The placement TU #includes this file to get its private copy; the whitebox
 * unit tests also #include it directly.  NOT listed in CMakeLists.txt as a
 * standalone TU.
 */
#ifdef ARTS_PROTOCOL_INV

#include "arts/coherence/inv/types.h"

#include <stdbool.h>
#include <stdint.h>

/* ===== inv_cache_compute_next ==========================================
 *
 * The reader plane.  A copy is durable: once ro reaches VALID it survives
 * reader release, so a covering acquire is a pure load — no message, no CAS,
 * no version compare.  A copy dies exactly one way: an INVALIDATE arrives.
 *
 * At most one fetch is open per rank (the inflight flag), and the parked chain
 * belongs to it: parking IS the acquire's own decision CAS, so a parked node
 * can never miss the fetch it joined, and the publishing CAS grabs the whole
 * chain in the same atom, so the committer owns exactly its fetch's cohort and
 * serves it as the continuation of its own transition, however late that runs.
 *
 * self_idx is the caller's pre-written pool node, consumed only by the actions
 * that park it (SEND_RO / PARK).
 */
uint64_t inv_cache_compute_next(uint64_t cur, int op, uint32_t self_idx,
                                uint32_t *out_action) {
  uint32_t ro = MSI_CACHE_RO(cur);
  uint32_t inflight = MSI_CACHE_INFLIGHT(cur);
  uint32_t hro = MSI_CACHE_HEAD_RO(cur);
  uint32_t act = MSI_CACHE_ACT_NONE;

  switch (op) {
  case MSI_CACHE_OP_ACQ_RO:
    if (ro == MSI_RO_VALID) {
      act = MSI_CACHE_ACT_SELF_SERVE; /* the wait-free path */
      break;
    }
    /* No covering copy.  Join the open fetch if there is one, else open it.
     * A kill-marked fetch is joinable too: its doomed reply still serves the
     * chain once before the reserved purge retires the copy, so nobody is
     * stranded and no second fetch is needed. */
    if (inflight != 0u) {
      hro = self_idx;
      act = MSI_CACHE_ACT_PARK;
      break;
    }
    ro = MSI_RO_REQ;
    inflight = 1u;
    hro = self_idx;
    act = MSI_CACHE_ACT_SEND_RO;
    break;

  case MSI_CACHE_OP_DELIVER:
    /* A reply landed.  It belongs to the unique open fetch, so a word with no
     * fetch open means this reply was superseded — discard it. */
    if (inflight == 0u) {
      act = MSI_CACHE_ACT_DROP;
      break;
    }
    inflight = 0u;
    hro = 0u; /* the committer takes the whole chain */
    if (ro == MSI_RO_REQ_KILL) {
      /* An INVALIDATE marked this fetch mid-flight.  Serve the cohort that
       * joined it — they are entitled to these bytes — and only then run the
       * reserved purge and send the ack this rank owes the round. */
      ro = MSI_RO_VALID;
      act = MSI_CACHE_ACT_PUBLISH_KILL;
      break;
    }
    ro = MSI_RO_VALID;
    act = MSI_CACHE_ACT_PUBLISH;
    break;

  case MSI_CACHE_OP_INVALIDATE:
    if (ro == MSI_RO_VALID) {
      ro = MSI_RO_IDLE;
      act = MSI_CACHE_ACT_PURGE_ACK;
      break;
    }
    if (ro == MSI_RO_REQ) {
      /* Mark the in-flight fetch; its reply serves the cohort once, then
       * purges and fires the ack this transition owes the round. */
      ro = MSI_RO_REQ_KILL;
      act = MSI_CACHE_ACT_KILL_MARKED;
      break;
    }
    /* IDLE or already REQ_KILL: idempotent.  The roster over-approximates, and
     * a second invalidate on a kill-marked fetch is unreachable — the ack it
     * owes keeps that round open, and rounds serialize. */
    act = MSI_CACHE_ACT_NOOP_ACK;
    break;

  case MSI_CACHE_OP_KILL_PURGE:
    /* Precondition (caller-asserted): ro == VALID.  Between the kill-publish
     * and this purge nothing can retire the copy — the owed ack blocks the
     * round chain, and local activity never leaves VALID. */
    ro = MSI_RO_IDLE;
    act = MSI_CACHE_ACT_NONE; /* the caller sends the owed ack LAST */
    break;

  default:
    break;
  }
  *out_action = act;
  return MSI_CACHE_MAKE(ro, inflight, hro);
}

/* ===== inv_dir_compute_next ============================================
 *
 * The invalidation round: one round at a time (the round_open claim) and one
 * outstanding-ack count for it.  A round is opened by a release, armed with the
 * size of its roster snapshot, and closed by the ack that drives the count to
 * zero.  That closer is unique, which is what makes it safe for it to wake the
 * releaser and re-open for whatever queued up meanwhile.
 *
 * arg: only ACKS_ARM reads it (the target count).
 */
uint64_t inv_dir_compute_next(uint64_t cur, int op, unsigned int arg,
                              uint32_t *out_action) {
  uint32_t open = MSI_DIR_ROUND_OPEN(cur);
  uint32_t acks = MSI_DIR_ACKS(cur);
  uint32_t act = MSI_DIR_ACT_NONE;

  switch (op) {
  case MSI_DIR_OP_ROUND_CLAIM:
    if (open == 0u) {
      open = 1u;
      act = MSI_DIR_ACT_CLAIMED;
    } /* else: a round is already in flight; its closer re-arms (act NONE) */
    break;

  case MSI_DIR_OP_ACKS_ARM:
    /* Under the claim, before the multicast: arm the outstanding-ack count for
     * this round's snapshot.  A CAS rather than a store because the word is
     * live — acks from this very round can already be arriving. */
    acks = (uint32_t)arg;
    break;

  case MSI_DIR_OP_ACK_DEC:
    /* Precondition (caller-asserted): acks > 0. */
    acks -= 1u;
    if (acks == 0u) {
      act = MSI_DIR_ACT_CLOSE; /* this committer closes the round */
    }
    break;

  case MSI_DIR_OP_ROUND_CLOSE:
    open = 0u;
    break;

  default:
    break;
  }
  *out_action = act;
  return MSI_DIR_MAKE(open, acks);
}

#endif /* ARTS_PROTOCOL_INV */
