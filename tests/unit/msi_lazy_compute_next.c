/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI-LAZY msi_lazy_cache_compute_next / msi_lazy_dir_compute_next — the PURE
 * transition functions over the packed cache word
 * [rw|ro|unc|mtp|inflight|wc|wq|head_ro|head_rw] and the packed directory word
 * [round_open|moving|acks|owner|w].
 *
 * Pins the FULL truth table with explicit, hand-derived expected values (no
 * oracle re-implementing the functions), plus the word-packing arithmetic
 * (compile-time), covering:
 *   cache:
 *   - a covering copy reads with the word UNCHANGED (no CAS at all), for the
 *     owner and for a plain sharer, and the read plane never inspects an
 *     in-flight ownership request — a reader must not end up behind a writer,
 *     not even one on its own rank;
 *   - the in-word chains: parking is the acquire's own decision CAS on both
 *     planes, and every publishing transition grabs its chain in the same atom;
 *   - the inflight gate: while a read reply is on the wire no new fetch opens,
 *     and only the landing CAS clears it — the ownership lane never touches it;
 *   - a local write acquire on an owned copy with an open door: one increment,
 *     no round, no message; and the two things that shut that door (a pending
 *     migration, an unconfirmed install);
 *   - the release 0-edge that folds {wc--, ownership dropped, target consumed,
 *     copy retained} and reports MIGRATE — and the guards that must not;
 *   - the ownership install (read chain grabbed and served at once, writers
 *     held behind the gate) and the ungate that admits them;
 *   - the invalidation alpha-rule, the unconditional purge, the kill mark that
 *     OWES its ack, and the idempotent repeat;
 *   - the read reply on a live fetch, on a killed one, and orphaned;
 *   - the serve decision — two branches, and neither reads a version.
 *   dir:
 *   - the counted migration claim (w++ folded with the claim decision);
 *   - the confirm flip folding {owner, moving drop, w--, chained re-claim};
 *   - round claim mutual exclusion, ack arming/decrement, the 0-edge close
 *     and the close's claim drop.
 *
 * Built standalone by #including coherence/msi/arbiters.c.  MSI+LAZY only;
 * self-skips elsewhere.
 */

#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MSI) || !defined(ARTS_TIMING_LAZY)
int main(void) {
  printf("PASS msi_lazy_compute_next: skipped (MSI+LAZY only; the packed "
         "cache/dir words exist only in that build)\n");
  return 0;
}
#else

#include "arts/coherence/msi/types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── word-packing arithmetic (compile-time pins) ───────────────────────── */
_Static_assert(MSI_LAZY_CACHE_RW_SHIFT + MSI_LAZY_CACHE_ST_BITS == 61,
               "cache word must pack into 61 bits (3 reserved)");
_Static_assert(MSI_LAZY_CACHE_HEAD_RW_SHIFT == 0 &&
                   MSI_LAZY_CACHE_HEAD_RO_SHIFT == 14 &&
                   MSI_LAZY_CACHE_WQ_SHIFT == 28 &&
                   MSI_LAZY_CACHE_WC_SHIFT == 41 &&
                   MSI_LAZY_CACHE_INFLIGHT_SHIFT == 54 &&
                   MSI_LAZY_CACHE_MTP_SHIFT == 55 &&
                   MSI_LAZY_CACHE_UNC_SHIFT == 56 &&
                   MSI_LAZY_CACHE_RO_SHIFT == 57 &&
                   MSI_LAZY_CACHE_RW_SHIFT == 59,
               "cache word field shifts drifted");
_Static_assert(MSI_LAZY_DIR_ROUND_SHIFT + MSI_LAZY_DIR_FLAG_BITS == 64,
               "dir word must pack to exactly 64 bits");
_Static_assert(MSI_LAZY_DIR_W_SHIFT == 10 && MSI_LAZY_DIR_OWNER_SHIFT == 34 &&
                   MSI_LAZY_DIR_ACKS_SHIFT == 48 &&
                   MSI_LAZY_DIR_MOVING_SHIFT == 62 &&
                   MSI_LAZY_DIR_ROUND_SHIFT == 63,
               "dir word field shifts drifted");

/* A counter owns its chain, so every waiter it admits must be addressable by
 * the matching head. */
_Static_assert(MSI_LAZY_CACHE_CNT_MASK <= MSI_LAZY_CACHE_HEAD_MASK,
               "every counted waiter must be addressable by its chain head");
/* The pool's index space is exactly what a head field can name. */
_Static_assert(MSI_WAITER_IDX_MAX == MSI_LAZY_CACHE_HEAD_MASK,
               "waiter index space must match the head field");

#define CACHE_PIN                                                              \
  MSI_LAZY_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_REQ_KILL, 1u, 1u, 1u, 0x1234u,      \
                      0xEDCu, 0x1ABCu, 0xF5u)
_Static_assert(MSI_LAZY_CACHE_RW(CACHE_PIN) == MSI_RW_GRANT,
               "cache rw round-trip");
_Static_assert(MSI_LAZY_CACHE_RO(CACHE_PIN) == MSI_RO_REQ_KILL,
               "cache ro round-trip");
_Static_assert(MSI_LAZY_CACHE_UNC(CACHE_PIN) == 1u, "cache unc round-trip");
_Static_assert(MSI_LAZY_CACHE_MTP(CACHE_PIN) == 1u, "cache mtp round-trip");
_Static_assert(MSI_LAZY_CACHE_INFLIGHT(CACHE_PIN) == 1u,
               "cache inflight round-trip");
_Static_assert(MSI_LAZY_CACHE_WC(CACHE_PIN) == 0x1234u, "cache wc round-trip");
_Static_assert(MSI_LAZY_CACHE_WQ(CACHE_PIN) == 0xEDCu, "cache wq round-trip");
_Static_assert(MSI_LAZY_CACHE_HEAD_RO(CACHE_PIN) == 0x1ABCu,
               "cache head_ro round-trip");
_Static_assert(MSI_LAZY_CACHE_HEAD_RW(CACHE_PIN) == 0xF5u,
               "cache head_rw round-trip");
/* Every field is disjoint: the fields must reconstruct the word they came
 * from. */
_Static_assert(MSI_LAZY_CACHE_MAKE(MSI_LAZY_CACHE_RW(CACHE_PIN),
                                   MSI_LAZY_CACHE_RO(CACHE_PIN),
                                   MSI_LAZY_CACHE_UNC(CACHE_PIN),
                                   MSI_LAZY_CACHE_MTP(CACHE_PIN),
                                   MSI_LAZY_CACHE_INFLIGHT(CACHE_PIN),
                                   MSI_LAZY_CACHE_WC(CACHE_PIN),
                                   MSI_LAZY_CACHE_WQ(CACHE_PIN),
                                   MSI_LAZY_CACHE_HEAD_RO(CACHE_PIN),
                                   MSI_LAZY_CACHE_HEAD_RW(CACHE_PIN)) ==
                   CACHE_PIN,
               "cache word fields must be disjoint");

#define DIR_PIN MSI_LAZY_DIR_MAKE(1u, 1u, 0x2AAAu, 0x1234u, 0xABCDEFu)
_Static_assert(MSI_LAZY_DIR_ROUND_OPEN(DIR_PIN) == 1u, "dir open round-trip");
_Static_assert(MSI_LAZY_DIR_MOVING(DIR_PIN) == 1u, "dir moving round-trip");
_Static_assert(MSI_LAZY_DIR_ACKS(DIR_PIN) == 0x2AAAu, "dir acks round-trip");
_Static_assert(MSI_LAZY_DIR_OWNER(DIR_PIN) == 0x1234u, "dir owner round-trip");
_Static_assert(MSI_LAZY_DIR_W(DIR_PIN) == 0xABCDEFu, "dir w round-trip");
_Static_assert(MSI_OWNER_NOBODY == (uint32_t)MSI_LAZY_DIR_OWNER_MASK,
               "the no-owner sentinel must be out of rank range");

/* The queued node type must satisfy its container contract. */
_Static_assert(offsetof(struct arts_db_msi_round_req_s, link) == 0,
               "round-request link must be the FIRST member");

/* ── truth-table checks ────────────────────────────────────────────────── */

static int g_fail;

#define CK(what, cond)                                                         \
  do {                                                                         \
    if (!(cond)) {                                                             \
      (void)fprintf(stderr, "FAIL msi_lazy_compute_next: %s\n", what);         \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

#define C(rw, ro, unc, mtp, inflight, wc, wq, hro, hrw)                        \
  MSI_LAZY_CACHE_MAKE(rw, ro, unc, mtp, inflight, wc, wq, hro, hrw)
#define D(open, moving, acks, owner, w)                                        \
  MSI_LAZY_DIR_MAKE(open, moving, acks, owner, w)

/* ---- reads: blind hits, the inflight gate, coalescing ------------------ */
static void cache_acquire_ro_table(void) {
  uint32_t act;
  uint64_t nx;

  /* plain sharer copy: word unchanged (wait-free read fast path) */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_ACQ_RO, 7, &act);
  CK("RO on VALID: self-serve", act == MSI_LAZY_CACHE_ACT_SELF_SERVE);
  CK("RO on VALID: word unchanged",
     nx == C(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0, 0, 0, 0, 0));

  /* the owner reads its own canonical copy, writers running or not */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 3, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_ACQ_RO, 7, &act);
  CK("RO under ownership: self-serve", act == MSI_LAZY_CACHE_ACT_SELF_SERVE);
  CK("RO under ownership: word unchanged",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 3, 0, 0, 0));

  /* no copy, nothing in flight: open the fetch, park self, send */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0,
                                     0),
                                   MSI_LAZY_CACHE_OP_ACQ_RO, 7, &act);
  CK("RO opener: send", act == MSI_LAZY_CACHE_ACT_SEND_RO);
  CK("RO opener: REQ + inflight + self chained",
     nx == C(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 1, 0, 0, 7, 0));

  /* an ownership request of THIS rank must not hold the reader up */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_REQ, MSI_RO_IDLE, 0, 0, 0, 0, 1, 0,
                                     4),
                                   MSI_LAZY_CACHE_OP_ACQ_RO, 7, &act);
  CK("RO beside an ownership request: opens its own fetch",
     act == MSI_LAZY_CACHE_ACT_SEND_RO);
  CK("RO beside an ownership request: write plane untouched",
     nx == C(MSI_RW_REQ, MSI_RO_REQ, 0, 0, 1, 0, 1, 7, 4));

  /* a reply is already coming: coalesce onto its chain */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 1, 0, 0, 7,
                                     0),
                                   MSI_LAZY_CACHE_OP_ACQ_RO, 9, &act);
  CK("RO coalesce: park", act == MSI_LAZY_CACHE_ACT_PARK);
  CK("RO coalesce: head is self",
     nx == C(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 1, 0, 0, 9, 0));

  /* killed fetch still in flight: same coalesce */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 1, 0,
                                     0, 7, 0),
                                   MSI_LAZY_CACHE_OP_ACQ_RO, 9, &act);
  CK("RO coalesce on kill-marked: park", act == MSI_LAZY_CACHE_ACT_PARK);
  CK("RO coalesce on kill-marked: head is self",
     nx == C(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 1, 0, 0, 9, 0));

  /* purged copy with the old reply still on the wire: no second fetch */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 1, 0, 0, 0,
                                     0),
                                   MSI_LAZY_CACHE_OP_ACQ_RO, 9, &act);
  CK("RO under an orphaned reply: park", act == MSI_LAZY_CACHE_ACT_PARK);
  CK("RO under an orphaned reply: no new fetch opens",
     nx == C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 1, 0, 0, 9, 0));
}

/* ---- writes: the local run, the door, the request ---------------------- */
static void cache_acquire_rw_table(void) {
  uint32_t act;
  uint64_t nx;

  /* owned, door open: one increment and nothing else */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW on an idle owner: self-serve", act == MSI_LAZY_CACHE_ACT_SELF_SERVE);
  CK("RW on an idle owner: wc++ only",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 1, 0, 0, 0));

  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 2, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW joining live writers: self-serve",
     act == MSI_LAZY_CACHE_ACT_SELF_SERVE);
  CK("RW joining live writers: wc++",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 3, 0, 0, 0));

  /* a pending migration shuts the door: queue for after the hand-off */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 1, 0, 1, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW at a shut door: park", act == MSI_LAZY_CACHE_ACT_PARK);
  CK("RW at a shut door: wq++ + chained",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 1, 0, 1, 1, 0, 4));

  /* an unconfirmed install also holds writers back */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW behind the confirm gate: park", act == MSI_LAZY_CACHE_ACT_PARK);
  CK("RW behind the confirm gate: wq++ + chained",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0, 0, 1, 0, 4));

  /* not owned: open the ownership request */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0,
                                     0),
                                   MSI_LAZY_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW opener: send", act == MSI_LAZY_CACHE_ACT_SEND_RW);
  CK("RW opener: REQ + wq + chained",
     nx == C(MSI_RW_REQ, MSI_RO_IDLE, 0, 0, 0, 0, 1, 0, 4));

  /* a request is in flight: one per rank, coalesce */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_REQ, MSI_RO_IDLE, 0, 0, 0, 0, 1, 0,
                                     4),
                                   MSI_LAZY_CACHE_OP_ACQ_RW, 6, &act);
  CK("RW coalesce: park", act == MSI_LAZY_CACHE_ACT_PARK);
  CK("RW coalesce: wq++ + head is self",
     nx == C(MSI_RW_REQ, MSI_RO_IDLE, 0, 0, 0, 0, 2, 0, 6));

  /* an ownership request over a held read copy leaves the read plane alone */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW opener over VALID: send", act == MSI_LAZY_CACHE_ACT_SEND_RW);
  CK("RW opener over VALID: ro untouched",
     nx == C(MSI_RW_REQ, MSI_RO_VALID, 0, 0, 0, 0, 1, 0, 4));
}

/* ---- release + the hand-off fold --------------------------------------- */
static void cache_release_table(void) {
  uint32_t act;
  uint64_t nx;

  /* other writers remain: count only */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 2, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_REL_RW, 0, &act);
  CK("release with writers left: no hand-off", act == MSI_LAZY_CACHE_ACT_NONE);
  CK("release with writers left: wc--",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 1, 0, 0, 0));

  /* last writer, no migration pending: ownership STAYS (sticky owner) */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 1, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_REL_RW, 0, &act);
  CK("release with no order pending: no hand-off",
     act == MSI_LAZY_CACHE_ACT_NONE);
  CK("release with no order pending: ownership retained",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 0, 0, 0, 0));

  /* last writer with a migration pending: hand off, retain a read copy */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 1, 0, 1, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_REL_RW, 0, &act);
  CK("release 0-edge with an order: migrate",
     act == MSI_LAZY_CACHE_ACT_MIGRATE);
  CK("release 0-edge: ownership dropped, order consumed, copy retained",
     nx == C(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0, 0, 0, 0, 0));

  /* ...and with local writers queued the rank re-requests ownership */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 1, 0, 1, 2,
                                     0, 5),
                                   MSI_LAZY_CACHE_OP_REL_RW, 0, &act);
  CK("release 0-edge with queued writers: migrate",
     act == MSI_LAZY_CACHE_ACT_MIGRATE);
  CK("release 0-edge with queued writers: re-request, chain kept",
     nx == C(MSI_RW_REQ, MSI_RO_VALID, 0, 1 - 1, 0, 0, 2, 0, 5));

  /* the confirm gate blocks the hand-off even at wc == 0 */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 1, 1, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_MIGRATE, 0, &act);
  CK("hand-off under the confirm gate: refused",
     act == MSI_LAZY_CACHE_ACT_NONE);
  CK("hand-off under the confirm gate: word unchanged",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 1, 1, 0, 0, 0, 0, 0));

  /* no order published: nothing to hand off to */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_MIGRATE, 0, &act);
  CK("hand-off with no order: refused", act == MSI_LAZY_CACHE_ACT_NONE);
  CK("hand-off with no order: word unchanged",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 0, 0, 0, 0));

  /* writers still holding: the door is shut but the copy cannot leave yet */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 1, 0, 1, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_MIGRATE, 0, &act);
  CK("hand-off with a live writer: refused", act == MSI_LAZY_CACHE_ACT_NONE);
}

/* ---- a migration order arriving --------------------------------------- */
static void cache_fwdm_table(void) {
  uint32_t act;
  uint64_t nx;

  /* the owner is busy: arm only, the 0-edge ships */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 2, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_FWDM, 0, &act);
  CK("order on a busy owner: armed only", act == MSI_LAZY_CACHE_ACT_NONE);
  CK("order on a busy owner: door shut",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 1, 0, 2, 0, 0, 0));

  /* the owner is idle: arming and shipping are ONE atom */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_FWDM, 0, &act);
  CK("order on an idle owner: migrate", act == MSI_LAZY_CACHE_ACT_MIGRATE);
  CK("order on an idle owner: ownership dropped in the same atom",
     nx == C(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0, 0, 0, 0, 0));

  /* an order that lands during the gate waits for the ungate's retry */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_FWDM, 0, &act);
  CK("order during the gate: armed only", act == MSI_LAZY_CACHE_ACT_NONE);
  CK("order during the gate: door shut, ownership kept",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 1, 1, 0, 0, 0, 0, 0));
}

/* ---- read replies ------------------------------------------------------ */
static void cache_deliver_table(void) {
  uint32_t act;
  uint64_t nx;

  /* live fetch: a DURABLE copy, chain grabbed in the same atom */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 1, 0, 0, 9,
                                     0),
                                   MSI_LAZY_CACHE_OP_DELIVER, 0, &act);
  CK("reply on REQ: publish", act == MSI_LAZY_CACHE_ACT_PUBLISH);
  CK("reply on REQ: VALID + inflight cleared + chain grabbed",
     nx == C(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0, 0, 0, 0, 0));

  /* killed fetch: serve the chain, keep NO copy, and the ack is still owed */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 1, 0,
                                     0, 9, 0),
                                   MSI_LAZY_CACHE_OP_DELIVER, 0, &act);
  CK("reply on a kill mark: publish-kill",
     act == MSI_LAZY_CACHE_ACT_PUBLISH_KILL);
  CK("reply on a kill mark: no copy left, chain grabbed",
     nx == C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0, 0));

  /* orphaned by an ownership install: drop, word only loses inflight */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 1, 1, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_DELIVER, 0, &act);
  CK("reply on an owned copy: drop", act == MSI_LAZY_CACHE_ACT_DROP);
  CK("reply on an owned copy: only the fetch flag clears",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 1, 0, 0, 0));

  /* orphaned and purged, nobody waiting: plain drop */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 1, 0, 0, 0,
                                     0),
                                   MSI_LAZY_CACHE_OP_DELIVER, 0, &act);
  CK("reply on IDLE with no waiters: drop", act == MSI_LAZY_CACHE_ACT_DROP);
  CK("reply on IDLE with no waiters: fetch flag clears",
     nx == C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0, 0));

  /* orphaned and purged with readers chained behind it: reopen in the same
   * atom so the chain cannot strand */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 1, 0, 0, 9,
                                     0),
                                   MSI_LAZY_CACHE_OP_DELIVER, 0, &act);
  CK("reply on IDLE with waiters: drop + refetch",
     act == MSI_LAZY_CACHE_ACT_DROP_REFETCH);
  CK("reply on IDLE with waiters: fetch reopened, chain kept",
     nx == C(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 1, 0, 0, 9, 0));
}

/* ---- ownership arriving, and the gate opening ------------------------- */
static void cache_ownership_table(void) {
  uint32_t act;
  uint64_t nx;

  /* readers are served AT ONCE; the writers stay behind the gate */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_REQ, MSI_RO_REQ, 0, 0, 1, 0, 2, 9,
                                     6),
                                   MSI_LAZY_CACHE_OP_DELIVER_RW, 0, &act);
  CK("ownership install: install", act == MSI_LAZY_CACHE_ACT_INSTALL);
  CK("ownership install: owned + VALID + RO chain grabbed + gate shut, "
     "the read reply still in flight and the writers still queued",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 1, 0, 2, 0, 6));

  /* ...and a kill-marked fetch is resolved by owning the copy outright */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_REQ, MSI_RO_REQ_KILL, 0, 0, 1, 0, 1,
                                     9, 6),
                                   MSI_LAZY_CACHE_OP_DELIVER_RW, 0, &act);
  CK("ownership install over a kill mark: install",
     act == MSI_LAZY_CACHE_ACT_INSTALL);
  CK("ownership install over a kill mark: VALID, chain grabbed",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 1, 0, 1, 0, 6));

  /* the flip is confirmed: admit every queued writer in one atom */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0, 0, 3,
                                     0, 6),
                                   MSI_LAZY_CACHE_OP_CONFIRM_ACK, 0, &act);
  CK("confirm ack: ungate", act == MSI_LAZY_CACHE_ACT_UNGATE);
  CK("confirm ack: gate open, queue promoted, chain grabbed",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 3, 0, 0, 0));

  /* an ack with no gate standing is inert */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 1, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_CONFIRM_ACK, 0, &act);
  CK("confirm ack with no gate: none", act == MSI_LAZY_CACHE_ACT_NONE);
  CK("confirm ack with no gate: word unchanged",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 1, 0, 0, 0));
}

/* ---- invalidation ------------------------------------------------------ */
static void cache_invalidate_table(void) {
  uint32_t act;
  uint64_t nx;

  /* ownership supersedes the roster's over-approximation */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 1, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_INV, 0, &act);
  CK("invalidate on the owner: no-op ack", act == MSI_LAZY_CACHE_ACT_NOOP_ACK);
  CK("invalidate on the owner: canonical bytes kept",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 1, 0, 0, 0));

  /* a copy dies unconditionally — no version is consulted */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_INV, 0, &act);
  CK("invalidate on a copy: purge + ack", act == MSI_LAZY_CACHE_ACT_PURGE_ACK);
  CK("invalidate on a copy: gone",
     nx == C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0, 0));

  /* an open fetch is marked and the ack is WITHHELD */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 1, 0, 0, 9,
                                     0),
                                   MSI_LAZY_CACHE_OP_INV, 0, &act);
  CK("invalidate on an open fetch: marked, ack owed",
     act == MSI_LAZY_CACHE_ACT_KILL_OWED);
  CK("invalidate on an open fetch: kill mark, chain untouched",
     nx == C(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 1, 0, 0, 9, 0));

  /* nothing here: idempotent (the roster over-approximates) */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0,
                                     0),
                                   MSI_LAZY_CACHE_OP_INV, 0, &act);
  CK("invalidate on nothing: no-op ack", act == MSI_LAZY_CACHE_ACT_NOOP_ACK);
  CK("invalidate on nothing: word unchanged",
     nx == C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0, 0));
}

/* ---- the serve decision ------------------------------------------------ */
static void cache_serve_table(void) {
  uint32_t act;
  uint64_t nx;

  /* the owner serves live bytes — a running writer does NOT change that */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 2, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_SERVE_DECIDE, 0, &act);
  CK("serve at a writing owner: serve", act == MSI_LAZY_CACHE_ACT_SERVE);
  CK("serve decision writes nothing",
     nx == C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 2, 0, 0, 0));

  nx = msi_lazy_cache_compute_next(C(MSI_RW_GRANT, MSI_RO_VALID, 0, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_SERVE_DECIDE, 0, &act);
  CK("serve at an idle owner: serve", act == MSI_LAZY_CACHE_ACT_SERVE);

  /* an ex-owner that retained its copy answers a stale redirect */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0, 0, 0,
                                     0, 0),
                                   MSI_LAZY_CACHE_OP_SERVE_DECIDE, 0, &act);
  CK("serve at a retaining ex-owner: serve", act == MSI_LAZY_CACHE_ACT_SERVE);

  /* no bytes here: hand the request back to the home */
  nx = msi_lazy_cache_compute_next(C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0,
                                     0),
                                   MSI_LAZY_CACHE_OP_SERVE_DECIDE, 0, &act);
  CK("serve with no bytes: bounce", act == MSI_LAZY_CACHE_ACT_BOUNCE);
  CK("bounce writes nothing",
     nx == C(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0, 0, 0, 0, 0));

  nx = msi_lazy_cache_compute_next(C(MSI_RW_REQ, MSI_RO_REQ, 0, 0, 1, 0, 1, 3,
                                     4),
                                   MSI_LAZY_CACHE_OP_SERVE_DECIDE, 0, &act);
  CK("serve while still fetching: bounce", act == MSI_LAZY_CACHE_ACT_BOUNCE);
}

/* ---- the directory ----------------------------------------------------- */
static void dir_table(void) {
  uint32_t act;
  uint64_t nx;

  /* first ownership request: count AND claim in one CAS */
  nx = msi_lazy_dir_compute_next(D(0, 0, 0, 5, 0), MSI_LAZY_DIR_OP_REQ_RW, 0,
                                 &act);
  CK("request: migration claim", act == MSI_LAZY_DIR_ACT_MIGRATE_CLAIM);
  CK("request: w++ and moving in one atom", nx == D(0, 1, 0, 5, 1));

  /* a second request while one migration is unconfirmed: count only */
  nx = msi_lazy_dir_compute_next(D(0, 1, 0, 5, 1), MSI_LAZY_DIR_OP_REQ_RW, 0,
                                 &act);
  CK("request during a migration: no claim", act == MSI_LAZY_DIR_ACT_NONE);
  CK("request during a migration: w++ only", nx == D(0, 1, 0, 5, 2));

  /* the flip: owner published, claim released, nobody else waiting */
  nx = msi_lazy_dir_compute_next(D(0, 1, 0, 5, 1),
                                 MSI_LAZY_DIR_OP_CONFIRM_FLIP, 7, &act);
  CK("confirm with no successor: no chain", act == MSI_LAZY_DIR_ACT_NONE);
  CK("confirm with no successor: owner flipped, moving dropped",
     nx == D(0, 0, 0, 7, 0));

  /* the flip with a queue behind it: the same CAS re-claims and chains */
  nx = msi_lazy_dir_compute_next(D(0, 1, 0, 5, 2),
                                 MSI_LAZY_DIR_OP_CONFIRM_FLIP, 7, &act);
  CK("confirm with a successor: chained claim",
     act == MSI_LAZY_DIR_ACT_MIGRATE_CLAIM);
  CK("confirm with a successor: owner flipped, moving re-claimed",
     nx == D(0, 1, 0, 7, 1));

  /* the flip must not disturb an open round */
  nx = msi_lazy_dir_compute_next(D(1, 1, 3, 5, 1),
                                 MSI_LAZY_DIR_OP_CONFIRM_FLIP, 7, &act);
  CK("confirm during a round: round state untouched",
     nx == D(1, 0, 3, 7, 0));

  /* round claim mutual exclusion */
  nx = msi_lazy_dir_compute_next(D(0, 0, 0, 5, 0),
                                 MSI_LAZY_DIR_OP_ROUND_CLAIM, 0, &act);
  CK("round claim: claimed", act == MSI_LAZY_DIR_ACT_CLAIMED);
  CK("round claim: open", nx == D(1, 0, 0, 5, 0));

  nx = msi_lazy_dir_compute_next(D(1, 0, 2, 5, 0),
                                 MSI_LAZY_DIR_OP_ROUND_CLAIM, 0, &act);
  CK("round claim while open: refused", act == MSI_LAZY_DIR_ACT_NONE);
  CK("round claim while open: word unchanged", nx == D(1, 0, 2, 5, 0));

  /* arming the ack count under the claim */
  nx = msi_lazy_dir_compute_next(D(1, 0, 0, 5, 1), MSI_LAZY_DIR_OP_ACKS_ARM, 3,
                                 &act);
  CK("acks arm: no action", act == MSI_LAZY_DIR_ACT_NONE);
  CK("acks arm: count set, everything else kept", nx == D(1, 0, 3, 5, 1));

  /* an empty round arms zero and closes without a multicast */
  nx = msi_lazy_dir_compute_next(D(1, 0, 5, 5, 0), MSI_LAZY_DIR_OP_ACKS_ARM, 0,
                                 &act);
  CK("acks arm to zero", nx == D(1, 0, 0, 5, 0));

  /* collecting */
  nx = msi_lazy_dir_compute_next(D(1, 0, 3, 5, 0), MSI_LAZY_DIR_OP_ACK_DEC, 0,
                                 &act);
  CK("ack: not the last", act == MSI_LAZY_DIR_ACT_NONE);
  CK("ack: count down", nx == D(1, 0, 2, 5, 0));

  nx = msi_lazy_dir_compute_next(D(1, 0, 1, 5, 0), MSI_LAZY_DIR_OP_ACK_DEC, 0,
                                 &act);
  CK("last ack: close", act == MSI_LAZY_DIR_ACT_CLOSE);
  CK("last ack: count reaches zero, the round is still open",
     nx == D(1, 0, 0, 5, 0));

  /* the close drops only the claim */
  nx = msi_lazy_dir_compute_next(D(1, 1, 0, 5, 2), MSI_LAZY_DIR_OP_ROUND_CLOSE,
                                 0, &act);
  CK("close: no action", act == MSI_LAZY_DIR_ACT_NONE);
  CK("close: claim dropped, migration state untouched",
     nx == D(0, 1, 0, 5, 2));
}

/* Pull in the arbiters (defines the two pure functions). */
#include "core/coherence/msi/arbiters.c"

int main(void) {
  cache_acquire_ro_table();
  cache_acquire_rw_table();
  cache_release_table();
  cache_fwdm_table();
  cache_deliver_table();
  cache_ownership_table();
  cache_invalidate_table();
  cache_serve_table();
  dir_table();
  if (g_fail) {
    return 1;
  }
  printf("PASS msi_lazy_compute_next: cache + directory truth tables\n");
  return 0;
}

#endif /* ARTS_PROTOCOL_MSI && ARTS_TIMING_LAZY */
