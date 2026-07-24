/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI msi_cache_compute_next / msi_dir_compute_next — the PURE transition
 * functions over the packed cache word [rw|ro|wc|head_rw|head_ro] and the
 * packed directory word [round_open|acks|owner|w].
 *
 * Pins the FULL truth table with explicit, hand-derived expected values (no
 * oracle re-implementing the functions), plus the word-packing arithmetic
 * (compile-time), covering:
 *   cache:
 *   - valid-copy read = SELF_SERVE with the word UNCHANGED (the wait-free
 *     read fast path: no CAS at all);
 *   - park is the decision CAS (self_idx becomes the chain head in the same
 *     atom, for both the fetch opener and the coalescers);
 *   - publish/grant grab the whole chain(s) in the same atom (head -> 0);
 *   - the kill mark (REQ -> REQ_KILL) and its serve-once + purge tail;
 *   - the last writer's release demotes GRANT -> plain VALID (readers never
 *     delay the ownership return) and flags the final writeback;
 *   dir:
 *   - the grant claim fires exactly on {w 0-edge && owner NOBODY};
 *   - round claim mutual exclusion;
 *   - ack arming/decrement and the 0-edge close;
 *   - ROUND_CLOSE folding {final w--, owner clear, reopen-claim} into one
 *     transition.
 *
 * Built standalone by #including coherence/msi/arbiters.c.  MSI-only;
 * self-skips elsewhere.
 */

#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MSI)
int main(void) {
  printf("PASS msi_compute_next: skipped (MSI-only; the packed cache/dir "
         "words exist only in the MSI build)\n");
  return 0;
}
#else

#include "arts/coherence/msi/types.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── word-packing arithmetic (compile-time pins) ───────────────────────── */
_Static_assert(MSI_CACHE_RW_SHIFT + MSI_CACHE_ST_BITS == 64,
               "cache word must pack to exactly 64 bits");
_Static_assert(MSI_CACHE_HEAD_RO_SHIFT == 0 && MSI_CACHE_HEAD_RW_SHIFT == 18 &&
                   MSI_CACHE_WC_SHIFT == 36 && MSI_CACHE_RO_SHIFT == 60 &&
                   MSI_CACHE_RW_SHIFT == 62,
               "cache word field shifts drifted");
_Static_assert(MSI_DIR_ROUND_SHIFT == 63 && MSI_DIR_ACKS_SHIFT == 49 &&
                   MSI_DIR_OWNER_SHIFT == 35 && MSI_DIR_W_SHIFT == 11,
               "dir word field shifts drifted");
_Static_assert(MSI_CACHE_RW(MSI_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_REQ_KILL,
                                           0xABCDEFu, 0x2AAAAu, 0x15555u)) ==
                   MSI_RW_GRANT,
               "cache rw round-trip");
_Static_assert(MSI_CACHE_RO(MSI_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_REQ_KILL,
                                           0xABCDEFu, 0x2AAAAu, 0x15555u)) ==
                   MSI_RO_REQ_KILL,
               "cache ro round-trip");
_Static_assert(MSI_CACHE_WC(MSI_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_REQ_KILL,
                                           0xABCDEFu, 0x2AAAAu, 0x15555u)) ==
                   0xABCDEFu,
               "cache wc round-trip");
_Static_assert(MSI_CACHE_HEAD_RW(MSI_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_REQ_KILL,
                                                0xABCDEFu, 0x2AAAAu,
                                                0x15555u)) == 0x2AAAAu,
               "cache head_rw round-trip");
_Static_assert(MSI_CACHE_HEAD_RO(MSI_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_REQ_KILL,
                                                0xABCDEFu, 0x2AAAAu,
                                                0x15555u)) == 0x15555u,
               "cache head_ro round-trip");
_Static_assert(MSI_DIR_ROUND_OPEN(MSI_DIR_MAKE(1u, 0x1FFFu,
                                               MSI_OWNER_GRANTING,
                                               0x9ABCDEu)) == 1u,
               "dir round-trip: round_open");
_Static_assert(MSI_DIR_ACKS(MSI_DIR_MAKE(1u, 0x1FFFu, MSI_OWNER_GRANTING,
                                         0x9ABCDEu)) == 0x1FFFu,
               "dir round-trip: acks");
_Static_assert(MSI_DIR_OWNER(MSI_DIR_MAKE(1u, 0x1FFFu, MSI_OWNER_GRANTING,
                                          0x9ABCDEu)) == MSI_OWNER_GRANTING,
               "dir round-trip: owner (GRANTING sentinel)");
_Static_assert(MSI_DIR_W(MSI_DIR_MAKE(1u, 0x1FFFu, MSI_OWNER_GRANTING,
                                      0x9ABCDEu)) == 0x9ABCDEu,
               "dir round-trip: w");
_Static_assert(MSI_OWNER_NOBODY != MSI_OWNER_GRANTING,
               "owner sentinels must be distinct");

/* The wb-queue entry must satisfy the Treiber-stack contract. */
_Static_assert(offsetof(struct arts_db_msi_wb_s, link) == 0,
               "wb entry link must be the FIRST member");

/* ── truth-table checks ────────────────────────────────────────────────── */

static int g_fail;

#define CK(what, cond)                                                         \
  do {                                                                         \
    if (!(cond)) {                                                             \
      (void)fprintf(stderr, "FAIL msi_compute_next: %s\n", what);              \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

#define CACHE(rw, ro, wc, hrw, hro) MSI_CACHE_MAKE(rw, ro, wc, hrw, hro)
#define DIR(open, acks, owner, w) MSI_DIR_MAKE(open, acks, owner, w)

static void cache_table(void) {
  uint32_t act;
  uint64_t nx;

  /* -- ACQ_RO ---------------------------------------------------------- */
  /* valid copy: word unchanged (wait-free read fast path) */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0),
                              MSI_CACHE_OP_ACQ_RO, 7, &act);
  CK("RO on VALID: self-serve", act == MSI_CACHE_ACT_SELF_SERVE);
  CK("RO on VALID: word unchanged",
     nx == CACHE(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0));
  /* under a held write grant: also a pure local hit */
  nx = msi_cache_compute_next(CACHE(MSI_RW_GRANT, MSI_RO_VALID, 2, 0, 0),
                              MSI_CACHE_OP_ACQ_RO, 7, &act);
  CK("RO under GRANT: self-serve", act == MSI_CACHE_ACT_SELF_SERVE);
  CK("RO under GRANT: word unchanged",
     nx == CACHE(MSI_RW_GRANT, MSI_RO_VALID, 2, 0, 0));
  /* both idle: open the fetch, park self, send */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0),
                              MSI_CACHE_OP_ACQ_RO, 7, &act);
  CK("RO opener: send", act == MSI_CACHE_ACT_SEND_RO);
  CK("RO opener: REQ + self chained",
     nx == CACHE(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 7));
  /* fetch in flight: coalesce (park, chain onto the head) */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 7),
                              MSI_CACHE_OP_ACQ_RO, 9, &act);
  CK("RO coalesce: park", act == MSI_CACHE_ACT_PARK);
  CK("RO coalesce: head is self",
     nx == CACHE(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 9));
  /* killed fetch still in flight: same coalesce */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 7),
                              MSI_CACHE_OP_ACQ_RO, 9, &act);
  CK("RO coalesce on kill-marked: park", act == MSI_CACHE_ACT_PARK);
  CK("RO coalesce on kill-marked: head is self",
     nx == CACHE(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 9));
  /* RW fetch in flight subsumes the read: park on the RO chain */
  nx = msi_cache_compute_next(CACHE(MSI_RW_REQ, MSI_RO_IDLE, 1, 4, 0),
                              MSI_CACHE_OP_ACQ_RO, 9, &act);
  CK("RO under RW fetch: park", act == MSI_CACHE_ACT_PARK);
  CK("RO under RW fetch: chained",
     nx == CACHE(MSI_RW_REQ, MSI_RO_IDLE, 1, 4, 9));

  /* -- ACQ_RW ---------------------------------------------------------- */
  /* held grant: join (wc++ only) */
  nx = msi_cache_compute_next(CACHE(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0),
                              MSI_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW join: self-serve", act == MSI_CACHE_ACT_SELF_SERVE);
  CK("RW join: wc++", nx == CACHE(MSI_RW_GRANT, MSI_RO_VALID, 2, 0, 0));
  /* idle: open the fetch */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0),
                              MSI_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW opener: send", act == MSI_CACHE_ACT_SEND_RW);
  CK("RW opener: REQ + wc + chained",
     nx == CACHE(MSI_RW_REQ, MSI_RO_IDLE, 1, 4, 0));
  /* fetch in flight: coalesce */
  nx = msi_cache_compute_next(CACHE(MSI_RW_REQ, MSI_RO_IDLE, 1, 4, 0),
                              MSI_CACHE_OP_ACQ_RW, 6, &act);
  CK("RW coalesce: park", act == MSI_CACHE_ACT_PARK);
  CK("RW coalesce: wc++ + head is self",
     nx == CACHE(MSI_RW_REQ, MSI_RO_IDLE, 2, 6, 0));
  /* RW opener while a read copy is valid: read plane untouched */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0),
                              MSI_CACHE_OP_ACQ_RW, 4, &act);
  CK("RW opener over VALID: send", act == MSI_CACHE_ACT_SEND_RW);
  CK("RW opener over VALID: ro untouched",
     nx == CACHE(MSI_RW_REQ, MSI_RO_VALID, 1, 4, 0));

  /* -- DELIVER --------------------------------------------------------- */
  /* live fetch: publish + whole-chain grab in the same atom */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 9),
                              MSI_CACHE_OP_DELIVER, 0, &act);
  CK("DELIVER on REQ: publish", act == MSI_CACHE_ACT_PUBLISH);
  CK("DELIVER on REQ: VALID + chain grabbed",
     nx == CACHE(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0));
  /* killed fetch: serve-once (publish + grab), purge tail follows */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 9),
                              MSI_CACHE_OP_DELIVER, 0, &act);
  CK("DELIVER on kill mark: publish-kill", act == MSI_CACHE_ACT_PUBLISH_KILL);
  CK("DELIVER on kill mark: VALID + chain grabbed",
     nx == CACHE(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0));
  /* superseded (grant absorbed the fetch): drop, word unchanged */
  nx = msi_cache_compute_next(CACHE(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0),
                              MSI_CACHE_OP_DELIVER, 0, &act);
  CK("DELIVER on VALID: drop", act == MSI_CACHE_ACT_DROP);
  CK("DELIVER on VALID: word unchanged",
     nx == CACHE(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0));
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0),
                              MSI_CACHE_OP_DELIVER, 0, &act);
  CK("DELIVER on IDLE: drop", act == MSI_CACHE_ACT_DROP);

  /* -- GRANT ----------------------------------------------------------- */
  /* both chains grabbed with the states in ONE atom */
  nx = msi_cache_compute_next(CACHE(MSI_RW_REQ, MSI_RO_REQ, 2, 6, 9),
                              MSI_CACHE_OP_GRANT, 0, &act);
  CK("GRANT: publish", act == MSI_CACHE_ACT_GRANT_PUBLISH);
  CK("GRANT: GRANT+VALID + both chains grabbed",
     nx == CACHE(MSI_RW_GRANT, MSI_RO_VALID, 2, 0, 0));

  /* -- REL_RW ---------------------------------------------------------- */
  /* non-final: count only */
  nx = msi_cache_compute_next(CACHE(MSI_RW_GRANT, MSI_RO_VALID, 2, 0, 0),
                              MSI_CACHE_OP_REL_RW, 0, &act);
  CK("release non-final: wb", act == MSI_CACHE_ACT_WB);
  CK("release non-final: wc--",
     nx == CACHE(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0));
  /* final: demote to plain VALID + final writeback */
  nx = msi_cache_compute_next(CACHE(MSI_RW_GRANT, MSI_RO_VALID, 1, 0, 0),
                              MSI_CACHE_OP_REL_RW, 0, &act);
  CK("release final: wb-final", act == MSI_CACHE_ACT_WB_FINAL);
  CK("release final: demoted",
     nx == CACHE(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0));

  /* -- INVALIDATE ------------------------------------------------------ */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0),
                              MSI_CACHE_OP_INVALIDATE, 0, &act);
  CK("INV on VALID: purge+ack", act == MSI_CACHE_ACT_PURGE_ACK);
  CK("INV on VALID: idle", nx == CACHE(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0));
  /* in-flight fetch: kill mark, ack owed */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_REQ, 0, 0, 9),
                              MSI_CACHE_OP_INVALIDATE, 0, &act);
  CK("INV on REQ: kill-marked (ack owed)", act == MSI_CACHE_ACT_KILL_MARKED);
  CK("INV on REQ: mark set, chain kept",
     nx == CACHE(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 9));
  /* idempotent cases */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0),
                              MSI_CACHE_OP_INVALIDATE, 0, &act);
  CK("INV on IDLE: noop+ack", act == MSI_CACHE_ACT_NOOP_ACK);
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_REQ_KILL, 0, 0, 9),
                              MSI_CACHE_OP_INVALIDATE, 0, &act);
  CK("INV on kill mark: noop+ack (single kill per fetch)",
     act == MSI_CACHE_ACT_NOOP_ACK);

  /* -- KILL_PURGE ------------------------------------------------------ */
  nx = msi_cache_compute_next(CACHE(MSI_RW_IDLE, MSI_RO_VALID, 0, 0, 0),
                              MSI_CACHE_OP_KILL_PURGE, 0, &act);
  CK("kill purge: none (owed ack is the caller's LAST step)",
     act == MSI_CACHE_ACT_NONE);
  CK("kill purge: idle", nx == CACHE(MSI_RW_IDLE, MSI_RO_IDLE, 0, 0, 0));
}

static void dir_table(void) {
  uint32_t act;
  uint64_t nx;

  /* -- REQ_RW ---------------------------------------------------------- */
  /* first writer of an idle directory: the claim fires in the same CAS */
  nx = msi_dir_compute_next(DIR(0, 0, MSI_OWNER_NOBODY, 0), MSI_DIR_OP_REQ_RW,
                            0, &act);
  CK("first REQ: grant-claim", act == MSI_DIR_ACT_GRANT_CLAIM);
  CK("first REQ: GRANTING + w=1",
     nx == DIR(0, 0, MSI_OWNER_GRANTING, 1));
  /* tenure active: queue only */
  nx = msi_dir_compute_next(DIR(0, 0, 3, 1), MSI_DIR_OP_REQ_RW, 0, &act);
  CK("REQ under tenure: none", act == MSI_DIR_ACT_NONE);
  CK("REQ under tenure: w++", nx == DIR(0, 0, 3, 2));
  /* w 0-edge but owner mid-publish (GRANTING): no second claim */
  nx = msi_dir_compute_next(DIR(0, 0, MSI_OWNER_GRANTING, 1),
                            MSI_DIR_OP_REQ_RW, 0, &act);
  CK("REQ under GRANTING: none", act == MSI_DIR_ACT_NONE);
  CK("REQ under GRANTING: w++", nx == DIR(0, 0, MSI_OWNER_GRANTING, 2));

  /* -- ROUND_CLAIM / ACKS_ARM / ACK_DEC -------------------------------- */
  nx = msi_dir_compute_next(DIR(0, 0, 3, 2), MSI_DIR_OP_ROUND_CLAIM, 0, &act);
  CK("claim on closed: claimed", act == MSI_DIR_ACT_CLAIMED);
  CK("claim on closed: open", nx == DIR(1, 0, 3, 2));
  nx = msi_dir_compute_next(DIR(1, 0, 3, 2), MSI_DIR_OP_ROUND_CLAIM, 0, &act);
  CK("claim on open: refused", act == MSI_DIR_ACT_NONE);
  CK("claim on open: unchanged", nx == DIR(1, 0, 3, 2));
  nx = msi_dir_compute_next(DIR(1, 0, 3, 2), MSI_DIR_OP_ACKS_ARM, 2, &act);
  CK("arm: none", act == MSI_DIR_ACT_NONE);
  CK("arm: acks set", nx == DIR(1, 2, 3, 2));
  nx = msi_dir_compute_next(DIR(1, 2, 3, 2), MSI_DIR_OP_ACK_DEC, 0, &act);
  CK("dec 2->1: none", act == MSI_DIR_ACT_NONE);
  CK("dec 2->1: acks--", nx == DIR(1, 1, 3, 2));
  nx = msi_dir_compute_next(DIR(1, 1, 3, 2), MSI_DIR_OP_ACK_DEC, 0, &act);
  CK("dec 1->0: close", act == MSI_DIR_ACT_CLOSE);
  CK("dec 1->0: acks 0", nx == DIR(1, 0, 3, 2));

  /* -- OWNER_PUBLISH --------------------------------------------------- */
  nx = msi_dir_compute_next(DIR(0, 0, MSI_OWNER_GRANTING, 2),
                            MSI_DIR_OP_OWNER_PUBLISH, 3, &act);
  CK("owner publish: none", act == MSI_DIR_ACT_NONE);
  CK("owner publish: named", nx == DIR(0, 0, 3, 2));

  /* -- ROUND_CLOSE ----------------------------------------------------- */
  /* no final in the batch: drop the round bit only */
  nx = msi_dir_compute_next(DIR(1, 0, 3, 2), MSI_DIR_OP_ROUND_CLOSE,
                            MSI_OWNER_NOBODY, &act);
  CK("close no-final: none", act == MSI_DIR_ACT_NONE);
  CK("close no-final: closed", nx == DIR(0, 0, 3, 2));
  /* final with writers left: {w--, owner clear, re-claim} in one atom */
  nx = msi_dir_compute_next(DIR(1, 0, 3, 2), MSI_DIR_OP_ROUND_CLOSE, 3, &act);
  CK("close final chain: grant-claim", act == MSI_DIR_ACT_GRANT_CLAIM);
  CK("close final chain: GRANTING + w--",
     nx == DIR(0, 0, MSI_OWNER_GRANTING, 1));
  /* final, no writers left: directory idles */
  nx = msi_dir_compute_next(DIR(1, 0, 3, 1), MSI_DIR_OP_ROUND_CLOSE, 3, &act);
  CK("close final idle: none", act == MSI_DIR_ACT_NONE);
  CK("close final idle: NOBODY + w=0",
     nx == DIR(0, 0, MSI_OWNER_NOBODY, 0));
  /* final by a NON-owner rank is unit-neutral: a legitimate final always
   * names the published owner, so a mismatched releaser is a surplus
   * creation hold (concurrent creates of one GUID) the directory never
   * counted — no w--, no owner change, just the round-bit drop. */
  nx = msi_dir_compute_next(DIR(1, 0, 3, 2), MSI_DIR_OP_ROUND_CLOSE, 5, &act);
  CK("close final other-rank: neutral", nx == DIR(0, 0, 3, 2));
  CK("close final other-rank: no claim", act == MSI_DIR_ACT_NONE);
}

int main(void) {
  cache_table();
  dir_table();
  if (g_fail) {
    return 1;
  }
  printf("PASS msi_compute_next: cache + dir truth tables\n");
  return 0;
}

/* Pull in the arbiters (defines the two pure functions). */
#include "core/coherence/msi/arbiters.c"

#endif /* ARTS_PROTOCOL_MSI */
