/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol (write-invalidate) type layouts.  Two timing arms share this
 * header; the build selects exactly one.
 *
 * ── EAGER ──────────────────────────────────────────────────────────────
 * Cache side: ONE 64-bit word carries the whole per-rank protocol state,
 * INCLUDING the parked-waiter chain heads:
 *
 *   [ rw_st:2 (63..62) | ro_st:2 (61..60) | wc:24 (59..36) |
 *     head_rw:18 (35..18) | head_ro:18 (17..0) ]
 *
 *   rw_st ∈ {IDLE, REQ, GRANT}; ro_st ∈ {IDLE, REQ, REQ_KILL, VALID}.
 *   wc      = local writer-EDT count (acquired, not yet released).
 *   head_*  = parked-waiter chain heads, as indices into the per-DB waiter
 *             pool (0 = empty; pool indices start at 1).
 *
 * The heads live IN the word so that (a) parking is the acquire's own
 * decision CAS — a parked node can never miss the fetch it joined — and
 * (b) the install-publishing CAS grabs the whole chain in the same atom —
 * the committer owns exactly its fetch's cohort and serves it as the
 * continuation of its own transition, however late that runs.  There is no
 * reader count in the word: a valid copy persists across reader release
 * (no read-return wire), so readers on a valid copy are pure loads.
 *
 * Home side: ONE 64-bit directory word
 *
 *   [ round_open:1 (63) | acks:14 (62..49) | owner:14 (48..35) |
 *     w:24 (34..11) | spare:11 (10..0) ]
 *
 *   round_open = at most one invalidation round open at a time.
 *   acks       = outstanding INVALIDATE_ACKs of the open round.
 *   owner      = writer-tenure owner rank; MSI_OWNER_NOBODY / MSI_OWNER_GRANTING
 *                sentinels (14-bit field, same width as the GUID rank field).
 *   w          = writer-NODE count (active tenure + queued requesters).
 *
 * The canonical home version is the installed buffer's version
 * (cache.buffer->version); there is no separate version field.
 *
 * ── LAZY ───────────────────────────────────────────────────────────────
 * The canonical copy lives at the owner and moves owner → owner by
 * migration only; the home is a pure directory that never holds bytes.
 * There is NO writeback and no writeback ack — that is the only thing this
 * timing drops.  The invalidation ack it does NOT drop: every RW release
 * asks the home for a round, the home snapshots the copy roster,
 * multicasts, collects EVERY ack, and only then does the release return.
 *
 * A reader never waits anywhere — not at the home, not at the owner, not
 * behind a remote writer, and not behind its own rank's ownership request.
 * The home registers the requester in the roster and redirects
 * unconditionally; the redirect target serves immediately and the requester
 * ALWAYS installs a durable valid copy; every later read on that rank is a
 * pure load with no message and no CAS.  A copy dies exactly one way: an
 * INVALIDATE arrived — the validity axis carries no version at all, and no
 * read path anywhere compares one (see the install-lane stamp below, whose
 * job is disjoint from validity).
 *
 * A rank holding no bytes bounces the redirect back through the home — a
 * retry, not a park: nothing is queued and nothing sleeps, and each pass
 * re-resolves against a later ownership generation.
 *
 * Cache word:
 *
 *   [ reserved:3 (63..61) | rw:2 (60..59) | ro:2 (58..57) | unc:1 (56) |
 *     mtp:1 (55) | inflight:1 (54) | wc:13 (53..41) | wq:13 (40..28) |
 *     head_ro:14 (27..14) | head_rw:14 (13..0) ]
 *
 *   rw   ∈ {IDLE, REQ, GRANT}.  GRANT doubles as the ownership bit and
 *          outlives any individual writer: only the migrate CAS clears it.
 *   ro   ∈ {IDLE, REQ, REQ_KILL, VALID} — same reader plane as EAGER.
 *   unc  = ownership is installed but the directory flip has not been
 *          confirmed.  While it is set no local writer may run and NO store
 *          may happen; ownership may not migrate on.  Without it a new owner
 *          could publish writes while the directory still names the previous
 *          one, and a reader registered afterwards would be redirected to
 *          that rank's retained (older) copy and install it durably.
 *   mtp  = a migration target is published off-word and awaits the 0-edge.
 *          It also closes the door: a local acquire may not join the current
 *          writers while it is set, or a stream of local writers starves the
 *          migration.  (Fairness, not safety.)
 *   inflight = a read reply is on the wire to this rank.  Set by the CAS that
 *          issues a fetch, cleared by the CAS the reply lands in, and no new
 *          fetch may open while it is set — so at most one read reply is ever
 *          in flight per rank and a landing reply always belongs to the
 *          unique open fetch.  The OWNERSHIP request lane is independent of
 *          this bit: reader and writer requests never wait on each other.
 *   wc / wq = local writers holding the DB / queued behind a closed door or
 *          an unconfirmed install.  A counter owns its own chain: wq owns
 *          head_rw, and no transition moves one without the other.
 *
 * Parking is the in-word chain idiom, and there are exactly two chains (one
 * per plane), so both heads fit the word: parking IS the acquire's decision
 * CAS (a parked node can never miss what it joined) and the publishing CAS
 * grabs the whole chain in the same atom (the committer owns exactly the
 * waiters its own transition closed and serves them as its continuation).
 *
 * Field widths are contract limits the caller asserts, not errors this layout
 * can report: 8,191 concurrent local writers and 16,383 parked nodes per DB
 * per rank.  A shape that needs more has two escapes — a 128-bit
 * compare-exchange, or moving a head behind a claim bit.
 *
 * The install-lane arbitration stamp is NOT in this word: it is the buffer's
 * own version field.  Two asynchronous installs can target one rank's buffer
 * slot (a read reply and an ownership migration), neither can put the pointer
 * swap inside its own decision CAS, so the swap is stamp-conditional (larger
 * wins, a stale one retreats and recycles) and always ordered BEFORE the
 * publishing CAS.  That is its only use: nothing reads it to decide whether a
 * copy is still valid — only the arrival of an INVALIDATE decides that.
 *
 * Directory word:
 *
 *   [ round_open:1 (63) | moving:1 (62) | acks:14 (61..48) |
 *     owner:14 (47..34) | w:24 (33..10) | reserved:10 (9..0) ]
 *
 *   owner  = the rank the directory believes holds the canonical copy;
 *            written only by a CONFIRM.
 *   moving = one migration is unconfirmed; its claimant is the sole
 *            forwarder, so exactly one migration is ever in flight.
 *   w      = RW-requesting ranks not yet granted ownership.  The count and
 *            the claim decision move in ONE CAS: split, a claim can be
 *            decided on a population that does not include the requester
 *            that triggered it, and the same waiter gets forwarded twice.
 *   round_open / acks = mutual exclusion and outstanding-ack count of the
 *            one open invalidation round.
 *
 * Runtime invariants (assert targets; the word packing depends on them):
 *   W1 rw==GRANT ⟹ ro==VALID
 *   W2 unc ⟹ rw==GRANT ∧ wc==0 (nobody holds it while the gate is shut)
 *   W3 |RW chain| == wq
 *   W4 ro∈{REQ,REQ_KILL} ⟹ inflight (an open fetch always has a reply coming)
 *   W5 ro==VALID ⟹ the RO chain is empty (a valid copy is a blind hit)
 *   W6 a second INVALIDATE on a kill-marked fetch is unreachable: the ack it
 *      owes keeps the round that marked it open, and rounds serialize
 *   W7 at most one read reply in flight per rank
 *   W8 an ownership install's base is >= the slot's current stamp
 */
#ifndef ARTS_COHERENCE_MSI_TYPES_H
#define ARTS_COHERENCE_MSI_TYPES_H

#include "arts/coherence/types_common.h"
#include "arts/rank_bitset.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ARTS_TIMING_LAZY

/* ── cache word layout ──────────────────────────────────────────────────── */
#define MSI_LAZY_CACHE_HEAD_BITS 14
#define MSI_LAZY_CACHE_CNT_BITS 13 /* wc / wq */
#define MSI_LAZY_CACHE_FLAG_BITS 1
#define MSI_LAZY_CACHE_ST_BITS 2

#define MSI_LAZY_CACHE_HEAD_RW_SHIFT 0
#define MSI_LAZY_CACHE_HEAD_RO_SHIFT                                           \
  (MSI_LAZY_CACHE_HEAD_RW_SHIFT + MSI_LAZY_CACHE_HEAD_BITS)
#define MSI_LAZY_CACHE_WQ_SHIFT                                                \
  (MSI_LAZY_CACHE_HEAD_RO_SHIFT + MSI_LAZY_CACHE_HEAD_BITS)
#define MSI_LAZY_CACHE_WC_SHIFT                                                \
  (MSI_LAZY_CACHE_WQ_SHIFT + MSI_LAZY_CACHE_CNT_BITS)
#define MSI_LAZY_CACHE_INFLIGHT_SHIFT                                          \
  (MSI_LAZY_CACHE_WC_SHIFT + MSI_LAZY_CACHE_CNT_BITS)
#define MSI_LAZY_CACHE_MTP_SHIFT                                               \
  (MSI_LAZY_CACHE_INFLIGHT_SHIFT + MSI_LAZY_CACHE_FLAG_BITS)
#define MSI_LAZY_CACHE_UNC_SHIFT                                               \
  (MSI_LAZY_CACHE_MTP_SHIFT + MSI_LAZY_CACHE_FLAG_BITS)
#define MSI_LAZY_CACHE_RO_SHIFT                                                \
  (MSI_LAZY_CACHE_UNC_SHIFT + MSI_LAZY_CACHE_FLAG_BITS)
#define MSI_LAZY_CACHE_RW_SHIFT                                                \
  (MSI_LAZY_CACHE_RO_SHIFT + MSI_LAZY_CACHE_ST_BITS)

#define MSI_LAZY_CACHE_HEAD_MASK                                               \
  ((uint64_t)((1ULL << MSI_LAZY_CACHE_HEAD_BITS) - 1))
#define MSI_LAZY_CACHE_CNT_MASK                                                \
  ((uint64_t)((1ULL << MSI_LAZY_CACHE_CNT_BITS) - 1))
#define MSI_LAZY_CACHE_FLAG_MASK ((uint64_t)0x1ULL)
#define MSI_LAZY_CACHE_ST_MASK ((uint64_t)0x3ULL)

/* rw values */
#define MSI_RW_IDLE 0u
#define MSI_RW_REQ 1u
#define MSI_RW_GRANT 2u
/* ro values */
#define MSI_RO_IDLE 0u
#define MSI_RO_REQ 1u
#define MSI_RO_REQ_KILL 2u /* in-flight fetch marked by an INVALIDATE */
#define MSI_RO_VALID 3u

#define MSI_LAZY_CACHE_RW(s)                                                   \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_RW_SHIFT) & MSI_LAZY_CACHE_ST_MASK))
#define MSI_LAZY_CACHE_RO(s)                                                   \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_RO_SHIFT) & MSI_LAZY_CACHE_ST_MASK))
#define MSI_LAZY_CACHE_UNC(s)                                                  \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_UNC_SHIFT) & MSI_LAZY_CACHE_FLAG_MASK))
#define MSI_LAZY_CACHE_MTP(s)                                                  \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_MTP_SHIFT) & MSI_LAZY_CACHE_FLAG_MASK))
#define MSI_LAZY_CACHE_INFLIGHT(s)                                             \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_INFLIGHT_SHIFT) &                         \
              MSI_LAZY_CACHE_FLAG_MASK))
#define MSI_LAZY_CACHE_WC(s)                                                   \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_WC_SHIFT) & MSI_LAZY_CACHE_CNT_MASK))
#define MSI_LAZY_CACHE_WQ(s)                                                   \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_WQ_SHIFT) & MSI_LAZY_CACHE_CNT_MASK))
#define MSI_LAZY_CACHE_HEAD_RO(s)                                              \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_HEAD_RO_SHIFT) & MSI_LAZY_CACHE_HEAD_MASK))
#define MSI_LAZY_CACHE_HEAD_RW(s)                                              \
  ((uint32_t)(((s) >> MSI_LAZY_CACHE_HEAD_RW_SHIFT) & MSI_LAZY_CACHE_HEAD_MASK))
#define MSI_LAZY_CACHE_MAKE(rw, ro, unc, mtp, inflight, wc, wq, hro, hrw)      \
  ((((uint64_t)(rw) & MSI_LAZY_CACHE_ST_MASK) << MSI_LAZY_CACHE_RW_SHIFT) |    \
   (((uint64_t)(ro) & MSI_LAZY_CACHE_ST_MASK) << MSI_LAZY_CACHE_RO_SHIFT) |    \
   (((uint64_t)(unc) & MSI_LAZY_CACHE_FLAG_MASK)                               \
    << MSI_LAZY_CACHE_UNC_SHIFT) |                                             \
   (((uint64_t)(mtp) & MSI_LAZY_CACHE_FLAG_MASK)                               \
    << MSI_LAZY_CACHE_MTP_SHIFT) |                                             \
   (((uint64_t)(inflight) & MSI_LAZY_CACHE_FLAG_MASK)                          \
    << MSI_LAZY_CACHE_INFLIGHT_SHIFT) |                                        \
   (((uint64_t)(wc) & MSI_LAZY_CACHE_CNT_MASK) << MSI_LAZY_CACHE_WC_SHIFT) |   \
   (((uint64_t)(wq) & MSI_LAZY_CACHE_CNT_MASK) << MSI_LAZY_CACHE_WQ_SHIFT) |   \
   (((uint64_t)(hro) & MSI_LAZY_CACHE_HEAD_MASK)                               \
    << MSI_LAZY_CACHE_HEAD_RO_SHIFT) |                                         \
   (((uint64_t)(hrw) & MSI_LAZY_CACHE_HEAD_MASK)                               \
    << MSI_LAZY_CACHE_HEAD_RW_SHIFT))

/* ── directory word layout ──────────────────────────────────────────────── */
#define MSI_LAZY_DIR_ACKS_BITS 14
#define MSI_LAZY_DIR_OWNER_BITS 14 /* = ARTS_GUID_RANK_BITS */
#define MSI_LAZY_DIR_W_BITS 24
#define MSI_LAZY_DIR_FLAG_BITS 1

#define MSI_LAZY_DIR_W_SHIFT 10 /* 10 spare low bits */
#define MSI_LAZY_DIR_OWNER_SHIFT (MSI_LAZY_DIR_W_SHIFT + MSI_LAZY_DIR_W_BITS)
#define MSI_LAZY_DIR_ACKS_SHIFT                                                \
  (MSI_LAZY_DIR_OWNER_SHIFT + MSI_LAZY_DIR_OWNER_BITS)
#define MSI_LAZY_DIR_MOVING_SHIFT                                              \
  (MSI_LAZY_DIR_ACKS_SHIFT + MSI_LAZY_DIR_ACKS_BITS)
#define MSI_LAZY_DIR_ROUND_SHIFT                                               \
  (MSI_LAZY_DIR_MOVING_SHIFT + MSI_LAZY_DIR_FLAG_BITS)

#define MSI_LAZY_DIR_ACKS_MASK                                                 \
  ((uint64_t)((1ULL << MSI_LAZY_DIR_ACKS_BITS) - 1))
#define MSI_LAZY_DIR_OWNER_MASK                                                \
  ((uint64_t)((1ULL << MSI_LAZY_DIR_OWNER_BITS) - 1))
#define MSI_LAZY_DIR_W_MASK ((uint64_t)((1ULL << MSI_LAZY_DIR_W_BITS) - 1))
#define MSI_LAZY_DIR_FLAG_MASK ((uint64_t)0x1ULL)

/* A LAZY directory names a real owner from creation onward (the creator boots
 * as owner), so the only sentinel is the pre-install / destroyed value. */
#define MSI_OWNER_NOBODY ((uint32_t)MSI_LAZY_DIR_OWNER_MASK)

#define MSI_LAZY_DIR_ROUND_OPEN(s)                                             \
  ((uint32_t)(((s) >> MSI_LAZY_DIR_ROUND_SHIFT) & MSI_LAZY_DIR_FLAG_MASK))
#define MSI_LAZY_DIR_MOVING(s)                                                 \
  ((uint32_t)(((s) >> MSI_LAZY_DIR_MOVING_SHIFT) & MSI_LAZY_DIR_FLAG_MASK))
#define MSI_LAZY_DIR_ACKS(s)                                                   \
  ((uint32_t)(((s) >> MSI_LAZY_DIR_ACKS_SHIFT) & MSI_LAZY_DIR_ACKS_MASK))
#define MSI_LAZY_DIR_OWNER(s)                                                  \
  ((uint32_t)(((s) >> MSI_LAZY_DIR_OWNER_SHIFT) & MSI_LAZY_DIR_OWNER_MASK))
#define MSI_LAZY_DIR_W(s)                                                      \
  ((uint32_t)(((s) >> MSI_LAZY_DIR_W_SHIFT) & MSI_LAZY_DIR_W_MASK))
#define MSI_LAZY_DIR_MAKE(open, moving, acks, owner, w)                        \
  ((((uint64_t)(open) & MSI_LAZY_DIR_FLAG_MASK)                                \
    << MSI_LAZY_DIR_ROUND_SHIFT) |                                             \
   (((uint64_t)(moving) & MSI_LAZY_DIR_FLAG_MASK)                              \
    << MSI_LAZY_DIR_MOVING_SHIFT) |                                            \
   (((uint64_t)(acks) & MSI_LAZY_DIR_ACKS_MASK)                                \
    << MSI_LAZY_DIR_ACKS_SHIFT) |                                              \
   (((uint64_t)(owner) & MSI_LAZY_DIR_OWNER_MASK)                              \
    << MSI_LAZY_DIR_OWNER_SHIFT) |                                             \
   (((uint64_t)(w) & MSI_LAZY_DIR_W_MASK) << MSI_LAZY_DIR_W_SHIFT))

#ifndef __cplusplus
/* Three bits of the cache word are reserved; every other bit is claimed. */
_Static_assert(MSI_LAZY_CACHE_RW_SHIFT + MSI_LAZY_CACHE_ST_BITS == 61,
               "LAZY cache word must pack into 61 bits (3 reserved)");
_Static_assert(MSI_LAZY_DIR_ROUND_SHIFT + MSI_LAZY_DIR_FLAG_BITS == 64,
               "LAZY directory word must pack to exactly 64 bits");
/* A counter owns its chain (W3: |RW chain| == wq), so every waiter the count
 * admits must be addressable by the matching head. */
_Static_assert(MSI_LAZY_CACHE_CNT_MASK <= MSI_LAZY_CACHE_HEAD_MASK,
               "every counted waiter must be addressable by its chain head");
/* Both state fields must be encodable, and the three flags the invariants
 * relate (W2/W4 and the migration door) must be distinct bits. */
_Static_assert(MSI_RO_VALID <= MSI_LAZY_CACHE_ST_MASK &&
                   MSI_RW_GRANT <= MSI_LAZY_CACHE_ST_MASK,
               "state encodings must fit their 2-bit fields");
_Static_assert(MSI_LAZY_CACHE_UNC_SHIFT != MSI_LAZY_CACHE_INFLIGHT_SHIFT &&
                   MSI_LAZY_CACHE_MTP_SHIFT !=
                       MSI_LAZY_CACHE_INFLIGHT_SHIFT &&
                   MSI_LAZY_CACHE_UNC_SHIFT != MSI_LAZY_CACHE_MTP_SHIFT,
               "confirm-gate / migration / fetch flags must be distinct bits");

#define MSI_LAZY_CACHE_PIN_                                                    \
  MSI_LAZY_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_REQ_KILL, 1u, 1u, 1u, 0x1234u,      \
                      0xEDCu, 0x1ABCu, 0xF5u)
_Static_assert(MSI_LAZY_CACHE_RW(MSI_LAZY_CACHE_PIN_) == MSI_RW_GRANT &&
                   MSI_LAZY_CACHE_RO(MSI_LAZY_CACHE_PIN_) == MSI_RO_REQ_KILL &&
                   MSI_LAZY_CACHE_UNC(MSI_LAZY_CACHE_PIN_) == 1u &&
                   MSI_LAZY_CACHE_MTP(MSI_LAZY_CACHE_PIN_) == 1u &&
                   MSI_LAZY_CACHE_INFLIGHT(MSI_LAZY_CACHE_PIN_) == 1u &&
                   MSI_LAZY_CACHE_WC(MSI_LAZY_CACHE_PIN_) == 0x1234u &&
                   MSI_LAZY_CACHE_WQ(MSI_LAZY_CACHE_PIN_) == 0xEDCu &&
                   MSI_LAZY_CACHE_HEAD_RO(MSI_LAZY_CACHE_PIN_) == 0x1ABCu &&
                   MSI_LAZY_CACHE_HEAD_RW(MSI_LAZY_CACHE_PIN_) == 0xF5u,
               "LAZY cache word field round-trip");
#undef MSI_LAZY_CACHE_PIN_

#define MSI_LAZY_DIR_PIN_                                                      \
  MSI_LAZY_DIR_MAKE(1u, 1u, 0x2AAAu, 0x1234u, 0xABCDEFu)
_Static_assert(MSI_LAZY_DIR_ROUND_OPEN(MSI_LAZY_DIR_PIN_) == 1u &&
                   MSI_LAZY_DIR_MOVING(MSI_LAZY_DIR_PIN_) == 1u &&
                   MSI_LAZY_DIR_ACKS(MSI_LAZY_DIR_PIN_) == 0x2AAAu &&
                   MSI_LAZY_DIR_OWNER(MSI_LAZY_DIR_PIN_) == 0x1234u &&
                   MSI_LAZY_DIR_W(MSI_LAZY_DIR_PIN_) == 0xABCDEFu,
               "LAZY directory word field round-trip");
#undef MSI_LAZY_DIR_PIN_
#endif /* __cplusplus */

#else /* EAGER */

/* ── cache word layout ──────────────────────────────────────────────────── */
#define MSI_CACHE_HEAD_BITS 18
#define MSI_CACHE_WC_BITS 24
#define MSI_CACHE_ST_BITS 2

#define MSI_CACHE_HEAD_RO_SHIFT 0
#define MSI_CACHE_HEAD_RW_SHIFT (MSI_CACHE_HEAD_RO_SHIFT + MSI_CACHE_HEAD_BITS)
#define MSI_CACHE_WC_SHIFT (MSI_CACHE_HEAD_RW_SHIFT + MSI_CACHE_HEAD_BITS)
#define MSI_CACHE_RO_SHIFT (MSI_CACHE_WC_SHIFT + MSI_CACHE_WC_BITS)
#define MSI_CACHE_RW_SHIFT (MSI_CACHE_RO_SHIFT + MSI_CACHE_ST_BITS)

#define MSI_CACHE_HEAD_MASK ((uint64_t)((1ULL << MSI_CACHE_HEAD_BITS) - 1))
#define MSI_CACHE_WC_MASK ((uint64_t)((1ULL << MSI_CACHE_WC_BITS) - 1))
#define MSI_CACHE_ST_MASK ((uint64_t)0x3ULL)

/* rw_st values */
#define MSI_RW_IDLE 0u
#define MSI_RW_REQ 1u
#define MSI_RW_GRANT 2u
/* ro_st values */
#define MSI_RO_IDLE 0u
#define MSI_RO_REQ 1u
#define MSI_RO_REQ_KILL 2u /* in-flight fetch marked by an INVALIDATE */
#define MSI_RO_VALID 3u

#define MSI_CACHE_RW(s)                                                        \
  ((uint32_t)(((s) >> MSI_CACHE_RW_SHIFT) & MSI_CACHE_ST_MASK))
#define MSI_CACHE_RO(s)                                                        \
  ((uint32_t)(((s) >> MSI_CACHE_RO_SHIFT) & MSI_CACHE_ST_MASK))
#define MSI_CACHE_WC(s)                                                        \
  ((uint32_t)(((s) >> MSI_CACHE_WC_SHIFT) & MSI_CACHE_WC_MASK))
#define MSI_CACHE_HEAD_RW(s)                                                   \
  ((uint32_t)(((s) >> MSI_CACHE_HEAD_RW_SHIFT) & MSI_CACHE_HEAD_MASK))
#define MSI_CACHE_HEAD_RO(s)                                                   \
  ((uint32_t)(((s) >> MSI_CACHE_HEAD_RO_SHIFT) & MSI_CACHE_HEAD_MASK))
#define MSI_CACHE_MAKE(rw, ro, wc, hrw, hro)                                   \
  ((((uint64_t)(rw) & MSI_CACHE_ST_MASK) << MSI_CACHE_RW_SHIFT) |              \
   (((uint64_t)(ro) & MSI_CACHE_ST_MASK) << MSI_CACHE_RO_SHIFT) |              \
   (((uint64_t)(wc) & MSI_CACHE_WC_MASK) << MSI_CACHE_WC_SHIFT) |              \
   (((uint64_t)(hrw) & MSI_CACHE_HEAD_MASK) << MSI_CACHE_HEAD_RW_SHIFT) |      \
   (((uint64_t)(hro) & MSI_CACHE_HEAD_MASK) << MSI_CACHE_HEAD_RO_SHIFT))

/* ── directory word layout ──────────────────────────────────────────────── */
#define MSI_DIR_ACKS_BITS 14
#define MSI_DIR_OWNER_BITS 14 /* = ARTS_GUID_RANK_BITS */
#define MSI_DIR_W_BITS 24

#define MSI_DIR_W_SHIFT 11 /* 11 spare low bits */
#define MSI_DIR_OWNER_SHIFT (MSI_DIR_W_SHIFT + MSI_DIR_W_BITS)
#define MSI_DIR_ACKS_SHIFT (MSI_DIR_OWNER_SHIFT + MSI_DIR_OWNER_BITS)
#define MSI_DIR_ROUND_SHIFT (MSI_DIR_ACKS_SHIFT + MSI_DIR_ACKS_BITS)

#define MSI_DIR_ACKS_MASK ((uint64_t)((1ULL << MSI_DIR_ACKS_BITS) - 1))
#define MSI_DIR_OWNER_MASK ((uint64_t)((1ULL << MSI_DIR_OWNER_BITS) - 1))
#define MSI_DIR_W_MASK ((uint64_t)((1ULL << MSI_DIR_W_BITS) - 1))

#define MSI_OWNER_NOBODY ((uint32_t)MSI_DIR_OWNER_MASK)         /* 0x3FFF */
#define MSI_OWNER_GRANTING ((uint32_t)(MSI_DIR_OWNER_MASK - 1)) /* 0x3FFE */

#define MSI_DIR_ROUND_OPEN(s) ((uint32_t)(((s) >> MSI_DIR_ROUND_SHIFT) & 0x1ULL))
#define MSI_DIR_ACKS(s)                                                        \
  ((uint32_t)(((s) >> MSI_DIR_ACKS_SHIFT) & MSI_DIR_ACKS_MASK))
#define MSI_DIR_OWNER(s)                                                       \
  ((uint32_t)(((s) >> MSI_DIR_OWNER_SHIFT) & MSI_DIR_OWNER_MASK))
#define MSI_DIR_W(s) ((uint32_t)(((s) >> MSI_DIR_W_SHIFT) & MSI_DIR_W_MASK))
#define MSI_DIR_MAKE(open, acks, owner, w)                                     \
  ((((uint64_t)(open) & 0x1ULL) << MSI_DIR_ROUND_SHIFT) |                      \
   (((uint64_t)(acks) & MSI_DIR_ACKS_MASK) << MSI_DIR_ACKS_SHIFT) |            \
   (((uint64_t)(owner) & MSI_DIR_OWNER_MASK) << MSI_DIR_OWNER_SHIFT) |         \
   (((uint64_t)(w) & MSI_DIR_W_MASK) << MSI_DIR_W_SHIFT))

#endif /* ARTS_TIMING_LAZY */

/* ── parked waiter (chain node, index-addressed) ────────────────────────── */
/* Nodes live in a per-DB grow-only pool; the chain link and the free list
 * both use pool indices (0 = none), so a head fits the word's head field.
 * A node is written (guid/slot/next) before the CAS that links it and is
 * read only by the committer that grabbed the chain — single-owner after
 * the grab, freed back to the pool after its serve. */
struct arts_db_msi_waiter_s {
  uint32_t next;         /* pool index of the next node; 0 = end */
  unsigned int slot;     /* dep slot of the parked acquire */
  arts_guid_t edt_guid;  /* parked EDT (guid-addressed idempotent serve) */
};

#define MSI_WAITER_CHUNK_CAP 256u /* nodes per pool chunk (chunked growth) */
#ifdef ARTS_TIMING_LAZY
#define MSI_WAITER_IDX_MAX MSI_LAZY_CACHE_HEAD_MASK
#else
#define MSI_WAITER_IDX_MAX MSI_CACHE_HEAD_MASK
#endif

struct arts_db_msi_waiter_pool_s {
#ifdef __cplusplus
  void *chunks;
  uint32_t next_fresh;
  uint32_t free_head;
#else
  _Atomic(uintptr_t) chunks;   /* chunk directory (grow-only) */
  _Atomic uint32_t next_fresh; /* bump allocator over chunk capacity */
  _Atomic uint32_t free_head;  /* Treiber free list by index (0 = empty) */
#endif
};

#ifndef ARTS_TIMING_LAZY
/* ── home writeback-queue entry (Treiber, whole-chain drain) ────────────── */
/* One entry per RW-release WB in flight at home; drained in a batch by the
 * round opener (order-insensitive: a single max-version install, commuting
 * cv ACKs, at most one final per batch). */
struct arts_db_msi_wb_s {
  arts_lf_link_t link; /* FIRST — required by arts_lf_stack_t */
  uint64_t vnew;
  unsigned int releaser_rank;
  uint32_t final_flag;
  uint64_t cv;                     /* releaser's ack cookie (sem identity) */
  struct arts_rdzv_landing_s rdzv; /* payload landing for this WB */
};
#endif /* !ARTS_TIMING_LAZY */

#ifdef ARTS_TIMING_LAZY
/* ── home round request (one per RW release awaiting its round) ─────────── */
/* A round closes exactly ONE request.  Coalescing several would require the
 * home to know that its roster snapshot followed EVERY coalesced release's
 * version bump, which it cannot: a request's arrival and its bump are separate
 * events.  Their ORDER is free (each round's snapshot provably follows its own
 * request's bump), so the queue only has to hand them out one at a time. */
struct arts_db_msi_round_req_s {
  arts_lf_link_t link; /* FIRST — required by arts_mpsc_t */
  unsigned int rank;   /* releaser to wake at close */
  uint64_t cv;         /* releaser's rendezvous address (pointer identity) */
};
#endif /* ARTS_TIMING_LAZY */

/* Home request queue: same Vyukov MPSC shape as the other arms so the shared
 * home.h declarations apply (producers = request handlers, single consumer =
 * the grant-claim holder). */
#ifdef __cplusplus
struct arts_home_lockreq_node_s {
  struct arts_home_lockreq_node_s *next;
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv;
};
struct arts_home_lockreq_queue_s {
  struct arts_home_lockreq_node_s *tail;
  struct arts_home_lockreq_node_s *head;
  struct arts_home_lockreq_node_s stub;
};
#else
struct arts_home_lockreq_node_s {
  _Atomic(struct arts_home_lockreq_node_s *) next;
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv;
};
struct arts_home_lockreq_queue_s {
  _Atomic(struct arts_home_lockreq_node_s *) tail;
  _Atomic(struct arts_home_lockreq_node_s *) head;
  struct arts_home_lockreq_node_s stub;
};
#endif

/* ── per-rank DB cache ──────────────────────────────────────────────────── */
/* EAGER: the writeback axis is a DEDICATED counter, decoupled from the
 * buffer's install version: buffer versions are pure install machinery (each
 * install = previous+1, locally monotone), while wb_next numbers this
 * tenure's releases from the grant's base (seeded base+1 by the grant/create
 * handler, fetch_add per release).  Sharing buf->version for both would let
 * local install counts overtake the grant base and stale-reject the grant's
 * own install.
 * LAZY: the owner's buffer IS canonical and carries the whole version axis —
 * the release's fetch_add on it is the publication atom, and an ownership
 * migration carries that same number as the receiving slot's install stamp.
 * pending_snapshot stays empty here: a read racing a creation hold is served
 * the live bytes, never held. */
#ifdef __cplusplus
struct arts_db_cache_s {
  uint64_t cache_state; /* single coherence word — see MSI_*CACHE_* */
  arts_atomic_shared_ptr_t buffer;
  arts_lockfree_pool_t buf_freelist;
  arts_lf_stack_t pending_snapshot; /* pre-publication read holds (home) */
  struct arts_db_msi_waiter_pool_s waiters; /* chain-node pool (index) */
#ifdef ARTS_TIMING_LAZY
  unsigned int migrate_target;
  struct arts_rdzv_landing_s migrate_rdzv;
#else
  uint64_t wb_next; /* next release's writeback version (grant-seeded) */
#endif
  arts_guid_t db_guid;
  uint64_t db_size;
};
#else
struct arts_db_cache_s {
  _Atomic uint64_t cache_state;
  arts_atomic_shared_ptr_t buffer;
  arts_lockfree_pool_t buf_freelist;
  arts_lf_stack_t pending_snapshot;
  struct arts_db_msi_waiter_pool_s waiters;
#ifdef ARTS_TIMING_LAZY
  /* Migration target and its landing, published by the word's mtp bit: the
   * stores land before the bit, and the directory's moving claim admits one
   * migration at a time, so the fields have a single writer per publication.
   * With mtp clear they are dead — the word CAS that consumes a migration
   * cannot clear them, so an arriving order overwrites unconditionally. */
  _Atomic unsigned int migrate_target;
  struct arts_rdzv_landing_s migrate_rdzv;
#else
  _Atomic uint64_t wb_next;
#endif
  arts_guid_t db_guid;
  uint64_t db_size;
};
#endif

/** Internal DataBlock descriptor (MSI protocol).
 *
 *  The per-rank coherence cache is embedded by value as the FIRST member.
 *  Non-home ranks allocate a cache-only footprint
 *  (arts_db_cache_stub_size() = offsetof(dir_state)) that includes
 *  home_initialized but omits the home-directory fields. */
struct arts_db_s {
  struct arts_db_cache_s cache; /**< FIRST — coherence state. */
  arts_db_types_t db_type;
  bool home_initialized;
#ifdef __cplusplus
  uint64_t dir_state;
#ifdef ARTS_TIMING_LAZY
  void *cur_round;
  uint32_t round_pending;
#else
  uint64_t hver;
  bool opening_pending;
#endif
#else
  _Atomic uint64_t dir_state; /* single directory word — see MSI_*DIR_* */
#ifdef ARTS_TIMING_LAZY
  /* The request the open round must answer, captured under the round claim
   * (its holder is the only writer) and consumed by exchange at the close. */
  _Atomic(struct arts_db_msi_round_req_s *) cur_round;
  /* Population of the request queue, incremented AFTER a push completes and
   * claimed by a decrement before the matching pop.  A Vyukov pop reports
   * empty while a producer is mid-link, so the count — not the pop — is what
   * tells the round engine whether work exists. */
  _Atomic uint32_t round_pending;
#else
  /* Canonical home version (the protocol axis).  Decoupled from the
   * installed buffer's version lane: a same-rank release moves no data
   * (home buffer == the tenure's working buffer), so its round advances
   * hver WITHOUT an install swap — swapping would orphan the buffer the
   * tenure's writers keep writing.  Written only under the round claim;
   * read (serve stamps, grant bases) at any time. */
  _Atomic uint64_t hver;
  _Atomic bool opening_pending; /* tenure-opening round re-arm latch (XCHG) */
#endif
#endif
  struct arts_home_lockreq_queue_s rw_waiters; /* Vyukov MPSC, pop-one */
#ifdef ARTS_TIMING_LAZY
  arts_mpsc_t round_q; /* Vyukov MPSC of struct arts_db_msi_round_req_s */
#else
  arts_lf_stack_t wb_queue;   /* Treiber, batch drain */
#endif
  struct arts_rank_bitset_s roster;            /* copy roster (INV targets) */
  struct arts_rank_bitset_s cached_ranks;      /* destroy fan-out roster */
#ifndef ARTS_TIMING_LAZY
  /* Single-slot state of the (at most one) open round / pending grant —
   * written only under the round claim / the GRANTING claim. */
  struct arts_rdzv_landing_s grant_rdzv; /* grantee's deliver landing */
  struct arts_db_msi_wb_s *round_entries; /* open round's drained batch */
#endif
  /* GPU staging fields (full arts_db_s alloc; unused on the CPU MSI path). */
  volatile unsigned int reader;
  volatile unsigned int writer;
  volatile unsigned int version;
  unsigned int time_stamp;
} ARTS_ALIGNED_MAX;

/* ── pure arbiters (CAS-retry loop bodies; see arbiters.c) ──────────────── */
#ifdef ARTS_TIMING_LAZY

/* arg: the acquires take the parking node's pool index; the rest ignore it. */
#define MSI_LAZY_CACHE_OP_ACQ_RO 0
#define MSI_LAZY_CACHE_OP_ACQ_RW 1
#define MSI_LAZY_CACHE_OP_REL_RW 2     /* wc--, and the 0-edge hand-off */
#define MSI_LAZY_CACHE_OP_DELIVER 3    /* a read reply landed */
#define MSI_LAZY_CACHE_OP_DELIVER_RW 4 /* ownership + data arrived */
#define MSI_LAZY_CACHE_OP_CONFIRM_ACK 5 /* the directory flip is confirmed */
#define MSI_LAZY_CACHE_OP_INV 6
#define MSI_LAZY_CACHE_OP_FWDM 7    /* migration order from the home */
#define MSI_LAZY_CACHE_OP_MIGRATE 8 /* re-attempt the hand-off, word only */
#define MSI_LAZY_CACHE_OP_SERVE_DECIDE 9 /* a redirected reader arrived */

#define MSI_LAZY_CACHE_ACT_NONE 0
#define MSI_LAZY_CACHE_ACT_SELF_SERVE 1 /* covering copy / open door: run */
#define MSI_LAZY_CACHE_ACT_SEND_RO 2    /* opened a read fetch: send */
#define MSI_LAZY_CACHE_ACT_SEND_RW 3    /* opened an ownership request: send */
#define MSI_LAZY_CACHE_ACT_PARK 4       /* chained onto the in-word waiters */
#define MSI_LAZY_CACHE_ACT_MIGRATE 5    /* ownership dropped: ship */
#define MSI_LAZY_CACHE_ACT_PUBLISH 6    /* durable copy: serve the chain */
#define MSI_LAZY_CACHE_ACT_PUBLISH_KILL 7 /* + the owed ack, after the serve */
#define MSI_LAZY_CACHE_ACT_DROP 8         /* orphan reply: discard */
#define MSI_LAZY_CACHE_ACT_DROP_REFETCH 9 /* + reopen for a stranded chain */
#define MSI_LAZY_CACHE_ACT_INSTALL 10  /* owner now: serve reads, confirm */
#define MSI_LAZY_CACHE_ACT_UNGATE 11   /* gate open: serve the writer chain */
#define MSI_LAZY_CACHE_ACT_PURGE_ACK 12 /* INV on VALID: purged, ack */
#define MSI_LAZY_CACHE_ACT_KILL_OWED 13 /* INV on REQ: marked, ack OWED */
#define MSI_LAZY_CACHE_ACT_NOOP_ACK 14  /* INV idempotent / owner: ack */
#define MSI_LAZY_CACHE_ACT_SERVE 15     /* redirect: deliver from here */
#define MSI_LAZY_CACHE_ACT_BOUNCE 16 /* no bytes here: re-request at home */

uint64_t msi_lazy_cache_compute_next(uint64_t cur, int op, uint32_t arg,
                                     uint32_t *out_action);

/* The home's REQ_RO and ROUND_REQ handlers touch no directory word (a roster
 * fetch_or plus an owner load; a queue push) — they have no op here. */
#define MSI_LAZY_DIR_OP_REQ_RW 0       /* w++ folded with the moving claim */
#define MSI_LAZY_DIR_OP_CONFIRM_FLIP 1 /* arg = the new owner rank */
#define MSI_LAZY_DIR_OP_ROUND_CLAIM 2
#define MSI_LAZY_DIR_OP_ACKS_ARM 3 /* under the claim: arg = target count */
#define MSI_LAZY_DIR_OP_ACK_DEC 4
#define MSI_LAZY_DIR_OP_ROUND_CLOSE 5

#define MSI_LAZY_DIR_ACT_NONE 0
#define MSI_LAZY_DIR_ACT_MIGRATE_CLAIM 1 /* this committer peeks + forwards */
#define MSI_LAZY_DIR_ACT_CLAIMED 2
#define MSI_LAZY_DIR_ACT_CLOSE 3 /* acks reached 0 with the round open */

uint64_t msi_lazy_dir_compute_next(uint64_t cur, int op, unsigned int arg,
                                   uint32_t *out_action);

#else /* EAGER */

#define MSI_CACHE_OP_ACQ_RW 0
#define MSI_CACHE_OP_ACQ_RO 1
#define MSI_CACHE_OP_DELIVER 2    /* RO data arrived (publish attempt) */
#define MSI_CACHE_OP_GRANT 3      /* RW grant arrived */
#define MSI_CACHE_OP_REL_RW 4
#define MSI_CACHE_OP_INVALIDATE 5
#define MSI_CACHE_OP_KILL_PURGE 6 /* reserved invalidate after the kill serve */

#define MSI_CACHE_ACT_NONE 0
#define MSI_CACHE_ACT_SELF_SERVE 1 /* covering VALID/GRANT: serve directly */
#define MSI_CACHE_ACT_SEND_RW 2    /* opened the RW fetch: park + send */
#define MSI_CACHE_ACT_SEND_RO 3    /* opened the RO fetch: park + send */
#define MSI_CACHE_ACT_PARK 4       /* chained onto the in-flight fetch */
#define MSI_CACHE_ACT_PUBLISH 5    /* installed: serve the grabbed RO chain */
#define MSI_CACHE_ACT_PUBLISH_KILL 6 /* + reserved purge + owed ack after */
#define MSI_CACHE_ACT_DROP 7       /* superseded DELIVER */
#define MSI_CACHE_ACT_GRANT_PUBLISH 8 /* serve grabbed RW + RO chains */
#define MSI_CACHE_ACT_WB 9         /* release: send writeback */
#define MSI_CACHE_ACT_WB_FINAL 10  /* last writer: demoted, final writeback */
#define MSI_CACHE_ACT_PURGE_ACK 11 /* INVALIDATE on VALID: purged, ack */
#define MSI_CACHE_ACT_KILL_MARKED 12 /* INVALIDATE on REQ: ack owed */
#define MSI_CACHE_ACT_NOOP_ACK 13  /* INVALIDATE idempotent: ack */

uint64_t msi_cache_compute_next(uint64_t cur, int op, uint32_t self_idx,
                                uint32_t *out_action);

#define MSI_DIR_OP_REQ_RW 0      /* w++, grant-claim on the 0-edge */
#define MSI_DIR_OP_ROUND_CLAIM 1 /* open an invalidation round */
#define MSI_DIR_OP_ACKS_ARM 2    /* under the claim: arm the ack count */
#define MSI_DIR_OP_ACK_DEC 3
#define MSI_DIR_OP_ROUND_CLOSE 4 /* close + (<=1) final + re-claim chain */
#define MSI_DIR_OP_OWNER_PUBLISH 5 /* under GRANTING: name the tenure owner */

#define MSI_DIR_ACT_NONE 0
#define MSI_DIR_ACT_GRANT_CLAIM 1 /* this CAS's committer pops + publishes */
#define MSI_DIR_ACT_CLAIMED 2
#define MSI_DIR_ACT_CLOSE 3 /* acks reached 0 with the round open */

uint64_t msi_dir_compute_next(uint64_t cur, int op, unsigned int arg,
                              uint32_t *out_action);

#endif /* ARTS_TIMING_LAZY */

#ifdef __cplusplus
}
#endif

#endif /* ARTS_COHERENCE_MSI_TYPES_H */
