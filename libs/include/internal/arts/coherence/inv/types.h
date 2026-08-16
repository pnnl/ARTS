/* SPDX-License-Identifier: Apache-2.0
 *
 * INV protocol (write-invalidate) type layouts.  Two write-policy arms share this
 * header; the build selects exactly one.
 *
 * ── WT ──────────────────────────────────────────────────────────────
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
  *   [ round_open:1 (63) | acks:14 (62..49) | spare:49 (48..0) ]
 *
 *   round_open = at most one invalidation round open at a time.
 *   acks       = outstanding INVALIDATE_ACKs of the open round.
 *
 * The word carries the round and nothing else.  Write ownership is not here:
 * it lives in the shared migrating grant (the holder's writer_count plus the
 * home's rw_holder / pending_rw), which every grant-bearing arm shares, so
 * this directory has no owner field and no writer count of its own.
 *
 * The canonical home version is the installed buffer's version
 * (cache.buffer->version); there is no separate version field.
 *
 * ── WB ───────────────────────────────────────────────────────────────
 * The canonical copy lives at the owner and moves owner → owner by
 * migration only; the home is a pure directory that never holds bytes.
 * There is NO publish and no publish ack — that is the only thing this
 * placement drops.  The invalidation ack it does NOT drop: every RW release
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
 *   ro   ∈ {IDLE, REQ, REQ_KILL, VALID} — same reader plane as WT.
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
 *          unique open fetch.  The GRANT request lane is independent of
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
#ifndef ARTS_COHERENCE_INV_TYPES_H
#define ARTS_COHERENCE_INV_TYPES_H

#include "arts/coherence/types_common.h"
#include "arts/rank_bitset.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── cache word: the READER plane, and only the reader plane ─────────────
 * Identical under both placements — what differs between HOME and OWNER is
 * where a fetch is SERVED FROM, never what the requester's word does.
 *
 *   [ ro:2 (63..62) | inflight:1 (61) | spare:43 | head_ro:18 (17..0) ]
 *
 *   ro       ∈ {IDLE, REQ, REQ_KILL, VALID}.  VALID survives reader release:
 *              a reader holding it is a pure load — no message, no CAS, no
 *              version compare.  A copy dies exactly one way: an INVALIDATE
 *              arrived.
 *   inflight = a read reply is on the wire to this rank.  Set by the CAS that
 *              issues a fetch, cleared by the CAS the reply lands in; no new
 *              fetch may open while it is set, so at most one reply is ever in
 *              flight per rank and a landing reply always belongs to the unique
 *              open fetch.
 *   head_ro  = head of the parked-reader chain, an index into the per-DB waiter
 *              pool (0 = empty; pool indices start at 1).  The head lives IN
 *              the word so parking IS the acquire's own decision CAS (a parked
 *              node can never miss the fetch it joined) and the publishing CAS
 *              grabs the whole chain in the same atom (the committer owns
 *              exactly its fetch's cohort and serves it as the continuation of
 *              its own transition, however late that runs).
 *
 * The WRITER plane is not here.  Write ownership is the migrating sentinel
 * grant (coherence/grant.c): writer_count carries local writers plus a +1
 * sentinel while this rank holds the grant, RW waiters park on the cache's
 * pending_rw Treiber stack, and writer_count > 0 IS the ownership predicate.
 * This word therefore carries no rw state, no writer count and no
 * ownership bit: write turns are never arbitrated or returned at the home
 * under either placement.
 *
 * The install-lane arbitration stamp is not in this word either: it is the
 * buffer's own version field.  Two asynchronous installs can target one rank's
 * buffer slot (a read reply and an ownership transfer), neither can put the
 * pointer swap inside its own decision CAS, so the swap is stamp-conditional
 * (larger wins, a stale one retreats and recycles) and always ordered BEFORE
 * the publishing CAS.  Nothing reads it to decide whether a copy is still
 * valid — only the arrival of an INVALIDATE decides that.
 */
#define INV_CACHE_HEAD_BITS 18
#define INV_CACHE_ST_BITS 2
#define INV_CACHE_FLAG_BITS 1

#define INV_CACHE_HEAD_RO_SHIFT 0
#define INV_CACHE_INFLIGHT_SHIFT 61
#define INV_CACHE_RO_SHIFT (INV_CACHE_INFLIGHT_SHIFT + INV_CACHE_FLAG_BITS)

#define INV_CACHE_HEAD_MASK ((uint64_t)((1ULL << INV_CACHE_HEAD_BITS) - 1))
#define INV_CACHE_ST_MASK ((uint64_t)0x3ULL)
#define INV_CACHE_FLAG_MASK ((uint64_t)0x1ULL)

/* ro values */
#define INV_RO_IDLE 0u
#define INV_RO_REQ 1u
#define INV_RO_REQ_KILL 2u /* in-flight fetch marked by an INVALIDATE */
#define INV_RO_VALID 3u

#define INV_CACHE_RO(s)                                                        \
  ((uint32_t)(((s) >> INV_CACHE_RO_SHIFT) & INV_CACHE_ST_MASK))
#define INV_CACHE_INFLIGHT(s)                                                  \
  ((uint32_t)(((s) >> INV_CACHE_INFLIGHT_SHIFT) & INV_CACHE_FLAG_MASK))
#define INV_CACHE_HEAD_RO(s)                                                   \
  ((uint32_t)(((s) >> INV_CACHE_HEAD_RO_SHIFT) & INV_CACHE_HEAD_MASK))
#define INV_CACHE_MAKE(ro, inflight, hro)                                      \
  ((((uint64_t)(ro) & INV_CACHE_ST_MASK) << INV_CACHE_RO_SHIFT) |              \
   (((uint64_t)(inflight) & INV_CACHE_FLAG_MASK) << INV_CACHE_INFLIGHT_SHIFT) |\
   (((uint64_t)(hro) & INV_CACHE_HEAD_MASK) << INV_CACHE_HEAD_RO_SHIFT))

/* ── directory word: the invalidation round, and only the round ──────────
 * Identical under both placements.
 *
 *   [ round_open:1 (63) | acks:14 (62..49) | spare:49 ]
 *
 *   round_open = at most one invalidation round is open at a time.
 *   acks       = outstanding INVALIDATE_ACKs of the open round.
 *
 * Who holds the write grant, who is queued for it, and whether a transfer is
 * in flight are NOT here: they are the grant plane's rw_holder / pending_rw /
 * invalidate_in_flight, shared with every other migrating-grant arm.
 */
#define INV_DIR_ACKS_BITS 14
#define INV_DIR_FLAG_BITS 1

#define INV_DIR_ACKS_SHIFT 49
#define INV_DIR_ROUND_SHIFT (INV_DIR_ACKS_SHIFT + INV_DIR_ACKS_BITS)

#define INV_DIR_ACKS_MASK ((uint64_t)((1ULL << INV_DIR_ACKS_BITS) - 1))
#define INV_DIR_FLAG_MASK ((uint64_t)0x1ULL)

#define INV_DIR_ROUND_OPEN(s)                                                  \
  ((uint32_t)(((s) >> INV_DIR_ROUND_SHIFT) & INV_DIR_FLAG_MASK))
#define INV_DIR_ACKS(s)                                                        \
  ((uint32_t)(((s) >> INV_DIR_ACKS_SHIFT) & INV_DIR_ACKS_MASK))
#define INV_DIR_MAKE(open, acks)                                               \
  ((((uint64_t)(open) & INV_DIR_FLAG_MASK) << INV_DIR_ROUND_SHIFT) |           \
   (((uint64_t)(acks) & INV_DIR_ACKS_MASK) << INV_DIR_ACKS_SHIFT))

#ifndef __cplusplus
_Static_assert(INV_CACHE_RO_SHIFT + INV_CACHE_ST_BITS == 64,
               "cache word must pack to exactly 64 bits");
_Static_assert(INV_DIR_ROUND_SHIFT + INV_DIR_FLAG_BITS == 64,
               "directory word must pack to exactly 64 bits");
_Static_assert(INV_RO_VALID <= INV_CACHE_ST_MASK,
               "reader state encodings must fit the 2-bit field");
#define INV_CACHE_PIN_ INV_CACHE_MAKE(INV_RO_REQ_KILL, 1u, 0x1ABCDu)
_Static_assert(INV_CACHE_RO(INV_CACHE_PIN_) == INV_RO_REQ_KILL &&
                   INV_CACHE_INFLIGHT(INV_CACHE_PIN_) == 1u &&
                   INV_CACHE_HEAD_RO(INV_CACHE_PIN_) == 0x1ABCDu,
               "cache word field round-trip");
#undef INV_CACHE_PIN_
#define INV_DIR_PIN_ INV_DIR_MAKE(1u, 0x2AAAu)
_Static_assert(INV_DIR_ROUND_OPEN(INV_DIR_PIN_) == 1u &&
                   INV_DIR_ACKS(INV_DIR_PIN_) == 0x2AAAu,
               "directory word field round-trip");
#undef INV_DIR_PIN_
#endif /* __cplusplus */

/* ── parked waiter (chain node, index-addressed) ────────────────────────── */
/* Nodes live in a per-DB grow-only pool; the chain link and the free list
 * both use pool indices (0 = none), so a head fits the word's head field.
 * A node is written (guid/slot/next) before the CAS that links it and is
 * read only by the committer that grabbed the chain — single-owner after
 * the grab, freed back to the pool after its serve. */
struct arts_db_inv_waiter_s {
  uint32_t next;         /* pool index of the next node; 0 = end */
  unsigned int slot;     /* dep slot of the parked acquire */
  arts_guid_t edt_guid;  /* parked EDT (guid-addressed idempotent serve) */
};

#define INV_WAITER_CHUNK_CAP 256u /* nodes per pool chunk (chunked growth) */
#define INV_WAITER_IDX_MAX INV_CACHE_HEAD_MASK

struct arts_db_inv_waiter_pool_s {
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

/* ── publish-queue entry (Treiber, whole-chain drain) ───────────────────── */
/* One entry per RW release awaiting its invalidation round at the home.  Both
 * placements use it; only `rdzv` differs — HOME advertises a landing so the
 * release's payload lands before the round opens, OWNER leaves it zero because
 * the canonical copy never travels to the home.  A batch is drained whole and
 * is order-insensitive: at most one install (the max-version entry that
 * carries data) and a set of commuting ack wakes. */
struct arts_db_inv_pub_s {
  arts_lf_link_t link; /* FIRST — required by arts_lf_stack_t */
  uint64_t vnew;
  unsigned int releaser_rank;
  uint64_t cv;                     /* releaser's ack cookie (sem identity) */
  struct arts_rdzv_landing_s rdzv; /* payload landing (HOME); zero under OWNER */
};

/* Home request queue: same Vyukov MPSC shape as the other arms so the shared
 * home.h declarations apply (producers = request handlers, single consumer =
 * the grant-claim holder). */
#ifdef __cplusplus
struct arts_home_grantreq_node_s {
  struct arts_home_grantreq_node_s *next;
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv;
};
struct arts_home_grantreq_queue_s {
  struct arts_home_grantreq_node_s *tail;
  struct arts_home_grantreq_node_s *head;
  struct arts_home_grantreq_node_s stub;
};
#else
struct arts_home_grantreq_node_s {
  _Atomic(struct arts_home_grantreq_node_s *) next;
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv;
};
struct arts_home_grantreq_queue_s {
  _Atomic(struct arts_home_grantreq_node_s *) tail;
  _Atomic(struct arts_home_grantreq_node_s *) head;
  struct arts_home_grantreq_node_s stub;
};
#endif

/* ── per-rank DB cache ──────────────────────────────────────────────────── */
/* The reader plane lives in cache_state (durable copies, one in-flight fetch,
 * one parked chain).  The writer plane is the migrating sentinel grant, held in
 * plain fields shared with every other grant-bearing arm: writer_count carries
 * local writers plus a +1 sentinel while this rank holds the grant, so
 * writer_count > 0 IS "this rank may write".  Nothing withdraws the sentinel
 * except an INVALIDATE, which is what makes the grant sticky.  Under WB the
 * same word also carries ARTS_GRANT_UNCONFIRMED, so "may write" additionally
 * means the home has published the directory flip; the negative value it
 * produces is why the signed test is the predicate and not a convenience.
 *
 * The version axis is the buffer's own version field, bumped by each release.
 * A second, publication-numbering counter would be redundant: no home-issued
 * grant sequence exists for a local install count to race against. */
#ifdef __cplusplus
struct arts_db_cache_s {
  uint64_t cache_state;
  unsigned int writer_count;
  arts_atomic_shared_ptr_t buffer;
  arts_lockfree_pool_t buf_freelist;
  arts_lf_stack_t pending_snapshot;
  arts_lf_stack_t pending_rw;
  struct arts_db_inv_waiter_pool_s waiters;
  unsigned int grant_req_in_flight;
  unsigned int incoming_new_owner;
  struct arts_rdzv_landing_s incoming_new_owner_rdzv;
  struct arts_rank_to_u64_map_s *cached_version;
  arts_guid_t db_guid;
  uint64_t db_size;

  /* WT publish write-combining — the write-side twin of the RO combining
   * window below.  pub_flight is the one-word flight state ({FLYING,DIRTY}:
   * claim and join are each one CAS); pub_waiters holds heap {sem, version}
   * nodes for every releaser blocked until a covering publish is ACKed
   * (heap for the same reason the publish rendezvous is — a shutdown-
   * escaped waiter leaks its node and a late drain may still post into
   * it).  home_pub_* is the durable publish credit: the home's stable
   * buffer plus a receiver-minted txid, refilled by every publish ACK;
   * txid is the presence flag (0 = none), written last with release
   * order so a reader that sees it sees the whole triple. */
  volatile unsigned int pub_flight;
  arts_lf_stack_t pub_waiters;
  /* Waiters drained by a completing flight but not covered by its version:
   * held here (plain field — touched only by the flight owner, and only one
   * flight is in flight) until the trailing flight's ACK re-examines them.
   * Non-NULL implies the flight word is FLYING. */
  void *pub_parked;
  uint64_t home_pub_addr;
  uint64_t home_pub_rkey;
  volatile uint64_t home_pub_txid;
};
#else
struct arts_db_cache_s {
  _Atomic uint64_t cache_state; /* reader plane — see INV_CACHE_* */
  /* Local writers + the ownership sentinel.  A CAS-loop "increment if > 0"
   * is the whole RW fast path: > 0 means this rank holds the grant. */
  volatile unsigned int writer_count;
  arts_atomic_shared_ptr_t buffer;
  arts_lockfree_pool_t buf_freelist;
  arts_lf_stack_t pending_snapshot; /* pre-publication read holds */
  /* RW waiters parked on this rank (Treiber, order-free drain-all): every
   * waiter is woken regardless of order, so a LIFO stack suffices. */
  arts_lf_stack_t pending_rw;
  struct arts_db_inv_waiter_pool_s waiters; /* reader chain-node pool */
  /* Ownership-request coalescing: only the actor that CASes 0->1 sends
   * GRANT_REQUEST; same-node RW acquires piggyback and are picked up by
   * the transfer's drain. */
  volatile unsigned int grant_req_in_flight;
  /* Next transfer target, published by the INVALIDATE handler BEFORE it
   * withdraws the sentinel.  ARTS_NO_PENDING_OWNER = none pending.  Publishing
   * before the withdrawal is what makes a separate flag redundant: whichever
   * actor drives writer_count to exactly 0 reads this field and ships. */
  unsigned int incoming_new_owner;
  struct arts_rdzv_landing_s incoming_new_owner_rdzv;
  /* Always NULL in this arm.  INV's sharer plane deliberately carries no
   * version ledger — a dedup watermark on it would stop the plane being
   * write-driven — but the shared transfer helper reads this field to decide
   * whether a map travels with ownership, and NULL selects its empty branch. */
  struct arts_rank_to_u64_map_s *cached_version;
  arts_guid_t db_guid;
  uint64_t db_size;

  /* WT publish write-combining — the write-side twin of the RO combining
   * window below.  pub_flight is the one-word flight state ({FLYING,DIRTY}:
   * claim and join are each one CAS); pub_waiters holds heap {sem, version}
   * nodes for every releaser blocked until a covering publish is ACKed
   * (heap for the same reason the publish rendezvous is — a shutdown-
   * escaped waiter leaks its node and a late drain may still post into
   * it).  home_pub_* is the durable publish credit: the home's stable
   * buffer plus a receiver-minted txid, refilled by every publish ACK;
   * txid is the presence flag (0 = none), written last with release
   * order so a reader that sees it sees the whole triple. */
  volatile unsigned int pub_flight;
  arts_lf_stack_t pub_waiters;
  /* Waiters drained by a completing flight but not covered by its version:
   * held here (plain field — touched only by the flight owner, and only one
   * flight is in flight) until the trailing flight's ACK re-examines them.
   * Non-NULL implies the flight word is FLYING. */
  void *pub_parked;
  uint64_t home_pub_addr;
  uint64_t home_pub_rkey;
  volatile uint64_t home_pub_txid;
};
#endif

/** Internal DataBlock descriptor (INV protocol).
 *
 *  The per-rank coherence cache is embedded by value as the FIRST member.
 *  Non-home ranks allocate a cache-only footprint
 *  (arts_db_cache_stub_size() = offsetof(rw_holder)) that includes
 *  home_initialized but omits every home-directory field.
 *
 *  The home directory is two independent planes that share no word:
 *    - the migrating write grant (rw_holder / pending_rw / invalidate_in_flight
 *      / pending_install_owner), identical to every other grant-bearing arm and
 *      driven by coherence/grant.c;
 *    - the invalidation round (dir_state / roster / pub_queue), which is what
 *      makes this arm INV.
 *  Both write policies carry exactly these fields; the write policy decides only
 *  whether a release's publish entry carries payload. */
struct arts_db_s {
  struct arts_db_cache_s cache; /**< FIRST — coherence state. */
  arts_db_types_t db_type;
  bool home_initialized;

  /* ---- migrating write grant (home side; see coherence/grant.c) ---- */
  arts_db_atomic_uint_t rw_holder;
  struct arts_home_grantreq_queue_s pending_rw; /* Vyukov MPSC, pop-one */
  arts_db_atomic_uint_t invalidate_in_flight;  /* transfer-round baton */
  unsigned int pending_install_owner; /* baton-holder-written transfer target */

  /* ---- invalidation round (home side; INV's own) ---- */
#ifdef __cplusplus
  uint64_t dir_state;
  bool opening_pending;
#else
  _Atomic uint64_t dir_state;   /* round_open + acks — see INV_DIR_* */
  _Atomic bool opening_pending; /* round re-arm latch (XCHG carry) */
#endif
  /* The home's canonical publication axis, advanced to each round's highest
   * published version.  It is NOT the installed buffer's version, and must not
   * be: a home-resident owner writes THROUGH the home buffer in place, so its
   * releases change the bytes without ever swapping an install — the buffer's
   * own lane stands still while the data moves on.  Serving readers off that
   * lane hands two different rounds the same stamp, and the receiver's
   * version-conditional install then rejects the newer copy while its state
   * machine still marks it valid: a reader left on bytes from a write window
   * that has already closed.  Advanced only under the round claim; read by the
   * serve path at any time.  (Under WB the home holds no bytes and the
   * field idles — there the owner's own buffer version IS the axis, because
   * ownership migration carries it.) */
#ifdef __cplusplus
  uint64_t hver;
#else
  _Atomic uint64_t hver;
#endif
  arts_lf_stack_t pub_queue;              /* releases awaiting a round */
  struct arts_db_inv_pub_s *round_entries; /* the open round's drained batch */
  struct arts_rank_bitset_s roster;        /* copy roster (INV targets) */
  struct arts_rank_bitset_s cached_ranks;  /* destroy fan-out roster */

  /* GPU staging fields (full arts_db_s alloc; unused on the CPU INV path). */
  volatile unsigned int reader;
  volatile unsigned int writer;
  volatile unsigned int version;
  unsigned int time_stamp;
} ARTS_ALIGNED_MAX;

/* ── pure arbiters (CAS-retry loop bodies; see arbiters.c) ──────────────── */
/* Both arbiters are pure functions of one word, run inside a CAS-retry loop,
 * and are identical under both write policies: the reader word decides fetch /
 * park / publish / invalidate, and the directory word decides round mutual
 * exclusion and ack accounting.  Neither knows anything about ownership —
 * that is the grant plane's, and it does not live in a word. */

/* arg: ACQ_RO takes the parking node's pool index; the rest ignore it. */
#define INV_CACHE_OP_ACQ_RO 0
#define INV_CACHE_OP_DELIVER 1    /* a read reply landed (publish attempt) */
#define INV_CACHE_OP_INVALIDATE 2
#define INV_CACHE_OP_KILL_PURGE 3 /* reserved invalidate after the kill serve */

#define INV_CACHE_ACT_NONE 0
#define INV_CACHE_ACT_SELF_SERVE 1   /* covering VALID copy: pure load, run */
#define INV_CACHE_ACT_SEND_RO 2      /* opened the fetch: park + send */
#define INV_CACHE_ACT_PARK 3         /* chained onto the in-flight fetch */
#define INV_CACHE_ACT_PUBLISH 4      /* installed: serve the grabbed chain */
#define INV_CACHE_ACT_PUBLISH_KILL 5 /* + reserved purge + owed ack after */
#define INV_CACHE_ACT_DROP 6         /* superseded reply: discard */
#define INV_CACHE_ACT_PURGE_ACK 7    /* INVALIDATE on VALID: purged, ack */
#define INV_CACHE_ACT_KILL_MARKED 8  /* INVALIDATE on REQ: ack owed */
#define INV_CACHE_ACT_NOOP_ACK 9     /* INVALIDATE idempotent: ack */

uint64_t inv_cache_compute_next(uint64_t cur, int op, uint32_t self_idx,
                                uint32_t *out_action);

#define INV_DIR_OP_ROUND_CLAIM 0 /* open an invalidation round */
#define INV_DIR_OP_ACKS_ARM 1    /* under the claim: arm the ack count */
#define INV_DIR_OP_ACK_DEC 2
#define INV_DIR_OP_ROUND_CLOSE 3

#define INV_DIR_ACT_NONE 0
#define INV_DIR_ACT_CLAIMED 1
#define INV_DIR_ACT_CLOSE 2 /* acks reached 0 with the round open */

uint64_t inv_dir_compute_next(uint64_t cur, int op, unsigned int arg,
                              uint32_t *out_action);

/* ── shared machinery the write-policy TUs call (inv/directory.c) ──────────── */
/* Reader chain-node pool: a node is written before the word CAS that links it
 * and read only by the committer that grabbed the chain. */
uint32_t inv_waiter_alloc(struct arts_db_cache_s *c);
void inv_waiter_free(struct arts_db_cache_s *c, uint32_t idx);
struct arts_db_inv_waiter_s *inv_waiter_ptr(struct arts_db_cache_s *c,
                                            uint32_t idx);
/* Wake a grabbed chain: the committer's continuation, however late it runs. */
void inv_serve_chain(struct arts_db_cache_s *cache, uint32_t head,
                     bool serialized);
/* Drive the invalidation round: called by every producer (a publish push, a
 * close re-arm).  A claim that raced past the work it saw closes empty. */
void inv_home_round_try_open(struct arts_db_s *db);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_COHERENCE_INV_TYPES_H */
