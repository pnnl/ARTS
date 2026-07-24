/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol (write-invalidate, EAGER timing) type layouts.
 *
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

/* ── parked waiter (chain node, index-addressed) ────────────────────────── */
/* Nodes live in a per-DB grow-only pool; the chain link and the free list
 * both use pool indices (0 = none), so a head fits the word's 18-bit field.
 * A node is written (guid/slot/next) before the CAS that links it and is
 * read only by the committer that grabbed the chain — single-owner after
 * the grab, freed back to the pool after its serve. */
struct arts_db_msi_waiter_s {
  uint32_t next;         /* pool index of the next node; 0 = end */
  unsigned int slot;     /* dep slot of the parked acquire */
  arts_guid_t edt_guid;  /* parked EDT (guid-addressed idempotent serve) */
};

#define MSI_WAITER_CHUNK_CAP 256u /* nodes per pool chunk (chunked growth) */
#define MSI_WAITER_IDX_MAX MSI_CACHE_HEAD_MASK

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
/* The writeback axis is a DEDICATED counter, decoupled from the buffer's
 * install version: buffer versions are pure install machinery (each install
 * = previous+1, locally monotone), while wb_next numbers this tenure's
 * releases from the grant's base (seeded base+1 by the grant/create
 * handler, fetch_add per release).  Sharing buf->version for both would
 * let local install counts overtake the grant base and stale-reject the
 * grant's own install. */
#ifdef __cplusplus
struct arts_db_cache_s {
  uint64_t cache_state; /* single coherence word — see MSI_CACHE_* */
  arts_atomic_shared_ptr_t buffer;
  arts_lockfree_pool_t buf_freelist;
  arts_lf_stack_t pending_snapshot; /* pre-publication read holds (home) */
  struct arts_db_msi_waiter_pool_s waiters; /* chain-node pool (index) */
  uint64_t wb_next; /* next release's writeback version (grant-seeded) */
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
  _Atomic uint64_t wb_next;
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
  uint64_t hver;
  bool opening_pending;
#else
  _Atomic uint64_t dir_state; /* single directory word — see MSI_DIR_* */
  /* Canonical home version (the protocol axis).  Decoupled from the
   * installed buffer's version lane: a same-rank release moves no data
   * (home buffer == the tenure's working buffer), so its round advances
   * hver WITHOUT an install swap — swapping would orphan the buffer the
   * tenure's writers keep writing.  Written only under the round claim;
   * read (serve stamps, grant bases) at any time. */
  _Atomic uint64_t hver;
  _Atomic bool opening_pending; /* tenure-opening round re-arm latch (XCHG) */
#endif
  struct arts_home_lockreq_queue_s rw_waiters; /* Vyukov MPSC, pop-one */
  arts_lf_stack_t wb_queue;                    /* Treiber, batch drain */
  struct arts_rank_bitset_s roster;            /* copy roster (INV targets) */
  struct arts_rank_bitset_s cached_ranks;      /* destroy fan-out roster */
  /* Single-slot state of the (at most one) open round / pending grant —
   * written only under the round claim / the GRANTING claim. */
  struct arts_rdzv_landing_s grant_rdzv; /* grantee's deliver landing */
  struct arts_db_msi_wb_s *round_entries; /* open round's drained batch */
  /* GPU staging fields (full arts_db_s alloc; unused on the CPU MSI path). */
  volatile unsigned int reader;
  volatile unsigned int writer;
  volatile unsigned int version;
  unsigned int time_stamp;
} ARTS_ALIGNED_MAX;

/* ── pure arbiters (CAS-retry loop bodies; see arbiters.c) ──────────────── */
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

#ifdef __cplusplus
}
#endif

#endif /* ARTS_COHERENCE_MSI_TYPES_H */
