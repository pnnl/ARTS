/* SPDX-License-Identifier: Apache-2.0
 *
 * LOCK protocol cache/db layout.
 *
 * Selected by arts/coherence/types.h when ARTS_PROTOCOL_LOCK is defined.
 * The LOCK protocol uses a single lock_state word on the home rank to
 * serialize all access-mode transitions.  There is no ownership transfer
 * (no invalidate push, no WRITEBACK_AND_TRANSFER); the home simply grants
 * or queues each requester and the requester writes back (RW) or returns
 * silently (RO) at release time.
 *
 * @note Internal header.  User code should include @c arts.h.
 */

#ifndef ARTS_COHERENCE_LOCK_TYPES_H
#define ARTS_COHERENCE_LOCK_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/coherence/types_common.h"
#include "arts/rank_bitset.h"
#include <stdint.h>

/* ── EAGER cache_state ────────────────────────────────────────────────────
 * Per-rank single 64-bit coherence word (EAGER timing only).
 * Layout: [ rw_state:2 (63..62) | ro_state:2 (61..60) |
 *           rw_count:30 (59..30) | ro_count:30 (29..0) ]
 *
 *   rw_state/ro_state ∈ {IDLE, REQ (request sent to home), GRANT (grant held)}
 *   rw_count/ro_count = #RW/#RO EDTs on this rank acquired-not-released.
 *
 * The counts are bumped by fetch_add(CACHE_RW_UNIT/CACHE_RO_UNIT); those
 * additions never touch the upper 4 state bits.  (GRANT,GRANT) is unreachable:
 * home never grants both to the same rank in EAGER. */
#ifdef ARTS_TIMING_EAGER
#define CACHE_ST_IDLE 0u
#define CACHE_ST_REQ 1u
#define CACHE_ST_GRANT 2u
#define CACHE_CNT_BITS 30
#define CACHE_CNT_MASK ((uint64_t)0x3fffffffULL)
#define CACHE_RO_CNT(s) ((uint32_t)((s) & CACHE_CNT_MASK))
#define CACHE_RW_CNT(s) ((uint32_t)(((s) >> CACHE_CNT_BITS) & CACHE_CNT_MASK))
#define CACHE_RO_ST(s) ((uint32_t)(((s) >> 60) & 0x3ULL))
#define CACHE_RW_ST(s) ((uint32_t)(((s) >> 62) & 0x3ULL))
#define CACHE_MAKE(rws, ros, wc, rc)                                           \
  (((uint64_t)((rws) & 0x3ULL) << 62) | ((uint64_t)((ros) & 0x3ULL) << 60) |   \
   (((uint64_t)(wc) & CACHE_CNT_MASK) << CACHE_CNT_BITS) |                     \
   ((uint64_t)(rc) & CACHE_CNT_MASK))
#define CACHE_RW_UNIT                                                          \
  ((uint64_t)1 << CACHE_CNT_BITS)   /* fetch_add → rw_count++ */
#define CACHE_RO_UNIT ((uint64_t)1) /* fetch_add → ro_count++ */
#endif                              /* ARTS_TIMING_EAGER */

/* ── LAZY cache_state ─────────────────────────────────────────────────────
 * Per-rank (owner) single 64-bit coherence word (LAZY timing only).
 * Layout (MSB→LSB):
 *   [ spare:1 (63) | owner:1 (62) | rw_st:2 (61..60) | ro_st:2 (59..58) |
 *     migrate_target:14 (57..44) | wc:22 (43..22) | rc:22 (21..0) ]
 *
 *   owner-bit  = 1 when this node holds data ownership (fast-path RW/RO).
 *   rw_st/ro_st ∈ {IDLE=0, REQ=1, GRANT=2} — local RW/RO protocol state.
 *   migrate_target = 14-bit rank of the next RW owner (set by home FORWARD);
 *                    ARTS_LOCK_NO_TARGET (0x3FFF) = no pending migration.
 *   wc/rc      = #RW/#RO EDTs on this rank acquired-not-released (local EDT
 *                count; 22 bits → ≥4 M concurrent EDTs, well above any limit).
 *
 * Single-word CAS discipline: owner-bit + migrate_target are cleared atomically
 * with the wc 0-edge in cache_lazy_compute_next (REL_RW path), so migration
 * needs no split-decrement or separate flag.
 *
 * Note: under LAZY the home does NOT hold canonical data — the current owner
 * (owner-bit set) does.  home.cache.buffer is unused by the LOCK-LAZY path. */
#ifdef ARTS_TIMING_LAZY
#define CACHE_ST_IDLE 0u
#define CACHE_ST_REQ 1u
#define CACHE_ST_GRANT 2u

/* Field widths and shifts. */
#define CACHE_LAZY_RC_BITS 22
#define CACHE_LAZY_WC_BITS 22
#define CACHE_LAZY_MT_BITS 14 /* migrate_target = ARTS_GUID_RANK_BITS */
#define CACHE_LAZY_ST_BITS 2  /* rw_st / ro_st */
#define CACHE_LAZY_OW_BITS 1  /* owner bit */

#define CACHE_LAZY_RC_SHIFT 0
#define CACHE_LAZY_WC_SHIFT                                                    \
  (CACHE_LAZY_RC_SHIFT + CACHE_LAZY_RC_BITS) /* 22                             \
                                              */
#define CACHE_LAZY_MT_SHIFT                                                    \
  (CACHE_LAZY_WC_SHIFT + CACHE_LAZY_WC_BITS) /* 44                             \
                                              */
#define CACHE_LAZY_RO_ST_SHIFT                                                 \
  (CACHE_LAZY_MT_SHIFT + CACHE_LAZY_MT_BITS) /* 58 */
#define CACHE_LAZY_RW_ST_SHIFT                                                 \
  (CACHE_LAZY_RO_ST_SHIFT + CACHE_LAZY_ST_BITS) /* 60 */
#define CACHE_LAZY_OW_SHIFT                                                    \
  (CACHE_LAZY_RW_ST_SHIFT + CACHE_LAZY_ST_BITS) /* 62 */

#define CACHE_LAZY_RC_MASK ((uint64_t)((1ULL << CACHE_LAZY_RC_BITS) - 1))
#define CACHE_LAZY_WC_MASK ((uint64_t)((1ULL << CACHE_LAZY_WC_BITS) - 1))
#define CACHE_LAZY_MT_MASK ((uint64_t)((1ULL << CACHE_LAZY_MT_BITS) - 1))
#define CACHE_LAZY_ST_MASK ((uint64_t)0x3ULL)
#define CACHE_LAZY_OW_MASK ((uint64_t)0x1ULL)

/* Getters. */
#define CACHE_RO_CNT(s)                                                        \
  ((uint32_t)(((s) >> CACHE_LAZY_RC_SHIFT) & CACHE_LAZY_RC_MASK))
#define CACHE_RW_CNT(s)                                                        \
  ((uint32_t)(((s) >> CACHE_LAZY_WC_SHIFT) & CACHE_LAZY_WC_MASK))
#define CACHE_MIGRATE_TARGET(s)                                                \
  ((uint32_t)(((s) >> CACHE_LAZY_MT_SHIFT) & CACHE_LAZY_MT_MASK))
#define CACHE_RO_ST(s)                                                         \
  ((uint32_t)(((s) >> CACHE_LAZY_RO_ST_SHIFT) & CACHE_LAZY_ST_MASK))
#define CACHE_RW_ST(s)                                                         \
  ((uint32_t)(((s) >> CACHE_LAZY_RW_ST_SHIFT) & CACHE_LAZY_ST_MASK))
#define CACHE_OWNER(s)                                                         \
  ((uint32_t)(((s) >> CACHE_LAZY_OW_SHIFT) & CACHE_LAZY_OW_MASK))

/* ro_leased flag — the single spare bit (63).  Set when this rank's current RO
 * grant arrived from the home via DELIVER(RO) (a home-COUNTED lease), as
 * opposed to an owner-fast-path local RO (never counted at home).  The
 * owner-local RO release path takes owner_try_execute and sends NO RO_RETURN;
 * without this flag a rank that requested an RO and THEN became the owner (a
 * held reader promoted to owner at the RW→RO flip) would self-serve and
 * self-release its leased RO without ever returning the home's r count → the
 * RO→RW flip deadlocks.  With the flag, the RO release sends a RO_RETURN
 * whenever the released RO was leased, regardless of owner-bit, keeping the
 * home's r balanced.  RO grants on a rank are homogeneous per generation (the
 * first RO acquire fixes ro_st to GRANT via the owner fast-path, or to REQ via
 * a home request, and later RO acquires join that state), so a single flag
 * suffices. */
#define CACHE_LAZY_LEASED_SHIFT 63
#define CACHE_LAZY_LEASED_MASK ((uint64_t)1ULL << CACHE_LAZY_LEASED_SHIFT)
#define CACHE_LEASED(s) (((s) & CACHE_LAZY_LEASED_MASK) != 0u)

/* Constructor (all fields). */
#define CACHE_MAKE(rws, ros, wc, rc)                                           \
  CACHE_MAKE_FULL(0u, (rws), (ros), ARTS_LOCK_NO_TARGET, (wc), (rc))

/* CACHE_MAKE_FULL: full constructor including owner-bit and migrate_target. */
#define CACHE_MAKE_FULL(own, rws, ros, mt, wc, rc)                             \
  (((uint64_t)((own) & CACHE_LAZY_OW_MASK) << CACHE_LAZY_OW_SHIFT) |           \
   ((uint64_t)((rws) & CACHE_LAZY_ST_MASK) << CACHE_LAZY_RW_ST_SHIFT) |        \
   ((uint64_t)((ros) & CACHE_LAZY_ST_MASK) << CACHE_LAZY_RO_ST_SHIFT) |        \
   ((uint64_t)((mt) & CACHE_LAZY_MT_MASK) << CACHE_LAZY_MT_SHIFT) |            \
   ((uint64_t)((wc) & CACHE_LAZY_WC_MASK) << CACHE_LAZY_WC_SHIFT) |            \
   ((uint64_t)((rc) & CACHE_LAZY_RC_MASK) << CACHE_LAZY_RC_SHIFT))

/* Sentinel for "no pending migration target". */
#define ARTS_LOCK_NO_TARGET                                                    \
  ((uint32_t)((1u << CACHE_LAZY_MT_BITS) - 1)) /* 0x3FFF */
#endif                                         /* ARTS_TIMING_LAZY */

/* ── EAGER home lock_state ────────────────────────────────────────────────
 * Layout: [ state_bit:1 (62) | w:31 (61..31) | r:31 (30..0) ]
 *
 * state_bit is meaningful only when w>0 && r>0:
 *   LOCK_PHASE_BIT_RW (0) = RW phase held / RO waiters queued
 *   LOCK_PHASE_BIT_RO (1) = RO phase held / RW waiters queued
 * When w==0 || r==0 the state_bit is normalized to 0.
 * w = #RW participant ranks; r = #RO participant ranks. */
#ifdef ARTS_TIMING_EAGER
#define LOCK_STATE_R_BITS 31
#define LOCK_STATE_W_BITS 31
#define LOCK_STATE_R_MASK ((uint64_t)0x7fffffffULL)
#define LOCK_STATE_W_MASK ((uint64_t)0x7fffffffULL)
#define LOCK_STATE_R(s) ((uint32_t)((s) & LOCK_STATE_R_MASK))
#define LOCK_STATE_W(s)                                                        \
  ((uint32_t)(((s) >> LOCK_STATE_R_BITS) & LOCK_STATE_W_MASK))
#define LOCK_STATE_BIT(s)                                                      \
  ((uint32_t)(((s) >> (LOCK_STATE_R_BITS + LOCK_STATE_W_BITS)) & 0x1ULL))
#define LOCK_MAKE_STATE(bit, w, r)                                             \
  (((uint64_t)((bit) & 0x1ULL) << (LOCK_STATE_R_BITS + LOCK_STATE_W_BITS)) |   \
   (((uint64_t)(w) & LOCK_STATE_W_MASK) << LOCK_STATE_R_BITS) |                \
   ((uint64_t)(r) & LOCK_STATE_R_MASK))
#define LOCK_PHASE_BIT_RW 0u
#define LOCK_PHASE_BIT_RO 1u
#endif /* ARTS_TIMING_EAGER */

/* ── LAZY home lock_state ─────────────────────────────────────────────────
 * Layout (MSB→LSB): [ phase:2 (63..62) | owner:14 (61..48) |
 *                     w:24 (47..24) | r:24 (23..0) ]
 * (= 2+14+24+24 = 64 bits, no spare)
 *
 *   phase  ∈ {IDLE=0, RW=1, RO=2} — current arbitration phase.
 *   owner  = 14-bit rank of the current data-owning node (ARTS_GUID_RANK_BITS).
 *            Packing owner here lets CONFIRM set owner+w--+phase in ONE CAS.
 *   w      = count of pending RW participants (held + queued).
 *   r      = count of outstanding RO grants (issued but not yet returned);
 *            the RO→RW flip fires at r==0. r may transiently exceed the number
 *            of distinct reader ranks under network reorder (commutative ±1
 *            accounting tolerates this). */
#ifdef ARTS_TIMING_LAZY
#define LOCK_LAZY_R_BITS 24
#define LOCK_LAZY_W_BITS 24
#define LOCK_LAZY_OWNER_BITS 14 /* = ARTS_GUID_RANK_BITS */
#define LOCK_LAZY_PHASE_BITS 2

#define LOCK_LAZY_R_SHIFT 0
#define LOCK_LAZY_W_SHIFT (LOCK_LAZY_R_SHIFT + LOCK_LAZY_R_BITS)     /* 24 */
#define LOCK_LAZY_OWNER_SHIFT (LOCK_LAZY_W_SHIFT + LOCK_LAZY_W_BITS) /* 48 */
#define LOCK_LAZY_PHASE_SHIFT                                                  \
  (LOCK_LAZY_OWNER_SHIFT + LOCK_LAZY_OWNER_BITS) /* 62 */

#define LOCK_LAZY_R_MASK ((uint64_t)((1ULL << LOCK_LAZY_R_BITS) - 1))
#define LOCK_LAZY_W_MASK ((uint64_t)((1ULL << LOCK_LAZY_W_BITS) - 1))
#define LOCK_LAZY_OWNER_MASK ((uint64_t)((1ULL << LOCK_LAZY_OWNER_BITS) - 1))
#define LOCK_LAZY_PHASE_MASK ((uint64_t)0x3ULL)

/* Phase constants. */
#define LOCK_PHASE_IDLE 0u
#define LOCK_PHASE_RW 1u
#define LOCK_PHASE_RO 2u

/* Getters. */
#define LOCK_R(s) ((uint32_t)(((s) >> LOCK_LAZY_R_SHIFT) & LOCK_LAZY_R_MASK))
#define LOCK_W(s) ((uint32_t)(((s) >> LOCK_LAZY_W_SHIFT) & LOCK_LAZY_W_MASK))
#define LOCK_OWNER(s)                                                          \
  ((uint32_t)(((s) >> LOCK_LAZY_OWNER_SHIFT) & LOCK_LAZY_OWNER_MASK))
#define LOCK_PHASE(s)                                                          \
  ((uint32_t)(((s) >> LOCK_LAZY_PHASE_SHIFT) & LOCK_LAZY_PHASE_MASK))

/* Constructor. */
#define LOCK_MAKE(phase, owner, w, r)                                          \
  (((uint64_t)((phase) & LOCK_LAZY_PHASE_MASK) << LOCK_LAZY_PHASE_SHIFT) |     \
   ((uint64_t)((owner) & LOCK_LAZY_OWNER_MASK) << LOCK_LAZY_OWNER_SHIFT) |     \
   ((uint64_t)((w) & LOCK_LAZY_W_MASK) << LOCK_LAZY_W_SHIFT) |                 \
   ((uint64_t)((r) & LOCK_LAZY_R_MASK) << LOCK_LAZY_R_SHIFT))
#endif /* ARTS_TIMING_LAZY */

/* ── home lock_state transition ops + grant codes (EAGER) ────────────────
 * lock_compute_next(cur, op, &grant) is the home arbiter, a pure function of
 * the single lock_state word run inside a CAS-retry loop.  Exposed here (not
 * just in home.c) so the acquire/release handlers can run the SAME arbiter
 * LOCALLY when home == self — a local hit runs the handler logic directly, with
 * no wire / loopback round (HPC: a local op must not touch the network layer).
 */
#ifdef ARTS_TIMING_EAGER
#define LOCK_OP_RW_ACQ 0
#define LOCK_OP_RO_ACQ 1
#define LOCK_OP_RW_REL 2
#define LOCK_OP_RO_REL 3
#define LOCK_GRANT_NONE 0
#define LOCK_GRANT_ONE_RW 1
#define LOCK_GRANT_ALL_RO 2
uint64_t lock_compute_next(uint64_t cur, int op, uint32_t *out_grant);
#endif /* ARTS_TIMING_EAGER */

/* ── home lock_state transition ops + action codes (LAZY) ────────────────
 * lock_lazy_compute_next(cur, op, &action) is the LAZY home arbiter.
 * Same CAS-retry pattern as EAGER's lock_compute_next. */
#ifdef ARTS_TIMING_LAZY
#define LOCK_OP_RW_ACQ 0
#define LOCK_OP_RO_ACQ 1
#define LOCK_OP_CONFIRM 2 /* new owner confirms migration complete */
#define LOCK_OP_RO_RET 3  /* reader returns RO (r--) */

#define LOCK_ACTION_NONE 0
#define LOCK_ACTION_FORWARD_MIGRATE                                            \
  1 /* home→owner: FORWARD(migrate→target)                                 \
     */
#define LOCK_ACTION_FORWARD_SERVE_ONE                                          \
  2 /* home→owner: FORWARD(serve→1 reader) */
#define LOCK_ACTION_FORWARD_SERVE_ALL                                          \
  3 /* home→owner: FORWARD(serve→held RO)                                  \
     */
/* Home arbiter — pure function of the single lock_state word, run in a
 * CAS-retry loop.  new_owner is read only by LOCK_OP_CONFIRM (sets owner in the
 * same word as w--/phase); ignored by the other ops.  The FORWARD recipient is
 * OWNER() of the returned word; the migrate target / served reader(s) are
 * resolved by the caller from rw_waiters/ro_waiters (a pure arbiter cannot peek
 * the queues). */
uint64_t lock_lazy_compute_next(uint64_t cur, int op, unsigned int new_owner,
                                uint32_t *out_action);
#endif /* ARTS_TIMING_LAZY */

/* ── cache_state transition ops + actions (EAGER) ────────────────────────
 * cache_compute_next(cur, op, &action) is the cache-side analogue of the home's
 * lock_compute_next: a pure function of the current word, run inside a
 * CAS-retry loop.  The ACQ_* ops decide the request/join state ONLY (the count
 * was bumped by the separate fetch_add cas1 BEFORE the push); the REL_* ops
 * carry the count-- inside the CAS (the decrement and the 0-edge state change
 * must be atomic together). */
#ifdef ARTS_TIMING_EAGER
#define CACHE_OP_ACQ_RW 0
#define CACHE_OP_ACQ_RO 1
#define CACHE_OP_GRANT_RW 2
#define CACHE_OP_GRANT_RO 3
#define CACHE_OP_REL_RW 4
#define CACHE_OP_REL_RO 5

#define CACHE_ACT_NONE 0
#define CACHE_ACT_SEND_RW 1    /* send RW REQUEST to home */
#define CACHE_ACT_SEND_RO 2    /* send RO REQUEST to home */
#define CACHE_ACT_DRAIN_RW 3   /* serve rw_pending */
#define CACHE_ACT_DRAIN_BOTH 4 /* serve rw_pending + ro_pending (RW grant) */
#define CACHE_ACT_DRAIN_RO 5   /* serve ro_pending */
#define CACHE_ACT_REL_RW 6     /* send RW RELEASE (writeback, ACK-gated) */
#define CACHE_ACT_REL_RO 7     /* send RO RELEASE (notify) */

/* Defined in arbiters.c (included by eager.c); exposed non-static for the
 * cache-state model test (mirrors lock_compute_next's exposure). */
uint64_t cache_compute_next(uint64_t cur, int op, uint32_t *out_action);
#endif /* ARTS_TIMING_EAGER */

/* ── cache_state transition ops + actions (LAZY) ─────────────────────────
 * cache_lazy_compute_next(cur, op, &action) is the LAZY owner-cache arbiter. */
#ifdef ARTS_TIMING_LAZY
#define CACHE_OP_ACQ_RW 0
#define CACHE_OP_ACQ_RO 1
/* No FORWARD ops: FORWARD-migrate / FORWARD-serve do not transit
 * cache_lazy_compute_next — arts_handler_db_lock_forward CASes migrate_target /
 * pushes ro_serve directly (the target/reader is not a pure-function output).
 */
#define CACHE_OP_REL_RW 4 /* local RW release: wc-- + 0-edge → migrate/noop */
#define CACHE_OP_REL_RO 5 /* local RO release: rc-- */

#define CACHE_ACT_NONE 0
#define CACHE_ACT_SEND_RW 1    /* send RW REQUEST to home */
#define CACHE_ACT_SEND_RO 2    /* send RO REQUEST to home */
#define CACHE_ACT_DRAIN_RW 3   /* serve rw_pending */
#define CACHE_ACT_DRAIN_BOTH 4 /* serve rw_pending + ro_pending */
#define CACHE_ACT_DRAIN_RO 5   /* serve ro_pending */
#define CACHE_ACT_MIGRATE 6 /* wc 0-edge + migrate_target set: send DELIVER */
#define CACHE_ACT_REL_RO 7  /* rc-- only (no wire needed) */

uint64_t cache_lazy_compute_next(uint64_t cur, int op, uint32_t *out_action);
#endif /* ARTS_TIMING_LAZY */

/* Pending waiter (parked EDT dep slot) used in both ro_pending and rw_pending.
 * Same shape as MRNEW's RW waiter for consistency. */
struct arts_db_lock_waiter_s {
  arts_lf_link_t link; /* FIRST — required by arts_lf_stack_t */
  arts_guid_t edt_guid;
  unsigned int slot;
};

/* arts_home_lockreq_node_s and arts_home_lockreq_queue_s are defined in
 * coherence/types_common.h (via mrnew/types.h when MRNEW) — but for LOCK
 * they are defined in mrnew/types.h only when MRNEW.  The LOCK build defines
 * its own arts_home_lockreq_queue_s here, matching the Vyukov MPSC shape
 * used by mrnew/types.h so the same home.h declarations apply. */
#ifdef __cplusplus
struct arts_home_lockreq_node_s {
  struct arts_home_lockreq_node_s *next;
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv; /* requester's grant/deliver landing */
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
  struct arts_rdzv_landing_s rdzv; /* requester's grant/deliver landing */
};

struct arts_home_lockreq_queue_s {
  _Atomic(struct arts_home_lockreq_node_s *) tail; /* producer end */
  _Atomic(struct arts_home_lockreq_node_s *) head; /* consumer end */
  struct arts_home_lockreq_node_s stub;            /* permanent sentinel */
};
#endif

/* Per-rank DB cache for the LOCK protocol.
 * The layout is shared between EAGER and LAZY; LAZY adds ro_serve (the list
 * of reader ranks the owner must serve during the RO phase).  EAGER omits it
 * so sizeof(arts_db_cache_s) is unchanged in EAGER builds. */
#ifdef __cplusplus
struct arts_db_cache_s {
  uint64_t cache_state; /* single coherence word — see CACHE_* */
  /* Single stable backing store: allocated once (first grant/create), written
   * in place by arts_db_buf_write_inplace on every later grant/writeback, freed
   * only at destroy.  The address never moves, so DBs holding internal
   * self-pointers stay valid.  Under LAZY, data stays with the owner; the
   * home-rank cache.buffer is unused by the coherence path. */
  arts_atomic_shared_ptr_t buffer;
  arts_lockfree_pool_t
      buf_freelist; /* per-DB recycled-buffer pool (push on deleter, pull on
                       install); unbounded, drained at cache destroy */
  arts_lf_stack_t pending_snapshot; /* unused by LOCK; kept for common_init */
  arts_lf_stack_t ro_pending;       /* parked RO EDT waiters on this rank */
  arts_lf_stack_t rw_pending;       /* parked RW EDT waiters on this rank */
#ifdef ARTS_TIMING_LAZY
  /* Reader ranks the owner must serve during the RO phase.  Home pushes via
   * FORWARD; the owner drains (DELIVER copy to each reader) when wc reaches 0.
   * This list cannot be packed into the single cache_state word. */
  arts_lf_stack_t ro_serve;
  /* The pending migrate target's landing (its stable buffer), written by the
   * FORWARD(migrate) handler BEFORE the CAS that publishes migrate_target
   * (single in-flight migration per owner — home serializes by CONFIRM), read
   * by whichever actor ships the DELIVER on the wc 0-edge. */
  struct arts_rdzv_landing_s migrate_rdzv;
#else
  /* Home's writeback landing for THIS grant's release (advertised in the
   * grant, 1:1 with the eventual RW release).  Written by the grant handler
   * before any local writer runs; consumed by the single ACK-gated releaser. */
  struct arts_rdzv_landing_s home_wb_rdzv;
#endif
  arts_guid_t db_guid;
  uint64_t db_size;
};
#else
struct arts_db_cache_s {
  _Atomic uint64_t cache_state;
  arts_atomic_shared_ptr_t buffer;
  arts_lockfree_pool_t
      buf_freelist; /* per-DB recycled-buffer pool (push on deleter, pull on
                       install); unbounded, drained at cache destroy */
  arts_lf_stack_t pending_snapshot;
  arts_lf_stack_t ro_pending;
  arts_lf_stack_t rw_pending;
#ifdef ARTS_TIMING_LAZY
  arts_lf_stack_t ro_serve; /* RO-phase serve list (owner only; LAZY only) */
  struct arts_rdzv_landing_s migrate_rdzv; /* pending migrate target landing */
#else
  struct arts_rdzv_landing_s home_wb_rdzv; /* grant's writeback landing */
#endif
  arts_guid_t db_guid;
  uint64_t db_size;
};
#endif

/** Internal DataBlock descriptor (LOCK protocol).
 *
 *  The per-rank coherence cache (struct arts_db_cache_s) is embedded by value
 *  as the FIRST member: the cb object the route_table wraps is the db_s, and
 *  cache-to-db_s recovery is a zero-cost cast.  Non-home ranks allocate a
 *  cache-only footprint (arts_db_cache_stub_size() = offsetof(lock_state))
 *  that includes home_initialized but omits the home-directory fields. */
struct arts_db_s {
  struct arts_db_cache_s
      cache;               /**< FIRST — coherence state (embedded by value). */
  arts_db_types_t db_type; /**< Storage subtype; in bounds on cache stubs. */
  /* Home-directory fields — present only on the GUID home rank.  Cache-only
   * stubs (arts_db_cache_stub_size()) end at lock_state (the first home field),
   * so they include home_initialized but omit every home-arm field. */
  bool home_initialized; /**< One-shot init sentinel (arts_db_home_init). */
#ifdef __cplusplus
  uint64_t lock_state;
#else
  _Atomic uint64_t lock_state;
#endif
  struct arts_home_lockreq_queue_s rw_waiters; /* Vyukov MPSC, pop-one */
  arts_lf_stack_t ro_waiters;                  /* Treiber, XCHG drain */
  struct arts_rank_bitset_s cached_ranks;      /* destroy fan-out roster */
  /* GPU staging fields (full arts_db_s alloc; unused on the CPU LOCK path). */
  volatile unsigned int reader;
  volatile unsigned int writer;
  volatile unsigned int version;
  unsigned int time_stamp;
} ARTS_ALIGNED_MAX;

#ifdef __cplusplus
}
#endif

#endif /* ARTS_COHERENCE_LOCK_TYPES_H */
