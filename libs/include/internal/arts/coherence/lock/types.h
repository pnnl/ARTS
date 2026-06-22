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

/* ── cache_state: per-rank single 64-bit coherence word ──────────────────
 * Mirrors the home lock_state design (single atomic + pure compute_next +
 * CAS-retry).  Replaces the old three-word cache state (held_mode +
 * local_count + request_in_flight), whose mutual non-atomicity was the cache
 * race surface.  Layout:
 *
 *   [ rw_state:2 (62..63) | ro_state:2 (60..61) | rw_count:30 (30..59) |
 *     ro_count:30 (0..29) ]
 *
 *   rw_state/ro_state ∈ {IDLE, REQ (request sent to home), GRANT (grant held)}
 *   rw_count/ro_count = #RW/#RO EDTs on this rank acquired-not-released
 *     (local per-EDT count; the rank sends ONE request per round on IDLE→REQ
 *      and ONE release per round on the count 0-edge).
 *
 * The counts are bumped by adding CACHE_RW_UNIT / CACHE_RO_UNIT (a plain
 * fetch_add/sub: they never touch the 2-bit state fields).  Reachable
 * (rw_state,ro_state) combos are 8; (GRANT,GRANT) is impossible (home never
 * grants both to one rank — see the single-word design doc). */
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

/* lock_state word layout:
 *   [state_bit:1 (bit 62) | w:31 (bits 31..61) | r:31 (bits 0..30)]
 *
 * state_bit is meaningful only when w>0 && r>0:
 *   LOCK_PHASE_BIT_RW (0) = RW phase held / RO waiters queued
 *   LOCK_PHASE_BIT_RO (1) = RO phase held / RW waiters queued
 * When w==0 || r==0 the state_bit is normalized to 0.
 *
 * w = #RW participant ranks (held + waiting);
 * r = #RO participant ranks (held + waiting). */
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

/* ── home lock_state transition ops + grant codes ────────────────────────
 * lock_compute_next(cur, op, &grant) is the home arbiter, a pure function of
 * the single lock_state word run inside a CAS-retry loop.  Exposed here (not
 * just in home.c) so the acquire/release handlers can run the SAME arbiter
 * LOCALLY when home == self — a local hit runs the handler logic directly, with
 * no wire / loopback round (HPC: a local op must not touch the network layer).
 */
#define LOCK_OP_RW_ACQ 0
#define LOCK_OP_RO_ACQ 1
#define LOCK_OP_RW_REL 2
#define LOCK_OP_RO_REL 3
#define LOCK_GRANT_NONE 0
#define LOCK_GRANT_ONE_RW 1
#define LOCK_GRANT_ALL_RO 2
uint64_t lock_compute_next(uint64_t cur, int op, uint32_t *out_grant);

/* ── cache_state transition ops + actions ────────────────────────────────
 * cache_compute_next(cur, op, &action) is the cache-side analogue of the home's
 * lock_compute_next: a pure function of the current word, run inside a
 * CAS-retry loop.  The ACQ_* ops decide the request/join state ONLY (the count
 * was bumped by the separate fetch_add cas1 BEFORE the push); the REL_* ops
 * carry the count-- inside the CAS (the decrement and the 0-edge state change
 * must be atomic together). */
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

/* Defined in acquire.c; exposed non-static for the cache-state model test
 * (mirrors lock_compute_next's exposure). */
uint64_t cache_compute_next(uint64_t cur, int op, uint32_t *out_action);

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
};

struct arts_home_lockreq_queue_s {
  _Atomic(struct arts_home_lockreq_node_s *) tail; /* producer end */
  _Atomic(struct arts_home_lockreq_node_s *) head; /* consumer end */
  struct arts_home_lockreq_node_s stub;            /* permanent sentinel */
};
#endif

/* Per-rank DB cache for the LOCK protocol. */
#ifdef __cplusplus
struct arts_db_cache_s {
  uint64_t cache_state; /* single coherence word — see CACHE_* */
  /* Single stable backing store: allocated once (first grant/create), written
   * in place by arts_db_buf_write_inplace on every later grant/writeback, freed
   * only at destroy.  The address never moves, so DBs holding internal
   * self-pointers stay valid.  LOCK needs no buffer versioning: exclusive-lock
   * serialization means no reader touches the buffer while it is written. */
  arts_atomic_shared_ptr_t buffer;
  arts_lf_stack_t pending_snapshot; /* unused by LOCK; kept for common_init */
  arts_lf_stack_t ro_pending;       /* parked RO EDT waiters on this rank */
  arts_lf_stack_t rw_pending;       /* parked RW EDT waiters on this rank */
  arts_guid_t db_guid;
  uint64_t db_size;
};
#else
struct arts_db_cache_s {
  _Atomic uint64_t cache_state;
  arts_atomic_shared_ptr_t buffer;
  arts_lf_stack_t pending_snapshot;
  arts_lf_stack_t ro_pending;
  arts_lf_stack_t rw_pending;
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
