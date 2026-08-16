/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
******************************************************************************/
#ifndef ARTS_MEMORY_COHERENCE_RCU_TYPES_H
#define ARTS_MEMORY_COHERENCE_RCU_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file val/types.h
 * @brief VAL (Multi-Reader, Node-Exclusive-Writer) cache/db layout.
 *
 * Selected by arts/coherence/types.h when ARTS_PROTOCOL_WRF_VAL is NOT defined.
 * The WT vs WB write-policy variant is chosen here by ARTS_WRITE_POLICY_WB.  The
 * protocol-agnostic pieces (buffer, snapshot waiter, defines, container_of)
 * come from types_common.h; the of_cache/total_size/stub_size helpers live in
 * the dispatcher (types.h), after this header defines cache + db_s.
 *
 * @note Internal header.  User code should include @c arts.h.
 */

#include "arts/coherence/types_common.h"

/* ========================================================================= */
/** @addtogroup internal_db_structs
 *  @{ */


/*--- Per-DB home metadata ------------------------------------------------
 * Lives only on the rank that hosts a given DB (GUID home decides).  Holds
 * the directory state needed for ownership transfer and dedup.  The MPSC
 * queues are defined inline below (needed for struct embedding in arts_db_s).
 *
 * Vyukov MPSC queue node carrying a requester rank (home GRANT_REQUEST
 * queue). The embedded `next` pointer is owned by the queue (push/pop manage
 * it). Producers are foreign-rank GRANT_REQUEST handlers; the single
 * consumer is the home-side dispatcher holding the invalidate_in_flight baton.
 */
#ifdef __cplusplus
struct arts_home_grantreq_node_s {
  struct arts_home_grantreq_node_s *next;
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv; /* requester's transfer landing */
};

struct arts_home_grantreq_queue_s {
  struct arts_home_grantreq_node_s
      *tail; /* producer end (push exchanges here) */
  struct arts_home_grantreq_node_s *head; /* consumer end (pop advances here) */
  struct arts_home_grantreq_node_s stub;
};
#else
struct arts_home_grantreq_node_s {
  _Atomic(struct arts_home_grantreq_node_s *) next;
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv; /* requester's transfer landing */
};

struct arts_home_grantreq_queue_s {
  _Atomic(struct arts_home_grantreq_node_s *)
      tail; /* producer end (push exchanges here) */
  _Atomic(struct arts_home_grantreq_node_s *)
      head;                             /* consumer end (pop advances here) */
  struct arts_home_grantreq_node_s stub; /* permanent sentinel */
};
#endif

/*--- Per-rank DB cache ---------------------------------------------------
 * Every rank that has acquired or hosts a given DB has one of these.  The
 * cache is the runtime's coherence-protocol state: ownership, the live
 * buffer, parked waiters, and destroy lifecycle.
 *
 *   writer_count   flat ownership counter.  > 0 ⇒ this rank holds RW
 *                  ownership; 0 ⇒ invalidated.
 *   buffer         currently-installed buffer, an atomic shared_ptr slot;
 *                  readers acquire via acquire_buf's acquire-and-validate load.
 *   pending_snapshot  Treiber stack of snapshot-response reorder-buffer
 *                  waiters (case-3 push only; drained whole on next install).
 *
 * The home-directory fields (rw_holder, pending_rw, invalidate_in_flight,
 * cached_ranks, cached_version, ...) are NOT here — they are inlined
 * directly in the wrapping struct arts_db_s, after this cache + db_type, and
 * reached via arts_db_of_cache(cache).  Non-home ranks allocate a cache-only
 * footprint that stops before them. */
struct arts_db_cache_s {
  volatile unsigned int writer_count;
  arts_atomic_shared_ptr_t buffer;
  arts_lockfree_pool_t
      buf_freelist; /* per-DB recycled-buffer pool (push on deleter, pull on
                       install); unbounded, drained at cache destroy */
  arts_lf_stack_t pending_snapshot;
  /* db_guid stored here for symmetry with the protocol pseudocode —
   * acquire_remote_* needs it for the route_table_return_db pairing on
   * ARTS_DB_DESTROYED early-returns, where the cache is in scope but the
   * original guid argument has been lost in the call chain. */
  arts_guid_t db_guid;
  uint64_t db_size;
  /* New owner rank for the next ownership transfer, published by this round's
   * GRANT_INVALIDATE handler BEFORE it withdraws the sentinel.  Sentinel
   * ARTS_NO_PENDING_OWNER == no transfer pending.  The publish-before-
   * sentinel-withdraw ordering makes a separate transfer_pending flag
   * redundant: whichever actor drives writer_count to 0 (the INVALIDATE, or the
   * last release_rw) reads this field — a non-sentinel value names the target,
   * so that actor fires the commit-PROCEED to it and ships the transfer
   * (GRANT_RESPONSE in WB, PUBLISH_AND_TRANSFER/local in WT). Single
   * writer per round (home baton gate), so no atomic needed.  Shared by both
   * write policies (WT's INVALIDATE now carries new_owner too). */
  unsigned int incoming_new_owner;
  /* The pending new owner's transfer landing, published together with (and
   * under the same single-writer / publish-before-withdraw discipline as)
   * incoming_new_owner: the 0-edge actor PUTs the transfer payload here. */
  struct arts_rdzv_landing_s incoming_new_owner_rdzv;
#ifdef ARTS_WRITE_POLICY_WB
  /* WB: owner-side dedup map.  Allocated lazily on first ownership; preserved
   * across ownership transfer (GRANT_RESPONSE serializes it). */
  struct arts_rank_to_u64_map_s *cached_version;
  /* WB per-cache RW exclusivity machinery.  RW GRANT_REQUEST coalescing
   * flag — only the actor that CASes false->true sends GRANT_REQUEST;
   * same-node RW EDTs piggyback on the in-flight one and are picked up by
   * GRANT's drain. */
  volatile unsigned int grant_req_in_flight;
  /* Set (1) when a GRANT_RESPONSE installs the buffer on this rank but home
   * has not yet flipped rw_holder to us; cleared (0) when home's CONFIRM
   * arrives. While set, this rank holds the data + ownership sentinel for
   * accounting but must NOT run RW EDTs (their writes would be observable
   * before the directory names us — the stale-RO window). Gates both parked and
   * fresh RW acquires. */
  volatile unsigned int grant_unconfirmed;
  /* Treiber stack of RW waiters parked on this rank (order-free drain-all). */
  arts_lf_stack_t pending_rw;
#else
  /* WT: the PUBLISH ACK rendezvous is a stack-local sem_t created per
   * release_rw, matched by pointer identity (the &sem address rides the
   * PUBLISH packet and is echoed verbatim in the ACK).  Multiple concurrent
   * releases each get their own sem — no per-cache seq state. */
  /* WT per-cache RW exclusivity machinery.  RW GRANT_REQUEST coalescing
   * flag — only the actor that CASes false->true sends GRANT_REQUEST;
   * same-node RW EDTs piggyback on the in-flight one and are picked up by
   * GRANT's drain. */
  volatile unsigned int grant_req_in_flight;
  /* Owner-side dedup map.  WT never creates it (always NULL): home serves RO
   * via SNAPSHOT_REQUEST with the home->cached_version watermark, so the WT
   * owner→owner transfer ships an EMPTY map.  The field exists so the shared
   * arts_db_send_grant_response (coherence/grant.c) compiles for
   * both write policies (it gates the map build on this being non-NULL). */
  struct arts_rank_to_u64_map_s *cached_version;
  /* Treiber stack of RW waiters parked on this rank (order-free drain-all). */
  arts_lf_stack_t pending_rw;

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
#endif
  /* Kept at the tail so compiling the option in cannot shift the offsets
   * of the fields (and the inlined home directory beyond them) that every
   * acquire path touches. */
#ifdef ARTS_RO_REQUEST_COMBINING
  /* Remote-read request-combining window (see coherence/coherence.c).
   * snapshot_req_in_flight admits ONE outstanding snapshot request per cache
   * (0->1 CAS claims the window).  ro_combine accumulates waiters that arrive
   * while a request is in flight; ro_combine_group is the batch the current
   * request rides for — isolated from ro_combine in one exchange at send time,
   * resumed wholesale by the response terminal.  ro_combine_group is
   * single-actor (written by the window owner before the send, consumed by
   * the one response that ends that window), so it needs no atomics. */
  volatile unsigned int snapshot_req_in_flight;
  arts_lf_stack_t ro_combine;
  arts_lf_link_t *ro_combine_group;
#endif
};

/** Internal DataBlock descriptor.
 *
 *  The per-rank coherence cache (struct arts_db_cache_s) is embedded by value
 *  as the FIRST member: the cb object the route_table wraps is the db_s, and
 *  cache-to-db_s recovery is a zero-cost cast (cache == &db->cache, and since
 *  cache is first, (struct arts_db_s *)cache aliases the wrapping db_s).  Use
 *  arts_db_of_cache() for that recovery.  Non-coherent pinned subtypes
 *  (ARTS_DB_PIN/ARTS_DB_GPU_PIN/ARTS_DB_GPU/ARTS_DB_CXL) leave the cache
 *  zeroed (no DB-level coherence) and store their payload at (db + 1). */
struct arts_db_s {
  struct arts_db_cache_s cache; /**< FIRST — coherence state (embedded by
                                     value).  Zeroed for non-coherent pinned
                                     subtypes. */
  arts_db_types_t db_type;      /**< Storage subtype (DB/PIN/GPU/CXL).  Placed
                                     right after the cache so a cache-only stub
                                     (arts_db_cache_stub_size()) still covers it
                                     — every coherence lookup/free reads it. */
  /* Home-directory metadata — present only on the rank that is the GUID home
   * for this DB.  Non-home / WB / creator-remote ranks allocate a cache-only
   * footprint of arts_db_cache_stub_size() bytes: it ends at the first home-arm
   * field below (rw_holder), so it INCLUDES home_initialized but omits every
   * home-arm field.  home_initialized MUST stay in bounds: the cache destructor
   * reads it on every free to decide whether to tear the home directory down —
   * on a stub it reads zeroed false and skips teardown.  The home-arm fields
   * are touched only on the home rank, never through a stub. */
  bool home_initialized; /**< one-shot init sentinel (set by arts_db_home_init).
                          */
#ifdef ARTS_WRITE_POLICY_WB
  arts_db_atomic_uint_t rw_holder;
  struct arts_home_grantreq_queue_s pending_rw; /* embedded Vyukov MPSC */
  arts_db_atomic_uint_t invalidate_in_flight;
  struct arts_rank_bitset_s
      cached_ranks; /* RO cached-rank roster, destroy fan-out */
  unsigned int pending_install_owner; /* baton-holder-written transfer target */
#else                                 /* WT */
  arts_db_atomic_uint_t rw_holder;
  struct arts_home_grantreq_queue_s pending_rw; /* embedded Vyukov MPSC */
  arts_db_atomic_uint_t invalidate_in_flight;
  /* WT uses cached_version as both the RO dedup watermark (SNAPSHOT_REQUEST
   * reply) AND its destroy roster (no cached_ranks bit-set). */
  struct arts_rank_to_u64_map_s *cached_version;
  /* Baton-holder-written transfer target (CONFIRM-driven advance, same as
   * WB). */
  unsigned int pending_install_owner;
#endif
  /* GPU staging locks / version stamps (GPU DB path; full arts_db_s alloc). */
  volatile unsigned int reader;  /**< GPU staging reader lock. */
  volatile unsigned int writer;  /**< GPU staging writer lock. */
  volatile unsigned int version; /**< GPU LC version counter. */
  unsigned int time_stamp;       /**< GPU staging timestamp. */
} ARTS_ALIGNED_MAX;

/** @} */ /* end internal_db_structs */

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_RCU_TYPES_H */
