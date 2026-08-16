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
#ifndef ARTS_MEMORY_COHERENCE_WRF_RCU_TYPES_H
#define ARTS_MEMORY_COHERENCE_WRF_RCU_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file wrf_val/types.h
 * @brief WRF_VAL (Multi-Reader, Multi-Writer / DB-WRF) cache/db layout.
 *
 * Selected by arts/coherence/types.h when ARTS_PROTOCOL_WRF_VAL is defined.
 * WRF_VAL carries no ownership grant and no RW waiter queue: writer_count is a
 * pure ref count and same-DB write-write conflicts are app-ordered (DB-WRF), so
 * there are no rw_waiter / pending_rw / home_grantreq structures here.  The RO
 * snapshot reorder-buffer (cache.pending_snapshot) is still present — under
 * WRF_VAL it parks all modes.  The protocol-agnostic pieces come from
 * types_common.h; the of_cache/total_size/stub_size helpers live in the
 * dispatcher (types.h), after this header defines cache + db_s.
 *
 * @note Internal header.  User code should include @c arts.h.
 */

#include "arts/coherence/types_common.h"

/* ========================================================================= */
/** @addtogroup internal_db_structs
 *  @{ */

/*--- Per-rank DB cache ---------------------------------------------------
 * Every rank that has acquired or hosts a given DB has one of these.
 *
 *   writer_count   pure ref count under WRF_VAL (no node-exclusive ownership).
 *   buffer         currently-installed buffer, an atomic shared_ptr slot;
 *                  readers acquire via acquire_buf's acquire-and-validate load.
 *   pending_snapshot  Treiber stack of snapshot-response reorder-buffer
 *                  waiters (case-3 push only; drained whole on next install).
 *                  Parks all modes under WRF_VAL (no ownership / pending_rw).
 *
 * The single home-directory field (cached_version) is NOT here — it is
 * inlined in the wrapping struct arts_db_s, after this cache + db_type, and
 * reached via arts_db_of_cache(cache).  Non-home ranks allocate a cache-only
 * footprint that stops before it. */
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
  /* WRF_VAL: writer_count is a pure ref count.  The PUBLISH ACK rendezvous is
   * a stack-local sem_t created per release_rw, matched by pointer identity
   * (the &sem address rides the PUBLISH packet and is echoed in the ACK) —
   * no per-cache seq state. */
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
   * the one response that ends that window), so it needs no atomics.  Under
   * this protocol the same pull path also carries non-home RW data fetches,
   * so those combine too — each waiter's mode accounting already happened at
   * acquire time, before the park. */
  volatile unsigned int snapshot_req_in_flight;
  arts_lf_stack_t ro_combine;
  arts_lf_link_t *ro_combine_group;
#endif
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
   * for this DB.  Non-home / creator-remote ranks allocate a cache-only
   * footprint of arts_db_cache_stub_size() bytes: it ends at the first home-arm
   * field below (cached_version), so it INCLUDES home_initialized but omits
   * it.  home_initialized MUST stay in bounds: the cache destructor reads it on
   * every free to decide whether to tear the home directory down — on a stub it
   * reads zeroed false and skips teardown.  The home-arm field is touched only
   * on the home rank, never through a stub. */
  bool home_initialized; /**< one-shot init sentinel (set by arts_db_home_init).
                          */
  struct arts_rank_to_u64_map_s *cached_version;
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

#endif /* ARTS_MEMORY_COHERENCE_WRF_RCU_TYPES_H */
