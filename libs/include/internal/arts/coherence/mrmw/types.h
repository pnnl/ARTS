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
#ifndef ARTS_MEMORY_COHERENCE_MRMW_TYPES_H
#define ARTS_MEMORY_COHERENCE_MRMW_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file mrmw/types.h
 * @brief MRMW (Multi-Reader, Multi-Writer / DB-DRF) cache/db layout.
 *
 * Selected by arts/coherence/types.h when ARTS_PROTOCOL_MRMW is defined.
 * MRMW carries no ownership lease and no RW waiter queue: writer_count is a
 * pure ref count and conflicting same-DB accesses are app-ordered (DB-DRF), so
 * there are no rw_waiter / pending_rw / home_lockreq structures here.  The RO
 * snapshot reorder-buffer (cache.pending_snapshot) is still present — under
 * MRMW it parks all modes.  The protocol-agnostic pieces come from
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
 *   writer_count   pure ref count under MRMW (no node-exclusive ownership).
 *   buffer         currently-installed buffer, an atomic shared_ptr slot;
 *                  readers acquire via acquire_buf's acquire-and-validate load.
 *   pending_snapshot  Treiber stack of snapshot-response reorder-buffer
 *                  waiters (case-3 push only; drained whole on next install).
 *                  Parks all modes under MRMW (no ownership / pending_rw).
 *
 * The single home-directory field (last_sent_version) is NOT here — it is
 * inlined in the wrapping struct arts_db_s, after this cache + db_type, and
 * reached via arts_db_of_cache(cache).  Non-home ranks allocate a cache-only
 * footprint that stops before it. */
struct arts_db_cache_s {
  volatile unsigned int writer_count;
  arts_atomic_shared_ptr_t buffer;
  arts_lf_stack_t pending_snapshot;
  /* db_guid stored here for symmetry with the protocol pseudocode —
   * acquire_remote_* needs it for the route_table_return_db pairing on
   * ARTS_DB_DESTROYED early-returns, where the cache is in scope but the
   * original guid argument has been lost in the call chain. */
  arts_guid_t db_guid;
  uint64_t db_size;
  /* MRMW: writer_count is a pure ref count.  The WRITEBACK ACK rendezvous is
   * a stack-local sem_t created per release_rw, matched by pointer identity
   * (the &sem address rides the WRITEBACK packet and is echoed in the ACK) —
   * no per-cache seq state. */
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
   * field below (last_sent_version), so it INCLUDES home_initialized but omits
   * it.  home_initialized MUST stay in bounds: the cache destructor reads it on
   * every free to decide whether to tear the home directory down — on a stub it
   * reads zeroed false and skips teardown.  The home-arm field is touched only
   * on the home rank, never through a stub. */
  bool home_initialized; /**< one-shot init sentinel (set by arts_db_home_init).
                          */
  struct arts_rank_to_u64_map_s *last_sent_version;
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

#endif /* ARTS_MEMORY_COHERENCE_MRMW_TYPES_H */
