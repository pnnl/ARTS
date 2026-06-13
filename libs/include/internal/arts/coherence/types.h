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
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
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
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#ifndef ARTS_MEMORY_COHERENCE_TYPES_H
#define ARTS_MEMORY_COHERENCE_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file coherence_types.h
 * @brief DB-coherence layout types (home-directory + node-cache protocol).
 *
 * These struct definitions live here — rather than in arts/runtime_types.h —
 * so the DB/coherence types are co-located with the coherence module.
 * struct arts_db_s embeds the per-rank cache (struct arts_db_cache_s) by value
 * as its FIRST member, so db_s requires the complete cache type; the whole
 * buffer/home/cache/db_s chain is therefore defined together here.
 * arts/coherence/coherence.h keeps the protocol function declarations and
 * includes this header for the layouts.
 *
 * @note This is an internal header.  User code should include @c arts.h.
 */

#include "arts.h"

#include "arts/defs.h"
#include "arts/utils/lockfree_lifo.h"  /* arts_lf_stack_t */
#include "arts/utils/lockfree_stack.h" /* arts_lockfree_stack_t */
#include "arts/utils/mpsc.h"           /* arts_mpsc_t */
#include "arts/utils/shared.h" /* arts_atomic_shared_ptr_t (cache.buffer slot) */
#include <stdbool.h>
#include <stddef.h> /* offsetof — container_of(cache, arts_db_s, cache) + stub size */
#include <stdint.h>
#ifndef __cplusplus
#include <stdatomic.h>
#endif
/* Lazy home metadata embeds a per-rank reader bit-set by value. */
#ifdef ARTS_COHERENCE_PROTOCOL_LAZY
#include "arts/rank_bitset.h"
#endif

/* Sentinel for arts_db_cache_s.incoming_new_owner meaning "no ownership
 * transfer pending".  A real rank is always < rank_count, so UINT_MAX is a safe
 * out-of-band value (and rank 0 is a valid owner, so 0 cannot be the sentinel).
 */
#define ARTS_LAZY_NO_PENDING_OWNER ((unsigned int)-1)

/* Portable atomic unsigned-int for struct fields visible to both C and the
 * C++/nvcc layout-only TUs (which cannot parse C11 _Atomic).  C accesses these
 * via arts_atomic_* on the underlying uint; nvcc only needs the layout. */
#ifdef __cplusplus
typedef unsigned int arts_db_atomic_uint_t;
#else
typedef _Atomic(unsigned int) arts_db_atomic_uint_t;
#endif

/* ========================================================================= */
/** @defgroup internal_db_structs Internal DB-Coherence Structures
 *  These structures are exposed for layout visibility but are managed
 *  entirely by the runtime.  User code should not manipulate them directly.
 *  @{ */

/* ========================================================================= */
/* DB coherence layout (home-directory + node-cache protocol).
 *
 * These struct definitions live here — rather than in
 * arts/coherence/coherence.h — because struct arts_db_s embeds the per-rank
 * cache (struct arts_db_cache_s) by value as its FIRST member, so db_s
 * requires the complete cache type.  coherence.h keeps the protocol
 * function declarations and includes this header for the layouts.
 *
 * Atomic discipline: fields the runtime reads/writes concurrently are
 * declared `volatile` and accessed exclusively through arts_atomic_*
 * (or arts_db_atomic_uint_t for the C11 _Atomic / C++-layout split).
 * The per-model #if defined(ARTS_COHERENCE_PROTOCOL_{EAGER,LAZY}) /
 * defined(ARTS_MEMORY_MODEL_RELAXED) selects which
 * machinery is compiled in for each consistency model.
 */

/*--- Buffer ---------------------------------------------------------------
 * Holds version + user-visible data bytes (FAM).  Lifetime is managed by an
 * arts_shared_ptr_t control block (cache.buffer is the atomic slot; each
 * acquirer holds a strong ref).  No embedded refcount: the cb's strong count
 * IS the "cache-hold + per-acquirer" count, and the cb deleter frees the
 * buffer once the last holder releases — so a destroy concurrent with an
 * in-flight acquire can never free the bytes out from under a reader.
 *   version  monotonic per-buffer version stamp.
 *   cb       this buffer's own control block (== the slot's cb while
 *            installed).  An EDT recovers it via buf_from_data(dep->ptr)->cb
 *            to drop its acquire ref at release — safe because the EDT's own
 *            ref keeps the buffer (hence buf->cb) alive until that release.
 *   data     FAM holding db_size bytes — user-visible canonical payload,
 *            64-byte aligned (cache-line / CXL atomicity). */
struct arts_db_buffer_s {
  uint64_t version;     /* monotonic per buffer */
  arts_shared_ptr_t cb; /* this buffer's control block */
  char _pad[48];        /* data[] lands at offset 64 (cache-line aligned) */
  char data[];          /* db_size bytes — user-visible */
};

/*--- Pending RW / RO waiters ---------------------------------------------
 * edt_guid + slot together identify the parked EDT's dep slot to fill on
 * trigger.  RW path uses Vyukov MPSC: the embedded `next` pointer is owned
 * by the queue (init/push/pop manage it).  Producers are foreign-rank
 * acquire paths; the single consumer is the home-side dispatcher. */
#ifdef __cplusplus
struct arts_db_rw_waiter_s {
  struct arts_db_rw_waiter_s *next;
  arts_guid_t edt_guid;
  unsigned int slot;
};
#else
struct arts_db_rw_waiter_s {
  _Atomic(struct arts_db_rw_waiter_s *) next;
  arts_guid_t edt_guid;
  unsigned int slot;
};
#endif

/* Per-cache RW waiter queue (Vyukov MPSC).  The stub waiter never carries a
 * payload — it is the permanent sentinel required by the algorithm. */
#ifdef __cplusplus
struct arts_pending_rw_queue_s {
  struct arts_db_rw_waiter_s *head;
  struct arts_db_rw_waiter_s *tail;
  struct arts_db_rw_waiter_s stub;
};
#else
struct arts_pending_rw_queue_s {
  _Atomic(struct arts_db_rw_waiter_s *) head;
  _Atomic(struct arts_db_rw_waiter_s *) tail;
  struct arts_db_rw_waiter_s stub;
};
#endif

/* Snapshot-response reorder-buffer node.  Pushed ONLY in case 3 of
 * arts_handler_db_snapshot_response (a NO_DATA reply arrived with version >
 * buf->version, i.e. transport reordered the with-data reply behind it).
 * Drained in full by the next case-2 install via a single atomic_exchange on
 * the Treiber stack — monotonic version guarantees every parked node's
 * target_version <= the just-installed version. */
struct arts_db_snapshot_waiter_s {
  arts_lf_link_t link; /* FIRST — required by arts_lf_stack_t */
  arts_guid_t edt_guid;
  unsigned int slot;
  uint64_t target_version;
};

/*--- Per-DB home metadata ------------------------------------------------
 * Lives only on the rank that hosts a given DB (GUID home decides).  Holds
 * the directory state needed for ownership transfer and dedup.  The sparse
 * map is referred to by opaque forward-decl; the MPSC queues are defined
 * inline below (needed for struct embedding in arts_db_s). */
struct arts_rank_to_u64_map_s; /* forward decl; sparse rank-keyed u64 map */

/* Vyukov MPSC queue node carrying a requester rank (home OWNERSHIP_REQUEST
 * queue). The embedded `next` pointer is owned by the queue (push/pop manage
 * it). Producers are foreign-rank OWNERSHIP_REQUEST handlers; the single
 * consumer is the home-side dispatcher holding the invalidate_in_flight baton.
 */
#ifdef __cplusplus
struct arts_home_lockreq_node_s {
  struct arts_home_lockreq_node_s *next;
  unsigned int rank;
};

struct arts_home_lockreq_queue_s {
  struct arts_home_lockreq_node_s
      *tail; /* producer end (push exchanges here) */
  struct arts_home_lockreq_node_s *head; /* consumer end (pop advances here) */
  struct arts_home_lockreq_node_s stub;
};
#else
struct arts_home_lockreq_node_s {
  _Atomic(struct arts_home_lockreq_node_s *) next;
  unsigned int rank;
};

struct arts_home_lockreq_queue_s {
  _Atomic(struct arts_home_lockreq_node_s *)
      tail; /* producer end (push exchanges here) */
  _Atomic(struct arts_home_lockreq_node_s *)
      head;                             /* consumer end (pop advances here) */
  struct arts_home_lockreq_node_s stub; /* permanent sentinel */
};
#endif

/*--- Per-rank DB cache ---------------------------------------------------
 * Every rank that has acquired or hosts a given DB has one of these.  The
 * cache is the runtime's coherence-protocol state: ownership, the live
 * buffer, parked waiters, and destroy lifecycle.
 *
 *   writer_count   flat ownership counter.  > 0 ⇒ this rank holds RW
 *                  ownership; 0 ⇒ invalidated.
 *   buffer         currently-installed buffer pointer; readers acquire via
 *                  acquire_buf's CAS-loop against buffer->ref_count.
 *   pending_snapshot  Treiber stack of snapshot-response reorder-buffer
 *                  waiters (case-3 push only; drained whole on next install).
 *                  Parks all modes under the relaxed model (no
 * ownership/pending_rw). buffer_pool    per-DB recycle pool of arts_db_buffer_s
 * (intrusive Treiber stack); buffers are never freed during the DB's lifetime.
 *
 * The home-directory fields (rw_holder, pending_rw, invalidate_in_flight,
 * cached_ranks, last_sent_version, ...) are NOT here — they are inlined
 * directly in the wrapping struct arts_db_s, after this cache + db_type, and
 * reached via arts_db_of_cache(cache).  Non-home ranks allocate a cache-only
 * footprint that stops before them. */
struct arts_db_cache_s {
  volatile unsigned int writer_count;
  /* Installed buffer, managed as an atomic shared_ptr: the slot holds the
   * "cache-hold" strong ref; acquire_buf takes an additional ref via the
   * acquire-and-validate load; the cb deleter frees the buffer on the last
   * drop.  No separate recycle pool — the cb pool (never freed) provides the
   * UAF-safety that the old hand-rolled ref_count + buffer_pool did. */
  arts_atomic_shared_ptr_t buffer;
  arts_lf_stack_t pending_snapshot;
  /* db_guid stored here for symmetry with the protocol pseudocode —
   * acquire_remote_* needs it for the route_table_return_db pairing on
   * ARTS_DB_DESTROYED early-returns, where the cache is in scope but the
   * original guid argument has been lost in the call chain. */
  arts_guid_t db_guid;
  uint64_t db_size;
#if defined(ARTS_MEMORY_MODEL_RELAXED)
  /* Relaxed: writer_count is a pure ref count.  The WRITEBACK ACK rendezvous is
   * a stack-local sem_t created per release_rw, matched by pointer identity
   * (the &sem address rides the WRITEBACK packet and is echoed in the ACK) —
   * no per-cache seq state. */
#elif defined(ARTS_COHERENCE_PROTOCOL_LAZY)
  /* Lazy: owner-side dedup map.  Allocated lazily on first ownership; preserved
   * across ownership transfer (TRANSFER_OWNERSHIP serializes it). */
  struct arts_rank_to_u64_map_s *last_sent_version;
  /* New owner rank published by the INVALIDATE_NOTICE handler.  Sentinel
   * ARTS_LAZY_NO_PENDING_OWNER == no transfer pending.  The publish-before-
   * sentinel-withdraw ordering makes a separate transfer_pending flag
   * redundant: when release_rw sees writer_count reach 0 it reads this field;
   * a non-sentinel value means the INVALIDATE already named the target, so this
   * (last) releaser ships TRANSFER_OWNERSHIP.  Single writer per round (home
   * baton gate), so no atomic needed. */
  unsigned int incoming_new_owner;
  /* Lazy per-cache RW exclusivity machinery.  RW OWNERSHIP_REQUEST coalescing
   * flag — only the actor that CASes false->true sends OWNERSHIP_REQUEST;
   * same-node RW EDTs piggyback on the in-flight one and are picked up by
   * GRANT's drain. */
  volatile unsigned int ownership_req_in_flight;
  /* Vyukov MPSC queue of RW waiters parked on this rank. */
  struct arts_pending_rw_queue_s pending_rw;
#else
  /* Eager: the WRITEBACK ACK rendezvous is a stack-local sem_t created per
   * release_rw, matched by pointer identity (the &sem address rides the
   * WRITEBACK packet and is echoed verbatim in the ACK).  Multiple concurrent
   * releases each get their own sem — no per-cache seq state. */
  /* Eager per-cache RW exclusivity machinery.  RW OWNERSHIP_REQUEST coalescing
   * flag — only the actor that CASes false->true sends OWNERSHIP_REQUEST;
   * same-node RW EDTs piggyback on the in-flight one and are picked up by
   * GRANT's drain. */
  volatile unsigned int ownership_req_in_flight;
  /* Vyukov MPSC queue of RW waiters parked on this rank. */
  struct arts_pending_rw_queue_s pending_rw;
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
   * for this DB.  Non-home / lazy / creator-remote ranks allocate a cache-only
   * footprint of arts_db_cache_stub_size() bytes: it ends at the first home-arm
   * field below (rw_holder for eager/lazy, last_sent_version for relaxed), so
   * it INCLUDES
   * home_initialized but omits every home-arm field.  home_initialized MUST
   * stay in bounds: the cache destructor reads it on every free to decide
   * whether to tear the home directory down — on a stub it reads zeroed false
   * and skips teardown.  The home-arm fields are touched only on the home rank,
   * never through a stub. */
  bool home_initialized; /**< one-shot init sentinel (set by arts_db_home_init).
                          */
#if defined(ARTS_COHERENCE_PROTOCOL_LAZY)
  arts_db_atomic_uint_t rw_holder;
  struct arts_home_lockreq_queue_s pending_rw; /* embedded Vyukov MPSC */
  arts_db_atomic_uint_t invalidate_in_flight;
  struct arts_rank_bitset_s
      cached_ranks; /* RO cached-rank roster, destroy fan-out */
  unsigned int pending_install_owner; /* baton-holder-written transfer target */
#elif defined(ARTS_MEMORY_MODEL_RELAXED)
  struct arts_rank_to_u64_map_s *last_sent_version;
#else /* eager/lazy (OCR model) */
  arts_db_atomic_uint_t rw_holder;
  struct arts_home_lockreq_queue_s pending_rw; /* embedded Vyukov MPSC */
  arts_db_atomic_uint_t invalidate_in_flight;
  struct arts_rank_to_u64_map_s *last_sent_version;
#endif
  /* GPU staging locks / version stamps (GPU DB path; full arts_db_s alloc). */
  volatile unsigned int reader;  /**< GPU staging reader lock. */
  volatile unsigned int writer;  /**< GPU staging writer lock. */
  volatile unsigned int version; /**< GPU LC version counter. */
  unsigned int time_stamp;       /**< GPU staging timestamp. */
} ARTS_ALIGNED_MAX;

/* Recover the wrapping struct arts_db_s from a coherence cache pointer.  cache
 * is the FIRST member of arts_db_s; container_of degenerates to the cache
 * address but is written as container_of for correctness-by-construction (and
 * to match the master plan's home-access idiom).  NULL-safe. */
#ifndef ARTS_CONTAINER_OF
#define ARTS_CONTAINER_OF(ptr, type, member)                                   \
  ((type *)((char *)(ptr) - offsetof(type, member)))
#endif
static inline struct arts_db_s *arts_db_of_cache(struct arts_db_cache_s *c) {
  if (c == NULL) {
    return NULL;
  }
  return ARTS_CONTAINER_OF(c, struct arts_db_s, cache);
}

/* Total allocation size of a DB = wrapping struct + user payload.  A DB always
 * carries its own length in the cache (cache.db_size, set for every db_type at
 * create); there is no separate object header storing it. */
static inline uint64_t arts_db_total_size(const struct arts_db_s *db) {
  return sizeof(struct arts_db_s) + db->cache.db_size;
}

/* Footprint of a non-home / lazy / creator-remote DB stub: the cache prefix +
 * db_type + home_initialized, stopping before the home-directory queues/maps
 * (which only the GUID home rank ever touches).  home_initialized MUST be in
 * bounds: the cache destructor reads it on EVERY free to decide whether to tear
 * down the home directory — on a stub it reads (zeroed) false and skips the
 * teardown.  Allocating this much keeps db_type + home_initialized in bounds
 * while shedding the bulky home directory (MPSC queue + rank->u64 map +
 * cached_ranks bitset).  The home rank allocates the full sizeof(struct
 * arts_db_s) instead.
 * The stub ends at the first home-directory field after home_initialized
 * (protocol-dependent: rw_holder for eager/lazy, last_sent_version for
 * relaxed). */
static inline uint64_t arts_db_cache_stub_size(void) {
#if defined(ARTS_MEMORY_MODEL_RELAXED)
  return offsetof(struct arts_db_s, last_sent_version);
#else
  return offsetof(struct arts_db_s, rw_holder);
#endif
}

/** @} */ /* end internal_db_structs */

/* ========================================================================= */
#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_TYPES_H */
