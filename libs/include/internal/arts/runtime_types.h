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
#ifndef ARTS_RUNTIME_TYPES_H
#define ARTS_RUNTIME_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file rt.h
 * @brief Internal structures for the ARTS runtime.
 *
 * This header defines internal types used by the runtime implementation:
 * EDT and DataBlock descriptors, event structures, and termination-detection
 * state.  Public types (arts_guid_t, arts_guid_kind_t, etc.) live in arts.h.
 *
 * @note This is an internal header.  User code should include @c arts.h.
 */

#include "arts.h"

#include "arts/defs.h"
#include "arts/sync/mpsc.h"   /* arts_mpsc_t */
#include "arts/sync/shared.h" /* arts_atomic_shared_ptr_t (cache.buffer slot) */
#include "arts/utils/lockfree_lifo.h"  /* arts_lf_stack_t */
#include "arts/utils/lockfree_stack.h" /* arts_lockfree_stack_t */
#include "arts/utils/marked_list.h"    /* arts_marked_list_t (RO waiters) */
#include <stdbool.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdatomic.h>
#endif
/* LRC home metadata embeds a per-rank reader bit-set by value. */
#ifdef ARTS_MEMORY_MODEL_LRC
#include "arts/memory/coherence_readers.h"
#endif

/* Portable atomic unsigned-int for struct fields visible to both C and the
 * C++/nvcc layout-only TUs (which cannot parse C11 _Atomic).  C accesses these
 * via arts_atomic_* on the underlying uint; nvcc only needs the layout. */
#ifdef __cplusplus
typedef unsigned int arts_coh_atomic_uint;
#else
typedef _Atomic(unsigned int) arts_coh_atomic_uint;
#endif

/* ========================================================================= */
/** @defgroup internal_structs Internal Runtime Structures
 *  These structures are exposed for layout visibility but are managed
 *  entirely by the runtime.  User code should not manipulate them directly.
 *  @{ */

/* ========================================================================= */
/* DB coherence layout (home-directory + node-cache protocol).
 *
 * These struct definitions live here — rather than in
 * arts/memory/coherence.h — because struct arts_db_s embeds the per-rank
 * cache (struct arts_db_cache_s) by value as its FIRST member, so db_s
 * requires the complete cache type.  coherence.h keeps the protocol
 * function declarations and includes this header for the layouts.
 *
 * Atomic discipline: fields the runtime reads/writes concurrently are
 * declared `volatile` and accessed exclusively through arts_atomic_*
 * (or arts_coh_atomic_uint for the C11 _Atomic / C++-layout split).
 * The per-model #if defined(ARTS_MEMORY_MODEL_{RC,LRC,LC}) selects which
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

/* RO waiters carry target_version, set by the DATA_RESPONSE handler when
 * home replies with the version that satisfies this acquire.  Initial value
 * U64_MAX so traversal-time `t <= cache.buffer.version` only fires once
 * target_version is actually set. */
struct arts_db_ro_waiter_s {
  arts_marked_list_node_t link; /* FIRST — required by marked-list */
  arts_guid_t edt_guid;
  unsigned int slot;
  volatile uint64_t target_version; /* U64_MAX until DATA_RESPONSE */
};

/*--- Per-DB home metadata ------------------------------------------------
 * Lives only on the rank that hosts a given DB (GUID home decides).  Holds
 * the directory state needed for ownership transfer and dedup.  The sparse
 * map is referred to by opaque forward-decl; the MPSC queues are defined
 * inline below (needed for struct embedding in arts_db_home_s). */
struct arts_rank_to_u64_map_s; /* forward decl; sparse rank-keyed u64 map */

/* Vyukov MPSC queue node carrying a requester rank (home LOCK_REQ queue).
 * The embedded `next` pointer is owned by the queue (push/pop manage it).
 * Producers are foreign-rank LOCK_REQ handlers; the single consumer is the
 * home-side dispatcher holding the invalidate_in_flight baton. */
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

/* RO_REQ queue deferred while a gate is set.  LRC builds use a full Vyukov
 * MPSC queue (multi-producer; single consumer drains when the gate clears).
 * RC/LC builds keep a no-op stub because RO acquires are served directly
 * from home without deferral. */
#ifdef ARTS_MEMORY_MODEL_LRC
#ifdef __cplusplus
struct arts_home_ro_node_s {
  struct arts_home_ro_node_s *next;
  unsigned int requester_rank;
  void *waiter_addr;
};

struct arts_home_pending_ro_queue_s {
  struct arts_home_ro_node_s *tail;
  struct arts_home_ro_node_s *head;
  struct arts_home_ro_node_s stub;
};
#else
struct arts_home_ro_node_s {
  _Atomic(struct arts_home_ro_node_s *) next;
  unsigned int requester_rank;
  void *waiter_addr;
};

struct arts_home_pending_ro_queue_s {
  _Atomic(struct arts_home_ro_node_s *) tail; /* producer end */
  _Atomic(struct arts_home_ro_node_s *) head; /* consumer end */
  struct arts_home_ro_node_s stub;            /* permanent sentinel */
};
#endif /* __cplusplus */
#else  /* !ARTS_MEMORY_MODEL_LRC */
/* RC/LC: no RO deferral queue needed — stub keeps the struct layout neutral. */
struct arts_home_pending_ro_queue_s {
  void *_reserved;
};
#endif /* ARTS_MEMORY_MODEL_LRC */

struct arts_db_home_s {
  /* Common across all three coherence models. */
  /* Destroy fan-out single-flight gate.  Set (CAS 0->1) before iterating the
   * readers roster; cleared only implicitly when the home_s struct is freed.
   * Ensures at most one destroy fan-out runs per DB lifetime. */
  arts_coh_atomic_uint destroy_in_flight;
  /* Outstanding DESTROY_DONE ack count; finalize when it reaches 0. */
  arts_coh_atomic_uint destroy_ack_outstanding;

#if defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC: exclusivity machinery (LOCK_REQ ownership rounds) + RO-forward
   * deferral queue + per-RO-reader bit-set for destroy fan-out. */
  arts_coh_atomic_uint rw_holder;
  struct arts_home_lockreq_queue_s pending_rw; /* embedded Vyukov MPSC */
  /* Active-directory in-flight tracking for INVALIDATE_NOTICE.  Set by the
   * home-side LOCK_REQ handler (acq_rel CAS 0->1) when it dispatches an
   * INVALIDATE to the current rw_holder; cleared once the transfer completes.
   */
  arts_coh_atomic_uint invalidate_in_flight;
  /* RO_REQs deferred while an ownership-transfer gate is set. */
  struct arts_home_pending_ro_queue_s pending_ro_forwards;
  /* Bit per reader rank set when a DATA_RESPONSE (RO grant) is sent; iterated
   * during DESTROY fan-out to reach all ranks that held a cached copy. */
  struct arts_readers_bits_s readers;
  /* Set by the LOCK_REQ handler when it kicks off an invalidate round; read
   * by the INSTALL_ACK handler.  Written only by the baton holder. */
  unsigned int pending_install_owner;
#elif defined(ARTS_MEMORY_MODEL_LC)
  /* LC: thin canonical-data home.  No per-node exclusive owner, no LOCK_REQ /
   * INVALIDATE machinery.  master_version is implicit in the home rank's own
   * cache_s.buffer.version.  last_sent_version is a per-rank dedup watermark
   * used by the DESTROY fan-out (same as RC) and by the WRITEBACK handler to
   * skip redundant installs. */
  struct arts_rank_to_u64_map_s *last_sent_version;
#else
  /* RC: exclusivity machinery (LOCK_REQ ownership rounds) and per-rank dedup
   * watermark for GRANT / DATA_RESPONSE. */
  arts_coh_atomic_uint rw_holder;
  struct arts_home_lockreq_queue_s pending_rw; /* embedded Vyukov MPSC */
  /* Active-directory in-flight tracking for INVALIDATE_NOTICE. */
  arts_coh_atomic_uint invalidate_in_flight;
  /* RO_REQs deferred while an ownership-transfer gate is set (no-op stub in
   * RC; real queue in LRC). */
  struct arts_home_pending_ro_queue_s pending_ro_forwards;
  /* Per-rank watermark of the latest DB version sent to each rank.  Used by
   * GRANT / DATA_RESPONSE to skip redundant data transfers when the
   * receiver's cached version is already current. */
  struct arts_rank_to_u64_map_s *last_sent_version;
#endif
};

/*--- Per-rank DB cache ---------------------------------------------------
 * Every rank that has acquired or hosts a given DB has one of these.  The
 * cache is the runtime's coherence-protocol state: ownership, the live
 * buffer, parked waiters, and destroy lifecycle.
 *
 *   writer_count   flat ownership counter.  > 0 ⇒ this rank holds RW
 *                  ownership; 0 ⇒ invalidated.
 *   buffer         currently-installed buffer pointer; readers acquire via
 *                  acquire_buf's CAS-loop against buffer->ref_count.
 *   pending_ro     Harris marked-next list of RO waiters parked on this rank
 *                  (selective drain by target_version).
 *   pending_count  unified live-waiter counter (RW + RO summed).  Consulted
 *                  only by destroy finalization.
 *   buffer_pool    per-DB recycle pool of arts_db_buffer_s (intrusive Treiber
 *                  stack); buffers are never freed during the DB's lifetime.
 *   home           home metadata.  Meaningful only on the rank that is the
 *                  GUID home for this DB; other ranks leave it zeroed.
 *   home_initialized  one-shot init sentinel for the embedded home. */
struct arts_db_cache_s {
  volatile unsigned int writer_count;
  /* Installed buffer, managed as an atomic shared_ptr: the slot holds the
   * "cache-hold" strong ref; acquire_buf takes an additional ref via the
   * acquire-and-validate load; the cb deleter frees the buffer on the last
   * drop.  No separate recycle pool — the cb pool (never freed) provides the
   * UAF-safety that the old hand-rolled ref_count + buffer_pool did. */
  arts_atomic_shared_ptr_t buffer;
  arts_marked_list_t pending_ro;
  volatile unsigned int pending_count;
  /* db_guid stored here for symmetry with the protocol pseudocode —
   * acquire_remote_* needs it for the route_table_return_db pairing on
   * ARTS_DB_DESTROYED early-returns, where the cache is in scope but the
   * original guid argument has been lost in the call chain. */
  arts_guid_t db_guid;
  uint64_t db_size;
  /* Home metadata embedded by value (composition — no separate allocation,
   * no pointer chase).  Meaningful only on the GUID home rank; on non-home
   * ranks it stays zeroed and untouched.  `home_initialized` is the one-shot
   * init sentinel set once arts_db_home_init has run. */
  struct arts_db_home_s home;
  bool home_initialized;
#if defined(ARTS_MEMORY_MODEL_LC)
  /* LC: writer_count is a pure ref count.  The WRITEBACK ACK rendezvous is a
   * stack-local sem_t created per release_rw, matched by pointer identity
   * (the &sem address rides the WRITEBACK packet and is echoed in the ACK) —
   * no per-cache seq state. */
#elif defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC: owner-side dedup map.  Allocated lazily on first ownership; preserved
   * across ownership transfer (TRANSFER_OWNERSHIP serializes it). */
  struct arts_rank_to_u64_map_s *last_sent_version;
  /* Set by the INVALIDATE_NOTICE handler when writers are still live;
   * release_rw observes this flag and ships TRANSFER_OWNERSHIP when
   * writer_count reaches 0. */
  arts_coh_atomic_uint transfer_pending;
  /* New owner rank extracted from the INVALIDATE_NOTICE message payload. */
  unsigned int incoming_new_owner;
  /* LRC per-cache RW exclusivity machinery.  RW LOCK_REQ coalescing flag —
   * only the actor that CASes false->true sends LOCK_REQ; same-node RW EDTs
   * piggyback on the in-flight one and are picked up by GRANT's drain. */
  volatile unsigned int lock_req_in_flight;
  /* Vyukov MPSC queue of RW waiters parked on this rank. */
  struct arts_pending_rw_queue_s pending_rw;
#else
  /* RC: the WRITEBACK ACK rendezvous is a stack-local sem_t created per
   * release_rw, matched by pointer identity (the &sem address rides the
   * WRITEBACK packet and is echoed verbatim in the ACK).  Multiple concurrent
   * releases each get their own sem — no per-cache seq state. */
  /* RC per-cache RW exclusivity machinery.  RW LOCK_REQ coalescing flag —
   * only the actor that CASes false->true sends LOCK_REQ; same-node RW EDTs
   * piggyback on the in-flight one and are picked up by GRANT's drain. */
  volatile unsigned int lock_req_in_flight;
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
 *  arts_db_of_cache() for that recovery.  Non-RC subtypes (PIN/GPU/CXL) leave
 *  the cache zeroed (no DB-level coherence) and store their payload at
 *  (db + 1). */
struct arts_db_s {
  struct arts_db_cache_s cache; /**< FIRST — RC coherence state (embedded
                                     by value).  Zeroed for non-RC subtypes. */
  /* GUID lives in the cache (cache.db_guid); kind comes from the GUID bits.
   * reader/writer/version/time_stamp are GPU-staging locks/version stamps used
   * by the GPU DB path (fold into the cache when GPU-LC is redesigned). */
  volatile unsigned int reader;  /**< GPU staging reader lock. */
  volatile unsigned int writer;  /**< GPU staging writer lock. */
  volatile unsigned int version; /**< GPU LC version counter. */
  unsigned int time_stamp;       /**< GPU staging timestamp. */
  arts_db_types_t db_type;       /**< Storage subtype (RC/PIN/GPU/CXL). */
} ARTS_ALIGNED_MAX;

/* Recover the wrapping struct arts_db_s from a coherence cache pointer.
 * cache is the FIRST member of arts_db_s, so the cache address aliases the
 * db_s address.  Caller contract: `c` must be a cache embedded in a db_s
 * (always true for route_table-installed DBs).  NULL-safe. */
static inline struct arts_db_s *arts_db_of_cache(struct arts_db_cache_s *c) {
#ifdef __cplusplus
  return reinterpret_cast<struct arts_db_s *>(c);
#else
  return (struct arts_db_s *)c;
#endif
}

/* Total allocation size of a DB = wrapping struct + user payload.  A DB always
 * carries its own length in the cache (cache.db_size, set for every db_type at
 * create); there is no separate object header storing it. */
static inline uint64_t arts_db_total_size(const struct arts_db_s *db) {
  return sizeof(struct arts_db_s) + db->cache.db_size;
}

/** Internal EDT descriptor. */
struct arts_edt_s {
  uint64_t arts_id;    /**< Compiler-assigned unique id (0 = unset). */
  arts_edt_t func_ptr; /**< User function to execute. */
  uint32_t paramc;     /**< Number of static parameters. */
  uint32_t depc;       /**< Number of dependency slots. */
  /* The EDT's own GUID.  Unlike DB/Event/Epoch (always reached via a
   * route_table lookup whose key already IS the GUID), an EDT is dispatched
   * by raw pointer through the lock-free work-stealing deques — at dispatch
   * there is no key and no handler args, so the GUID must travel inside the
   * struct.  This is a load-bearing identity carrier, NOT a redundant
   * self-GUID: do not remove it. */
  arts_guid_t guid;
  arts_guid_t epoch_guid;    /**< Enclosing epoch GUID (NULL_GUID = none). */
  arts_guid_t finish_event;  /**< LATCH event for finish-scope tracking.
                                  NULL_GUID = no finish-scope (legacy path). */
  arts_edt_types_t edt_type; /**< EDT subtype (DEFAULT=CPU, GPU). */
  volatile unsigned int depc_needed; /**< Remaining unsatisfied deps. */
  volatile unsigned int
      invalidate_count; /**< Outstanding cache invalidations. */
} ARTS_ALIGNED_MAX;

/* Total allocation size of an EDT = struct + trailing [paramv | depv].
 * Computed from paramc/depc; no separate object header stores it. */
static inline uint64_t arts_edt_total_size(const struct arts_edt_s *edt) {
  return sizeof(struct arts_edt_s) +
         ((uint64_t)edt->paramc * sizeof(uint64_t)) +
         ((uint64_t)edt->depc * sizeof(arts_edt_dep_t));
}

/** An individual dependent registered on an event (legacy structure;
 *  retained only for the legacy event.c body until follow-up). */
struct arts_dependent_s {
  uint8_t type;               /**< Dependent kind (EDT or event). */
  volatile unsigned int slot; /**< Target dependency slot. */
  volatile arts_guid_t addr;  /**< GUID of the dependent EDT/event. */
  volatile bool done_writing; /**< Write completion flag. */
  arts_db_access_mode_t mode; /**< Access mode for signaling. */
  uint64_t byte_offset;       /**< Byte offset for slice dependencies. */
  uint64_t size;              /**< Slice size in bytes. */
};

/** Linked list node containing an array of dependents. */
struct arts_dependent_list_s {
  unsigned int size; /**< Number of dependents in this node. */
  struct arts_dependent_list_s *volatile next; /**< Next list node. */
  struct arts_dependent_s dependents[];        /**< Flexible array. */
};

/** Forward-declared dep node (definition in arts/sync/event.h, Task 8). */
struct arts_event_dep_s;

/** Internal event descriptor — single generic type, hint-driven behavior.
 *  The `is_channel` discriminator selects which union arm is active.
 *
 *  Non-CHANNEL semantics (`is_channel == 0`):
 *    - `latch` decrements per LATCH_DECR satisfy; fires at <= 0.
 *    - Fire is a pure state transition (sets `fired`, publishes `data`,
 *      drains `deps_stack`) and never destroys: the event lingers to serve
 *      late binders until an explicit `arts_event_destroy`.  Over-satisfy
 *      past the fire is silently absorbed.
 *
 *  CHANNEL semantics (`is_channel == 1`):
 *    - `nb_sat` increments per satisfy; `nb_deps` increments per add_dep.
 *    - Drainer pops one from each queue, decrements both counters, signals.
 *    - No auto-destroy; only explicit `arts_event_destroy`.
 *
 *  Two declarations: the C path uses C11 `_Atomic`; the C++/nvcc path
 *  drops the qualifier so the layout is visible without requiring C11
 *  atomics — same approach as memory/coherence.h. */
#ifdef __cplusplus
struct arts_event_s {
  uint8_t is_channel; /* discriminator: 0=simple, 1=channel */

  union {
    struct {
      int32_t latch;
      bool fired;
      arts_guid_t data;
      arts_lf_stack_t deps_stack;
    } simple;
    struct {
      uint32_t nb_sat;
      uint32_t nb_deps;
      arts_mpsc_t data_queue;
      arts_mpsc_t dep_queue;
      uint8_t draining;
    } channel;
  };
} ARTS_ALIGNED_MAX;
#else
struct arts_event_s {
  uint8_t is_channel; /* discriminator: 0=simple, 1=channel */

  union {
    struct {
      _Atomic(int32_t) latch;     /* fires at <= 0 */
      _Atomic(bool) fired;        /* single-fire CAS gate */
      arts_guid_t data;           /* last satisfy data; late binders read */
      arts_lf_stack_t deps_stack; /* Treiber stack of pending consumers */
    } simple;
    struct {
      _Atomic(uint32_t) nb_sat;  /* incremented per satisfy */
      _Atomic(uint32_t) nb_deps; /* incremented per add_dep */
      arts_mpsc_t data_queue;    /* satisfy FIFO */
      arts_mpsc_t dep_queue;     /* dep FIFO */
      _Atomic(uint8_t) draining; /* single-flight drainer gate */
    } channel;
  };
} ARTS_ALIGNED_MAX;
#endif

/** @} */ /* end internal_structs */

/* ========================================================================= */
/** @defgroup td_types Termination Detection
 *  @{ */

/** Three-phase termination detection state machine. */
typedef enum {
  PHASE_1, /**< Initial quiescence check. */
  PHASE_2, /**< Counter stabilization. */
  PHASE_3  /**< Termination confirmed. */
} termination_detection_phase_t;

/**
 * @brief Per-epoch termination detection state.
 *
 * Tracks active/finished task counts across nodes to determine when
 * all work within the epoch has completed.
 */
struct arts_epoch_s {
  volatile unsigned int local_lock;   /**< Single-node active/finished lock. */
  volatile unsigned int phase;        /**< Current TD phase (PHASE_*). */
  volatile unsigned int active_count; /**< Local active task count. */
  volatile unsigned int finished_count;      /**< Local finished task count. */
  volatile unsigned int global_active_count; /**< Cluster-wide active count. */
  volatile unsigned int
      global_finished_count;               /**< Cluster-wide finished count. */
  volatile unsigned int last_active_count; /**< Previous-round active count. */
  volatile unsigned int
      last_finished_count;            /**< Previous-round finished count. */
  volatile uint64_t queued;           /**< Number of queued operations. */
  volatile uint64_t outstanding;      /**< Number of outstanding remote ops. */
  unsigned int termination_exit_slot; /**< EDT slot to signal on completion. */
  arts_guid_t termination_exit_guid;  /**< EDT to signal on completion. */
  arts_guid_t guid;                   /**< GUID of this epoch. */
  arts_guid_t pool_guid;              /**< Associated resource pool GUID. */
};
typedef struct arts_epoch_s arts_epoch_t;

/** @} */ /* end td_types */

/* ========================================================================= */
#ifdef __cplusplus
}
#endif

#endif /* ARTS_RUNTIME_TYPES_H */
