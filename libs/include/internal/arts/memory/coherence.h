/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol data structures.
 *
 * Per-DB state under the home-directory + node-cache protocol.
 * See the corresponding design plan for the full protocol; this
 * header captures only the struct layouts the runtime needs to
 * manipulate.
 *
 * Atomic discipline: ARTS uses GCC __sync_* builtins (full memory
 * fences) wrapped in arts_atomic_* helpers — no <stdatomic.h>.  All
 * fields the runtime reads/writes concurrently are declared `volatile`
 * and accessed exclusively through arts_atomic_*; the volatile keeps
 * the compiler from caching reloads inside CAS loops.
 *
 * Routing pointer convention: 48-bit virtual address space.  Tagged-
 * pointer encodings (lock-free stack `top`, marked-list `next`) live
 * at the primitive layer; this header just embeds those primitives.
 */

#ifndef ARTS_MEMORY_COHERENCE_H
#define ARTS_MEMORY_COHERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "arts/runtime_types.h"
#include "arts/utils/lockfree_stack.h"
#include "arts/utils/marked_list.h"
#ifdef ARTS_MEMORY_MODEL_LRC
#include "arts/memory/coherence_readers.h"
#endif

/*--- Buffer ---------------------------------------------------------------
 *
 * arts_db_buffer_s holds version + ref_count + user-visible data bytes
 * (FAM).  Per design plan §Buffer (line 60-64): one atomic load of
 * cache.buffer snapshots both data pointer and version.  acquire
 * returns &buf->data[0] with ref held; release drops via release_buf.
 *
 *   pool_link   intrusive stack link (offset 0) for cache.buffer_pool.
 *   version     monotonic per-buffer version stamp.
 *   ref_count   sentinel +1 + per-acquire +1; whoever brings it to 0
 *               recycles into buffer_pool (never `free` until destroy).
 *   data        FAM holding db_size bytes — user-visible canonical
 *               payload.  All buffers in a cache share the same db_size
 *               so pool recycle is size-safe.
 */
struct arts_db_buffer_s {
  arts_lockfree_stack_node_t pool_link; /* FIRST — for cache.buffer_pool */
  volatile uint64_t version;            /* monotonic per buffer */
  volatile unsigned int ref_count;      /* sentinel +1 + per-acquire +1 */
  unsigned int _pad;                    /* keep data 16-byte aligned */
  char data[];                          /* db_size bytes — user-visible */
};

/*--- Pending RW / RO waiters ---------------------------------------------
 *
 * Each parked acquire registers a waiter on the per-rank cache.  The
 * waiter is recycled into the marked-list module's private pool after
 * helping-unlink physically removes it from the chain.
 *
 * RW waiters carry only the parked EDT guid: at GRANT time the
 * traversal triggers every unmarked RW waiter en bloc.
 *
 * RO waiters additionally carry target_version, set by the
 * DATA_RESPONSE handler when home replies with the version that
 * satisfies this acquire.  Initial value U64_MAX so traversal-time
 * `t <= cache.buffer.version` only fires after target_version is
 * actually set.
 */
/* edt_guid + slot together identify the parked EDT's dep slot to
 * fill on trigger.  Both fields are passed via the queue payload
 * because the protocol's "trigger an EDT" semantics require both —
 * slot index decides which depv entry to populate with the acquired
 * buffer pointer.
 *
 * RW path uses Vyukov MPSC: the embedded `next` pointer is owned by
 * the queue (init/push/pop manage it).  Producers are foreign-rank
 * acquire_remote_rw paths; the single consumer is the home-side
 * dispatcher (drain_pending_rw_after_grant / fail_trigger_pending /
 * destroy fan-out).  See arts/memory/coherence_pending_rw.h. */
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

/* Per-cache RW waiter queue (Vyukov MPSC).  Embedded in struct
 * arts_db_cache_s.  The stub waiter never carries a payload — it is
 * the permanent sentinel required by the algorithm. */
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

/* Lifecycle helpers. */
void arts_pending_rw_queue_init(struct arts_pending_rw_queue_s *q);
/* Push a waiter (multi-producer).  Caller fills edt_guid/slot before
 * calling.  Waiter must be heap-allocated; queue takes ownership and
 * frees it during pop or destroy. */
void arts_pending_rw_queue_push(struct arts_pending_rw_queue_s *q,
                                struct arts_db_rw_waiter_s *w);
/* Pop the head waiter (single consumer).  On success, *out_edt and
 * *out_slot are populated and the function returns true; the popped
 * node has been freed (or is the embedded stub on first call) before
 * return.  Returns false on empty.
 *
 * Why copy-out instead of returning the waiter pointer: in Vyukov's
 * algorithm the popped node is freed on the NEXT pop (it becomes the
 * "old head" we walk past).  Returning a pointer that becomes a
 * dangling reference one call later is footgun-prone, so we copy
 * fields here and free immediately. */
bool arts_pending_rw_queue_pop(struct arts_pending_rw_queue_s *q,
                               arts_guid_t *out_edt, unsigned int *out_slot);
/* Drain everything (single consumer); invokes cb(edt_guid, slot, ctx)
 * on each popped waiter in FIFO order.  cb must NOT block — drain
 * holds no lock but is intended for short tasks (mark-EDT-ready). */
void arts_pending_rw_queue_drain(struct arts_pending_rw_queue_s *q,
                                 void (*cb)(arts_guid_t edt_guid,
                                            unsigned int slot, void *ctx),
                                 void *ctx);
/* Destroy: free every queued waiter.  Stub is embedded in the queue
 * and not freed. */
void arts_pending_rw_queue_destroy(struct arts_pending_rw_queue_s *q);

struct arts_db_ro_waiter_s {
  arts_marked_list_node_t link; /* FIRST — required by marked-list */
  arts_guid_t edt_guid;
  unsigned int slot;
  volatile uint64_t target_version; /* U64_MAX until DATA_RESPONSE */
};

/*--- Per-DB home metadata ------------------------------------------------
 *
 * Lives only on the rank that hosts a given DB (GUID_HOME_RANK
 * decides).  Holds the directory state needed for ownership transfer
 * and dedup:
 *
 *   rw_holder           current RW owner rank (always defined from
 *                       creation onward).
 *   pending_rw          MPSC queue of foreign LOCK_REQ requesters
 *                       awaiting transfer.  Lock-free; single
 *                       consumer = home's network handler thread.
 *   last_sent_version   sparse map [rank → uint64].  Watermark of the
 *                       latest DB version this home sent to that
 *                       rank.  Used by GRANT / DATA_RESPONSE to skip
 *                       redundant data when last_sent[R] >= master_v.
 *
 * The sparse map is referred to by opaque forward-decl; the MPSC queue
 * is defined inline below (needed for struct embedding in arts_db_home_s).
 */
struct arts_rank_to_u64_map_s; /* forward decl; sparse rank-keyed u64 map */

/* Vyukov MPSC queue node carrying a requester rank.  Used by
 * arts_home_lockreq_queue_s.  The embedded `next` pointer is owned by
 * the queue (push/pop manage it).  Producers are foreign-rank LOCK_REQ
 * handlers; the single consumer is the home-side dispatcher holding the
 * invalidate_in_flight baton. */
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

/* RO_REQ queue deferred while a gate is set.  Each entry carries the
 * requester rank and the opaque waiter pointer (valid only at the
 * requester's address space).  Drained when invalidate_in_flight clears.
 *
 * LRC builds use a full Vyukov MPSC queue (multi-producer: any network
 * handler thread may enqueue while the gate is set; single consumer:
 * the INSTALL_ACK handler that clears the gate drains it).
 * RC builds keep a no-op stub because the RC GET_DATA path serves RO
 * acquires directly from home without deferral. */
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
/* RC: no RO deferral queue needed — stub keeps the struct layout neutral. */
struct arts_home_pending_ro_queue_s {
  void *_reserved;
};
#endif /* ARTS_MEMORY_MODEL_LRC */

struct arts_db_home_s {
  /* Common across all three coherence models. */
  /* Destroy fan-out single-flight gate.  Set (CAS 0→1) by handle_destroy_req
   * before it iterates the readers roster; cleared only implicitly when the
   * home_s struct is freed.  Ensures at most one destroy fan-out runs per DB
   * lifetime, even under concurrent DESTROY_REQ arrivals. */
  _Atomic(unsigned int) destroy_in_flight;
  /* Outstanding DESTROY_DONE ack count; finalize when it reaches 0.  Used by
   * all three models during destroy fan-out. */
  _Atomic(unsigned int) destroy_ack_outstanding;

#if defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC: includes exclusivity machinery (LOCK_REQ ownership rounds) +
   * RO-forward deferral queue + per-RO-reader bit-set for destroy fan-out. */
  _Atomic(unsigned int) rw_holder;
  struct arts_home_lockreq_queue_s pending_rw; /* embedded Vyukov MPSC */
  /* Active-directory in-flight tracking for INVALIDATE_NOTICE.  Set by
   * the home-side LOCK_REQ handler (acq_rel CAS 0->1) when it dispatches
   * an INVALIDATE to the current rw_holder; cleared once the ownership
   * transfer round completes. */
  _Atomic(unsigned int) invalidate_in_flight;
  /* RO_REQs deferred while an ownership-transfer gate is set. */
  struct arts_home_pending_ro_queue_s pending_ro_forwards;
  /* Bit per reader rank set when a DATA_RESPONSE (RO grant) is sent; iterated
   * during DESTROY fan-out to reach all ranks that held a cached copy. */
  struct arts_readers_bits_s readers;
  /* Set by the LOCK_REQ handler when it kicks off an invalidate round;
   * read by the INSTALL_ACK handler.  Written only by the baton holder. */
  unsigned int pending_install_owner;
#elif defined(ARTS_MEMORY_MODEL_LC)
  /* LC: thin canonical-data home.  No per-node exclusive owner, no
   * LOCK_REQ / INVALIDATE machinery.  master_version is implicit in the
   * home rank's own cache_s.buffer.version.  last_sent_version is a
   * per-rank dedup watermark used by the DESTROY fan-out (same as RC) and
   * by the WRITEBACK handler to skip redundant installs. */
  struct arts_rank_to_u64_map_s *last_sent_version;
#else
  /* RC: includes exclusivity machinery (LOCK_REQ ownership rounds) and
   * per-rank dedup watermark for GRANT / DATA_RESPONSE. */
  _Atomic(unsigned int) rw_holder;
  struct arts_home_lockreq_queue_s pending_rw; /* embedded Vyukov MPSC */
  /* Active-directory in-flight tracking for INVALIDATE_NOTICE.  Set by
   * the home-side LOCK_REQ handler (acq_rel CAS 0->1) when it dispatches
   * an INVALIDATE to the current rw_holder; cleared once the ownership
   * transfer round completes. */
  _Atomic(unsigned int) invalidate_in_flight;
  /* RO_REQs deferred while an ownership-transfer gate is set (no-op stub
   * in RC; real queue in LRC). */
  struct arts_home_pending_ro_queue_s pending_ro_forwards;
  /* Per-rank watermark of the latest DB version sent to each rank.
   * Used by GRANT / DATA_RESPONSE to skip redundant data transfers when
   * the receiver's cached version is already current. */
  struct arts_rank_to_u64_map_s *last_sent_version;
#endif
};

/*--- Destroy state -------------------------------------------------------
 *
 * Forward-only tri-state.  See the destroy section of the design plan
 * for the lifecycle and CAS gating discipline.
 */
typedef enum {
  ARTS_DB_DESTROY_NONE = 0,
  ARTS_DB_DESTROY_MARKED,
  ARTS_DB_DESTROY_CLEANING,
} arts_db_destroy_state_t;

/*--- Per-rank DB cache ---------------------------------------------------
 *
 * Every rank that has acquired or hosts a given DB has one of these.
 * The cache is the runtime's coherence-protocol state: it tracks
 * ownership, the live buffer, parked waiters, and destroy lifecycle.
 *
 *   writer_count        flat ownership counter.  > 0 ⇒ this rank
 *                       holds RW ownership; 0 ⇒ invalidated.
 *                       Sentinel discipline:
 *                         GRANT install ⇒ writer_count = 1 (sentinel)
 *                         each successful waiter-claim ⇒ +1
 *                         INVALIDATE_NOTICE ⇒ -1 (sentinel withdrawal)
 *                         each release ⇒ -1
 *                       The thread whose fetch_sub returns 1 (i.e.
 *                       brings the count to 0) is the unique
 *                       transfer actor.
 *   buffer              currently-installed buffer pointer; readers
 *                       acquire via acquire_buf's CAS-loop "increment
 *                       if positive" against buffer->ref_count.
 *   lock_req_in_flight  RW LOCK_REQ coalescing flag — only the actor
 *                       that CASes false→true sends LOCK_REQ; same-
 *                       node RW EDTs piggyback on the in-flight one
 *                       and are picked up by GRANT's drain.  Cleared
 *                       by the GRANT handler.
 *   pending_rw          Vyukov MPSC queue of RW waiters parked on this
 *                       rank.  Multi-producer (foreign acquires);
 *                       single consumer (home-side dispatcher).  Pure
 *                       FIFO LOCK_REQ ordering.
 *   pending_ro          Harris marked-next list of RO waiters parked
 *                       on this rank.  RO retains marked-list because
 *                       its drain is selective (target_version filter)
 *                       and does not match pure FIFO MPSC semantics.
 *   pending_count       unified live-waiter counter (RW + RO summed).
 *                       Maintained: +1 on every push, -1 on every
 *                       successful mark.  Consulted only by destroy
 *                       finalization (pending_count == 0 ⇒ no live
 *                       waiters).
 *   destroy_state       forward-only tri-state; see arts_db_destroy_state_t.
 *   buffer_pool         per-DB recycle pool of arts_db_buffer_s
 *                       (intrusive Treiber stack).  Buffers are never
 *                       freed during the DB's lifetime — pushed here
 *                       when ref_count → 0, popped on next install.
 *   home                home metadata.  NON-NULL only on the rank that
 *                       is GUID_HOME_RANK for this DB (i.e. the home
 *                       rank).  Other ranks leave it NULL.
 */
/* Adapter: route_table stores arts_db_s* (v2's data layout); v3
 * cache_s lives inside db->coherence_cache.  All v3 paths look up
 * cache via this helper, so when the cutover removes arts_db_s the
 * change is local to one function.  Returns NULL if either the
 * route_table entry doesn't exist or the entry has no cache_s
 * (e.g. PIN/CXL DBs).  Defined in coherence_acquire.c. */
struct arts_db_cache_s *arts_coh_route_table_lookup_cache(arts_guid_t db_guid);

/* Cache_s init kinds — selects how writer_count / home / buffer get
 * initialized.  Per coherence design plan §1006-1031 / §968-988. */
typedef enum {
  /* Creator side, home == self: install buffer, writer_count = 2
   * (sentinel + creator EDT), home struct with rw_holder = self. */
  ARTS_COH_INIT_CREATOR_HOME = 0,
  /* Creator side, home != self: install buffer (creator local),
   * writer_count = 2 (sentinel + creator EDT), no home struct. */
  ARTS_COH_INIT_CREATOR_REMOTE,
  /* Home side, creator != self (DB_CREATE handler): install
   * buffer (zero-init), writer_count = 0, home struct with rw_holder
   * = creator_rank. */
  ARTS_COH_INIT_HOME_RECV,
  /* Lazy install on a sharer that is neither creator nor home, or
   * pre-DB_CREATE arrival on home: no buffer, writer_count = 0. */
  ARTS_COH_INIT_LAZY,
} arts_coh_init_kind_t;

/* Allocate + initialize a fresh coherence cache_s for db_guid.
 * Used by arts_db_create_internal (creator side), the DB_CREATE wire
 * handler (home side), and lazy install on consumer ranks.
 * creator_rank: only consulted when kind == ARTS_COH_INIT_HOME_RECV
 * (used to set home->rw_holder).  Defined in db.c. */
struct arts_db_cache_s *arts_coh_alloc_cache_s(arts_guid_t db_guid,
                                               uint64_t db_size,
                                               arts_coh_init_kind_t kind,
                                               unsigned int creator_rank);

/* Public destroy entry — sends DESTROY_REQ to home; home runs the
 * fan-out and finalize.  Defined in coherence_destroy.c. */
void arts_coh_db_destroy(arts_guid_t db_guid);

/* Cache_s destructor: drains the buffer pool and frees home_s.  Called
 * from arts_db_free when db->coherence_cache is non-NULL (Phase 3.1).
 * The caller (arts_db_free) frees the cache_s struct itself after this
 * routine returns.  Defined in coherence_destroy.c. */
void arts_coh_cache_destructor(struct arts_db_cache_s *cache);

struct arts_db_cache_s {
  volatile unsigned int writer_count;
  /* buffer is read/written via arts_atomic_swap_ptr; declare as
   * `volatile void *` so the helper signature matches. */
  volatile struct arts_db_buffer_s *buffer;
  arts_marked_list_t pending_ro;
  volatile unsigned int pending_count;
  volatile unsigned int destroy_state;
  arts_lockfree_stack_t buffer_pool;
  /* db_guid stored here for symmetry with the protocol pseudocode —
   * acquire_remote_* needs it for the route_table_return_db pairing
   * on ARTS_DB_DESTROYED early-returns, where the cache is in scope
   * but the original guid argument has been lost in the call chain. */
  arts_guid_t db_guid;
  uint64_t db_size;
  struct arts_db_home_s *home;
  /* Back-pointer to the wrapping struct arts_db_s (Phase 3.1).  Set by
   * arts_db_create_internal (and equivalent install paths) right after
   * cache_s is allocated.  Used by arts_coh_try_finalize_destroy to free
   * the db_s + cache_s + buffers in one call (arts_db_free), eliminating
   * the legacy route_table-managed lifecycle.  Stored as void * to avoid
   * a circular include between coherence.h and runtime_types.h. */
  void *db_owner;
#if defined(ARTS_MEMORY_MODEL_LC)
  /* LC: writer_count is a pure ref count (no exclusive ownership; every
   * writer is a non-home node that writes then pushes back to home).
   * No LOCK_REQ coalescing flag, no per-cache RW waiter queue — LC routes
   * RW acquires through acquire_remote_ro so RW waiters use pending_ro.
   *
   * Per-cache WRITEBACK ACK rendezvous: release_rw atomically claims a
   * fresh seq via fetch_add on writeback_seq, embeds it in the WRITEBACK
   * packet, then spin-waits on writeback_acked_seq >= my_seq.  The ACK
   * handler does an atomic monotonic-max on writeback_acked_seq. */
  volatile uint64_t writeback_seq;
  volatile uint64_t writeback_acked_seq;
#elif defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC: owner-side dedup map.  Allocated lazily on first ownership;
   * preserved across ownership transfer (TRANSFER_OWNERSHIP serializes
   * it). */
  struct arts_rank_to_u64_map_s *last_sent_version;
  /* Set by the INVALIDATE_NOTICE handler when writers are still live;
   * release_rw observes this flag and ships TRANSFER_OWNERSHIP when
   * writer_count reaches 0. */
  _Atomic(unsigned int) transfer_pending;
  /* New owner rank extracted from the INVALIDATE_NOTICE message payload;
   * read by the transfer-ship path when transfer_pending is observed. */
  unsigned int incoming_new_owner;
  /* LRC per-cache RW exclusivity machinery. */
  /* RW LOCK_REQ coalescing flag — only the actor that CASes false→true
   * sends LOCK_REQ; same-node RW EDTs piggyback on the in-flight one and
   * are picked up by GRANT's drain.  Cleared by the GRANT handler. */
  volatile unsigned int lock_req_in_flight;
  /* Vyukov MPSC queue of RW waiters parked on this rank. */
  struct arts_pending_rw_queue_s pending_rw;
#else
  /* RC: per-cache WRITEBACK ACK rendezvous.  release_rw on a non-home
   * owner atomically claims a fresh seq via fetch_add on writeback_seq,
   * embeds it in the WRITEBACK packet, then spin-waits on
   * writeback_acked_seq >= my_seq.  ACK handler does an atomic
   * monotonic-max on writeback_acked_seq.  Multiple concurrent releases
   * on the same cache rendezvous independently because each holds a
   * unique seq. */
  volatile uint64_t writeback_seq;
  volatile uint64_t writeback_acked_seq;
  /* RC per-cache RW exclusivity machinery. */
  /* RW LOCK_REQ coalescing flag — only the actor that CASes false→true
   * sends LOCK_REQ; same-node RW EDTs piggyback on the in-flight one and
   * are picked up by GRANT's drain.  Cleared by the GRANT handler. */
  volatile unsigned int lock_req_in_flight;
  /* Vyukov MPSC queue of RW waiters parked on this rank. */
  struct arts_pending_rw_queue_s pending_rw;
#endif
};

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_H */
