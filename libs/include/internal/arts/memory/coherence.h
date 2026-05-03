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

#include <stdbool.h>
#include <stdint.h>

#include "arts/runtime_types.h"
#include "arts/utils/lockfree_stack.h"
#include "arts/utils/marked_list.h"

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
 * fill on trigger.  Both fields are passed via the marked-list
 * payload because the protocol's "trigger an EDT" semantics require
 * both — slot index decides which depv entry to populate with the
 * acquired buffer pointer. */
struct arts_db_rw_waiter_s {
  arts_marked_list_node_t link; /* FIRST — required by marked-list */
  arts_guid_t edt_guid;
  unsigned int slot;
};

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
 * The MPSC queue and sparse map are referred to here by opaque
 * forward-decl; their concrete implementations live in their own
 * modules and only the home metadata struct needs to embed them.
 */
struct arts_lockfree_mpsc_s;   /* forward decl; defined alongside the impl */
struct arts_rank_to_u64_map_s; /* forward decl; sparse rank-keyed u64 map */

struct arts_db_home_s {
  unsigned int rw_holder;
  struct arts_lockfree_mpsc_s *pending_rw;
  struct arts_rank_to_u64_map_s *last_sent_version;
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
 *   pending_rw          Harris marked-next list of RW waiters parked
 *                       on this rank.
 *   pending_ro          Harris marked-next list of RO waiters parked
 *                       on this rank.
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

struct arts_db_cache_s {
  volatile unsigned int writer_count;
  /* buffer is read/written via arts_atomic_swap_ptr; declare as
   * `volatile void *` so the helper signature matches. */
  volatile struct arts_db_buffer_s *buffer;
  volatile unsigned int lock_req_in_flight;
  arts_marked_list_t pending_rw;
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
  /* Per-cache WRITEBACK ACK rendezvous.  release_rw on a non-home
   * owner atomically claims a fresh seq via fetch_add on writeback_seq,
   * embeds it in the WRITEBACK packet, then spin-waits on
   * writeback_acked_seq >= my_seq.  ACK handler does an atomic
   * monotonic-max on writeback_acked_seq.  Multiple concurrent
   * releases on the same cache rendezvous independently because each
   * holds a unique seq. */
  volatile uint64_t writeback_seq;
  volatile uint64_t writeback_acked_seq;
};

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_H */
