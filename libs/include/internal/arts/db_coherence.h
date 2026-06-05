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

/* The DB coherence layout structs (arts_db_buffer_s, arts_pending_rw_queue_s
 * + node, arts_db_snapshot_waiter_s, arts_home_lockreq_queue_s + node,
 * arts_db_cache_s, arts_db_s) and the arts_coh_atomic_uint typedef live in
 * coherence_types.h (pulled in via runtime_types.h), because struct arts_db_s
 * embeds arts_db_cache_s by value as its first member and inlines the
 * home-directory fields, so it needs the complete cache type.  This header
 * keeps only the protocol function declarations. */
#include "arts/runtime_types.h"

/*--- Pending RW queue lifecycle helpers ----------------------------------*/
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

/* Adapter: route_table stores arts_db_s*; the cache_s is embedded by value
 * as the first member.  All coherence paths look up cache via this helper.
 * Returns NULL if either the route_table entry doesn't exist or the entry
 * has no live cache (e.g. PIN/CXL DBs).  Defined in coherence.c. */
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

/* Initialize a coherence cache_s in place for db_guid.  The cache is
 * embedded by value as the first member of struct arts_db_s; the caller
 * allocates (and zeroes) the db_s and passes &db->cache.  Used by
 * db_create_in_place (creator side), the DB_CREATE wire handler (home
 * side), and lazy install on consumer ranks.  creator_rank: only consulted
 * when kind == ARTS_COH_INIT_HOME_RECV (used to set home->rw_holder).
 * Defined in coherence.c. */
void arts_coh_init_cache_s(struct arts_db_cache_s *c, arts_guid_t db_guid,
                           uint64_t db_size, arts_coh_init_kind_t kind,
                           unsigned int creator_rank);

/* Public destroy entry — sends DESTROY_REQ to home; home runs the
 * fan-out and finalize.  Defined in coherence.c. */
void arts_coh_db_destroy(arts_guid_t db_guid);

/* Cache_s destructor: drains the buffer pool and tears down home_s in place.
 * Called from arts_db_free.  Because the cache is embedded by value as the
 * first member of arts_db_s, the caller frees the wrapping db_s after this
 * routine returns — it does NOT free the cache separately.  Defined in
 * coherence.c. */
void arts_coh_cache_destructor(struct arts_db_cache_s *cache);

/*--- Acquire path --------------------------------------------------------
 *
 * Implements the 8-case dispatcher (HOME × OWNER × {RO, RW}), the remote
 * acquire helpers (LOCK_REQ for RW, GET_DATA for RO), the
 * GRANT/DATA_RESPONSE-side drain routines, and the lazy first-touch cache_s
 * allocation for foreign ranks.
 *
 * Return contract:
 *   On success the call returns ARTS_DB_ACQUIRE_OK and writes the
 *   buffer-data pointer into dep->ptr.  Cases 1/3/5/6/2-fast-path take the
 *   synchronous path and finish before returning.  ARTS_DB_ACQUIRE_PARK
 *   indicates the EDT was parked (cases 4/7/8 + case 6 fall-through).  The
 *   caller leaves the dep slot in the "depc_needed not yet decremented"
 *   state; the protocol's trigger path will fill the slot and decrement when
 *   the ownership/data lands. */
typedef enum {
  ARTS_DB_ACQUIRE_OK = 0,
  ARTS_DB_ACQUIRE_PARK,
} arts_db_acquire_result_t;

/* Lazy first-touch: allocate cache_s on this rank if absent.  Used
 * by the dispatcher when arts_route_table_lookup_db_safe returns NULL.
 * Returns NULL only if a concurrent destroy raced ahead. */
struct arts_db_cache_s *arts_coh_lazy_install_cache_s(arts_guid_t db_guid,
                                                      uint64_t db_size);

/* 8-case acquire dispatcher.  edt_guid and slot identify the parked
 * EDT's dep slot if the path requires parking.  The caller is
 * responsible for the route_table ref on the underlying DB entry —
 * this function neither acquires nor releases it. */
arts_db_acquire_result_t arts_handler_db_acquire(struct arts_db_cache_s *cache,
                                                 arts_edt_dep_t *dep,
                                                 arts_guid_t edt_guid,
                                                 unsigned int slot);

/*--- Release path --------------------------------------------------------
 *
 * release_rw is the user-visible RW release entry point.  It bumps the
 * in-buffer version, decrements writer_count, and depending on
 * (rest_count, home == self) issues the appropriate wire message:
 *
 *   R1 (home == self,  rest > 0):  version++; nothing else
 *   R2 (home == self,  rest == 0): version++; local_transfer
 *   R3 (home != self,  rest > 0):  version++; WRITEBACK_NORMAL + await ACK
 *   R4 (home != self,  rest == 0): version++; WRITEBACK_AND_TRANSFER + await
 * ACK
 *
 * If destroy is locally marked when release_rw enters, all wire sends are
 * skipped — destroy commits the runtime to teardown and any state we would
 * have written back is moot. */

/* Release a RW acquire.  Cache-only signature: the dual-stack model keeps
 * user data at cache->user_data and the buf is just a coherence handle owned
 * by the cache, so callers don't track a per-acquire buf pointer. */
void arts_coh_release_rw(struct arts_db_cache_s *cache);

/* Release a RO acquire — currently a no-op (no held ref to drop) but kept as
 * a separate symbol for symmetry and future cutover. */
void arts_coh_release_ro(struct arts_db_cache_s *cache);

#if defined(ARTS_MEMORY_MODEL_RC)
/* RC ownership-transfer chain advance: pop the next queued RW requester,
 * publish it as the new rw_holder, GRANT it the buffer (monotonic-dedup), and
 * decide whether the transfer chain continues.  Holds home.invalidate_in_flight
 * (the ownership-transfer-in-progress baton) at 1 across the WHOLE has_next
 * chain; the baton is cleared at exactly one point — when the queue drains and
 * a race-recheck confirms no successor raced in.  Shared by the three RC GRANT
 * drivers (local-transfer on home, writeback-and-transfer, ownership-return) so
 * the gate lifetime is identical and auditable.  RC build only; LRC drives its
 * chain through the separate INSTALL_ACK round. */
void arts_coh_rc_advance_chain(struct arts_db_cache_s *cache);
#endif

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_H */
