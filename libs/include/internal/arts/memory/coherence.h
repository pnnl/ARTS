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
 * + node, arts_db_ro_waiter_s, arts_home_lockreq_queue_s + node,
 * arts_home_pending_ro_queue_s, arts_db_home_s, arts_db_cache_s) and the
 * arts_coh_atomic_uint typedef live in runtime_types.h, because struct
 * arts_db_s embeds arts_db_cache_s by value as its first member and therefore
 * needs the complete type.  This header keeps only the protocol function
 * declarations. */
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
 * has no live cache (e.g. PIN/CXL DBs).  Defined in coherence_acquire.c. */
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
 * arts_db_create_internal (creator side), the DB_CREATE wire handler (home
 * side), and lazy install on consumer ranks.  creator_rank: only consulted
 * when kind == ARTS_COH_INIT_HOME_RECV (used to set home->rw_holder).
 * Defined in coherence_cache.c. */
void arts_coh_init_cache_s(struct arts_db_cache_s *c, arts_guid_t db_guid,
                           uint64_t db_size, arts_coh_init_kind_t kind,
                           unsigned int creator_rank);

/* Public destroy entry — sends DESTROY_REQ to home; home runs the
 * fan-out and finalize.  Defined in coherence_destroy.c. */
void arts_coh_db_destroy(arts_guid_t db_guid);

/* Cache_s destructor: drains the buffer pool and tears down home_s in place.
 * Called from arts_db_free.  Because the cache is embedded by value as the
 * first member of arts_db_s, the caller frees the wrapping db_s after this
 * routine returns — it does NOT free the cache separately.  Defined in
 * coherence_destroy.c. */
void arts_coh_cache_destructor(struct arts_db_cache_s *cache);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_H */
