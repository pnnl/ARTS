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

#include <semaphore.h> /* sem_t — await_writeback_ack signature */
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

/* The DB coherence layout structs (arts_db_buffer_s, arts_db_rw_waiter_s,
 * arts_db_snapshot_waiter_s, arts_home_lockreq_queue_s + node,
 * arts_db_cache_s, arts_db_s) and the arts_db_atomic_uint_t typedef live in
 * coherence_types.h (pulled in via runtime_types.h), because struct arts_db_s
 * embeds arts_db_cache_s by value as its first member and inlines the
 * home-directory fields, so it needs the complete cache type.  This header
 * keeps only the protocol function declarations. */
#include "arts/runtime_types.h"
#include "arts/utils/shared.h" /* arts_shared_ptr_t — lazy_install return type */

/*--- Pending RW Treiber-stack lifecycle helpers --------------------------
 * The per-cache RW waiter chain exists only under MRNEW: it parks same-node RW
 * EDTs that piggyback on an in-flight ownership round.  MRSW removes it — every
 * writer is ordered through the home FIFO and woken by a route-table lookup, so
 * the worker never parks RW EDTs in a per-cache queue. */
#if defined(ARTS_PROTOCOL_MRNEW)
void arts_pending_rw_queue_init(arts_lf_stack_t *q);
/* Push a waiter (multi-producer).  Caller fills edt_guid/slot before
 * calling.  Waiter must be heap-allocated; the stack takes ownership and
 * frees it during drain or destroy. */
void arts_pending_rw_queue_push(arts_lf_stack_t *q,
                                struct arts_db_rw_waiter_s *w);
/* Drain everything (single consumer): atomic-exchange the whole chain out,
 * then invoke cb(edt_guid, slot, ctx) on each waiter and free it.  Order is
 * LIFO and immaterial (every waiter is woken).  cb must NOT block — drain
 * holds no lock but is intended for short tasks (mark-EDT-ready). */
void arts_pending_rw_queue_drain(arts_lf_stack_t *q,
                                 void (*cb)(arts_guid_t edt_guid,
                                            unsigned int slot, void *ctx),
                                 void *ctx);
/* Destroy: free every queued waiter (single-threaded at teardown). */
void arts_pending_rw_queue_destroy(arts_lf_stack_t *q);
#endif /* MRNEW */

/* Cache_s init kinds — selects how writer_count / home / buffer get
 * initialized.  Per coherence design plan §1006-1031 / §968-988. */
typedef enum {
  /* Creator side, home == self: install buffer, writer_count = 2
   * (sentinel + creator EDT), home struct with rw_holder = self. */
  ARTS_DB_INIT_CREATOR_HOME = 0,
  /* Creator side, home != self: install buffer (creator local),
   * writer_count = 2 (sentinel + creator EDT), no home struct. */
  ARTS_DB_INIT_CREATOR_REMOTE,
  /* Home side, creator != self (DB_CREATE handler): install
   * buffer (zero-init), writer_count = 0, home struct with rw_holder
   * = creator_rank. */
  ARTS_DB_INIT_HOME_RECV,
  /* Lazy install on a sharer that is neither creator nor home, or
   * pre-DB_CREATE arrival on home: no buffer, writer_count = 0. */
  ARTS_DB_INIT_LAZY,
} arts_db_init_kind_t;

/* Initialize a coherence cache_s in place for db_guid.  The cache is
 * embedded by value as the first member of struct arts_db_s; the caller
 * allocates (and zeroes) the db_s and passes &db->cache.  Used by
 * db_create_in_place (creator side), the DB_CREATE wire handler (home
 * side), and lazy install on consumer ranks.  creator_rank: only consulted
 * when kind == ARTS_DB_INIT_HOME_RECV (used to set home->rw_holder).
 * Defined in coherence.c. */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank);

/* Public destroy entry — sends DESTROY_REQ to home; home runs the
 * fan-out and finalize.  Defined in coherence.c. */
void arts_db_destroy_remote(arts_guid_t db_guid);

/* Cache_s destructor: drops the buffer slot's shared_ptr ref (the cb deleter
 * frees the buffer once the last in-flight acquirer releases) and tears down
 * home_s in place.
 * Called from arts_db_free.  Because the cache is embedded by value as the
 * first member of arts_db_s, the caller frees the wrapping db_s after this
 * routine returns — it does NOT free the cache separately.  Defined in
 * coherence.c. */
void arts_db_cache_destructor(struct arts_db_cache_s *cache);

/*--- Acquire path --------------------------------------------------------
 *
 * Implements the 8-case dispatcher (HOME × OWNER × {RO, RW}), the remote
 * acquire helpers (OWNERSHIP_REQUEST for RW, GET_DATA for RO), the
 * GRANT/DATA_RESPONSE-side drain routines, and the lazy first-touch cache_s
 * allocation for foreign ranks.
 *
 * Result enum, returned by the remote acquire helpers
 * (arts_db_acquire_remote_ro / arts_db_acquire_remote_rw): OK means data was
 * resolved synchronously into dep->ptr; PARK means the EDT was parked on the
 * coherence protocol and a later wake (DATA_RESPONSE / GRANT / TRANSFER) will
 * deliver it.  The acquire handler itself (arts_handler_db_acquire, below) is a
 * void self-accounting body: a synchronous resolve calls
 * arts_db_acquire_resolved (count the dep + advance the RW cursor); a remote
 * request parks. */
typedef enum {
  ARTS_DB_ACQUIRE_OK = 0,
  ARTS_DB_ACQUIRE_PARK,
} arts_db_acquire_result_t;

/* Lazy first-touch: allocate cache_s on this rank if absent.  Used by the
 * acquire path when a lookup_db miss is observed on a remote-owned DB.
 * Returns a PINNED handle to the db_s whose cache it installed; the caller
 * MUST arts_shared_release it on every control-flow path.  A NULL handle =
 * the DB was destroyed before the install could be observed. */
arts_shared_ptr_t arts_db_cache_lazy_install(arts_guid_t db_guid,
                                             uint64_t db_size);

/* OOO_DB_ACQUIRE Cat-B body (per model). item = the installed db_s; args =
 * arts_ooo_args_db_acquire_s {edt, db_guid, slot}. Attempts the one dep's
 * acquire (mode x ownership dispatch); on a synchronous resolve calls
 * arts_db_acquire_resolved; on a remote request, parks. Used by BOTH the driver
 * (acquire_one_dep) and the OoO drain — this IS the OOO_DB_ACQUIRE handler. */
void arts_handler_db_acquire(void *item, void *args);

/* Decrement acquire_remaining; schedule the EDT if it reaches 0. Caller must
 * not touch the EDT afterward (it may have been scheduled + run). */
void arts_db_acquire_account(struct arts_edt_s *edt);

/* A dep resolved locally (data now in dep->ptr): count it down, and if it is a
 * serialized (RW) dep advance the cursor + fire the next serialized dep. */
void arts_db_acquire_resolved(struct arts_edt_s *edt, unsigned int slot);

/* Secured wake (position-idempotent): advance the RW cursor past `slot` if it
 * still points there, then fire the next serialized dep. Does NOT touch
 * acquire_remaining. Callers: PROCEED handler + GRANT/TRANSFER drain. */
void mark_edt_secured_by_guid(arts_guid_t edt_guid, unsigned int slot);

/* Per-protocol classification used by the arts_db_acquire_all driver: returns
 * true for deps that take exclusive ownership through the home directory and
 * must be GUID-serialized (eager/lazy RW). RO is never serialized; MRMW
 * serializes nothing (every acquire is a home snapshot). Defined in
 * coherence/{eager,lazy,mrmw}.c. */
bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode);

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
void arts_db_release_rw(struct arts_db_cache_s *cache);

/* Release a RO acquire — currently a no-op (no held ref to drop) but kept as
 * a separate symbol for symmetry and future cutover. */
void arts_db_release_ro(struct arts_db_cache_s *cache);

/* ===== Shared coherence services (defined in coherence.c) ============
 * Model TUs (and the release-family TU) call back into these model-agnostic
 * helpers.  These are the genuinely shared coherence internals the per-model
 * handler bodies reuse; they carry no model #ifdef. */

/* Block on a stack-local semaphore until the matching WRITEBACK_ACK posts it
 * (pointer identity); returns early if teardown begins.  Used by the eager and
 * MRMW release-tail bodies. */
void await_writeback_ack(sem_t *cv);

/* Take the EDT's strong buffer ref and return buf->data (NULL when no buffer is
 * installed).  Used by the per-protocol acquire bodies. */
void *arts_db_acquire_local(struct arts_db_cache_s *cache);

/* Fire SNAPSHOT_REQUEST (edt_guid + slot) to home and PARK.  Shared by all
 * three protocol acquire bodies. */
arts_db_acquire_result_t
arts_db_acquire_remote_ro(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
                          unsigned int slot);

/* Wake a parked EDT's dep slot (re-derives dep->ptr from the installed buffer).
 * Used by the drain paths and the response handlers. */
void mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot);

/* Drain the snapshot reorder buffer in one atomic_exchange (monotonic version
 * guarantees a full drain is always correct).  Called from the install paths
 * (GRANT / TRANSFER_OWNERSHIP / DATA_RESPONSE case 2) and destroy fan-out. */
void arts_db_drain_pending_snapshot(struct arts_db_cache_s *cache);

/* Destroy/fail fan-out: wake every parked waiter (RW FIFO + snapshot reorder
 * buffer) with a NULL ptr so the EDT observes the destroyed DB.  The RW-queue
 * drain is protocol-specific (eager/lazy drain pending_rw, MRMW has none);
 * all arms drain pending_snapshot via arts_db_drain_pending_snapshot. */
void arts_db_fail_trigger_pending(struct arts_db_cache_s *cache);

/* Shared cache_s construct/destruct sub-helpers.  The per-protocol
 * arts_db_cache_init / arts_db_cache_destructor wrap these, preserving the
 * exact order: protocol field-init runs BEFORE arts_db_cache_common_init on
 * construct; on destruct the wrapper runs arts_db_cache_common_destroy_pre
 * (buffer-NULL) → protocol field-destroy → arts_db_cache_common_destroy_post
 * (snapshot drain → home teardown).  Splitting the destruct into pre/post lets
 * the protocol field-destroy land between the buffer-NULL and the snapshot/home
 * teardown, matching the original single-TU ordering. */
void arts_db_cache_common_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                               uint64_t db_size, arts_db_init_kind_t kind,
                               unsigned int creator_rank);
void arts_db_cache_common_destroy_pre(struct arts_db_cache_s *cache);
void arts_db_cache_common_destroy_post(struct arts_db_cache_s *cache);

/* Case-D (arts_handler_db_create) per-protocol leaf functions.
 * publish_holder: eager/lazy store creator_rank as the home rw_holder, MRMW
 * no-op; install_home_buffer: MRMW installs a version-1 zero buffer (home
 * is always canonical), eager/lazy defer the install to the creator's first
 * WRITEBACK (no-op here).  Defined once per protocol TU. */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank);
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size);

#if defined(ARTS_PROTOCOL_MRNEW) || defined(ARTS_PROTOCOL_MRSW)
/* Single-owner ownership machinery shared by the MRNEW and MRSW protocols
 * (defined in coherence/<proto>/ownership.c).  Called by the EAGER/LAZY
 * arts_handler_db_acquire bodies; the RO-path predicate is the only divergence
 * between EAGER and LAZY, so it stays inline in each protocol's handler.
 *
 * arts_db_acquire_remote_rw: the remote-RW acquire path.  MRNEW pushes a
 * pending_rw waiter and kicks a OWNERSHIP_REQUEST if none is in flight; MRSW
 * always sends OWNERSHIP_REQUEST(edt_guid, slot) to home (no per-cache queue,
 * single-writer ordering) — both return PARK. */
arts_db_acquire_result_t
arts_db_acquire_remote_rw(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
                          unsigned int slot);

/* Per-protocol ownership-round seams (defined in coherence/eager.c and
 * coherence/lazy.c, called from the OWNERSHIP_REQUEST / RELEASE_OWNERSHIP
 * handlers).  start: the eager protocol INVALIDATEs the current holder; the
 * lazy protocol pops the FIFO target + starts the invalidate round.  return:
 * the eager protocol advances the chain; the lazy protocol never receives
 * RELEASE_OWNERSHIP (no-op). */
void arts_db_start_ownership_round(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester);
#endif /* MRNEW || MRSW */

#if defined(ARTS_PROTOCOL_MRNEW)
/* arts_db_acquire_rw_local_fast: the case-2/6 RW local fast path (MRNEW).
 * CAS-increments writer_count "if positive"; on success writes dep->ptr
 * (acquire_local) and returns true; returns false when writer_count went to 0
 * (ownership invalidated) so the caller falls through to
 * arts_db_acquire_remote_rw. */
bool arts_db_acquire_rw_local_fast(struct arts_db_cache_s *cache,
                                   arts_edt_dep_t *dep);

/* Non-destructive enumeration of a cache pending_rw queue (single consumer):
 * invokes cb(edt_guid, slot, ctx) for each parked waiter without popping. */
void arts_pending_rw_queue_for_each(arts_lf_stack_t *q,
                                    void (*cb)(arts_guid_t edt_guid,
                                               unsigned int slot, void *ctx),
                                    void *ctx);
#endif /* MRNEW */

#if defined(ARTS_PROTOCOL_MRSW)
/* arts_db_acquire_rw_local_fast: the case-2/6 RW local fast path (MRSW).  MRSW
 * caps the local active-writer count at one (the token): it pushes (edt_guid,
 * slot) onto cache.pending_rw, then claims the token (idle owner: CAS 1->2 +
 * run the FIFO head) or returns true to be popped by the active writer's
 * release / the drain.  Returns false only when this rank is NOT the owner, so
 * the caller falls through to arts_db_acquire_remote_rw.  Does NOT write
 * dep->ptr (the run path delivers it). */
bool arts_db_acquire_rw_local_fast(struct arts_db_cache_s *cache,
                                   arts_edt_dep_t *dep, arts_guid_t edt_guid,
                                   unsigned int slot);

/* Pop exactly ONE waiter from cache.pending_rw and deliver it (mark secured +
 * ready).  No per-waiter writer_count bump — the single token already accounts
 * the one active writer.  Called by the idle-owner token claim, the GRANT
 * drain, and the release token hand-off.  Returns true iff a waiter was popped
 * (false ⇒ queue empty: the caller may need to drop an orphan token). */
bool arts_db_mrsw_run_one(struct arts_db_cache_s *cache);

/* MRSW local RW release: pop-then-conditional-sub with a cheap LOCAL Dekker
 * re-check (defined in coherence/mrsw/ownership.c; called from
 * arts_db_release_rw after the per-timing writeback/version bump). */
void arts_db_release_rw_local(struct arts_db_cache_s *cache);
#endif /* MRSW */

#if defined(ARTS_PROTOCOL_MRNEW) || defined(ARTS_PROTOCOL_MRSW)
/* GRANT drain: install the granted waiters.  MRNEW pops every pending_rw waiter
 * (bump writer_count per waiter); MRSW pops exactly one (the token provides the
 * single-writer floor).  Defined in coherence/<proto>/ownership.c; called from
 * the EAGER GRANT handler and the LAZY CONFIRM_ACK handler. */
void arts_db_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                          uint64_t version, bool has_next);
#endif /* MRNEW || MRSW */

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_H */
