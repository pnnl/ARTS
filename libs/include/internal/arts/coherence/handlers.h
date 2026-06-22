/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message handlers.
 *
 * The 11 wire messages defined in arts/transport/protocol.h each
 * land in their own handler here.  Handlers are dispatched from the
 * single per-rank network handler thread (one of S1's invariants),
 * so home-side state — home.pending_rw, home.last_sent_version,
 * home.rw_holder — is touched by exactly one writer.  Worker threads
 * concurrently read/write cache.* via the atomic primitives wired up
 * in the data structures (writer_count, the buffer slot's atomic shared_ptr,
 * the pending_rw/pending_snapshot queues).
 *
 * Drop discipline — two categories:
 *   Cat-B (deferrable: OWNERSHIP_REQUEST / GET_DATA / WRITEBACK / DESTROY /
 *     OWNERSHIP_INVALIDATE): the wire dispatcher routes through
 *     arts_ooo_dispatch_or_defer_guid, which acquires the home db_s
 *     ref-pinned and hands the pure (item, args) body a live cache on
 *     HIT, or defers the args for replay once DB_CREATE installs.
 *   Cat-C (non-deferrable: DATA_RESPONSE / DESTROY_NOTIFY / WRITEBACK_ACK /
 *     RELEASE_OWNERSHIP / REDIRECT_RO / CONFIRM / CONFIRM_ACK): the wire
 * dispatcher (and the matching self-send shortcut) does a ref-pinned lookup via
 *     arts_route_table_lookup_db; on HIT it calls the pure (item, args)
 *     body, on MISS it applies the handler's exact miss-action (silent
 *     drop, DESTROY_NOTIFY reply, or sem-post — see each body).
 *   Either way the handler body itself performs NO route-table lookup; it
 *   operates on the already-acquired, ref-pinned cache passed by the
 *   dispatcher or OoO engine.
 *
 * Handlers receive the full wire packet; size and rank are accessed
 * via the embedded arts_msg_header_s header. */

#ifndef ARTS_MEMORY_COHERENCE_HANDLERS_H
#define ARTS_MEMORY_COHERENCE_HANDLERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arts/transport/protocol.h"

/* Forward declaration for lazy handler parameters. */
struct arts_db_cache_s;

/* ===== Cat-C handler args ============================================
 * Cat-C (silent-drop / response-ACK) handlers are NOT OoO-deferrable: they
 * carry no OOO_* kind and are never enqueued on a slot's ooo_list.  The wire
 * dispatcher (and the matching self-send shortcut) does the ref-pinned
 * lookup-acquire; on a HIT it calls the pure (item, args) body, and on a MISS
 * it applies that handler's exact miss-action (silent drop, DESTROY_NOTIFY
 * reply, or sem-post — see each handler's contract).  These args structs are
 * stack-built by the dispatcher from the decoded wire packet and passed by
 * pointer to the body; unlike the OoO Cat-B args they are never copied into a
 * heap payload, so a trailing data pointer (snapshot_response) stays valid for
 * the handler's duration (it points into the live receive buffer). */

/* arts_handler_db_snapshot_response body args.  `data`/`data_size` point into
 * the live wire receive buffer (data == NULL when data_present == 0). */
struct arts_db_snapshot_response_args_s {
  arts_guid_t edt_guid;
  uint32_t slot;
  uint32_t data_present;
  uint64_t version;
  const void *data;
  uint64_t data_size;
};

/* arts_handler_db_cache_destroy body args. */
struct arts_db_cache_destroy_args_s {
  arts_guid_t db_guid;
};

/* arts_handler_db_writeback_ack body args.  The body is cache-independent
 * (pointer-identity sem-post on cv); the dispatcher posts on both HIT and MISS
 * so a torn-down cache never strands the blocked releaser. */
struct arts_db_writeback_ack_args_s {
  uint64_t cv; /* releaser's stack-local sem_t address */
};

#ifdef ARTS_TIMING_LAZY
/* arts_handler_db_snapshot_redirect body args (owner side). */
struct arts_db_snapshot_redirect_args_s {
  arts_guid_t db_guid;
  arts_guid_t edt_guid;
  unsigned int requester_rank;
  uint32_t slot;
};

/* arts_handler_db_ownership_confirm body args (home side). */
struct arts_db_ownership_response_ack_args_s {
  arts_guid_t db_guid;
  uint64_t version;
};
#endif /* ARTS_TIMING_LAZY */

/* ===== Home-side handlers ============================================ */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_OWNERSHIP_REQUEST]): item_v is the
 * home db_s the engine acquired (cache is its first member); args_v is an
 * arts_ooo_args_db_ownership_request_s.  The wire dispatcher decodes
 * OWNERSHIP_REQUEST into those args and routes through
 * arts_ooo_dispatch_or_defer_guid.  Defined for the release-consistency family
 * (coherence/ownership.c) where OWNERSHIP_REQUEST exists; MRMW provides a
 * no-op body (coherence/mrmw.c) — MRMW never enqueues this kind. */
void arts_handler_db_ownership_request(void *item_v, void *args_v);
/* Cat-B pure body (OoO g_ooo_table[OOO_DB_SNAPSHOT_REQUEST]): item_v is the
 * home db_s the engine acquired (cache is its first member); args_v is an
 * arts_ooo_args_db_snapshot_request_s.  The wire dispatcher decodes the packet
 * into those args and routes through arts_ooo_dispatch_or_defer_guid. */
void arts_handler_db_snapshot_request(void *item_v, void *args_v);
/* Cat-B pure body (OoO g_ooo_table[OOO_DB_WRITEBACK]): item_v is the home db_s
 * the engine acquired (cache is its first member); args_v is an
 * arts_ooo_args_db_writeback_s followed by data_size payload bytes.  The wire
 * dispatcher copies the trailing wire payload into the args blob (after the
 * struct) and routes through arts_ooo_dispatch_or_defer_guid; this body reads
 * the payload from (char *)args_v + sizeof(arts_ooo_args_db_writeback_s). */
void arts_handler_db_writeback(void *item_v, void *args_v);
/* Cat-B pure body (OoO g_ooo_table[OOO_DB_DESTROY]): item_v is the home db_s
 * the engine acquired (cache is its first member); args_v is an
 * arts_ooo_args_db_destroy_s.  The wire dispatcher decodes DESTROY_REQ into
 * those args and routes through arts_ooo_dispatch_or_defer_guid. */
void arts_handler_db_destroy(void *item_v, void *args_v);
void arts_handler_db_create(struct arts_msg_db_create_coherent_packet_s *p);

/* ===== Sharer-side (response) handlers =============================== */

/* OWNERSHIP_RESPONSE at new owner C: payload = full contiguous wire buffer
 * (header + map + data); size is total bytes.  One signature for both timings —
 * EAGER drains + runs immediately, LAZY defers the RW drain to CONFIRM_ACK. */
void arts_handler_db_ownership_response(void *payload, size_t size);
/* Cat-C pure body (DATA_RESPONSE): item_v is the db_s the dispatcher acquired
 * (cache is its first member); args_v is an
 * arts_db_snapshot_response_args_s.  NOT OoO-deferrable — the dispatcher
 * looks the cache up with a held ref and, on a MISS (DB destroyed / slot
 * NULL-stored), SILENTLY DROPS (this 1:1 response resumes a parked EDT; if the
 * cache is gone the EDT was already torn down). */
void arts_handler_db_snapshot_response(void *item_v, void *args_v);
/* Pure (cache, args) body: item_v is the db_s (cache is its first member);
 * args_v is an arts_ooo_args_db_ownership_invalidate_s.  The wire dispatcher /
 * self-send looks the cache up and calls this body DIRECTLY — no OoO defer.  In
 * both timings the home publishes the invalidate target (rw_holder) only at the
 * post-install CONFIRM owner-swap, so the cache is provably installed when
 * INVALIDATE arrives.  Eager/lazy define the real body (sentinel withdrawal /
 * transfer trigger); MRMW provides a no-op body (MRMW never receives
 * INVALIDATE). */
void arts_handler_db_ownership_invalidate(void *item_v, void *args_v);
/* Cat-C pure body (WRITEBACK_ACK): item_v is the db_s the dispatcher acquired
 * (unused — the wake is a cache-independent pointer-identity sem-post on
 * args->cv); args_v is an arts_db_writeback_ack_args_s.  NOT
 * OoO-deferrable.  The dispatcher posts the sem on BOTH a HIT (via this body)
 * and a MISS so a torn-down home cache never strands the blocked releaser. */
void arts_handler_db_writeback_ack(void *item_v, void *args_v);
/* Cat-C pure body (DESTROY_NOTIFY): item_v is the db_s the dispatcher acquired
 * (cache is its first member); args_v is an arts_db_cache_destroy_args_s.
 * NOT OoO-deferrable — the dispatcher looks the cache up with a held ref and,
 * on a MISS (already torn down on this rank), SILENTLY DROPS (idempotent). */
void arts_handler_db_cache_destroy(void *item_v, void *args_v);

/* ===== Lazy-only handlers ============================================== */

#ifdef ARTS_TIMING_LAZY
/* Cat-C pure body (REDIRECT_RO, owner side): item_v is the db_s the dispatcher
 * acquired (cache is its first member); args_v is an
 * arts_db_snapshot_redirect_args_s.  NOT OoO-deferrable — the dispatcher
 * looks the cache up with a held ref and, on a MISS (DB destroyed / not yet
 * installed on this rank), sends DESTROY_NOTIFY to the requester (so the
 * requester's parked RO waiter wakes and observes DB_DESTROYED). */
void arts_handler_db_snapshot_redirect(void *item_v, void *args_v);
#endif /* ARTS_TIMING_LAZY */

#if defined(ARTS_PROTOCOL_MRNEW) || defined(ARTS_PROTOCOL_MRSW)
/* Cat-C pure body (CONFIRM, home side; both timings): item_v is the db_s the
 * dispatcher acquired (cache is its first member); args_v is unused (the new
 * owner is read from db->pending_install_owner).  NOT OoO-deferrable — the
 * dispatcher looks the cache up with a held ref and, on a MISS (DB destroyed),
 * SILENTLY DROPS.  Records the new rw_holder, then starts the next transfer
 * round or releases the invalidate_in_flight baton.  LAZY additionally replies
 * with CONFIRM_ACK; EAGER does not (the new owner already drained at
 * OWNERSHIP_RESPONSE). */
void arts_handler_db_ownership_confirm(void *item_v, void *args_v);

/* Send CONFIRM from new owner C back to home A once the ownership transfer is
 * complete (both timings; home flips rw_holder + advances the round). */
void arts_send_db_ownership_confirm(unsigned int home_rank, arts_guid_t db_guid,
                                    uint64_t version);
#endif /* MRNEW || MRSW */

/* ===== Sender helpers ================================================ */

/* Send a coherence wire packet of the given type with optional
 * trailing payload (data + data_size).  data == NULL ⇒ no payload.
 * Used by handlers that emit replies and by acquire/release.  The
 * OWNERSHIP_REQUEST carries only db_guid (the home FIFO orders rank-by-rank).
 */
void arts_send_db_ownership_request(unsigned int home_rank,
                                    arts_guid_t db_guid);
/* OWNERSHIP_RESPONSE wire sender (shared, both timings): carries the serialized
 * last_sent_version map + buffer payload (no edt — the EDT rides CONFIRM). */
void arts_send_db_ownership_response(unsigned int new_owner_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     const void *map_buf, size_t map_size,
                                     const void *data, size_t data_size);
/* cv: address of the releaser's stack-local sem_t (as uint64_t), forwarded
 * verbatim to the home and echoed back in the ACK for pointer-identity wakeup.
 */
void arts_send_db_writeback(unsigned int home_rank, arts_guid_t db_guid,
                            uint64_t version, uint64_t cv, const void *data,
                            uint64_t data_size);
void arts_send_db_writeback_ack(unsigned int releaser_rank, arts_guid_t db_guid,
                                uint64_t cv);
/* new_owner_rank: rank that home has selected as next owner.  Eager builds
 * pass 0 (receiver ignores it); lazy builds embed it in the packet so the
 * holder knows where to ship TRANSFER_OWNERSHIP without a home round-trip. */
void arts_send_db_ownership_invalidate(unsigned int owner_rank,
                                       arts_guid_t db_guid,
                                       unsigned int new_owner_rank);
void arts_send_db_snapshot_request(unsigned int home_rank, arts_guid_t db_guid,
                                   arts_guid_t edt_guid, uint32_t slot);
void arts_send_db_snapshot_response(unsigned int requester_rank,
                                    arts_guid_t db_guid, uint64_t version,
                                    arts_guid_t edt_guid, uint32_t slot,
                                    const void *data, uint64_t data_size);
void arts_send_db_create_coherent(unsigned int home_rank, arts_guid_t db_guid,
                                  uint64_t db_size, uint16_t flags,
                                  uint16_t db_type);
void arts_send_db_destroy(unsigned int home_rank, arts_guid_t db_guid);
void arts_send_db_cache_destroy(unsigned int sharer_rank, arts_guid_t db_guid);

#ifdef ARTS_TIMING_LAZY
/* Send REDIRECT_RO from home to the current owner, asking the owner to
 * serve DATA_RESPONSE or a no-data response directly to requester_rank. */
void arts_send_db_snapshot_redirect(unsigned int owner_rank,
                                    arts_guid_t db_guid,
                                    unsigned int requester_rank,
                                    arts_guid_t edt_guid, uint32_t slot);

/* Send CONFIRM_ACK from home to the new owner C after home has flipped
 * rw_holder to C. Self-send dispatches the handler inline.
 * piggyback_new_owner == ARTS_LAZY_NO_PENDING_OWNER ⇒ plain CONFIRM_ACK (no
 * pending requester).  Otherwise the round advances and the ack carries the
 * next transfer target, so the new owner's handler applies the INVALIDATE
 * effect in the same message (no separate INVALIDATE, no reorder window). */
void arts_send_db_ownership_confirm_ack(unsigned int new_owner_rank,
                                        arts_guid_t db_guid,
                                        unsigned int piggyback_new_owner);
/* Cat-C pure body (CONFIRM_ACK, new-owner side): item_v is the db_s the
 * dispatcher acquired (cache is its first member); args_v is the
 * CONFIRM_ACK packet (its new_owner_rank carries the piggybacked invalidate
 * target, or ARTS_LAZY_NO_PENDING_OWNER for a plain ack). Drains the parked RW
 * waiters that the TRANSFER handler deferred, clears the gate, applies the
 * piggybacked invalidate effect (publish incoming_new_owner + withdraw the
 * sentinel), and removes the drain guard (the relocated 0-edge ship-check). NOT
 * OoO-deferrable — the dispatcher looks the cache up with a held ref and, on a
 * MISS (DB destroyed), SILENTLY DROPS (the gated waiters are woken by the
 * destroy fan-out instead). */
void arts_handler_db_ownership_confirm_ack(void *item_v, void *args_v);

/* Kick a new INVALIDATE_NOTICE round: read rw_holder, send notice to
 * holder carrying new_owner as the TRANSFER_OWNERSHIP target. */
void arts_db_lazy_start_invalidate_round(struct arts_db_cache_s *cache,
                                         unsigned int new_owner);
#endif /* ARTS_TIMING_LAZY */

#if defined(ARTS_PROTOCOL_MRNEW) || defined(ARTS_PROTOCOL_MRSW)
/* Shared owner→owner transfer ship (defined in coherence/<proto>/ownership.c):
 * ship the current buffer (+ serialized owner-side map for LAZY, empty map for
 * EAGER) to cache->incoming_new_owner via the OWNERSHIP_RESPONSE wire,
 * re-arming incoming_new_owner to the sentinel before sending.  Called on the
 * 0-edge of release_rw / the INVALIDATE handler (both timings) and the LAZY
 * CONFIRM_ACK / EAGER OWNERSHIP_RESPONSE drain-guard removal. */
void arts_db_send_ownership_response(struct arts_db_cache_s *cache);
#endif /* MRNEW || MRSW */

#ifdef ARTS_PROTOCOL_LOCK
/* ===== LOCK protocol handlers ======================================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_LOCK_REQUEST]) @home: item_v is the
 * home db_s the engine acquired (cache is its first member); args_v is an
 * arts_ooo_args_db_lock_request_s.  push-before-CAS on rw_waiters/ro_waiters,
 * then a single lock_state CAS, then grant-send per the transition case. */
void arts_handler_db_lock_request(void *item_v, void *args_v);

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_LOCK_RELEASE]) @home: same item_v
 * shape; args_v is an arts_ooo_args_db_lock_release_s followed by data_size
 * writeback bytes when mode==RW.  RW installs writeback first, then transitions
 * lock_state, then grant-sends the next holder(s). */
void arts_handler_db_lock_release(void *item_v, void *args_v);

/* Cat-C pure body @requester: payload = full contiguous wire buffer
 * (header + data); size is total bytes.  Buffer install + cache_state CAS
 * (REQ→GRANT, or phantom RO release) + drain of the cache pending stacks. */
void arts_handler_db_lock_grant(void *payload, size_t size);

/* NOTE: arts_handler_db_acquire is NOT re-declared here.  It is already
 * declared protocol-agnostically in coherence.h as void
 * arts_handler_db_acquire(void *item, void *args) and serves as both the
 * engine's acquire_one_dep body and the OOO_DB_ACQUIRE replay.  LOCK only
 * *defines* it (lock/acquire.c). */

/* ===== LOCK senders =================================================== */
void arts_send_db_lock_request(unsigned int home_rank, arts_guid_t db_guid,
                               arts_db_access_mode_t mode);
/* arts_send_db_lock_grant: version is the monotone round counter; home bumps
 * and forwards it so the requester's buf_install rejects stale grants. */
void arts_send_db_lock_grant(unsigned int requester_rank, arts_guid_t db_guid,
                             arts_db_access_mode_t mode, uint64_t version,
                             const void *data, uint64_t data_size);
/* arts_send_db_lock_release: cv is the releaser's stack-local sem_t address
 * (RW only; 0 for RO); version is the monotone counter (RW only; 0 for RO).
 * Home echoes cv in the LOCK_RELEASE_ACK to unblock the releaser. */
void arts_send_db_lock_release(unsigned int home_rank, arts_guid_t db_guid,
                               arts_db_access_mode_t mode, uint64_t version,
                               uint64_t cv, const void *data,
                               uint64_t data_size);
/* arts_send_db_lock_release_ack: home → releaser after installing writeback.
 * Mirrors arts_send_db_writeback_ack; cv is echoed verbatim so the releaser
 * wakes by pointer identity (no seq tracking). */
void arts_send_db_lock_release_ack(unsigned int releaser_rank,
                                   arts_guid_t db_guid, uint64_t cv);

#endif /* ARTS_PROTOCOL_LOCK */

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_HANDLERS_H */
