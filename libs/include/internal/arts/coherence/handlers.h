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

/* arts_handler_db_snapshot_response body args.  data_present mirrors the wire
 * packet's tri-state (0 = no data moved, 1 = data, 2 = size-only CTS).  For
 * data_present == 1 the payload either landed one-sided (rdzv_txid != 0; the
 * landing is named by rdzv_cookie and install pairs by txid) or — same-rank
 * self-serve only — rides inline via `data`/`data_size`. */
struct arts_db_snapshot_response_args_s {
  arts_guid_t edt_guid;
  uint32_t slot;
  uint32_t data_present;
  uint64_t version;
  const void *data;
  uint64_t data_size;
  uint64_t db_size;     /* CTS: allocation size; data: landed byte count */
  uint64_t rdzv_txid;   /* != 0: payload PUT into our landing; pair by txid */
  uint64_t rdzv_cookie; /* our landing handle, echoed back */
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
  struct arts_rdzv_landing_s rdzv; /* requester's landing, forwarded by home */
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
/* Consume an in-flight rendezvous whose receiver-side target is gone (the
 * metadata packet reached a destroyed object): registers a discard
 * continuation so the landing frees and the txid pairing table stays
 * leak-free.  No-op for txid == 0. */
void arts_db_rdzv_discard_landing(uint64_t txid, uint64_t cookie);
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

/* Send OWNERSHIP_REQUEST to the DB's home (the home FIFO orders
 * rank-by-rank).  When the requester knows db_size a fresh transfer landing
 * is allocated and advertised in the request; a size-unknown first touch
 * sends landing-less (txid 0) and home answers OWNERSHIP_CTS, whose handler
 * re-issues through this sender with the landing attached. */
void arts_send_db_ownership_request(struct arts_db_cache_s *cache);
/* Cat-C pure body (OWNERSHIP_CTS at the requester): item_v is the db_s the
 * dispatcher acquired; args_v is the OWNERSHIP_CTS packet.  Learns db_size
 * and re-issues the in-flight OWNERSHIP_REQUEST with a landing (the
 * coalescing flag stays held — this is the same round continuing).  MISS
 * (DB destroyed) silently drops. */
void arts_handler_db_ownership_cts(void *item_v, void *args_v);
/* OWNERSHIP_CTS sender: home → first-touch requester (db_size reply). */
void arts_send_db_ownership_cts(unsigned int requester_rank,
                                arts_guid_t db_guid, uint64_t db_size);
/* cv: address of the releaser's stack-local writeback rendezvous
 * (arts_db_wb_rendezvous_s; sem first) as uint64_t, forwarded verbatim and
 * echoed back in WRITEBACK_CTS / WRITEBACK_ACK for pointer-identity wakeup.
 * Same-rank sends carry `data` inline through the OoO args; remote sends are
 * control-only (announce txid==0 / commit txid!=0 — the payload travels
 * one-sided between them, see arts_db_writeback_sync). */
void arts_send_db_writeback(unsigned int home_rank, arts_guid_t db_guid,
                            uint64_t version, uint64_t cv, const void *data,
                            uint64_t data_size, uint64_t rdzv_txid,
                            uint64_t rdzv_cookie);
/* WRITEBACK_CTS: home → releaser, carrying a fresh home landing for an
 * announced dirty writeback (cv echoed verbatim). */
void arts_send_db_writeback_cts(unsigned int releaser_rank, arts_guid_t db_guid,
                                const struct arts_rdzv_landing_s *landing,
                                uint64_t cv);
void arts_send_db_writeback_ack(unsigned int releaser_rank, arts_guid_t db_guid,
                                uint64_t cv);
/* new_owner_rank: rank home selected as next owner; new_owner_rdzv: that
 * rank's transfer landing (from its queued request), forwarded so the current
 * holder can PUT the transfer payload directly (NULL = zero landing —
 * sentinel/data-less round). */
void arts_send_db_ownership_invalidate(
    unsigned int owner_rank, arts_guid_t db_guid, unsigned int new_owner_rank,
    const struct arts_rdzv_landing_s *new_owner_rdzv);
/* Send SNAPSHOT_REQUEST (GET_DATA) to the DB's home, advertising a fresh
 * snapshot landing when db_size is known (multi-rank runs); a size-unknown
 * first touch sends landing-less and the server answers a size-only CTS
 * response (data_present == 2), whose handler re-enters this sender. */
void arts_send_db_snapshot_request(struct arts_db_cache_s *cache,
                                   arts_guid_t edt_guid, uint32_t slot);
/* Send DATA_RESPONSE.  kind (-> wire data_present): 0 = no data moved (a
 * non-NULL landing's cookie is echoed so the requester recycles it), 1 =
 * payload (PUT from src_h's buffer into `landing`; src_h — a strong buffer
 * ref — is CONSUMED: transferred to the PUT's local completion, or released
 * after an inline self-serve), 2 = size-only CTS (db_size).  src_h must be
 * NULL for kinds 0/2. */
void arts_send_db_snapshot_response(unsigned int requester_rank,
                                    arts_guid_t db_guid, uint64_t version,
                                    arts_guid_t edt_guid, uint32_t slot,
                                    uint32_t kind, uint64_t db_size,
                                    const struct arts_rdzv_landing_s *landing,
                                    arts_shared_ptr_t src_h);
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
                                    arts_guid_t edt_guid, uint32_t slot,
                                    const struct arts_rdzv_landing_s *rdzv);

/* Send CONFIRM_ACK from home to the new owner C after home has flipped
 * rw_holder to C. Self-send dispatches the handler inline.
 * piggyback_new_owner == ARTS_LAZY_NO_PENDING_OWNER ⇒ plain CONFIRM_ACK (no
 * pending requester).  Otherwise the round advances and the ack carries the
 * next transfer target, so the new owner's handler applies the INVALIDATE
 * effect in the same message (no separate INVALIDATE, no reorder window). */
void arts_send_db_ownership_confirm_ack(
    unsigned int new_owner_rank, arts_guid_t db_guid,
    unsigned int piggyback_new_owner,
    const struct arts_rdzv_landing_s *piggyback_rdzv);
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
void arts_db_lazy_start_invalidate_round(
    struct arts_db_cache_s *cache, unsigned int new_owner,
    const struct arts_rdzv_landing_s *new_owner_rdzv);
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
/* Send LOCK_REQUEST to the home, advertising this rank's grant/deliver
 * landing: the stable buffer for RW (in-place install is LOCK's fixed-address
 * contract), a FRESH landing for a LAZY RO serve (a stale RO grant racing a
 * newer owner install must not clobber the stable buffer — the ro_return arm
 * discards it).  Size-unknown first touch sends landing-less; home answers
 * LOCK_CTS and the handler re-enters this sender. */
void arts_send_db_lock_request(struct arts_db_cache_s *cache,
                               arts_db_access_mode_t mode);
/* Cat-C pure body (LOCK_CTS at the requester): item_v is the db_s; args_v is
 * the LOCK_CTS packet.  Learns db_size and re-issues the request (echoed
 * mode) with a landing. */
void arts_handler_db_lock_cts(void *item_v, void *args_v);
/* LOCK_CTS sender: home → first-touch requester (db_size + echoed mode). */
void arts_send_db_lock_cts(unsigned int requester_rank, arts_guid_t db_guid,
                           uint64_t db_size, uint32_t mode);
/* arts_send_db_lock_grant: version is the monotone round counter.  The grant
 * payload PUTs into req_rdzv (the requester's advertised landing; src_h — a
 * strong ref on the home buffer — is CONSUMED); `wb`, for RW grants, is
 * home's writeback landing for this grant's release (NULL/zero for RO). */
void arts_send_db_lock_grant(unsigned int requester_rank, arts_guid_t db_guid,
                             arts_db_access_mode_t mode, uint64_t version,
                             const struct arts_rdzv_landing_s *req_rdzv,
                             const struct arts_rdzv_landing_s *wb,
                             arts_shared_ptr_t src_h, uint64_t data_size);
/* arts_send_db_lock_release: cv is the releaser's stack-local sem_t address
 * (RW only; 0 for RO); version is the monotone counter (RW only; 0 for RO).
 * Home echoes cv in the LOCK_RELEASE_ACK to unblock the releaser.  A remote
 * RW release PUTs the dirty bytes into the grant's `wb` landing first and
 * echoes {rdzv_txid, rdzv_cookie} here; `data` rides inline only same-rank. */
void arts_send_db_lock_release(unsigned int home_rank, arts_guid_t db_guid,
                               arts_db_access_mode_t mode, uint64_t version,
                               uint64_t cv, const void *data,
                               uint64_t data_size, uint64_t rdzv_txid,
                               uint64_t rdzv_cookie);
/* arts_send_db_lock_release_ack: home → releaser after installing writeback.
 * Mirrors arts_send_db_writeback_ack; cv is echoed verbatim so the releaser
 * wakes by pointer identity (no seq tracking). */
void arts_send_db_lock_release_ack(unsigned int releaser_rank,
                                   arts_guid_t db_guid, uint64_t cv);

#ifdef ARTS_TIMING_LAZY
/* ===== LOCK-LAZY handlers (Cat-C direct dispatch; stubs until Tasks 5-6) = */

/* arts_handler_db_lock_forward: Cat-C @owner.  item_v is the full wire packet
 * (arts_msg_lock_forward_packet_s *); the handler looks up the db_s internally
 * by ->db_guid.  Handles home→owner forward: either migrate RW to a new owner
 * or serve one RO reader directly from the owner's buffer. */
void arts_handler_db_lock_forward(void *item_v);

/* arts_handler_db_lock_deliver: Cat-C @target.  payload is the full wire
 * buffer (header + inline data); size is total bytes (data starts at
 * sizeof(arts_msg_lock_deliver_packet_s)).  Installs data + transitions cache
 * state (GRANT or RO-hold) and drains any pending local acquires. */
void arts_handler_db_lock_deliver(void *payload, size_t size);

/* arts_handler_db_lock_confirm: Cat-C @home.  item_v is the full wire packet
 * (arts_msg_lock_confirm_packet_s *); the handler looks up the db_s internally
 * by ->db_guid.  New owner → home: migration complete; home transitions
 * lock_state and serves the next waiter if any.  packet->rank = new owner. */
void arts_handler_db_lock_confirm(void *item_v);

/* arts_handler_db_lock_roret: Cat-C @home.  item_v is the full wire packet
 * (arts_msg_lock_confirm_packet_s *); the handler looks up the db_s internally
 * by ->db_guid.  Reader → home: RO release (data-less); home decrements the
 * reader count and serves the next waiter when the count reaches zero.
 * packet->rank = the returning reader. */
void arts_handler_db_lock_roret(void *item_v);

/* ===== LOCK-LAZY senders (stubs until Tasks 5-6) ========================= */

/* arts_send_db_lock_forward: home → current owner.  mode=DB_MODE_RW requests
 * migration to target rank; mode=DB_MODE_RO requests serving one RO reader. */
void arts_send_db_lock_forward(unsigned int owner_rank, arts_guid_t db_guid,
                               uint32_t mode, uint32_t target,
                               const struct arts_rdzv_landing_s *target_rdzv);

/* arts_send_db_lock_deliver: owner → target.  Versionless (LOCK serialization
 * guarantees ordering).  The payload PUTs into the target's forwarded landing
 * (rdzv); src_h — a strong ref on the owner buffer — is CONSUMED. */
void arts_send_db_lock_deliver(unsigned int target_rank, arts_guid_t db_guid,
                               uint32_t mode,
                               const struct arts_rdzv_landing_s *rdzv,
                               arts_shared_ptr_t src_h, uint64_t data_size);

/* arts_send_db_lock_confirm: new owner → home: migration done. */
void arts_send_db_lock_confirm(unsigned int home_rank, arts_guid_t db_guid);

/* arts_send_db_lock_roret: reader → home: RO release (data-less). */
void arts_send_db_lock_roret(unsigned int home_rank, arts_guid_t db_guid);

#endif /* ARTS_TIMING_LAZY */

#endif /* ARTS_PROTOCOL_LOCK */

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_HANDLERS_H */
