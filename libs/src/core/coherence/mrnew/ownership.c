/* SPDX-License-Identifier: Apache-2.0
 *
 * MRNEW shared ownership machinery (compiled for both coherence
 * protocols; MRMW does not compile it).
 *
 * Compiled only when ARTS_COHERENCE_PROTOCOL is MRNEW (selected in
 * libs/src/core/CMakeLists.txt).  Holds the home-directory / single-owner
 * ownership machinery that EAGER and LAZY share but MRMW does not have
 * (MRMW keeps the canonical buffer at home via synchronous WRITEBACK and has
 * no OWNERSHIP_REQUEST / transfer / pending_rw round).  The small points where
 * EAGER and LAZY themselves differ (the drain-now vs CONFIRM_ACK-gated drain)
 * live in their per-protocol OWNERSHIP_RESPONSE / CONFIRM handlers
 * (coherence/eager.c / coherence/lazy.c); the owner→owner transfer ship + wire
 * sender + CONFIRM sender + home-side OWNERSHIP_REQUEST/INVALIDATE machinery
 * are shared here.  Contains a single ARTS_TIMING_LAZY guard (the INVALIDATE
 * self-send direct-call vs eager OoO-defer).
 */
#include <assert.h> /* lazy INVALIDATE direct-call invariant assert */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h" /* arts_db_buf_acquire (invalidate xfer) */
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/home.h" /* arts_home_lockreq_queue_push */
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C self-send) */
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"     /* arts_global_rank_id */
#include "arts/transport/outbox.h"   /* arts_transport_send_async */
#include "arts/transport/protocol.h" /* arts_fill_packet_header, MSG_* */
#include "arts/utils/malloc.h" /* arts_malloc / arts_free (transfer ship) */

/* ===== Case 2/6: RW local fast path ================================
 * Shared by the EAGER and LAZY arts_handler_db_acquire bodies
 * (coherence/eager.c / coherence/lazy.c).  CAS-increments writer_count "if
 * positive"; on success writes dep->ptr (acquire_local) and returns true;
 * returns false when writer_count went to 0 (ownership invalidated) so the
 * caller falls through to arts_db_acquire_remote_rw. */
bool arts_db_acquire_rw_local_fast(struct arts_db_cache_s *cache,
                                   arts_edt_dep_t *dep) {
  /* CAS-loop "increment if owner": never bump from <= 0 (0 == invalidated).
   * writer_count is non-negative by construction: the post-install rw_holder
   * flip (CONFIRM-driven, both timings) guarantees an INVALIDATE never reaches
   * a rank before its GRANT install, so GRANT(+sentinel) strictly precedes
   * INVALIDATE(-1); the install's sentinel+guard (+2) further holds the count
   * >= 0 if the next round's INVALIDATE lands mid-install on another receiver
   * thread.  The (int) cast is therefore defensive — with a non-negative count
   * `<= 0` reduces to `== 0` (not owner). */
  while (1) {
    unsigned int wc = arts_atomic_read(&cache->writer_count);
    if ((int)wc <= 0) {
      return false; /* not owner (incl. transient negative) — go remote-RW. */
    }
    if (arts_atomic_cswap(&cache->writer_count, wc, wc + 1) == wc) {
      /* writer_count bumped.  acquire_local NULL is fine (sentinel /
       * version-0); release_rw will decrement the matching bump. */
      dep->ptr = arts_db_acquire_local(cache);
      return true;
    }
  }
}

/* ===== Case 4/8: remote-RW path ==================================== */

arts_db_acquire_result_t
arts_db_acquire_remote_rw(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
                          unsigned int slot) {
  /* No destroy precheck: handle_destroy_req NULL-stores route_item->data BEFORE
   * detaching the slot, so route_table_lookup_db already misses and the
   * caller's OoO defer handles "DB destroyed".  Destroying a DB an EDT still
   * has a pending acquire on is undefined per OCR; a waiter left parked at that
   * point is freed (not woken) by the refcount-0 cache destructor. */
  /* Allocate + push waiter into MPSC queue. */
  struct arts_db_rw_waiter_s *w =
      (struct arts_db_rw_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt_guid;
  w->slot = slot;
  /* Note: under MPSC there is no per-node "mark" — the consumer simply pops in
   * FIFO order and wakes each popped waiter (drain_pending_rw_after_grant on
   * the grant path); the refcount-0 destructor frees any waiter still parked at
   * destroy. */
  arts_pending_rw_queue_push(&cache->pending_rw, w);

  /* Kick OWNERSHIP_REQUEST if no one else has — GRANT is what eventually
   * triggers our drain in FIFO order. */
  if (arts_atomic_cswap(&cache->ownership_req_in_flight, 0, 1) == 0) {
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_ownership_request(home_rank, cache->db_guid);
  }
  return ARTS_DB_ACQUIRE_PARK;
}

/* The arts_handler_db_acquire 8-case body lives per-protocol in
 * coherence/eager.c and coherence/lazy.c — the two builds differ only on the
 * RO-has-local-data predicate (eager: is_home||is_owner; lazy: is_owner), which
 * the C-preprocessor seam forbids in a shared TU.  Both call the shared
 * arts_db_acquire_rw_local_fast / arts_db_acquire_remote_rw above and
 * arts_db_acquire_remote_ro (coherence/coherence.c).
 */

/* ===== GRANT drain (called from coherence/eager.c and coherence/lazy.c) === */

/* Drain callback context for the RW MPSC pop loop. */
struct rw_drain_ctx_s {
  struct arts_db_cache_s *cache;
};

static void rw_drain_cb(arts_guid_t edt_guid, unsigned int slot, void *vctx) {
  struct rw_drain_ctx_s *ctx = (struct rw_drain_ctx_s *)vctx;
  /* Each popped waiter claims exactly one writer_count slot (FIFO). */
  arts_atomic_add(&ctx->cache->writer_count, 1);
  /* Advance the RW cursor first (position-idempotent; never schedules), THEN
   * deliver data (may schedule + let another worker run/free the EDT).
   * Sentinel DBs (db_size==0) have cache->buffer==NULL by design;
   * mark_edt_ready_by_guid handles that cleanly (depv[slot].ptr=NULL, still
   * accounts the dep). */
  mark_edt_secured_by_guid(edt_guid, slot);
  mark_edt_ready_by_guid(edt_guid, slot);
}

void arts_db_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                          uint64_t version, bool has_next) {
  (void)version;
  (void)
      has_next; /* chain continuation is home-driven (advance_chain INVALIDATEs
                 * the new owner when the queue is still non-empty); the owner
                 * no longer self-withdraws its sentinel. */
  struct rw_drain_ctx_s ctx = {.cache = cache};
  arts_pending_rw_queue_drain(&cache->pending_rw, rw_drain_cb, &ctx);
}

/* ===== Shared owner→owner transfer ship (EAGER + LAZY) =============
 * Ship the current buffer to cache->incoming_new_owner via the
 * OWNERSHIP_RESPONSE wire, re-arming incoming_new_owner to the sentinel BEFORE
 * the send (a self-transfer dispatches the new owner's install inline, which
 * can recursively republish incoming_new_owner for the next round; clearing it
 * up front leaves that fresh publish intact).  LAZY serializes its owner-side
 * dedup map; EAGER has no owner-side map (last_sent_version == NULL) and emits
 * an empty map (count=0): the receiver unconditionally reconstructs map_size >=
 * 8, so an omitted header would underflow data_size and corrupt the install. */
void arts_db_send_ownership_response(struct arts_db_cache_s *cache) {
  unsigned int new_owner = cache->incoming_new_owner;
  cache->incoming_new_owner =
      ARTS_LAZY_NO_PENDING_OWNER; /* re-arm before send */
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf == NULL) {
    /* Sentinel DB (db_size==0) or pre-publication: empty transfer + empty map
     * count-header. */
    uint32_t empty_map[2] = {0u, 0u};
    arts_send_db_ownership_response(new_owner, cache->db_guid, /*version=*/0,
                                    empty_map, sizeof(empty_map), /*data=*/NULL,
                                    /*data_size=*/0);
    return;
  }
  size_t map_max =
      (sizeof(uint32_t) * 2) + ((size_t)arts_global_rank_count *
                                sizeof(struct arts_msg_rank_version_pair_s));
  void *map_buf = arts_malloc(map_max);
  size_t map_size;
  if (cache->last_sent_version != NULL) {
    map_size = arts_rank_u64_map_serialize(cache->last_sent_version, map_buf);
  } else {
    uint32_t *p = (uint32_t *)map_buf;
    p[0] = 0u;
    p[1] = 0u;
    map_size = sizeof(uint32_t) * 2;
  }
  arts_send_db_ownership_response(new_owner, cache->db_guid, buf->version,
                                  map_buf, map_size, buf->data, cache->db_size);
  arts_free(map_buf);
  arts_db_buf_release(&buf_h);
}

/* ===== OWNERSHIP_RESPONSE wire sender (EAGER + LAZY) ================
 * Carries the serialized last_sent_version map + buffer payload.  A self-send
 * (new_owner == this rank) constructs a contiguous buffer and dispatches the
 * handler inline. */
void arts_send_db_ownership_response(unsigned int new_owner_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     const void *map_buf, size_t map_size,
                                     const void *data, size_t data_size) {
  struct arts_msg_ownership_response_packet_s hdr;
  uint32_t entry_count = (map_buf != NULL && map_size >= sizeof(uint32_t) * 2)
                             ? ((const uint32_t *)map_buf)[0]
                             : 0u;
  uint64_t total =
      (uint64_t)sizeof(hdr) + (uint64_t)map_size + (uint64_t)data_size;
  arts_fill_packet_header(&hdr.header, total, MSG_DB_OWNERSHIP_RESPONSE);
  hdr.header.rank = arts_global_rank_id;
  hdr.db_guid = db_guid;
  hdr.version = version;
  hdr.map_entry_count = entry_count;
  hdr.pad = 0;
  if (new_owner_rank == arts_global_rank_id) {
    /* Self-transfer: construct a contiguous buffer and call handler inline. */
    void *buf = arts_malloc((size_t)total);
    memcpy(buf, &hdr, sizeof(hdr));
    if (map_buf != NULL && map_size > 0) {
      memcpy((char *)buf + sizeof(hdr), map_buf, map_size);
    }
    if (data != NULL && data_size > 0) {
      memcpy((char *)buf + sizeof(hdr) + map_size, data, data_size);
    }
    arts_handler_db_ownership_response(buf, (size_t)total);
    arts_free(buf);
    return;
  }
  if ((map_size > 0 || data_size > 0) && (map_buf != NULL || data != NULL)) {
    size_t payload_size = map_size + data_size;
    void *payload = arts_malloc(payload_size);
    if (map_buf != NULL && map_size > 0) {
      memcpy(payload, map_buf, map_size);
    }
    if (data != NULL && data_size > 0) {
      memcpy((char *)payload + map_size, data, data_size);
    }
    arts_transport_send_payload_async_free(
        (int)new_owner_rank, (char *)&hdr, sizeof(hdr), (char *)payload,
        /*offset=*/0, (uint64_t)payload_size, arts_free);
  } else {
    arts_transport_send_async((int)new_owner_rank, (char *)&hdr, sizeof(hdr));
  }
}

/* ===== Home-side ownership handlers (MRNEW; moved from handlers.c) =====
 * OWNERSHIP_REQUEST / RELEASE_OWNERSHIP exist only under MRNEW (MRMW routes
 * all acquires through GET_DATA / DATA_RESPONSE), so these handlers are
 * compiled only for MRNEW.  The home-directory machinery they touch
 * (pending_rw, invalidate_in_flight, rw_holder) is shared by both protocols;
 * the point where EAGER and LAZY diverge is delegated to per-protocol seams in
 * coherence/eager.c / coherence/lazy.c. */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_OWNERSHIP_REQUEST]): the OoO engine
 * has already acquired the home db_s for db_guid and pinned a ref across this
 * call, so there is no lookup / NULL-check / defer here.  cache is the FIRST
 * member of arts_db_s (offset 0), so the slot object the engine hands us IS the
 * cache.  The wire dispatcher decodes OWNERSHIP_REQUEST into the args struct
 * and routes through the engine via OOO_DB_OWNERSHIP_REQUEST; a missing home
 * db_s defers the args and re-issues this body once DB_CREATE installs and
 * drains.  (MRMW never enqueues this kind — coherence/mrmw.c provides a
 * no-op definition that satisfies the single g_ooo_table slot in the MRMW
 * build.) */
void arts_handler_db_ownership_request(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_ownership_request_s *a =
      (struct arts_ooo_args_db_ownership_request_s *)args_v;
  unsigned int requester = a->requester;

  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_home_lockreq_queue_push(&db->pending_rw, requester);

  /* Active-directory invariant: AT MOST ONE INVALIDATE_NOTICE in flight
   * to the current rw_holder per ownership-transfer round.  CAS 0->1
   * gates the dispatch — only the thread that flips the bit sends.
   * Concurrent OWNERSHIP_REQUESTs whose CAS loses simply piggyback on the
   * outstanding round; their requester is queued in pending_rw and is served by
   * the next CONFIRM-driven round.  The baton is held across the INVALIDATE →
   * owner→owner transfer → CONFIRM round-trip and cleared in the CONFIRM
   * handler once the queue drains.  Replaces the pre-fix `was_empty` heuristic
   * which over-sent on concurrent enqueues. */
  unsigned int iif_zero = 0;
  if (!atomic_compare_exchange_strong_explicit(
          &db->invalidate_in_flight, &iif_zero, 1u, memory_order_acq_rel,
          memory_order_acquire)) {
    return; /* another round is in flight; requester stays queued */
  }
  /* Baton won: the eager protocol INVALIDATEs the current rw_holder; the lazy
   * protocol pops the FIFO transfer target, publishes pending_install_owner,
   * and starts the invalidate round. */
  arts_db_start_ownership_round(cache, db, requester);
}

/* ===== Ownership wire senders (MRNEW; moved from coherence/senders.c) =====
 * OWNERSHIP_REQUEST / INVALIDATE_NOTICE exist only under MRNEW.  Ownership now
 * transfers owner→owner (arts_db_send_ownership_response); there is no
 * RELEASE_OWNERSHIP message.  Self-sends dispatch the matching handler inline.
 */

void arts_send_db_ownership_request(unsigned int home_rank,
                                    arts_guid_t db_guid) {
  struct arts_msg_ownership_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine exactly as the wire RX
     * dispatcher does — HIT runs the OWNERSHIP_REQUEST body inline, MISS defers
     * the args and replays once the home db_s is installed + drained.  (The
     * handler is now a pure (item, args) body; it no longer does its own
     * lookup-or-defer, so the inline shortcut must enter through
     * dispatch_or_defer.) */
    struct arts_ooo_args_db_ownership_request_s args = {
        .requester = p.header.rank,
        .db_guid = db_guid,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_OWNERSHIP_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_ownership_invalidate(unsigned int owner_rank,
                                       arts_guid_t db_guid,
                                       unsigned int new_owner_rank) {
  struct arts_msg_ownership_invalidate_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_INVALIDATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.new_owner_rank = new_owner_rank;
  memset(p.pad, 0, sizeof(p.pad));
  if (owner_rank == arts_global_rank_id) {
    /* Self-send: both timings now dispatch the INVALIDATE handler directly.
     *
     * The home publishes the invalidate target (rw_holder) only AFTER that
     * rank's cache install — the post-install CONFIRM owner-swap (true for BOTH
     * timings now) or the DB_CREATE on the creator — so an INVALIDATE always
     * targets an already-installed cache; the before-install reorder that once
     * forced eager through the OoO engine is gone.
     *
     * Direct (non-deferred) self-dispatch is also safe in the new_owner==home
     * self-CONFIRM chain: OWNERSHIP_RESPONSE installs a sentinel(+1)+guard(+1)
     * on writer_count, then self-sends CONFIRM, whose drain loop self-sends
     * THIS INVALIDATE inline.  Because the guard +1 is still present at that
     * point, this inline -1 cannot drive writer_count to the 0-edge
     * (writer_count >= 2 before it) — the commutative signed counter's unique
     * 0-crossing remains the RESPONSE handler's guard-removal -1, which then
     * ships exactly one owner→owner transfer.  Inline vs deferred only moves
     * WHICH actor observes the 0-edge; the guard guarantees it is still exactly
     * one.  Do not re-introduce a defer here. */
    struct arts_ooo_args_db_ownership_invalidate_s args = {
        .db_guid = db_guid,
        .new_owner_rank = new_owner_rank,
    };
    /* Pin the db_s for the handler's duration (cache is its FIRST member,
     * offset 0) — keeps it alive against a concurrent DESTROY.  Target is the
     * rw_holder, published only post-install, so it is provably present. */
    arts_shared_ptr_t db_h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
    assert(db != NULL); /* target is rw_holder, published post-install */
    arts_handler_db_ownership_invalidate(db, &args);
    arts_shared_release(&db_h);
    return;
  }
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

/* CONFIRM (new owner C → home A), both timings.  Home flips rw_holder +
 * advances the next round; LAZY additionally replies CONFIRM_ACK (its handler),
 * EAGER does not.  The handler reads pending_install_owner and ignores args, so
 * the self-send passes NULL. */
void arts_send_db_ownership_confirm(unsigned int home_rank, arts_guid_t db_guid,
                                    uint64_t version) {
  struct arts_msg_ownership_confirm_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_CONFIRM);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  if (home_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop.
     * HIT advances the transfer round on the ref-pinned home db_s; MISS (DB
     * destroyed) silently drops. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_ownership_confirm(db, NULL);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}
