/* SPDX-License-Identifier: Apache-2.0
 *
 * The migrating sentinel grant — the ownership plane shared by every arm whose
 * write right MOVES between owners rather than being arbitrated per acquire.
 *
 * The grant is a per-rank sentinel on writer_count: an ownership install adds
 * +1, and nothing withdraws it except an INVALIDATE.  A grant therefore
 * OUTLIVES the writer that opened it (sticky), and consecutive local writers
 * cost no wire traffic at all.  The home is a pure directory: it names the
 * holder (rw_holder), queues requesters (pending_rw, Vyukov MPSC), and admits
 * one transfer round at a time (invalidate_in_flight baton).  The payload
 * itself never routes through the home on a transfer — the INVALIDATE carries
 * the next owner, and whichever actor drives writer_count to exactly 0 (the
 * INVALIDATE handler, or the last release) ships owner -> owner directly.
 *
 * Nothing here is protocol-specific: what a release does BESIDES decrementing
 * the count (publish the payload to home, run an invalidation round, both, or
 * neither) belongs to the arm, not to the grant.  WRF_VAL is the one arm that
 * does not link this TU — it has no grant at all, keeping the canonical buffer
 * at home via a synchronous PUBLISH on every release.
 *
 * Write-policy-divergent steps are delegated to seams the arm defines
 * (arts_db_start_grant_round, and the GRANT_RESPONSE / CONFIRM
 * handlers, where WT drains immediately and WB gates on CONFIRM_ACK).
 * One ARTS_WRITE_POLICY_WB guard remains here: the INVALIDATE self-send is a
 * direct call under WB and an OoO defer under WT.
 */
#include <assert.h> /* WB INVALIDATE direct-call invariant assert */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h" /* arts_db_buf_acquire (invalidate xfer) */
#include "arts/counter/object_counter.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/directory.h" /* arts_home_grantreq_queue_push */
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C self-send) */
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/schedfuzz.h"
#include "arts/system/threads.h"     /* arts_global_rank_id */
#include "arts/transport/net.h"   /* arts_transport_send_async */
#include "arts/transport/protocol.h" /* arts_fill_packet_header, MSG_* */
#include "arts/utils/malloc.h" /* arts_malloc / arts_free (transfer ship) */
#include "arts/counter/Preamble.h"

/* ===== Case 2/6: RW local fast path ================================
 * Shared by the WT and WB arts_handler_db_acquire bodies
 * (coherence/val/wt.c / coherence/val/wb.c).  CAS-increments writer_count "if
 * runnable"; on success writes dep->ptr (acquire_local) and returns true;
 * returns false when this rank may not take a write turn, so the caller falls
 * through to arts_db_acquire_remote_rw. */
bool arts_db_acquire_rw_local_fast(struct arts_db_cache_s *cache,
                                   arts_edt_dep_t *dep) {
  /* The CAS below is the WHOLE test, not a commit behind one.  writer_count
   * carries both facts a fresh write turn needs — a positive count IS "this
   * rank holds the grant", and ARTS_GRANT_UNCONFIRMED makes an install whose
   * directory flip the home has not yet published read back negative — so a
   * turn is taken only if, AT THE INSTANT THE CAS COMMITS, the word is a
   * positive count.  Testing the two facts as separate loads cannot achieve
   * this: a grant lost and re-granted (unconfirmed) between them is
   * indistinguishable from one never lost, and the turn lands on an install
   * that may not run.  Here that interleaving changes the word, the CAS fails,
   * and the loop re-reads.  The predicate is a property of the VALUE, not of
   * its identity, so the value's reuse (ABA) is harmless.
   *
   * `(int)wc <= 0` therefore rejects two distinct states: 0 (no grant — never
   * held, or invalidated) and negative (held but unconfirmed, where a write
   * would be observable while the directory still names the previous owner and
   * a reader is redirected to that rank's retained, now stale copy).  Both go
   * remote-RW, which parks; the in-flight round's drain wakes them. */
  while (1) {
    unsigned int wc = arts_atomic_read(&cache->writer_count);
    if ((int)wc <= 0) {
      return false; /* no grant, or one not yet confirmed — go remote-RW. */
    }
    if (arts_atomic_cswap(&cache->writer_count, wc, wc + 1) == wc) {
      /* writer_count bumped; release_rw decrements the matching bump.
       * acquire_local may only come back NULL for a zero-sized (sentinel)
       * block: holding the grant on a SIZED block means holding its buffer,
       * and a buffer installed but never published is present at version 0,
       * not absent. */
      /* A write turn taken on a grant this rank already held: the round trip
       * the sticky grant removed. */
      INCREMENT_NUM_GRANT_LOCAL_REUSE_BY(1);
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
  INCREMENT_NUM_DB_ACQUIRE_REMOTE_BY(1);
  arts_object_acquire(true);
  arts_pending_rw_queue_push(&cache->pending_rw, w);

  /* Kick GRANT_REQUEST if no one else has — GRANT is what eventually
   * triggers our drain in FIFO order.  One request per round, so this counts
   * grants that had to move rather than waiters. */
  if (arts_atomic_cswap(&cache->grant_req_in_flight, 0, 1) == 0) {
    INCREMENT_NUM_GRANT_MIGRATE_BY(1);
    arts_send_db_grant_request(cache);
  }
  return ARTS_DB_ACQUIRE_PARK;
}

/* The arts_handler_db_acquire 8-case body lives per-write-policy in
 * coherence/val/wt.c and coherence/val/wb.c — the two builds differ only on the
 * RO-has-local-data predicate (WT: is_home||is_owner; WB: is_owner), which
 * the C-preprocessor seam forbids in a shared TU.  Both call the shared
 * arts_db_acquire_rw_local_fast / arts_db_acquire_remote_rw above and
 * arts_db_acquire_remote_ro (coherence/coherence.c).
 */

/* ===== GRANT drain (called from coherence/val/wt.c and coherence/val/wb.c) === */

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
   * A waiter woken here is being handed the write turn, so it must find
   * storage of the DB's declared size; the one buffer-less case is a
   * zero-sized (sentinel) block, whose defined value is the NULL pointer
   * mark_edt_ready_by_guid then stamps (still accounting the dep). */
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

/* ===== Shared owner→owner transfer ship (WT + WB) =============
 * Ship the current buffer to cache->incoming_new_owner via the
 * GRANT_RESPONSE wire, re-arming incoming_new_owner to the sentinel BEFORE
 * the send (a self-transfer dispatches the new owner's install inline, which
 * can recursively republish incoming_new_owner for the next round; clearing it
 * up front leaves that fresh publish intact).  WB serializes its owner-side
 * dedup map; WT has no owner-side map (cached_version == NULL) and emits
 * an empty map (count=0): the receiver unconditionally reconstructs map_size >=
 * 8, so an omitted header would underflow data_size and corrupt the install. */
void arts_db_send_grant_response(struct arts_db_cache_s *cache) {
  unsigned int new_owner = cache->incoming_new_owner;
  struct arts_rdzv_landing_s rdzv = cache->incoming_new_owner_rdzv;
  cache->incoming_new_owner =
      ARTS_NO_PENDING_OWNER; /* re-arm before send */
  cache->incoming_new_owner_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);

  /* Serialize the owner-side dedup map — small, and it rides INLINE in the
   * response packet (the control plane); only the buffer payload moves
   * one-sided.  An empty count-header is emitted when there is no map: the
   * receiver unconditionally parses map_size >= 8, so omitting it would
   * corrupt the parse. */
  size_t map_max =
      (sizeof(uint32_t) * 2) + ((size_t)arts_global_rank_count *
                                sizeof(struct arts_msg_rank_version_pair_s));
  void *map_buf = arts_malloc(map_max);
  size_t map_size;
#ifdef ARTS_WRITE_POLICY_WB
  /* Self-credit before the map migrates: this (ex-)owner keeps its buffer
   * after the ship, and its self-produced version was never "sent" by anyone —
   * without this entry the inherited ledger would re-ship a payload the
   * ex-owner already holds on its next read.  Only advance an EXISTING map
   * (slot CAS — safe against the redirect handler); when no map exists yet the
   * self entry is appended to the serialized bytes below instead, because
   * creating the map here would race the network thread's OWNER create. */
  if (buf != NULL && cache->cached_version != NULL) {
    arts_rank_u64_map_advance(cache->cached_version, arts_global_rank_id,
                              buf->version);
  }
#endif
  if (buf != NULL && cache->cached_version != NULL) {
    map_size = arts_rank_u64_map_serialize(cache->cached_version, map_buf);
  } else {
    uint32_t *mp = (uint32_t *)map_buf;
    mp[0] = 0u;
    mp[1] = 0u;
    map_size = sizeof(uint32_t) * 2;
#ifdef ARTS_WRITE_POLICY_WB
    if (buf != NULL) {
      /* No owner-side map (never served a reader): emit a one-entry map
       * carrying only the self-credit. */
      struct arts_msg_rank_version_pair_s *self_pair =
          (struct arts_msg_rank_version_pair_s *)((char *)map_buf +
                                                  (sizeof(uint32_t) * 2));
      self_pair->rank = arts_global_rank_id;
      self_pair->pad = 0;
      self_pair->version = buf->version;
      mp[0] = 1u;
      map_size += sizeof(*self_pair);
    }
#endif
  }

  uint64_t version = (buf != NULL) ? buf->version : 0;
  /* data_size carries the DB's size ALWAYS (descriptor state — known even
   * when no buffer exists yet): a hinted first-touch requester may have
   * skipped the size CTS, and a data-less grant is its only chance to learn
   * the exact size before its release publishes one.  Payload presence is
   * signaled by rdzv_txid (wire) / trailing bytes (self), never by
   * data_size. */
  uint64_t payload_size = (buf != NULL) ? cache->db_size : 0;
  struct arts_msg_grant_response_packet_s hdr;
  arts_fill_packet_header(&hdr.header, sizeof(hdr) + map_size,
                          MSG_DB_GRANT_RESPONSE);
  hdr.header.rank = arts_global_rank_id;
  hdr.db_guid = cache->db_guid;
  hdr.version = version;
  hdr.map_entry_count = (map_size >= sizeof(uint32_t) * 2)
                            ? ((const uint32_t *)map_buf)[0]
                            : 0u;
  hdr.pad = 0;

  if (new_owner == arts_global_rank_id) {
    /* Self-transfer: no wire, no RDMA.  Ex-owner and new owner are the same
     * cache, so the landing this rank advertised in its own request carries
     * nothing it does not already have — unless there is nothing to have.
     * With a buffer present the bytes are already in place and the landing
     * goes back to the pool; with none, this rank is about to take the write
     * turn on a sized block, so the landing BECOMES its storage (nothing else
     * on this path would give it any).  The handler then runs inline on a
     * contiguous same-rank buffer (map + data trailing). */
    if (rdzv.cookie != 0) {
      struct arts_db_buffer_s *landing =
          (struct arts_db_buffer_s *)(uintptr_t)rdzv.cookie;
      if (buf != NULL) {
        arts_db_buf_landing_recycle(cache, landing);
      } else {
        (void)arts_db_buf_adopt_landing(cache, version, landing,
                                        cache->db_size);
      }
    }
    uint64_t total = sizeof(hdr) + map_size + payload_size;
    hdr.header.size = total;
    hdr.data_size = cache->db_size;
    hdr.rdzv_txid = 0;
    hdr.rdzv_cookie = 0;
    char *pkt = (char *)arts_malloc((size_t)total);
    memcpy(pkt, &hdr, sizeof(hdr));
    memcpy(pkt + sizeof(hdr), map_buf, map_size);
    if (buf != NULL && payload_size > 0) {
      memcpy(pkt + sizeof(hdr) + map_size, buf->data, (size_t)payload_size);
    }
    arts_free(map_buf);
    if (buf != NULL) {
      arts_db_buf_release(&buf_h);
    }
    arts_handler_db_grant_response(pkt, (size_t)total);
    arts_free(pkt);
    return;
  }

  if (buf != NULL && payload_size > 0 && rdzv.txid != 0) {
    /* One-sided ship: PUT straight from the live buffer into the new owner's
     * landing — zero copy at the source.  The strong buffer ref transfers to
     * the PUT's local completion, keeping the bytes valid until the fabric no
     * longer reads them (the protocol additionally keeps this cache's buffer
     * slot untouched until the new owner CONFIRMs, but the ref makes the
     * lifetime explicit rather than assumed). */
    hdr.data_size = cache->db_size;
    hdr.rdzv_txid = rdzv.txid;
    hdr.rdzv_cookie = rdzv.cookie;
    arts_net_put_payload((int)new_owner, rdzv.addr, rdzv.key, rdzv.txid,
                         buf->data, payload_size, arts_db_buf_ref_release_cb,
                         (void *)buf_h);
    buf_h = NULL; /* transferred to the PUT completion */
  } else {
    /* Nothing to ship: this ex-owner holds no buffer, or the block is
     * zero-sized.  Echo the advertised landing (if any) back — for a sized
     * block it is the new owner's only storage, and the receiving handler
     * adopts it rather than returning it to the pool; data_size tells that
     * handler the exact size either way. */
    hdr.data_size = cache->db_size;
    hdr.rdzv_txid = 0;
    hdr.rdzv_cookie = rdzv.cookie;
    if (buf != NULL) {
      arts_db_buf_release(&buf_h);
    }
  }
  arts_transport_send_payload_async_free((int)new_owner, (char *)&hdr,
                                         sizeof(hdr), (char *)map_buf,
                                         /*offset=*/0, (uint64_t)map_size,
                                         arts_free);
}

/* ===== Home-side ownership handlers (VAL; moved from handlers.c) =====
 * GRANT_REQUEST exists only under VAL (WRF_VAL routes
 * all acquires through SNAPSHOT_REQUEST / SNAPSHOT_RESPONSE), so these handlers are
 * compiled only for VAL.  The home-directory machinery they touch
 * (pending_rw, invalidate_in_flight, rw_holder) is shared by both write policies;
 * the point where WT and WB diverge is delegated to per-write-policy seams in
 * coherence/val/wt.c / coherence/val/wb.c. */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_GRANT_REQUEST]): the OoO engine
 * has already acquired the home db_s for db_guid and pinned a ref across this
 * call, so there is no lookup / NULL-check / defer here.  cache is the FIRST
 * member of arts_db_s (offset 0), so the slot object the engine hands us IS the
 * cache.  The wire dispatcher decodes GRANT_REQUEST into the args struct
 * and routes through the engine via OOO_DB_GRANT_REQUEST; a missing home
 * db_s defers the args and re-issues this body once DB_CREATE installs and
 * drains.  (WRF_VAL never enqueues this kind — coherence/wrf_val.c provides a
 * no-op definition that satisfies the single g_ooo_table slot in the WRF_VAL
 * build.) */
void arts_handler_db_grant_request(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_grant_request_s *a =
      (struct arts_ooo_args_db_grant_request_s *)args_v;
  unsigned int requester = a->requester;

  struct arts_db_s *db = arts_db_of_cache(cache);
  if (a->rdzv.txid == 0 && requester != arts_global_rank_id &&
      cache->db_size > 0 && arts_global_rank_count > 1) {
    /* First-touch request without a landing: the requester did not know
     * db_size.  Answer with the size (CTS) and do NOT enqueue — home only
     * queues requests that carry a landing (or target a sentinel DB, whose
     * transfers are data-less).  The requester re-issues with a landing.
     * The home's OWN landing-less request is not a size probe: it wants the
     * permission without payload and queues like any other. */
    arts_send_db_grant_cts(requester, cache->db_guid, cache->db_size);
    return;
  }
  arts_home_grantreq_queue_push(&db->pending_rw, requester, &a->rdzv);
  arts_sched_fuzz_point(); /* widen the push<->baton-CAS window */

  /* Active-directory invariant: AT MOST ONE GRANT_INVALIDATE in flight
   * to the current rw_holder per ownership-transfer round.  CAS 0->1
   * gates the dispatch — only the thread that flips the bit sends.
   * Concurrent GRANT_REQUESTs whose CAS loses simply piggyback on the
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
  /* Baton won: the WT write policy INVALIDATEs the current rw_holder; the WB
   * write policy pops the FIFO transfer target, publishes pending_install_owner,
   * and starts the invalidate round. */
  arts_db_start_grant_round(cache, db, requester);
}

/* ===== Ownership wire senders (VAL; moved from coherence/senders.c) =====
 * GRANT_REQUEST / GRANT_INVALIDATE exist only under VAL.  Ownership
 * transfers owner→owner (arts_db_send_grant_response); there is no
 * separate release message.  Self-sends dispatch the matching handler inline.
 */

void arts_send_db_grant_request(struct arts_db_cache_s *cache) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  /* Advertise a fresh transfer landing sized by the exact size when known,
   * else by the GUID's szhint bound — a first touch usually carries a
   * landing already, and home's GRANT_CTS round survives only as the
   * sentinel fallback (its handler re-enters this sender with
   * cache->db_size learned).  The rendezvous plane exists only when a peer
   * could PUT (multi-rank run): a single-rank run has no fabric, every
   * transfer is a same-rank inline dispatch, and the registered pool
   * carries no MRs to advertise. */
  struct arts_rdzv_landing_s rdzv = {0, 0, 0, 0};
  /* Under write-back the landing is advertised unconditionally, even though a
   * transfer of a DB nobody has published yet carries no bytes: whether the
   * current owner HAS bytes is remote state, so the only way to know before
   * asking is to ask — a round trip added to every acquire to save one pooled
   * buffer on the rare pre-publication path.  A transfer that ships no data
   * echoes the advertisement back, and the requester keeps it as this block's
   * storage (it is entitled to a pointer of the declared size whether or not
   * anyone has written the block) or returns it to the pool when it already
   * holds a buffer. */
#ifdef ARTS_WRITE_POLICY_WT
  /* The write-through home already holds the newest bytes (every release
   * published them), so its own request needs the PERMISSION only: no
   * landing, and the ex-owner's transfer takes the data-less arm.  The
   * stable home buffer is never re-installed through the grant plane. */
  bool wants_payload = (home_rank != arts_global_rank_id);
#else
  bool wants_payload = true;
#endif
  uint64_t fetch_size = arts_db_first_fetch_size(cache);
  if (wants_payload && fetch_size > 0 && arts_global_rank_count > 1) {
    (void)arts_db_buf_landing_alloc(cache, fetch_size, &rdzv);
  } else if (!wants_payload) {
    INCREMENT_NUM_GRANT_HOME_DATALESS_BY(1);
  }
  struct arts_msg_grant_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_GRANT_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.rdzv.addr = rdzv.addr;
  p.rdzv.key = rdzv.key;
  p.rdzv.txid = rdzv.txid;
  p.rdzv.cookie = rdzv.cookie;
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine exactly as the wire RX
     * dispatcher does — HIT runs the GRANT_REQUEST body inline, MISS defers
     * the args and replays once the home db_s is installed + drained.  (The
     * handler is now a pure (item, args) body; it no longer does its own
     * lookup-or-defer, so the inline shortcut must enter through
     * dispatch_or_defer.) */
    struct arts_ooo_args_db_grant_request_s args = {
        .requester = p.header.rank,
        .db_guid = db_guid,
        .rdzv = rdzv,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_GRANT_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* GRANT_CTS sender (home → first-touch requester) + requester-side body.
 * The requester learns db_size and re-issues the in-flight request with a
 * landing; the coalescing flag stays held (same round continuing). */
void arts_send_db_grant_cts(unsigned int requester_rank,
                                arts_guid_t db_guid, uint64_t db_size) {
  INCREMENT_NUM_GRANT_SIZE_CTS_BY(1);
  struct arts_msg_grant_cts_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_GRANT_CTS);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.db_size = db_size;
  if (requester_rank == arts_global_rank_id) {
    /* Self-send cannot occur in a consistent state (home == requester shares
     * ONE cache, so a size-unknown request implies a size-unknown home), but
     * mirror the Cat-C lookup-acquire-or-drop for uniformity. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_grant_cts(db, &p);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

void arts_handler_db_grant_cts(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_msg_grant_cts_packet_s *p =
      (struct arts_msg_grant_cts_packet_s *)args_v;
  if (cache->db_size == 0) {
    cache->db_size = p->db_size;
  }
  arts_send_db_grant_request(cache);
}

void arts_send_db_grant_invalidate(
    unsigned int owner_rank, arts_guid_t db_guid, unsigned int new_owner_rank,
    const struct arts_rdzv_landing_s *new_owner_rdzv) {
  struct arts_rdzv_landing_s rdzv =
      (new_owner_rdzv != NULL) ? *new_owner_rdzv
                               : (struct arts_rdzv_landing_s){0, 0, 0, 0};
  struct arts_msg_grant_invalidate_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_GRANT_INVALIDATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.new_owner_rank = new_owner_rank;
  memset(p.pad, 0, sizeof(p.pad));
  p.new_owner_rdzv.addr = rdzv.addr;
  p.new_owner_rdzv.key = rdzv.key;
  p.new_owner_rdzv.txid = rdzv.txid;
  p.new_owner_rdzv.cookie = rdzv.cookie;
  if (owner_rank == arts_global_rank_id) {
    /* Self-send: both placements now dispatch the INVALIDATE handler directly.
     *
     * The home publishes the invalidate target (rw_holder) only AFTER that
     * rank's cache install — the post-install CONFIRM owner-swap (true for BOTH
     * write policies now) or the DB_CREATE on the creator — so an INVALIDATE always
     * targets an already-installed cache; the before-install reorder that once
     * forced WT through the OoO engine is gone.
     *
     * Direct (non-deferred) self-dispatch is also safe in the new_owner==home
     * self-CONFIRM chain: GRANT_RESPONSE installs a sentinel(+1)+guard(+1)
     * on writer_count, then self-sends CONFIRM, whose drain loop self-sends
     * THIS INVALIDATE inline.  Because the guard +1 is still present at that
     * point, this inline -1 cannot drive writer_count to the 0-edge
     * (writer_count >= 2 before it) — the commutative signed counter's unique
     * 0-crossing remains the RESPONSE handler's guard-removal -1, which then
     * ships exactly one owner→owner transfer.  Inline vs deferred only moves
     * WHICH actor observes the 0-edge; the guard guarantees it is still exactly
     * one.  Do not re-introduce a defer here. */
    struct arts_ooo_args_db_grant_invalidate_s args = {
        .db_guid = db_guid,
        .new_owner_rank = new_owner_rank,
        .new_owner_rdzv = rdzv,
    };
    /* Pin the db_s for the handler's duration (cache is its FIRST member,
     * offset 0) — keeps it alive against a concurrent DESTROY.  Target is the
     * rw_holder, published only post-install, so it is provably present. */
    arts_shared_ptr_t db_h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
    assert(db != NULL); /* target is rw_holder, published post-install */
    arts_handler_db_grant_invalidate(db, &args);
    arts_shared_release(&db_h);
    return;
  }
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

/* CONFIRM (new owner C → home A), both placements.  Home flips rw_holder +
 * advances the next round; OWNER additionally replies CONFIRM_ACK (its handler),
 * HOME does not.  The handler reads pending_install_owner and ignores args, so
 * the self-send passes NULL. */
void arts_send_db_grant_confirm(unsigned int home_rank, arts_guid_t db_guid,
                                    uint64_t version) {
  struct arts_msg_grant_confirm_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_GRANT_CONFIRM);
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
      arts_handler_db_grant_confirm(db, NULL);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}
