/* SPDX-License-Identifier: Apache-2.0
 *
 * MRSW shared ownership machinery (compiled for both timing variants of the
 * MRSW protocol; MRMW/MRNEW do not compile it).
 *
 * Compiled only when ARTS_COHERENCE_PROTOCOL is MRSW (selected in
 * libs/src/core/CMakeLists.txt).  MRSW is rank-granular on the MRNEW engine:
 * the home-directory / single-owner ownership machinery here (home FIFO,
 * OWNERSHIP_REQUEST/RESPONSE/INVALIDATE/CONFIRM(+ACK), incoming_new_owner,
 * ownership_req_in_flight, invalidate_in_flight baton, owner→owner transfer
 * ship + wire senders) is byte-for-byte MRNEW.  The ONLY MRSW deltas are the
 * local active-writer cap (a single writer_count {2,1,0} = sentinel+token) and
 * its consequences:
 *   - local RW acquire claims the TOKEN (CAS 1->2) instead of joining a count;
 *   - the cache.pending_rw RW-waiter queue is consumed ONE waiter at a time
 *     (pop-one), the token (not a per-waiter bump) accounting the single active
 *     writer;
 *   - release pops-then-conditionally-subtracts with a cheap LOCAL Dekker
 *     re-check (counter+queue ⇒ MCS-style double-check).
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
#include "arts/utils/atomics.h"      /* arts_atomic_* */
#include "arts/utils/malloc.h" /* arts_malloc / arts_free (transfer ship) */

/* ===== single-consumer invariant for cache.pending_rw =============
 * The Vyukov MPSC FIFO is multi-producer / SINGLE-consumer.  MRSW upholds the
 * single-consumer invariant WITHOUT a lock: the ONLY consumer that pops-and-
 * delivers is the token holder (run_one / release hand-off), which is unique by
 * the writer-token serialization.  Destroy does not pop or wake this FIFO;
 * destroying a DB with a still-parked waiter is OCR undefined behavior, so the
 * refcount-0 cache destructor merely FREES the leftover nodes
 * (arts_db_rw_waiter_queue_destroy) without delivering them.  The raw queue
 * helpers (arts_db_rw_waiter_queue_pop / _peek_empty) are therefore called
 * directly. */

/* ===== pop-one delivery (token already accounts the one active writer) =====
 * Pop exactly ONE waiter from cache.pending_rw and deliver it.  Unlike MRNEW's
 * rw_drain_cb, there is NO per-waiter writer_count bump: the single token (the
 * +1 above the sentinel) accounts the one active writer for the whole epoch.
 * Used by the idle-owner token claim, the GRANT drain, and the release token
 * hand-off.  Sentinel DBs (db_size==0) have cache->buffer==NULL by design;
 * mark_edt_ready_by_guid handles that cleanly (depv[slot].ptr=NULL, still
 * accounts the dep). */
bool arts_db_mrsw_run_one(struct arts_db_cache_s *cache) {
  arts_guid_t edt_guid;
  unsigned int slot;
  if (!arts_db_rw_waiter_queue_pop(&cache->pending_rw, &edt_guid, &slot)) {
    return false;
  }
  /* Advance the RW cursor first (position-idempotent; never schedules), THEN
   * deliver data (may schedule + let another worker run/free the EDT). */
  mark_edt_secured_by_guid(edt_guid, slot);
  mark_edt_ready_by_guid(edt_guid, slot);
  return true;
}

/* ===== Kick a remote-RW round (OWNERSHIP_REQUEST) =================
 * The waiter is ALREADY pushed on cache.pending_rw by the single push site
 * (arts_db_acquire_rw_local_fast).  Coalesce: only the actor that CASes
 * ownership_req_in_flight false->true sends OWNERSHIP_REQUEST; same-node RW
 * EDTs piggyback on the in-flight one and are picked up by the GRANT (eager) /
 * CONFIRM_ACK (lazy) drain in FIFO order.
 *
 * No destroy_state precheck: per spec 4.11, handle_destroy_req NULL-stores
 * route_item->data BEFORE flipping destroy_state, so route_table_lookup_db
 * already misses and the caller's OoO defer handles "DB destroyed".  If a
 * destroy races in at this point, the waiter on cache.pending_rw is abandoned
 * (destroying a DB while an EDT holds a pending dependence is OCR undefined
 * behavior). */
static void arts_db_kick_remote_rw(struct arts_db_cache_s *cache) {
  if (arts_atomic_cswap(&cache->ownership_req_in_flight, 0, 1) == 0) {
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_ownership_request(home_rank, cache->db_guid);
  }
}

/* ===== Case 2/6: RW local fast path (single push site) =============
 * Whole RW acquire path for both timings (the handler RW branch calls ONLY
 * this — no separate remote-RW fall-through, so the waiter is pushed exactly
 * once).  MRSW caps the local active writer at one (the token).  Push the
 * waiter ONCE, then decide by state:
 *   wc<=0:  not owner — kick OWNERSHIP_REQUEST (coalesced); the round's drain
 *           pops me.
 *   wc>=2:  an active writer is in charge — it pops me on its release.
 *   invalidated-drain (incoming_new_owner != NONE): the drain pops me.
 *   wc==1 idle owner: CAS 1->2 claims the token + runs the FIFO head.
 * Always returns true (the waiter is queued; the run path delivers the dep).
 * Does NOT write dep->ptr. */
bool arts_db_acquire_rw_local_fast(struct arts_db_cache_s *cache,
                                   arts_edt_dep_t *dep, arts_guid_t edt_guid,
                                   unsigned int slot) {
  (void)dep;
  arts_db_rw_waiter_queue_push(&cache->pending_rw, edt_guid, slot);
  while (1) {
    unsigned int wc = arts_atomic_read(&cache->writer_count);
    if ((int)wc <= 0) {
      /* Not owner → kick a remote round; the drain pops my queued waiter. */
      arts_db_kick_remote_rw(cache);
      return true;
    }
    if (wc >= 2) {
      return true; /* active writer present → it pops me on release. */
    }
    if (cache->incoming_new_owner != ARTS_LAZY_NO_PENDING_OWNER) {
      return true; /* invalidated-drain → drain pops me. */
    }
    if (arts_atomic_cswap(&cache->writer_count, 1u, 2u) == 1u) {
      /* idle owner: claimed the token, run the FIFO head (mine or an earlier
       * pushed waiter — FIFO order). */
      arts_db_mrsw_run_one(cache);
      return true;
    }
    /* CAS lost (state moved 1->2 or 1->0): re-read, re-decide. */
  }
}

/* ===== Case 4/8: remote-RW path ==================================== */

/* Retained for the non-owner acquire seam used by the per-timing handlers when
 * this rank is provably NOT the owner (the RO-gated lazy path).  The waiter is
 * pushed here and a remote round kicked — symmetric with the local-fast
 * not-owner branch (which is the common case). */
arts_db_acquire_result_t
arts_db_acquire_remote_rw(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
                          unsigned int slot) {
  arts_db_rw_waiter_queue_push(&cache->pending_rw, edt_guid, slot);
  arts_db_kick_remote_rw(cache);
  return ARTS_DB_ACQUIRE_PARK;
}

/* The arts_handler_db_acquire 8-case body lives per-protocol in
 * coherence/mrsw/eager.c and coherence/mrsw/lazy.c — the two builds differ only
 * on the RO-has-local-data predicate (eager: is_home||is_owner; lazy:
 * is_owner), which the C-preprocessor seam forbids in a shared TU.  Both call
 * the shared arts_db_acquire_rw_local_fast / arts_db_acquire_remote_rw above
 * and arts_db_acquire_remote_ro (coherence/coherence.c).
 */

/* ===== GRANT drain (called from coherence/mrsw/eager.c and lazy.c) ====
 * MRSW pops EXACTLY ONE waiter (the single-writer cap): the install was 0->2
 * (sentinel + token), so the token already accounts the one active writer; run
 * the one waiter the token is for. */
void arts_db_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                          uint64_t version, bool has_next) {
  (void)version;
  (void)has_next;
  /* Pop exactly one waiter and run it.  If the queue is EMPTY on arrival (the
   * waiter that motivated this round's OWNERSHIP_REQUEST was already served by
   * an earlier grant / token hand-off, or a re-kick double-requested), the
   * install's token has NO writer to account it — it would be orphaned, holding
   * writer_count at 2 forever and stalling the transfer chain.  Drop the orphan
   * token via the normal release path so this idle grant either ships onward (a
   * transfer is pending) or quiesces as an idle owner. */
  if (!arts_db_mrsw_run_one(cache)) {
    arts_db_release_rw_local(cache);
  }
}

/* ===== local RW release (pop-then-conditional-sub + LOCAL Dekker recheck) ==
 * Called from the per-timing arts_db_release_rw after the writeback (eager) /
 * version bump.  If the queue has a waiter, hand it the token (NO sub — token
 * stays, no home round-trip: bulk locality).  Otherwise withdraw the token
 * (2->1 idle | 1->0 invalidated), then a cheap LOCAL Dekker re-check catches an
 * acquirer that raced our pop-empty/sub: if a waiter is now visible, reclaim
 * the token (purely local) and run it; otherwise on the true 0-edge with a
 * transfer target pending, ship the owner→owner transfer.
 *
 * Why the LOCAL re-check is mandatory (counter + queue ⇒ Dekker): if a fresh
 * acquirer pushes between our pop-empty and the sub, it reads the pre-sub
 * positive count and returns "active writer present" expecting US to pop it —
 * without the re-check it would strand.  This re-check is purely local (one
 * peek load + one reclaim CAS) — no home round-trip, no MPSC rescan. */
void arts_db_release_rw_local(struct arts_db_cache_s *cache) {
  arts_guid_t g;
  unsigned int slot;
  if (arts_db_rw_waiter_queue_pop(&cache->pending_rw, &g, &slot)) {
    /* Hand the token to the next waiter — NO writer_count sub (locality). */
    mark_edt_secured_by_guid(g, slot);
    mark_edt_ready_by_guid(g, slot);
    return;
  }
  /* Empty pop: withdraw the token (2->1 idle | 1->0 invalidated). */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1);

  if (rest > 0) {
    /* rest == 1: we still hold the SENTINEL (idle owner, incoming == NONE);
     * ownership stays here.  Cheap LOCAL Dekker re-check: an acquirer that
     * pushed between our pop-empty and the sub read the pre-sub positive count
     * and returned "active writer present" expecting US to pop it — reclaim the
     * token (1->2, purely local) and run it.  CAS lost → another actor (a fresh
     * idle-owner claim) took the token; it runs the waiter. */
    if (!arts_db_rw_waiter_queue_peek_empty(&cache->pending_rw)) {
      if (arts_atomic_cswap(&cache->writer_count, (unsigned)rest,
                            (unsigned)rest + 1) == (unsigned)rest) {
        arts_db_mrsw_run_one(cache);
      }
    }
    return;
  }

  /* rest == 0: the SENTINEL was already withdrawn by this round's INVALIDATE
   * (which published incoming_new_owner), so ownership is LEAVING this rank. We
   * are the unique actor on the true 0-edge — ship the owner→owner transfer. Do
   * NOT reclaim on a raced push here: resurrecting the count would lose the
   * transfer (the next owner would be stranded).  A waiter that raced in reads
   * writer_count == 0 in its own acquire and re-requests; a waiter that already
   * returned "parked" expecting a local pop is re-kicked by the ship helper's
   * stranded-waiter re-request tail. */
  if (cache->incoming_new_owner != ARTS_LAZY_NO_PENDING_OWNER) {
    arts_db_send_ownership_response(cache);
  }
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
  } else {
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
                                    map_buf, map_size, buf->data,
                                    cache->db_size);
    arts_free(map_buf);
    arts_db_buf_release(&buf_h);
  }
  /* Stranded-waiter re-request tail: ownership has left this rank to new_owner.
   * A CROSS-RANK transfer cannot pop a local waiter that raced into pending_rw
   * after we crossed the 0-edge (it returned "parked" expecting a local pop, or
   * pushed during the ship), so re-kick an OWNERSHIP_REQUEST when the queue is
   * non-empty — a future grant pops it.  Skip on a self-transfer (new_owner ==
   * self): the inline install dispatched above already runs this rank's
   * waiters, and re-kicking would double-serve. */
  if (new_owner != arts_global_rank_id &&
      !arts_db_rw_waiter_queue_peek_empty(&cache->pending_rw)) {
    arts_db_kick_remote_rw(cache);
  }
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

/* ===== Home-side ownership handlers (MRSW; rank-granular like MRNEW) =====
 * OWNERSHIP_REQUEST exists only under the ownership protocols (MRMW routes all
 * acquires through GET_DATA / DATA_RESPONSE).  The home-directory machinery
 * (pending_rw FIFO, invalidate_in_flight, rw_holder) is shared by both timings;
 * the point where EAGER and LAZY diverge is delegated to per-timing seams in
 * coherence/mrsw/eager.c / coherence/mrsw/lazy.c. */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_OWNERSHIP_REQUEST]): the OoO engine
 * has already acquired the home db_s for db_guid and pinned a ref across this
 * call, so there is no lookup / NULL-check / defer here.  cache is the FIRST
 * member of arts_db_s (offset 0), so the slot object the engine hands us IS the
 * cache. */
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
   * outstanding round; their requester is queued in pending_rw and served by
   * the next CONFIRM-driven round.  The baton is held across the INVALIDATE →
   * owner→owner transfer → CONFIRM round-trip and cleared in the CONFIRM
   * handler once the queue drains. */
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

/* ===== Ownership wire senders (MRSW; rank-granular like MRNEW) =====
 * OWNERSHIP_REQUEST / INVALIDATE_NOTICE exist only under the ownership
 * protocols.  Ownership transfers owner→owner
 * (arts_db_send_ownership_response); there is no RELEASE_OWNERSHIP message.
 * Self-sends dispatch the matching handler inline. */

void arts_send_db_ownership_request(unsigned int home_rank,
                                    arts_guid_t db_guid) {
  struct arts_msg_ownership_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine exactly as the wire RX
     * dispatcher does — HIT runs the OWNERSHIP_REQUEST body inline, MISS defers
     * the args and replays once the home db_s is installed + drained. */
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
    /* Self-send: both timings dispatch the INVALIDATE handler directly.  The
     * home publishes the invalidate target (rw_holder) only AFTER that rank's
     * cache install (the post-install CONFIRM owner-swap, both timings, or the
     * DB_CREATE on the creator), so an INVALIDATE always targets an
     * already-installed cache. */
    struct arts_ooo_args_db_ownership_invalidate_s args = {
        .db_guid = db_guid,
        .new_owner_rank = new_owner_rank,
    };
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
