/* SPDX-License-Identifier: Apache-2.0
 *
 * LAZY timing translation unit for MRSW: defines the LAZY-specific
 * arts_handler_db_* / arts_db_* bodies directly (CMake links exactly this TU
 * for an MRSW+LAZY build) plus the LAZY-only wire handlers/senders.
 * Compiled only for ARTS_COHERENCE_PROTOCOL=MRSW with
 * ARTS_PROTOCOL_TIMING=LAZY (selected in libs/src/core/CMakeLists.txt).
 * Contains NO protocol/timing preprocessor logic.
 *
 * Mirrors coherence/mrnew/lazy.c; the MRSW deltas are the single-writer cap:
 * the local RW acquire claims the TOKEN (arts_db_acquire_rw_local_fast pushes a
 * waiter + returns without writing dep->ptr), the TRANSFER install jumps
 * writer_count 0->2 (sentinel + token — no transient drain guard; the token IS
 * the running writer's account), the CONFIRM_ACK drains ONE waiter (the token's
 * writer) and applies the piggybacked sentinel withdrawal but does NOT remove a
 * guard, and release routes through arts_db_release_rw_local
 * (pop-then-conditional-sub).
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/home.h"
#include "arts/db.h"
#include "arts/edt.h"             /* arts_edt_dep_t (acquire body) */
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C self-send) */
#include "arts/ooo.h"             /* OOO_DB_* args (handler bodies) */
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"   /* arts_global_rank_id */
#include "arts/transport/outbox.h" /* arts_transport_send_async */
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h" /* arts_atomic_* */
#include "arts/utils/malloc.h"  /* arts_malloc / arts_free (transfer sender) */

/* ===== 8-case acquire dispatch (LAZY arm) ==========================
 * Whole arts_handler_db_acquire body for the LAZY build.  Diverges from EAGER
 * only on the RO-has-local-data predicate (LAZY: is_owner — only the current
 * owner holds an installed buffer; a home-but-not-owner rank goes through
 * acquire_remote_ro and home forwards via REDIRECT_RO).  On a successful local
 * RW fast path the run path delivers the dep — the handler does NOT call
 * arts_db_acquire_resolved. */
void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *dep = &((arts_edt_dep_t *)arts_get_depv(edt))[slot];
  arts_db_access_mode_t mode = dep->mode;
  /* writer_count > 0 means owner ({2,1}); the (int) cast is defensive. */
  bool is_owner = ((int)arts_atomic_read(&cache->writer_count) > 0);

  if (mode == DB_MODE_RO) {
    if (is_owner) { /* lazy RO predicate (only the owner holds the canonical
                       copy) */
      dep->ptr = arts_db_acquire_local(cache);
      arts_db_acquire_resolved(edt, slot);
      return;
    }
    arts_db_acquire_remote_ro(cache, edt->guid, slot); /* parks (SNAPSHOT) */
    return;
  }
  /* RW */
  /* Gate: a TRANSFER installed our buffer + sentinel but home has not confirmed
   * the rw_holder flip yet. Running now would make this write observable before
   * the directory names us (the stale-RO window), so a fresh RW acquire parks
   * until CONFIRM_ACK drains it. ownership_req_in_flight is held by the
   * in-flight round, so acquire_remote_rw parks without issuing a duplicate
   * request. */
  bool can_run_rw =
      is_owner && (arts_atomic_read(&cache->ownership_unconfirmed) == 0);
  if (can_run_rw &&
      arts_db_acquire_rw_local_fast(cache, dep, edt->guid, slot)) {
    /* The run path delivers this dep; do NOT resolve here. */
    return;
  }
  arts_db_acquire_remote_rw(cache, edt->guid,
                            slot); /* parks (OWNERSHIP_REQUEST) */
}

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW;
}

/* ===== release_rw (lazy arm) =======================================
 * The lazy protocol drops the buffer ref BEFORE relinquishing the token, so the
 * slot's cache-hold is the only ref that can keep the buffer alive past
 * writer_count==0 (a concurrent teardown then frees it via the cb deleter with
 * no dangling local ref).  The single-writer token release (pop-then-
 * conditional-sub + LOCAL Dekker re-check, and the 0-edge owner→owner ship) is
 * in arts_db_release_rw_local (coherence/mrsw/ownership.c). */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped ownership;
   * the release token logic would underflow.  Atomic acquire-load avoids a
   * TSan race. */
  if (arts_atomic_read(&cache->writer_count) == 0) {
    return;
  }
  /* Version bump on the current buffer (local ref scoped to release_rw). */
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
  }
  /* Drop the buffer ref BEFORE the token release (close the writer_count==0
   * teardown window). */
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  /* Single-writer token release: hand the token to the next waiter (pop) or
   * withdraw it (sub) with a LOCAL Dekker re-check, shipping TRANSFER_OWNERSHIP
   * only on the true 0-edge with a transfer target pending. */
  arts_db_release_rw_local(cache);
}

/* ===== cache_s lifecycle (lazy: pending_rw FIFO + dedup map + sentinel) =
 * Construct: the lazy protocol's field-init (the per-cache RW-waiter FIFO + the
 * owner-side dedup map [lazy-allocated] + the transfer sentinel) runs BEFORE
 * arts_db_cache_common_init.  Destruct order: buffer-NULL (pre) → pending_rw
 * destroy → snapshot drain + home teardown (post). */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  arts_db_rw_waiter_queue_init(&c->pending_rw);
  /* Lazy owner-side fields: dedup map allocated lazily on first ownership
   * grant; incoming_new_owner starts at the sentinel (no transfer pending). */
  c->last_sent_version = NULL;
  c->incoming_new_owner = ARTS_LAZY_NO_PENDING_OWNER;
  c->ownership_unconfirmed = 0u;
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache); /* buffer-NULL FIRST */
  /* Refcount hit 0 → this destructor is the SOLE owner of the cache: no other
   * ref-holder exists, so it is the unique safe single consumer of the pop-one
   * pending_rw FIFO.  Wake every still-parked RW waiter with NULL data (the
   * buffer slot was NULLed by destroy_pre, so mark_edt_ready_by_guid delivers
   * depv[slot].ptr=NULL and accounts the dep) BEFORE freeing the queue nodes.
   * This is where the destroy fan-out's deferred RW wake lands (the handler no
   * longer drains pending_rw; the releasing token holder relinquished without
   * popping).  Snapshot waiters are woken too (arts_db_drain_pending_snapshot
   * wakes; the common-post path only frees), then the queue is torn down.
   *
   * Skip the wake during final runtime teardown (shutdown_state != 0): the
   * worker scheduler is gone by the time arts_clean_up_dbs frees the route
   * table, so accounting a dep / scheduling a parked EDT would dereference a
   * destroyed deque.  At shutdown the parked EDTs are abandoned with the rest
   * of the graph — only the FIFO nodes still need freeing. */
  if (arts_node_info.shutdown_state == 0) {
    arts_guid_t edt_guid;
    unsigned int slot;
    while (arts_db_rw_waiter_queue_pop(&cache->pending_rw, &edt_guid, &slot)) {
      mark_edt_ready_by_guid(edt_guid, slot);
    }
    arts_db_drain_pending_snapshot(cache); /* wake parked snapshot waiters */
  }
  arts_db_rw_waiter_queue_destroy(&cache->pending_rw);
  arts_db_cache_common_destroy_post(cache); /* snapshot free → home teardown */
}

/* ===== home-directory lifecycle (inlined in arts_db_s) ============= */

void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks) {
  atomic_store_explicit(&db->rw_holder, rw_holder, memory_order_relaxed);
  arts_home_lockreq_queue_init(&db->pending_rw);
  atomic_store_explicit(&db->invalidate_in_flight, 0, memory_order_relaxed);
  arts_rank_bitset_init(&db->cached_ranks, nranks);
  db->pending_install_owner = 0;
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_lockreq_queue_destroy(&db->pending_rw);
  arts_rank_bitset_destroy(&db->cached_ranks);
  /* No free: home fields are inlined in the arts_db_s. */
}

/* ===== Lazy start_invalidate_round ==================================== */

void arts_db_lazy_start_invalidate_round(struct arts_db_cache_s *cache,
                                         unsigned int new_owner) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  unsigned int current_owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  /* INVALIDATE target is always rw_holder, which home publishes only after that
   * rank's cache install (creator at DB_CREATE, or the new owner at CONFIRM),
   * so the target's cache is provably already installed; the lazy protocol
   * never defers INVALIDATE. */
  arts_send_db_ownership_invalidate(current_owner, cache->db_guid, new_owner);
}

/* ===== Lazy TRANSFER_OWNERSHIP handler (new owner C) ===================
 * MRSW: install jumps writer_count 0->2 (sentinel + TOKEN).  The RW drain is
 * DEFERRED to CONFIRM_ACK (home has not flipped rw_holder yet) — the token is
 * the gate-held running-writer account, relinquished only by
 * arts_db_release_rw_local after CONFIRM_ACK runs the writer. */
void arts_handler_db_ownership_response(void *payload, size_t size) {
  struct arts_msg_ownership_response_packet_s *hdr =
      (struct arts_msg_ownership_response_packet_s *)payload;
  arts_guid_t db_guid = hdr->db_guid;

  arts_shared_ptr_t db_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    db_h = arts_db_cache_lazy_install(db_guid, /*db_size=*/0);
    db = (struct arts_db_s *)arts_shared_get(db_h);
    if (db == NULL) {
      arts_shared_release(&db_h);
      return; /* DB destroyed before we became owner — drop. */
    }
  }
  struct arts_db_cache_s *cache = &db->cache;

  /* Wire layout: header | map (count pairs) | data bytes */
  char *map_start = (char *)payload + sizeof(*hdr);
  size_t map_size =
      (sizeof(uint32_t) * 2) + ((size_t)hdr->map_entry_count *
                                sizeof(struct arts_msg_rank_version_pair_s));
  char *data_start = map_start + map_size;
  size_t data_size = size - sizeof(*hdr) - map_size;

  /* Reconstruct the owner-side dedup map so this rank can skip redundant
   * DATA_RESPONSE sends to readers that already hold a fresh-enough copy. */
  if (cache->last_sent_version != NULL) {
    arts_rank_u64_map_destroy(cache->last_sent_version);
  }
  cache->last_sent_version = arts_rank_u64_map_deserialize(
      map_start, map_size, arts_global_rank_count);

  /* Install the transferred buffer. */
  if (data_size > 0) {
    arts_db_buf_install(cache, hdr->version, data_start, data_size);
    if (cache->db_size == 0) {
      cache->db_size = data_size;
    }
  }

  /* Gate this rank's RW execution until home confirms the rw_holder flip. Set
   * the flag BEFORE the writer_count bump (plain store ordered before the
   * atomic RMW, same discipline as incoming_new_owner vs the INVALIDATE sub).
   */
  cache->ownership_unconfirmed = 1u;
  /* Sentinel(+1) + token(+1), single op (0->2): no intermediate 1 a racing
   * INVALIDATE could catch at 0.  The token is the running writer's account,
   * held until release_rw_local after the CONFIRM_ACK drain runs it.  An
   * absolute swap(1) would clobber a racing INVALIDATE's decrement. */
  arts_atomic_add(&cache->writer_count, 2u);

  /* RW drain is DEFERRED to CONFIRM_ACK (home has not flipped rw_holder yet).
   * The snapshot + OoO drains stay: a parked RO waiter served here gets the
   * transferred (pre-write) version, which is correct, and a reordered
   * INVALIDATE deferred on a previously-missing cache replays now. */
  arts_db_drain_pending_snapshot(cache);
  arts_ooo_drain_guid(db_guid);

  /* No racing INVALIDATE can have reached us yet (home targets this rank as an
   * INVALIDATE recipient only after the rw_holder flip, which needs this
   * CONFIRM).  Send CONFIRM unconditionally.  ownership_req_in_flight stays 1
   * until CONFIRM_ACK so fresh RW acquires in the gate window park without
   * issuing a duplicate request. */
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  arts_send_db_ownership_confirm(home_rank, db_guid, hdr->version);

  arts_shared_release(&db_h);
}

/* ===== Lazy CONFIRM handler (home A) =============================== */

/* Cat-C pure body (CONFIRM, home side).  args_v is unused (the new owner is
 * read from db->pending_install_owner).  Records the new rw_holder, then either
 * advances the round in the SAME message (CONFIRM_ACK piggybacks the next
 * transfer target — the merged INVALIDATE) or releases the baton. */
void arts_handler_db_ownership_confirm(void *item_v, void *args_v) {
  (void)args_v;
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_s *db = arts_db_of_cache(cache);

  /* Publish the new rw_holder (visible to GET_DATA redirect path). */
  unsigned int new_owner = db->pending_install_owner;
  atomic_store_explicit(&db->rw_holder, new_owner, memory_order_release);

  /* If a next requester is queued, advance the round in the SAME message — the
   * CONFIRM_ACK piggybacks the next transfer target, and the new owner's
   * handler applies the INVALIDATE effect itself (publish incoming_new_owner +
   * withdraw the sentinel), removing the CONFIRM_ACK↔INVALIDATE reorder
   * window. */
  unsigned int piggyback = ARTS_LAZY_NO_PENDING_OWNER;
  {
    unsigned int next_owner;
    if (arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
      db->pending_install_owner = next_owner;
      piggyback = next_owner;
    }
  }
  arts_send_db_ownership_confirm_ack(new_owner, cache->db_guid, piggyback);
  if (piggyback != ARTS_LAZY_NO_PENDING_OWNER) {
    /* Round advanced via the merged ack; the baton stays held until the new
     * owner releases (transfer-commit). */
    return;
  }

  /* No pending requester — release the baton (with the freshly-enqueued-racer
   * recheck retry loop). */
  while (1) {
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    if (arts_home_lockreq_queue_empty(&db->pending_rw)) {
      return;
    }
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      return;
    }
    unsigned int next_owner;
    if (arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
      db->pending_install_owner = next_owner;
      arts_db_lazy_start_invalidate_round(cache, next_owner);
      return;
    }
  }
}

/* ===== Lazy CONFIRM_ACK handler (new owner C) ==================== */

/* Cat-C pure body (CONFIRM_ACK, new-owner side). Home has flipped rw_holder to
 * this rank; it is now safe for this rank's RW EDT to run.  Open the gate,
 * drain the ONE RW waiter deferred at TRANSFER (the token's writer), apply the
 * piggybacked invalidate effect (if the round advanced), and — MRSW — do NOT
 * remove a drain guard: the token relinquished by release_rw_local IS the
 * deferred 0-edge.  After pop-one + the piggyback sentinel sub the running
 * token writer's release reaches the 0-edge and transfers.
 *
 * args_v is the CONFIRM_ACK packet: its new_owner_rank carries the next
 * transfer target when the round advanced, or ARTS_LAZY_NO_PENDING_OWNER for a
 * plain ack. */
void arts_handler_db_ownership_confirm_ack(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_msg_ownership_confirm_ack_packet_s *p =
      (struct arts_msg_ownership_confirm_ack_packet_s *)args_v;

  /* Open the gate: fresh RW acquires may now take the fast path, and the
   * coalescing flag is released so a future round can re-issue. */
  cache->ownership_unconfirmed = 0u;
  cache->ownership_req_in_flight = 0u;

  /* Drain the ONE RW waiter the TRANSFER handler deferred (the token's writer).
   * The token (the +1 above the sentinel) already accounts it; run-one delivers
   * the dep and the writer will release via arts_db_release_rw_local. */
  arts_db_drain_pending_rw_after_grant(cache, /*version=*/0,
                                       /*has_next=*/false);

  /* Piggybacked INVALIDATE effect (merged from the former standalone
   * INVALIDATE_NOTICE).  Mirror the INVALIDATE handler's discipline: publish
   * the transfer target BEFORE withdrawing the sentinel (the publish-before-sub
   * is the dedup against the running writer's release that observes the
   * 0-edge).
   *
   * The sub withdraws the SENTINEL.  Two outcomes:
   *   - the drain popped a writer (token held): rest >= 1 here — the running
   *     writer's release_rw_local crosses the 0-edge and ships.  Do NOT ship.
   *   - the drain found the queue empty and dropped the orphan token to idle
   *     (writer_count == 1, sentinel-only): this sub takes it to 0, and NO
   *     writer release is coming — so THIS handler is the unique 0-edge actor
   *     and must ship now. */
  if (p != NULL && p->new_owner_rank != ARTS_LAZY_NO_PENDING_OWNER) {
    cache->incoming_new_owner = p->new_owner_rank;
    if ((int)arts_atomic_sub(&cache->writer_count, 1) == 0 &&
        cache->incoming_new_owner != ARTS_LAZY_NO_PENDING_OWNER) {
      arts_db_send_ownership_response(cache);
    }
  }
}

/* ===== Per-model wire-handler bodies =============================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_SNAPSHOT_REQUEST]): the OoO engine
 * has already acquired the home db_s and pinned a ref across this call (cache
 * is its FIRST member). */
void arts_handler_db_snapshot_request(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_snapshot_request_s *a =
      (struct arts_ooo_args_db_snapshot_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_guid_t edt_guid = a->edt_guid;
  uint32_t slot = a->slot;

  /* Lazy home-side RO routing: home does not hold the canonical copy — the
   * current owner does.  Record the requester in the cached-ranks set, then
   * redirect to the current owner (REDIRECT_RO) so the owner serves
   * DATA_RESPONSE directly, applying the owner-side last_sent_version dedup. */
  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_rank_bitset_set(&db->cached_ranks, requester);
  unsigned int owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  arts_send_db_snapshot_redirect(owner, cache->db_guid, requester, edt_guid,
                                 slot);
}

/* Fan-out callback for arts_rank_bitset_for_each during destroy. */
static void lazy_destroy_fanout_cb(unsigned int rank, void *ctx) {
  arts_guid_t db_guid = (arts_guid_t)(uintptr_t)ctx;
  unsigned int self = arts_global_rank_id;
  if (rank != self) {
    arts_send_db_cache_destroy(rank, db_guid);
  }
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_DESTROY]): the OoO engine has already
 * acquired the home db_s and pinned a ref across this call (cache is its FIRST
 * member).  Order: remote DESTROY_NOTIFY roster fan-out FIRST (the cache stays
 * alive — only the install ref is dropped), then arts_route_table_set_destroyed
 * LAST.  Lazy roster source = rw_holder (current RW owner) + the RO
 * cached-ranks bit-set + the queued ownership requesters + the in-flight
 * transfer target.
 *
 * MRSW does NOT drain cache.pending_rw here (it is a pop-one FIFO a token
 * holder can be releasing concurrently — a second consumer would corrupt the
 * chain). The parked RW + snapshot waiters are woken by the refcount-0 cache
 * destructor (arts_db_cache_destructor), the SOLE owner once set_destroyed
 * drops the last install ref; a token holder racing its release defers to the
 * destructor via the arts_route_table_was_destroyed guard in
 * arts_db_release_rw_local. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  {
    unsigned int holder =
        atomic_load_explicit(&db->rw_holder, memory_order_acquire);
    if (holder != self) {
      arts_send_db_cache_destroy(holder, a->db_guid);
    }
  }
  arts_rank_bitset_for_each(&db->cached_ranks, lazy_destroy_fanout_cb,
                            (void *)(uintptr_t)a->db_guid);
  {
    unsigned int q_rank;
    while (arts_home_lockreq_queue_pop(&db->pending_rw, &q_rank)) {
      if (q_rank != self) {
        arts_send_db_cache_destroy(q_rank, a->db_guid);
      }
    }
  }
  /* In-flight ownership transfer: the new owner C lives only in
   * pending_install_owner during [pop at round-start .. rw_holder flip] and is
   * in none of the rosters above.  With the confirm gate it parks its RW waiter
   * until CONFIRM_ACK, so a destroy that races the transfer must wake it here
   * or it hangs.  Notify it (dedup against rw_holder / self). */
  if (atomic_load_explicit(&db->invalidate_in_flight, memory_order_acquire) !=
      0u) {
    unsigned int in_flight = db->pending_install_owner;
    unsigned int holder =
        atomic_load_explicit(&db->rw_holder, memory_order_acquire);
    if (in_flight != self && in_flight != holder) {
      arts_send_db_cache_destroy(in_flight, a->db_guid);
    }
  }
  /* Do NOT drain cache.pending_rw / pending_snapshot here: a token holder can
   * be releasing pending_rw concurrently, and that pop-one FIFO must stay
   * single-consumer.  set_destroyed drops the last install ref → the refcount-0
   * destructor (arts_db_cache_destructor) is the sole owner and wakes every
   * parked waiter. */
  (void)cache;
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* Case-D leaf: lazy publishes creator_rank as the home rw_holder. */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

/* ===== Ownership-round seams (called from coherence/mrsw/ownership.c) == */

void arts_db_start_ownership_round(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester) {
  (void)requester;
  /* Lazy: pop the OLDEST requester (FIFO) to be the transfer target and embed
   * its rank in the INVALIDATE_NOTICE so the current holder ships
   * TRANSFER_OWNERSHIP directly.  pending_install_owner is only written by the
   * baton holder (single writer), so no atomic. */
  unsigned int next_owner;
  if (!arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
    /* Defensive: we just pushed, so empty is impossible under correct usage. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    return;
  }
  db->pending_install_owner = next_owner;
  arts_db_lazy_start_invalidate_round(cache, next_owner);
}

/* ===== Lazy INVALIDATE_NOTICE handler (pure body, DIRECT-call) ======
 * The lazy protocol does NOT route INVALIDATE through the OoO engine: the
 * invalidate target is always the rw_holder, whose CACHE the requester
 * lazy-installs before it ever sends OWNERSHIP_REQUEST, so the cache is present
 * and the wire dispatcher / self-send call this body directly.  cache is the
 * FIRST member of arts_db_s (offset 0).
 *
 * MRSW: the count here is sentinel + token (or sentinel-only if the writer has
 * quiesced).  The sub withdraws the SENTINEL; the 0-edge fires only when no
 * token remains (a running writer holds the +1), so transfer-gating is
 * automatic — the last release_rw_local ships instead. */
void arts_handler_db_ownership_invalidate(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_ownership_invalidate_s *a =
      (struct arts_ooo_args_db_ownership_invalidate_s *)args_v;
  /* Publish the transfer target BEFORE withdrawing the sentinel.  This ordering
   * is the dedup (no separate transfer_pending flag): a concurrent release that
   * observes the 0-edge is guaranteed to see incoming_new_owner already
   * published, so exactly one of {this handler, the last releaser} ships. */
  cache->incoming_new_owner = a->new_owner_rank;
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1);
  if (rest != 0) {
    /* rest > 0: a token (running writer) still held — the last
     * release_rw_local, seeing rest==0 with incoming_new_owner already
     * published, ships instead. */
    return;
  }
  /* rest == 0: we are the unique transfer actor (no token remained). */
  arts_db_send_ownership_response(cache);
}

/* ===== Lazy REDIRECT_RO handler (owner side) ======================= */

/* Cat-C pure body (REDIRECT_RO, owner side).  The wire dispatcher / self-send
 * shortcut has already looked the owner-side db_s up with a held ref and passes
 * it as item_v (cache is its FIRST member).  On a MISS the dispatcher sends
 * DESTROY_NOTIFY to the requester. */
void arts_handler_db_snapshot_redirect(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_snapshot_redirect_args_s *a =
      (struct arts_db_snapshot_redirect_args_s *)args_v;
  unsigned int requester = a->requester_rank;
  arts_guid_t edt_guid = a->edt_guid;
  uint32_t slot = a->slot;

  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf == NULL) {
    /* No buffer installed yet (pre-publication or sentinel DB).  Respond
     * version=0, no data — requester's RO waiter fires with undefined content
     * (per spec). */
    arts_send_db_snapshot_response(requester, a->db_guid, /*version=*/0,
                                   edt_guid, slot, /*data=*/NULL,
                                   /*data_size=*/0);
    return;
  }
  /* Invariant: last_sent_version is created at ownership-install
   * (TRANSFER_OWNERSHIP, retained permanently).  The INITIAL owner (the
   * creator, which never received a transfer) has no map yet, so lazily create
   * it on its first served REDIRECT. */
  if (cache->last_sent_version == NULL) {
    cache->last_sent_version = arts_rank_u64_map_create(arts_global_rank_count);
  }

  uint64_t cur_v = buf->version;
  uint64_t last_sent =
      arts_rank_u64_map_get(cache->last_sent_version, requester);

  if (last_sent >= cur_v) {
    /* Requester already holds this version — send no-data response. */
    arts_send_db_snapshot_response(requester, a->db_guid, cur_v, edt_guid, slot,
                                   NULL, 0);
  } else {
    /* Advance dedup watermark (monotonic max) then send data. */
    arts_rank_u64_map_advance(cache->last_sent_version, requester, cur_v);
    arts_send_db_snapshot_response(requester, a->db_guid, cur_v, edt_guid, slot,
                                   buf->data, cache->db_size);
  }
  arts_db_buf_release(&buf_h);
}

/* ===== Lazy wire senders =========================================
 * The OWNERSHIP_RESPONSE wire sender + the owner→owner ship are shared
 * (coherence/mrsw/ownership.c); the CONFIRM sender is shared too.  Only the
 * lazy-only CONFIRM_ACK + REDIRECT_RO senders remain here. */

void arts_send_db_ownership_confirm_ack(unsigned int new_owner_rank,
                                        arts_guid_t db_guid,
                                        unsigned int piggyback_new_owner) {
  struct arts_msg_ownership_confirm_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_CONFIRM_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.new_owner_rank = piggyback_new_owner;
  memset(p.pad, 0, sizeof(p.pad));
  if (new_owner_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop.
     * HIT runs the confirm_ack body (applying the piggybacked invalidate effect
     * if any); MISS (DB destroyed) silently drops (gated waiters are woken by
     * the refcount-0 cache destructor). */
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_ownership_confirm_ack(db, &p);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)new_owner_rank, (char *)&p, sizeof(p));
}

void arts_send_db_snapshot_redirect(unsigned int owner_rank,
                                    arts_guid_t db_guid,
                                    unsigned int requester_rank,
                                    arts_guid_t edt_guid, uint32_t slot) {
  struct arts_msg_snapshot_redirect_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_SNAPSHOT_REDIRECT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.edt_guid = edt_guid;
  p.requester_rank = requester_rank;
  p.slot = slot;
  if (owner_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire.  HIT
     * serves DATA_RESPONSE from the ref-pinned owner-side db_s; MISS (DB
     * destroyed / not yet installed) sends DESTROY_NOTIFY to the requester. */
    struct arts_db_snapshot_redirect_args_s args = {
        .db_guid = db_guid,
        .edt_guid = edt_guid,
        .requester_rank = requester_rank,
        .slot = slot,
    };
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_snapshot_redirect(db, &args);
    } else {
      arts_send_db_cache_destroy(requester_rank, db_guid);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

/* Case-D leaf: the lazy protocol defers the home-buffer install to the
 * creator's first release_rw (WRITEBACK / GRANT path); nothing to do at
 * create. */
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  (void)cache;
  (void)db_size;
}
