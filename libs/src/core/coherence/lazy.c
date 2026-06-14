/* SPDX-License-Identifier: Apache-2.0
 *
 * LAZY protocol translation unit: defines the LAZY-specific
 * arts_handler_db_* / arts_db_* bodies directly (CMake links exactly this TU
 * for an MRNEW+LAZY build) plus the LAZY-only wire handlers/senders.
 * Compiled only for ARTS_COHERENCE_PROTOCOL=MRNEW with
 * ARTS_PROTOCOL_TIMING=LAZY (selected in libs/src/core/CMakeLists.txt).
 * Contains NO protocol/timing preprocessor logic.
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

/* The lazy protocol transfers ownership owner->owner via
 * arts_db_lazy_send_ownership_response, not a home-side GRANT, so there is no
 * local chain-advance.  invalidate_transfer still calls this on the home path;
 * it is a no-op stub for the lazy build. */
void arts_db_local_transfer_now(struct arts_db_cache_s *cache) { (void)cache; }

/* ===== 8-case acquire dispatch (LAZY arm) ==========================
 * Whole arts_handler_db_acquire body for the LAZY build.  Diverges from EAGER
 * only on the RO-has-local-data predicate (LAZY: is_owner — the home rank does
 * NOT hold the canonical copy; only the current owner has an installed buffer,
 * so a home-but-not-owner rank goes through acquire_remote_ro and home forwards
 * to the owner via REDIRECT_RO). */
void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *dep = &((arts_edt_dep_t *)arts_get_depv(edt))[slot];
  arts_db_access_mode_t mode = dep->mode;
  /* Signed: the commutative writer_count is transiently negative when an
   * INVALIDATE races ahead of its GRANT (multi-receiver wire reorder) —
   * negative means NOT owner.  An unsigned compare would treat it as owner
   * and serve RO from a non-owned (stale) buffer. */
  bool is_owner = ((int)cache->writer_count > 0);

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
   * until CONFIRM drains it. ownership_req_in_flight is held by the in-flight
   * round, so acquire_remote_rw parks without issuing a duplicate request. */
  bool can_run_rw =
      is_owner && (arts_atomic_read(&cache->ownership_unconfirmed) == 0);
  if (can_run_rw && arts_db_acquire_rw_local_fast(cache, dep)) {
    arts_db_acquire_resolved(edt, slot); /* data here, writer_count bumped */
    return;
  }
  arts_db_acquire_remote_rw(cache, edt->guid,
                            slot); /* parks (OWNERSHIP_REQUEST) */
}

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW;
}

/* ===== release_rw (lazy arm) =======================================
 * The lazy protocol drops the buffer ref BEFORE decrementing writer_count, so
 * the slot's cache-hold is the only ref that can keep the buffer alive past
 * writer_count==0 (a concurrent teardown then frees it via the cb deleter with
 * no dangling local ref).  In the eager protocol this window does not exist
 * (local_transfer_now restores the sentinel); the lazy protocol has no sentinel
 * restoration, so it must close the window by releasing the ref before exposing
 * writer_count==0. */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped ownership;
   * decrementing would underflow.  Atomic acquire-load avoids a TSan race. */
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
  /* Drop the buffer ref BEFORE the writer_count decrement (close the
   * writer_count==0 teardown window). */
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  /* Signed, like the INVALIDATE/guard fire sites: the commutative counter can
   * be transiently negative (an INVALIDATE racing ahead of its add), which must
   * read
   * != 0 here.  A true 1->0 release reads 0 and ships; a transient 0->-1 reads
   * -1 and does not. */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1); /* post value */
  if (rest == 0) {
    /* If an INVALIDATE_NOTICE already published a transfer target while writers
     * were live, this (last) releaser is the unique actor that ships
     * TRANSFER_OWNERSHIP — sentinel invariant, no flag.  Identical for home and
     * non-home owners.  Otherwise no transfer is pending: home retains
     * ownership until a future OWNERSHIP_REQUEST; a non-home owner quiesces. */
    if (cache->incoming_new_owner != ARTS_LAZY_NO_PENDING_OWNER) {
      arts_db_lazy_send_ownership_response(cache);
    }
  }
}

/* ===== cache_s lifecycle (lazy: pending_rw + dedup map + sentinel) =
 * Construct: the lazy protocol's field-init (the Vyukov MPSC pending_rw queue +
 * the owner-side dedup map [lazy-allocated] + the transfer sentinel) runs
 * BEFORE arts_db_cache_common_init.  Destruct order: buffer-NULL (pre) →
 * pending_rw destroy → snapshot drain + home teardown (post). */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  arts_pending_rw_queue_init(&c->pending_rw);
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
  arts_pending_rw_queue_destroy(&cache->pending_rw);
  arts_db_cache_common_destroy_post(cache); /* snapshot drain → home teardown */
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

/* ===== last_sent_version map serialization (lazy ownership response) = */

size_t arts_rank_u64_map_serialize(const struct arts_rank_to_u64_map_s *m,
                                   void *out) {
  uint32_t *count_field = (uint32_t *)out;
  struct arts_msg_rank_version_pair_s *entries =
      (struct arts_msg_rank_version_pair_s *)((char *)out +
                                              (sizeof(uint32_t) * 2));
  uint32_t n = 0;
  for (unsigned int r = 0; r < m->nranks; r++) {
    uint64_t v = atomic_load_explicit(&m->slots[r], memory_order_acquire);
    if (v == 0) {
      continue;
    }
    entries[n].rank = (uint32_t)r;
    entries[n].pad = 0;
    entries[n].version = v;
    n++;
  }
  count_field[0] = n;
  count_field[1] = 0; /* alignment pad */
  return (sizeof(uint32_t) * 2) + ((size_t)n * sizeof(*entries));
}

struct arts_rank_to_u64_map_s *
arts_rank_u64_map_deserialize(const void *in, size_t size,
                              unsigned int nranks) {
  (void)size; /* used by debug assertions; production ignores it */
  struct arts_rank_to_u64_map_s *m = arts_rank_u64_map_create(nranks);
  const uint32_t *count_field = (const uint32_t *)in;
  uint32_t n = count_field[0];
  const struct arts_msg_rank_version_pair_s *entries =
      (const struct arts_msg_rank_version_pair_s *)((const char *)in +
                                                    (sizeof(uint32_t) * 2));
  for (uint32_t i = 0; i < n; i++) {
    arts_rank_u64_map_set(m, (unsigned int)entries[i].rank, entries[i].version);
  }
  return m;
}

/* ===== Lazy ownership-transfer wire handlers (moved from handlers.c) ===
 * (arts_db_drain_pending_snapshot / arts_db_drain_pending_rw_after_grant are
 * declared in coherence/coherence.h.) */

/* ===== Lazy ship_transfer / start_invalidate_round ==================== */

void arts_db_lazy_send_ownership_response(struct arts_db_cache_s *cache) {
  unsigned int new_owner = cache->incoming_new_owner;
  /* Re-arm the sentinel BEFORE the send, not after.  A self-transfer
   * (new_owner == this rank) ships via an inline self-dispatch that recursively
   * runs the new owner's install → CONFIRM → INSTALL_ACK → home's next transfer
   * round → that round's INVALIDATE, which republishes incoming_new_owner.  If
   * the re-arm ran after the send it would clobber that freshly-published next
   * target with the sentinel, so the release that later drives writer_count to
   * 0 would read "no transfer pending" and ship nothing — stranding the next
   * owner.  Clearing it up front (we already captured new_owner) leaves any
   * nested round's publish intact. */
  cache->incoming_new_owner = ARTS_LAZY_NO_PENDING_OWNER;
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf == NULL) {
    /* Sentinel DB (db_size==0) or pre-publication: send an empty transfer.
     * Still emit the 8-byte map count-header (count=0, pad=0): the receiver
     * unconditionally reconstructs map_size >= 8, so omitting it would
     * underflow data_size to (size_t)-8 and corrupt the install. */
    uint32_t empty_map[2] = {0u, 0u};
    arts_send_db_ownership_response(new_owner, cache->db_guid,
                                    /*version=*/0, empty_map, sizeof(empty_map),
                                    /*data=*/NULL, /*data_size=*/0);
    return;
  }

  /* Serialize the owner-side dedup map for transfer. */
  size_t map_max =
      (sizeof(uint32_t) * 2) + ((size_t)arts_global_rank_count *
                                sizeof(struct arts_msg_rank_version_pair_s));
  void *map_buf = arts_malloc(map_max);
  size_t map_size;
  if (cache->last_sent_version != NULL) {
    map_size = arts_rank_u64_map_serialize(cache->last_sent_version, map_buf);
  } else {
    /* No map yet: emit an empty map (count=0, pad=0). */
    uint32_t *p = (uint32_t *)map_buf;
    p[0] = 0u;
    p[1] = 0u;
    map_size = sizeof(uint32_t) * 2;
  }

  arts_send_db_ownership_response(new_owner, cache->db_guid, buf->version,
                                  map_buf, map_size, buf->data, cache->db_size);
  arts_free(map_buf);

  /* Release the local buffer ref taken above.  The slot's cache-hold ref keeps
   * the buffer alive: the old RW owner retains its buffer + last_sent_version
   * map permanently (until DB destroy), so in-flight RO REDIRECTs that still
   * name this rank as owner are served from its own copy. */
  arts_db_buf_release(&buf_h);
}

void arts_db_lazy_start_invalidate_round(struct arts_db_cache_s *cache,
                                         unsigned int new_owner) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  unsigned int current_owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  /* INVALIDATE target is always rw_holder, which home publishes only after that
   * rank's cache install (creator at DB_CREATE, or the new owner at
   * INSTALL_ACK).  So the target's cache is provably already installed when the
   * INVALIDATE arrives — the lazy protocol never defers INVALIDATE; do not
   * route it through dispatch_or_defer (the dispatcher / self-send call the
   * handler body directly, guarded by assert(cache != NULL)). */
  arts_send_db_ownership_invalidate(current_owner, cache->db_guid, new_owner);
}

/* ===== Lazy TRANSFER_OWNERSHIP handler (new owner C) =================== */

void arts_handler_db_ownership_response(void *payload, size_t size) {
  struct arts_msg_ownership_response_packet_s *hdr =
      (struct arts_msg_ownership_response_packet_s *)payload;
  arts_guid_t db_guid = hdr->db_guid;

  struct arts_db_cache_s *cache = arts_db_cache_lookup(db_guid);
  if (cache == NULL) {
    cache = arts_db_cache_lazy_install(db_guid, /*db_size=*/0);
    if (cache == NULL) {
      return; /* DB destroyed before we became owner — drop. */
    }
  }

  /* Wire layout: header | map (count pairs) | data bytes */
  char *map_start = (char *)payload + sizeof(*hdr);
  size_t map_size =
      (sizeof(uint32_t) * 2) + ((size_t)hdr->map_entry_count *
                                sizeof(struct arts_msg_rank_version_pair_s));
  char *data_start = map_start + map_size;
  size_t data_size = size - sizeof(*hdr) - map_size;

  /* Reconstruct the owner-side dedup map so this rank can skip
   * redundant DATA_RESPONSE sends to readers that already hold a
   * sufficiently fresh copy. */
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

  /* ADD the ownership sentinel (+1) PLUS a transient DRAIN GUARD (+1) in a
   * single atomic op (jump 0->2, no intermediate 1 a racing INVALIDATE could
   * catch at 0) — the same scheme the eager GRANT uses.  The guard keeps
   * writer_count >= 1 across the per-waiter +1 drain below, so a commutative
   * INVALIDATE(-1) cannot zero the count mid-drain and ship the transfer before
   * this rank's parked writers are counted+scheduled; the 0-crossing is
   * deferred to guard-removal (after the drain).  Jumping to 2 also makes a
   * fresh local RW acquire take the fast path (cswap +1) instead of parking, so
   * the drain sees exactly the pre-TRANSFER waiter set.  An absolute swap(1)
   * would clobber a racing INVALIDATE's decrement and lose the transfer
   * (distributed hang). */
  /* Gate this rank's RW execution until home confirms the rw_holder flip. Set
   * the flag BEFORE the writer_count bump (plain store ordered before the
   * atomic RMW, same discipline as incoming_new_owner vs the INVALIDATE sub): a
   * worker doing a fresh RW acquire reads writer_count then the gate, so any
   * observer of the bumped count must also observe the gate. */
  cache->ownership_unconfirmed = 1u;
  /* Sentinel (+1) + drain guard (+1), single op (0->2). The guard is held until
   * the CONFIRM handler, so a next-round INVALIDATE racing ahead of CONFIRM
   * cannot zero the count and ship before this rank has used its ownership. */
  arts_atomic_add(&cache->writer_count, 2u);

  /* RW drain is DEFERRED to the CONFIRM handler (home has not flipped rw_holder
   * to us yet). The snapshot + OoO drains stay: a parked RO waiter served here
   * gets the transferred (pre-write) version, which is correct, and a reordered
   * INVALIDATE deferred on a previously-missing cache replays now. */
  arts_db_drain_pending_snapshot(cache);
  arts_ooo_drain_guid(db_guid);

  /* No racing INVALIDATE can have reached us yet: home targets this rank as an
   * INVALIDATE recipient only after the rw_holder flip, which needs this
   * INSTALL_ACK. So incoming_new_owner == NONE here — send INSTALL_ACK
   * unconditionally. ownership_req_in_flight stays 1 until CONFIRM so fresh RW
   * acquires in the gate window park without issuing a duplicate request. */
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  arts_send_db_ownership_response_ack(home_rank, db_guid, hdr->version);
}

/* ===== Lazy INSTALL_ACK handler (home A) =============================== */

/* Cat-C pure body (INSTALL_ACK, home side).  The wire dispatcher / self-send
 * shortcut has already looked the home db_s up with a held ref and passes it as
 * item_v (cache is its FIRST member, offset 0).  No lookup/NULL-check here —
 * the dispatcher's MISS branch SILENTLY DROPS (DB destroyed).  args_v is unused
 * (the new owner is read from db->pending_install_owner, published by the baton
 * holder; INSTALL_ACK only confirms the install completed). */
void arts_handler_db_ownership_response_ack(void *item_v, void *args_v) {
  (void)args_v;
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_s *db = arts_db_of_cache(cache);

  /* Publish the new rw_holder (visible to GET_DATA redirect path). */
  unsigned int new_owner = db->pending_install_owner;
  atomic_store_explicit(&db->rw_holder, new_owner, memory_order_release);

  /* The directory now names the new owner: tell it to run its gated RW EDTs.
   * CONFIRM and the next-round INVALIDATE below are both home→new_owner and may
   * reorder under multiple receivers; the new owner's drain guard absorbs that,
   * so send order does not matter — CONFIRM is sent first. */
  arts_send_db_ownership_confirm(new_owner, cache->db_guid);

  /* Drain-or-release retry loop: try to start the next transfer round
   * if there are pending_rw requests, otherwise release the baton. */
  while (1) {
    unsigned int next_owner;
    if (arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
      db->pending_install_owner = next_owner;
      arts_db_lazy_start_invalidate_round(cache, next_owner);
      /* Pipeline: the popped next_owner is the genuine next owner — PROCEED it
       * so it overlaps its RW acquire with the in-flight invalidate round. */
      arts_send_db_ownership_proceed(next_owner, cache->db_guid);
      return;
    }
    /* No pending requester — release the baton. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    /* Re-check for a freshly-enqueued requester that raced the baton
     * release.  If the queue is still empty, we're done. */
    if (arts_home_lockreq_queue_empty(&db->pending_rw)) {
      return;
    }
    /* There is a new requester; try to re-acquire the baton. */
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      /* Another OWNERSHIP_REQUEST handler already picked up the baton (race);
       * that thread will drain the queue. */
      return;
    }
    /* Re-acquired the baton; loop to pop and start the next round. */
  }
}

/* ===== Lazy OWNERSHIP_CONFIRM handler (new owner C) ==================== */

/* Cat-C pure body (OWNERSHIP_CONFIRM, new-owner side). Home has flipped
 * rw_holder to this rank; it is now safe for this rank's RW EDTs to run and
 * make their writes observable. Drain the RW waiters deferred at TRANSFER,
 * clear the gate, and remove the drain guard (the relocated 0-edge ship-check).
 */
void arts_handler_db_ownership_confirm(void *item_v, void *args_v) {
  (void)args_v;
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;

  /* Open the gate: fresh RW acquires may now take the fast path, and the
   * coalescing flag is released so a future round can re-issue. */
  cache->ownership_unconfirmed = 0u;
  cache->ownership_req_in_flight = 0u;

  /* Drain the RW waiters that the TRANSFER handler deferred (this is the work
   * moved out of arts_handler_db_ownership_response). */
  arts_db_drain_pending_rw_after_grant(cache, /*version=*/0,
                                       /*has_next=*/false);

  /* Remove the drain guard held across the INSTALL_ACK→CONFIRM round trip. If a
   * next-round INVALIDATE arrived between the flip and this CONFIRM it withdrew
   * the sentinel (commutative signed counter); if that drives the count to 0
   * and a transfer target is pending and no local writer remains, we are the
   * unique actor that ships TRANSFER_OWNERSHIP to the next owner. Otherwise
   * this rank retains ownership and its drained EDTs ship on their own release
   * 0-edge. */
  if ((int)arts_atomic_sub(&cache->writer_count, 1) == 0 &&
      cache->incoming_new_owner != ARTS_LAZY_NO_PENDING_OWNER) {
    arts_db_lazy_send_ownership_response(cache);
  }
}

/* ===== Per-model wire-handler bodies =============================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_SNAPSHOT_REQUEST]): the OoO engine
 * has already acquired the home db_s and pinned a ref across this call (cache
 * is its FIRST member), so there is no lookup / NULL-check / defer here. */
void arts_handler_db_snapshot_request(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_snapshot_request_s *a =
      (struct arts_ooo_args_db_snapshot_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_guid_t edt_guid = a->edt_guid;
  uint32_t slot = a->slot;

  /* Lazy home-side RO routing.
   *
   * Under the lazy protocol home does not hold the canonical data copy — the
   * current owner does.  Home's job is to redirect the requester to the owner
   * (via REDIRECT_RO) so the owner can send DATA_RESPONSE directly,
   * applying the owner-side last_sent_version dedup.
   *
   * Record the requester in the cached-ranks set, then redirect to the current
   * owner.  A destroyed DB is handled by the route_table lookup miss (slot
   * value NULL-stored before destroy) + the handler single-actor invariant —
   * no per-home destroy flag. */
  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_rank_bitset_set(&db->cached_ranks, requester);
  /* Forward to the current owner unconditionally: even mid-transfer, rw_holder
   * still names the OLD owner, which retains its buffer + last_sent_version
   * permanently and serves the REDIRECT from its own copy.  Client-side
   * monotonic version compare keeps stale snapshots safe.  No defer queue. */
  unsigned int owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  arts_send_db_snapshot_redirect(owner, cache->db_guid, requester, edt_guid,
                                 slot);
}

/* Fan-out callback for arts_rank_bitset_for_each during destroy.
 * ctx carries the db_guid encoded as uintptr_t (no heap allocation
 * needed since the callback is synchronous). */
static void lazy_destroy_fanout_cb(unsigned int rank, void *ctx) {
  arts_guid_t db_guid = (arts_guid_t)(uintptr_t)ctx;
  unsigned int self = arts_global_rank_id;
  if (rank != self) {
    arts_send_db_cache_destroy(rank, db_guid);
  }
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_DESTROY]): the OoO engine has already
 * acquired the home db_s and pinned a ref across this call (cache is its FIRST
 * member).  Order: roster fan-out + fail_trigger wake parked waiters FIRST,
 * then arts_route_table_set_destroyed LAST.  Lazy roster source = rw_holder
 * (current RW owner) + the RO cached-ranks bit-set + the queued ownership
 * requesters. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  /* Lazy: notify the current RW owner first (rw_holder, not the cached-ranks
   * bit-set), then the RO cached-ranks bit-set, then the queued requesters. */
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
   * in none of the rosters above. With the confirm gate it parks its RW waiter
   * until CONFIRM, so a destroy that races the transfer must wake it here or it
   * hangs. Notify it (dedup against rw_holder / self). */
  if (atomic_load_explicit(&db->invalidate_in_flight, memory_order_acquire) !=
      0u) {
    unsigned int in_flight = db->pending_install_owner;
    unsigned int holder =
        atomic_load_explicit(&db->rw_holder, memory_order_acquire);
    if (in_flight != self && in_flight != holder) {
      arts_send_db_cache_destroy(in_flight, a->db_guid);
    }
  }
  arts_db_fail_trigger_pending(cache);
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* Case-D leaf: lazy publishes creator_rank as the home rw_holder (coalesce
 * path). */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

/* ===== Ownership-round seams (called from coherence/ownership.c) ==
 * family→protocol: the ownership-family OWNERSHIP_REQUEST / RELEASE_OWNERSHIP
 * handlers delegate the EAGER/LAZY-divergent steps here. */

void arts_db_start_ownership_round(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester) {
  (void)requester;
  /* Lazy: pop the OLDEST requester (FIFO) to be the transfer target and
   * embed its rank in the INVALIDATE_NOTICE so the current holder ships
   * TRANSFER_OWNERSHIP directly, without a home round-trip.
   * pending_install_owner is only written by the baton holder (single
   * writer invariant), so no atomic needed. */
  unsigned int next_owner;
  if (!arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
    /* Defensive: we just pushed, so empty is impossible under correct
     * usage.  Release the baton and return. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    return;
  }
  db->pending_install_owner = next_owner;
  arts_db_lazy_start_invalidate_round(cache, next_owner);
  /* Pipeline: the popped next_owner is the genuine next owner — PROCEED it so
   * it overlaps its RW acquire with the in-flight initial invalidate round. */
  arts_send_db_ownership_proceed(next_owner, cache->db_guid);
}

void arts_db_ownership_return(struct arts_db_cache_s *cache) {
  /* The lazy protocol transfers ownership owner→owner via TRANSFER_OWNERSHIP
   * and never sends RELEASE_OWNERSHIP to home, so this body is unreachable in
   * the lazy build. */
  (void)cache;
}

/* ===== Lazy INVALIDATE_NOTICE handler (pure body, DIRECT-call) ====== */

/* Pure (item, args) body.  The lazy protocol does NOT route INVALIDATE through
 * the OoO engine
 * (its engine slot is an inert no-op): the invalidate target is always the
 * rw_holder, whose CACHE the requester lazy-installs before it ever sends
 * OWNERSHIP_REQUEST, so the cache is present and the wire dispatcher /
 * self-send shortcut call this body directly (guarded by assert(cache !=
 * NULL)).  cache is the FIRST member of arts_db_s (offset 0), so the item_v
 * handed in IS the cache.
 *
 * The cache being present does NOT mean the ownership sentinel (+1) is present:
 * the owner's TRANSFER_OWNERSHIP sentinel (+1) and this home->owner INVALIDATE
 * (-1, sent for the NEXT requester) are two messages to the SAME rank that can
 * reorder under multiple receiver threads.  So writer_count uses the same
 * commutative signed scheme as the eager protocol — TRANSFER adds +1,
 * INVALIDATE/release subtract a signed -1, and only the decrement that drives
 * writer_count from a positive value to EXACTLY 0 ships TRANSFER_OWNERSHIP;
 * a transient 0 -> -1 (an INVALIDATE racing ahead of its sentinel) reads
 * rest != 0 and does NOT ship.  Order-independent. */
void arts_handler_db_ownership_invalidate(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_ownership_invalidate_s *a =
      (struct arts_ooo_args_db_ownership_invalidate_s *)args_v;
  /* Sentinel withdrawal (writer_count -= 1).  Home's invalidate_in_flight gate
   * sends AT MOST ONE INVALIDATE_NOTICE to this rank per transfer round, after
   * rw_holder has been advanced to a rank that already holds the sentinel (+1).
   * The decrement that drives writer_count to 0 is the unique actor that
   * performs the ownership transfer; while local writers are still active
   * (rest > 0) the last release_rw drives it instead.
   *
   * Lazy: publish the transfer target BEFORE withdrawing the sentinel.  This
   * ordering is the dedup (no separate transfer_pending flag): a concurrent
   * release_rw that observes rest==0 is guaranteed to see incoming_new_owner
   * already published, so exactly one of {this handler, the last releaser}
   * ships.  Only one INVALIDATE_NOTICE is in flight per round (home baton
   * gate), so there is no concurrent writer to incoming_new_owner. */
  cache->incoming_new_owner = a->new_owner_rank;
  /* Signed, commutative (same scheme as the eager protocol).  Ship ONLY on the
   * positive->0 edge: */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1);
  if (rest != 0) {
    /* rest > 0: local writers still active — the last release_rw, seeing
     * rest==0 with incoming_new_owner already published, ships.  rest < 0: this
     * INVALIDATE raced ahead of its TRANSFER sentinel (multi-receiver reorder);
     * transient, do NOT ship — the TRANSFER's +1 (and drain/release) drive the
     * count back through exactly 0, and that decrement ships. */
    return;
  }
  /* rest == 0: we are the unique transfer actor. */
  arts_db_lazy_send_ownership_response(cache);
}

/* ===== Lazy REDIRECT_RO handler (owner side, moved from handlers.c) === */

/* Cat-C pure body (REDIRECT_RO, owner side).  The wire dispatcher / self-send
 * shortcut has already looked the owner-side db_s up with a held ref and passes
 * it as item_v (cache is its FIRST member, offset 0).  No lookup/NULL-check
 * here — the dispatcher's MISS branch sends DESTROY_NOTIFY to the requester (DB
 * destroyed / not yet installed on this rank) so the requester's parked RO
 * waiter wakes and observes DB_DESTROYED rather than hanging. */
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
    /* No buffer installed yet (pre-publication or sentinel DB).
     * Respond with version=0, no data — requester's RO waiter fires
     * with undefined content (per spec). */
    arts_send_db_snapshot_response(requester, a->db_guid, /*version=*/0,
                                   edt_guid, slot, /*data=*/NULL,
                                   /*data_size=*/0);
    return;
  }
  /* Invariant: last_sent_version is created at ownership-install
   * (TRANSFER_OWNERSHIP, retained permanently thereafter).  The INITIAL
   * owner (the creator, which never received a transfer) has no map yet, so
   * lazily create it on its first served REDIRECT — otherwise the producer-on-
   * home + RO-consumers-elsewhere DAG would be served no-data (NULL/stale).
   * Single-actor here (the redirect handler runs on the network thread). */
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

/* ===== Lazy wire senders (moved from coherence/senders.c) ========= */

/* Lazy OWNERSHIP_RESPONSE = TRANSFER_OWNERSHIP: carries the serialized
 * last_sent_version map + buffer payload. */
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

void arts_send_db_ownership_response_ack(unsigned int home_rank,
                                         arts_guid_t db_guid,
                                         uint64_t version) {
  struct arts_msg_install_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_RESPONSE_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  if (home_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop.
     * HIT advances the transfer round on the ref-pinned home db_s; MISS (DB
     * destroyed) silently drops. */
    struct arts_db_ownership_response_ack_args_s args = {.db_guid = db_guid,
                                                         .version = version};
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_ownership_response_ack(db, &args);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_ownership_confirm(unsigned int new_owner_rank,
                                    arts_guid_t db_guid) {
  struct arts_msg_ownership_confirm_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_CONFIRM);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (new_owner_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop.
     * HIT runs the confirm body on the ref-pinned db_s; MISS (DB destroyed)
     * silently drops (gated waiters are woken by the destroy fan-out). */
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_ownership_confirm(db, NULL);
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
     * destroyed / not yet installed) sends DESTROY_NOTIFY to the requester so
     * its parked RO waiter wakes and observes DB_DESTROYED. */
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
