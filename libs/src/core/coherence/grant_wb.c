/* SPDX-License-Identifier: Apache-2.0
 *
 * The WB write policy's half of the migrating sentinel grant.
 *
 * coherence/grant.c holds what every grant-bearing arm shares (the sentinel,
 * the home FIFO, the transfer baton, the owner->owner ship).  This TU holds
 * what the WB write policy decides on top of it: a new owner installs but may
 * NOT run until the home confirms the directory flip (CONFIRM -> CONFIRM_ACK).
 * The gate is load-bearing here and absent under WT for one reason — with no
 * bytes at the home, a reader registered after the install would be redirected
 * to whichever rank the directory still names, and that rank's retained copy is
 * stale the instant the new owner stores.  WT has no such window because the
 * home serves reads itself.
 *
 * Nothing here is protocol-specific: VAL and INV link this same TU.  What a
 * transfer means for the ex-holder's copy IS protocol-specific and is the
 * arts_db_grant_note_ex_holder seam.
 */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/directory.h"
#include "arts/coherence/handlers.h"
#include "arts/db.h"
#include "arts/gas/route_table.h"
#include "arts/memory/regpool.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/counter/Preamble.h"

/* Rendezvous continuation state for a transfer whose payload travels
 * one-sided: the packet and the write completion may arrive in either order,
 * so both are paired before the install + commit run. */
struct owner_response_landed_ctx_s {
  arts_shared_ptr_t db_h; /* pin transferred from the handler */
  arts_guid_t db_guid;
  struct arts_db_buffer_s *landing;
  uint64_t version;
  uint64_t data_size;
};
static void owner_response_commit(arts_shared_ptr_t db_h, arts_guid_t db_guid,
                                 uint64_t version);
static void owner_response_landed_cb(void *arg);

/* ===== start the transfer round =======================================
 * The INVALIDATE target is always rw_holder, which the home names only after
 * that rank's cache install (the creator at DB_CREATE, a new owner at
 * CONFIRM).  The target's cache is therefore provably installed by the time
 * the INVALIDATE arrives, so this placement never defers it: the dispatcher
 * and the self-send call the handler body directly. */
void arts_db_owner_start_invalidate_round(
    struct arts_db_cache_s *cache, unsigned int new_owner,
    const struct arts_rdzv_landing_s *new_owner_rdzv) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  unsigned int current_owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  /* INVALIDATE target is always rw_holder, which home publishes only after that
   * rank's cache install (creator at DB_CREATE, or the new owner at
   * CONFIRM).  So the target's cache is provably already installed when the
   * INVALIDATE arrives — the WB write policy never defers INVALIDATE; do not
   * route it through dispatch_or_defer (the dispatcher / self-send call the
   * handler body directly, guarded by assert(cache != NULL)). */
  arts_send_db_grant_invalidate(current_owner, cache->db_guid, new_owner,
                                    new_owner_rdzv);
}


void arts_handler_db_grant_response(void *payload, size_t size) {
  struct arts_msg_grant_response_packet_s *hdr =
      (struct arts_msg_grant_response_packet_s *)payload;
  arts_guid_t db_guid = hdr->db_guid;

  /* Pin the db_s for the whole handler body (map deserialize + buffer install +
   * snapshot/OoO drains + writer_count RMW): the embedded cache is its FIRST
   * member (offset 0), so one cb ref keeps it alive against a concurrent
   * DESTROY on another receiver thread.  Released on every return path. */
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    db_h = arts_db_cache_stub_install(db_guid, /*db_size=*/0);
    db = (struct arts_db_s *)arts_shared_get(db_h);
    if (db == NULL) {
      arts_shared_release(&db_h);
      /* Consume any in-flight one-sided payload so the pairing table stays
       * leak-free (the landing frees on arrival). */
      arts_db_rdzv_discard_landing(hdr->rdzv_txid, hdr->rdzv_cookie);
      return; /* DB destroyed before we became owner — drop. */
    }
  }
  struct arts_db_cache_s *cache = &db->cache;

  /* A hinted first touch may reach its first grant with the size still
   * unlearned (no CTS leg ran); the response's data_size is the sender's
   * descriptor size, valid payload or not. */
  if (cache->db_size == 0 && hdr->data_size > 0) {
    cache->db_size = hdr->data_size;
  }

  /* Wire layout: header | map (count pairs, inline).  The buffer payload does
   * not trail on the wire — it travels one-sided into our advertised landing
   * (a same-rank self-transfer still trails it inline). */
  char *map_start = (char *)payload + sizeof(*hdr);
  size_t map_size =
      (sizeof(uint32_t) * 2) + ((size_t)hdr->map_entry_count *
                                sizeof(struct arts_msg_rank_version_pair_s));

  /* Reconstruct the owner-side dedup map so this rank can skip redundant
   * SNAPSHOT_RESPONSE sends to readers that already hold a sufficiently fresh
   * copy.  Must happen HERE (the map bytes live in the packet, which is freed
   * after dispatch) — harmless ahead of the install: the map is only consulted
   * once this rank serves REDIRECTs, which requires the ownership this round
   * is still delivering. */
  if (cache->cached_version != NULL) {
    arts_rank_u64_map_destroy(cache->cached_version);
  }
  cache->cached_version = arts_rank_u64_map_deserialize(
      map_start, map_size, arts_global_rank_count);

  if (hdr->rdzv_txid != 0) {
    /* Payload travels one-sided: pair this packet with the write completion
     * (either order) and install+commit when both are in. */
    struct owner_response_landed_ctx_s *ctx =
        (struct owner_response_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->db_guid = db_guid;
    ctx->landing = (struct arts_db_buffer_s *)(uintptr_t)hdr->rdzv_cookie;
    ctx->version = hdr->version;
    ctx->data_size = hdr->data_size;
    arts_net_rdzv_expect(hdr->rdzv_txid, owner_response_landed_cb, ctx);
    return;
  }

  /* No PUT: recycle the unused landing (echoed for a data-less transfer),
   * install any inline same-rank payload, and commit. */
  if (hdr->rdzv_cookie != 0) {
    arts_db_buf_landing_recycle(
        cache, (struct arts_db_buffer_s *)(uintptr_t)hdr->rdzv_cookie);
  }
  char *data_start = map_start + map_size;
  size_t data_size = size - sizeof(*hdr) - map_size;
  if (data_size > 0) {
    arts_db_buf_install(cache, hdr->version, data_start, data_size);
    if (cache->db_size == 0) {
      cache->db_size = data_size;
    }
  }
  owner_response_commit(db_h, db_guid, hdr->version); /* releases db_h */
}

/* Post-install tail of the GRANT_RESPONSE handler — everything that must
 * run only once the transferred bytes are in place.  db_h is the caller's pin
 * on the db_s; consumed (released) here. */
static void owner_response_commit(arts_shared_ptr_t db_h, arts_guid_t db_guid,
                                 uint64_t version) {
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  struct arts_db_cache_s *cache = &db->cache;

  /* ADD the ownership sentinel (+1) PLUS a transient DRAIN GUARD (+1) in a
   * single atomic op (jump 0->2, no intermediate 1 a racing INVALIDATE could
   * catch at 0) — the same scheme the WT GRANT uses.  The guard keeps
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
  cache->grant_unconfirmed = 1u;
  /* Sentinel (+1) + drain guard (+1), single op (0->2). The guard is held until
   * the CONFIRM_ACK handler, so a next-round INVALIDATE racing ahead of
   * CONFIRM_ACK cannot zero the count and ship before this rank has used its
   * ownership. */
  arts_atomic_add(&cache->writer_count, 2u);

  /* RW drain is DEFERRED to the CONFIRM_ACK handler (home has not flipped
   * rw_holder to us yet). The snapshot + OoO drains stay: a parked RO waiter
   * served here gets the transferred (pre-write) version, which is correct, and
   * a reordered INVALIDATE deferred on a previously-missing cache replays now.
   */
  arts_db_drain_pending_snapshot(cache);
#ifdef ARTS_RO_COMBINING_LIVE
  /* Read waiters batched while this rank had no local copy resume here
   * against the transferred buffer instead of re-fetching remotely. */
  arts_db_ro_combine_grant_drain(cache);
#endif
  arts_ooo_drain_guid(db_guid);

  /* No racing INVALIDATE can have reached us yet: home targets this rank as an
   * INVALIDATE recipient only after the rw_holder flip, which needs this
   * CONFIRM. So incoming_new_owner == NONE here — send CONFIRM
   * unconditionally. grant_req_in_flight stays 1 until CONFIRM_ACK so fresh
   * RW acquires in the gate window park without issuing a duplicate request. */
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  arts_send_db_grant_confirm(home_rank, db_guid, version);

  arts_shared_release(&db_h);
}

/* Rendezvous continuation: transfer payload fully landed — install without a
 * copy, then run the commit tail.  Fires on {packet, write-completion}
 * pairing, either arrival order, on a dispatch-path progress thread. */
static void owner_response_landed_cb(void *arg) {
  struct owner_response_landed_ctx_s *ctx =
      (struct owner_response_landed_ctx_s *)arg;
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(ctx->db_h);
  arts_db_buf_install_landed(&db->cache, ctx->version, ctx->landing,
                             ctx->data_size);
  owner_response_commit(ctx->db_h, ctx->db_guid, ctx->version); /* releases */
  arts_free(ctx);
}

/* ===== WB CONFIRM handler (home A) =============================== */

/* Cat-C pure body (CONFIRM, home side).  The wire dispatcher / self-send
 * shortcut has already looked the home db_s up with a held ref and passes it as
 * item_v (cache is its FIRST member, offset 0).  No lookup/NULL-check here —
 * the dispatcher's MISS branch SILENTLY DROPS (DB destroyed).  args_v is unused
 * (the new owner is read from db->pending_install_owner, published by the baton
 * holder; CONFIRM only confirms the install completed). */
void arts_handler_db_grant_confirm(void *item_v, void *args_v) {
  (void)args_v;
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_s *db = arts_db_of_cache(cache);

  /* Publish the new rw_holder (visible to the read redirect path). */
  unsigned int new_owner = db->pending_install_owner;
  unsigned int prev_holder =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  atomic_store_explicit(&db->rw_holder, new_owner, memory_order_release);
  /* Protocol seam: the ex-holder keeps the bytes it wrote.  Whether that
   * retained copy needs registering is the protocol's question, not the
   * placement's. */
  arts_db_grant_note_ex_holder(db, prev_holder);

  /* The directory now names the new owner: tell it to run its gated RW EDTs.
   * If a next requester is queued, the round advances in the SAME message —
   * the CONFIRM_ACK piggybacks the next transfer target, and the new owner's
   * handler applies the INVALIDATE effect (publish incoming_new_owner +
   * withdraw the sentinel) itself.  This merges what used to be two separate
   * home→new_owner messages (CONFIRM_ACK + INVALIDATE) into one, removing the
   * CONFIRM_ACK↔INVALIDATE reorder window.  With no pending requester the ack
   * carries ARTS_NO_PENDING_OWNER and the baton is released below. */
  unsigned int piggyback = ARTS_NO_PENDING_OWNER;
  struct arts_rdzv_landing_s piggyback_rdzv = {0, 0, 0, 0};
  {
    unsigned int next_owner;
    if (arts_home_grantreq_queue_pop(&db->pending_rw, &next_owner,
                                    &piggyback_rdzv)) {
      db->pending_install_owner = next_owner;
      piggyback = next_owner;
      /* No early PROCEED: the next owner's RW cursor is advanced by the CURRENT
       * owner at transfer-commit (writer_count->0 in
       * arts_db_send_grant_response) — the earliest moment its
       * acquisition of this db is irrevocably committed.  PROCEEDing here
       * (round start, before the current owner has released) could strand it in
       * a hold-and-wait cycle. */
    }
  }
  arts_send_db_grant_confirm_ack(new_owner, cache->db_guid, piggyback,
                                     &piggyback_rdzv);
  if (piggyback != ARTS_NO_PENDING_OWNER) {
    /* Round advanced via the merged ack; the baton stays held until the new
     * owner releases (transfer-commit), as in the standalone-INVALIDATE case.
     */
    return;
  }

  /* No pending requester — release the baton (with the freshly-enqueued-racer
   * recheck retry loop). */
  while (1) {
    /* Release the baton. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    /* Dekker-style publication: the baton-release store must be globally
     * visible BEFORE the emptiness re-check loads, or a requester that
     * pushed and lost its baton CAS inside the window is missed — a plain
     * release-store followed by loads permits exactly that StoreLoad
     * reordering. */
    atomic_thread_fence(memory_order_seq_cst);
    /* Re-check for a freshly-enqueued requester that raced the baton
     * release.  If the queue is still empty, we're done. */
    if (arts_home_grantreq_queue_empty(&db->pending_rw)) {
      return;
    }
    /* There is a new requester; try to re-acquire the baton. */
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      /* Another GRANT_REQUEST handler already picked up the baton (race);
       * that thread will drain the queue. */
      return;
    }
    INCREMENT_NUM_GRANT_BATON_RECLAIM_BY(1);
    /* Re-acquired the baton: pop the racer and start a fresh round.  This path
     * starts AFTER the just-confirmed owner is already running (no in-flight
     * CONFIRM_ACK to piggyback on), so it issues a STANDALONE INVALIDATE to the
     * current rw_holder, exactly like the first-round request-handler path. */
    unsigned int next_owner;
    struct arts_rdzv_landing_s next_rdzv;
    if (arts_home_grantreq_queue_pop(&db->pending_rw, &next_owner,
                                    &next_rdzv)) {
      db->pending_install_owner = next_owner;
      arts_db_owner_start_invalidate_round(cache, next_owner, &next_rdzv);
      return;
    }
    /* The racer was already consumed by whoever we contended with; loop to
     * release and recheck. */
  }
}

/* ===== WB CONFIRM_ACK handler (new owner C) ==================== */

/* Cat-C pure body (CONFIRM_ACK, new-owner side). Home has flipped
 * rw_holder to this rank; it is now safe for this rank's RW EDTs to run and
 * make their writes observable. Drain the RW waiters deferred at TRANSFER,
 * clear the gate, apply the piggybacked invalidate effect (if the round
 * advanced — see arts_handler_db_grant_confirm), and remove the drain
 * guard (the relocated 0-edge ship-check).
 *
 * args_v is the CONFIRM_ACK packet: its new_owner_rank carries the next
 * transfer target when the round advanced, or ARTS_NO_PENDING_OWNER for a
 * plain ack.  Merging the INVALIDATE into this message removes the former
 * CONFIRM_ACK↔INVALIDATE reorder window: both effects now happen here, in a
 * fixed order, under the drain guard.
 */
void arts_handler_db_grant_confirm_ack(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_msg_grant_confirm_ack_packet_s *p =
      (struct arts_msg_grant_confirm_ack_packet_s *)args_v;

  /* Open the gate: fresh RW acquires may now take the fast path, and the
   * coalescing flag is released so a future round can re-issue. */
  cache->grant_unconfirmed = 0u;
  cache->grant_req_in_flight = 0u;

  /* Drain the RW waiters that the GRANT_RESPONSE handler deferred (this is the work
   * moved out of arts_handler_db_grant_response). */
  arts_db_drain_pending_rw_after_grant(cache, /*version=*/0,
                                       /*has_next=*/false);

  /* Piggybacked INVALIDATE effect (merged from the former standalone
   * GRANT_INVALIDATE).  Mirror the INVALIDATE handler's discipline: publish
   * the transfer target BEFORE withdrawing the sentinel (the publish-before-sub
   * is the dedup against a concurrent release_rw that observes the 0-edge).
   * Do NOT ship on this sub's edge: the drain guard (+1) is still held, so the
   * count is >= 1 here and the 0-crossing is deferred to the guard-removal
   * below — exactly the invariant the held guard always provided. */
  if (p != NULL && p->new_owner_rank != ARTS_NO_PENDING_OWNER) {
    cache->incoming_new_owner_rdzv.addr = p->new_owner_rdzv.addr;
    cache->incoming_new_owner_rdzv.key = p->new_owner_rdzv.key;
    cache->incoming_new_owner_rdzv.txid = p->new_owner_rdzv.txid;
    cache->incoming_new_owner_rdzv.cookie = p->new_owner_rdzv.cookie;
    cache->incoming_new_owner = p->new_owner_rank;
    arts_atomic_sub(&cache->writer_count, 1); /* sentinel withdrawal */
  }

  /* Remove the drain guard held across the CONFIRM→CONFIRM_ACK round trip. If
   * the round advanced (sentinel withdrawn above) and that drives the count to
   * 0 with no local writer remaining, we are the unique actor that ships
   * GRANT_RESPONSE to the next owner.  Otherwise this rank retains
   * ownership and its drained EDTs ship on their own release 0-edge. */
  if ((int)arts_atomic_sub(&cache->writer_count, 1) == 0 &&
      cache->incoming_new_owner != ARTS_NO_PENDING_OWNER) {
    arts_db_send_grant_response(cache);
  }
}

/* ===== Per-model wire-handler bodies =============================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_SNAPSHOT_REQUEST]): the OoO engine
 * has already acquired the home db_s and pinned a ref across this call (cache
 * is its FIRST member), so there is no lookup / NULL-check / defer here. */

/* ===== Ownership-round seams (called from coherence/grant.c) ==
 * family→protocol: the ownership-family GRANT_REQUEST handler
 * delegates the WT/WB-divergent steps here. */

void arts_db_start_grant_round(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester) {
  (void)requester;
  /* WB: pop the OLDEST requester (FIFO) to be the transfer target and
   * embed its rank in the GRANT_INVALIDATE so the current holder ships
   * GRANT_RESPONSE directly, without a home round-trip.
   * pending_install_owner is only written by the baton holder (single
   * writer invariant), so no atomic needed. */
  unsigned int next_owner;
  struct arts_rdzv_landing_s next_rdzv;
  while (!arts_home_grantreq_queue_pop(&db->pending_rw, &next_owner,
                                       &next_rdzv)) {
    /* Empty despite our own push: an earlier round already served it (rounds
     * can complete between the push and this claim).  Release the baton with
     * the SAME re-check discipline as the round close: a requester that
     * pushed and lost the baton race while we held it would otherwise stay
     * queued forever with no baton holder to serve it. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    /* Dekker-style publication: the baton-release store must be globally
     * visible BEFORE the emptiness re-check loads, or a requester that
     * pushed and lost its baton CAS inside the window is missed — a plain
     * release-store followed by loads permits exactly that StoreLoad
     * reordering. */
    atomic_thread_fence(memory_order_seq_cst);
    if (arts_home_grantreq_queue_empty(&db->pending_rw)) {
      return;
    }
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      return; /* another claimant owns the queue now */
    }
    INCREMENT_NUM_GRANT_BATON_RECLAIM_BY(1);
  }
  db->pending_install_owner = next_owner;
  arts_db_owner_start_invalidate_round(cache, next_owner, &next_rdzv);
  /* No early PROCEED here (see arts_db_send_grant_response): the next
   * owner's RW cursor advances at transfer-commit, not at round start. */
}

/* ===== WB GRANT_INVALIDATE handler (pure body, DIRECT-call) ====== */

/* Pure (item, args) body.  The WB write policy does NOT route INVALIDATE through
 * the OoO engine
 * (its engine slot is an inert no-op): the invalidate target is always the
 * rw_holder, whose CACHE the requester stub-installs before it ever sends
 * GRANT_REQUEST, so the cache is present and the wire dispatcher /
 * self-send shortcut call this body directly (guarded by assert(cache !=
 * NULL)).  cache is the FIRST member of arts_db_s (offset 0), so the item_v
 * handed in IS the cache.
 *
 * When this INVALIDATE arrives the ownership sentinel (+1) is already
 * installed: the post-install rw_holder flip (CONFIRM-driven) advances
 * rw_holder to a rank only after its GRANT install, so home sends this notice
 * strictly after that install — GRANT(+sentinel) precedes INVALIDATE(-1), they
 * never reorder.  The install's sentinel+guard (+2) further holds writer_count
 * >= 0 against the next round's INVALIDATE on another receiver thread.
 * writer_count is therefore non-negative; only the decrement that drives it
 * from a positive value to EXACTLY 0 ships the owner->owner transfer, and the
 * (int) cast is defensive. */
void arts_handler_db_grant_invalidate(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_grant_invalidate_s *a =
      (struct arts_ooo_args_db_grant_invalidate_s *)args_v;
  /* Sentinel withdrawal (writer_count -= 1).  Home's invalidate_in_flight gate
   * sends AT MOST ONE GRANT_INVALIDATE to this rank per transfer round, after
   * rw_holder has been advanced to a rank that already holds the sentinel (+1).
   * The decrement that drives writer_count to 0 is the unique actor that
   * performs the ownership transfer; while local writers are still active
   * (rest > 0) the last release_rw drives it instead.
   *
   * WB: publish the transfer target BEFORE withdrawing the sentinel.  This
   * ordering is the dedup (no separate transfer_pending flag): a concurrent
   * release_rw that observes rest==0 is guaranteed to see incoming_new_owner
   * already published, so exactly one of {this handler, the last releaser}
   * ships.  Only one GRANT_INVALIDATE is in flight per round (home baton
   * gate), so there is no concurrent writer to incoming_new_owner. */
  cache->incoming_new_owner_rdzv = a->new_owner_rdzv;
  cache->incoming_new_owner = a->new_owner_rank;
  /* Ship ONLY on the positive->0 edge (writer_count is non-negative; see the
   * handler header). */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1);
  if (rest != 0) {
    /* rest > 0: local writers still active — the last release_rw, seeing
     * rest==0 with incoming_new_owner already published, ships instead.  (The
     * (int) cast is defensive: the post-install flip + install guard keep the
     * count non-negative, so rest < 0 does not occur.) */
    return;
  }
  /* rest == 0: we are the unique transfer actor. */
  arts_db_send_grant_response(cache);
}

/* ===== WB SNAPSHOT_REDIRECT handler (owner side, moved from handlers.c) === */

/* Cat-C pure body (SNAPSHOT_REDIRECT, owner side).  The wire dispatcher / self-send
 * shortcut has already looked the owner-side db_s up with a held ref and passes
 * it as item_v (cache is its FIRST member, offset 0).  No lookup/NULL-check
 * here — the dispatcher's MISS branch sends DESTROY_NOTIFY to the requester (DB
 * destroyed / not yet installed on this rank) so the requester's parked RO
 * waiter wakes and observes DB_DESTROYED rather than hanging. */

/* ===== WB wire senders (moved from coherence/senders.c) =========
 * The GRANT_RESPONSE wire sender + the owner→owner ship are now shared
 * (coherence/grant.c); the CONFIRM sender is shared too.  Only the
 * WB-only CONFIRM_ACK + SNAPSHOT_REDIRECT senders remain here. */

void arts_send_db_grant_confirm_ack(
    unsigned int new_owner_rank, arts_guid_t db_guid,
    unsigned int piggyback_new_owner,
    const struct arts_rdzv_landing_s *piggyback_rdzv) {
  struct arts_msg_grant_confirm_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_GRANT_CONFIRM_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.new_owner_rank = piggyback_new_owner;
  if (piggyback_rdzv != NULL) {
    p.new_owner_rdzv.addr = piggyback_rdzv->addr;
    p.new_owner_rdzv.key = piggyback_rdzv->key;
    p.new_owner_rdzv.txid = piggyback_rdzv->txid;
    p.new_owner_rdzv.cookie = piggyback_rdzv->cookie;
  } else {
    p.new_owner_rdzv = (struct arts_msg_rdzv_landing_s){0, 0, 0, 0};
  }
  if (new_owner_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop.
     * HIT runs the confirm_ack body on the ref-pinned db_s, passing the packet
     * so the piggybacked invalidate effect (if any) is applied; MISS (DB
     * destroyed) silently drops (gated waiters are woken by the destroy
     * fan-out). */
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_grant_confirm_ack(db, &p);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)new_owner_rank, (char *)&p, sizeof(p));
}

