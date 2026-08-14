/* SPDX-License-Identifier: Apache-2.0
 *
 * The HOME placement's half of the migrating sentinel grant.
 *
 * coherence/grant.c holds what every grant-bearing arm shares (the sentinel,
 * the home FIFO, the transfer baton, the owner->owner ship).  This TU holds
 * what the HOME placement decides on top of it: a new owner installs, drains,
 * and RUNS immediately — there is no confirm gate, because the home serves
 * reads, so no reader can be sent to a stale copy while the directory catches
 * up.  (The OWNER placement gates on CONFIRM_ACK for exactly that reason; see
 * its own TU.)
 *
 * Nothing here is protocol-specific — RCU and MSI link this same TU.  What a
 * transfer means for the ex-holder's copy IS protocol-specific, so it is a
 * seam: arts_db_grant_note_ex_holder.  RCU has nothing to do (its readers
 * re-check a version at every acquire, so a retained copy is harmless); MSI
 * must register the ex-holder as a sharer, or the new owner's first release
 * would leave a live copy uninvalidated.
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/directory.h"
#include "arts/coherence/handlers.h"
#include "arts/db.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== Ownership-round seams (called from coherence/grant.c) ==
 * family→protocol: the ownership-family OWNERSHIP_REQUEST / RELEASE_OWNERSHIP
 * handlers delegate the HOME/OWNER-divergent steps here. */

void arts_db_start_grant_round(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester) {
  (void)requester;
  /* Pop the OLDEST requester (FIFO) to be the transfer target, publish it as
   * pending_install_owner, and INVALIDATE the current holder carrying the new
   * owner so the holder ships the owner→owner OWNERSHIP_RESPONSE directly.
   * pending_install_owner is written only by the baton holder (single writer),
   * so no atomic.  Structurally identical to the OWNER start round; the HOME /
   * OWNER divergence is only the drain point in the response/confirm handlers.
   */
  unsigned int next_owner;
  struct arts_rdzv_landing_s next_rdzv;
  if (!arts_home_grantreq_queue_pop(&db->pending_rw, &next_owner, &next_rdzv)) {
    /* Defensive: we just pushed, so empty is impossible under correct usage.
     * Release the baton and return. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    return;
  }
  db->pending_install_owner = next_owner;
  unsigned int current_owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  /* Credit the outgoing holder as a sharer HERE, not at the CONFIRM: it keeps
   * the bytes it wrote, and under HOME the incoming owner runs the moment it
   * installs — there is no confirm gate — so it can close a release round
   * before the home ever processes the CONFIRM.  A roster credited only then
   * would miss that round, leaving the ex-holder on bytes from a write window
   * that has already closed, with nothing left to retire them.  Deciding the
   * transfer strictly precedes the new owner existing, so crediting here
   * cannot be too late. */
  arts_db_grant_note_ex_holder(db, current_owner);
  arts_send_db_grant_invalidate(current_owner, cache->db_guid, next_owner,
                                    &next_rdzv);
}

/* ===== Eager OWNERSHIP_RESPONSE handler (new owner C) ================ */

/* HOME converged onto the OWNER owner→owner transfer, with the CRITICAL
 * divergence: HOME drains pending_rw + runs its RW EDTs IMMEDIATELY here (no
 * CONFIRM_ACK gate — home serves RO, so there is no stale-RO window to close),
 * then sends CONFIRM to home (flips rw_holder + advances the next round). */
/* Post-install tail of the OWNERSHIP_RESPONSE (GRANT) handler — everything
 * that must run only once the transferred bytes are in place.  db_h is the
 * caller's pin on the db_s; consumed (released) here. */
static void home_response_commit(arts_shared_ptr_t db_h, arts_guid_t db_guid,
                                  uint64_t version) {
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  struct arts_db_cache_s *cache = &db->cache;

  /* Sentinel(+1) + drain guard(+1), single op (0->2): the guard keeps
   * writer_count >= 1 across the per-waiter +1 drain so a commutative
   * INVALIDATE(-1) cannot zero the count mid-drain and ship prematurely.  The
   * 0-crossing is deferred to guard-removal below. */
  arts_atomic_add(&cache->writer_count, 2u);
  cache->grant_req_in_flight = 0;

  /* HOME drains + runs NOW (no confirm-ack gate): home serves RO so there is
   * no stale-RO window.  Then tell home we installed (CONFIRM), which flips
   * rw_holder + advances the next round. */
  arts_db_drain_pending_rw_after_grant(cache, version, /*has_next=*/false);
  arts_db_drain_pending_snapshot(cache);
#ifdef ARTS_RO_COMBINING_LIVE
  /* Read waiters batched while this rank had no local copy resume here
   * against the transferred buffer instead of re-fetching remotely. */
  arts_db_ro_combine_grant_drain(cache);
#endif
  arts_ooo_drain_guid(db_guid);

  unsigned int home_rank = arts_guid_get_rank(db_guid);
  arts_send_db_grant_confirm(home_rank, db_guid, version);

  /* Remove the drain guard: the relocated 0-edge ship-check.  If a next-round
   * INVALIDATE already withdrew the sentinel and no local writer remains, we
   * are the unique actor that ships the next transfer. */
  if ((int)arts_atomic_sub(&cache->writer_count, 1) == 0 &&
      cache->incoming_new_owner != ARTS_NO_PENDING_OWNER) {
    arts_db_send_grant_response(cache);
  }

  arts_shared_release(&db_h);
}

/* Rendezvous continuation: the transfer payload has fully landed in the
 * advertised landing buffer ("imm seen => landing valid"); install it without
 * a copy and run the commit tail.  Fires on {response packet, write
 * completion} pairing, in either arrival order, on a dispatch-path progress
 * thread. */
struct home_response_landed_ctx_s {
  arts_shared_ptr_t db_h; /* pin transferred from the handler */
  arts_guid_t db_guid;
  struct arts_db_buffer_s *landing;
  uint64_t version;
  uint64_t data_size;
};

static void home_response_landed_cb(void *arg) {
  struct home_response_landed_ctx_s *ctx =
      (struct home_response_landed_ctx_s *)arg;
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(ctx->db_h);
  arts_db_buf_install_landed(&db->cache, ctx->version, ctx->landing,
                             ctx->data_size);
  home_response_commit(ctx->db_h, ctx->db_guid, ctx->version); /* releases */
  arts_free(ctx);
}

void arts_handler_db_grant_response(void *payload, size_t size) {
  struct arts_msg_grant_response_packet_s *hdr =
      (struct arts_msg_grant_response_packet_s *)payload;
  arts_guid_t db_guid = hdr->db_guid;

  /* Pin the db_s across the install + commit (and, on the rendezvous path,
   * across the pairing wait): the embedded cache is its FIRST member (offset
   * 0), so one cb ref keeps it alive against a concurrent DESTROY on another
   * receiver thread.  Released by the commit tail. */
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

  /* Wire layout: header | map (count pairs, inline).  HOME ignores the map
   * (it dedups RO via home->cached_version), but parses past it. */
  char *map_start = (char *)payload + sizeof(*hdr);
  size_t map_size =
      (sizeof(uint32_t) * 2) + ((size_t)hdr->map_entry_count *
                                sizeof(struct arts_msg_rank_version_pair_s));

  if (hdr->rdzv_txid != 0) {
    /* The payload travels one-sided: pair this packet with the write
     * completion (either may arrive first) and install+commit when both are
     * in.  The landing is our own buffer, named by the echoed cookie. */
    struct home_response_landed_ctx_s *ctx =
        (struct home_response_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->db_guid = db_guid;
    ctx->landing =
        (struct arts_db_buffer_s *)(uintptr_t)hdr->rdzv_cookie;
    ctx->version = hdr->version;
    ctx->data_size = hdr->data_size;
    arts_net_rdzv_expect(hdr->rdzv_txid, home_response_landed_cb, ctx);
    return;
  }

  /* No PUT: recycle the unused landing (echoed back for a data-less
   * transfer), install any inline same-rank payload, and commit. */
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
  home_response_commit(db_h, db_guid, hdr->version); /* releases db_h */
}

/* ===== Eager CONFIRM handler (home A) =============================== */

/* Cat-C pure body (CONFIRM, home side).  The wire dispatcher / self-send
 * shortcut has already looked the home db_s up with a held ref and passes it as
 * item_v (cache is its FIRST member).  args_v is unused — the new owner is read
 * from db->pending_install_owner (published by the baton holder).  HOME does
 * NOT send CONFIRM_ACK (the new owner already drained + ran at
 * OWNERSHIP_RESPONSE); it only flips rw_holder + advances the next round. */
void arts_handler_db_grant_confirm(void *item_v, void *args_v) {
  (void)args_v;
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_s *db = arts_db_of_cache(cache);

  unsigned int new_owner = db->pending_install_owner;
  unsigned int prev_holder =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  atomic_store_explicit(&db->rw_holder, new_owner, memory_order_release);
  /* Protocol seam: the ex-holder keeps the bytes it wrote.  Whether that
   * retained copy needs registering is the protocol's question, not the
   * placement's. */
  arts_db_grant_note_ex_holder(db, prev_holder);

  /* Drain-or-release retry loop: start the next transfer round if there are
   * pending_rw requests, otherwise release the baton (same loop as OWNER, minus
   * the CONFIRM_ACK send). */
  while (1) {
    unsigned int next_owner;
    struct arts_rdzv_landing_s next_rdzv;
    if (arts_home_grantreq_queue_pop(&db->pending_rw, &next_owner, &next_rdzv)) {
      db->pending_install_owner = next_owner;
      unsigned int current =
          atomic_load_explicit(&db->rw_holder, memory_order_acquire);
      arts_send_db_grant_invalidate(current, cache->db_guid, next_owner,
                                        &next_rdzv);
      return;
    }
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    if (arts_home_grantreq_queue_empty(&db->pending_rw)) {
      return;
    }
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      return;
    }
  }
}

/* ===== Eager INVALIDATE_NOTICE handler (pure body) =================== */

/* Pure (cache, args) body.  The wire dispatcher / self-send shortcut has
 * already looked the cache up (the target is rw_holder, published only at the
 * post-install CONFIRM owner-swap, so the cache is provably installed when
 * INVALIDATE arrives) and passes the db_s as item_v — cache is its FIRST member
 * (offset 0).  No OoO defer is needed (and none exists for this kind anymore):
 * since rw_holder flips post-install in both placements, the before-install
 * GRANT/INVALIDATE reorder that once forced HOME through the engine cannot
 * occur.  GRANT(+sentinel) strictly precedes INVALIDATE(-1); the install guard
 * (+2) keeps writer_count non-negative even if the next round's INVALIDATE
 * lands mid-install on another receiver thread. */
void arts_handler_db_grant_invalidate(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_grant_invalidate_s *a =
      (struct arts_ooo_args_db_grant_invalidate_s *)args_v;
  /* Publish the transfer target BEFORE withdrawing the sentinel: whichever
   * actor drives writer_count to 0 (this handler, or a concurrent last
   * release_rw) then reads the same new_owner and PROCEEDs it.  Same
   * store-before-decrement discipline the OWNER arm uses for incoming_new_owner;
   * the HOME INVALIDATE carries new_owner (home embeds the FIFO front),
   * making the HOME and OWNER transfer paths structurally identical. */
  cache->incoming_new_owner_rdzv = a->new_owner_rdzv;
  cache->incoming_new_owner = a->new_owner_rank;
  /* Sentinel withdrawal (writer_count -= 1).  Home's invalidate_in_flight gate
   * sends AT MOST ONE INVALIDATE_NOTICE to this rank per transfer round, after
   * rw_holder has been advanced to a rank that already holds the sentinel (+1).
   * The decrement that drives writer_count to 0 is the unique actor that
   * performs the ownership transfer; while local writers are still active
   * (rest > 0) the last release_rw drives it instead.
   *
   * writer_count is non-negative.  The post-install rw_holder flip means this
   * notice targets a rank whose sentinel +1 is already installed (GRANT
   * precedes INVALIDATE); the install guard (+2) holds the count >= 0 if the
   * next round's INVALIDATE lands mid-install on another receiver thread.  It
   * bottoms out at exactly 0, so the (int) cast is defensive — only the
   * decrement that drives writer_count from a positive value to exactly 0 owns
   * the transfer. */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1);
  if (rest == 0) {
    /* The decrement that drives writer_count to exactly 0 is the unique actor
     * that ships the owner→owner transfer to incoming_new_owner (published
     * above, before the sentinel withdrawal). */
    arts_db_send_grant_response(cache);
  }
}

/* Case-D leaf: the HOME placement defers the home-buffer install to the
 * creator's first release_rw (PUBLISH / GRANT path); nothing to do at
 * create. */
