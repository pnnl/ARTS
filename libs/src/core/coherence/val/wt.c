/* SPDX-License-Identifier: Apache-2.0
 *
 * WT write-policy translation unit: defines the WT-specific
 * arts_handler_db_* / arts_db_* bodies directly (CMake links exactly this TU
 * for a VAL+WT build) plus the WT-only wire handlers/senders.
 * Compiled only for ARTS_COHERENCE_PROTOCOL=VAL with
 * ARTS_WRITE_POLICY=WT (selected in libs/src/core/CMakeLists.txt).
 * Contains NO protocol/write-policy preprocessor logic.
 */
#include <semaphore.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/memory/regpool.h" /* arts_regpool_free (orphaned landing) */
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/directory.h"
#include "arts/db.h"
#include "arts/edt.h"             /* arts_edt_dep_t (acquire body) */
#include "arts/gas/guid.h" /* creator slice arithmetic */
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (pin db_s) */
#include "arts/ooo.h"             /* OOO_DB_* args (handler bodies) */
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"       /* ARTS_ERROR */
#include "arts/system/threads.h"     /* arts_global_rank_id */
#include "arts/transport/net.h"   /* arts_transport_send_async */
#include "arts/transport/protocol.h" /* arts_fill_packet_header, MSG_* */
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h" /* pairing ctx */      /* arts_atomic_* */

/* ===== 8-case acquire dispatch (HOME arm) ==========================
 * Whole arts_handler_db_acquire body for the HOME build.  Diverges from OWNER
 * only on the RO-has-local-data predicate (HOME: is_home||is_owner — home
 * always holds current data via synchronous PUBLISH).  The remote-RO,
 * RW-local-fast, and remote-RW paths are the shared helpers
 * (coherence/coherence.c / coherence/grant.c). */
void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *dep = &((arts_edt_dep_t *)arts_get_depv(edt))[slot];
  arts_db_access_mode_t mode = dep->mode;
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  /* writer_count is non-negative (post-install rw_holder flip + install guard),
   * so > 0 means owner and <= 0 means not-owner; the (int) cast is defensive.
   * An owned read here is safe to serve RO from the local buffer. */
  bool is_owner = ((int)arts_atomic_read(&cache->writer_count) > 0);

  if (mode == DB_MODE_RO) {
    if (is_home ||
        is_owner) { /* HOME RO predicate (home holds current data) */
      /* Pre-check the publication signal so an unpublished (v0, installed
       * at create) stable buffer parks WITHOUT taking the EDT's acquire
       * hold — the resume re-derives dep->ptr from the published buffer. */
      bool unpublished = false;
      {
        arts_shared_ptr_t ph = arts_db_buf_acquire(cache);
        struct arts_db_buffer_s *pb =
            (struct arts_db_buffer_s *)arts_shared_get(ph);
        unpublished =
            (pb != NULL &&
             __atomic_load_n(&pb->version, __ATOMIC_ACQUIRE) == 0);
        arts_db_buf_release(&ph);
      }
      dep->ptr = (unpublished && cache->db_size > 0)
                     ? NULL
                     : arts_db_acquire_local(cache);
      if (dep->ptr == NULL && cache->db_size > 0) {
        /* Home before the creator's first PUBLISH: a cross-rank create
         * installs home metadata only — no buffer exists until the creator's
         * release publishes one.  A dependence satisfied before that release
         * (add-dependence satisfies immediately) may legally race here; the
         * RELEASE is the publication point, so PARK on the snapshot reorder
         * buffer — the first publish install drains us and the resume
         * re-derives dep->ptr from the installed buffer.  (db_size == 0 is
         * the sentinel DB: NULL is its defined value.) */
        struct arts_db_snapshot_waiter_s *w =
            (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
        w->edt_guid = edt->guid;
        w->slot = slot;
        w->target_version = 1;
        w->serve = NULL;
        arts_lf_stack_push(&cache->pending_snapshot, &w->link);
        /* Race recovery: an install may have landed between the NULL read
         * and the push — drain (our own node included) so nobody parks
         * forever.  The atomic_exchange drain is single-actor safe. */
        arts_shared_ptr_t rh = arts_db_buf_acquire(cache);
        struct arts_db_buffer_s *rb =
            (struct arts_db_buffer_s *)arts_shared_get(rh);
        bool published =
            (rb != NULL &&
             __atomic_load_n(&rb->version, __ATOMIC_ACQUIRE) > 0);
        arts_db_buf_release(&rh);
        if (published) {
          arts_db_drain_pending_snapshot(cache);
        }
        return;
      }
      arts_db_acquire_resolved(edt, slot);
      return;
    }
    arts_db_acquire_remote_ro(cache, edt->guid, slot); /* parks (SNAPSHOT) */
    return;
  }
  /* RW */
  if (is_owner && arts_db_acquire_rw_local_fast(cache, dep)) {
    arts_db_acquire_resolved(edt, slot); /* data here, writer_count bumped */
    return;
  }
  arts_db_acquire_remote_rw(cache, edt->guid,
                            slot); /* parks (GRANT_REQUEST) */
}

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW;
}

/* ===== release_rw (WT arm) ========================================
 * The WT write policy keeps home's RO copy fresh with a pure synchronous
 * PUBLISH on every non-home release (home owners already hold the canonical
 * buffer).  Ownership transfer is a SEPARATE owner→owner ship
 * (arts_db_send_grant_response), fired on the 0-edge when a transfer target
 * is pending — structurally identical to the WB arm.  The drain-now /
 * confirm-ack divergence lives only in the GRANT_RESPONSE / CONFIRM
 * handlers, not here. */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: no local hold to drop means our acquire never bumped ownership
   * (e.g. a cache already torn down by a destroy fan-out); decrementing would
   * underflow.  Which word states carry no hold is the release policy's —
   * possession alone is one of them under a voluntary-return policy and not
   * under a revoked one. */
  if (arts_db_grant_release_skip(cache)) {
    return;
  }
  /* Acquire current buffer for the version bump + PUBLISH send.  Local ref
   * scoped to release_rw — the EDT's own ref is dropped by release_one_dep. */
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t new_version = 0;
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
    new_version = arts_atomic_read_u64(&buf->version);
  }
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  /* HOME keeps home's RO copy fresh: pure synchronous publish every release
   * (a non-home owner; home owners already hold the canonical buffer).  No
   * WB_AND_TRANSFER mode — ownership transfer is the separate owner→owner ship
   * below.  Derived ONCE: the hand-back rides this decision, and deriving it
   * twice is how the two drift apart. */
  bool will_publish = (!is_home && buf != NULL);
  /* Where the write right goes back unasked it can travel with these bytes
   * instead of behind them, which is the whole saving — so the claim is taken
   * BEFORE the publish, and a won claim replaces the count-dropping edge. */
  bool handed_back = arts_db_grant_release_claim(cache, will_publish);
  if (will_publish) {
    /* The flight pins its own source ref; this caller's ref only covers the
     * version bump above. */
    arts_db_publish_sync(cache, new_version);
  }
  /* WT: release buffer ref after the publish (which reads buf->data). */
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  /* The count-dropping edge, and what this rank owes at it, belong to the
   * release policy — a revoked grant ships onward to the target the round
   * named, a voluntarily returned one goes back to the home.  Sequenced after
   * the publish above either way: the bytes must be at the home before the
   * write right can move.  A claim taken above already dropped the count, and
   * only has to settle which vehicle carried it. */
  if (handed_back) {
    arts_db_grant_release_settle(cache);
  } else {
    arts_db_grant_release_commit(cache);
  }
}

/* ===== cache_s lifecycle (WT: pending_rw Vyukov MPSC) =============
 * Construct: the WT write policy's field-init (the Vyukov MPSC pending_rw queue
 * — cannot
 * be zero-initialized, head/tail must point at the embedded stub) runs BEFORE
 * arts_db_cache_common_init so the queue is wired before any push could land.
 * Destruct order: buffer-NULL (pre) → pending_rw destroy → snapshot drain +
 * home teardown (post). */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  arts_pending_rw_queue_init(&c->pending_rw);
  /* Transfer target for the commit-PROCEED, published by each round's
   * INVALIDATE before it withdraws the sentinel.  Start at the sentinel (no
   * transfer pending); writer_count->0 always implies a prior INVALIDATE set
   * it. */
  c->incoming_new_owner = ARTS_NO_PENDING_OWNER;
  c->incoming_new_owner_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  /* WT has no owner-side dedup map (home serves RO via SNAPSHOT_REQUEST): the
   * owner→owner transfer always ships an empty map.  NULL so the shared ship
   * helper's map-build gate takes its empty-map branch. */
  c->cached_version = NULL;
#ifdef ARTS_RELEASE_PURGE
  c->pending_grant_return = 0u;
#endif
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
  arts_home_grantreq_queue_init(&db->pending_rw);
  atomic_store_explicit(&db->invalidate_in_flight, 0, memory_order_relaxed);
  db->cached_version = arts_rank_u64_map_create(nranks);
  db->pending_install_owner = 0;
#ifdef ARTS_RELEASE_PURGE
  atomic_store_explicit(&db->pending_return_from, ARTS_GRANT_NO_RETURNER,
                        memory_order_relaxed);
#endif
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_grantreq_queue_destroy(&db->pending_rw);
  arts_rank_u64_map_destroy(db->cached_version);
  /* No free: home fields are inlined in the arts_db_s. */
}

/* ===== SNAPSHOT_REQUEST reply (home.cached_version atomic-monotonic) ===== */

/* update_cached_version_max: the SNAPSHOT_REQUEST reply path.  Decide send-with-
 * data vs send-no-data based on the home watermark, then advance the
 * watermark.  Under single-threaded handler dispatch the "atomic
 * CAS-loop" the design plan specifies collapses to a plain compare/
 * advance — but we keep the helper signature so future MPMC upgrades
 * are localized. */
/* SNAPSHOT_REQUEST reply path: decide CTS / no-data / one-sided-data from the home
 * watermark and the requester's landing, then advance the watermark only when
 * payload actually moves.  Consumes master_h (the pinned canonical buffer). */
static void update_cached_version_max(struct arts_db_cache_s *cache,
                                 unsigned int requester, uint64_t master_v,
                                 arts_shared_ptr_t master_h,
                                 arts_guid_t edt_guid, uint32_t slot,
                                 const struct arts_rdzv_landing_s *rdzv) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  /* Monotonic dedup — if the requester already holds this version
   * (cur >= master_v), no payload moves: reply no-data, echoing the unused
   * landing for recycling.  The ledger is credited by serves AND by the
   * requester's own publishs, so a self-produced copy dedups too. */
  uint64_t cur = arts_rank_u64_map_get(db->cached_version, requester);
  if (cur >= master_v) {
    arts_db_buf_release(&master_h);
    arts_send_db_snapshot_response(requester, cache->db_guid, master_v,
                                   edt_guid, slot, /*kind=*/0,
                                   cache->db_size, rdzv, NULL);
    return;
  }
  if (rdzv->txid == 0 && arts_global_rank_count > 1) {
    /* Data must move but the requester advertised no landing (first touch —
     * db_size unknown there).  Size-only CTS; the re-request carries a
     * landing.  The watermark does NOT advance on this leg. */
    arts_db_buf_release(&master_h);
    arts_send_db_snapshot_response(requester, cache->db_guid, master_v,
                                   edt_guid, slot, /*kind=*/2,
                                   cache->db_size, rdzv, NULL);
    return;
  }
  arts_rank_u64_map_advance(db->cached_version, requester, master_v);
  /* master_h transfers into the sender (PUT source-lifetime pin). */
  arts_send_db_snapshot_response(requester, cache->db_guid, master_v, edt_guid,
                                 slot, /*kind=*/1, cache->db_size, rdzv,
                                 master_h);
}

/* Deferred SNAPSHOT_REQUEST serve (snapshot-waiter `serve` arm): the request arrived
 * at home before the creator's first PUBLISH installed a buffer, and the
 * install's drain now re-issues it.  Runs the same serve tail as the request
 * handler.  The waiter is freed by the drain after this returns. */
static void home_serve_parked_snapshot(struct arts_db_cache_s *cache,
                                        struct arts_db_snapshot_waiter_s *w) {
  arts_shared_ptr_t master_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *master =
      (struct arts_db_buffer_s *)arts_shared_get(master_h);
  if (master != NULL &&
      __atomic_load_n(&master->version, __ATOMIC_ACQUIRE) == 0) {
    /* v0 = still unpublished (drain paths fire only on a positive bump, so
     * this is teardown-adjacent defensiveness, not a live path). */
    arts_db_buf_release(&master_h);
    master = NULL;
    master_h = NULL;
  }
  if (master == NULL) {
    /* Only reachable when the drain runs from cache teardown with the serve
     * still parked (destroy-during-pending-acquire is app UB): fall back to
     * the no-data reply so the requester is never stranded. */
    arts_send_db_snapshot_response(w->requester, cache->db_guid, /*version=*/0,
                                   w->edt_guid, w->slot, /*kind=*/0,
                                   cache->db_size, &w->rdzv, NULL);
    return;
  }
  /* Acquire pairs with the publish commit's release-store version stamp: a
   * serve must never stamp a version whose bytes it cannot yet see. */
  uint64_t master_v = __atomic_load_n(&master->version, __ATOMIC_ACQUIRE);
  /* master_h transfers into the reply path (consumed there). */
  update_cached_version_max(cache, w->requester, master_v, master_h, w->edt_guid,
                       w->slot, &w->rdzv);
}

/* ===== Per-protocol wire-handler bodies =============================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_SNAPSHOT_REQUEST]): the OoO engine
 * has already acquired the home db_s for db_guid and pinned a ref across this
 * call (cache is its FIRST member), so there is no lookup / NULL-check / defer
 * here.  The HOME protocol serves from home's canonical buffer with
 * cached_version dedup. */
void arts_handler_db_snapshot_request(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_snapshot_request_s *a =
      (struct arts_ooo_args_db_snapshot_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_guid_t edt_guid = a->edt_guid;
  uint32_t slot = a->slot;

  arts_shared_ptr_t master_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *master =
      (struct arts_db_buffer_s *)arts_shared_get(master_h);
  if (master != NULL &&
      __atomic_load_n(&master->version, __ATOMIC_ACQUIRE) == 0) {
    /* Present-but-unpublished (v0, installed at create): park exactly as
     * if no buffer existed — the creator's first publish is the
     * publication point. */
    arts_db_buf_release(&master_h);
    master = NULL;
    master_h = NULL;
  }
  if (master == NULL) {
    /* Two cases produce master==NULL post-precheck:
     *   (a) Sentinel DB (db_size==0): no buffer is ever installed — respond
     *       with version=0, NULL data (NULL is the sentinel's defined value).
     *   (b) HOME_RECV pre-PUBLISH (db_size>0): cross-rank create has
     *       happened but the creator's first PUBLISH has not landed.  The
     *       creator's RELEASE is the publication point a satisfied consumer
     *       is entitled to observe, so PARK the serve on the snapshot
     *       reorder buffer; the first publish install drains it and
     *       re-issues the serve with the original requester/landing.
     * NOT a destroy condition -- the precheck above (destroy_state) is
     * authoritative for that. */
    if (cache->db_size > 0) {
      struct arts_db_snapshot_waiter_s *w =
          (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
      w->edt_guid = edt_guid;
      w->slot = slot;
      w->target_version = 1;
      w->serve = home_serve_parked_snapshot;
      w->requester = requester;
      w->rdzv = a->rdzv;
      arts_lf_stack_push(&cache->pending_snapshot, &w->link);
      /* Race recovery: a publish may have landed between the version read
       * and the push — drain (our own node included) so no serve parks
       * forever. */
      arts_shared_ptr_t rh = arts_db_buf_acquire(cache);
      struct arts_db_buffer_s *rb =
          (struct arts_db_buffer_s *)arts_shared_get(rh);
      bool published =
          (rb != NULL && __atomic_load_n(&rb->version, __ATOMIC_ACQUIRE) > 0);
      arts_db_buf_release(&rh);
      if (published) {
        arts_db_drain_pending_snapshot(cache);
      }
      return;
    }
    arts_send_db_snapshot_response(requester, cache->db_guid, /*version=*/0,
                                   edt_guid, slot, /*kind=*/0, cache->db_size,
                                   &a->rdzv, NULL);
    return;
  }
  /* Acquire pairs with the publish commit's release-store version stamp: a
   * serve must never stamp a version whose bytes it cannot yet see. */
  uint64_t master_v = __atomic_load_n(&master->version, __ATOMIC_ACQUIRE);
  /* master_h transfers into the reply path (consumed there). */
  update_cached_version_max(cache, requester, master_v, master_h, edt_guid, slot,
                       &a->rdzv);
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_PUBLISH]): the OoO engine has
 * already acquired the home db_s and pinned a ref across this call (cache is
 * its FIRST member); the dispatcher copies PUBLISH's trailing data payload
 * into the args blob after arts_ooo_args_db_publish_s and this body reads it
 * back from (char *)a + sizeof(*a).  A PUBLISH that races ahead of DB_CREATE
 * MISSes the engine, which defers (trailing data preserved) and re-issues this
 * body on the install's drain.
 *
 * ACK-exactly-once: the body acks (cv != 0) only when it runs (cache live).  A
 * deferred PUBLISH does NOT ack on the defer — the single PUBLISH_ACK is
 * sent on the drain replay.  Pure: it only installs (version-guarded) + acks;
 * ownership transfer is a separate owner→owner GRANT_RESPONSE ship. */
/* Rendezvous continuation for a committed publish: the dirty bytes have
 * fully landed IN PLACE in the stable home buffer; publication is the version
 * stamp alone (release-store, sequenced after the {commit packet, write
 * completion} pairing — "landing valid" precedes the stamp).  The ACK fires
 * even if the DB was destroyed while the pairing was outstanding — a
 * torn-down home cache must never strand the blocked releaser (the
 * PUBLISH_ACK-on-MISS rule). */
struct pub_landed_ctx_s {
  arts_shared_ptr_t db_h;
  uint64_t version;
  unsigned int releaser;
  arts_guid_t db_guid;
  uint64_t cv;
  bool returns_grant;
};

static void pub_landed_cb(void *arg) {
  struct pub_landed_ctx_s *ctx = (struct pub_landed_ctx_s *)arg;
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(ctx->db_h);
  if (db != NULL) {
    arts_db_buf_bump_inplace(&db->cache, ctx->version);
    /* The releaser holds the version it just published — credit the ledger
     * so its own next acquire dedups to no-data. */
    arts_rank_u64_map_advance(db->cached_version, ctx->releaser, ctx->version);
    /* First-publish publication (bump from version 0): wake home-parked RO
     * acquires and re-issue home-parked SNAPSHOT_REQUEST serves. */
    arts_db_drain_pending_snapshot(&db->cache);
  }
  arts_send_db_publish_ack(ctx->releaser, ctx->db_guid, ctx->cv, ctx->version,
                             db != NULL ? &db->cache : NULL);
  if (ctx->returns_grant && db != NULL) {
    /* The releaser handed the write right back with these bytes.  Accepting
     * here and not a moment earlier is what makes the two one event: the
     * stamp above is already in place, so a block handed straight on carries
     * a version this release has not moved past.  A block destroyed while the
     * bytes were in flight owes nothing — its directory went with it. */
    arts_db_grant_return_arrived(db, ctx->releaser);
  }
  arts_shared_release(&ctx->db_h);
  arts_free(ctx);
}

void arts_handler_db_publish(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_publish_s *a =
      (struct arts_ooo_args_db_publish_s *)args_v;

  if (a->data_size == 0) {
    /* Data-less ordering round (sentinel DB): install nothing, ACK. */
    arts_send_db_publish_ack(a->releaser, a->db_guid, a->cv, a->version,
                               cache);
    return;
  }
  if (a->data_inline != 0) {
    /* A home-resident writer mutates the stable buffer directly and its
     * release publishes nothing over this plane — an inline publish reaching
     * this arm means the release path regressed. */
    ARTS_ERROR("coherence: inline same-rank publish is unreachable under a "
               "home-canonical write-through release path");
  }
  if (a->rdzv_txid == 0) {
    /* Announce: the releaser holds a->data_size dirty bytes and no home
     * address yet.  Advertise the STABLE buffer as the landing (the PUT lands
     * in place; a concurrent snapshot serve may ship a torn old/new word mix,
     * which the memory model already permits for unordered readers) and mint
     * the pairing txid.  Nothing is allocated; nothing installs at commit. */
    if (a->data_size > cache->db_size) {
      ARTS_ERROR("coherence: publish announce of %llu bytes exceeds the "
                 "DB's %llu-byte stable buffer",
                 (unsigned long long)a->data_size,
                 (unsigned long long)cache->db_size);
    }
    struct arts_rdzv_landing_s landing = {0, 0, 0, 0};
    arts_shared_ptr_t mh = arts_db_buf_acquire(cache);
    struct arts_db_buffer_s *master =
        (struct arts_db_buffer_s *)arts_shared_get(mh);
    if (master == NULL ||
        !arts_net_rdzv_local(master->data, cache->db_size, &landing.addr,
                             &landing.key)) {
      ARTS_ERROR("coherence: publish announce found no stable home buffer");
    }
    landing.txid = arts_net_rdzv_txid_next();
    arts_db_buf_release(&mh);
    arts_send_db_publish_cts(a->releaser, a->db_guid, &landing, a->cv);
    return;
  }
  /* Commit: the dirty bytes were PUT in place into the stable buffer.  Pair
   * with the write completion — either arrival order — then stamp the
   * version and ACK the blocked releaser. */
  struct pub_landed_ctx_s *ctx =
      (struct pub_landed_ctx_s *)arts_malloc(sizeof(*ctx));
  ctx->db_h = arts_route_table_lookup_db(cache->db_guid);
  ctx->version = a->version;
  ctx->releaser = a->releaser;
  ctx->db_guid = a->db_guid;
  ctx->cv = a->cv;
  ctx->returns_grant = (a->returns_grant != 0u);
  arts_net_rdzv_expect(a->rdzv_txid, pub_landed_cb, ctx);
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_DESTROY]): the OoO engine has already
 * acquired the home db_s and pinned a ref across this call (cache is its FIRST
 * member).  Order: roster fan-out, then
 * arts_route_table_set_destroyed LAST (detach the slot cb + drop the install
 * ref); a waiter left parked at destroy (UB) is cleaned up by the destructor detaches the slot cb + drops the install
 * ref; the cb deleter frees the cache once outstanding lookup refs drain.  A
 * second DESTROY_REQ finds the slot absent and is a no-op.  WT roster
 * source = home->cached_version + the creator-slice probe. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  /* WT: home->cached_version is the readers roster. */
  {
    unsigned int n = arts_global_rank_count;
    for (unsigned int r = 0; r < n; r++) {
      if (r == self) {
        continue;
      }
      if (arts_rank_u64_map_get(db->cached_version, r) > 0) {
        arts_send_db_cache_destroy(r, a->db_guid);
      }
    }
  }
  /* The queued ownership requesters are NOT drained for the roster: the FIFO
   * has exactly one consumer — the rank holding the transfer baton — and a
   * destroy popping it concurrently is a second one.  A requester that has
   * touched this block is already in the version ledger above, and a
   * first-touch requester is covered by the creator-slice probe below. */
  /* The creator may hold an in-place publish credit taught at create
   * (CREATE_RETURN) without ever having published, so the version ledger
   * cannot name it.  A credit holder must join the teardown roster, or a
   * labeled-GUID reuse would find the stale credit still armed against a
   * buffer that no longer exists. */
  if (arts_db_seq_budget != 0) {
    uint64_t slice = ARTS_GUID_DB_GET_SEQ(a->db_guid) / arts_db_seq_budget;
    if (slice < (uint64_t)arts_global_rank_count &&
        (unsigned int)slice != self &&
        arts_rank_u64_map_get(db->cached_version, (unsigned int)slice) == 0) {
      arts_send_db_cache_destroy((unsigned int)slice, a->db_guid);
    }
  }
  (void)arts_route_table_set_destroyed(a->db_guid);
  /* After the slot withdrawal — see arts_db_pub_flight_abandon. */
  arts_db_pub_flight_abandon(cache);
}

/* Case-D leaf: HOME publishes creator_rank as the home rw_holder (coalesce
 * path). */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  /* The stable home buffer exists from create so its address can travel as
   * a durable publish credit — but at VERSION 0: "unpublished".  The
   * has-the-creator-published predicate that used to be buffer PRESENCE is
   * version > 0 everywhere on this arm; the first publish's 0->v bump is
   * the publication point that drains the parked readers/serves. */
  if (db_size > 0) {
    (void)arts_db_buf_install(cache, /*new_version=*/0, /*data_payload=*/NULL,
                              db_size);
  }
}

/* The home's installed buffer IS the canonical copy under this write policy,
 * and every publish stamps it in place, so its version is the axis a serve is
 * judged against. */
uint64_t arts_db_grant_serve_version(struct arts_db_s *db) {
  arts_shared_ptr_t h = arts_db_buf_acquire(&db->cache);
  struct arts_db_buffer_s *b = (struct arts_db_buffer_s *)arts_shared_get(h);
  uint64_t v = (b != NULL) ? arts_atomic_read_u64(&b->version)
                           : ARTS_GRANT_VERSION_NONE;
  arts_db_buf_release(&h);
  return v;
}

/* Readers re-check a version at every acquire, so an ex-holder's retained
 * buffer can never be mistaken for current: nothing to register. */
void arts_db_grant_note_ex_holder(struct arts_db_s *db, unsigned int rank) {
  (void)db;
  (void)rank;
}
