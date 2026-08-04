/* SPDX-License-Identifier: Apache-2.0
 *
 * HOME protocol translation unit: defines the HOME-specific
 * arts_handler_db_* / arts_db_* bodies directly (CMake links exactly this TU
 * for an RCU+HOME build) plus the HOME-only wire handlers/senders.
 * Compiled only for ARTS_COHERENCE_PROTOCOL=RCU with
 * ARTS_WRITE_POLICY=WT (selected in libs/src/core/CMakeLists.txt).
 * Contains NO protocol/placement preprocessor logic.
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
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (pin db_s) */
#include "arts/ooo.h"             /* OOO_DB_* args (handler bodies) */
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
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
      dep->ptr = arts_db_acquire_local(cache);
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
        if (arts_shared_get(rh) != NULL) {
          arts_db_buf_release(&rh);
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
                            slot); /* parks (OWNERSHIP_REQUEST) */
}

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW;
}

/* ===== release_rw (HOME arm) ========================================
 * The HOME protocol keeps home's RO copy fresh with a pure synchronous
 * PUBLISH on every non-home release (home owners already hold the canonical
 * buffer).  Ownership transfer is a SEPARATE owner→owner ship
 * (arts_db_send_grant_response), fired on the 0-edge when a transfer target
 * is pending — structurally identical to the OWNER arm.  The drain-now /
 * confirm-ack divergence lives only in the OWNERSHIP_RESPONSE / CONFIRM
 * handlers, not here. */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped ownership (e.g. a
   * cache already torn down by a destroy fan-out); decrementing would
   * underflow. Atomic acquire-load avoids a TSan race against concurrent
   * writes. */
  if (arts_atomic_read(&cache->writer_count) == 0) {
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
   * below. */
  if (!is_home && buf != NULL) {
    /* buf_h (held until after this call) pins buf->data for the whole round —
     * the sync helper's ACK follows the target-side write completion, which
     * implies the fabric has fully drained the source. */
    arts_db_publish_sync(cache, new_version, buf->data, cache->db_size);
  }
  /* Eager: release buffer ref after the publish (which reads buf->data). */
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  /* writer_count is non-negative (post-install flip + install guard absorb any
   * INVALIDATE that lands during install).  A true 1->0 release reads 0 and
   * ships the transfer; the (int) cast is defensive. */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1); /* post value */
  if (rest == 0 && cache->incoming_new_owner != ARTS_NO_PENDING_OWNER) {
    /* An INVALIDATE published a transfer target while writers were live; this
     * (last) releaser is the unique actor that ships the owner→owner transfer.
     * Identical for home and non-home owners. */
    arts_db_send_grant_response(cache);
  }
}

/* ===== cache_s lifecycle (HOME: pending_rw Vyukov MPSC) =============
 * Construct: the HOME placement's field-init (the Vyukov MPSC pending_rw queue
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
  /* HOME has no owner-side dedup map (home serves RO via GET_DATA): the
   * owner→owner transfer always ships an empty map.  NULL so the shared ship
   * helper's map-build gate takes its empty-map branch. */
  c->cached_version = NULL;
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
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_grantreq_queue_destroy(&db->pending_rw);
  arts_rank_u64_map_destroy(db->cached_version);
  /* No free: home fields are inlined in the arts_db_s. */
}

/* ===== GET_DATA reply (home.cached_version atomic-monotonic) ===== */

/* update_cached_version_max: the GET_DATA reply path.  Decide send-with-
 * data vs send-no-data based on the home watermark, then advance the
 * watermark.  Under single-threaded handler dispatch the "atomic
 * CAS-loop" the design plan specifies collapses to a plain compare/
 * advance — but we keep the helper signature so future MPMC upgrades
 * are localized. */
/* GET_DATA reply path: decide CTS / no-data / one-sided-data from the home
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

/* Deferred GET_DATA serve (snapshot-waiter `serve` arm): the request arrived
 * at home before the creator's first PUBLISH installed a buffer, and the
 * install's drain now re-issues it.  Runs the same serve tail as the request
 * handler.  The waiter is freed by the drain after this returns. */
static void home_serve_parked_snapshot(struct arts_db_cache_s *cache,
                                        struct arts_db_snapshot_waiter_s *w) {
  arts_shared_ptr_t master_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *master =
      (struct arts_db_buffer_s *)arts_shared_get(master_h);
  if (master == NULL) {
    /* Only reachable when the drain runs from cache teardown with the serve
     * still parked (destroy-during-pending-acquire is app UB): fall back to
     * the no-data reply so the requester is never stranded. */
    arts_send_db_snapshot_response(w->requester, cache->db_guid, /*version=*/0,
                                   w->edt_guid, w->slot, /*kind=*/0,
                                   cache->db_size, &w->rdzv, NULL);
    return;
  }
  uint64_t master_v = master->version;
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
      /* Race recovery: an install may have landed between the NULL read and
       * the push — drain (our own node included) so no serve parks forever. */
      arts_shared_ptr_t rh = arts_db_buf_acquire(cache);
      if (arts_shared_get(rh) != NULL) {
        arts_db_buf_release(&rh);
        arts_db_drain_pending_snapshot(cache);
      }
      return;
    }
    arts_send_db_snapshot_response(requester, cache->db_guid, /*version=*/0,
                                   edt_guid, slot, /*kind=*/0, cache->db_size,
                                   &a->rdzv, NULL);
    return;
  }
  uint64_t master_v = master->version;
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
 * ownership transfer is a separate owner→owner OWNERSHIP_RESPONSE ship. */
/* Rendezvous continuation for a committed publish: the dirty bytes have
 * fully landed in the home landing; install them without a copy and ACK the
 * blocked releaser.  The ACK fires even if the DB was destroyed while the
 * pairing was outstanding — a torn-down home cache must never strand the
 * blocked releaser (the PUBLISH_ACK-on-MISS rule). */
struct pub_landed_ctx_s {
  arts_shared_ptr_t db_h;
  struct arts_db_buffer_s *landing;
  uint64_t version;
  uint64_t data_size;
  unsigned int releaser;
  arts_guid_t db_guid;
  uint64_t cv;
};

static void pub_landed_cb(void *arg) {
  struct pub_landed_ctx_s *ctx = (struct pub_landed_ctx_s *)arg;
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(ctx->db_h);
  if (db != NULL) {
    arts_db_buf_install_landed(&db->cache, ctx->version, ctx->landing,
                               ctx->data_size);
    /* The releaser holds the version it just published — credit the ledger
     * so its own next acquire dedups to no-data. */
    arts_rank_u64_map_advance(db->cached_version, ctx->releaser, ctx->version);
    /* First-publish publication: wake home-parked RO acquires and re-issue
     * home-parked GET_DATA serves (pre-install parkers). */
    arts_db_drain_pending_snapshot(&db->cache);
  } else {
    /* Destroyed mid-round (app UB): the cache's recycle pool is gone — return
     * the landing's storage straight to the registered pool. */
    arts_regpool_free(ctx->landing);
  }
  if (ctx->cv != 0) {
    arts_send_db_publish_ack(ctx->releaser, ctx->db_guid, ctx->cv);
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
    if (a->cv != 0) {
      arts_send_db_publish_ack(a->releaser, a->db_guid, a->cv);
    }
    return;
  }
  if (a->data_inline != 0) {
    /* Same-rank publish: the payload trails the args blob.  Monotonic:
     * buf_install ignores a stale (lower/equal version) publish. */
    arts_db_buf_install(cache, a->version, (const void *)((char *)a + sizeof(*a)),
                        a->data_size);
    /* The releaser holds the version it just published — credit the ledger
     * so its own next acquire dedups to no-data. */
    arts_rank_u64_map_advance(arts_db_of_cache(cache)->cached_version,
                              a->releaser, a->version);
    /* First-publish publication: wake home-parked RO acquires and re-issue
     * home-parked GET_DATA serves (pre-install parkers). */
    arts_db_drain_pending_snapshot(cache);
    if (a->cv != 0) {
      arts_send_db_publish_ack(a->releaser, a->db_guid, a->cv);
    }
    return;
  }
  if (a->rdzv_txid == 0) {
    /* Announce: the releaser holds a->data_size dirty bytes.  Allocate a
     * fresh home landing for them and hand it back (PUBLISH_CTS); nothing
     * installs yet — the commit leg pairs with the write completion. */
    struct arts_rdzv_landing_s landing;
    (void)arts_db_buf_landing_alloc(cache, a->data_size, &landing);
    arts_send_db_publish_cts(a->releaser, a->db_guid, &landing, a->cv);
    return;
  }
  /* Commit: the dirty bytes were PUT into our landing (named by the echoed
   * cookie).  Pair with the write completion — either arrival order — then
   * install the landing without a copy (version-conditional; stale retreats
   * recycle) and ACK the blocked releaser. */
  struct pub_landed_ctx_s *ctx =
      (struct pub_landed_ctx_s *)arts_malloc(sizeof(*ctx));
  ctx->db_h = arts_route_table_lookup_db(cache->db_guid);
  ctx->landing = (struct arts_db_buffer_s *)(uintptr_t)a->rdzv_cookie;
  ctx->version = a->version;
  ctx->data_size = a->data_size;
  ctx->releaser = a->releaser;
  ctx->db_guid = a->db_guid;
  ctx->cv = a->cv;
  arts_net_rdzv_expect(a->rdzv_txid, pub_landed_cb, ctx);
}

/* Cat-C pure body (PUBLISH_ACK).  Cache-independent pointer-identity sem-post
 * on a->cv (the releaser's stack-local sem_t, valid on this rank — the ACK
 * always returns to the sender), so item_v is unused.  The dispatcher posts on
 * BOTH a HIT (this body) and a MISS so a torn-down home cache never strands the
 * blocked releaser. */
void arts_handler_db_publish_ack(void *item_v, void *args_v) {
  (void)item_v;
  struct arts_db_publish_ack_args_s *a =
      (struct arts_db_publish_ack_args_s *)args_v;
  sem_t *s = (sem_t *)(uintptr_t)a->cv;
  if (s != NULL) {
    sem_post(s);
  }
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_DESTROY]): the OoO engine has already
 * acquired the home db_s and pinned a ref across this call (cache is its FIRST
 * member).  Order: roster fan-out, then
 * arts_route_table_set_destroyed LAST (detach the slot cb + drop the install
 * ref); a waiter left parked at destroy (UB) is cleaned up by the destructor detaches the slot cb + drops the install
 * ref; the cb deleter frees the cache once outstanding lookup refs drain.  A
 * second DESTROY_REQ finds the slot absent and is a no-op.  Eager roster
 * source = home->cached_version + the queued ownership requesters. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  /* Eager: use home->cached_version as the readers roster, then the queued
   * ownership requesters. */
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
  {
    unsigned int q_rank;
    while (arts_home_grantreq_queue_pop(&db->pending_rw, &q_rank, NULL)) {
      if (q_rank != self) {
        arts_send_db_cache_destroy(q_rank, a->db_guid);
      }
    }
  }
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* Case-D leaf: HOME publishes creator_rank as the home rw_holder (coalesce
 * path). */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  (void)cache;
  (void)db_size;
}

/* Readers re-check a version at every acquire, so an ex-holder's retained
 * buffer can never be mistaken for current: nothing to register. */
void arts_db_grant_note_ex_holder(struct arts_db_s *db, unsigned int rank) {
  (void)db;
  (void)rank;
}
