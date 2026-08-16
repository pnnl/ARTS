/* SPDX-License-Identifier: Apache-2.0
 *
 * WRF_VAL protocol translation unit: defines the WRF_VAL (DB-WRF)-specific
 * arts_handler_db_* / arts_db_* bodies directly (CMake links exactly this TU
 * for a WRF_VAL build) plus the WRF_VAL-only wire handlers/senders.
 * Compiled only for the DB_WRF x VAL configuration (arm WRF_VAL; selected in
 * libs/src/core/CMakeLists.txt). Contains NO model/protocol preprocessor
 * logic.
 */
#include <semaphore.h>
#include <stdbool.h>
#include <stdint.h>

#include "arts/coherence/buffer.h"
#include "arts/memory/regpool.h" /* arts_regpool_free (orphaned landing) */
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/directory.h"
#include "arts/gas/guid.h" /* creator slice arithmetic */
#include "arts/gas/route_table.h" /* pairing-window descriptor pin */
#include "arts/db.h"
#include "arts/edt.h" /* arts_edt_dep_t (acquire body) */
#include "arts/ooo.h" /* OOO_DB_* args (handler bodies) */
#include "arts/runtime_state.h"
#include "arts/transport/net.h" /* arts_net_rdzv_expect */
#include "arts/runtime_types.h"
#include "arts/system/print.h"   /* ARTS_ERROR */
#include "arts/system/threads.h" /* arts_global_rank_id */
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h" /* pairing ctx */  /* arts_atomic_* */

/* ===== 8-case acquire dispatch (WRF_VAL arm) ==========================
 * Whole arts_handler_db_acquire body for the WRF_VAL build.  Home holds the
 * canonical buffer (maintained by sync PUBLISH from every non-home writer).
 * RW and RO are unified — non-home acquires go through acquire_remote_ro in
 * both modes so the EDT parks (no list registration) and is woken by
 * SNAPSHOT_RESPONSE once home delivers its current buffer.  There is no
 * GRANT_REQUEST / INVALIDATE / GRANT round, no per-cache pending_rw queue.
 *
 * RW acquires bump writer_count BEFORE parking (or before acquire_local on
 * home).  release_rw balances this decrement; without the bump, release_rw's
 * writer_count == 0 guard silently skips the PUBLISH, breaking cross-rank RW
 * visibility.  RO acquires do not bump because release_ro is a no-op. */
void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *dep = &((arts_edt_dep_t *)arts_get_depv(edt))[slot];
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  if (dep->mode == DB_MODE_RW) {
    arts_atomic_add(&cache->writer_count, 1); /* balanced by release_rw */
  }
  if (is_home) {
    dep->ptr = arts_db_acquire_local(cache);
    arts_db_acquire_resolved(edt, slot);
    return;
  }
  arts_db_acquire_remote_ro(cache, edt->guid, slot); /* parks (SNAPSHOT_REQUEST) */
}

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  (void)mode;
  return false; /* WRF_VAL: no ownership round; nothing is serialized */
}

/* ===== release_rw (WRF_VAL arm) =======================================
 * WRF_VAL drops its buffer ref in the tail (after the PUBLISH
 * reads buf->data).  Every non-home write must be pushed back to home
 * synchronously so home stays canonical before any subsequent acquire can see
 * fresh data; home itself needs no PUBLISH.  R3 (rest > 0, non-home) and
 * R4 (rest == 0, non-home) take the same sync publish. */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped ownership;
   * decrementing would underflow.  Atomic acquire-load avoids a TSan race. */
  if (arts_atomic_read(&cache->writer_count) == 0) {
    return;
  }
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t new_version = 0;
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
    new_version = arts_atomic_read_u64(&buf->version);
  }
  (void)arts_atomic_sub(&cache->writer_count,
                        1); /* WRF_VAL: rest is not consulted */
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  if (!is_home && buf != NULL) {
    /* buf_h (held until after this call) pins buf->data for the whole round —
     * the sync helper's ACK follows the target-side write completion, which
     * implies the fabric has fully drained the source. */
    arts_db_publish_sync(cache, new_version);
  }
  /* Release the buffer ref held for the version-bump and PUBLISH read. */
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }
}

/* ===== cache_s lifecycle (WRF_VAL: no pending_rw queue) ===============
 * The WRF_VAL cache struct omits the pending_rw field entirely, so there is no
 * protocol field-init / field-destroy: the wrapper just calls the
 * shared common steps (construct: common; destruct: buffer-NULL → snapshot
 * drain + home teardown).
 */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache);  /* buffer-NULL FIRST */
  arts_db_cache_common_destroy_post(cache); /* snapshot drain → home teardown */
}

/* ===== home-directory lifecycle (WRF_VAL: only cached_version) ===== */

void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks) {
  /* WRF_VAL: no exclusive owner — rw_holder is unused. */
  (void)rw_holder;
  db->cached_version = arts_rank_u64_map_create(nranks);
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
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

/* ===== Per-model wire-handler bodies =============================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_SNAPSHOT_REQUEST]): the OoO engine
 * has already acquired the home db_s and pinned a ref across this call (cache
 * is its FIRST member), so there is no lookup / NULL-check / defer here.  The
 * WRF_VAL serves from home's canonical buffer with cached_version dedup. */
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
    /* Home-canonical: every sized block gets its version-1 zero buffer at
     * create (arts_db_create_install_home_buffer below), so the only block
     * that reaches here is a zero-sized one — respond version=0, NULL data,
     * which is that block's defined value.  A sized block never resolves to a
     * NULL pointer: "nobody has published yet" bears on its contents, not on
     * whether it has storage.  NOT a destroy condition -- the precheck above
     * (destroy_state) is authoritative for that. */
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
 * its FIRST member); the dispatcher copies the trailing data payload into the
 * args blob after arts_ooo_args_db_publish_s and this body reads it back from
 * (char *)a + sizeof(*a).  A PUBLISH that races ahead of DB_CREATE defers and
 * re-issues on the install's drain.  WRF_VAL uses PUBLISH_NORMAL only (no
 * exclusive owner to transfer to), so the WB_AND_TRANSFER ownership-chain
 * relay is moot. */
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
};

static void pub_landed_cb(void *arg) {
  struct pub_landed_ctx_s *ctx = (struct pub_landed_ctx_s *)arg;
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(ctx->db_h);
  if (db != NULL) {
    arts_db_buf_bump_inplace(&db->cache, ctx->version);
    /* The releaser holds the version it just published — credit the ledger
     * so its own next acquire dedups to no-data. */
    arts_rank_u64_map_advance(db->cached_version, ctx->releaser, ctx->version);
  }
  arts_send_db_publish_ack(ctx->releaser, ctx->db_guid, ctx->cv, ctx->version,
                             db != NULL ? &db->cache : NULL);
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
  arts_net_rdzv_expect(a->rdzv_txid, pub_landed_cb, ctx);
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_DESTROY]): the OoO engine has already
 * acquired the home db_s and pinned a ref across this call (cache is its FIRST
 * member).  Order: roster fan-out, then arts_route_table_set_destroyed LAST (a
 * waiter left parked at destroy = UB, freed by the refcount-0 destructor).
 * The WRF_VAL roster source is
 * home->cached_version (same as the WT write policy); WRF_VAL has no
 * pending_rw queue, so no grantreq drain. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  unsigned int n = arts_global_rank_count;
  for (unsigned int r = 0; r < n; r++) {
    if (r == self) {
      continue;
    }
    if (arts_rank_u64_map_get(db->cached_version, r) > 0) {
      arts_send_db_cache_destroy(r, a->db_guid);
    }
  }
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

/* Case-D leaf: WRF_VAL has no exclusive owner — no rw_holder to publish
 * (no-op). */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  (void)db;
  (void)creator_rank;
}

/* WRF_VAL has no exclusive-ownership protocol (no GRANT_REQUEST, no
 * GRANT_INVALIDATE — its dispatcher fatals on both wire messages), so its
 * ooo_kind enum omits OOO_DB_GRANT_REQUEST and OOO_DB_GRANT_INVALIDATE
 * entirely.  The WRF_VAL build therefore defines no
 * arts_handler_db_grant_request / _grant_invalidate body — the real
 * bodies live in coherence/grant.c / each arm's own write-policy TU, which
 * WRF_VAL does not compile. */

/* ===== create-time home buffer (WRF_VAL: home is canonical) =========== */

/* Case-D leaf: the WRF_VAL home holds the canonical copy; there is no
 * creator PUBLISH to wait for, so publish a version-1 zero buffer
 * immediately.  Without it the first home RW acquire (acquire_local) hands the
 * EDT a NULL payload. */
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  bool buf_absent = (arts_shared_get(buf_h) == NULL);
  arts_db_buf_release(&buf_h);
  if (db_size > 0 && buf_absent) {
    arts_db_buf_install(cache, /*new_version=*/1, /*data_payload=*/NULL,
                        db_size);
  }
}
