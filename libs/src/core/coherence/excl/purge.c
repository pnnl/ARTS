/* SPDX-License-Identifier: Apache-2.0
 *
 * RWLOCK protocol, HOME placement.
 *
 * The home holds the canonical payload, so it both arbitrates and serves: a
 * write grant leaves the home carrying the bytes, and a reader phase is fanned
 * out from the home's own buffer.  For that to be sound the bytes must be back
 * before anyone else can be granted, which is why a release carries the payload
 * to the home and hands the grant back in the SAME message — the grant cannot
 * outlive the write-through, and returning it costs nothing once the payload has
 * to travel anyway.  A rank's next write therefore re-requests from the home and
 * receives the payload again.
 *
 * That is the whole difference from the OWNER placement, where the payload stays
 * with the last writer and a release with nothing pending sends nothing at all.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=EXCL + ARTS_RELEASE_POLICY=PURGE;
 * the placement variant is a link-time file choice, so this TU contains no
 * placement preprocessor guards.
 */

/* lock/types.h must precede all other coherence headers: it defines
 * arts_db_cache_s, arts_db_s, LOCK_HELD_*, arts_db_excl_waiter_s, and the
 * arts_home_grantreq_queue_s for the RWLOCK build (coherence.h and handlers.h
 * declare functions that take these types by pointer). */
#include "arts/counter/object_counter.h"
#include "arts/coherence/excl/types.h"

#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/directory.h"
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/identity.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== RO waiter node (rank-granular RO queue entry) ===================
 * Mirrors the definition in home.c (both TUs include lock/types.h which does
 * not define this node — it is a file-local type used only by home-side grant
 * and teardown logic in home.c, and by arts_send_db_excl_grant here which
 * calls arts_handler_db_excl_grant as a local-hit self-send). */
struct arts_lock_ro_node_s {
  arts_lf_link_t link; /* FIRST */
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv; /* requester's grant landing */
};

/* ===== Pure state-transition arbiters ==================================
 * excl_compute_next and cache_compute_next are pure functions (no external
 * calls, no global state).  They live in arbiters.c so the unit test
 * (tests/unit/excl_compute_next.c) can #include just the arbiters without
 * pulling in the full handler bodies. */
#include "arbiters.c"
#include "arts/counter/Preamble.h"

/* ===== Grant dispatcher (shared by request + release handlers) ========= */

/* Acquire the home buffer's data for a grant payload and fan-out grants.
 * After pop (rw) / drain (ro), for each target rank: self → direct handler
 * call (local hit, wire 0) / remote → MSG_DB_EXCL_GRANT. */
static void lock_home_grant(struct arts_db_s *db, struct arts_db_cache_s *cache,
                            uint32_t grant) {
  if (grant == LOCK_GRANT_NONE) {
    return;
  }
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t data_size = buf ? cache->db_size : 0;
  /* The grant carries the home buffer's own version.  No separate counter is
   * needed: home commits (RW publish installs) are sequentialized by the
   * protocol (RW single-owner inter-node + ACK-gated release), so buf->version
   * is monotone across rounds. */
  uint64_t version = buf ? buf->version : 0;
  if (grant == LOCK_GRANT_ONE_RW) {
    unsigned int rank;
    struct arts_rdzv_landing_s rdzv;
    if (arts_home_grantreq_queue_pop(&db->rw_waiters, &rank, &rdzv)) {
      /* RW grant: advertise home's stable buffer as THIS grant's publish
       * landing (grants and RW releases pair 1:1); the releaser PUTs its
       * dirty bytes straight into it.  In-place is safe at home for the same
       * reason the install always was: the global RW lock excludes every
       * reader while the publish is in flight. */
      struct arts_rdzv_landing_s pub = {0, 0, 0, 0};
      if (buf != NULL && rank != arts_global_rank_id &&
          arts_net_rdzv_local(buf->data, data_size, &pub.addr, &pub.key)) {
        pub.txid = arts_net_rdzv_txid_next();
        pub.cookie = 0; /* in-place: home finds its buffer via the cache */
      }
      arts_send_db_excl_grant(rank, cache->db_guid, DB_MODE_RW, version, &rdzv,
                              &pub, buf_h, data_size);
      buf_h = NULL; /* consumed by the grant sender */
    }
  } else { /* LOCK_GRANT_ALL_RO: drain ro_waiters, fan-out to each rank. */
    arts_lf_link_t *node = arts_lf_stack_drain(&db->ro_waiters);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_lock_ro_node_s *rn =
          ARTS_CONTAINER_OF(node, struct arts_lock_ro_node_s, link);
      /* Each fan-out PUT pins the source with its own strong ref. */
      arts_send_db_excl_grant(rn->rank, cache->db_guid, DB_MODE_RO, version,
                              &rn->rdzv, NULL,
                              (buf != NULL) ? arts_shared_copy(buf_h) : NULL,
                              data_size);
      arts_free(rn);
      node = nx;
    }
  }
  if (buf_h != NULL) {
    arts_db_buf_release(&buf_h);
  }
}

/* ===== arts_send_db_excl_grant ========================================= */

void arts_send_db_excl_grant(unsigned int requester_rank, arts_guid_t db_guid,
                             arts_db_access_mode_t mode, uint64_t version,
                             const struct arts_rdzv_landing_s *req_rdzv,
                             const struct arts_rdzv_landing_s *pub,
                             arts_shared_ptr_t src_h, uint64_t data_size) {
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  struct arts_msg_excl_grant_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_GRANT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.version = version;
  p.data_size = 0;
  p.rdzv_txid = 0;
  p.rdzv_cookie = (req_rdzv != NULL) ? req_rdzv->cookie : 0;
  if (pub != NULL) {
    p.pub.addr = pub->addr;
    p.pub.key = pub->key;
    p.pub.txid = pub->txid;
    p.pub.cookie = pub->cookie;
  } else {
    p.pub = (struct arts_msg_rdzv_landing_s){0, 0, 0, 0};
  }
  if (requester_rank == arts_global_rank_id) {
    /* Self-send (home == requester): the granted buffer already sits in this
     * rank's own cache (home and requester name the same single stable buffer),
     * so there is nothing to serialize or install — copying the whole DB out to
     * a packet and back into the same buffer would be pure waste.  Emit a
     * data-less self-grant: the handler skips the in-place install and runs the
     * grant commit (cache-state transition + waiter drain) against the live
     * buffer.  The requester's advertised landing goes unused — for RWLOCK it is
     * the stable buffer itself, so there is nothing to recycle. */
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    p.data_size = 0;
    p.header.size = sizeof(p);
    arts_handler_db_excl_grant(&p, sizeof(p));
    return;
  }
  if (src == NULL || req_rdzv == NULL || req_rdzv->txid == 0) {
    /* Data-less grant (sentinel DB / nothing published). */
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
    return;
  }
  /* One-sided grant: PUT straight from the home buffer into the requester's
   * stable-buffer landing (RWLOCK's fixed-address install), pairing packet and
   * write completion by txid.  The strong ref transfers to the PUT's local
   * completion, so a concurrent publish recycling the buffer cannot free
   * the bytes mid-read. */
  uint64_t ds = data_size;
  p.data_size = ds;
  p.rdzv_txid = req_rdzv->txid;
  arts_net_put_payload((int)requester_rank, req_rdzv->addr, req_rdzv->key,
                       req_rdzv->txid, src->data, ds,
                       arts_db_buf_ref_release_cb, (void *)src_h);
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

/* ===== arts_handler_db_excl_request ==================================== */

void arts_handler_db_excl_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_ooo_args_db_excl_request_s *a =
      (struct arts_ooo_args_db_excl_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)a->mode;

  if (a->rdzv.txid == 0 && cache->db_size > 0 && arts_global_rank_count > 1) {
    /* First-touch request without a landing: the requester did not know
     * db_size.  Answer with the size (CTS) and do NOT enqueue — the grant
     * plane requires a landing.  The requester re-issues with one. */
    arts_send_db_excl_cts(requester, cache->db_guid, cache->db_size,
                          (uint32_t)mode);
    return;
  }

  /* (1) push-before-CAS: enqueue this requester rank in its mode's queue
   * BEFORE reading lock_state, so the CAS transition already sees this
   * participant counted. */
  if (mode == DB_MODE_RW) {
    arts_home_grantreq_queue_push(&db->rw_waiters, requester, &a->rdzv);
  } else {
    struct arts_lock_ro_node_s *n =
        (struct arts_lock_ro_node_s *)arts_malloc(sizeof(*n));
    n->rank = requester;
    n->rdzv = a->rdzv;
    arts_lf_stack_push(&db->ro_waiters, &n->link);
  }
  arts_rank_bitset_set(&db->cached_ranks,
                       requester); /* destroy fan-out roster */

  /* (2) read -> compute next -> CAS retry on contention. */
  int op = (mode == DB_MODE_RW) ? LOCK_OP_RW_ACQ : LOCK_OP_RO_ACQ;
  uint32_t grant;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    next = excl_compute_next(cur, op, &grant);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_acq_rel, memory_order_acquire));

  /* (3) grant per the transition case. */
  lock_home_grant(db, cache, grant);
}

/* ===== arts_handler_db_excl_release ==================================== */

/* Steps (2)+(3) of the release: the lock_state transition + next grant.
 * Factored out so the rendezvous continuation (dirty bytes landed) runs the
 * identical commit. */
static void lock_release_commit(struct arts_db_s *db,
                                struct arts_db_cache_s *cache,
                                arts_db_access_mode_t mode) {
  int op = (mode == DB_MODE_RW) ? LOCK_OP_RW_REL : LOCK_OP_RO_REL;
  uint32_t grant;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    next = excl_compute_next(cur, op, &grant);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_release, memory_order_acquire));
  lock_home_grant(db, cache, grant);
}

/* Rendezvous continuation for a committed remote RW release: the dirty bytes
 * have fully landed IN PLACE in home's stable buffer ("imm seen => buffer
 * valid"; the global RW lock excluded every reader while they flew), so there
 * is nothing to install — ACK the blocked releaser, then transition + grant. */
struct lock_release_landed_ctx_s {
  arts_shared_ptr_t db_h;
  arts_guid_t db_guid;
  unsigned int releaser;
  uint64_t cv;
};

static void lock_release_landed_cb(void *arg) {
  struct lock_release_landed_ctx_s *ctx =
      (struct lock_release_landed_ctx_s *)arg;
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(ctx->db_h);
  if (ctx->cv != 0) {
    /* ACK even when the DB was destroyed mid-round — a torn-down home cache
     * must never strand the blocked releaser. */
    arts_send_db_excl_release_ack(ctx->releaser, ctx->db_guid, ctx->cv);
  }
  if (db != NULL) {
    lock_release_commit(db, &db->cache, DB_MODE_RW);
  }
  arts_shared_release(&ctx->db_h);
  arts_free(ctx);
}

void arts_handler_db_excl_release(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_ooo_args_db_excl_release_s *a =
      (struct arts_ooo_args_db_excl_release_s *)args_v;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)a->mode;

  if (mode == DB_MODE_RW && a->data_size > 0 && a->data_inline == 0 &&
      a->rdzv_txid == 0) {
    /* Announce: the releaser holds dirty bytes but its RW hold carried no
     * grant-provided publish landing (a creator-seeded hold never received
     * a grant).  Advertise home's stable buffer (fresh txid) and let the
     * releaser PUT + commit; nothing transitions yet.  In-place is safe by
     * the same exclusion argument as the grant-advertised landing: the
     * releaser holds the global RW lock, so no reader anywhere touches the
     * buffer while the publish flies. */
    struct arts_rdzv_landing_s landing = {0, 0, 0, 0};
    arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
    struct arts_db_buffer_s *buf =
        (struct arts_db_buffer_s *)arts_shared_get(buf_h);
    if (buf == NULL) {
      /* No home buffer to land in (never installed): materialize the stable
       * buffer first — the PUT fully overwrites it. */
      arts_db_buf_write_inplace(cache, NULL, a->data_size);
      buf_h = arts_db_buf_acquire(cache);
      buf = (struct arts_db_buffer_s *)arts_shared_get(buf_h);
    }
    if (buf == NULL ||
        !arts_net_rdzv_local(buf->data, a->data_size, &landing.addr,
                             &landing.key)) {
      ARTS_ERROR("lock: home stable buffer is not fabric-registered — "
                 "one-sided publish requires the registered pool");
    }
    landing.txid = arts_net_rdzv_txid_next();
    landing.cookie = 0; /* in-place: home finds its buffer via the cache */
    arts_db_buf_release(&buf_h);
    arts_send_db_publish_cts(a->releaser, a->db_guid, &landing, a->cv);
    return;
  }
  if (mode == DB_MODE_RW && a->data_size > 0 && a->data_inline == 0 &&
      a->rdzv_txid != 0) {
    /* Remote dirty release: the publish PUT straight into home's stable
     * buffer (the landing advertised in this grant).  Pair the release packet
     * with the write completion; ACK + transition + grant run only once the
     * bytes are fully placed. */
    struct lock_release_landed_ctx_s *ctx =
        (struct lock_release_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = arts_route_table_lookup_db(cache->db_guid);
    ctx->db_guid = a->db_guid;
    ctx->releaser = a->releaser;
    ctx->cv = a->cv;
    arts_net_rdzv_expect(a->rdzv_txid, lock_release_landed_cb, ctx);
    return;
  }

  /* (1) RW same-rank: write the publish into home's stable buffer in place,
   * then ACK.  Under exclusive-lock serialization the releaser held the sole
   * RW grant and home grants the next holder only after this publish
   * completes, so no reader is touching the buffer here — the in-place
   * overwrite is safe and the buffer address stays fixed (preserving DBs with
   * internal self-pointers).  No versioning: home commits are already
   * sequentialized by the protocol (RW single-owner + ACK-gated release). */
  if (mode == DB_MODE_RW && a->data_size > 0 && a->data_inline != 0) {
    const void *data = (const char *)a + sizeof(*a);
    arts_db_buf_write_inplace(cache, data, a->data_size);
  }
  if (mode == DB_MODE_RW && a->cv != 0) {
    arts_send_db_excl_release_ack(a->releaser, a->db_guid, a->cv);
  }

  /* (2)+(3) transition + grant. */
  lock_release_commit(db, cache, mode);
}

/* ===== arts_db_acquire_is_serialized ===================================
 * RWLOCK: both RW and RO are blocking locks — both are GUID-serialized so
 * the engine acquires them in a global order (deadlock-free lock ordering). */
bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW || mode == DB_MODE_RO;
}

/* ===== arts_db_cache_init ==============================================
 * Initialize the per-rank cache for the RWLOCK protocol.  For RWLOCK, the home-rank
 * lock_state and the per-cache cache_state single word are the coherence state;
 * arts_db_cache_common_init handles the shared fields (db_guid, db_size,
 * pending_snapshot, home-directory init).
 *
 * Creator hold: arts_db_create defaults to an RW acquire (OCR contract; only
 * ARTS_DB_PROP_NO_ACQUIRE skips it), so the creator EDT must hold the lock RW
 * exactly like any granted writer — otherwise its matching arts_db_release (or
 * the EDT-epilogue auto-release) would run against a hold that was never taken.
 * For the creator/home kinds we therefore SEED the cache held in RW
 * (rw_state=GRANT, rw_count=1); the home lock_state is seeded w=1 in
 * arts_db_home_init.  The release path drives both back to 0 — the standard
 * acquire/release pair, no protocol-specific create bookkeeping.  (NO_ACQUIRE
 * resets these to idle in arts_db_create, mirroring the single-owner arms.) */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  uint64_t seed = 0ULL; /* rw_state=ro_state=IDLE, both counts 0 */
  if (kind == ARTS_DB_INIT_CREATOR_HOME ||
      kind == ARTS_DB_INIT_CREATOR_REMOTE) {
    /* Creator holds RW: GRANT + one writer. */
    seed = CACHE_MAKE(CACHE_ST_GRANT, CACHE_ST_IDLE, 1u, 0u);
  }
  atomic_store_explicit(&c->cache_state, seed, memory_order_relaxed);
  arts_lf_stack_init(&c->ro_pending);
  arts_lf_stack_init(&c->rw_pending);
  c->home_pub_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

/* ===== arts_db_cache_destructor =========================================
 * free parked nodes only — destroying an in-use DB is undefined (OCR), no
 * wake.
 *
 * Destructor runs at refcount 0 as the SOLE owner of the object; no concurrent
 * actor remains, so it is the single safe drainer of the parked-waiter stacks.
 * Destroy itself is just the route-slot detach (CAS value→NULL + drop the
 * install ref). */
void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache); /* buffer-NULL FIRST */
  /* Drain + free the parked-waiter stacks (empty in the normal case — a legit
   * destroy has no outstanding acquirers). */
  for (arts_lf_stack_t *q = &cache->ro_pending;; q = &cache->rw_pending) {
    arts_lf_link_t *node = arts_lf_stack_drain(q);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_db_excl_waiter_s *w =
          ARTS_CONTAINER_OF(node, struct arts_db_excl_waiter_s, link);
      arts_free(w);
      node = nx;
    }
    if (q == &cache->rw_pending) {
      break;
    }
  }
  arts_db_cache_common_destroy_post(cache); /* snapshot free → home teardown */
}

/* ===== arts_db_create_publish_holder ====================================
 * RWLOCK home init: nothing to publish for the holder field — RWLOCK tracks
 * mode via lock_state, not rw_holder.  The creator becomes the first RW
 * holder via the normal acquire/self-grant chain, so there is nothing to
 * publish at create. */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  (void)db;
  (void)creator_rank;
}

/* ===== arts_db_create_install_home_buffer ================================
 * RWLOCK home init: install the zero-init home buffer at creation time so the
 * home always holds the canonical backing store.  The first GRANT carries
 * this data (empty / zero at first) to the requester; the requester's first
 * RW release sends back the updated contents via LOCK_RELEASE publish. */
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  if (db_size > 0) {
    arts_db_buf_write_inplace(cache, /*data=*/NULL, db_size); /* zero-init */
  }
}

/* ===== arts_send_db_excl_request ========================================
 * Send MSG_DB_EXCL_REQUEST to the home rank.  Self-send (home == this rank)
 * dispatches through the OoO engine (HIT runs inline; MISS defers until the
 * home db_s is installed).  Remote send goes via the transport. */
/* Materialize this rank's stable buffer (RWLOCK's fixed-address backing store)
 * and advertise it as the grant landing: the grant PUT installs IN PLACE,
 * preserving the address across the DB's whole lifetime.  A fresh txid is
 * drawn per request (each request is served by at most one grant). */
static bool lock_stable_landing(struct arts_db_cache_s *cache,
                                struct arts_rdzv_landing_s *out) {
  *out = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  if (cache->db_size == 0 || arts_global_rank_count <= 1) {
    return false;
  }
  arts_shared_ptr_t h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf = (struct arts_db_buffer_s *)arts_shared_get(h);
  if (buf == NULL) {
    /* First touch: allocate the one stable buffer (zero-filled — the grant
     * PUT fully overwrites it before any drained waiter reads). */
    arts_db_buf_write_inplace(cache, NULL, cache->db_size);
    h = arts_db_buf_acquire(cache);
    buf = (struct arts_db_buffer_s *)arts_shared_get(h);
  }
  if (buf == NULL ||
      !arts_net_rdzv_local(buf->data, cache->db_size, &out->addr, &out->key)) {
    arts_db_buf_release(&h);
    return false;
  }
  out->txid = arts_net_rdzv_txid_next();
  out->cookie = 0; /* in-place: this rank finds the buffer via its cache */
  arts_db_buf_release(&h);
  return true;
}

void arts_send_db_excl_request(struct arts_db_cache_s *cache,
                               arts_db_access_mode_t mode) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  struct arts_rdzv_landing_s rdzv;
  (void)lock_stable_landing(cache, &rdzv);
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine so before-create reorders are
     * handled correctly (the engine defers when the slot is absent). */
    struct arts_ooo_args_db_excl_request_s args = {
        .requester = arts_global_rank_id,
        .db_guid = db_guid,
        .mode = (uint32_t)mode,
        .rdzv = rdzv,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_EXCL_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  struct arts_msg_excl_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_REQUEST);
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.rdzv.addr = rdzv.addr;
  p.rdzv.key = rdzv.key;
  p.rdzv.txid = rdzv.txid;
  p.rdzv.cookie = rdzv.cookie;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* LOCK_CTS sender (home → first-touch requester) + requester-side body. */
void arts_send_db_excl_cts(unsigned int requester_rank, arts_guid_t db_guid,
                           uint64_t db_size, uint32_t mode) {
  struct arts_msg_excl_cts_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_CTS);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.db_size = db_size;
  p.mode = mode;
  p.pad = 0;
  if (requester_rank == arts_global_rank_id) {
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_excl_cts(db, &p);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

void arts_handler_db_excl_cts(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_msg_excl_cts_packet_s *p =
      (struct arts_msg_excl_cts_packet_s *)args_v;
  if (cache->db_size == 0) {
    cache->db_size = p->db_size;
  }
  arts_send_db_excl_request(cache, (arts_db_access_mode_t)p->mode);
}

/* Serve exactly `expected` waiters from `q` — the parked population the
 * caller's GRANT CAS observed in the counts, which is exact: counts return to
 * zero at every 0-edge, arrivals under a held grant self-serve (never push),
 * so everything counted at the grant is parked (or about to be).  The grant
 * committer is the SOLE drainer; each whole-stack atomic-exchange batch is
 * consumed once (exactly-once serve).  A counted waiter whose push has not
 * landed yet — the instruction window between its acquire CAS and its push —
 * is awaited by re-draining, the same bounded-window retry discipline as the
 * home queue's mid-push pop.  Serving = cursor-advance + account
 * (mark_edt_secured + mark_edt_ready); it does NOT touch the count (the
 * count was carried by the waiter's acquire CAS — "counted ⟹ will park"). */
static void lock_drain_pending(arts_lf_stack_t *q, uint32_t expected) {
  uint32_t served = 0;
  while (served < expected) {
    arts_lf_link_t *node = arts_lf_stack_drain(q);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_db_excl_waiter_s *w =
          ARTS_CONTAINER_OF(node, struct arts_db_excl_waiter_s, link);
      /* Position-idempotent cursor advance (fires next serialized dep), then
       * re-derive dep->ptr from the installed buffer + account (may schedule
       * the EDT when acquire_remaining reaches 0). */
      mark_edt_secured_by_guid(w->edt_guid, w->slot);
      mark_edt_ready_by_guid(w->edt_guid, w->slot);
      arts_free(w);
      served++;
      node = nx;
    }
  }
}

/* ===== arts_handler_db_acquire =========================================
 * OOO_DB_ACQUIRE Cat-B body — protocol-agnostic signature.
 *
 * item is the pre-pinned home db_s; args is {edt, db_guid, slot}; mode is read
 * from depv[slot].mode.
 *
 * Single-atom acquire: ONE CAS carries {count++, decision} together (see
 * cache_compute_next in arbiters.c for why the atomicity is load-bearing).
 * The action decides where this acquire is served:
 *   SELF_SERVE  a covering grant is held — serve the dep directly.  It never
 *               touches the pend stacks, which keeps the parked population
 *               equal to the counts a later grant CAS reads.
 *   SEND_*      this acquire opened the round's request: park the waiter
 *               FIRST, then send — by the time any grant for the request can
 *               exist, the opener's node is already drainable.
 *   PARK        covered by the in-flight request: park.  The grant committer
 *               serves exactly the population counted at its grant CAS, so a
 *               node whose push trails the grant is awaited, never lost. */
void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  arts_db_access_mode_t mode = depv[slot].mode;

  int op = (mode == DB_MODE_RW) ? CACHE_OP_ACQ_RW : CACHE_OP_ACQ_RO;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    next = cache_compute_next(cur, op, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));

  if (act == CACHE_ACT_SELF_SERVE) {
    /* Position-idempotent cursor advance + account, exactly as a drained
     * waiter is served.  The turn was answered from what this rank already
     * holds, so it joins the same census the other arms feed. */
    INCREMENT_NUM_DB_ACQUIRE_LOCAL_HIT_BY(1);
    arts_object_acquire(false);
    mark_edt_secured_by_guid(edt->guid, slot);
    mark_edt_ready_by_guid(edt->guid, slot);
    return;
  }
  INCREMENT_NUM_DB_ACQUIRE_REMOTE_BY(1);
  arts_object_acquire(true);

  /* SEND_* / PARK: park the waiter.  Push BEFORE any send so the node is
   * drainable before a grant for this round can arrive. */
  struct arts_db_excl_waiter_s *w =
      (struct arts_db_excl_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt->guid;
  w->slot = slot;
  arts_lf_stack_push(
      (mode == DB_MODE_RW) ? &cache->rw_pending : &cache->ro_pending, &w->link);

  if (act == CACHE_ACT_SEND_RW) {
    arts_send_db_excl_request(cache, DB_MODE_RW);
  } else if (act == CACHE_ACT_SEND_RO) {
    arts_send_db_excl_request(cache, DB_MODE_RO);
  }
}

/* ===== arts_handler_db_excl_grant =======================================
 * Cat-C pure body — grant arrived at the requester rank.  payload is the full
 * contiguous wire buffer (header + db_size data bytes); size is the byte count.
 *
 *   (1) Cat-C route-table lookup (NULL ⇒ DB destroyed concurrently ⇒ drop).
 *   (2) Install the grant data BEFORE the CAS so drained waiters observe it.
 *   (3) CAS the single word: RW grant Q→GRANT (action DRAIN_BOTH); RO grant
 *       Q→GRANT (DRAIN_RO) or, when rc==0 (the RO waiters were already served
 *       by an RW grant — RW⊇RO), a phantom Q→IDLE that is returned to home at
 *       once (REL_RO).
 *   (4) Run the action after the CAS commits (mirrors the home grant path:
 *       compute decides, the pop/drain/send runs after).
 *
 * The grant committer is the SOLE drainer of the pend stacks: the counts its
 * CAS observed are exactly the parked population (arrivals under the held
 * grant self-serve and never park), and lock_drain_pending consumes exactly
 * that many nodes — serving is exactly-once with no other drain path that
 * could run against a later round. */
/* Steps (3)+(4) of the grant: the cache_state transition + resulting drain /
 * phantom return.  Factored out so the rendezvous continuation (grant bytes
 * landed) runs the identical commit.  Consumes db_h. */
static void lock_grant_commit(arts_shared_ptr_t db_h, arts_guid_t db_guid,
                              arts_db_access_mode_t mode,
                              const struct arts_rdzv_landing_s *pub) {
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  struct arts_db_cache_s *cache = &db->cache;

  /* Stash home's publish landing for this grant's eventual RW release —
   * BEFORE the CAS that lets local writers run (the single ACK-gated releaser
   * consumes it). */
  if (mode == DB_MODE_RW && pub != NULL) {
    cache->home_pub_rdzv = *pub;
  }

  int op = (mode == DB_MODE_RW) ? CACHE_OP_GRANT_RW : CACHE_OP_GRANT_RO;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    next = cache_compute_next(cur, op, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));

  switch (act) {
  case CACHE_ACT_DRAIN_BOTH: /* RW grant serves this rank's RW + RO cohort */
    lock_drain_pending(&cache->rw_pending, CACHE_RW_CNT(cur));
    lock_drain_pending(&cache->ro_pending, CACHE_RO_CNT(cur));
    break;
  case CACHE_ACT_DRAIN_RO:
    lock_drain_pending(&cache->ro_pending, CACHE_RO_CNT(cur));
    break;
  case CACHE_ACT_REL_RO: /* phantom RO grant: nothing to serve, return home */
    arts_send_db_excl_release(arts_guid_get_rank(db_guid), db_guid, DB_MODE_RO,
                              /*version=*/0u, /*cv=*/0u, NULL, 0u,
                              /*rdzv_txid=*/0u, /*rdzv_cookie=*/0u);
    break;
  default:
    break;
  }

  arts_shared_release(&db_h);
}

/* Rendezvous continuation: the grant bytes have fully landed IN PLACE in this
 * rank's stable buffer (RWLOCK's fixed-address install; no local holder exists
 * while a grant is in flight — the global lock excluded us).  Nothing to
 * install; run the commit. */
struct lock_grant_landed_ctx_s {
  arts_shared_ptr_t db_h;
  arts_guid_t db_guid;
  arts_db_access_mode_t mode;
  struct arts_rdzv_landing_s pub;
};

static void lock_grant_landed_cb(void *arg) {
  struct lock_grant_landed_ctx_s *ctx = (struct lock_grant_landed_ctx_s *)arg;
  if (arts_shared_get(ctx->db_h) != NULL) {
    lock_grant_commit(ctx->db_h, ctx->db_guid, ctx->mode, &ctx->pub);
  } else {
    arts_shared_release(&ctx->db_h); /* destroyed mid-flight — drop */
  }
  arts_free(ctx);
}

void arts_handler_db_excl_grant(void *payload, size_t size) {
  struct arts_msg_excl_grant_packet_s *p =
      (struct arts_msg_excl_grant_packet_s *)payload;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)p->mode;
  const void *data = (const char *)p + sizeof(*p);
  uint64_t data_size = (uint64_t)size - (uint64_t)sizeof(*p);

  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    /* Destroyed mid-flight: still consume any pairing so the txid table stays
     * leak-free (RWLOCK landings are in-place, cookie 0 — nothing to free). */
    arts_db_rdzv_discard_landing(p->rdzv_txid, p->rdzv_cookie);
    arts_shared_release(&db_h);
    return;
  }
  struct arts_db_cache_s *cache = &db->cache;

  struct arts_rdzv_landing_s pub = {p->pub.addr, p->pub.key, p->pub.txid,
                                   p->pub.cookie};

  if (p->rdzv_txid != 0) {
    /* The grant payload travels one-sided into our stable buffer; pair this
     * packet with the write completion (either order), then commit. */
    struct lock_grant_landed_ctx_s *ctx =
        (struct lock_grant_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->db_guid = p->db_guid;
    ctx->mode = mode;
    ctx->pub = pub;
    arts_net_rdzv_expect(p->rdzv_txid, lock_grant_landed_cb, ctx);
    return;
  }

  /* Same-rank / data-less grant: install any inline bytes in place (exclusive-
   * lock serialization — no local holder; duplicate bytes are identical), then
   * commit. */
  if (data_size > 0u) {
    arts_db_buf_write_inplace(cache, data, data_size);
  }
  lock_grant_commit(db_h, p->db_guid, mode, &pub); /* consumes db_h */
}

/* ===== arts_send_db_excl_release ========================================
 * Send LOCK_RELEASE to the home.  RW carries publish data, version, and cv
 * (the releaser's stack-local sem_t address so home can echo it in the ACK);
 * RO carries none and passes version=0 / cv=0.
 * Self-send (home == this rank) routes through the OoO engine so reordering
 * against DB_CREATE is handled identically to the wire path. */
void arts_send_db_excl_release(unsigned int home_rank, arts_guid_t db_guid,
                               arts_db_access_mode_t mode, uint64_t version,
                               uint64_t cv, const void *data,
                               uint64_t data_size, uint64_t rdzv_txid,
                               uint64_t rdzv_cookie) {
  uint64_t ds = (mode == DB_MODE_RW) ? data_size : 0u;

  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine exactly as the wire RX
     * dispatcher does — HIT runs arts_handler_db_excl_release inline (which
     * posts cv); MISS defers the args until the home db_s is installed.  A
     * data-less release (the common self case) carries stack args directly;
     * only an inline-data release builds the contiguous args buffer. */
    if (data == NULL || ds == 0u) {
      struct arts_ooo_args_db_excl_release_s args = {
          .releaser = arts_global_rank_id,
          .db_guid = db_guid,
          .mode = (uint32_t)mode,
          .data_size = 0,
          .cv = cv,
          .version = version,
          .rdzv_txid = 0,
          .rdzv_cookie = 0,
          .data_inline = 0,
      };
      arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_EXCL_RELEASE, &args,
                                      sizeof(args));
      return;
    }
    uint32_t asz =
        (uint32_t)(sizeof(struct arts_ooo_args_db_excl_release_s) + ds);
    char *abuf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_excl_release_s *args =
        (struct arts_ooo_args_db_excl_release_s *)abuf;
    args->releaser = arts_global_rank_id;
    args->db_guid = db_guid;
    args->mode = (uint32_t)mode;
    args->data_size = ds;
    args->cv = cv;
    args->version = version;
    args->rdzv_txid = 0;
    args->rdzv_cookie = 0;
    args->data_inline = 1u;
    memcpy(abuf + sizeof(*args), data, (size_t)ds);
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_EXCL_RELEASE, abuf, asz);
    arts_free(abuf);
    return;
  }

  /* Remote send: control-only — a dirty RW release PUT its bytes into the
   * grant's home landing before this packet; {rdzv_txid, rdzv_cookie} echo it
   * for pairing. */
  (void)data;
  struct arts_msg_excl_release_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_RELEASE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.version = version;
  p.cv = cv;
  p.data_size = ds;
  p.rdzv_txid = rdzv_txid;
  p.rdzv_cookie = rdzv_cookie;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* ===== release-edge senders ============================================
 * Called AFTER the release CAS has already moved the state word to IDLE for
 * this phase (so for a self-send, the inline grant handler that follows will
 * CAS the next phase onto an already-IDLE word — no overwrite).
 *
 * RW → synchronous publish: ship buf->data with a stack-local sem_t cv
 *      token; home echoes it in LOCK_RELEASE_ACK and await_publish_ack
 *      returns only after the post (no lost update across TCP; self-send posts
 *      inline and returns at once).
 * RO → data-less fire-and-forget notify. */
static void lock_send_release_rw(struct arts_db_cache_s *cache) {
  unsigned int home = (unsigned int)arts_guid_get_rank(cache->db_guid);
  if (home == arts_global_rank_id) {
    /* Home-local RW release: this rank IS home, so the releaser already holds
     * home's authoritative buffer (its in-place writes have landed) and the
     * caller keeps the descriptor pinned across this call.  There is nothing to
     * ship and no ACK to await — apply the home lock_state commit + onward
     * grant directly on the live cache.  Routing this through a GUID-keyed
     * self-send would re-resolve the DB via its route slot, which a concurrent
     * (legal) destroy may have already detached while this holder still owed
     * its release; the self-send would then MISS, defer on the OoO list
     * forever, and strand this worker in await_publish_ack.  A release
     * provably follows a successful acquire, so it never needs the OoO
     * before-create deferral that the request path relies on. */
    lock_release_commit(arts_db_of_cache(cache), cache, DB_MODE_RW);
    return;
  }
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  const void *data = (buf != NULL) ? buf->data : NULL;
  uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
  /* Consume this grant's home publish landing (1:1 grant:release). */
  struct arts_rdzv_landing_s pub = cache->home_pub_rdzv;
  cache->home_pub_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  if (home != arts_global_rank_id && ds > 0u && pub.txid == 0) {
    /* Dirty release from a hold that never received a grant (the
     * creator-seeded RW grant): no cached landing — run the announce leg.
     * Heap rendezvous, deliberately leaked on the shutdown escape (a late
     * CTS/ACK writes/posts through the echoed cv; see the HOME-placement
     * publish sync for the same discipline). */
    struct arts_db_pub_rendezvous_s *wr =
        (struct arts_db_pub_rendezvous_s *)arts_malloc(sizeof(*wr));
    sem_init(&wr->sem, 0, 0);
    wr->landing = (struct arts_rdzv_landing_s){0, 0, 0, 0};
    arts_send_db_excl_release(home, cache->db_guid, DB_MODE_RW,
                              /*version=*/0u, (uint64_t)(uintptr_t)wr,
                              /*data=*/NULL, ds, /*rdzv_txid=*/0u,
                              /*rdzv_cookie=*/0u);
    await_publish_ack(&wr->sem); /* CTS wake — or the shutdown escape */
    if (wr->landing.txid == 0) {
      /* This read of wr->landing.txid is UNSYNCHRONIZED on the shutdown-
       * escape path — sem_timedwait returned via the shutdown timeout, not a
       * real post, so there is no happens-before edge against a CTS reply
       * that races in concurrently.  That is precisely why wr is leaked
       * instead of freed: do not "tighten" this into an immediate free, or a
       * late racing write turns it into a use-after-free. */
      arts_db_buf_release(&buf_h);
      return; /* shutdown escape: round abandoned with the runtime — leak wr */
    }
    /* buf_h (held until after the ACK) pins the source; the ACK follows the
     * target-side completion, which implies the fabric drained it. */
    arts_net_put_payload((int)home, wr->landing.addr, wr->landing.key,
                         wr->landing.txid, data, ds, /*on_local_done=*/NULL,
                         NULL);
    arts_send_db_excl_release(home, cache->db_guid, DB_MODE_RW,
                              /*version=*/0u, (uint64_t)(uintptr_t)wr,
                              /*data=*/NULL, ds, wr->landing.txid,
                              wr->landing.cookie);
    await_publish_ack(&wr->sem); /* install ACK */
    if (arts_atomic_read(&arts_node_info.shutdown_state) == 0) {
      sem_destroy(&wr->sem);
      arts_free(wr);
    }
    arts_db_buf_release(&buf_h);
    return;
  }
  sem_t cv;
  sem_init(&cv, 0, 0);
  uint64_t cv_token = (uint64_t)(uintptr_t)&cv;
  if (home != arts_global_rank_id && ds > 0u && pub.txid != 0) {
    /* Remote dirty release: PUT the dirty bytes straight into home's stable
     * buffer (the landing advertised in the grant) — the global RW lock
     * excludes every reader while they fly — then send the control-only
     * release packet; home pairs {packet, write completion} before it ACKs
     * and grants onward.  buf_h (held until after the ACK) pins the source
     * bytes; the ACK follows the target-side completion, which implies the
     * fabric fully drained them. */
    arts_net_put_payload((int)home, pub.addr, pub.key, pub.txid, data, ds,
                         /*on_local_done=*/NULL, NULL);
    arts_send_db_excl_release(home, cache->db_guid, DB_MODE_RW, /*version=*/0u,
                              cv_token, /*data=*/NULL, ds, pub.txid, pub.cookie);
  } else {
    /* Same-rank (inline) or data-less release. */
    arts_send_db_excl_release(home, cache->db_guid, DB_MODE_RW, /*version=*/0u,
                              cv_token, data, ds, /*rdzv_txid=*/0u,
                              /*rdzv_cookie=*/0u);
  }
  await_publish_ack(&cv);
  sem_destroy(&cv);
  arts_db_buf_release(&buf_h);
}

static void lock_send_release_ro(struct arts_db_cache_s *cache) {
  unsigned int home = (unsigned int)arts_guid_get_rank(cache->db_guid);
  if (home == arts_global_rank_id) {
    /* Home-local RO release: commit the home lock_state transition + onward
     * grant directly, for the same reason the RW path does — a destroy that
     * detached the route slot must not be able to defer this release forever. */
    lock_release_commit(arts_db_of_cache(cache), cache, DB_MODE_RO);
    return;
  }
  arts_send_db_excl_release(home, cache->db_guid, DB_MODE_RO, /*version=*/0u,
                            /*cv=*/0u, NULL, 0u, /*rdzv_txid=*/0u,
                            /*rdzv_cookie=*/0u);
}

/* ===== arts_db_release_rw / arts_db_release_ro =========================
 * Single-word release: CAS the count down (+ the 0-edge state transition,
 * atomic together) via cache_compute_next, then run the release action it
 * returns.  CACHE_ACT_REL_RW → publish (the last holder of an RW grant, incl.
 * the last RO joiner under it); CACHE_ACT_REL_RO → notify (last RO holder).
 *
 * Destroyed-guard: a dep NULL-woken at destroy never really held the lock; a
 * count of 0 means there is nothing to release — skip rather than underflow.
 * (Full destroy reconciliation under the acquire-time count is a separate
 * subtask.) */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_RW_CNT(cur) == 0u) {
      return; /* destroyed-guard / already released */
    }
    next = cache_compute_next(cur, CACHE_OP_REL_RW, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_REL_RW) {
    lock_send_release_rw(cache);
  }
}

void arts_db_release_ro(struct arts_db_cache_s *cache) {
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_RO_CNT(cur) == 0u) {
      return; /* destroyed-guard / already released */
    }
    next = cache_compute_next(cur, CACHE_OP_REL_RO, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_REL_RW) {
    lock_send_release_rw(
        cache); /* last RO joiner under an RW grant → publish */
  } else if (act == CACHE_ACT_REL_RO) {
    lock_send_release_ro(cache);
  }
}
