/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence acquire path.  Implements the 8-case dispatcher and its
 * supporting routines (lazy_install_cache_s, acquire_remote_rw,
 * acquire_remote_ro, drain_pending_rw_after_grant, drain_pending_ro,
 * trigger_ro_waiter).  See the design plan for the full algorithm;
 * inline comments highlight the trickier race resolutions.
 *
 * Wake mechanism: parked EDTs are tracked via the standard
 * arts_edt_s.depc_needed counter.  When a waiter is triggered, we
 * (1) look up the EDT, (2) write the buffer data pointer into
 * depv[slot].ptr, (3) atomic_sub(depc_needed); if it reaches 0 the
 * EDT becomes ready and is handed to the scheduler.
 */

#include "arts/memory/coherence_acquire.h"

#include <stdlib.h>
#include <string.h>

#include "arts/compute/edt.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/coherence_home.h"
#include "arts/memory/db.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"

/* Forward decl — strong def in coherence_destroy.c. */
void arts_coh_try_finalize_destroy(struct arts_db_cache_s *cache);

/* Adapter: route_table stores arts_db_s* (the previous data layout); the new RC
 * cache_s lives inside db->coherence_cache.  All RC paths look up
 * cache via this helper, so when the cutover removes arts_db_s the
 * change is local to one function.  Returns NULL if either the
 * route_table entry doesn't exist or the entry has no cache_s
 * (e.g. PIN/CXL DBs).
 *
 * this is the last surviving raw arts_route_table_lookup_data
 * caller in libs/.  Migrating it to lookup_db_safe + release would require
 * rewriting all ~15 callers across coherence_acquire.c / coherence_release.c
 * / coherence_handlers.c / db.c to balance the ref — out of scope for the
 * event subsystem rewrite.  Safe in practice because the returned cache_s
 * is heap-allocated separately from arts_db_s (it lives in
 * db->coherence_cache as its own malloc'd struct), and the RC protocol
 * keeps cache_s alive via its own destroy gate (arts_coh_try_finalize_destroy)
 * independent of the route_table slot's lifetime.  Documented as a known
 * exception. */
struct arts_db_cache_s *arts_coh_route_table_lookup_cache(arts_guid_t db_guid) {
  void *data = arts_route_table_lookup_data(db_guid);
  if (data == NULL) {
    return NULL;
  }
  struct arts_db_s *db = (struct arts_db_s *)data;
  if (db->coherence_cache == NULL) {
    return NULL;
  }
  return (struct arts_db_cache_s *)db->coherence_cache;
}
#define coh_lookup_cache arts_coh_route_table_lookup_cache

/* Forward decl from scheduler.c: hand a now-ready EDT to the scheduler. */
extern void arts_handle_remote_stolen_edt(struct arts_edt_s *edt);

/* ===== EDT wake helper ============================================== */

/* Trigger the parked EDT identified by (edt_guid, slot) by writing
 * the dep slot's data pointer and decrementing depc_needed.
 *
 * RC-only: the canonical user data pointer lives in cache->user_data,
 * regardless of whether cache_s was created in-place with arts_db_s
 * (user_data == (db+1)) or lazy-installed standalone (user_data ==
 * malloc'd buffer).  We look up the cache via the GUID of the dep
 * slot's DB and stamp depv[slot].ptr accordingly.
 *
 * Non-static so coherence_destroy.c can wake parked EDTs on
 * destroy-fail (NULL ptr semantics — EDT observes destroyed DB). */
static void mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot);

void arts_coh_mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot) {
  mark_edt_ready_by_guid(edt_guid, slot);
}

static void mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot) {
  if (edt_guid == NULL_GUID) {
    return;
  }
  /* lookup_edt_safe pairs with release at function exit. */
  struct arts_edt_s *edt = arts_route_table_lookup_edt_safe(edt_guid);
  if (edt == NULL) {
    ARTS_INFO("coherence: edt_guid %lu not found at trigger time", edt_guid);
    return;
  }
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  arts_guid_t db_guid = depv[slot].guid;
  if (db_guid != NULL_GUID) {
    struct arts_db_cache_s *cache = coh_lookup_cache(db_guid);
    if (cache != NULL) {
      /* Acquire a fresh buf ref for this EDT; release_one_dep's DIST
       * branch calls release_buf on its dep slot when the EDT finishes.
       * depv[slot].ptr aliases buf->data — the canonical user-visible
       * payload (design plan §Buffer). */
      struct arts_db_buffer_s *buf = arts_coherence_acquire_buf(cache);
      if (buf == NULL) {
        fprintf(stderr,
                "[COH-DBG rank %u] mark_edt_ready: NULL buf for db=%lu "
                "edt=%lu slot=%u wc=%u db_size=%lu\n",
                arts_global_rank_id, (unsigned long)db_guid,
                (unsigned long)edt_guid, slot, cache->writer_count,
                (unsigned long)cache->db_size);
        fflush(stderr);
      }
      depv[slot].ptr = buf ? buf->data : NULL;
    }
  }
  if (arts_atomic_sub(&edt->depc_needed, 1U) == 0) {
    arts_handle_remote_stolen_edt(edt);
  }
  arts_route_table_release(edt_guid);
}

/* ===== Lazy first-touch =========================================== */

/* Allocate a stub arts_db_s + cache_s for a DIST DB that this rank
 * has not yet touched, register it in the route_table, and return
 * the cache_s.  The stub has no user data payload — install_buffer
 * lazy-allocates cache->user_data on first wire arrival.
 *
 * Race-safe: route_table_add_item_race rejects if another thread
 * (e.g. a concurrent wire-receive) raced us; in that case we free
 * our stub and return the existing cache_s.
 *
 * Caller invariant: returns with the route_table ref bumped (via
 * lookup) so subsequent acquire_dbs flow can balance with the
 * usual return_db at release_one_dep time. */
struct arts_db_cache_s *arts_coh_lazy_install_cache_s(arts_guid_t db_guid,
                                                      uint64_t db_size) {
  /* First check if it already exists (someone else lazy-installed or
   * a wire-receive fired). */
  struct arts_db_cache_s *cache = coh_lookup_cache(db_guid);
  if (cache != NULL) {
    return cache;
  }

  /* Allocate stub arts_db_s with no payload (sizeof header only).
   * coherence_cache holds the RC protocol state.  arts_db_s exists purely to
   * satisfy the route_table's typed-entry contract during the
   * dual-stack period. */
  struct arts_db_s *stub =
      (struct arts_db_s *)arts_malloc_align(sizeof(struct arts_db_s), 16);
  memset(stub, 0, sizeof(struct arts_db_s));
  arts_shared_init(&stub->shared, arts_db_get_deleter());
  stub->header.type = ARTS_GUID_DB;
  stub->header.size = sizeof(struct arts_db_s);
  stub->guid = db_guid;
  stub->db_type = ARTS_DB;

  /* db_size==0 ⇒ lazy install: buffer alloc deferred until first wire
   * arrival (install_buffer with the actual db_size).  No home struct
   * yet — even for is_home, the home struct is created when DB_CREATE
   * arrives (with the proper rw_holder = creator_rank). */
  stub->coherence_cache = arts_coh_alloc_cache_s(
      db_guid, /*db_size=*/db_size, ARTS_COH_INIT_LAZY, /*creator_rank=*/0);
  /* back-pointer for try_finalize_destroy direct-free. */
  ((struct arts_db_cache_s *)stub->coherence_cache)->db_owner = stub;

  if (arts_route_table_add_item_race(stub, db_guid, arts_global_rank_id,
                                     /*used=*/true)) {
    arts_route_table_fire_oo(db_guid, arts_out_of_order_handler);
    return (struct arts_db_cache_s *)stub->coherence_cache;
  }

  /* Lost the race — another thread already installed.  Tear down our
   * stub and return the established cache. */
  arts_db_free(stub);
  return coh_lookup_cache(db_guid);
}

/* ===== Case 1/3/5: local-buffer acquire ============================= */

static void *acquire_local(struct arts_db_cache_s *cache) {
  /* Design plan §acquire_local: return buf->data with ref held.
   * release happens in arts_db_release → arts_coherence_release_buf
   * (called from release_one_dep's DIST branch). */
  struct arts_db_buffer_s *buf = arts_coherence_acquire_buf(cache);
  if (buf == NULL) {
    fprintf(stderr,
            "[COH-DBG rank %u] acquire_local: NULL buf for db=%lu "
            "wc=%u db_size=%lu\n",
            arts_global_rank_id, (unsigned long)cache->db_guid,
            cache->writer_count, (unsigned long)cache->db_size);
    fflush(stderr);
    return NULL;
  }
  return buf->data;
}

/* ===== Case 2/6: RW local fast path ================================ */

#ifndef ARTS_MEMORY_MODEL_LC
typedef enum { CASE26_OK = 0, CASE26_FAIL_FALLBACK } case26_result_t;

static case26_result_t acquire_rw_local_fast(struct arts_db_cache_s *cache) {
  /* CAS-loop "increment if positive": never bump from 0. */
  while (1) {
    unsigned int wc = cache->writer_count;
    if (wc == 0) {
      return CASE26_FAIL_FALLBACK; /* ownership invalidated. */
    }
    if (arts_atomic_cswap(&cache->writer_count, wc, wc + 1) == wc) {
      return CASE26_OK;
    }
  }
}

/* ===== Case 4/8: remote-RW path ==================================== */

static arts_db_acquire_result_t acquire_remote_rw(struct arts_db_cache_s *cache,
                                                  arts_guid_t edt_guid,
                                                  unsigned int slot) {
  /* No destroy_state precheck: per spec 4.11, handle_destroy_req NULL-stores
   * route_item->data BEFORE flipping destroy_state, so route_table_lookup_db
   * already misses and the caller's OoO defer handles "DB destroyed".  If
   * we did get here with destroy_state advancing concurrently, the
   * fail_trigger_pending pop/wake-with-NULL chain will catch our waiter.
   */
  /* Allocate + push waiter into MPSC queue. */
  struct arts_db_rw_waiter_s *w =
      (struct arts_db_rw_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt_guid;
  w->slot = slot;
  /* IMPORTANT: bump pending_count BEFORE push.  Reversing this order
   * opens an underflow window: a DESTROY_REQ that arrives between
   * push and fetch_add could pop our waiter and fetch_sub before our
   * increment, sending pending_count to UINT_MAX.
   *
   * Note: under MPSC there is no per-node "mark" — the consumer simply
   * pops in FIFO order and decrements pending_count once per popped
   * waiter (in drain_pending_rw_after_grant / fail_trigger_pending /
   * destroy fan-out).  The post-push destroy re-check is folded into
   * the consumer path: if destroy_state advances past NONE while we
   * are pushing, fail_trigger_pending will pop us and wake the EDT
   * with NULL ptr; pending_count is decremented there. */
  arts_atomic_add(&cache->pending_count, 1);
  arts_pending_rw_queue_push(&cache->pending_rw, w);

  /* Kick LOCK_REQ if no one else has — GRANT is what eventually
   * triggers our drain in FIFO order. */
  if (arts_atomic_cswap(&cache->lock_req_in_flight, 0, 1) == 0) {
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_coh_send_lock_req(home_rank, cache->db_guid);
  }
  return ARTS_DB_ACQUIRE_PARK;
}
#endif /* !ARTS_MEMORY_MODEL_LC */

/* ===== Case 7: remote-RO path ====================================== */

static arts_db_acquire_result_t acquire_remote_ro(struct arts_db_cache_s *cache,
                                                  arts_guid_t edt_guid,
                                                  unsigned int slot) {
  /* No destroy_state precheck (same rationale as acquire_remote_rw):
   * route_item NULL-store happens before destroy_state CAS, so caller's
   * lookup miss + OoO defer is the destroyed-DB path.  We push the waiter
   * unconditionally; fail_trigger_pending handles any concurrent destroy
   * by marking the waiter and waking the EDT with NULL ptr. */
  struct arts_db_ro_waiter_s *w =
      (struct arts_db_ro_waiter_s *)arts_marked_list_alloc(&cache->pending_ro);
  w->edt_guid = edt_guid;
  w->slot = slot;
  w->target_version = UINT64_MAX;
  arts_atomic_add(&cache->pending_count, 1);
  arts_marked_list_push(&cache->pending_ro, &w->link);

  unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
  arts_coh_send_get_data(home_rank, cache->db_guid, w);
  return ARTS_DB_ACQUIRE_PARK;
}

/* ===== 8-case dispatcher ========================================== */

arts_db_acquire_result_t arts_coh_db_acquire(struct arts_db_cache_s *cache,
                                             arts_guid_t edt_guid,
                                             unsigned int slot,
                                             arts_db_access_mode_t mode,
                                             void **out_data) {
  /* Caller is responsible for the route_table ref on the underlying
   * arts_db_s entry -- this function does not acquire or release it.
   * Caller passes the cache_s extracted from db->coherence_cache.
   *
   * Return contract:
   *   ARTS_DB_ACQUIRE_OK    -- ownership/visibility established; *out_data
   *                            is buf->data if a buffer is installed, or
   *                            NULL when only metadata exists (sentinel
   *                            db_size==0, or version-0 pre-install on
   *                            home for cross-rank create).  In the NULL
   *                            case the caller's body must treat the dep
   *                            as "no payload"; writer_count was bumped
   *                            (RW path) and release_rw will balance.
   *   ARTS_DB_ACQUIRE_PARK  -- waiter pushed to cache.pending_*; the
   *                            protocol's GRANT/DATA_RESPONSE drain will
   *                            wake the EDT.
   *
   * Per the route_item NULL/AVAILABLE invariant (spec 3.1), "DB does not
   * exist" means route_item->data == NULL, in which case the caller
   * (acquire_dbs) defers via the OoO list -- arts_coh_db_acquire is
   * never called with a non-existent DB.  Therefore there is no
   * DESTROYED return: a cache_s being passed in implies the DB exists.
   * cache->buffer == NULL is just "no payload yet", not destruction. */
  if (cache == NULL) {
    /* Defensive: caller misuse.  Park (caller can recover via OoO). */
    return ARTS_DB_ACQUIRE_PARK;
  }

  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  bool is_owner = (cache->writer_count > 0);

#if defined(ARTS_MEMORY_MODEL_LC)
  /* LC: home rank holds the canonical buffer (maintained by sync
   * WRITEBACK from every non-home writer).  RW and RO are unified —
   * non-home acquires go through acquire_remote_ro in both modes so
   * the EDT parks on pending_ro and is woken by DATA_RESPONSE once home
   * delivers its current buffer.  There is no LOCK_REQ / INVALIDATE /
   * GRANT round, no per-cache pending_rw queue.
   *
   * RW acquires bump writer_count BEFORE parking (or before acquire_local
   * on home).  release_rw balances this decrement; without the bump,
   * release_rw's writer_count == 0 guard silently skips the WRITEBACK,
   * breaking cross-rank RW visibility.  RO acquires do not bump because
   * release_ro is a no-op. */
  (void)is_owner;
  if (mode == DB_MODE_RW) {
    arts_atomic_add(&cache->writer_count, 1);
  }
  if (is_home) {
    *out_data = acquire_local(cache);
    return ARTS_DB_ACQUIRE_OK;
  }
  return acquire_remote_ro(cache, edt_guid, slot);
#else
  if (mode == DB_MODE_RO) {
#ifdef ARTS_MEMORY_MODEL_LRC
    /* LRC: the home rank does not hold the canonical data copy; only the
     * current owner (writer_count > 0) has an installed buffer.  A home-
     * but-not-owner rank has cache->buffer == NULL until TRANSFER_OWNERSHIP
     * arrives, so acquire_local would deliver NULL to the EDT.  Go through
     * acquire_remote_ro so that home forwards the request to the owner via
     * REDIRECT_RO and the owner sends the buffer back via DATA_RESPONSE. */
    bool has_local_data = is_owner;
#else
    /* RC: WRITEBACK is synchronous (sender spins on ACK), so home always
     * holds current data before any RO acquire can execute.  is_home is
     * sufficient to guarantee a non-NULL buffer. */
    bool has_local_data = is_home || is_owner;
#endif
    if (has_local_data) {
      /* acquire_local returns NULL when buffer is not installed (sentinel
       * or version-0 pre-install).  That is OK -- caller treats NULL as
       * "no payload".  No DESTROYED claim. */
      *out_data = acquire_local(cache);
      return ARTS_DB_ACQUIRE_OK;
    }
    /* Case 7: remote-RO. */
    return acquire_remote_ro(cache, edt_guid, slot);
  }

  /* mode == DB_MODE_RW (or RW-equivalent) */
  if (is_owner) {
    if (acquire_rw_local_fast(cache) == CASE26_OK) {
      /* writer_count bumped.  acquire_local NULL is fine (sentinel /
       * version-0); release_rw will decrement the matching bump.  No
       * undo, no DESTROYED. */
      *out_data = acquire_local(cache);
      return ARTS_DB_ACQUIRE_OK;
    }
    /* FAIL_FALLBACK: writer_count went to 0 between dispatch and CAS;
     * fall through to remote-RW. */
  }
  return acquire_remote_rw(cache, edt_guid, slot);
#endif /* ARTS_MEMORY_MODEL_LC */
}

/* ===== Drain helpers (called from coherence_handlers.c) ============= */

/* arts_coh_drain_pending_rw_after_grant: RC/LRC only.  LC has no
 * pending_rw queue and no GRANT message — non-home writers acquire via
 * acquire_remote_ro and are woken by DATA_RESPONSE. */
#ifndef ARTS_MEMORY_MODEL_LC

/* Drain callback context for the RW MPSC pop loop. */
struct rw_drain_ctx_s {
  struct arts_db_cache_s *cache;
};

static void rw_drain_cb(arts_guid_t edt_guid, unsigned int slot, void *vctx) {
  struct rw_drain_ctx_s *ctx = (struct rw_drain_ctx_s *)vctx;
  /* Each popped waiter claims exactly one writer_count slot (FIFO),
   * wakes its parked EDT, and decrements pending_count. */
  arts_atomic_add(&ctx->cache->writer_count, 1);
  /* Sentinel DBs (db_size==0) have cache->buffer==NULL by design;
   * mark_edt_ready_by_guid handles that cleanly (depv[slot].ptr=NULL,
   * still decrements depc_needed). */
  mark_edt_ready_by_guid(edt_guid, slot);
  if (arts_atomic_sub(&ctx->cache->pending_count, 1) == 0) {
    arts_coh_try_finalize_destroy(ctx->cache);
  }
}

void arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                           uint64_t version, bool has_next) {
  (void)version;
  struct rw_drain_ctx_s ctx = {.cache = cache};
  arts_pending_rw_queue_drain(&cache->pending_rw, rw_drain_cb, &ctx);

  /* Withdraw the install-time sentinel iff has_next.  If withdraw
   * brings writer_count to 0, no local waiters were drained — emit
   * RELEASE_OWNERSHIP (one-way) instead of WRITEBACK_AND_TRANSFER
   * to avoid blocking the network handler thread on ACK. */
  if (has_next) {
    unsigned int rest = arts_atomic_sub(&cache->writer_count, 1);
    if (rest == 0) {
      bool is_home = ((unsigned int)arts_guid_get_rank(cache->db_guid) ==
                      arts_global_rank_id);
      if (is_home) {
        /* local_transfer lives in B5; for now, delegate via stub. */
        extern void arts_coh_local_transfer_now(struct arts_db_cache_s * cache);
        arts_coh_local_transfer_now(cache);
      } else {
        unsigned int home_rank =
            (unsigned int)arts_guid_get_rank(cache->db_guid);
        arts_coh_send_release_ownership(home_rank, cache->db_guid);
      }
    }
  }
}

#endif /* !ARTS_MEMORY_MODEL_LC */

/* Visit context for the RO drain. */
struct ro_drain_ctx_s {
  struct arts_db_cache_s *cache;
  uint64_t buffer_version;
};

static void ro_drain_visit(arts_marked_list_node_t *node, void *vctx) {
  struct ro_drain_ctx_s *ctx = (struct ro_drain_ctx_s *)vctx;
  struct arts_db_ro_waiter_s *w = (struct arts_db_ro_waiter_s *)node;
  uint64_t t = w->target_version;
  if (t > ctx->buffer_version) {
    return; /* not yet satisfied. */
  }
  arts_guid_t edt_local = w->edt_guid;
  unsigned int slot_local = w->slot;
  if (arts_marked_list_mark(node)) {
    /* Always wake the parked EDT — see rw_drain_visit comment for the
     * sentinel-DB rationale. */
    mark_edt_ready_by_guid(edt_local, slot_local);
    if (arts_atomic_sub(&ctx->cache->pending_count, 1) == 0) {
      arts_coh_try_finalize_destroy(ctx->cache);
    }
  }
}

void arts_coh_drain_pending_ro(struct arts_db_cache_s *cache,
                               uint64_t version) {
  /* Use the just-installed version as the gate.  Any RO whose
   * target_version <= version is satisfied. */
  struct ro_drain_ctx_s ctx = {.cache = cache, .buffer_version = version};
  arts_marked_list_traverse(&cache->pending_ro, ro_drain_visit, &ctx);
}

void arts_coh_trigger_ro_waiter(struct arts_db_cache_s *cache,
                                struct arts_db_ro_waiter_s *w,
                                uint64_t version) {
  /* DATA_RESPONSE handler delivered (version, waiter_addr) — set the
   * waiter's target_version and try to fire it.  If cache.buffer is
   * already at >= version we trigger now; otherwise the next install
   * will pick it up via drain_pending_ro. */
  w->target_version = version;
  struct arts_db_buffer_s *buf = arts_coherence_acquire_buf(cache);
  uint64_t buf_v = buf ? buf->version : 0;
  /* Sentinel DBs (db_size==0) carry no buffer; treat the version gate as
   * satisfied so the waiter still fires (mark_edt_ready_by_guid handles
   * the NULL buf cleanly). */
  bool ready = (buf == NULL) || (buf_v >= version);
  if (ready) {
    arts_guid_t edt_local = w->edt_guid;
    unsigned int slot_local = w->slot;
    if (arts_marked_list_mark(&w->link)) {
      mark_edt_ready_by_guid(edt_local, slot_local);
      if (arts_atomic_sub(&cache->pending_count, 1) == 0) {
        arts_coh_try_finalize_destroy(cache);
      }
    }
  }
  if (buf != NULL) {
    arts_coherence_release_buf(cache, buf);
  }
}
