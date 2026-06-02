/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence cache lifecycle + acquire + release + destroy.
 *
 * This translation unit consolidates four phases of the per-rank
 * coherence cache_s:
 *
 *   1. Cache construction / destruction
 *      - arts_coh_init_cache_s: in-place cache_s initializer (creator-home,
 *        creator-remote, home-recv, lazy).
 *      - arts_coh_cache_destructor: chained from arts_db_free.
 *
 *   2. Acquire path (8-case dispatcher + supporting routines).  See the
 *      design plan for the full algorithm; inline comments highlight the
 *      trickier race resolutions.  Wake mechanism: parked EDTs are tracked
 *      via arts_edt_s.depc_needed; a triggered waiter looks up the EDT,
 *      writes the buffer data pointer into depv[slot].ptr, and
 *      atomic_sub(depc_needed); on reaching 0 the EDT is handed to the
 *      scheduler.
 *
 *   3. Release path (the four release cases R1-R4 and the WRITEBACK_ACK
 *      rendezvous).  Wait/wake mechanism: a stack-local binary semaphore;
 *      release_rw sem_init's a sem_t on its stack, embeds its address in the
 *      WRITEBACK packet, and sem_waits on it.  The home echoes that address
 *      verbatim in WRITEBACK_ACK; arts_handler_db_writeback_ack sem_posts it.
 *      Matching is by pointer identity (the address is valid only on the
 *      releaser rank, where the post runs) — no per-cache seq state, no
 *      busy-wait.
 *
 *   4. Destroy lifecycle (arts_coh_db_destroy public entry +
 *      fail_trigger_pending).  Final teardown is driven by the cb
 *      (shared-ptr) deferred-free model: destroy fans out, then
 *      arts_route_table_mark_delete frees the cache_s via
 *      arts_coh_cache_destructor once all refs drain.
 */

#include <errno.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arts/compute/edt.h"
#include "arts/gas/route_table.h"
#include "arts/memory/coherence.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/coherence_home.h"
#include "arts/memory/db.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

/* ===== same-TU forward declarations ================================
 *
 * local_transfer_now / invalidate_transfer are defined in the release
 * section below but called from the acquire section above it (same TU). */
#ifndef ARTS_MEMORY_MODEL_LC
void arts_coh_local_transfer_now(struct arts_db_cache_s *cache);
void arts_coh_invalidate_transfer(struct arts_db_cache_s *cache);
#endif

/* ================================================================== */
/* ===== Cache lifecycle ============================================ */
/* ================================================================== */

void arts_coh_init_cache_s(struct arts_db_cache_s *c, arts_guid_t db_guid,
                           uint64_t db_size, arts_coh_init_kind_t kind,
                           unsigned int creator_rank) {
  /* Caller provides a zeroed cache (embedded in a zeroed/calloc'd db_s, or
   * memset by the stub path).  We do not zero it here — the embedding db_s
   * owns the storage. */
  c->db_guid = db_guid;
  c->db_size = db_size;
  /* Vyukov MPSC queue cannot be zero-initialized: head and tail must
   * point at the embedded stub.  Initialize before any push could
   * land.  LC routes all RW acquires through the RO path and never
   * pushes to pending_rw, so skip in LC builds. */
#ifndef ARTS_MEMORY_MODEL_LC
  arts_pending_rw_queue_init(&c->pending_rw);
#endif
  /* Snapshot reorder-buffer: a Treiber stack (zero-initializable, but init
   * explicitly for clarity).  Nodes are heap-allocated on the case-3 push path
   * and freed when drained by the next install. */
  arts_lf_stack_init(&c->pending_snapshot);
  /* Initialize the inlined home-directory fields only on the rank that owns
   * this DB's GUID home; non-home ranks leave db_self->home_initialized false
   * (and allocate only the cache-only stub, so the home fields don't exist).
   * init_kind selects the initial rw_holder. */
  unsigned int self = arts_global_rank_id;
  unsigned int n = arts_global_rank_count;
  if (n == 0) {
    n = 1;
  }
  struct arts_db_s *db_self = arts_db_of_cache(c);
  if (kind == ARTS_COH_INIT_HOME_RECV) {
    arts_db_home_init(db_self, creator_rank, n);
    db_self->home_initialized = true;
  } else if (kind == ARTS_COH_INIT_CREATOR_HOME) {
    arts_db_home_init(db_self, self, n);
    db_self->home_initialized = true;
    c->writer_count = 2; /* sentinel + creator EDT */
  } else if (kind == ARTS_COH_INIT_CREATOR_REMOTE) {
    c->writer_count = 2;
  }
  /* RC/LC WRITEBACK ACK rendezvous is now a stack-local sem_t per release_rw
   * (pointer-identity match) — no per-cache seq fields to initialize. */
#if defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC owner-side fields: dedup map allocated lazily on first ownership
   * grant.  incoming_new_owner starts at the sentinel (no transfer pending). */
  c->last_sent_version = NULL;
  c->incoming_new_owner = ARTS_LRC_NO_PENDING_OWNER;
#endif
}

/* ================================================================== */
/* ===== Acquire path =============================================== */
/* ================================================================== */

/* Adapter: route_table stores arts_db_s*; the coherence cache_s is embedded
 * by value as the first member of db_s.  All coherence paths look up cache
 * via this helper.  Returns NULL if either the route_table entry doesn't
 * exist or the entry has no DB-level coherence (e.g. PIN/CXL DBs, which keep
 * the embedded cache zeroed).
 *
 * This is the last surviving raw arts_route_table_lookup_data caller in
 * libs/.  Migrating it to a typed handle lookup (arts_route_table_lookup_db +
 * arts_shared_release) would require rewriting all ~15 callers across
 * coherence.c / coherence_handlers.c / db.c to balance the ref — out of scope
 * here.  Documented as a known exception. */
struct arts_db_cache_s *arts_coh_route_table_lookup_cache(arts_guid_t db_guid) {
  void *data = arts_route_table_lookup_data(db_guid);
  if (data == NULL) {
    return NULL;
  }
  struct arts_db_s *db = (struct arts_db_s *)data;
  if (db->db_type != ARTS_DB) {
    return NULL;
  }
  return &db->cache;
}
#define coh_lookup_cache arts_coh_route_table_lookup_cache

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
 * Cross-TU: the snapshot-response handler (coherence_handlers.c) resumes a
 * parked EDT by (edt_guid, slot) directly; the destroy path
 * (fail_trigger_pending) wakes parked EDTs on destroy-fail (NULL ptr semantics
 * — EDT observes destroyed DB). */
void mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot) {
  if (edt_guid == NULL_GUID) {
    return;
  }
  /* lookup_edt handle pairs with release at function exit. */
  arts_shared_ptr_t edt_h = arts_route_table_lookup_edt(edt_guid);
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_shared_get(edt_h);
  if (edt == NULL) {
    ARTS_INFO("coherence: edt_guid %lu not found at trigger time", edt_guid);
    arts_shared_release(&edt_h);
    return;
  }
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  arts_guid_t db_guid = depv[slot].guid;
  if (db_guid != NULL_GUID) {
    struct arts_db_cache_s *cache = coh_lookup_cache(db_guid);
    if (cache != NULL) {
      /* Acquire the EDT's strong ref on the buffer; release_one_dep drops it
       * (via buf_from_data(ptr)->cb) when the EDT finishes.  The handle is not
       * released here — the ref is the EDT's hold.  depv[slot].ptr aliases
       * buf->data, the canonical user-visible payload. */
      arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
      struct arts_db_buffer_s *buf =
          (struct arts_db_buffer_s *)arts_shared_get(buf_h);
      depv[slot].ptr = buf ? buf->data : NULL;
    }
  }
  /* Frontier dep satisfied — advance past it and resume the strict sequential
   * acquire walk.  arts_db_acquire_all schedules the EDT (via
   * arts_schedule_ready_edt) once all deps are held.  The edt_h ref held across
   * this call keeps the EDT alive even if arts_db_acquire_all schedules it and
   * another worker runs it. */
  edt->resume_k++;
  arts_db_acquire_all(edt);
  arts_shared_release(&edt_h);
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
 * lookup) so subsequent arts_db_acquire_all flow can balance with the
 * usual return_db at release_one_dep time. */
struct arts_db_cache_s *arts_coh_lazy_install_cache_s(arts_guid_t db_guid,
                                                      uint64_t db_size) {
  /* First check if it already exists (someone else lazy-installed or
   * a wire-receive fired). */
  struct arts_db_cache_s *cache = coh_lookup_cache(db_guid);
  if (cache != NULL) {
    return cache;
  }

  /* Lazy install (non-home consumer first acquire): cache-only stub — no home
   * directory (this rank is not the GUID home).  arts_db_cache_stub_size()
   * spans cache + db_type, stopping before the home fields. */
  uint64_t stub_sz = arts_db_cache_stub_size();
  struct arts_db_s *stub = (struct arts_db_s *)arts_malloc_align(stub_sz, 16);
  memset(stub, 0, stub_sz);
  stub->db_type = ARTS_DB;

  /* db_size==0 ⇒ lazy install: buffer alloc deferred until first wire
   * arrival (install_buffer with the actual db_size).  No home struct
   * yet — even for is_home, the home struct is created when DB_CREATE
   * arrives (with the proper rw_holder = creator_rank). */
  arts_coh_init_cache_s(&stub->cache, db_guid, /*db_size=*/db_size,
                        ARTS_COH_INIT_LAZY, /*creator_rank=*/0);

  if (arts_route_table_install_if_absent(stub, db_guid, arts_global_rank_id,
                                         /*used=*/true)) {
    arts_ooo_drain_guid(db_guid);
    return &stub->cache;
  }

  /* Lost the race — another thread already installed.  Tear down our
   * stub and return the established cache. */
  arts_db_free(stub);
  return coh_lookup_cache(db_guid);
}

/* ===== Case 1/3/5: local-buffer acquire ============================= */

static void *acquire_local(struct arts_db_cache_s *cache) {
  /* Take the EDT's strong ref on the buffer and return buf->data.  The handle
   * is intentionally NOT released here — the ref is the EDT's hold for its
   * whole lifetime; release_one_dep drops it via buf_from_data(ptr)->cb.  The
   * ref keeps the buffer alive against a concurrent destroy. */
  arts_shared_ptr_t h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf = (struct arts_db_buffer_s *)arts_shared_get(h);
  if (buf == NULL) {
    return NULL; /* h is NULL — nothing installed, nothing held */
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
  /* Note: under MPSC there is no per-node "mark" — the consumer simply
   * pops in FIFO order and wakes each popped waiter (in
   * drain_pending_rw_after_grant / fail_trigger_pending / destroy
   * fan-out).  The post-push destroy re-check is folded into the
   * consumer path: if destroy_state advances past NONE while we are
   * pushing, fail_trigger_pending will pop us and wake the EDT with
   * NULL ptr. */
  arts_pending_rw_queue_push(&cache->pending_rw, w);

  /* Kick LOCK_REQ if no one else has — GRANT is what eventually
   * triggers our drain in FIFO order. */
  if (arts_atomic_cswap(&cache->ownership_req_in_flight, 0, 1) == 0) {
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_ownership_request(home_rank, cache->db_guid);
  }
  return ARTS_DB_ACQUIRE_PARK;
}
#endif /* !ARTS_MEMORY_MODEL_LC */

/* ===== Case 7: remote-RO / remote-snapshot path =================== */

static arts_db_acquire_result_t acquire_remote_ro(struct arts_db_cache_s *cache,
                                                  arts_guid_t edt_guid,
                                                  unsigned int slot) {
  /* No list registration (plan: "acquire 시 list 등록 안 함").  Fire
   * SNAPSHOT_REQUEST carrying edt_guid + slot and PARK; the matching
   * SNAPSHOT_RESPONSE at this rank resumes the EDT directly (case 1/2),
   * or — only under transport reorder — case 3 pushes a reorder-buffer
   * node onto pending_snapshot.  A concurrent destroy is handled by the
   * caller's lookup-miss + OoO defer (route_item NULL-store precedes the
   * destroy_state CAS); no per-waiter destroy precheck is needed here. */
  unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
  arts_send_db_snapshot_request(home_rank, cache->db_guid, edt_guid, slot);
  return ARTS_DB_ACQUIRE_PARK;
}

/* ===== 8-case dispatcher ========================================== */

arts_db_acquire_result_t arts_handler_db_acquire(struct arts_db_cache_s *cache,
                                                 arts_edt_dep_t *dep,
                                                 arts_guid_t edt_guid,
                                                 unsigned int slot) {
  /* Caller is responsible for the route_table ref on the underlying
   * arts_db_s entry -- this function does not acquire or release it.
   * Caller passes the cache_s embedded in the db_s (db->cache) and the dep
   * slot to fill; the handler owns the dep->ptr write on the OK path.
   *
   * Return contract:
   *   ARTS_DB_ACQUIRE_OK    -- ownership/visibility established; dep->ptr
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
   * (arts_db_acquire_all) defers via the OoO list -- arts_handler_db_acquire
   * is never called with a non-existent DB.  Therefore there is no
   * DESTROYED return: a cache_s being passed in implies the DB exists.
   * cache->buffer == NULL is just "no payload yet", not destruction. */
  if (cache == NULL) {
    /* Defensive: caller misuse.  Park (caller can recover via OoO). */
    return ARTS_DB_ACQUIRE_PARK;
  }

  arts_db_access_mode_t mode = dep->mode;
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  bool is_owner = (cache->writer_count > 0);

#if defined(ARTS_MEMORY_MODEL_LC)
  /* LC: home rank holds the canonical buffer (maintained by sync
   * WRITEBACK from every non-home writer).  RW and RO are unified —
   * non-home acquires go through acquire_remote_ro in both modes so
   * the EDT parks (no list registration) and is woken by DATA_RESPONSE once
   * home delivers its current buffer.  There is no LOCK_REQ / INVALIDATE /
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
    dep->ptr = acquire_local(cache);
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
      dep->ptr = acquire_local(cache);
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
      dep->ptr = acquire_local(cache);
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
  /* Each popped waiter claims exactly one writer_count slot (FIFO) and
   * wakes its parked EDT. */
  arts_atomic_add(&ctx->cache->writer_count, 1);
  /* Sentinel DBs (db_size==0) have cache->buffer==NULL by design;
   * mark_edt_ready_by_guid handles that cleanly (depv[slot].ptr=NULL,
   * still decrements depc_needed). */
  mark_edt_ready_by_guid(edt_guid, slot);
}

void arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                           uint64_t version, bool has_next) {
  (void)version;
  (void)
      has_next; /* chain continuation is home-driven (advance_chain INVALIDATEs
                 * the new owner when the queue is still non-empty); the owner
                 * no longer self-withdraws its sentinel. */
  struct rw_drain_ctx_s ctx = {.cache = cache};
  arts_pending_rw_queue_drain(&cache->pending_rw, rw_drain_cb, &ctx);
}

#endif /* !ARTS_MEMORY_MODEL_LC */

/* Drain the snapshot reorder buffer in one atomic_exchange.  Monotonic version
 * guarantees every parked node's target_version <= the buffer version that
 * triggers the drain, so a full drain (no partial pop) is always correct
 * (plan: "install 시 전체 drain").  Called from the case-2 install path, the
 * GRANT install, the LRC TRANSFER_OWNERSHIP install, and destroy fan-out. */
void arts_coh_drain_pending_snapshot(struct arts_db_cache_s *cache) {
  arts_lf_link_t *node = arts_lf_stack_drain(&cache->pending_snapshot);
  while (node != NULL) {
    struct arts_db_snapshot_waiter_s *w =
        (struct arts_db_snapshot_waiter_s *)node;
    arts_lf_link_t *next =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    arts_guid_t edt_local = w->edt_guid;
    unsigned int slot_local = w->slot;
    arts_free(w);
    /* Wake after free: mark_edt_ready_by_guid re-derives dep->ptr from the
     * (now-installed) cache buffer and resumes the acquire walk. */
    mark_edt_ready_by_guid(edt_local, slot_local);
    node = next;
  }
}

/* ================================================================== */
/* ===== Release path =============================================== */
/* ================================================================== */

/* ===== local_transfer / invalidate_transfer (RC/LRC only) =========
 *
 * LC has no LOCK_REQ / INVALIDATE / GRANT round; non-home writers push
 * data back to home via synchronous WRITEBACK and then release their
 * writer_count.  Home is always the canonical data holder, so no
 * local_transfer_now or invalidate_transfer is needed in LC builds. */
#ifndef ARTS_MEMORY_MODEL_LC

#if defined(ARTS_MEMORY_MODEL_RC)
/* RC ownership-transfer chain advance.  Caller guarantees the baton
 * (home.invalidate_in_flight) is held (==1): a transfer round is in progress
 * and this call owns it.  Pops the next waiter, publishes it as rw_holder and
 * GRANTs it, then drives the chain home-side: if more waiters remain it
 * INVALIDATEs the just-granted owner (baton stays 1; the chain continues when
 * that owner's writer_count falls back to 0 and it ownership_returns);
 * otherwise the owner retains ownership and the baton is cleared.  Holding the
 * baton across the whole chain serializes transfers so no fresh
 * OWNERSHIP_REQUEST can CAS 0->1 and fire a second INVALIDATE concurrently. The
 * baton is cleared at the two chain-end points — the queue-empty reclaim and
 * the terminal grant — each followed by a race-recheck for a requester that
 * enqueued after the clear. */
void arts_coh_rc_advance_chain(struct arts_db_cache_s *cache) {
  struct arts_db_s *db =
      arts_db_of_cache(cache); /* home fields inlined in db */
  for (;;) {
    unsigned int new_owner;
    if (!arts_home_lockreq_queue_pop(&db->pending_rw, &new_owner)) {
      /* Queue empty — chain-end candidate.  Home reclaims ownership so the
       * next foreign LOCK_REQ has a holder to invalidate.
       *
       * Ordering invariant: the reclaim (writer_count sentinel + rw_holder)
       * MUST be published BEFORE the baton clear.  A concurrent
       * OWNERSHIP_REQUEST producer that wins the baton does so by reading the
       * cleared (0) value of this baton store, establishing a synchronizes-with
       * edge from this store to its acquire-CAS; only what is sequenced-BEFORE
       * the baton clear is thereby published to it.  The producer then reads
       * rw_holder and INVALIDATEs it, so rw_holder=self must precede the clear
       * — otherwise the producer can observe a stale rw_holder naming a rank
       * whose sentinel was already withdrawn (writer_count==0) and underflow
       * it.  (The LRC arms preserve the same clear-baton-last ordering.) */
      cache->writer_count = 1;
      atomic_store_explicit(&db->rw_holder, arts_global_rank_id,
                            memory_order_release);
      atomic_store_explicit(&db->invalidate_in_flight, 0, memory_order_release);
      /* Re-check for a requester that raced the drain after the baton clear. */
      if (arts_home_lockreq_queue_empty(&db->pending_rw)) {
        return;
      }
      /* A requester raced in.  Try to re-assume the chain.  On loss, the
       * producer that won the baton reads rw_holder=self (published above,
       * before the clear it synchronizes-with) and INVALIDATEs home, whose
       * sentinel writer_count==1 drives the pushed requester's round. */
      unsigned int expected = 0u;
      if (!atomic_compare_exchange_strong_explicit(
              &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
              memory_order_acquire)) {
        return;
      }
      /* Re-assumed the baton; loop to pop the raced-in requester (the next pop
       * overwrites the transient rw_holder=self before any GRANT). */
      continue;
    }
    /* Queue non-empty — publish the new owner BEFORE the GRANT (invariant:
     * home's rw_holder is always the rank that has been GRANTed). */
    atomic_store_explicit(&db->rw_holder, new_owner, memory_order_release);
    bool has_next = !arts_home_lockreq_queue_empty(&db->pending_rw);
    arts_shared_ptr_t master_h = arts_coh_acquire_buf(cache);
    struct arts_db_buffer_s *master =
        (struct arts_db_buffer_s *)arts_shared_get(master_h);
    if (master == NULL) {
      /* Sentinel DB (db_size==0, no buffer): no-data GRANT so the new owner's
       * parked waiter still fires. */
      arts_send_db_ownership_response(new_owner, cache->db_guid, /*version=*/0,
                                      has_next, NULL, 0);
    } else {
      /* Monotonic dedup via home->last_sent_version watermark. */
      uint64_t cur = arts_rank_u64_map_get(db->last_sent_version, new_owner);
      if (cur >= master->version) {
        arts_send_db_ownership_response(new_owner, cache->db_guid,
                                        master->version, has_next, NULL, 0);
      } else {
        arts_rank_u64_map_set(db->last_sent_version, new_owner,
                              master->version);
        arts_send_db_ownership_response(new_owner, cache->db_guid,
                                        master->version, has_next, master->data,
                                        cache->db_size);
      }
      arts_coh_release_buf(&master_h);
    }
    /* Home-driven chain (replaces the old has_next self-relay, which raced a
     * late LOCK_REQ): the new owner does NOT self-withdraw its sentinel.
     * Re-read the queue AFTER the GRANT — if a waiter remains (incl. one that
     * raced in after the pop above), drive the next transfer by INVALIDATEing
     * the owner we just granted.  The commutative signed writer_count makes
     * this GRANT-then-INVALIDATE pair reorder-safe (the INVALIDATE's -1
     * commutes with the GRANT's +1 and the owner's drain/release; whichever
     * decrement drives writer_count from positive to 0 ownership_returns,
     * re-entering this chain). The baton stays 1 across the chain. */
    if (!arts_home_lockreq_queue_empty(&db->pending_rw)) {
      arts_send_db_ownership_invalidate(new_owner, cache->db_guid,
                                        /*new_owner_rank=*/0u);
      return;
    }
    /* Terminal grant: no waiter — the new owner retains ownership (sentinel
     * kept).  Clear the baton so a later LOCK_REQ can start a fresh round, then
     * re-check for one that raced the clear (same recovery as the queue-empty
     * reclaim path above, but ownership is held by new_owner, so drive the
     * transfer by INVALIDATEing it rather than re-popping). */
    atomic_store_explicit(&db->invalidate_in_flight, 0, memory_order_release);
    if (arts_home_lockreq_queue_empty(&db->pending_rw)) {
      return;
    }
    {
      unsigned int expected = 0u;
      if (!atomic_compare_exchange_strong_explicit(
              &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
              memory_order_acquire)) {
        return; /* a LOCK_REQ producer re-took the baton; it INVALIDATEs
                   rw_holder */
      }
      arts_send_db_ownership_invalidate(new_owner, cache->db_guid,
                                        /*new_owner_rank=*/0u);
    }
    return;
  }
}
#endif /* ARTS_MEMORY_MODEL_RC */

void arts_coh_local_transfer_now(struct arts_db_cache_s *cache) {
#if defined(ARTS_MEMORY_MODEL_RC)
  /* RC: the shared chain-advance holds the baton across the has_next chain and
   * clears it at the single queue-empty race-recovery point. */
  arts_coh_rc_advance_chain(cache);
#else
  /* LRC compiles this (it lives in the !LC block) but never calls it: LRC
   * transfers ownership owner→owner via arts_coh_lrc_send_ownership_response,
   * not a home-side GRANT.  No-op stub. */
  (void)cache;
#endif
}

void arts_coh_invalidate_transfer(struct arts_db_cache_s *cache) {
  bool is_home =
      ((unsigned int)arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  if (is_home) {
    arts_coh_local_transfer_now(cache);
    return;
  }
  unsigned int home_rank = (unsigned int)arts_guid_get_rank(cache->db_guid);
  /* The transfer trigger MUST carry the owner's data.  A data-less
   * ownership_return could overtake the releasing worker's in-flight WRITEBACK
   * (release_rw decrements writer_count BEFORE it sends its writeback, so this
   * INVALIDATE-driven decrement can bring the count to 0 while that data is
   * still in flight) and make home GRANT the next owner a stale version.  Ship
   * the current buffer as a one-way WB_AND_TRANSFER instead: home installs it
   * before advancing the chain, so the next owner always sees this owner's
   * write regardless of arrival order vs the worker's own WRITEBACK (identical
   * version => idempotent install).  cv==0 marks it fire-and-forget: home skips
   * the ACK, so the network receiver thread running this handler does not block
   * on an ACK it would itself have to dispatch (a self-deadlock under a single
   * receiver thread). */
  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf != NULL) {
    arts_send_db_writeback(home_rank, cache->db_guid, buf->version, /*cv=*/0,
                           ARTS_WB_AND_TRANSFER, buf->data, cache->db_size);
    arts_coh_release_buf(&buf_h);
  } else {
    /* Zero-size sentinel DB (no buffer): no data can be stale, so the data-less
     * ownership_return is correct. */
    arts_send_db_ownership_return(home_rank, cache->db_guid);
  }
}

#endif /* !ARTS_MEMORY_MODEL_LC — end of local_transfer_now +                \
          invalidate_transfer */

/* ===== writeback ACK wait (RC and LC) ================================
 *
 * RC and LC use synchronous WRITEBACK with a stack-local semaphore matched
 * by pointer identity.  LRC uses TRANSFER_OWNERSHIP instead and sends no
 * WRITEBACK_ACK, so this helper is excluded from LRC builds. */
#ifndef ARTS_MEMORY_MODEL_LRC

static void await_writeback_ack(sem_t *cv) {
  /* Block on the stack-local semaphore until arts_handler_db_writeback_ack
   * posts it.  No busy-wait: sem_timedwait sleeps the worker.  We re-arm on a
   * coarse cadence only to re-check the shutdown flag — once teardown starts
   * the network receiver stops draining and the ACK never arrives, so the EDT
   * epilogue must not block forever (returning lets the worker exit). */
  for (;;) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 1; /* shutdown re-check cadence, not a timeout on the ACK */
    if (sem_timedwait(cv, &ts) == 0) {
      return; /* ACK arrived (pointer-identity post) */
    }
    if (errno == ETIMEDOUT &&
        arts_atomic_read(&arts_node_info.shutdown_state) != 0) {
      return; /* teardown: ACK will never come */
    }
    /* ETIMEDOUT (not shutting down) or EINTR: re-arm the blocking wait. */
  }
}
#endif /* !ARTS_MEMORY_MODEL_LRC */

/* ===== release_rw =================================================== */

void arts_coh_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped
   * ownership (e.g. an RC-style call against a cache that's already
   * been torn down by a destroy fan-out).  Decrementing would
   * underflow; bail.  Atomic acquire-load avoids a TSan race against
   * concurrent writer_count writes. */
  if (arts_atomic_read(&cache->writer_count) == 0) {
    return;
  }

  /* Acquire current buffer for version bump + WRITEBACK send.  This is
   * a local ref scoped to release_rw — the EDT's own ref (from acquire)
   * is dropped separately by release_one_dep. */
  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t new_v = 0;
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
    new_v = arts_atomic_read_u64(&buf->version);
  }

#ifdef ARTS_MEMORY_MODEL_LRC
  /* LRC: drop the buffer ref BEFORE decrementing writer_count, so the slot's
   * cache-hold is the only ref that can keep the buffer alive past
   * writer_count==0 (a concurrent teardown then frees it via the cb deleter
   * with no dangling local ref). */
  if (buf != NULL) {
    arts_coh_release_buf(&buf_h);
    buf = NULL;
  }
#endif

  unsigned int rest =
      arts_atomic_sub(&cache->writer_count, 1); /* post-decrement value */

  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);

#if defined(ARTS_MEMORY_MODEL_LC)
  /* LC release: every non-home write must be pushed back to home
   * synchronously so home remains canonical before any subsequent
   * acquire can see fresh data.  Home itself needs no WRITEBACK.
   *
   * R3: intermediate release (rest > 0, non-home) — same sync writeback.
   * R4: last release (rest == 0, non-home) — sync writeback to home. */
  if (!is_home && buf != NULL) {
    sem_t cv;
    sem_init(&cv, 0, 0);
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_writeback(home_rank, cache->db_guid, new_v,
                           (uint64_t)(uintptr_t)&cv, ARTS_WB_NORMAL, buf->data,
                           cache->db_size);
    await_writeback_ack(&cv);
    sem_destroy(&cv);
  }
  /* Release the buffer ref held for the version-bump and WRITEBACK read. */
  if (buf != NULL) {
    arts_coh_release_buf(&buf_h);
  }
#elif defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC: drop our buffer ref BEFORE decrementing writer_count.
   *
   * Invariant: when writer_count reaches 0, no thread may hold an
   * outstanding buffer ref acquired in this call, because a concurrent
   * deferred-free teardown (triggered once refs drain) will free
   * cache->buffer_pool.  Any subsequent release_buf write to that pool
   * would corrupt freed memory.
   *
   * In RC this window does not exist because local_transfer_now restores
   * the sentinel (writer_count = 1) when no pending waiter is queued,
   * keeping writer_count above 0 until the next proper acquire.  LRC has
   * no such sentinel restoration, so we must close the window here by
   * releasing the ref before exposing writer_count == 0.
   * (buf was already released before the writer_count decrement.) */
  /* buf was dropped before decrement; skip further release below. */
  if (rest == 0) {
    /* If an INVALIDATE_NOTICE already published a transfer target while writers
     * were live, this (last) releaser is the unique actor that ships
     * TRANSFER_OWNERSHIP — sentinel invariant, no flag (spec :3173).  Identical
     * for home and non-home owners.  Otherwise no transfer is pending: home
     * retains ownership until a future LOCK_REQ; a non-home owner quiesces. */
    if (cache->incoming_new_owner != ARTS_LRC_NO_PENDING_OWNER) {
      arts_coh_lrc_send_ownership_response(cache);
    }
  }
  /* buf was dropped before decrement; home vs non-home no longer branch here.
   */
  (void)is_home;
  (void)new_v;
#else
  /* RC */
  if (rest == 0) {
    if (is_home) {
      arts_coh_local_transfer_now(cache);
    } else {
      if (buf != NULL) {
        /* RC R4: WRITEBACK_AND_TRANSFER + await ACK (stack-local sem). */
        sem_t cv;
        sem_init(&cv, 0, 0);
        unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
        arts_send_db_writeback(home_rank, cache->db_guid, new_v,
                               (uint64_t)(uintptr_t)&cv, ARTS_WB_AND_TRANSFER,
                               buf->data, cache->db_size);
        await_writeback_ack(&cv);
        sem_destroy(&cv);
      }
    }
  } else if (!is_home && buf != NULL) {
    /* RC R3: intermediate writeback so remote ROs see fresh data + await ACK.
     */
    sem_t cv;
    sem_init(&cv, 0, 0);
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_writeback(home_rank, cache->db_guid, new_v,
                           (uint64_t)(uintptr_t)&cv, ARTS_WB_NORMAL, buf->data,
                           cache->db_size);
    await_writeback_ack(&cv);
    sem_destroy(&cv);
  }
  /* RC: release buffer ref after the writeback (which reads buf->data). */
  if (buf != NULL) {
    arts_coh_release_buf(&buf_h);
  }
#endif /* model dispatch */
}

void arts_coh_release_ro(struct arts_db_cache_s *cache) {
  /* RO release is also no-op here — the EDT's buf ref is dropped by
   * release_one_dep's DIST branch via release_buf (matching the
   * acquire_buf in mark_edt_ready_by_guid / acquire_local). */
  (void)cache;
}

/* ================================================================== */
/* ===== Destroy lifecycle ========================================== */
/* ================================================================== */

/* ===== fail_trigger_pending ======================================= */

static void fail_trigger_rw_cb(arts_guid_t edt_guid, unsigned int slot,
                               void *vctx) {
  (void)vctx;
  mark_edt_ready_by_guid(edt_guid, slot);
}

void arts_coh_fail_trigger_pending(struct arts_db_cache_s *cache) {
  /* Destroy fan-out: wake every parked waiter with NULL ptr so the EDT
   * observes the destroyed DB (mark_edt_ready_by_guid delivers NULL when the
   * cache buffer is gone).  RW uses the Vyukov MPSC FIFO drain; the snapshot
   * reorder buffer drains via the same atomic_exchange (waking any case-3 nodes
   * that would otherwise never be satisfied).  LC has no pending_rw queue (all
   * modes park on pending_snapshot). */
#ifndef ARTS_MEMORY_MODEL_LC
  arts_pending_rw_queue_drain(&cache->pending_rw, fail_trigger_rw_cb, NULL);
#endif
  arts_coh_drain_pending_snapshot(cache);
}

/* ===== arts_coh_db_destroy public API ============================= */

/* Public destroy: forward DESTROY_REQ to home (uniform path; home ==
 * self gets the message via self-loop).  Caller is responsible for
 * the OCR-spec contract: no concurrent acquires/uses in flight. */
void arts_coh_db_destroy(arts_guid_t db_guid) {
  unsigned int home_rank = (unsigned int)arts_guid_get_rank(db_guid);
  arts_send_db_destroy(home_rank, db_guid);
}

/* ===== cache_s destructor (chained from arts_db_free) ============= */

/* Called from arts_db_free for ARTS_DB descriptors.  Drains the recycle pool
 * and tears down home_s in place; the cache is embedded by value as the first
 * member of db_s, so the caller (arts_db_free) frees the wrapping db_s — the
 * cache is not freed separately.  Order matters because step 1 covers the
 * rare race where a wire handler installed a buffer past
 * try_finalize_destroy's NULL-swap. */
void arts_coh_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  /* 1. Release the cache-hold on the buffer (store NULL into the shared
   *    slot).  If no acquirer holds a ref the cb deleter frees the buffer
   *    now; otherwise the buffer survives until the last in-flight acquirer
   *    releases (deferred free via the cb).  The buffer carries no back-ref
   *    to this cache, so it safely outlives us — eliminating the old
   *    destroy-vs-release use-after-free without a recycle pool. */
  arts_atomic_shared_store(&cache->buffer, NULL);
  /* 3. Free queued waiters.  RW uses the Vyukov MPSC; the snapshot reorder
   *    buffer is a Treiber stack drained and freed in one atomic_exchange.
   *    LC has no pending_rw queue (all modes park on pending_snapshot). */
#ifndef ARTS_MEMORY_MODEL_LC
  arts_pending_rw_queue_destroy(&cache->pending_rw);
#endif
  {
    arts_lf_link_t *n = arts_lf_stack_drain(&cache->pending_snapshot);
    while (n != NULL) {
      arts_lf_link_t *next =
          atomic_load_explicit(&n->next, memory_order_relaxed);
      arts_free(n);
      n = next;
    }
  }
  /* 4. Home-directory fields (inlined in arts_db_s; tear down sub-resources).
   */
  {
    struct arts_db_s *db_self = arts_db_of_cache(cache);
    if (db_self->home_initialized) {
      arts_db_home_teardown(db_self);
      db_self->home_initialized = false;
    }
  }
  /* 5. cache_s itself is freed by the route_table after this routine
   *    returns.  Buffers (FAM data lives there) are recycled to the
   *    pool / freed in step 1-3. */
}
