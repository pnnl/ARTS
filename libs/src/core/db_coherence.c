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

#include "arts/db.h"
#include "arts/db_coherence.h"
#include "arts/db_coherence_buffer.h"
#include "arts/db_coherence_handlers.h"
#include "arts/db_coherence_home.h"
#include "arts/db_coherence_model.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

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
  /* Model-specific cache-field init: RC/LRC initialize the Vyukov MPSC
   * pending_rw queue (cannot be zero-initialized — head/tail must point at the
   * embedded stub) before any push could land; LRC additionally arms its
   * owner-side dedup map + transfer sentinel.  LC has no pending_rw queue (all
   * RW acquires route through the RO path) so its hook is a no-op. */
  arts_coh_model_init_cache_s(c);
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
   * (pointer-identity match) — no per-cache seq fields to initialize.  LRC
   * owner-side fields (dedup map + transfer sentinel) are armed by the model
   * init hook above. */
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

void *arts_coh_acquire_local(struct arts_db_cache_s *cache) {
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

/* Case 2/6 (RW local fast path) and Case 4/8 (remote-RW path) live in
 * db_coherence_release.c — they touch the RC/LRC home-directory cache fields
 * (pending_rw, ownership_req_in_flight) that the LC cache layout does not have.
 */

/* ===== Case 7: remote-RO / remote-snapshot path =================== */

arts_db_acquire_result_t
arts_coh_acquire_remote_ro(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
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

  /* The 8-case body is model-specific: RC/LRC run the single-owner LOCK_REQ /
   * GRANT path (db_coherence_release.c), LC runs the unified home-canonical
   * path (db_coherence_lc.c). */
  return arts_coh_model_acquire_dispatch(cache, dep, edt_guid, slot, mode,
                                         is_home, is_owner);
}

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

/* ===== writeback ACK wait (shared coherence service) =================
 *
 * Synchronous WRITEBACK with a stack-local semaphore matched by pointer
 * identity.  Called by the RC and LC release-tail hooks (the LRC tail uses
 * TRANSFER_OWNERSHIP and never waits on a WRITEBACK_ACK).  Declared in
 * db_coherence_model.h so the model TUs can invoke it. */
void await_writeback_ack(sem_t *cv) {
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

  /* Pre-decrement model hook: a model may drop its buffer ref before
   * writer_count is decremented (LRC closes the teardown window here; RC/LC are
   * no-ops).  May NULL out buf to signal "already released". */
  arts_coh_model_release_rw_pre_decrement(cache, &buf_h, &buf);

  unsigned int rest =
      arts_atomic_sub(&cache->writer_count, 1); /* post-decrement value */

  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);

  /* Post-decrement model hook: owns the transfer/writeback decision and the
   * final buffer-ref release.  Per-model body lives in db_coherence_<model>.c.
   */
  arts_coh_model_release_rw_tail(cache, &buf_h, buf, new_v, rest, is_home);
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

void arts_coh_fail_trigger_pending(struct arts_db_cache_s *cache) {
  /* Destroy fan-out: wake every parked waiter with NULL ptr so the EDT
   * observes the destroyed DB (mark_edt_ready_by_guid delivers NULL when the
   * cache buffer is gone).  RW uses the Vyukov MPSC FIFO drain; the snapshot
   * reorder buffer drains via the same atomic_exchange (waking any case-3 nodes
   * that would otherwise never be satisfied).  LC has no pending_rw queue (all
   * modes park on pending_snapshot) so its hook is a no-op. */
  arts_coh_model_fail_trigger_pending_rw(cache);
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
  /* 3. Free queued waiters.  RW uses the Vyukov MPSC (torn down by the model
   *    hook — RC/LRC destroy pending_rw, LC is a no-op since it has none); the
   *    snapshot reorder buffer is a Treiber stack drained and freed below. */
  arts_coh_model_cache_destructor(cache);
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
