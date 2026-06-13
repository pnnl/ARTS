/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence cache lifecycle + acquire + release + destroy.
 *
 * This translation unit consolidates four phases of the per-rank
 * coherence cache_s:
 *
 *   1. Cache construction / destruction
 *      - arts_db_cache_init: in-place cache_s initializer (creator-home,
 *        creator-remote, home-recv, lazy).
 *      - arts_db_cache_destructor: chained from arts_db_free.
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
 *   4. Destroy lifecycle (arts_db_destroy_remote public entry +
 *      fail_trigger_pending).  Final teardown is driven by the cb
 *      (shared-ptr) deferred-free model: destroy fans out, then
 *      arts_route_table_set_destroyed frees the cache_s via
 *      arts_db_cache_destructor once all refs drain.
 */

#include <errno.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/home.h"
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
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

/* Protocol-agnostic cache_s field init.  The per-protocol arts_db_cache_init
 * wrapper (coherence/<protocol>.c) runs its protocol-specific field-init
 * (eager/lazy pending_rw queue + lazy dedup-map/sentinel; relaxed none) BEFORE
 * calling this, so the Vyukov MPSC stub is wired before any push could land. */
void arts_db_cache_common_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                               uint64_t db_size, arts_db_init_kind_t kind,
                               unsigned int creator_rank) {
  /* Caller provides a zeroed cache (embedded in a zeroed/calloc'd db_s, or
   * memset by the stub path).  We do not zero it here — the embedding db_s
   * owns the storage. */
  c->db_guid = db_guid;
  c->db_size = db_size;
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
  if (kind == ARTS_DB_INIT_HOME_RECV) {
    arts_db_home_init(db_self, creator_rank, n);
    db_self->home_initialized = true;
  } else if (kind == ARTS_DB_INIT_CREATOR_HOME) {
    arts_db_home_init(db_self, self, n);
    db_self->home_initialized = true;
    c->writer_count = 2; /* sentinel + creator EDT */
  } else if (kind == ARTS_DB_INIT_CREATOR_REMOTE) {
    c->writer_count = 2;
  }
  /* Eager/relaxed WRITEBACK ACK rendezvous is a stack-local sem_t per
   * release_rw (pointer-identity match) — no per-cache seq fields to
   * initialize.  Lazy owner-side fields (dedup map + transfer sentinel) are
   * armed by the protocol init hook above. */
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
struct arts_db_cache_s *arts_db_cache_lookup(arts_guid_t db_guid) {
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

/* ===== EDT wake helper ============================================== */

/* Trigger the parked EDT identified by (edt_guid, slot) by writing
 * the dep slot's data pointer and decrementing depc_needed.
 *
 * Eager-only: the canonical user data pointer lives in cache->user_data,
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
    struct arts_db_cache_s *cache = arts_db_cache_lookup(db_guid);
    if (cache != NULL) {
      /* Acquire the EDT's strong ref on the buffer; release_one_dep drops it
       * (via buf_from_data(ptr)->cb) when the EDT finishes.  The handle is not
       * released here — the ref is the EDT's hold.  depv[slot].ptr aliases
       * buf->data, the canonical user-visible payload. */
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      struct arts_db_buffer_s *buf =
          (struct arts_db_buffer_s *)arts_shared_get(buf_h);
      depv[slot].ptr = buf ? buf->data : NULL;
    }
  }
  /* Data resolved for this dep — count it down; the actor that reaches 0
   * schedules. The edt_h ref held across this call keeps the EDT alive even if
   * the schedule lets another worker run (and free) it; do NOT touch edt after
   * arts_db_acquire_account returns. */
  arts_db_acquire_account(edt);
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
struct arts_db_cache_s *arts_db_cache_lazy_install(arts_guid_t db_guid,
                                                   uint64_t db_size) {
  /* First check if it already exists (someone else lazy-installed or
   * a wire-receive fired). */
  struct arts_db_cache_s *cache = arts_db_cache_lookup(db_guid);
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
  arts_db_cache_init(&stub->cache, db_guid, /*db_size=*/db_size,
                     ARTS_DB_INIT_LAZY,
                     /*creator_rank=*/0);

  if (arts_route_table_install_if_absent(stub, db_guid, arts_global_rank_id,
                                         /*used=*/true)) {
    arts_ooo_drain_guid(db_guid);
    return &stub->cache;
  }

  /* Lost the race — another thread already installed.  Tear down our
   * stub and return the established cache. */
  arts_db_free(stub);
  return arts_db_cache_lookup(db_guid);
}

/* ===== Case 1/3/5: local-buffer acquire ============================= */

void *arts_db_acquire_local(struct arts_db_cache_s *cache) {
  /* Take the EDT's strong ref on the buffer and return buf->data.  The handle
   * is intentionally NOT released here — the ref is the EDT's hold for its
   * whole lifetime; release_one_dep drops it via buf_from_data(ptr)->cb.  The
   * ref keeps the buffer alive against a concurrent destroy. */
  arts_shared_ptr_t h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf = (struct arts_db_buffer_s *)arts_shared_get(h);
  if (buf == NULL) {
    return NULL; /* h is NULL — nothing installed, nothing held */
  }
  return buf->data;
}

/* Case 2/6 (RW local fast path) and Case 4/8 (remote-RW path) live in
 * coherence/ownership.c — they touch the OCR-model home-directory cache fields
 * (pending_rw, ownership_req_in_flight) that the RELAXED cache layout does not
 * have. */

/* ===== Case 7: remote-RO / remote-snapshot path =================== */

arts_db_acquire_result_t
arts_db_acquire_remote_ro(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
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

/* The 8-case acquire dispatcher arts_handler_db_acquire is protocol-specific:
 * EAGER and LAZY define it in coherence/ownership.c-backed
 * coherence/{eager,lazy}.c (single-owner OWNERSHIP_REQUEST / GRANT path,
 * differing only on the RO-has-local-data predicate); RELAXED defines its
 * unified home-canonical body in coherence/relaxed.c.  The shared remote-RO
 * path (arts_db_acquire_remote_ro) and the local-buffer fast read
 * (arts_db_acquire_local) above are reused by all three.
 */

/* Drain the snapshot reorder buffer in one atomic_exchange.  Monotonic version
 * guarantees every parked node's target_version <= the buffer version that
 * triggers the drain, so a full drain (no partial pop) is always correct
 * (plan: "install 시 전체 drain").  Called from the case-2 install path, the
 * GRANT install, the lazy TRANSFER_OWNERSHIP install, and destroy fan-out. */
void arts_db_drain_pending_snapshot(struct arts_db_cache_s *cache) {
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
 * identity.  Called by the eager and relaxed release-tail bodies (the lazy
 * tail uses TRANSFER_OWNERSHIP and never waits on a WRITEBACK_ACK).  Declared
 * in coherence/coherence.h so the protocol TUs can invoke it. */
void await_writeback_ack(sem_t *cv) {
  /* Block on the stack-local semaphore until arts_handler_db_writeback_ack
   * posts it.  No busy-wait: sem_timedwait sleeps the worker.  We re-arm on a
   * coarse cadence only to re-check the shutdown flag — once teardown starts
   * the network receiver stops draining and the ACK never arrives, so the EDT
   * epilogue must not block forever (returning lets the worker exit). */
  for (;;) {
    struct timespec ts;
    (void)clock_gettime(CLOCK_REALTIME, &ts);
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

/* arts_db_release_rw is protocol-specific (the version bump is shared, but the
 * pre-decrement buffer-ref drop and the post-decrement transfer/writeback
 * decision differ per protocol), so its whole body lives in
 * coherence/{eager,lazy,relaxed}.c.  Eager and relaxed call await_writeback_ack
 * above for the synchronous-WRITEBACK rendezvous. */

void arts_db_release_ro(struct arts_db_cache_s *cache) {
  /* RO release is also no-op here — the EDT's buf ref is dropped by
   * release_one_dep's DIST branch via release_buf (matching the
   * acquire_buf in mark_edt_ready_by_guid / acquire_local). */
  (void)cache;
}

/* ================================================================== */
/* ===== Destroy lifecycle ========================================== */
/* ================================================================== */

/* arts_db_fail_trigger_pending (destroy/fail wake of parked waiters) is
 * protocol-specific: EAGER/LAZY drain the pending_rw FIFO
 * (coherence/ownership.c), RELAXED has no pending_rw (coherence/relaxed.c).
 * Both arms then drain the snapshot reorder buffer via
 * arts_db_drain_pending_snapshot above. */

/* ===== arts_db_destroy_remote public API ============================= */

/* Public destroy: forward DESTROY_REQ to home (uniform path; home ==
 * self gets the message via self-loop).  Caller is responsible for
 * the OCR-spec contract: no concurrent acquires/uses in flight. */
void arts_db_destroy_remote(arts_guid_t db_guid) {
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  arts_send_db_destroy(home_rank, db_guid);
}

/* ===== cache_s destructor (chained from arts_db_free) =============
 *
 * The full destructor arts_db_cache_destructor is model-specific (it sequences
 * the model field-destroy between these two shared steps) and lives in
 * coherence/{eager,lazy,relaxed}.c.  The agnostic steps are split into pre
 * (the buffer-NULL that must run first) and post (snapshot drain + home
 * teardown);
 * the per-model wrapper runs pre → model-destroy → post.  cache_s itself is
 * freed by the route_table after the wrapper returns; buffers (FAM data) are
 * recycled / freed by the cb deleter chain once outstanding refs drain. */

/* Step 1: release the cache-hold on the buffer (store NULL into the shared
 * slot).  If no acquirer holds a ref the cb deleter frees the buffer now;
 * otherwise it survives until the last in-flight acquirer releases.  Runs
 * FIRST so it covers the rare race where a wire handler installed a buffer past
 * try_finalize_destroy's NULL-swap. */
void arts_db_cache_common_destroy_pre(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_atomic_shared_store(&cache->buffer, NULL);
}

/* Steps 3b+4: drain+free the snapshot reorder buffer (a Treiber stack), then
 * tear down the inlined home-directory sub-resources.  Runs AFTER the protocol
 * field-destroy (pending_rw in eager and lazy builds). */
void arts_db_cache_common_destroy_post(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  {
    arts_lf_link_t *n = arts_lf_stack_drain(&cache->pending_snapshot);
    while (n != NULL) {
      arts_lf_link_t *next =
          atomic_load_explicit(&n->next, memory_order_relaxed);
      arts_free(n);
      n = next;
    }
  }
  {
    struct arts_db_s *db_self = arts_db_of_cache(cache);
    if (db_self->home_initialized) {
      arts_db_home_teardown(db_self);
      db_self->home_initialized = false;
    }
  }
}
