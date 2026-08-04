/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence cache lifecycle + acquire + release + destroy.
 *
 * This translation unit consolidates four phases of the per-rank
 * coherence cache_s:
 *
 *   1. Cache construction / destruction
 *      - arts_db_cache_init: in-place cache_s initializer (creator-home,
 *        creator-remote, home-recv, OWNER).
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
 *   3. Release path (the four release cases R1-R4 and the PUBLISH_ACK
 *      rendezvous).  Wait/wake mechanism: a stack-local binary semaphore;
 *      release_rw sem_init's a sem_t on its stack, embeds its address in the
 *      PUBLISH packet, and sem_waits on it.  The home echoes that address
 *      verbatim in PUBLISH_ACK; arts_handler_db_publish_ack sem_posts it.
 *      Matching is by pointer identity (the address is valid only on the
 *      releaser rank, where the post runs) — no per-cache seq state, no
 *      busy-wait.
 *
 *   4. Destroy lifecycle (arts_db_destroy_remote public entry).  Final teardown
 *      is driven by the cb (shared-ptr) deferred-free model: destroy fans out,
 *      then
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
#include "arts/coherence/directory.h"
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h"
#include "arts/memory/regpool.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/net.h" /* arts_net_put_payload */
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

/* ================================================================== */
/* ===== Cache lifecycle ============================================ */
/* ================================================================== */

/* Protocol-agnostic cache_s field init.  The per-protocol arts_db_cache_init
 * wrapper (coherence/<protocol>.c) runs its protocol-specific field-init
 * (HOME/OWNER pending_rw queue + OWNER dedup-map/sentinel; WRF_RCU none) BEFORE
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
#if defined(ARTS_RO_REQUEST_COMBINING) && !defined(ARTS_PROTOCOL_EXCL) 
  arts_lf_stack_init(&c->ro_combine);
  c->ro_combine_group = NULL;
  c->snapshot_req_in_flight = 0;
#endif
  arts_lf_pool_init(&c->buf_freelist, 0);
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
#if !defined(ARTS_PROTOCOL_EXCL)
    /* RCU/WRF_RCU: writer_count tracks ownership (sentinel + creator). */
    c->writer_count = 2;
#endif
  } else if (kind == ARTS_DB_INIT_CREATOR_REMOTE) {
#if !defined(ARTS_PROTOCOL_EXCL)
    c->writer_count = 2;
#endif
  }
  /* Eager/WRF_RCU PUBLISH ACK rendezvous is a stack-local sem_t per
   * release_rw (pointer-identity match) — no per-cache seq fields to
   * initialize.  Lazy owner-side fields (dedup map + transfer sentinel) are
   * armed by the protocol init hook above. */
}

/* ================================================================== */
/* ===== Acquire path =============================================== */
/* ================================================================== */

/* ===== EDT wake helper ============================================== */

/* Trigger the parked EDT identified by (edt_guid, slot) by writing
 * the dep slot's data pointer and decrementing depc_needed.
 *
 * Eager-only: the canonical user data pointer lives in cache->user_data,
 * regardless of whether cache_s was created in-place with arts_db_s
 * (user_data == (db+1)) or stub-installed standalone (user_data ==
 * malloc'd buffer).  We look up the cache via the GUID of the dep
 * slot's DB and stamp depv[slot].ptr accordingly.
 *
 * Cross-TU: the snapshot-response handler (coherence_handlers.c) resumes a
 * parked EDT by (edt_guid, slot) directly; the refcount-0 cache destructor
 * delivers a NULL ptr to any waiter still parked at destroy. */
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
    /* Pin the db_s for the cache-deref window: the embedded cache is its FIRST
     * member (offset 0), so pinning the db_s keeps the cache alive against a
     * concurrent destroy while we read the buffer slot.  Released after
     * depv[slot].ptr is set. */
    arts_shared_ptr_t db_h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
    if (db != NULL && db->db_type == ARTS_DB) {
      struct arts_db_cache_s *cache = &db->cache;
      /* Acquire the EDT's strong ref on the buffer; release_one_dep drops it
       * (via buf_from_data(ptr)->cb) when the EDT finishes.  depv[slot].ptr
       * aliases buf->data, the canonical user-visible payload. */
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      struct arts_db_buffer_s *buf =
          (struct arts_db_buffer_s *)arts_shared_get(buf_h);
      void *data = buf ? buf->data : NULL;
      /* Idempotent slot claim.  Two delivery paths can wake the SAME
       * (edt, slot) — e.g. a snapshot_response case-2 drain racing a direct
       * response.  The slot resolves exactly once: CAS depv[slot].ptr
       * NULL->data so only the first wake keeps its buffer ref (the EDT's hold)
       * and accounts; a loser drops the extra ref it just took and returns
       * WITHOUT accounting — no double-decrement of acquire_remaining, no
       * buffer-ref leak.  (A NULL data resolution is the destroyed-DB / UB
       * path; it does not claim and falls through to account, matching legacy
       * behavior.) */
      if (data != NULL) {
        void *expected = NULL;
        if (!atomic_compare_exchange_strong((_Atomic(void *) *)&depv[slot].ptr,
                                            &expected, data)) {
          arts_db_buf_release(&buf_h); /* lost: release the extra ref */
          arts_shared_release(&db_h);
          arts_shared_release(&edt_h);
          return;
        }
        /* won: keep buf_h as the EDT's hold (do not release it here).  B1: pin
         * the descriptor for the slot's acquire->release span by MOVING db_h
         * into db_pin (released last in release_one_dep) instead of dropping it
         * below, so a concurrent destroy cannot free the cache (buffer slot +
         * recycle pool) under this outstanding buffer ref. */
        if (__atomic_load_n(&depv[slot].db_pin, __ATOMIC_ACQUIRE) == NULL) {
          /* Publish with release so the run/release thread (reached via the
           * work-stealing deque handoff) observes db_pin like the sibling ptr
           * field's atomic CAS — keeps TSan clean and the ARM ordering explicit
           * rather than relying on the deque's incidental HW fence. */
          __atomic_store_n(&depv[slot].db_pin, (void *)db_h, __ATOMIC_RELEASE);
          db_h = NULL;
        }
      }
    }
    arts_shared_release(&db_h); /* no-op when moved into db_pin above */
  }
  /* Data resolved for this dep — count it down; the actor that reaches 0
   * schedules. The edt_h ref held across this call keeps the EDT alive even if
   * the schedule lets another worker run (and free) it; do NOT touch edt after
   * arts_db_acquire_account returns. */
  arts_db_acquire_account(edt);
  arts_shared_release(&edt_h);
}

/* ===== Stub first-touch =========================================== */

/* Allocate a stub arts_db_s + cache_s for a DIST DB that this rank
 * has not yet touched, register it in the route_table, and return a
 * PINNED handle to the db_s whose cache it installed.  The stub has no
 * user data payload — install_buffer allocates on demand cache->user_data on
 * first wire arrival.
 *
 * Race-safe: route_table_install_if_absent rejects if another thread
 * (e.g. a concurrent wire-receive) raced us; in that case we free our
 * stub and return a pinned handle to the established db_s.
 *
 * Returns a pinned handle; the caller MUST arts_shared_release it once the
 * cache is no longer needed (on every control-flow path).  A NULL handle
 * means the DB was destroyed before the install could be observed (the
 * lost-race lookup found no live entry). */
arts_shared_ptr_t arts_db_cache_stub_install(arts_guid_t db_guid,
                                             uint64_t db_size) {
  /* First check if it already exists (someone else stub-installed or
   * a wire-receive fired). */
  arts_shared_ptr_t existing = arts_route_table_lookup_db(db_guid);
  if (arts_shared_get(existing) != NULL) {
    return existing;
  }
  arts_shared_release(&existing);

  /* Stub install (non-home consumer first acquire): cache-only stub — no home
   * directory (this rank is not the GUID home).  arts_db_cache_stub_size()
   * spans cache + db_type, stopping before the home fields. */
  uint64_t stub_sz = arts_db_cache_stub_size();
  struct arts_db_s *stub =
      (struct arts_db_s *)arts_malloc_aligned(stub_sz, ARTS_CACHE_LINE_SIZE);
  memset(stub, 0, stub_sz);
  stub->db_type = ARTS_DB;

  /* db_size==0 ⇒ stub install: buffer alloc deferred until first wire
   * arrival (install_buffer with the actual db_size).  No home struct
   * yet — even for is_home, the home struct is created when DB_CREATE
   * arrives (with the proper rw_holder = creator_rank). */
  arts_db_cache_init(&stub->cache, db_guid, /*db_size=*/db_size,
                     ARTS_DB_INIT_STUB,
                     /*creator_rank=*/0);

  if (arts_route_table_install_if_absent(stub, db_guid, arts_global_rank_id,
                                         /*used=*/true)) {
    arts_ooo_drain_guid(db_guid);
    /* Pin the just-installed db_s (one cb ref) for the caller. */
    return arts_route_table_lookup_db(db_guid);
  }

  /* Lost the race — another thread already installed.  Tear down our
   * stub and return a pinned handle to the established db_s. */
  arts_db_free(stub);
  return arts_route_table_lookup_db(db_guid);
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
 * coherence/grant.c — they touch the OCR-model home-directory cache fields
 * (pending_rw, grant_req_in_flight) that the WRF_RCU cache layout does not
 * have. */

/* ===== Case 7: remote-RO / remote-snapshot path =================== */

#if !defined(ARTS_PROTOCOL_EXCL)
#ifdef ARTS_RO_REQUEST_COMBINING
/* ===== Remote-read request combining ===============================
 *
 * One snapshot request per cache may be in flight ("the window").  The 0->1
 * CAS winner on snapshot_req_in_flight owns it: it isolates the accumulated
 * waiter stack in one exchange, sends ONE wire request naming an arbitrary
 * member (the leader — the response resumes it through the unchanged 1:1
 * path), and parks the rest as ro_combine_group.  The response terminal
 * resumes the whole group against the buffer the response made current, then
 * re-arms: isolate whatever accumulated meanwhile and send again, or release
 * the window.
 *
 * Correctness boundary: a waiter may only ride a request sent AFTER its
 * park.  Any release the waiter is event-ordered after published before its
 * park, hence before the send, hence is contained in the version the server
 * serves at receive time — so one response satisfies the whole batch.
 * Waiters that arrive while a request is in flight must NOT join it (its
 * response may predate their ordering obligations); they form the next
 * window.  Pull semantics stay intact: currency is established per-request
 * at serve time, the server tracks no readers, writers pay nothing.
 *
 * Single-rank runs bypass combining: the self-send serve chain is fully
 * synchronous, so window chaining would recurse through the inline response
 * handler — and there is no wire cost to amortize.  On multi-rank runs the
 * re-arm can inline-recurse only while a request resolves back onto this
 * rank (owner-is-self serve); that recursion is finite and small, because
 * once ownership is local, new read acquires resolve locally and stop
 * feeding the stack. */

/* Isolate the accumulated stack and launch one request.  Caller must own the
 * window.  Returns false when there was nothing to send (caller releases). */
static bool ro_combine_launch_owned(struct arts_db_cache_s *cache) {
  arts_lf_link_t *batch = arts_lf_stack_drain(&cache->ro_combine);
  if (batch == NULL) {
    return false;
  }
  struct arts_db_snapshot_waiter_s *leader =
      (struct arts_db_snapshot_waiter_s *)batch;
  cache->ro_combine_group =
      atomic_load_explicit(&batch->next, memory_order_relaxed);
  arts_guid_t leader_guid = leader->edt_guid;
  unsigned int leader_slot = leader->slot;
  arts_free(leader);
#ifdef ARTS_PROTOCOL_INV
  arts_send_db_inv_request(cache, DB_MODE_RO);
#else
  arts_send_db_snapshot_request(cache, leader_guid, leader_slot);
#endif
  return true;
}

/* Claim the window if free and launch.  Push-then-claim on the acquire side
 * plus release-then-recheck here close the missed-wakeup race: a pusher that
 * loses the claim is guaranteed its node is seen either by the owner's next
 * isolation or by this loop after the owner releases. */
static void ro_combine_pump(struct arts_db_cache_s *cache) {
  while (!arts_lf_stack_empty(&cache->ro_combine)) {
    if (arts_atomic_cswap(&cache->snapshot_req_in_flight, 0, 1) != 0) {
      return; /* someone owns the window; their terminal rechecks */
    }
    if (ro_combine_launch_owned(cache)) {
      return; /* window stays owned until the response terminal */
    }
    (void)arts_atomic_swap(&cache->snapshot_req_in_flight, 0);
  }
}

void arts_db_ro_combine_on_terminal(struct arts_db_cache_s *cache,
                                    bool buffer_live, uint64_t version) {
  arts_lf_link_t *node = cache->ro_combine_group;
  cache->ro_combine_group = NULL;
  while (node != NULL) {
    struct arts_db_snapshot_waiter_s *w =
        (struct arts_db_snapshot_waiter_s *)node;
    arts_lf_link_t *next =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    if (buffer_live) {
      /* The buffer the response made current is at least as new as anything
       * a batch member is ordered after — resume against it (the resume
       * re-derives dep->ptr from the installed buffer). */
      arts_guid_t edt_local = w->edt_guid;
      unsigned int slot_local = w->slot;
      arts_free(w);
      mark_edt_ready_by_guid(edt_local, slot_local);
    } else {
      /* Reorder case: the publish satisfying this response is still in
       * flight.  Park the member on the reorder buffer (its need is <=
       * `version`); the coming install drains it. */
      w->target_version = version;
      w->serve = NULL;
      arts_lf_stack_push(&cache->pending_snapshot, &w->link);
    }
    node = next;
  }
  /* Re-arm: whatever accumulated during this window rides the next request;
   * otherwise release the window and recheck (a push may race the release). */
  if (ro_combine_launch_owned(cache)) {
    return;
  }
  (void)arts_atomic_swap(&cache->snapshot_req_in_flight, 0);
  ro_combine_pump(cache);
}
void arts_db_ro_combine_grant_drain(struct arts_db_cache_s *cache) {
  /* Ownership-arrival drain: called from the transfer-commit body while its
   * sentinel/guard holds writer_count >= 1, so ownership cannot ship out from
   * under the walk.  Every waiter here parked before this drain, and the
   * transferred buffer contains every release completed before the transfer
   * (the ownership chain linearizes all writers), so resuming against it is
   * correct for any park time — unlike a snapshot install, which is only a
   * specific version.  The in-flight window group (ro_combine_group) is NOT
   * touched: its response terminal owns it.  Stragglers that push after this
   * exchange are picked up by their own pump (a fresh request round trip,
   * correct via the response path). */
  arts_lf_link_t *node = arts_lf_stack_drain(&cache->ro_combine);
  while (node != NULL) {
    struct arts_db_snapshot_waiter_s *w =
        (struct arts_db_snapshot_waiter_s *)node;
    arts_lf_link_t *next =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    arts_guid_t edt_local = w->edt_guid;
    unsigned int slot_local = w->slot;
    arts_free(w);
    mark_edt_ready_by_guid(edt_local, slot_local);
    node = next;
  }
}
#endif /* ARTS_RO_REQUEST_COMBINING */

arts_db_acquire_result_t
arts_db_acquire_remote_ro(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
                          unsigned int slot) {
  /* No list registration.  Fire SNAPSHOT_REQUEST carrying edt_guid + slot and
   * PARK; the matching SNAPSHOT_RESPONSE at this rank resumes the EDT directly
   * (case 1/2), or — only under transport reorder — case 3 pushes a
   * reorder-buffer node onto pending_snapshot.  A concurrent destroy is handled
   * by the caller's lookup-miss + OoO defer. */
#ifdef ARTS_RO_REQUEST_COMBINING
  if (arts_global_rank_count > 1) {
    struct arts_db_snapshot_waiter_s *w =
        (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
    w->edt_guid = edt_guid;
    w->slot = slot;
    w->target_version = 0;
    w->serve = NULL;
    arts_lf_stack_push(&cache->ro_combine, &w->link);
    ro_combine_pump(cache);
    return ARTS_DB_ACQUIRE_PARK;
  }
#endif
#ifdef ARTS_PROTOCOL_INV
  arts_send_db_inv_request(cache, DB_MODE_RO);
#else
  arts_send_db_snapshot_request(cache, edt_guid, slot);
#endif
  return ARTS_DB_ACQUIRE_PARK;
}
#endif /* !ARTS_PROTOCOL_EXCL */

/* The 8-case acquire dispatcher arts_handler_db_acquire is protocol-specific:
 * HOME and OWNER define it in coherence/grant.c-backed
 * each arm's own placement TU (single-owner OWNERSHIP_REQUEST / GRANT path,
 * differing only on the RO-has-local-data predicate); WRF_RCU defines its
 * unified home-canonical body in coherence/wrf_val.c.  The shared remote-RO
 * path (arts_db_acquire_remote_ro) and the local-buffer fast read
 * (arts_db_acquire_local) above are reused by all three.
 */

/* Drain the snapshot reorder buffer in one atomic_exchange.  Monotonic version
 * guarantees every parked node's target_version <= the buffer version that
 * triggers the drain, so a full drain (no partial pop) is always correct
 * (plan: "install 시 전체 drain").  Called from the case-2 install path, the
 * GRANT install, the OWNER TRANSFER_OWNERSHIP install, and destroy fan-out. */
void arts_db_drain_pending_snapshot(struct arts_db_cache_s *cache) {
  arts_lf_link_t *node = arts_lf_stack_drain(&cache->pending_snapshot);
  while (node != NULL) {
    struct arts_db_snapshot_waiter_s *w =
        (struct arts_db_snapshot_waiter_s *)node;
    arts_lf_link_t *next =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    if (w->serve != NULL) {
      /* Deferred remote serve (home parked a GET_DATA while it had no buffer
       * yet): re-issue against the now-installed buffer.  The callback must
       * not retain w past its return. */
      w->serve(cache, w);
      arts_free(w);
    } else {
      arts_guid_t edt_local = w->edt_guid;
      unsigned int slot_local = w->slot;
      arts_free(w);
      /* Wake after free: mark_edt_ready_by_guid re-derives dep->ptr from the
       * (now-installed) cache buffer and resumes the acquire walk. */
      mark_edt_ready_by_guid(edt_local, slot_local);
    }
    node = next;
  }
}

/* ================================================================== */
/* ===== Release path =============================================== */
/* ================================================================== */

/* ===== publish ACK wait (shared coherence service) =================
 *
 * Synchronous PUBLISH with a stack-local semaphore matched by pointer
 * identity.  Called by the HOME and WRF_RCU release-tail bodies (the OWNER
 * tail uses TRANSFER_OWNERSHIP and never waits on a PUBLISH_ACK).  Declared
 * in coherence/coherence.h so the protocol TUs can invoke it. */
void await_publish_ack(sem_t *cv) {
  /* Block on the stack-local semaphore until arts_handler_db_publish_ack
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
 * pre-decrement buffer-ref drop and the post-decrement transfer/publish
 * decision differ per protocol), so its whole body lives in
 * each arm's own placement TU.  Eager and WRF_RCU call arts_db_publish_sync
 * below for the synchronous-PUBLISH rendezvous. */

/* Compiled by every arm that publishes at a release.  Under HOME that is the
 * payload write-through; under OWNER only MSI publishes at all, and its
 * publish is control-only — the round request. */
#if !defined(ARTS_PROTOCOL_EXCL) &&                                          \
    (!defined(ARTS_WRITE_POLICY_WB) || defined(ARTS_PROTOCOL_INV))
void arts_db_publish_sync(struct arts_db_cache_s *cache, uint64_t version,
                            const void *data, uint64_t data_size) {
  unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
  /* The rendezvous lives on the HEAP, not the releaser's stack: the shutdown
   * escape in await_publish_ack can abandon the wait while a CTS/ACK reply
   * is still in flight, and that reply writes the landing fields through the
   * echoed cv before posting.  A heap block deliberately LEAKED on the
   * shutdown escape keeps that late write inside live memory (a bounded,
   * teardown-only leak); a popped stack frame would be corrupted. */
  struct arts_db_pub_rendezvous_s *wr =
      (struct arts_db_pub_rendezvous_s *)arts_malloc(sizeof(*wr));
  sem_init(&wr->sem, 0, 0);
  wr->landing = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  if (home_rank == arts_global_rank_id || data == NULL || data_size == 0) {
    /* Same-rank round (the payload rides inline through the OoO args copy —
     * no wire, no RDMA) or a data-less ordering round: single announce+ACK. */
    arts_send_db_publish(home_rank, cache->db_guid, version,
                           (uint64_t)(uintptr_t)wr, data, data_size,
                           /*rdzv_txid=*/0, /*rdzv_cookie=*/0);
    await_publish_ack(&wr->sem);
    if (arts_atomic_read(&arts_node_info.shutdown_state) != 0) {
      return; /* possible shutdown escape: a late ACK may still post — leak */
    }
    sem_destroy(&wr->sem);
    arts_free(wr);
    return;
  }
  /* Remote dirty round: announce (data_size, txid 0) -> home allocates a
   * fresh landing and replies PUBLISH_CTS -> PUT the dirty bytes -> commit
   * (same packet layout, txid set) -> home pairs {commit, write completion},
   * installs the landing, ACKs. */
  arts_send_db_publish(home_rank, cache->db_guid, version,
                         (uint64_t)(uintptr_t)wr, /*data=*/NULL, data_size,
                         /*rdzv_txid=*/0, /*rdzv_cookie=*/0);
  await_publish_ack(&wr->sem); /* CTS wake — or the shutdown escape */
  if (wr->landing.txid == 0) {
    /* Shutdown escape before the CTS landed: no landing to PUT into; the
     * round is abandoned with the runtime (never a silent data drop in a
     * live run — the CTS wake always carries a landing).  Leak wr: the late
     * CTS may still write/post through the echoed cv.
     *
     * This read of wr->landing.txid is UNSYNCHRONIZED on this escape path —
     * sem_timedwait returned via the shutdown timeout, not a real post, so
     * there is no happens-before edge against a CTS reply that races in
     * concurrently.  That is precisely why wr must be leaked here rather than
     * freed: freeing it and letting the racing write land afterward would be
     * a use-after-free.  Do not "tighten" this into an immediate free. */
    return;
  }
  /* Source lifetime: the caller's buffer ref pins `data` across the PUT; the
   * ACK below follows the target-side write completion, which implies the
   * fabric has fully drained the source — no per-PUT completion hook needed. */
  arts_net_put_payload((int)home_rank, wr->landing.addr, wr->landing.key,
                       wr->landing.txid, data, data_size,
                       /*on_local_done=*/NULL, NULL);
  arts_send_db_publish(home_rank, cache->db_guid, version,
                         (uint64_t)(uintptr_t)wr, /*data=*/NULL, data_size,
                         wr->landing.txid, wr->landing.cookie);
  await_publish_ack(&wr->sem); /* install ACK */
  if (arts_atomic_read(&arts_node_info.shutdown_state) != 0) {
    return; /* possible shutdown escape — leak (late ACK may post) */
  }
  sem_destroy(&wr->sem);
  arts_free(wr);
}
#endif /* !ARTS_WRITE_POLICY_WB && !ARTS_PROTOCOL_EXCL */

#if !defined(ARTS_PROTOCOL_EXCL)
void arts_db_release_ro(struct arts_db_cache_s *cache) {
  /* RO release is a no-op for RCU/WRF_RCU: the EDT's buf ref is dropped
   * by release_one_dep's DIST branch via release_buf (matching the
   * acquire_buf in mark_edt_ready_by_guid / acquire_local).
   * RWLOCK defines its own arts_db_release_ro in coherence/excl/release.c. */
  (void)cache;
}
#endif /* !ARTS_PROTOCOL_EXCL */

/* ================================================================== */
/* ===== Destroy lifecycle ========================================== */
/* ================================================================== */

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
 * The full destructor arts_db_cache_destructor is protocol-specific — it
 * sequences the protocol's own field teardown between these two shared steps —
 * and lives with each arm's placement TU.  The agnostic steps split into pre
 * (the buffer-NULL that must run first) and post (snapshot drain + home
 * teardown), so the per-arm wrapper runs pre, its own teardown, then post.
 * cache_s itself is
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
  /* Drain the per-DB recycled-buffer pool, returning leftovers to the
   * registered pool.  The buffer slot is already NULL'd above, and B1 keeps
   * the descriptor (hence this pool) alive until the last buffer ref drops,
   * so no late deleter can push in after this point; the cache is torn down
   * single-threaded here.  Every node was allocated by arts_db_buf_alloc via
   * arts_regpool_alloc_aligned (pool_link is at offset 0, so the node ptr is
   * the buffer base), so the matching free is arts_regpool_free, not the
   * pool's default arts_free. */
  arts_lf_pool_destroy_with(&cache->buf_freelist, arts_regpool_free);
}

/* Steps 3b+4: drain+free the snapshot reorder buffer (a Treiber stack), then
 * tear down the inlined home-directory sub-resources.  Runs AFTER the protocol
 * field-destroy (pending_rw in HOME and OWNER builds). */
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
#if defined(ARTS_RO_REQUEST_COMBINING) && !defined(ARTS_PROTOCOL_EXCL) 
  {
    /* Combining waiters still parked at destroy are freed, not woken —
     * destroying a DB with a pending acquire is undefined per the programming
     * model, same contract as the reorder-buffer drain above. */
    arts_lf_link_t *n = arts_lf_stack_drain(&cache->ro_combine);
    while (n != NULL) {
      arts_lf_link_t *next =
          atomic_load_explicit(&n->next, memory_order_relaxed);
      arts_free(n);
      n = next;
    }
    n = cache->ro_combine_group;
    cache->ro_combine_group = NULL;
    while (n != NULL) {
      arts_lf_link_t *next =
          atomic_load_explicit(&n->next, memory_order_relaxed);
      arts_free(n);
      n = next;
    }
  }
#endif
  {
    struct arts_db_s *db_self = arts_db_of_cache(cache);
    if (db_self->home_initialized) {
      arts_db_home_teardown(db_self);
      db_self->home_initialized = false;
    }
  }
}
