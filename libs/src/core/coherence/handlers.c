/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message handlers (receive side) + home-side
 * dedup / ownership-transfer helpers.
 *
 * The matching arts_send_db_* SENDERS (packet-fill + outbox enqueue) live
 * in coherence_senders.c.
 *
 * Lookup discipline.  Two handler categories, both lookup-then-operate but
 * differing on the MISS action:
 *   - Cat-B (deferrable home-side: OWNERSHIP_REQUEST / GET_DATA / WRITEBACK /
 * DESTROY): the wire dispatcher routes through the OoO engine, which acquires
 * the home db_s (ref-pinned) and hands a pure (item, args) body the live cache
 *     on a HIT, or DEFERS the args and replays them once DB_CREATE installs.
 *   - Cat-C (non-deferrable: DATA_RESPONSE / DESTROY_NOTIFY / WRITEBACK_ACK /
 *     RELEASE_OWNERSHIP / REDIRECT_RO / CONFIRM / CONFIRM_ACK): the wire
 * dispatcher (and the matching self-send shortcut) does the ref-pinned
 *     lookup-acquire; on a HIT it calls the pure (item, args) body, and on a
 *     MISS it applies that handler's exact miss-action (silent drop,
 *     DESTROY_NOTIFY reply, or the WRITEBACK_ACK sem-post — see each body).
 * Either way the handler body itself performs NO route-table lookup; it
 * operates on the already-acquired, ref-pinned cache (the FIRST member of the
 * db_s the caller hands it).
 *
 * Memory ordering: ARTS atomics are __sync_*-based (full fence) and
 * the per-rank network thread (S1) is the sole writer of home.*
 * state, so the trickier orderings are confined to:
 *   - cache.writer_count (worker ↔ network handler)
 *   - cache.buffer       (worker ↔ network handler installs)
 *   - cache.pending_count (worker push ↔ marker mark)
 * All accessed via arts_atomic_*, all matching the algorithm in the
 * coherence design plan.
 */

#include "arts/coherence/handlers.h"

#include <assert.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h" /* ARTS_DB_PROP_NO_ACQUIRE */
#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/home.h"
#include "arts/db.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

/* ===== Home-side handlers ========================================== */

/* arts_handler_db_ownership_request lives in coherence/ownership.c (MRNEW
 * only — MRMW has no OWNERSHIP_REQUEST / GRANT round). */

/* arts_handler_db_snapshot_request (GET_DATA) is protocol-specific —
 * EAGER/MRMW serve from home's canonical buffer (dedup), LAZY records the
 * sharer + REDIRECTs to the owner — so its whole body lives in
 * coherence/{eager,lazy,mrmw}.c. */

/* arts_handler_db_writeback (+_ack) is protocol-specific — EAGER/MRMW install
 * + ACK (pure: ownership transfer is a separate owner→owner OWNERSHIP_RESPONSE
 * ship), LAZY has no synchronous writeback (no-op fillers preserve the
 * OoO-table / link parity) — so their whole bodies live in
 * coherence/{eager,lazy,mrmw}.c. */

/* arts_handler_db_destroy is protocol-specific — the roster fan-out source
 * differs (eager/MRMW walk home->last_sent_version; lazy walks rw_holder +
 * cached_ranks + pending_rw) — so its whole body lives in
 * coherence/{eager,lazy,mrmw}.c.  All three skeletons run the roster fan-out,
 * then arts_route_table_set_destroyed; any waiter left parked at destroy time
 * (UB per OCR) is cleaned up by the refcount-0 cache destructor. */

/* NO_ACQUIRE home normalization.  The creator neither acquires nor releases,
 * so the home is the sole idle owner.  Every home create path seeds a
 * create-time RW hold (cache_init CREATOR_HOME / arts_db_home_init) that, for
 * NO_ACQUIRE, no EDT will ever release.  That seed must be undone so the first
 * real acquirer is granted rather than blocked behind a hold nothing releases.
 * Mirrors the local-create path: under a single-writer lock the unreleased hold
 * deadlocks every future writer; the ownership protocols collapse the seed to
 * the sentinel (writer_count = 1).  Idempotent and safe to call on every create
 * path (fresh install and lazy-stub coalesce). */
static inline void db_create_no_acquire_idle(struct arts_db_s *db,
                                             bool no_acquire) {
  if (!no_acquire) {
    return;
  }
#if defined(ARTS_PROTOCOL_LOCK)
#if defined(ARTS_TIMING_LAZY)
  /* LAZY: data lives with the owner, not the home.  With no creator hold there
   * is no owner unless we make one — so the home rank (this rank; the create
   * handler runs only on the GUID home, see the assert in
   * arts_handler_db_create) becomes the IDLE data owner: it holds the zero-init
   * buffer (installed by the create flow) with owner-bit set but rw_st=IDLE,
   * wc=0.  The first writer's REQUEST then migrates that zero buffer from here,
   * exactly like a sticky owner that has finished its writers.  lock_state is
   * the idle directory naming this rank as owner. */
  atomic_store_explicit(&db->cache.cache_state,
                        CACHE_MAKE_FULL(1u, CACHE_ST_IDLE, CACHE_ST_IDLE,
                                        ARTS_LOCK_NO_TARGET, 0u, 0u),
                        memory_order_relaxed);
  atomic_store_explicit(&db->lock_state,
                        LOCK_MAKE(LOCK_PHASE_IDLE, arts_global_rank_id, 0u, 0u),
                        memory_order_relaxed);
#else  /* ARTS_TIMING_EAGER */
  /* EAGER: the home holds the canonical buffer and grants from it; the creator
   * is a non-owner.  Idle both words (the first LOCK_REQUEST is granted, not
   * blocked behind the unreleased creator hold). */
  atomic_store_explicit(&db->cache.cache_state, 0ULL, memory_order_relaxed);
  atomic_store_explicit(&db->lock_state, 0ULL, memory_order_relaxed);
#endif /* ARTS_TIMING_* */
#else
  db->cache.writer_count = 1;
#endif
}

void arts_handler_db_create(struct arts_msg_db_create_coherent_packet_s *p) {
  /* Home-side init for non-home creator.  Per coherence design plan
   * §968-988: install zero-init buffer, home struct with rw_holder =
   * creator_rank, writer_count = 0 (home is non-owner). */
  unsigned int creator_rank = p->header.rank;
  arts_guid_t db_guid = p->db_guid;
  uint64_t db_size = p->db_size;
  /* NO_ACQUIRE: the creator neither acquires nor writes back, so home is the
   * sole idle owner (not a non-owner awaiting a creator writeback). */
  bool no_acquire = (p->flags & ARTS_DB_PROP_NO_ACQUIRE) != 0;

  /* This handler installs/initializes the home directory, so it is only ever
   * dispatched to the GUID's home rank — where the descriptor is always a
   * full allocation.  A cache-only stub (which omits the home-arm fields)
   * is only ever installed on non-home ranks, so the home-field writes below
   * (arts_db_home_init / rw_holder) stay in bounds. */
  assert((unsigned int)arts_guid_get_rank(db_guid) == arts_global_rank_id &&
         "home-directory init must run on the GUID home rank");

  /* Race against lazy_install or another path that already set up an
   * empty cache_s on this rank — coalesce by promoting the existing
   * lazy entry rather than allocating a duplicate. */
  arts_shared_ptr_t existing_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *existing = (struct arts_db_s *)arts_shared_get(existing_h);
  if (existing != NULL && existing->db_type == ARTS_DB) {
    struct arts_db_cache_s *cache = &existing->cache;
    struct arts_db_s *db = existing;
    arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
    bool buf_absent = (arts_shared_get(buf_h) == NULL);
    arts_db_buf_release(&buf_h);
    if (buf_absent && db_size > 0) {
      arts_db_buf_install(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (!db->home_initialized) {
      arts_db_home_init(db, creator_rank, arts_global_rank_count);
      db->home_initialized = true;
    } else {
      arts_db_create_publish_holder(db, creator_rank);
    }
    db_create_no_acquire_idle(db, no_acquire);
    arts_shared_release(&existing_h);
    return;
  }
  if (existing != NULL) {
    /* existing non-ARTS_DB entry (no coherence cache) — drop the ref and
     * proceed to the install/coalesce branch below. */
    arts_shared_release(&existing_h);
  }

  /* No existing entry -- allocate the db_s stub (cache embedded), install in
   * route_table.
   *
   * Lazy buffer install (OCR pattern): HOME_RECV does NOT install a
   * buffer here.  cache->buffer stays NULL with version 0 -- "metadata
   * only" state.  The first WRITEBACK from the creator's release_rw
   * installs the buffer at home (version 1+, with the creator's
   * payload).  Cross-rank GET_DATA before that point is served as a
   * no-payload DATA_RESPONSE (handle_get_data); the requesting rank
   * sees ptr=NULL (per OCR spec ch2:832-839 "value of the created data
   * block is undefined" -- ARTS interprets this as "before any writer
   * has published, no data exists; reading is application's
   * responsibility"). */
  struct arts_db_s *stub = (struct arts_db_s *)arts_malloc_align(
      sizeof(struct arts_db_s), ARTS_CACHE_LINE_SIZE);
  memset(stub, 0, sizeof(struct arts_db_s));
  stub->db_type = (arts_db_types_t)p->db_type;
  if (no_acquire) {
    /* Home is the idle RW owner from creation — identical to a locally created
     * DB that has already been released by its creator.  HOME_RECV (rw_holder =
     * creator) would route the first INVALIDATE_NOTICE to a creator that holds
     * no cache — a phantom holder — and the acquire would stall forever.
     *
     * CREATOR_HOME sets writer_count = sentinel(1) + creator_hold(1) = 2, but
     * NO_ACQUIRE means no EDT will ever release the creator hold.  Decrement to
     * 1 (sentinel only) so the first OWNERSHIP_REQUEST's INVALIDATE-to-self
     * drives writer_count to 0, triggering advance_chain and the GRANT. */
    arts_db_cache_init(&stub->cache, db_guid, db_size,
                       ARTS_DB_INIT_CREATOR_HOME, creator_rank);
    /* Collapse the create-time creator hold to the idle/sentinel state:
     * MRNEW/MRSW drop writer_count 2 -> 1 (sentinel only); LOCK frees the
     * lock+cache state so the first OWNERSHIP_REQUEST / LOCK_REQUEST is granted
     * rather than blocked behind a hold no EDT will ever release. */
    db_create_no_acquire_idle(stub, no_acquire);
    if (db_size > 0) {
      arts_db_buf_install(&stub->cache, /*new_version=*/1,
                          /*data_payload=*/NULL, db_size);
    }
  } else {
    arts_db_cache_init(&stub->cache, db_guid, db_size, ARTS_DB_INIT_HOME_RECV,
                       creator_rank);
    /* Case-D leaf: MRMW installs a version-1 zero buffer now (home is
     * canonical, no creator writeback to wait for); eager/lazy defer the
     * install to the creator's first WRITEBACK (no-op here). */
    arts_db_create_install_home_buffer(&stub->cache, db_size);
  }

  if (arts_route_table_install_if_absent(stub, db_guid, arts_global_rank_id,
                                         /*used=*/true)) {
    arts_ooo_drain_guid(db_guid);
    return;
  }

  /* Lost race — free our stub and coalesce into the existing entry. */
  arts_db_free(stub);
  arts_shared_ptr_t winner_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *winner = (struct arts_db_s *)arts_shared_get(winner_h);
  if (winner != NULL && winner->db_type == ARTS_DB) {
    struct arts_db_cache_s *cache = &winner->cache;
    struct arts_db_s *db = winner;
    arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
    bool buf_absent = (arts_shared_get(buf_h) == NULL);
    arts_db_buf_release(&buf_h);
    if (buf_absent && db_size > 0) {
      arts_db_buf_install(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (!db->home_initialized) {
      arts_db_home_init(db, creator_rank, arts_global_rank_count);
      db->home_initialized = true;
    } else {
      arts_db_create_publish_holder(db, creator_rank);
    }
    db_create_no_acquire_idle(db, no_acquire);
  }
  if (winner != NULL) {
    arts_shared_release(&winner_h);
  }
}

/* ===== Sharer-side response handlers =============================== */

/* The EAGER GRANT handler arts_handler_db_ownership_response lives in
 * coherence/eager.c; LAZY's TRANSFER_OWNERSHIP overload lives in
 * coherence/lazy.c; MRMW has no ownership transfer (dispatcher fatals). */

/* Cat-C pure body (DATA_RESPONSE).  The wire dispatcher / self-send shortcut
 * has already looked the home db_s up with a held ref and passes it as item_v
 * (cache is its FIRST member, offset 0, so item_v IS the cache).  No
 * lookup/NULL-check here — the dispatcher's MISS branch SILENTLY DROPS (this
 * 1:1 response resumes a parked EDT; a missing cache means it was torn down).
 *
 * No acquire-time list registration: this 1:1 response resumes the parked EDT
 * (a->edt_guid, a->slot) directly.  Three cases over a monotonic version:
 *   1. a->version <= buf->version : nothing newer to install — resume self
 *      against the live buffer.
 *   2. data + a->version > buf->version : install + drain-all the reorder
 *      buffer + resume self.
 *   3. NO_DATA + a->version > buf->version : the with-data reply was
 *      reordered behind us — push self onto pending_snapshot (a future
 *      case-2 install drains us) + re-check (race recovery).
 * Shared verbatim by eager/lazy/MRMW (MRMW routes RW through here too). */
void arts_handler_db_snapshot_response(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_snapshot_response_args_s *a =
      (struct arts_db_snapshot_response_args_s *)args_v;
  arts_guid_t edt_guid = a->edt_guid;
  uint32_t slot = a->slot;

  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t buf_v = buf ? buf->version : 0;
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  if (a->version <= buf_v) {
    /* Case 1: nothing newer to install — resume self. */
    mark_edt_ready_by_guid(edt_guid, slot);
    return;
  }
  if (a->data_present) {
    /* Case 2: install (version-conditional publish inside install_buffer;
     * stale installs retreat) + drain-all + resume self. */
    arts_db_buf_install(cache, a->version, a->data, a->data_size);
    arts_db_drain_pending_snapshot(cache);
    mark_edt_ready_by_guid(edt_guid, slot);
    return;
  }
  /* Case 3: NO_DATA arrived ahead of the with-data reply.  Park a
   * reorder-buffer node; a later case-2 install drains it. */
  struct arts_db_snapshot_waiter_s *w =
      (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt_guid;
  w->slot = slot;
  w->target_version = a->version;
  arts_lf_stack_push(&cache->pending_snapshot, &w->link);
  /* Race recovery: a concurrent case-2 install may have published the buffer
   * between our version read and the push.  If so, drain (our own node
   * included) so we don't park forever.  The atomic_exchange drain is the
   * single-actor primitive — a concurrent installer's drain and ours cannot
   * both claim the same node. */
  arts_shared_ptr_t rch = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *rbuf =
      (struct arts_db_buffer_s *)arts_shared_get(rch);
  uint64_t rv = rbuf ? rbuf->version : 0;
  if (rbuf != NULL) {
    arts_db_buf_release(&rch);
  }
  if (rv >= a->version) {
    arts_db_drain_pending_snapshot(cache);
  }
}

/* arts_handler_db_ownership_invalidate (INVALIDATE_NOTICE) lives per protocol:
 * coherence/eager.c (commutative signed counter) and coherence/lazy.c
 * (publish-target-then-withdraw).  MRMW never sends INVALIDATE (dispatcher
 * fatals). */

/* arts_handler_db_writeback_ack is protocol-specific — EAGER/MRMW post the
 * releaser's stack-local sem_t (pointer identity), LAZY has no synchronous
 * writeback (no-op filler for OoO-table / link parity) — so its whole body
 * lives in coherence/{eager,lazy,mrmw}.c. */

/* Cat-C pure body (DESTROY_NOTIFY).  The wire dispatcher / self-send shortcut
 * has already looked the cache up with a held ref and passes the db_s as item_v
 * (cache is its FIRST member, offset 0).  No lookup/NULL-check here — the
 * dispatcher's MISS branch SILENTLY DROPS (already torn down on this rank;
 * cb-NULL = idempotent, a second DESTROY_NOTIFY is a no-op).
 *
 * Destroy is just the route-slot detach (CAS value→NULL + drop the install
 * ref); the cb deleter (arts_db_cache_destructor) runs at refcount 0 and does
 * the cleanup (free the parked-waiter nodes).  Destroying a DB that an EDT
 * still has a pending dependence on is undefined per OCR (ocrDbDestroy: the
 * user ensures the DB is not in use), so no parked-EDT wake is attempted. */
void arts_handler_db_cache_destroy(void *item_v, void *args_v) {
  (void)item_v;
  struct arts_db_cache_destroy_args_s *a =
      (struct arts_db_cache_destroy_args_s *)args_v;
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* The LAZY REDIRECT_RO handler arts_handler_db_snapshot_redirect lives in
 * coherence/lazy.c (owner-side, REDIRECT only exists under LAZY). */
