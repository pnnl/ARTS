/* SPDX-License-Identifier: Apache-2.0
 *
 * RC (Release Consistency) coherence-model translation unit: defines the
 * RC-specific arts_handler_db_* / arts_db_* bodies directly (CMake links
 * exactly this TU for an RC build) plus the RC-only wire handlers/senders.
 * Compiled only when ARTS_MEMORY_MODEL=RC (selected in
 * libs/src/core/CMakeLists.txt). Contains NO ARTS_MEMORY_MODEL_* preprocessor
 * logic.
 */
#include <semaphore.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/home.h"
#include "arts/db.h"
#include "arts/edt.h" /* arts_edt_dep_t (acquire body) */
#include "arts/ooo.h" /* OOO_DB_* args (handler bodies) */
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"     /* arts_global_rank_id */
#include "arts/transport/outbox.h"   /* arts_transport_send_async */
#include "arts/transport/protocol.h" /* arts_fill_packet_header, MSG_* */
#include "arts/utils/atomics.h"      /* arts_atomic_* */

/* ===== 8-case acquire dispatch (RC arm) ============================
 * Whole arts_handler_db_acquire body for the RC build.  Diverges from LRC only
 * on the RO-has-local-data predicate (RC: is_home||is_owner — home always holds
 * current data via synchronous WRITEBACK).  The remote-RO, RW-local-fast, and
 * remote-RW paths are the shared helpers (coherence/coherence.c /
 * coherence/release.c). */
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
  bool is_owner = (cache->writer_count > 0);

  if (mode == DB_MODE_RO) {
    if (is_home || is_owner) { /* RC RO predicate (home holds current data) */
      dep->ptr = arts_db_acquire_local(cache);
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

/* ===== release_rw (RC arm) ========================================
 * RC drops its buffer ref in the tail (after the WRITEBACK reads buf->data), so
 * there is no pre-decrement ref drop.  local_transfer_now restores the sentinel
 * (writer_count = 1) when no waiter is queued, so the writer_count==0 teardown
 * window that LRC must close early does not exist for RC. */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped ownership (e.g. a
   * cache already torn down by a destroy fan-out); decrementing would
   * underflow. Atomic acquire-load avoids a TSan race against concurrent
   * writes. */
  if (arts_atomic_read(&cache->writer_count) == 0) {
    return;
  }
  /* Acquire current buffer for the version bump + WRITEBACK send.  Local ref
   * scoped to release_rw — the EDT's own ref is dropped by release_one_dep. */
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t new_version = 0;
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
    new_version = arts_atomic_read_u64(&buf->version);
  }

  /* Signed, like the INVALIDATE/guard fire sites: the commutative counter can
   * be transiently negative (an INVALIDATE racing ahead of its add), which must
   * read
   * != 0 here.  A true 1->0 release reads 0 and fires; a transient 0->-1 reads
   * -1 and does not. */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1); /* post value */
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);

  if (rest == 0) {
    if (is_home) {
      arts_db_local_transfer_now(cache);
    } else {
      if (buf != NULL) {
        /* RC R4: WRITEBACK_AND_TRANSFER + await ACK (stack-local sem). */
        sem_t cv;
        sem_init(&cv, 0, 0);
        unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
        arts_send_db_writeback(home_rank, cache->db_guid, new_version,
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
    arts_send_db_writeback(home_rank, cache->db_guid, new_version,
                           (uint64_t)(uintptr_t)&cv, ARTS_WB_NORMAL, buf->data,
                           cache->db_size);
    await_writeback_ack(&cv);
    sem_destroy(&cv);
  }
  /* RC: release buffer ref after the writeback (which reads buf->data). */
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }
}

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
void arts_db_rc_advance_chain(struct arts_db_cache_s *cache) {
  struct arts_db_s *db =
      arts_db_of_cache(cache); /* home fields inlined in db */
  for (;;) {
    unsigned int new_owner;
    if (!arts_home_lockreq_queue_pop(&db->pending_rw, &new_owner)) {
      /* Queue empty — chain-end candidate.  Home reclaims ownership so the
       * next foreign OWNERSHIP_REQUEST has a holder to invalidate.
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
    arts_shared_ptr_t master_h = arts_db_buf_acquire(cache);
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
      arts_db_buf_release(&master_h);
    }
    /* Home-driven chain (replaces the old has_next self-relay, which raced a
     * late OWNERSHIP_REQUEST): the new owner does NOT self-withdraw its
     * sentinel. Re-read the queue AFTER the GRANT — if a waiter remains (incl.
     * one that raced in after the pop above), drive the next transfer by
     * INVALIDATEing the owner we just granted.  The commutative signed
     * writer_count makes this GRANT-then-INVALIDATE pair reorder-safe (the
     * INVALIDATE's -1 commutes with the GRANT's +1 and the owner's
     * drain/release; whichever decrement drives writer_count from positive to 0
     * ownership_returns, re-entering this chain). The baton stays 1 across the
     * chain. */
    if (!arts_home_lockreq_queue_empty(&db->pending_rw)) {
      arts_send_db_ownership_invalidate(new_owner, cache->db_guid,
                                        /*new_owner_rank=*/0u);
      /* Pipeline: tell the genuine next owner (FIFO front, post-pop) to advance
       * its RW cursor while the transfer toward it is in flight. */
      unsigned int front;
      if (arts_home_lockreq_queue_peek(&db->pending_rw, &front)) {
        arts_send_db_ownership_proceed(front, cache->db_guid);
      }
      return;
    }
    /* Terminal grant: no waiter — the new owner retains ownership (sentinel
     * kept).  Clear the baton so a later OWNERSHIP_REQUEST can start a fresh
     * round, then re-check for one that raced the clear (same recovery as the
     * queue-empty reclaim path above, but ownership is held by new_owner, so
     * drive the transfer by INVALIDATEing it rather than re-popping). */
    atomic_store_explicit(&db->invalidate_in_flight, 0, memory_order_release);
    if (arts_home_lockreq_queue_empty(&db->pending_rw)) {
      return;
    }
    {
      unsigned int expected = 0u;
      if (!atomic_compare_exchange_strong_explicit(
              &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
              memory_order_acquire)) {
        return; /* a OWNERSHIP_REQUEST producer re-took the baton; it
                   INVALIDATEs rw_holder */
      }
      arts_send_db_ownership_invalidate(new_owner, cache->db_guid,
                                        /*new_owner_rank=*/0u);
      /* Pipeline: PROCEED the genuine next owner (FIFO front) of the
       * race-recovery transfer we just drove. */
      unsigned int front;
      if (arts_home_lockreq_queue_peek(&db->pending_rw, &front)) {
        arts_send_db_ownership_proceed(front, cache->db_guid);
      }
    }
    return;
  }
}

/* RC: the shared chain-advance holds the baton across the has_next chain and
 * clears it at the single queue-empty race-recovery point. */
void arts_db_local_transfer_now(struct arts_db_cache_s *cache) {
  arts_db_rc_advance_chain(cache);
}

/* ===== cache_s lifecycle (RC: pending_rw Vyukov MPSC) =============
 * Construct: RC's model field-init (the Vyukov MPSC pending_rw queue — cannot
 * be zero-initialized, head/tail must point at the embedded stub) runs BEFORE
 * arts_db_cache_common_init so the queue is wired before any push could land.
 * Destruct order: buffer-NULL (pre) → pending_rw destroy → snapshot drain +
 * home teardown (post). */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  arts_pending_rw_queue_init(&c->pending_rw);
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
  arts_home_lockreq_queue_init(&db->pending_rw);
  atomic_store_explicit(&db->invalidate_in_flight, 0, memory_order_relaxed);
  db->last_sent_version = arts_rank_u64_map_create(nranks);
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_lockreq_queue_destroy(&db->pending_rw);
  arts_rank_u64_map_destroy(db->last_sent_version);
  /* No free: home fields are inlined in the arts_db_s. */
}

/* ===== GET_DATA reply (home.last_sent_version atomic-monotonic) ===== */

/* update_last_sent_max: the GET_DATA reply path.  Decide send-with-
 * data vs send-no-data based on the home watermark, then advance the
 * watermark.  Under single-threaded handler dispatch the "atomic
 * CAS-loop" the design plan specifies collapses to a plain compare/
 * advance — but we keep the helper signature so future MPMC upgrades
 * are localized. */
static void update_last_sent_max(struct arts_db_cache_s *cache,
                                 unsigned int requester, uint64_t master_v,
                                 const void *data, uint64_t data_size,
                                 arts_guid_t edt_guid, uint32_t slot) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  /* Monotonic dedup — if the requester already received this version
   * (cur >= master_v), send NO_DATA.  Cache_s lifetime invariant
   * guarantees user_data persists until destroy (route_table ref). */
  uint64_t cur = arts_rank_u64_map_get(db->last_sent_version, requester);
  if (cur >= master_v) {
    arts_send_db_snapshot_response(requester, cache->db_guid, master_v,
                                   edt_guid, slot, NULL, 0);
    return;
  }
  arts_rank_u64_map_set(db->last_sent_version, requester, master_v);
  arts_send_db_snapshot_response(requester, cache->db_guid, master_v, edt_guid,
                                 slot, data, data_size);
}

/* ===== Per-model wire-handler bodies =============================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_SNAPSHOT_REQUEST]): the OoO engine
 * has already acquired the home db_s for db_guid and pinned a ref across this
 * call (cache is its FIRST member), so there is no lookup / NULL-check / defer
 * here.  RC serves from home's canonical buffer with last_sent_version dedup.
 */
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
     *   (a) Sentinel DB (db_size==0): no buffer is ever installed.
     *   (b) HOME_RECV pre-WRITEBACK: cross-rank create has happened but
     *       the creator's first WRITEBACK has not landed; we have a
     *       cache but no buffer (lazy install per OCR pattern).
     * Both cases: respond with version=0, NULL data.  The requester's
     * handle_data_response will deliver ptr=NULL to the parked RO waiter
     * (per spec, "value is undefined" before any writer publishes).
     * NOT a destroy condition -- the precheck above (destroy_state) is
     * authoritative for that. */
    arts_send_db_snapshot_response(requester, cache->db_guid, /*version=*/0,
                                   edt_guid, slot,
                                   /*data=*/NULL, /*data_size=*/0);
    return;
  }
  uint64_t master_v = master->version;
  update_last_sent_max(cache, requester, master_v, master->data, cache->db_size,
                       edt_guid, slot);
  arts_db_buf_release(&master_h);
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_WRITEBACK]): the OoO engine has
 * already acquired the home db_s and pinned a ref across this call (cache is
 * its FIRST member); the dispatcher copies WRITEBACK's trailing data payload
 * into the args blob after arts_ooo_args_db_writeback_s and this body reads it
 * back from (char *)a + sizeof(*a).  A WRITEBACK that races ahead of DB_CREATE
 * MISSes the engine, which defers (trailing data preserved) and re-issues this
 * body on the install's drain.
 *
 * ACK-exactly-once: the body acks (cv != 0) only when it runs (cache live).  A
 * deferred WRITEBACK does NOT ack on the defer — the single WRITEBACK_ACK is
 * sent on the drain replay.  cv==0 marks a fire-and-forget (INVALIDATE-driven
 * WB_AND_TRANSFER from a network handler that cannot block on an ACK). */
void arts_handler_db_writeback(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_writeback_s *a =
      (struct arts_ooo_args_db_writeback_s *)args_v;
  const void *data =
      a->data_size > 0 ? (const void *)((char *)a + sizeof(*a)) : NULL;

  arts_db_buf_install(cache, a->version, data, a->data_size);
  if (a->cv != 0) {
    arts_send_db_writeback_ack(a->releaser, a->db_guid, a->cv);
  }
  /* No snapshot drain here: home's RO acquires hit case 1 (resume self) and
   * never park; foreign ROs are served by GET_DATA, not by drain. */

  /* WB_AND_TRANSFER: the writing owner returned the buffer; advance the
   * transfer chain via the shared helper (baton held across has_next, cleared
   * only at the queue-empty race-recovery point). */
  if (a->flag == ARTS_WB_AND_TRANSFER) {
    arts_db_rc_advance_chain(cache);
  }
}

/* Cat-C pure body (WRITEBACK_ACK).  Cache-independent pointer-identity sem-post
 * on a->cv (the releaser's stack-local sem_t, valid on this rank — the ACK
 * always returns to the sender), so item_v is unused.  The dispatcher posts on
 * BOTH a HIT (this body) and a MISS so a torn-down home cache never strands the
 * blocked releaser. */
void arts_handler_db_writeback_ack(void *item_v, void *args_v) {
  (void)item_v;
  struct arts_db_writeback_ack_args_s *a =
      (struct arts_db_writeback_ack_args_s *)args_v;
  sem_t *s = (sem_t *)(uintptr_t)a->cv;
  if (s != NULL) {
    sem_post(s);
  }
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_DESTROY]): the OoO engine has already
 * acquired the home db_s and pinned a ref across this call (cache is its FIRST
 * member).  Order: roster fan-out + fail_trigger wake parked waiters FIRST (the
 * cache stays alive — only the install ref is dropped), then
 * arts_route_table_set_destroyed LAST detaches the slot cb + drops the install
 * ref; the cb deleter frees the cache once outstanding lookup refs drain.  A
 * second DESTROY_REQ finds the slot absent and is a no-op.  RC roster source =
 * home->last_sent_version + the queued ownership requesters. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  /* RC: use home->last_sent_version as the readers roster, then the queued
   * ownership requesters. */
  {
    unsigned int n = arts_global_rank_count;
    for (unsigned int r = 0; r < n; r++) {
      if (r == self) {
        continue;
      }
      if (arts_rank_u64_map_get(db->last_sent_version, r) > 0) {
        arts_send_db_cache_destroy(r, a->db_guid);
      }
    }
  }
  {
    unsigned int q_rank;
    while (arts_home_lockreq_queue_pop(&db->pending_rw, &q_rank)) {
      if (q_rank != self) {
        arts_send_db_cache_destroy(q_rank, a->db_guid);
      }
    }
  }
  arts_db_fail_trigger_pending(cache);
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* Case-D leaf: RC publishes creator_rank as the home rw_holder (coalesce path).
 */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

/* ===== Ownership-round seams (called from coherence/release.c) ===
 * family→model: the release-family OWNERSHIP_REQUEST / RELEASE_OWNERSHIP
 * handlers delegate the RC/LRC-divergent steps here. */

void arts_db_start_ownership_round(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester) {
  (void)requester;
  arts_send_db_ownership_invalidate(
      atomic_load_explicit(&db->rw_holder, memory_order_acquire),
      cache->db_guid,
      /*new_owner_rank=*/0u);
  /* Pipeline: PROCEED the genuine next owner (FIFO front) so it overlaps its RW
   * acquire with the in-flight initial transfer. */
  unsigned int front;
  if (arts_home_lockreq_queue_peek(&db->pending_rw, &front)) {
    arts_send_db_ownership_proceed(front, cache->db_guid);
  }
}

void arts_db_ownership_return(struct arts_db_cache_s *cache) {
  /* RC: advance the transfer chain via the shared helper (baton held across
   * has_next, cleared only at the queue-empty race-recovery point).  The
   * helper's master==NULL path sends a no-data GRANT so the new owner's waiter
   * still fires, rather than tearing the new owner's cache mid-chain. */
  arts_db_rc_advance_chain(cache);
}

/* ===== RC GRANT handler (moved from handlers.c) ==================== */

void arts_handler_db_ownership_response(
    struct arts_msg_ownership_response_packet_s *p, const void *data,
    uint64_t data_size) {
  struct arts_db_cache_s *cache = arts_db_cache_lookup(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (p->data_present) {
    arts_db_buf_install(cache, p->version, data, data_size);
  }
  /* ADD the ownership sentinel (+1) PLUS a transient DRAIN GUARD (+1) in a
   * single atomic op (so writer_count jumps 0->2 with no intermediate 1 an
   * INVALIDATE could catch at 0).  The guard keeps writer_count >= 1 across the
   * entire grant — the per-waiter +1 drain below counts this rank's parked
   * writers ONE AT A TIME, and a follow-up INVALIDATE (home's has_next chain
   * step, a separate wire) can be processed concurrently on another receiver
   * thread.  Without the guard, an INVALIDATE's -1 landing after the sentinel
   * +1 but BEFORE the first waiter's +1 would drive writer_count to 0 and ship
   * the transfer prematurely, stranding this rank's not-yet-counted waiters.
   * The guard defers that 0-crossing to guard-removal time (after the drain),
   * where it is performed once, correctly. jumping to 2 also closes the
   * late-arrival window: a fresh local RW acquire now sees writer_count > 0 and
   * takes the fast path (cswap +1) instead of parking, so the drain below sees
   * exactly the pre-grant waiter set.  The whole scheme is commutative+signed:
   * sentinel(+1)+guard(+1)+drain(+1 each)+INVALIDATE(-1)+ release(-1)+guard(-1)
   * settle to 0 in any order; only the decrement that crosses to exactly 0
   * ships ownership.  GRANT is RC/LRC only. */
  arts_atomic_add(&cache->writer_count, 2u);
  cache->ownership_req_in_flight = 0;
  /* Drain pending_rw — pop every queued waiter in FIFO order via the
   * Vyukov MPSC consumer path. */
  arts_db_drain_pending_rw_after_grant(cache, p->version, p->has_next != 0);
  /* Drain pending_snapshot: any case-3 reorder-buffer waiter the new buffer
   * install now satisfies. */
  arts_db_drain_pending_snapshot(cache);
  /* Drain the OoO slot: a GRANT installs this rank's cache, so an
   * INVALIDATE_NOTICE that raced ahead of the GRANT (two-wire reorder) and
   * deferred on a missing cache now replays against the just-installed cache
   * (§6.1).  Idempotent when no INVALIDATE is queued. */
  arts_ooo_drain_guid(p->db_guid);
  /* Remove the drain guard.  All of this rank's parked writers are now counted
   * (+1 each) and any deferred INVALIDATE has replayed, so the count is stable.
   * The guard's signed -1 performs the deferred 0-crossing check: if a racing
   * INVALIDATE withdrew the sentinel while no local writer remains, the count
   * reaches exactly 0 here (not transiently mid-drain) and this is the unique
   * actor that returns ownership to home. */
  if ((int)arts_atomic_sub(&cache->writer_count, 1) == 0) {
    arts_db_invalidate_transfer(cache);
  }
}

/* ===== RC INVALIDATE_NOTICE handler (Cat-B pure body) ============== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_OWNERSHIP_INVALIDATE]): the engine
 * has already acquired the db_s for db_guid and pinned a ref across this call,
 * so there is no lookup / NULL-check here.  cache is the FIRST member of
 * arts_db_s (offset 0), so the slot object the engine hands us IS the cache.
 * The wire dispatcher decodes INVALIDATE_NOTICE into the args and routes
 * through the engine via OOO_DB_OWNERSHIP_INVALIDATE; an INVALIDATE that races
 * ahead of the cache install (a reordered GRANT/INVALIDATE on two wires, or a
 * before-create race) MISSes the engine, which DEFERS the args and re-issues
 * this body once the cache installs and drains (§6.1) — the invalidation is not
 * lost.  The commutative signed writer_count makes the replayed decrement
 * order-independent w.r.t. the GRANT's sentinel +1. */
void arts_handler_db_ownership_invalidate(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_ownership_invalidate_s *a =
      (struct arts_ooo_args_db_ownership_invalidate_s *)args_v;
  (void)a; /* RC ignores new_owner_rank (LRC consumes it). */
  /* Sentinel withdrawal (writer_count -= 1).  Home's invalidate_in_flight gate
   * sends AT MOST ONE INVALIDATE_NOTICE to this rank per transfer round, after
   * rw_holder has been advanced to a rank that already holds the sentinel (+1).
   * The decrement that drives writer_count to 0 is the unique actor that
   * performs the ownership transfer; while local writers are still active
   * (rest > 0) the last release_rw drives it instead.
   *
   * Commutative signed counter.  The baton gate above makes GRANT-then-
   * INVALIDATE the order this holder normally sees (the sentinel +1 is already
   * installed when the notice arrives).  The signed counter additionally
   * tolerates the reordered case: an INVALIDATE that races ahead of its GRANT
   * leaves writer_count transiently negative, which is not a defect — hence no
   * underflow assert.  Only the decrement that drives writer_count from a
   * positive value to exactly 0 owns the transfer; a transient 0 -> -1 reads as
   * rest<0 and does NOT trigger. */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1);
  if (rest == 0) {
    arts_db_invalidate_transfer(cache);
  }
}

/* ===== RC GRANT sender (moved from coherence/senders.c) ========== */

void arts_send_db_ownership_response(unsigned int requester_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     bool has_next, const void *data,
                                     uint64_t data_size) {
  struct arts_msg_ownership_response_packet_s p;
  uint64_t total = sizeof(p) + (data ? data_size : 0);
  arts_fill_packet_header(&p.header, total, MSG_DB_OWNERSHIP_RESPONSE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.has_next = has_next ? 1u : 0u;
  p.data_present = data ? 1u : 0u;
  memset(p.pad, 0, sizeof(p.pad));
  if (requester_rank == arts_global_rank_id) {
    arts_handler_db_ownership_response(&p, data, data ? data_size : 0);
    return;
  }
  if (data && data_size > 0) {
    arts_transport_send_payload_async((int)requester_rank, (char *)&p,
                                      sizeof(p), (char *)data, data_size);
  } else {
    arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
  }
}

/* Case-D leaf: RC keeps the lazy OCR home-buffer install (the creator's
 * release_rw WRITEBACK / GRANT path publishes the buffer); nothing to do at
 * create. */
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  (void)cache;
  (void)db_size;
}
