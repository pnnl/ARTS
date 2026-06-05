/* SPDX-License-Identifier: Apache-2.0
 *
 * RC (Release Consistency) coherence-model hook implementations plus RC-only
 * wire handlers/senders. Compiled only when ARTS_MEMORY_MODEL=RC (selected in
 * libs/src/core/CMakeLists.txt). Contains NO ARTS_MEMORY_MODEL_* preprocessor
 * logic.
 */
#include <semaphore.h>
#include <stdint.h>
#include <string.h>

#include "arts/db.h"
#include "arts/db_coherence_home.h"
#include "arts/db_coherence_model.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"     /* arts_global_rank_id */
#include "arts/transport/outbox.h"   /* arts_remote_send_request_async */
#include "arts/transport/protocol.h" /* arts_fill_packet_header, MSG_* */

/* Forward declarations for the snapshot drain helper defined in
 * db_coherence_handlers.c, called from the RC handlers below.
 * (arts_coh_drain_pending_rw_after_grant and mark_edt_ready_by_guid are
 * declared in db_coherence_model.h.) */
void arts_coh_drain_pending_snapshot(struct arts_db_cache_s *cache);

/* RC: WRITEBACK is synchronous (sender spins on ACK), so home always holds
 * current data before any RO acquire can execute.  is_home is sufficient to
 * guarantee a non-NULL buffer. */
bool arts_coh_model_ro_has_local_data(bool is_home, bool is_owner) {
  return is_home || is_owner;
}

/* RC drops its buffer ref in the tail (after the writeback reads buf->data), so
 * the pre-decrement seam is a no-op.  local_transfer_now restores the sentinel
 * (writer_count = 1) when no waiter is queued, so the writer_count==0 teardown
 * window that LRC must close here does not exist for RC. */
void arts_coh_model_release_rw_pre_decrement(struct arts_db_cache_s *cache,
                                             arts_shared_ptr_t *buf_h,
                                             struct arts_db_buffer_s **buf) {
  (void)cache;
  (void)buf_h;
  (void)buf;
}

void arts_coh_model_release_rw_tail(struct arts_db_cache_s *cache,
                                    arts_shared_ptr_t *buf_h,
                                    struct arts_db_buffer_s *buf,
                                    uint64_t new_version, unsigned int rest,
                                    bool is_home) {
  if (rest == 0) {
    if (is_home) {
      arts_coh_local_transfer_now(cache);
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
    arts_coh_release_buf(buf_h);
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

/* RC: the shared chain-advance holds the baton across the has_next chain and
 * clears it at the single queue-empty race-recovery point. */
void arts_coh_local_transfer_now(struct arts_db_cache_s *cache) {
  arts_coh_rc_advance_chain(cache);
}

/* ===== cache_s lifecycle (model field init/teardown) =============== */

void arts_coh_model_init_cache_s(struct arts_db_cache_s *c) {
  arts_pending_rw_queue_init(&c->pending_rw);
}

void arts_coh_model_cache_destructor(struct arts_db_cache_s *c) {
  arts_pending_rw_queue_destroy(&c->pending_rw);
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

/* ===== Per-model wire-handler body hooks =========================== */

void arts_coh_model_snapshot_request_serve(struct arts_db_cache_s *cache,
                                           unsigned int requester,
                                           arts_guid_t edt_guid,
                                           uint32_t slot) {
  arts_shared_ptr_t master_h = arts_coh_acquire_buf(cache);
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
  arts_coh_release_buf(&master_h);
}

void arts_coh_model_writeback_transfer(struct arts_db_cache_s *cache) {
  /* RC: the writing owner returned the buffer; advance the transfer chain via
   * the shared helper (baton held across has_next, cleared only at the
   * queue-empty race-recovery point). */
  arts_coh_rc_advance_chain(cache);
}

void arts_coh_model_writeback_ack(uint64_t cv) {
  /* Pointer-identity wakeup: cv is the address of the releaser's stack-local
   * sem_t (valid on this rank — the ACK always returns to the rank that sent
   * the WRITEBACK).  Post it to wake the parked release_rw.  No route_table
   * lookup, no seq matching. */
  sem_t *s = (sem_t *)(uintptr_t)cv;
  if (s != NULL) {
    sem_post(s);
  }
}

void arts_coh_model_destroy_fanout(struct arts_db_cache_s *cache,
                                   unsigned int self) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_guid_t db_guid = cache->db_guid;
  /* RC: use home->last_sent_version as the readers roster. */
  {
    unsigned int n = arts_global_rank_count;
    for (unsigned int r = 0; r < n; r++) {
      if (r == self) {
        continue;
      }
      if (arts_rank_u64_map_get(db->last_sent_version, r) > 0) {
        arts_send_db_cache_destroy(r, db_guid);
      }
    }
  }
  {
    unsigned int q_rank;
    while (arts_home_lockreq_queue_pop(&db->pending_rw, &q_rank)) {
      if (q_rank != self) {
        arts_send_db_cache_destroy(q_rank, db_guid);
      }
    }
  }
}

void arts_coh_model_db_create_set_holder(struct arts_db_s *db,
                                         unsigned int creator_rank) {
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

/* ===== Ownership-round seams (called from db_coherence_release.c) === */

void arts_coh_model_start_ownership_round(struct arts_db_cache_s *cache,
                                          struct arts_db_s *db,
                                          unsigned int requester) {
  (void)requester;
  arts_send_db_ownership_invalidate(
      atomic_load_explicit(&db->rw_holder, memory_order_acquire),
      cache->db_guid,
      /*new_owner_rank=*/0u);
}

void arts_coh_model_ownership_return(struct arts_db_cache_s *cache) {
  /* RC: advance the transfer chain via the shared helper (baton held across
   * has_next, cleared only at the queue-empty race-recovery point).  The
   * helper's master==NULL path sends a no-data GRANT so the new owner's waiter
   * still fires, rather than tearing the new owner's cache mid-chain. */
  arts_coh_rc_advance_chain(cache);
}

/* ===== RC GRANT handler (moved from handlers.c) ==================== */

void arts_handler_db_ownership_response(
    struct arts_remote_ownership_response_packet_s *p, const void *data,
    uint64_t data_size) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  if (p->data_present) {
    arts_coh_install_buffer(cache, p->version, data, data_size);
  }
  /* ADD the ownership sentinel (+1), do NOT set.  With a single receiver thread
   * the follow-up INVALIDATE is delivered to this owner after the GRANT (same
   * per-destination outbox, program order), so writer_count is 0 here and +=1
   * equals the old =1.  The additive form does not DEPEND on that ordering: if
   * the GRANT/INVALIDATE pair is ever reordered (e.g. multiple receiver
   * threads), an INVALIDATE arriving first pre-decrements writer_count to
   * (signed) -1, and the additive sentinel commutes with it — interpreted
   * signed, INVALIDATE(-1)+GRANT(+1)+drain(+1)+release(-1) settle to 0 in any
   * order, and the op that drives writer_count from positive to 0 returns
   * ownership.  GRANT is RC/LRC only (LC uses DATA_RESPONSE). */
  arts_atomic_add(&cache->writer_count, 1u);
  cache->ownership_req_in_flight = 0;
  /* Drain pending_rw — pop every queued waiter in FIFO order via the
   * Vyukov MPSC consumer path. */
  arts_coh_drain_pending_rw_after_grant(cache, p->version, p->has_next != 0);
  /* Drain pending_snapshot: any case-3 reorder-buffer waiter the new buffer
   * install now satisfies. */
  arts_coh_drain_pending_snapshot(cache);
}

/* ===== RC INVALIDATE_NOTICE handler (moved from handlers.c) ======== */

void arts_handler_db_ownership_invalidate(
    struct arts_remote_ownership_invalidate_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    /* cache==NULL here means the DB was destroyed: the route_item value is
     * atomic-exchanged to NULL only by the destroy fan-out.  An INVALIDATE is
     * sent exclusively to the rank that currently holds the DB (home publishes
     * rw_holder before any INVALIDATE can name a target, and that holder
     * installed its cache at acquire time), so a live, never-destroyed target
     * always has a cache.  A missing cache therefore means the holder is gone
     * and the sentinel withdrawal is moot — dropping is correct.  Deferring
     * onto the OoO list would be WRONG: a sharer-side message gets no
     * CREATE-driven drain, so it could only replay on a future same-GUID
     * re-creation and underflow that fresh DB's writer_count. */
    return;
  }
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
    arts_coh_invalidate_transfer(cache);
  }
}

/* ===== RC GRANT sender (moved from db_coherence_senders.c) ========== */

void arts_send_db_ownership_response(unsigned int requester_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     bool has_next, const void *data,
                                     uint64_t data_size) {
  struct arts_remote_ownership_response_packet_s p;
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
    arts_remote_send_request_payload_async((int)requester_rank, (char *)&p,
                                           sizeof(p), (char *)data, data_size);
  } else {
    arts_remote_send_request_async((int)requester_rank, (char *)&p, sizeof(p));
  }
}

/* RC/LRC keep the lazy OCR home-buffer install (the creator's release_rw
 * WRITEBACK / GRANT path publishes the buffer); nothing to do at create. */
void arts_coh_model_create_home_buffer(struct arts_db_cache_s *cache,
                                       uint64_t db_size) {
  (void)cache;
  (void)db_size;
}
