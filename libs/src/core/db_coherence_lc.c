/* SPDX-License-Identifier: Apache-2.0
 *
 * LC (Location Consistency) coherence-model hook implementations plus LC-only
 * wire handlers/senders. Compiled only when ARTS_MEMORY_MODEL=LC (selected in
 * libs/src/core/CMakeLists.txt). Contains NO ARTS_MEMORY_MODEL_* preprocessor
 * logic.
 */
#include <semaphore.h>
#include <stdint.h>

#include "arts/db.h"
#include "arts/db_coherence_home.h"
#include "arts/db_coherence_model.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"

/* LC 8-case dispatch: home holds the canonical buffer (maintained by sync
 * WRITEBACK from every non-home writer).  RW and RO are unified — non-home
 * acquires go through acquire_remote_ro in both modes so the EDT parks (no list
 * registration) and is woken by DATA_RESPONSE once home delivers its current
 * buffer.  There is no LOCK_REQ / INVALIDATE / GRANT round, no per-cache
 * pending_rw queue.
 *
 * RW acquires bump writer_count BEFORE parking (or before acquire_local on
 * home).  release_rw balances this decrement; without the bump, release_rw's
 * writer_count == 0 guard silently skips the WRITEBACK, breaking cross-rank RW
 * visibility.  RO acquires do not bump because release_ro is a no-op. */
arts_db_acquire_result_t
arts_coh_model_acquire_dispatch(struct arts_db_cache_s *cache,
                                arts_edt_dep_t *dep, arts_guid_t edt_guid,
                                unsigned int slot, arts_db_access_mode_t mode,
                                bool is_home, bool is_owner) {
  (void)is_owner;
  if (mode == DB_MODE_RW) {
    arts_atomic_add(&cache->writer_count, 1);
  }
  if (is_home) {
    dep->ptr = arts_coh_acquire_local(cache);
    return ARTS_DB_ACQUIRE_OK;
  }
  return arts_coh_acquire_remote_ro(cache, edt_guid, slot);
}

/* LC has no pending_rw queue (all modes park on pending_snapshot), so the
 * destroy/fail fan-out of pending_rw is a no-op; the caller still drains
 * pending_snapshot separately. */
void arts_coh_model_fail_trigger_pending_rw(struct arts_db_cache_s *cache) {
  (void)cache;
}

/* LC drops its buffer ref in the tail (after the writeback reads buf->data), so
 * the pre-decrement seam is a no-op. */
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
  (void)rest;
  /* LC release: every non-home write must be pushed back to home synchronously
   * so home remains canonical before any subsequent acquire can see fresh data.
   * Home itself needs no WRITEBACK.
   *
   * R3: intermediate release (rest > 0, non-home) — same sync writeback.
   * R4: last release (rest == 0, non-home) — sync writeback to home. */
  if (!is_home && buf != NULL) {
    sem_t cv;
    sem_init(&cv, 0, 0);
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_writeback(home_rank, cache->db_guid, new_version,
                           (uint64_t)(uintptr_t)&cv, ARTS_WB_NORMAL, buf->data,
                           cache->db_size);
    await_writeback_ack(&cv);
    sem_destroy(&cv);
  }
  /* Release the buffer ref held for the version-bump and WRITEBACK read. */
  if (buf != NULL) {
    arts_coh_release_buf(buf_h);
  }
}

/* ===== cache_s lifecycle (LC has no pending_rw queue) ============== */

void arts_coh_model_init_cache_s(struct arts_db_cache_s *c) { (void)c; }

void arts_coh_model_cache_destructor(struct arts_db_cache_s *c) { (void)c; }

/* ===== home-directory lifecycle (LC: only last_sent_version) ======= */

void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks) {
  /* LC has no exclusive owner — rw_holder is unused. */
  (void)rw_holder;
  db->last_sent_version = arts_rank_u64_map_create(nranks);
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
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
  /* LC uses WRITEBACK_NORMAL only (no exclusive owner to transfer to), so the
   * WB_AND_TRANSFER ownership-chain relay is a no-op. */
  (void)cache;
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
  /* LC: use home->last_sent_version as the readers roster (same as RC).
   * LC has no pending_rw queue: skip the lockreq drain. */
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

void arts_coh_model_db_create_set_holder(struct arts_db_s *db,
                                         unsigned int creator_rank) {
  /* LC has no exclusive owner — no rw_holder to publish. */
  (void)db;
  (void)creator_rank;
}

/* LC never sends LOCK_REQ, so OOO_DB_OWNERSHIP_REQUEST is never enqueued; the
 * OoO dispatch table entry is satisfied by this no-op. */
void arts_coh_ooo_replay_ownership_request(void *item, void *vargs) {
  (void)item;
  (void)vargs;
}

/* ===== create-time home buffer (LC home is canonical) ============== */

void arts_coh_model_create_home_buffer(struct arts_db_cache_s *cache,
                                       uint64_t db_size) {
  /* LC home holds the canonical copy; there is no creator WRITEBACK to wait
   * for, so publish a version-1 zero buffer immediately.  Without it the first
   * home RW acquire (acquire_local) hands the EDT a NULL payload. */
  if (db_size > 0 && arts_coh_buffer_peek(cache) == NULL) {
    arts_coh_install_buffer(cache, /*new_version=*/1, /*data_payload=*/NULL,
                            db_size);
  }
}
