/* SPDX-License-Identifier: Apache-2.0
 *
 * LRC (Lazy Release Consistency) coherence-model hook implementations plus
 * LRC-only wire handlers/senders. Compiled only when ARTS_MEMORY_MODEL=LRC
 * (selected in libs/src/core/CMakeLists.txt). Contains NO ARTS_MEMORY_MODEL_*
 * preprocessor logic.
 */
#include <stdint.h>
#include <string.h>

#include "arts/db.h"
#include "arts/db_coherence_home.h"
#include "arts/db_coherence_model.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"   /* arts_global_rank_id */
#include "arts/transport/outbox.h" /* arts_remote_send_request_async */
#include "arts/transport/protocol.h"
#include "arts/utils/malloc.h" /* arts_malloc / arts_free (transfer sender) */

/* LRC transfers ownership owner->owner via
 * arts_coh_lrc_send_ownership_response, not a home-side GRANT, so there is no
 * local chain-advance.  invalidate_transfer still calls this on the home path;
 * it is a no-op stub for LRC. */
void arts_coh_local_transfer_now(struct arts_db_cache_s *cache) { (void)cache; }

/* LRC: the home rank does not hold the canonical data copy; only the current
 * owner (writer_count > 0) has an installed buffer.  A home-but-not-owner rank
 * has cache->buffer == NULL until TRANSFER_OWNERSHIP arrives, so acquire_local
 * would deliver NULL to the EDT.  Go through acquire_remote_ro so that home
 * forwards the request to the owner via REDIRECT_RO and the owner sends the
 * buffer back via DATA_RESPONSE. */
bool arts_coh_model_ro_has_local_data(bool is_home, bool is_owner) {
  (void)is_home;
  return is_owner;
}

/* LRC: drop the buffer ref BEFORE decrementing writer_count, so the slot's
 * cache-hold is the only ref that can keep the buffer alive past
 * writer_count==0 (a concurrent teardown then frees it via the cb deleter with
 * no dangling local ref).
 *
 * Invariant: when writer_count reaches 0, no thread may hold an outstanding
 * buffer ref acquired in this call, because a concurrent deferred-free teardown
 * (triggered once refs drain) will free cache->buffer_pool.  Any subsequent
 * release_buf write to that pool would corrupt freed memory.
 *
 * In RC this window does not exist because local_transfer_now restores the
 * sentinel (writer_count = 1) when no pending waiter is queued, keeping
 * writer_count above 0 until the next proper acquire.  LRC has no such sentinel
 * restoration, so we must close the window here by releasing the ref before
 * exposing writer_count == 0. */
void arts_coh_model_release_rw_pre_decrement(struct arts_db_cache_s *cache,
                                             arts_shared_ptr_t *buf_h,
                                             struct arts_db_buffer_s **buf) {
  (void)cache;
  if (*buf != NULL) {
    arts_coh_release_buf(buf_h);
    *buf = NULL;
  }
}

void arts_coh_model_release_rw_tail(struct arts_db_cache_s *cache,
                                    arts_shared_ptr_t *buf_h,
                                    struct arts_db_buffer_s *buf,
                                    uint64_t new_version, unsigned int rest,
                                    bool is_home) {
  (void)buf_h;
  (void)buf; /* buf was dropped before the writer_count decrement. */
  (void)new_version;
  (void)is_home; /* home vs non-home no longer branch here. */
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
}

/* ===== cache_s lifecycle (model field init/teardown) =============== */

void arts_coh_model_init_cache_s(struct arts_db_cache_s *c) {
  arts_pending_rw_queue_init(&c->pending_rw);
  /* LRC owner-side fields: dedup map allocated lazily on first ownership grant;
   * incoming_new_owner starts at the sentinel (no transfer pending). */
  c->last_sent_version = NULL;
  c->incoming_new_owner = ARTS_LRC_NO_PENDING_OWNER;
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
  arts_rank_bitset_init(&db->cached_ranks, nranks);
  db->pending_install_owner = 0;
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_lockreq_queue_destroy(&db->pending_rw);
  arts_rank_bitset_destroy(&db->cached_ranks);
  /* No free: home fields are inlined in the arts_db_s. */
}

/* ===== last_sent_version map serialization (LRC ownership response) = */

size_t arts_rank_u64_map_serialize(const struct arts_rank_to_u64_map_s *m,
                                   void *out) {
  uint32_t *count_field = (uint32_t *)out;
  struct arts_remote_rank_version_pair_s *entries =
      (struct arts_remote_rank_version_pair_s *)((char *)out +
                                                 sizeof(uint32_t) * 2);
  uint32_t n = 0;
  for (unsigned int r = 0; r < m->nranks; r++) {
    uint64_t v = atomic_load_explicit(&m->slots[r], memory_order_acquire);
    if (v == 0) {
      continue;
    }
    entries[n].rank = (uint32_t)r;
    entries[n].pad = 0;
    entries[n].version = v;
    n++;
  }
  count_field[0] = n;
  count_field[1] = 0; /* alignment pad */
  return sizeof(uint32_t) * 2 + (size_t)n * sizeof(*entries);
}

struct arts_rank_to_u64_map_s *
arts_rank_u64_map_deserialize(const void *in, size_t size,
                              unsigned int nranks) {
  (void)size; /* used by debug assertions; production ignores it */
  struct arts_rank_to_u64_map_s *m = arts_rank_u64_map_create(nranks);
  const uint32_t *count_field = (const uint32_t *)in;
  uint32_t n = count_field[0];
  const struct arts_remote_rank_version_pair_s *entries =
      (const struct arts_remote_rank_version_pair_s *)((const char *)in +
                                                       sizeof(uint32_t) * 2);
  for (uint32_t i = 0; i < n; i++) {
    arts_rank_u64_map_set(m, (unsigned int)entries[i].rank, entries[i].version);
  }
  return m;
}

/* ===== LRC ownership-transfer wire handlers (moved from handlers.c) === */

/* Forward declaration for the snapshot drain helper defined in
 * db_coherence_handlers.c, called from the LRC handlers below.
 * (arts_coh_drain_pending_rw_after_grant is declared in
 * db_coherence_model.h.) */
void arts_coh_drain_pending_snapshot(struct arts_db_cache_s *cache);

/* ===== LRC ship_transfer / start_invalidate_round ==================== */

void arts_coh_lrc_send_ownership_response(struct arts_db_cache_s *cache) {
  unsigned int new_owner = cache->incoming_new_owner;
  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf == NULL) {
    /* Sentinel DB (db_size==0) or pre-publication: send an empty transfer.
     * Still emit the 8-byte map count-header (count=0, pad=0): the receiver
     * unconditionally reconstructs map_size >= 8, so omitting it would
     * underflow data_size to (size_t)-8 and corrupt the install. */
    uint32_t empty_map[2] = {0u, 0u};
    arts_send_db_ownership_response(new_owner, cache->db_guid,
                                    /*version=*/0, empty_map, sizeof(empty_map),
                                    /*data=*/NULL, /*data_size=*/0);
    /* Re-arm the sentinel: a future INVALIDATE round publishes a fresh target.
     */
    cache->incoming_new_owner = ARTS_LRC_NO_PENDING_OWNER;
    return;
  }

  /* Serialize the owner-side dedup map for transfer. */
  size_t map_max =
      sizeof(uint32_t) * 2 + (size_t)arts_global_rank_count *
                                 sizeof(struct arts_remote_rank_version_pair_s);
  void *map_buf = arts_malloc(map_max);
  size_t map_size;
  if (cache->last_sent_version != NULL) {
    map_size = arts_rank_u64_map_serialize(cache->last_sent_version, map_buf);
  } else {
    /* No map yet: emit an empty map (count=0, pad=0). */
    uint32_t *p = (uint32_t *)map_buf;
    p[0] = 0u;
    p[1] = 0u;
    map_size = sizeof(uint32_t) * 2;
  }

  arts_send_db_ownership_response(new_owner, cache->db_guid, buf->version,
                                  map_buf, map_size, buf->data, cache->db_size);
  arts_free(map_buf);

  /* Release the local buffer ref taken above.  The slot's cache-hold ref keeps
   * the buffer alive: the old RW owner retains its buffer + last_sent_version
   * map permanently (until DB destroy), so in-flight RO REDIRECTs that still
   * name this rank as owner are served from its own copy. */
  arts_coh_release_buf(&buf_h);
  /* Re-arm the sentinel: this owner has fully shipped; a future INVALIDATE
   * round will publish a fresh target. */
  cache->incoming_new_owner = ARTS_LRC_NO_PENDING_OWNER;
}

void arts_coh_lrc_start_invalidate_round(struct arts_db_cache_s *cache,
                                         unsigned int new_owner) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  unsigned int current_owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  arts_send_db_ownership_invalidate(current_owner, cache->db_guid, new_owner);
}

/* ===== LRC TRANSFER_OWNERSHIP handler (new owner C) =================== */

void arts_handler_db_ownership_response(void *payload, size_t size) {
  struct arts_remote_ownership_response_packet_s *hdr =
      (struct arts_remote_ownership_response_packet_s *)payload;
  arts_guid_t db_guid = hdr->db_guid;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(db_guid);
  if (cache == NULL) {
    cache = arts_coh_lazy_install_cache_s(db_guid, /*db_size=*/0);
    if (cache == NULL) {
      return; /* DB destroyed before we became owner — drop. */
    }
  }

  /* Wire layout: header | map (count pairs) | data bytes */
  char *map_start = (char *)payload + sizeof(*hdr);
  size_t map_size =
      sizeof(uint32_t) * 2 + (size_t)hdr->map_entry_count *
                                 sizeof(struct arts_remote_rank_version_pair_s);
  char *data_start = map_start + map_size;
  size_t data_size = size - sizeof(*hdr) - map_size;

  /* Reconstruct the owner-side dedup map so this rank can skip
   * redundant DATA_RESPONSE sends to readers that already hold a
   * sufficiently fresh copy. */
  if (cache->last_sent_version != NULL) {
    arts_rank_u64_map_destroy(cache->last_sent_version);
  }
  cache->last_sent_version = arts_rank_u64_map_deserialize(
      map_start, map_size, arts_global_rank_count);

  /* Install the transferred buffer. */
  if (data_size > 0) {
    arts_coh_install_buffer(cache, hdr->version, data_start, data_size);
    if (cache->db_size == 0) {
      cache->db_size = data_size;
    }
  }

  /* Install ownership sentinel: writer_count = 1. */
  arts_atomic_swap(&cache->writer_count, 1u);
  /* Clear ownership_req_in_flight so subsequent RW acquires can kick new
   * LOCK_REQ rounds if needed. */
  cache->ownership_req_in_flight = 0;

  /* Drain local RW waiters + any case-3 snapshot reorder-buffer waiters that
   * this install now satisfies. */
  arts_coh_drain_pending_rw_after_grant(cache, hdr->version,
                                        /*has_next=*/false);
  arts_coh_drain_pending_snapshot(cache);

  /* Confirm installation to home so home can update rw_holder. */
  unsigned int home_rank = (unsigned int)arts_guid_get_rank(db_guid);
  arts_send_db_ownership_response_ack(home_rank, db_guid, hdr->version);
}

/* ===== LRC INSTALL_ACK handler (home A) =============================== */

void arts_handler_db_ownership_response_ack(
    struct arts_remote_install_ack_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return;
  }
  struct arts_db_s *db = arts_db_of_cache(cache);

  /* Publish the new rw_holder (visible to GET_DATA redirect path). */
  unsigned int new_owner = db->pending_install_owner;
  atomic_store_explicit(&db->rw_holder, new_owner, memory_order_release);

  /* Drain-or-release retry loop: try to start the next transfer round
   * if there are pending_rw requests, otherwise release the baton. */
  while (1) {
    unsigned int next_owner;
    if (arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
      db->pending_install_owner = next_owner;
      arts_coh_lrc_start_invalidate_round(cache, next_owner);
      return;
    }
    /* No pending requester — release the baton. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    /* Re-check for a freshly-enqueued requester that raced the baton
     * release.  If the queue is still empty, we're done. */
    if (arts_home_lockreq_queue_empty(&db->pending_rw)) {
      return;
    }
    /* There is a new requester; try to re-acquire the baton. */
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      /* Another LOCK_REQ handler already picked up the baton (race);
       * that thread will drain the queue. */
      return;
    }
    /* Re-acquired the baton; loop to pop and start the next round. */
  }
}

/* ===== Per-model wire-handler body hooks =========================== */

void arts_coh_model_snapshot_request_serve(struct arts_db_cache_s *cache,
                                           unsigned int requester,
                                           arts_guid_t edt_guid,
                                           uint32_t slot) {
  /* LRC home-side RO routing.
   *
   * Home does not hold the canonical data copy under LRC — the current
   * owner does.  Home's job is to redirect the requester to the owner
   * (via REDIRECT_RO) so the owner can send DATA_RESPONSE directly,
   * applying the owner-side last_sent_version dedup.
   *
   * Record the requester in the cached-ranks set, then redirect to the current
   * owner.  A destroyed DB is handled by the route_table lookup miss (slot
   * value NULL-stored before destroy) + the handler single-actor invariant —
   * no per-home destroy flag. */
  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_rank_bitset_set(&db->cached_ranks, requester);
  /* Forward to the current owner unconditionally: even mid-transfer, rw_holder
   * still names the OLD owner, which retains its buffer + last_sent_version
   * permanently and serves the REDIRECT from its own copy.  Client-side
   * monotonic version compare keeps stale snapshots safe.  No defer queue. */
  unsigned int owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  arts_send_db_snapshot_redirect(owner, cache->db_guid, requester, edt_guid,
                                 slot);
}

void arts_coh_model_writeback_transfer(struct arts_db_cache_s *cache) {
  /* LRC has no synchronous writeback (the dispatcher fatals on
   * MSG_DB_WRITEBACK), so the WB_AND_TRANSFER tail is unreachable in LRC;
   * ownership moves owner→owner via arts_coh_lrc_send_ownership_response. */
  (void)cache;
}

void arts_coh_model_writeback_ack(uint64_t cv) {
  /* LRC does not use WRITEBACK_ACK; this message type is not sent in LRC
   * builds.  Drop silently. */
  (void)cv;
}

/* Fan-out callback for arts_rank_bitset_for_each during destroy.
 * ctx carries the db_guid encoded as uintptr_t (no heap allocation
 * needed since the callback is synchronous). */
static void lrc_destroy_fanout_cb(unsigned int rank, void *ctx) {
  arts_guid_t db_guid = (arts_guid_t)(uintptr_t)ctx;
  unsigned int self = arts_global_rank_id;
  if (rank != self) {
    arts_send_db_cache_destroy(rank, db_guid);
  }
}

void arts_coh_model_destroy_fanout(struct arts_db_cache_s *cache,
                                   unsigned int self) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_guid_t db_guid = cache->db_guid;
  /* LRC: notify the current RW owner first (tracked by rw_holder, not
   * the cached-ranks bit-set), then iterate the RO cached-ranks bit-set.  The
   * handler single-actor invariant keeps the readers roster stable across this
   * scan; a later LOCK_REQ/RO_REQ for a destroyed DB finds the slot absent
   * (lookup → NULL) and is a no-op. */
  {
    unsigned int holder =
        atomic_load_explicit(&db->rw_holder, memory_order_acquire);
    if (holder != self) {
      arts_send_db_cache_destroy(holder, db_guid);
    }
  }
  arts_rank_bitset_for_each(&db->cached_ranks, lrc_destroy_fanout_cb,
                            (void *)(uintptr_t)db_guid);
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
  /* LRC: pop the OLDEST requester (FIFO) to be the transfer target and
   * embed its rank in the INVALIDATE_NOTICE so the current holder ships
   * TRANSFER_OWNERSHIP directly, without a home round-trip.
   * pending_install_owner is only written by the baton holder (single
   * writer invariant), so no atomic needed. */
  unsigned int next_owner;
  if (!arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
    /* Defensive: we just pushed, so empty is impossible under correct
     * usage.  Release the baton and return. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    return;
  }
  db->pending_install_owner = next_owner;
  arts_coh_lrc_start_invalidate_round(cache, next_owner);
}

void arts_coh_model_ownership_return(struct arts_db_cache_s *cache) {
  /* LRC transfers ownership owner→owner via TRANSFER_OWNERSHIP and never sends
   * RELEASE_OWNERSHIP to home, so this is unreachable in LRC. */
  (void)cache;
}

/* ===== LRC INVALIDATE_NOTICE handler (moved from handlers.c) ======== */

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
   * LRC: publish the transfer target BEFORE withdrawing the sentinel.  This
   * ordering is the dedup (no separate transfer_pending flag): a concurrent
   * release_rw that observes rest==0 is guaranteed to see incoming_new_owner
   * already published, so exactly one of {this handler, the last releaser}
   * ships.  Only one INVALIDATE_NOTICE is in flight per round (home baton
   * gate), so there is no concurrent writer to incoming_new_owner. */
  cache->incoming_new_owner = p->new_owner_rank;
  unsigned int rest = arts_atomic_sub(&cache->writer_count, 1);
  if (rest > 0) {
    /* Local writers still active; the last release_rw will see rest==0 with
     * incoming_new_owner already published and ship TRANSFER_OWNERSHIP. */
    return;
  }
  /* rest == 0: we are the unique transfer actor. */
  arts_coh_lrc_send_ownership_response(cache);
}

/* ===== LRC REDIRECT_RO handler (owner side, moved from handlers.c) === */

void arts_handler_db_snapshot_redirect(
    struct arts_remote_snapshot_redirect_packet_s *p) {
  unsigned int requester = p->requester_rank;
  arts_guid_t edt_guid = p->edt_guid;
  uint32_t slot = p->slot;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    /* DB has been destroyed or not yet installed on this rank. */
    arts_send_db_cache_destroy(requester, p->db_guid);
    return;
  }

  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf == NULL) {
    /* No buffer installed yet (pre-publication or sentinel DB).
     * Respond with version=0, no data — requester's RO waiter fires
     * with undefined content (per spec). */
    arts_send_db_snapshot_response(requester, p->db_guid, /*version=*/0,
                                   edt_guid, slot, /*data=*/NULL,
                                   /*data_size=*/0);
    return;
  }
  /* Invariant: last_sent_version is created at ownership-install
   * (TRANSFER_OWNERSHIP, retained permanently thereafter).  The INITIAL
   * owner (the creator, which never received a transfer) has no map yet, so
   * lazily create it on its first served REDIRECT — otherwise the producer-on-
   * home + RO-consumers-elsewhere DAG would be served no-data (NULL/stale).
   * Single-actor here (the redirect handler runs on the network thread). */
  if (cache->last_sent_version == NULL) {
    cache->last_sent_version = arts_rank_u64_map_create(arts_global_rank_count);
  }

  uint64_t cur_v = (uint64_t)buf->version;
  uint64_t last_sent =
      arts_rank_u64_map_get(cache->last_sent_version, requester);

  if (last_sent >= cur_v) {
    /* Requester already holds this version — send no-data response. */
    arts_send_db_snapshot_response(requester, p->db_guid, cur_v, edt_guid, slot,
                                   NULL, 0);
  } else {
    /* Advance dedup watermark (monotonic max) then send data. */
    arts_rank_u64_map_advance(cache->last_sent_version, requester, cur_v);
    arts_send_db_snapshot_response(requester, p->db_guid, cur_v, edt_guid, slot,
                                   buf->data, cache->db_size);
  }
  arts_coh_release_buf(&buf_h);
}

/* ===== LRC wire senders (moved from db_coherence_senders.c) ========= */

/* LRC OWNERSHIP_RESPONSE = TRANSFER_OWNERSHIP: carries the serialized
 * last_sent_version map + buffer payload. */
void arts_send_db_ownership_response(unsigned int new_owner_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     const void *map_buf, size_t map_size,
                                     const void *data, size_t data_size) {
  struct arts_remote_ownership_response_packet_s hdr;
  uint32_t entry_count = (map_buf != NULL && map_size >= sizeof(uint32_t) * 2)
                             ? ((const uint32_t *)map_buf)[0]
                             : 0u;
  uint64_t total =
      (uint64_t)sizeof(hdr) + (uint64_t)map_size + (uint64_t)data_size;
  arts_fill_packet_header(&hdr.header, total, MSG_DB_OWNERSHIP_RESPONSE);
  hdr.header.rank = arts_global_rank_id;
  hdr.db_guid = db_guid;
  hdr.version = version;
  hdr.map_entry_count = entry_count;
  hdr.pad = 0;
  if (new_owner_rank == arts_global_rank_id) {
    /* Self-transfer: construct a contiguous buffer and call handler inline. */
    void *buf = arts_malloc((size_t)total);
    memcpy(buf, &hdr, sizeof(hdr));
    if (map_buf != NULL && map_size > 0) {
      memcpy((char *)buf + sizeof(hdr), map_buf, map_size);
    }
    if (data != NULL && data_size > 0) {
      memcpy((char *)buf + sizeof(hdr) + map_size, data, data_size);
    }
    arts_handler_db_ownership_response(buf, (size_t)total);
    arts_free(buf);
    return;
  }
  if ((map_size > 0 || data_size > 0) && (map_buf != NULL || data != NULL)) {
    size_t payload_size = map_size + data_size;
    void *payload = arts_malloc(payload_size);
    if (map_buf != NULL && map_size > 0) {
      memcpy(payload, map_buf, map_size);
    }
    if (data != NULL && data_size > 0) {
      memcpy((char *)payload + map_size, data, data_size);
    }
    arts_remote_send_request_payload_async_free(
        (int)new_owner_rank, (char *)&hdr, sizeof(hdr), (char *)payload,
        /*offset=*/0, (uint64_t)payload_size, arts_free);
  } else {
    arts_remote_send_request_async((int)new_owner_rank, (char *)&hdr,
                                   sizeof(hdr));
  }
}

void arts_send_db_ownership_response_ack(unsigned int home_rank,
                                         arts_guid_t db_guid,
                                         uint64_t version) {
  struct arts_remote_install_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_RESPONSE_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_ownership_response_ack(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_snapshot_redirect(unsigned int owner_rank,
                                    arts_guid_t db_guid,
                                    unsigned int requester_rank,
                                    arts_guid_t edt_guid, uint32_t slot) {
  struct arts_remote_snapshot_redirect_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_SNAPSHOT_REDIRECT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.edt_guid = edt_guid;
  p.requester_rank = requester_rank;
  p.slot = slot;
  if (owner_rank == arts_global_rank_id) {
    arts_handler_db_snapshot_redirect(&p);
    return;
  }
  arts_remote_send_request_async((int)owner_rank, (char *)&p, sizeof(p));
}

/* RC/LRC keep the lazy OCR home-buffer install (the creator's release_rw
 * WRITEBACK / GRANT path publishes the buffer); nothing to do at create. */
void arts_coh_model_create_home_buffer(struct arts_db_cache_s *cache,
                                       uint64_t db_size) {
  (void)cache;
  (void)db_size;
}
