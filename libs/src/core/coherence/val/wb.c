/* SPDX-License-Identifier: Apache-2.0
 *
 * WB write-policy translation unit: defines the WB-specific
 * arts_handler_db_* / arts_db_* bodies directly (CMake links exactly this TU
 * for a VAL+WB build) plus the WB-only wire handlers/senders.
 * Compiled only for ARTS_COHERENCE_PROTOCOL=VAL with
 * ARTS_WRITE_POLICY=WB (selected in libs/src/core/CMakeLists.txt).
 * Contains NO protocol/write-policy preprocessor logic.
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/directory.h"
#include "arts/db.h"
#include "arts/edt.h"             /* arts_edt_dep_t (acquire body) */
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C self-send) */
#include "arts/ooo.h"             /* OOO_DB_* args (handler bodies) */
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"   /* arts_global_rank_id */
#include "arts/transport/net.h" /* arts_transport_send_async */
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h" /* arts_atomic_* */
#include "arts/utils/malloc.h"  /* arts_malloc / arts_free (transfer sender) */

/* ===== 8-case acquire dispatch (WB arm) ==========================
 * Whole arts_handler_db_acquire body for the WB build.  Diverges from WT
 * only on the RO-has-local-data predicate (WB: is_owner — the home rank does
 * NOT hold the canonical copy; only the current owner has an installed buffer,
 * so a home-but-not-owner rank goes through acquire_remote_ro and home forwards
 * to the owner via SNAPSHOT_REDIRECT). */
void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *dep = &((arts_edt_dep_t *)arts_get_depv(edt))[slot];
  arts_db_access_mode_t mode = dep->mode;
  /* A NON-ZERO writer_count means this rank holds the grant and therefore the
   * canonical buffer — an install the home has not confirmed yet included,
   * which ARTS_GRANT_UNCONFIRMED makes negative rather than zero.  Reading in
   * that window is safe precisely because no RW EDT can have run here: the
   * bytes are exactly the ones the transfer delivered.  A zero count is the
   * only state in which this rank's copy may be stale. */
  bool has_canonical_copy = (arts_atomic_read(&cache->writer_count) != 0u);

  if (mode == DB_MODE_RO) {
    if (has_canonical_copy) { /* WB RO predicate (only a grant holder has the
                                 canonical copy; home holds no bytes) */
      dep->ptr = arts_db_acquire_local(cache);
      arts_db_acquire_resolved(edt, slot);
      return;
    }
    arts_db_acquire_remote_ro(cache, edt->guid, slot); /* parks (SNAPSHOT) */
    return;
  }
  /* RW: the fast path's CAS is the whole test.  It takes a turn only against a
   * grant this rank holds AND the home has already confirmed, in one atom — an
   * install whose rw_holder flip is still unpublished reads back negative and
   * is refused, because writing now would be observable while the directory
   * names the previous owner (the stale-RO window).  Everything else parks;
   * grant_req_in_flight is held by the in-flight round, so acquire_remote_rw
   * parks without issuing a duplicate request, and CONFIRM_ACK drains it. */
  if (arts_db_acquire_rw_local_fast(cache, dep)) {
    arts_db_acquire_resolved(edt, slot); /* data here, writer_count bumped */
    return;
  }
  arts_db_acquire_remote_rw(cache, edt->guid,
                            slot); /* parks (GRANT_REQUEST) */
}

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW;
}

/* ===== release_rw (WB arm) =======================================
 * The WB write policy drops the buffer ref BEFORE decrementing writer_count, so
 * the slot's cache-hold is the only ref that can keep the buffer alive past
 * writer_count==0 (a concurrent teardown then frees it via the cb deleter with
 * no dangling local ref).  In the WT write policy this window does not exist
 * (local_transfer_now restores the sentinel); the WB write policy has no sentinel
 * restoration, so it must close the window by releasing the ref before exposing
 * writer_count==0. */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped ownership;
   * decrementing would underflow.  Atomic acquire-load avoids a TSan race. */
  if (arts_atomic_read(&cache->writer_count) == 0) {
    return;
  }
  /* Version bump on the current buffer (local ref scoped to release_rw). */
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
    /* Self-credit: this rank now holds the bumped version.  Keeps the
     * owner-side ledger's invariant ("floor of what each rank's cache holds")
     * true for the producer itself; the ship path re-credits before the map
     * migrates, so a NULL map (never-served owner) can stay NULL here. */
    if (cache->cached_version != NULL) {
      arts_rank_u64_map_advance(cache->cached_version, arts_global_rank_id,
                                arts_atomic_read_u64(&buf->version));
    }
  }
  /* Drop the buffer ref BEFORE the writer_count decrement (close the
   * writer_count==0 teardown window). */
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  /* writer_count is non-negative (post-install flip + install guard absorb any
   * INVALIDATE that lands during install).  A true 1->0 release reads 0 and
   * ships the transfer; the (int) cast is defensive. */
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1); /* post value */
  if (rest == 0) {
    /* If a GRANT_INVALIDATE already published a transfer target while writers
     * were live, this (last) releaser is the unique actor that ships
     * GRANT_RESPONSE — sentinel invariant, no flag.  Identical for home and
     * non-home owners.  Otherwise no transfer is pending: home retains
     * ownership until a future GRANT_REQUEST; a non-home owner quiesces. */
    if (cache->incoming_new_owner != ARTS_NO_PENDING_OWNER) {
      arts_db_send_grant_response(cache);
    }
  }
}

/* ===== cache_s lifecycle (WB: pending_rw + dedup map + sentinel) =
 * Construct: the WB write policy's field-init (the Vyukov MPSC pending_rw queue +
 * the owner-side dedup map [allocated on demand] + the transfer sentinel) runs
 * BEFORE arts_db_cache_common_init.  Destruct order: buffer-NULL (pre) →
 * pending_rw destroy → snapshot drain + home teardown (post). */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  arts_pending_rw_queue_init(&c->pending_rw);
  /* WB owner-side fields.  The dedup ledger is allocated HERE, once, rather
   * than on first use: its readers hold no reference to it, so a ledger that
   * could be created (or replaced) later would have to carry a lifetime
   * protocol to be read safely.  Allocated with the cache and freed with it,
   * a reader can only ever meet the live object. */
  c->cached_version = arts_rank_u64_map_create(arts_global_rank_count);
  c->incoming_new_owner = ARTS_NO_PENDING_OWNER;
  c->incoming_new_owner_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache); /* buffer-NULL FIRST */
  arts_pending_rw_queue_destroy(&cache->pending_rw);
  arts_rank_u64_map_destroy(cache->cached_version);
  cache->cached_version = NULL;
  arts_db_cache_common_destroy_post(cache); /* snapshot drain → home teardown */
}

/* ===== home-directory lifecycle (inlined in arts_db_s) ============= */

void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks) {
  atomic_store_explicit(&db->rw_holder, rw_holder, memory_order_relaxed);
  arts_home_grantreq_queue_init(&db->pending_rw);
  atomic_store_explicit(&db->invalidate_in_flight, 0, memory_order_relaxed);
  arts_rank_bitset_init(&db->cached_ranks, nranks);
  db->pending_install_owner = 0;
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_grantreq_queue_destroy(&db->pending_rw);
  arts_rank_bitset_destroy(&db->cached_ranks);
  /* No free: home fields are inlined in the arts_db_s. */
}

void arts_handler_db_snapshot_request(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_snapshot_request_s *a =
      (struct arts_ooo_args_db_snapshot_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_guid_t edt_guid = a->edt_guid;
  uint32_t slot = a->slot;

  /* WB home-side RO routing.
   *
   * Under the WB write policy home does not hold the canonical data copy — the
   * current owner does.  Home's job is to redirect the requester to the owner
   * (via SNAPSHOT_REDIRECT) so the owner can send SNAPSHOT_RESPONSE directly,
   * applying the owner-side cached_version dedup.
   *
   * Record the requester in the cached-ranks set, then redirect to the current
   * owner.  A destroyed DB is handled by the route_table lookup miss (slot
   * value NULL-stored before destroy) + the handler single-actor invariant —
   * no per-home destroy flag. */
  struct arts_db_s *db = arts_db_of_cache(cache);
  arts_rank_bitset_set(&db->cached_ranks, requester);
  /* Forward to the current owner unconditionally: even mid-transfer, rw_holder
   * still names the OLD owner, which retains its buffer + cached_version
   * permanently and serves the REDIRECT from its own copy.  Client-side
   * monotonic version compare keeps stale snapshots safe.  No defer queue. */
  if (a->rdzv.txid == 0 && cache->db_size > 0 && arts_global_rank_count > 1) {
    /* First-touch request without a landing: the requester did not know
     * db_size.  Home is the size authority even though the data lives with
     * the owner — answer a size-only CTS directly; the re-request carries a
     * landing and is then redirected for real. */
    arts_send_db_snapshot_response(requester, cache->db_guid, /*version=*/0,
                                   edt_guid, slot, /*kind=*/2, cache->db_size,
                                   &a->rdzv, NULL);
    return;
  }
  unsigned int owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  arts_send_db_snapshot_redirect(owner, cache->db_guid, requester, edt_guid,
                                 slot, &a->rdzv);
}

/* Fan-out callback for arts_rank_bitset_for_each during destroy.
 * ctx carries the db_guid encoded as uintptr_t (no heap allocation
 * needed since the callback is synchronous). */
static void owner_destroy_fanout_cb(unsigned int rank, void *ctx) {
  arts_guid_t db_guid = (arts_guid_t)(uintptr_t)ctx;
  unsigned int self = arts_global_rank_id;
  if (rank != self) {
    arts_send_db_cache_destroy(rank, db_guid);
  }
}

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_DESTROY]): the OoO engine has already
 * acquired the home db_s and pinned a ref across this call (cache is its FIRST
 * member).  Order: roster fan-out, then
 * arts_route_table_set_destroyed LAST (parked waiter at destroy = UB, cleaned
 * up by the refcount-0 destructor).  WB roster source = rw_holder
 * (current RW owner) + the RO cached-ranks bit-set + the queued ownership
 * requesters. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  /* WB: notify the current RW owner first (rw_holder, not the cached-ranks
   * bit-set), then the RO cached-ranks bit-set, then the queued requesters. */
  {
    unsigned int holder =
        atomic_load_explicit(&db->rw_holder, memory_order_acquire);
    if (holder != self) {
      arts_send_db_cache_destroy(holder, a->db_guid);
    }
  }
  arts_rank_bitset_for_each(&db->cached_ranks, owner_destroy_fanout_cb,
                            (void *)(uintptr_t)a->db_guid);
  {
    unsigned int q_rank;
    while (arts_home_grantreq_queue_pop(&db->pending_rw, &q_rank, NULL)) {
      if (q_rank != self) {
        arts_send_db_cache_destroy(q_rank, a->db_guid);
      }
    }
  }
  /* In-flight ownership transfer: the new owner C lives only in
   * pending_install_owner during [pop at round-start .. rw_holder flip] and is
   * in none of the rosters above. With the confirm gate it parks its RW waiter
   * until CONFIRM_ACK, so a destroy that races the transfer must wake it here
   * or it hangs. Notify it (dedup against rw_holder / self). */
  if (atomic_load_explicit(&db->invalidate_in_flight, memory_order_acquire) !=
      0u) {
    unsigned int in_flight = db->pending_install_owner;
    unsigned int holder =
        atomic_load_explicit(&db->rw_holder, memory_order_acquire);
    if (in_flight != self && in_flight != holder) {
      arts_send_db_cache_destroy(in_flight, a->db_guid);
    }
  }
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* Case-D leaf: OWNER publishes creator_rank as the home rw_holder (coalesce
 * path). */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

void arts_handler_db_snapshot_redirect(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_snapshot_redirect_args_s *a =
      (struct arts_db_snapshot_redirect_args_s *)args_v;
  unsigned int requester = a->requester_rank;
  arts_guid_t edt_guid = a->edt_guid;
  uint32_t slot = a->slot;

  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  if (buf == NULL) {
    /* A grant holder of a SIZED block always holds its buffer — every create
     * seeds the initial holder's, and a transfer either carries the payload or
     * hands the requester its own advertised landing as storage — so the only
     * block that reaches here is a zero-sized one, whose defined value is the
     * NULL pointer the requester's waiter then sees.  Respond version=0, no
     * data; the unused landing is echoed for recycling. */
    arts_send_db_snapshot_response(requester, a->db_guid, /*version=*/0,
                                   edt_guid, slot, /*kind=*/0, cache->db_size,
                                   &a->rdzv, NULL);
    return;
  }
  uint64_t cur_v = buf->version;
  uint64_t known_v =
      arts_rank_u64_map_get(cache->cached_version, requester);

  if (known_v >= cur_v) {
    /* Requester already holds this version — no payload moves; echo the
     * unused landing for recycling. */
    arts_db_buf_release(&buf_h);
    arts_send_db_snapshot_response(requester, a->db_guid, cur_v, edt_guid, slot,
                                   /*kind=*/0, cache->db_size, &a->rdzv, NULL);
  } else if (a->rdzv.txid == 0 && arts_global_rank_count > 1) {
    /* Data must move but no landing was forwarded (a first-touch request that
     * slipped past home's CTS gate can only mean home learned the size after
     * forwarding — defensive).  Size-only CTS; the re-request (via home)
     * carries a landing.  The watermark does NOT advance. */
    arts_db_buf_release(&buf_h);
    arts_send_db_snapshot_response(requester, a->db_guid, cur_v, edt_guid, slot,
                                   /*kind=*/2, cache->db_size, &a->rdzv, NULL);
  } else {
    /* Advance dedup watermark (monotonic max), then PUT the payload into the
     * forwarded landing (buf_h transfers into the sender). */
    arts_rank_u64_map_advance(cache->cached_version, requester, cur_v);
    arts_send_db_snapshot_response(requester, a->db_guid, cur_v, edt_guid, slot,
                                   /*kind=*/1, cache->db_size, &a->rdzv,
                                   buf_h);
  }
}

void arts_send_db_snapshot_redirect(unsigned int owner_rank,
                                    arts_guid_t db_guid,
                                    unsigned int requester_rank,
                                    arts_guid_t edt_guid, uint32_t slot,
                                    const struct arts_rdzv_landing_s *rdzv) {
  struct arts_rdzv_landing_s fwd =
      (rdzv != NULL) ? *rdzv : (struct arts_rdzv_landing_s){0, 0, 0, 0};
  struct arts_msg_snapshot_redirect_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_SNAPSHOT_REDIRECT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.edt_guid = edt_guid;
  p.requester_rank = requester_rank;
  p.slot = slot;
  p.rdzv.addr = fwd.addr;
  p.rdzv.key = fwd.key;
  p.rdzv.txid = fwd.txid;
  p.rdzv.cookie = fwd.cookie;
  if (owner_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire.  HIT
     * serves SNAPSHOT_RESPONSE from the ref-pinned owner-side db_s; MISS (DB
     * destroyed / not yet installed) sends DESTROY_NOTIFY to the requester so
     * its parked RO waiter wakes and observes DB_DESTROYED. */
    struct arts_db_snapshot_redirect_args_s args = {
        .db_guid = db_guid,
        .edt_guid = edt_guid,
        .requester_rank = requester_rank,
        .slot = slot,
        .rdzv = fwd,
    };
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_snapshot_redirect(db, &args);
    } else {
      arts_send_db_cache_destroy(requester_rank, db_guid);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

/* Case-D leaf: the WB write policy defers the home-buffer install to the
 * creator's first release_rw (PUBLISH / GRANT path); nothing to do at
 * create. */
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  (void)cache;
  (void)db_size;
}

/* Readers re-check a version at every acquire, so an ex-holder's retained
 * buffer can never be mistaken for current: nothing to register. */
void arts_db_grant_note_ex_holder(struct arts_db_s *db, unsigned int rank) {
  (void)db;
  (void)rank;
}
