/* SPDX-License-Identifier: Apache-2.0
 *
 * EAGER timing translation unit for MRSW: defines the EAGER-specific
 * arts_handler_db_* / arts_db_* bodies directly (CMake links exactly this TU
 * for an MRSW+EAGER build) plus the EAGER-only wire handlers/senders.
 * Compiled only for ARTS_COHERENCE_PROTOCOL=MRSW with
 * ARTS_PROTOCOL_TIMING=EAGER (selected in libs/src/core/CMakeLists.txt).
 * Contains NO protocol/timing preprocessor logic.
 *
 * Mirrors coherence/mrnew/eager.c; the MRSW deltas are the single-writer cap:
 * the local RW acquire claims the TOKEN (arts_db_acquire_rw_local_fast pushes a
 * waiter + returns without writing dep->ptr — the run path delivers it), the
 * GRANT install jumps writer_count 0->2 (sentinel + token, no guard removal —
 * the token is the running writer's account, relinquished only by
 * release_rw_local on an empty pop), and release routes through
 * arts_db_release_rw_local (pop-then-conditional-sub).
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
#include "arts/edt.h"             /* arts_edt_dep_t (acquire body) */
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (pin db_s) */
#include "arts/ooo.h"             /* OOO_DB_* args (handler bodies) */
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/threads.h"     /* arts_global_rank_id */
#include "arts/transport/outbox.h"   /* arts_transport_send_async */
#include "arts/transport/protocol.h" /* arts_fill_packet_header, MSG_* */
#include "arts/utils/atomics.h"      /* arts_atomic_* */

/* ===== 8-case acquire dispatch (EAGER arm) ==========================
 * Whole arts_handler_db_acquire body for the EAGER build.  Diverges from LAZY
 * only on the RO-has-local-data predicate (EAGER: is_home||is_owner — home
 * always holds current data via synchronous WRITEBACK).  The remote-RO,
 * RW-local-fast, and remote-RW paths are the shared helpers
 * (coherence/coherence.c / coherence/mrsw/ownership.c).  On a successful local
 * RW fast path the run path (token claim / drain / release hand-off) delivers
 * the dep — the handler does NOT call arts_db_acquire_resolved. */
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
  /* writer_count > 0 means owner ({2,1} = sentinel+token / sentinel-only); the
   * (int) cast is defensive.  An owned read here is safe to serve RO from the
   * local buffer, and in a non-invalidated epoch the count never reaches 0. */
  bool is_owner = ((int)arts_atomic_read(&cache->writer_count) > 0);

  if (mode == DB_MODE_RO) {
    if (is_home ||
        is_owner) { /* eager RO predicate (home holds current data) */
      dep->ptr = arts_db_acquire_local(cache);
      arts_db_acquire_resolved(edt, slot);
      return;
    }
    arts_db_acquire_remote_ro(cache, edt->guid, slot); /* parks (SNAPSHOT) */
    return;
  }
  /* RW */
  if (is_owner && arts_db_acquire_rw_local_fast(cache, dep, edt->guid, slot)) {
    /* The run path (idle-owner token claim / active-writer release / GRANT
     * drain) delivers this dep via mark_edt_secured + mark_edt_ready; do NOT
     * resolve here (would double-account). */
    return;
  }
  arts_db_acquire_remote_rw(cache, edt->guid,
                            slot); /* parks (OWNERSHIP_REQUEST) */
}

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW;
}

/* ===== release_rw (eager arm) ========================================
 * The eager protocol keeps home's RO copy fresh with a pure synchronous
 * WRITEBACK on every non-home release (home owners already hold the canonical
 * buffer).  The single-writer token release (pop-then-conditional-sub + LOCAL
 * Dekker re-check, and the 0-edge owner→owner ship) is in
 * arts_db_release_rw_local (coherence/mrsw/ownership.c). */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped ownership (e.g. a
   * cache already torn down by a destroy fan-out); the release token logic
   * would underflow. Atomic acquire-load avoids a TSan race. */
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
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  /* EAGER keeps home's RO copy fresh: pure synchronous writeback every release
   * (a non-home owner; home owners already hold the canonical buffer). */
  if (!is_home && buf != NULL) {
    sem_t cv;
    sem_init(&cv, 0, 0);
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_writeback(home_rank, cache->db_guid, new_version,
                           (uint64_t)(uintptr_t)&cv, buf->data, cache->db_size);
    await_writeback_ack(&cv);
    sem_destroy(&cv);
  }
  /* Eager: release buffer ref after the writeback (which reads buf->data). */
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  /* Single-writer token release: hand the token to the next waiter (pop) or
   * withdraw it (sub) with a LOCAL Dekker re-check, shipping the owner→owner
   * transfer only on the true 0-edge with a transfer target pending. */
  arts_db_release_rw_local(cache);
}

/* ===== cache_s lifecycle (eager: pending_rw Vyukov MPSC FIFO) ========
 * Construct: the eager protocol's field-init (the per-cache RW-waiter FIFO —
 * cannot be zero-initialized, head/tail must point at the embedded stub) runs
 * BEFORE arts_db_cache_common_init so the queue is wired before any push could
 * land.  Destruct order: buffer-NULL (pre) → pending_rw destroy → snapshot
 * drain + home teardown (post). */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  arts_db_rw_waiter_queue_init(&c->pending_rw);
  /* Transfer target for the owner→owner ship, published by each round's
   * INVALIDATE before it withdraws the sentinel.  Start at the sentinel (no
   * transfer pending). */
  c->incoming_new_owner = ARTS_LAZY_NO_PENDING_OWNER;
  /* EAGER has no owner-side dedup map (home serves RO via GET_DATA): the
   * owner→owner transfer always ships an empty map.  NULL so the shared ship
   * helper's map-build gate takes its empty-map branch. */
  c->last_sent_version = NULL;
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache); /* buffer-NULL FIRST */
  /* Refcount hit 0 → this destructor is the SOLE owner of the cache: no other
   * ref-holder exists, so it is the unique safe single consumer of the pop-one
   * pending_rw FIFO.  Wake every still-parked RW waiter with NULL data (the
   * buffer slot was NULLed by destroy_pre, so mark_edt_ready_by_guid delivers
   * depv[slot].ptr=NULL and accounts the dep) BEFORE freeing the queue nodes.
   * This is where the destroy fan-out's deferred RW wake lands (the handler no
   * longer drains pending_rw; the releasing token holder relinquished without
   * popping).  Snapshot waiters are woken too (arts_db_drain_pending_snapshot
   * wakes; the common-post path only frees), then the queue is torn down.
   *
   * Skip the wake during final runtime teardown (shutdown_state != 0): the
   * worker scheduler is gone by the time arts_clean_up_dbs frees the route
   * table, so accounting a dep / scheduling a parked EDT would dereference a
   * destroyed deque.  At shutdown the parked EDTs are abandoned with the rest
   * of the graph — only the FIFO nodes still need freeing. */
  if (arts_node_info.shutdown_state == 0) {
    arts_guid_t edt_guid;
    unsigned int slot;
    while (arts_db_rw_waiter_queue_pop(&cache->pending_rw, &edt_guid, &slot)) {
      mark_edt_ready_by_guid(edt_guid, slot);
    }
    arts_db_drain_pending_snapshot(cache); /* wake parked snapshot waiters */
  }
  arts_db_rw_waiter_queue_destroy(&cache->pending_rw);
  arts_db_cache_common_destroy_post(cache); /* snapshot free → home teardown */
}

/* ===== home-directory lifecycle (inlined in arts_db_s) ============= */

void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks) {
  atomic_store_explicit(&db->rw_holder, rw_holder, memory_order_relaxed);
  arts_home_lockreq_queue_init(&db->pending_rw);
  atomic_store_explicit(&db->invalidate_in_flight, 0, memory_order_relaxed);
  db->last_sent_version = arts_rank_u64_map_create(nranks);
  db->pending_install_owner = 0;
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

static void update_last_sent_max(struct arts_db_cache_s *cache,
                                 unsigned int requester, uint64_t master_v,
                                 const void *data, uint64_t data_size,
                                 arts_guid_t edt_guid, uint32_t slot) {
  struct arts_db_s *db = arts_db_of_cache(cache);
  /* Monotonic dedup — if the requester already received this version
   * (cur >= master_v), send NO_DATA. */
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

/* ===== Per-protocol wire-handler bodies =============================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_SNAPSHOT_REQUEST]): the OoO engine
 * has already acquired the home db_s for db_guid and pinned a ref across this
 * call (cache is its FIRST member).  The eager protocol serves from home's
 * canonical buffer with last_sent_version dedup. */
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
    /* Sentinel DB (db_size==0) or HOME_RECV pre-WRITEBACK: respond version=0,
     * NULL data (per spec "value is undefined" before any writer publishes). */
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
 * back from (char *)a + sizeof(*a). */
void arts_handler_db_writeback(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_writeback_s *a =
      (struct arts_ooo_args_db_writeback_s *)args_v;
  const void *data =
      a->data_size > 0 ? (const void *)((char *)a + sizeof(*a)) : NULL;

  /* Monotonic: buf_install ignores a stale (lower/equal version) writeback. */
  arts_db_buf_install(cache, a->version, data, a->data_size);
  if (a->cv != 0) {
    arts_send_db_writeback_ack(a->releaser, a->db_guid, a->cv);
  }
}

/* Cat-C pure body (WRITEBACK_ACK).  Cache-independent pointer-identity sem-post
 * on a->cv (the releaser's stack-local sem_t, valid on this rank); item_v is
 * unused.  The dispatcher posts on BOTH a HIT (this body) and a MISS so a
 * torn-down home cache never strands the blocked releaser. */
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
 * member).  Order: remote DESTROY_NOTIFY roster fan-out FIRST (the cache stays
 * alive — only the install ref is dropped), then arts_route_table_set_destroyed
 * LAST.  Eager roster source = home->last_sent_version + the queued ownership
 * requesters.
 *
 * MRSW does NOT drain cache.pending_rw here (it is a pop-one FIFO a token
 * holder can be releasing concurrently — a second consumer would corrupt the
 * chain). The parked RW + snapshot waiters are woken by the refcount-0 cache
 * destructor (arts_db_cache_destructor), the SOLE owner once set_destroyed
 * drops the last install ref; a token holder racing its release defers to the
 * destructor via the arts_route_table_was_destroyed guard in
 * arts_db_release_rw_local. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
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
  /* Do NOT drain cache.pending_rw / pending_snapshot here: a token holder can
   * be releasing pending_rw concurrently, and that pop-one FIFO must stay
   * single-consumer.  set_destroyed drops the last install ref → the refcount-0
   * destructor (arts_db_cache_destructor) is the sole owner and wakes every
   * parked waiter. */
  (void)cache;
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* Case-D leaf: eager publishes creator_rank as the home rw_holder. */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

/* ===== Ownership-round seams (called from coherence/mrsw/ownership.c) == */

void arts_db_start_ownership_round(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester) {
  (void)requester;
  /* Pop the OLDEST requester (FIFO) to be the transfer target, publish it as
   * pending_install_owner, and INVALIDATE the current holder carrying the new
   * owner so the holder ships the owner→owner OWNERSHIP_RESPONSE directly.
   * pending_install_owner is written only by the baton holder (single writer),
   * so no atomic. */
  unsigned int next_owner;
  if (!arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
    /* Defensive: we just pushed, so empty is impossible under correct usage. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    return;
  }
  db->pending_install_owner = next_owner;
  unsigned int current_owner =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  arts_send_db_ownership_invalidate(current_owner, cache->db_guid, next_owner);
}

/* ===== Eager OWNERSHIP_RESPONSE handler (new owner C) ================
 * EAGER drains pending_rw + runs its one RW EDT IMMEDIATELY here (no
 * CONFIRM_ACK gate — home serves RO, so there is no stale-RO window to close),
 * then sends CONFIRM to home (flips rw_holder + advances the next round).
 *
 * MRSW delta: install jumps writer_count 0->2 (sentinel + TOKEN).  Unlike
 * MRNEW there is NO guard-removal -1 at the end: the +1 above the sentinel is
 * the running writer's token, relinquished only by arts_db_release_rw_local on
 * an empty pop.  This is safe because no INVALIDATE can land during the
 * install: home flips rw_holder to this rank only at the post-install CONFIRM,
 * so home does not target this rank with an INVALIDATE until after CONFIRM is
 * processed — there is no concurrent decrement to drive the count below the
 * token floor during the install body. */
void arts_handler_db_ownership_response(void *payload, size_t size) {
  struct arts_msg_ownership_response_packet_s *hdr =
      (struct arts_msg_ownership_response_packet_s *)payload;
  arts_guid_t db_guid = hdr->db_guid;

  arts_shared_ptr_t db_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    db_h = arts_db_cache_lazy_install(db_guid, /*db_size=*/0);
    db = (struct arts_db_s *)arts_shared_get(db_h);
    if (db == NULL) {
      arts_shared_release(&db_h);
      return; /* DB destroyed before we became owner — drop. */
    }
  }
  struct arts_db_cache_s *cache = &db->cache;

  /* Wire layout: header | map (count pairs) | data bytes.  EAGER ignores the
   * map (it dedups RO via home->last_sent_version), but parses past it. */
  char *map_start = (char *)payload + sizeof(*hdr);
  size_t map_size =
      (sizeof(uint32_t) * 2) + ((size_t)hdr->map_entry_count *
                                sizeof(struct arts_msg_rank_version_pair_s));
  char *data_start = map_start + map_size;
  size_t data_size = size - sizeof(*hdr) - map_size;

  if (data_size > 0) {
    arts_db_buf_install(cache, hdr->version, data_start, data_size);
    if (cache->db_size == 0) {
      cache->db_size = data_size;
    }
  }

  /* Sentinel(+1) + token(+1), single op (0->2): the token is the running
   * writer's account.  No INVALIDATE can land before CONFIRM advances
   * rw_holder, so the count holds >= 2 across this install body. */
  arts_atomic_add(&cache->writer_count, 2u);
  cache->ownership_req_in_flight = 0;

  /* EAGER drains + runs NOW (no confirm-ack gate): home serves RO so there is
   * no stale-RO window.  Pop exactly ONE waiter (the token's writer).  Then
   * tell home we installed (CONFIRM), which flips rw_holder + advances the next
   * round. */
  arts_db_drain_pending_rw_after_grant(cache, hdr->version, /*has_next=*/false);
  arts_db_drain_pending_snapshot(cache);
  arts_ooo_drain_guid(db_guid);

  unsigned int home_rank = arts_guid_get_rank(db_guid);
  arts_send_db_ownership_confirm(home_rank, db_guid, hdr->version);

  /* No guard removal: the token stays until release_rw_local on an empty pop
   * (which is also the relocated 0-edge ship-check). */
  arts_shared_release(&db_h);
}

/* ===== Eager CONFIRM handler (home A) =============================== */

/* Cat-C pure body (CONFIRM, home side).  args_v is unused — the new owner is
 * read from db->pending_install_owner (published by the baton holder).  EAGER
 * does NOT send CONFIRM_ACK (the new owner already drained + ran at
 * OWNERSHIP_RESPONSE); it only flips rw_holder + advances the next round. */
void arts_handler_db_ownership_confirm(void *item_v, void *args_v) {
  (void)args_v;
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_s *db = arts_db_of_cache(cache);

  unsigned int new_owner = db->pending_install_owner;
  atomic_store_explicit(&db->rw_holder, new_owner, memory_order_release);

  /* Drain-or-release retry loop: start the next transfer round if there are
   * pending_rw requests, otherwise release the baton. */
  while (1) {
    unsigned int next_owner;
    if (arts_home_lockreq_queue_pop(&db->pending_rw, &next_owner)) {
      db->pending_install_owner = next_owner;
      unsigned int current =
          atomic_load_explicit(&db->rw_holder, memory_order_acquire);
      arts_send_db_ownership_invalidate(current, cache->db_guid, next_owner);
      return;
    }
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    if (arts_home_lockreq_queue_empty(&db->pending_rw)) {
      return;
    }
    unsigned int expected = 0u;
    if (!atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      return;
    }
  }
}

/* ===== Eager INVALIDATE_NOTICE handler (pure body) =================== */

/* Pure (cache, args) body.  The wire dispatcher / self-send shortcut has
 * already looked the cache up (the target is rw_holder, published only at the
 * post-install CONFIRM owner-swap, so the cache is provably installed when
 * INVALIDATE arrives) and passes the db_s as item_v — cache is its FIRST member
 * (offset 0).
 *
 * MRSW: the count here is sentinel + token (or sentinel-only if the writer has
 * already quiesced).  The sub withdraws the SENTINEL; the 0-edge fires only
 * when no token remains (a running writer holds the +1), so transfer-gating is
 * automatic — the last release_rw_local ships instead. */
void arts_handler_db_ownership_invalidate(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_ownership_invalidate_s *a =
      (struct arts_ooo_args_db_ownership_invalidate_s *)args_v;
  /* Publish the transfer target BEFORE withdrawing the sentinel: whichever
   * actor drives writer_count to 0 (this handler, or a concurrent last
   * release) then reads the same new_owner and ships the owner→owner transfer.
   */
  cache->incoming_new_owner = a->new_owner_rank;
  int rest = (int)arts_atomic_sub(&cache->writer_count, 1);
  if (rest == 0) {
    /* The decrement that drives writer_count to exactly 0 (no token remaining)
     * is the unique actor that ships the owner→owner transfer. */
    arts_db_send_ownership_response(cache);
  }
}

/* Case-D leaf: the eager protocol defers the home-buffer install to the
 * creator's first release_rw (WRITEBACK / GRANT path); nothing to do at
 * create. */
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  (void)cache;
  (void)db_size;
}
