/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol, EAGER timing: arbiters + acquire/release/handler/sender
 * bodies.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=MSI.
 */
#include "arts/coherence/msi/types.h"

#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/home.h"
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h"
#include "arts/memory/regpool.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/system/identity.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== pure arbiters (private copy — see arbiters.c) ==================== */
#include "arbiters.c"

/* ===== waiter pool (index-addressed chain nodes) ========================
 * Chain nodes live in per-DB chunked storage addressed by an 18-bit index
 * (0 = none) so a chain head fits the cache word.  Alloc = tagged free-list
 * pop (the tag defeats index ABA under concurrent alloc/free) with a
 * bump-allocator fallback; free = tagged push.  A node's fields are written
 * before the word CAS that links it and read only by the committer that
 * grabbed the chain. */

#define MSI_WAITER_NCHUNKS                                                     \
  ((uint32_t)(((uint64_t)MSI_WAITER_IDX_MAX + 1u) / MSI_WAITER_CHUNK_CAP))
#define MSI_FREE_IDX(h) ((uint32_t)((h) & MSI_CACHE_HEAD_MASK))
#define MSI_FREE_TAG(h) ((uint32_t)((h) >> MSI_CACHE_HEAD_BITS))
#define MSI_FREE_MAKE(tag, idx)                                                \
  ((uint32_t)((((tag) & 0x3FFFu) << MSI_CACHE_HEAD_BITS) |                     \
              ((idx) & MSI_CACHE_HEAD_MASK)))

struct msi_waiter_dir_s {
  _Atomic(struct arts_db_msi_waiter_s *) chunk[MSI_WAITER_NCHUNKS];
};

static struct msi_waiter_dir_s *msi_waiter_dir(struct arts_db_cache_s *c) {
  uintptr_t d = atomic_load_explicit(&c->waiters.chunks, memory_order_acquire);
  if (d == 0) {
    struct msi_waiter_dir_s *fresh =
        (struct msi_waiter_dir_s *)arts_calloc(1, sizeof(*fresh));
    uintptr_t expect = 0;
    if (atomic_compare_exchange_strong_explicit(
            &c->waiters.chunks, &expect, (uintptr_t)fresh,
            memory_order_acq_rel, memory_order_acquire)) {
      d = (uintptr_t)fresh;
    } else {
      arts_free(fresh);
      d = expect;
    }
  }
  return (struct msi_waiter_dir_s *)d;
}

static struct arts_db_msi_waiter_s *msi_waiter_ptr(struct arts_db_cache_s *c,
                                                   uint32_t idx) {
  struct msi_waiter_dir_s *dir = msi_waiter_dir(c);
  struct arts_db_msi_waiter_s *chunk = atomic_load_explicit(
      &dir->chunk[idx / MSI_WAITER_CHUNK_CAP], memory_order_acquire);
  return &chunk[idx % MSI_WAITER_CHUNK_CAP];
}

static uint32_t msi_waiter_alloc(struct arts_db_cache_s *c) {
  for (;;) {
    uint32_t h =
        atomic_load_explicit(&c->waiters.free_head, memory_order_acquire);
    uint32_t idx = MSI_FREE_IDX(h);
    if (idx == 0u) {
      break; /* free list empty: bump-allocate */
    }
    uint32_t next = msi_waiter_ptr(c, idx)->next;
    uint32_t nh = MSI_FREE_MAKE(MSI_FREE_TAG(h) + 1u, next);
    if (atomic_compare_exchange_weak_explicit(&c->waiters.free_head, &h, nh,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      return idx;
    }
  }
  uint32_t idx =
      atomic_fetch_add_explicit(&c->waiters.next_fresh, 1u, memory_order_acq_rel);
  if (idx > (uint32_t)MSI_WAITER_IDX_MAX) {
    (void)fprintf(stderr,
                  "arts msi: parked-waiter pool exhausted (index space)\n");
    abort();
  }
  struct msi_waiter_dir_s *dir = msi_waiter_dir(c);
  _Atomic(struct arts_db_msi_waiter_s *) *slot =
      &dir->chunk[idx / MSI_WAITER_CHUNK_CAP];
  struct arts_db_msi_waiter_s *chunk =
      atomic_load_explicit(slot, memory_order_acquire);
  if (chunk == NULL) {
    struct arts_db_msi_waiter_s *fresh = (struct arts_db_msi_waiter_s *)
        arts_calloc(MSI_WAITER_CHUNK_CAP, sizeof(*fresh));
    struct arts_db_msi_waiter_s *expect = NULL;
    if (!atomic_compare_exchange_strong_explicit(slot, &expect, fresh,
                                                 memory_order_acq_rel,
                                                 memory_order_acquire)) {
      arts_free(fresh);
    }
  }
  return idx;
}

static void msi_waiter_free(struct arts_db_cache_s *c, uint32_t idx) {
  struct arts_db_msi_waiter_s *node = msi_waiter_ptr(c, idx);
  for (;;) {
    uint32_t h =
        atomic_load_explicit(&c->waiters.free_head, memory_order_acquire);
    node->next = MSI_FREE_IDX(h);
    uint32_t nh = MSI_FREE_MAKE(MSI_FREE_TAG(h) + 1u, idx);
    if (atomic_compare_exchange_weak_explicit(&c->waiters.free_head, &h, nh,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      return;
    }
  }
}

/* ===== chain serve (the committer's continuation) =======================
 * Serve every node of a privately-owned chain.  The chain was grabbed in the
 * same CAS that published the install, so it holds exactly the fetch's
 * cohort — however late this loop runs, no other actor can reach these
 * nodes.  The secured signal (serialized-cursor advance) is reserved for
 * serialized deps: raising it for a non-serialized dep that happens to sit
 * at the cursor position falsely advances the cursor and enqueues a
 * duplicate resume, double-driving the next serialized dep (double writer
 * count, one release — the tenure never finals).  A non-serialized waiter
 * gets the data-arrival signal only. */
static void msi_serve_chain(struct arts_db_cache_s *cache, uint32_t head,
                            bool serialized) {
  while (head != 0u) {
    struct arts_db_msi_waiter_s *w = msi_waiter_ptr(cache, head);
    uint32_t next = w->next;
    arts_guid_t edt_guid = w->edt_guid;
    unsigned int slot = w->slot;
    msi_waiter_free(cache, head);
    if (serialized) {
      mark_edt_secured_by_guid(edt_guid, slot);
    }
    mark_edt_ready_by_guid(edt_guid, slot);
    head = next;
  }
}

/* ===== cache lifecycle ================================================== */

void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  uint64_t seed = 0ULL; /* both states IDLE, wc 0, both chain heads empty */
  if (kind == ARTS_DB_INIT_CREATOR_HOME ||
      kind == ARTS_DB_INIT_CREATOR_REMOTE) {
    /* Creator holds the write grant (create-default RW acquire): one writer
     * counted, and the granted copy is a valid read copy too. */
    seed = MSI_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_VALID, 1u, 0u, 0u);
  }
  atomic_store_explicit(&c->cache_state, seed, memory_order_relaxed);
  atomic_store_explicit(&c->waiters.chunks, (uintptr_t)0,
                        memory_order_relaxed);
  atomic_store_explicit(&c->waiters.next_fresh, 1u, memory_order_relaxed);
  atomic_store_explicit(&c->waiters.free_head, 0u, memory_order_relaxed);
  /* The home's create-installed zero buffer carries version 1; a creator
   * grant therefore numbers its releases from base 1 (first vnew = 2). */
  atomic_store_explicit(&c->wb_next, 2u, memory_order_relaxed);
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache);
  /* Parked chains live in the cache word; with no outstanding acquirers a
   * legit destroy has both heads 0 (fetch-open states only ever park), so
   * only the node-pool storage itself is torn down here. */
  arts_db_cache_common_destroy_post(cache);
}

void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  /* MSI tracks the tenure owner in dir_state (seeded by arts_db_home_init);
   * there is no separate holder field to publish at create. */
  (void)db;
  (void)creator_rank;
}

void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  /* Metadata-only home until the creator's first writeback: a creator-held
   * create has NO published state yet, and a read acquire that is
   * event-ordered after the creator's writes (write -> add_dependence ->
   * release) must observe the release-published bytes — a zero serve here
   * would be a serve-before-publication.  Pre-publication read serves hold
   * on pending_snapshot and the first install drains them.  (The NO_ACQUIRE
   * create path zero-installs in the shared create flow instead: with no
   * creator hold, creation itself is the publication.) */
  (void)cache;
  (void)db_size;
}

/* ===== acquire / release ================================================ */

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  /* A write acquire can wait on other tenures' releases (cross-DB cyclic
   * hazard: acquire-all must issue it in strict slot order).  A read acquire
   * waits only on the home's reply — never on another EDT's release — so it
   * carries no cycle risk and may issue in parallel. */
  return mode == DB_MODE_RW;
}

void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  arts_db_access_mode_t mode = depv[slot].mode;

  if (mode == DB_MODE_RO) {
    /* Home rank: the installed buffer IS the canonical copy — serve
     * directly (an overlap with an in-flight remote tenure is a race the
     * memory model already permits; the floor holds by home monotonicity).
     * Pre-publication (no buffer yet: a creator-held create whose first
     * writeback has not landed) holds instead — push-then-recheck against
     * the concurrent first install. */
    if (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id) {
      if (cache->db_size != 0) {
        arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
        bool published = (arts_shared_get(buf_h) != NULL);
        arts_db_buf_release(&buf_h);
        if (!published) {
          struct arts_db_snapshot_waiter_s *w =
              (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
          w->edt_guid = edt->guid;
          w->slot = slot;
          w->target_version = 0;
          w->serve = NULL; /* local waiter: the drain resumes the EDT */
          w->requester = arts_global_rank_id;
          w->rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
          arts_lf_stack_push(&cache->pending_snapshot, &w->link);
          buf_h = arts_db_buf_acquire(cache);
          published = (arts_shared_get(buf_h) != NULL);
          arts_db_buf_release(&buf_h);
          if (published) {
            arts_db_drain_pending_snapshot(cache);
          }
          return;
        }
      }
      mark_edt_ready_by_guid(edt->guid, slot);
      return;
    }
    /* Valid-copy fast path: pure loads, no CAS, no node. */
    uint64_t peek =
        atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_CACHE_RW(peek) == MSI_RW_GRANT ||
        MSI_CACHE_RO(peek) == MSI_RO_VALID) {
      mark_edt_ready_by_guid(edt->guid, slot);
      return;
    }
  }

  /* Park-or-open: the decision CAS links our node as the chain head in the
   * same atom (node fields written before every attempt). */
  uint32_t idx = msi_waiter_alloc(cache);
  struct arts_db_msi_waiter_s *node = msi_waiter_ptr(cache, idx);
  node->edt_guid = edt->guid;
  node->slot = slot;

  int op = (mode == DB_MODE_RW) ? MSI_CACHE_OP_ACQ_RW : MSI_CACHE_OP_ACQ_RO;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    node->next = (mode == DB_MODE_RW) ? MSI_CACHE_HEAD_RW(cur)
                                      : MSI_CACHE_HEAD_RO(cur);
    next = msi_cache_compute_next(cur, op, idx, &act);
    if (next == cur) {
      break; /* read on a copy that turned valid mid-retry: no word write */
    }
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));

  switch (act) {
  case MSI_CACHE_ACT_SELF_SERVE:
    msi_waiter_free(cache, idx);
    if (mode == DB_MODE_RW) {
      mark_edt_secured_by_guid(edt->guid, slot);
    }
    mark_edt_ready_by_guid(edt->guid, slot);
    break;
  case MSI_CACHE_ACT_SEND_RW:
    arts_send_db_msi_request(cache, DB_MODE_RW);
    break;
  case MSI_CACHE_ACT_SEND_RO:
    arts_send_db_msi_request(cache, DB_MODE_RO);
    break;
  default: /* PARK: the fetch's committer will serve us */
    break;
  }
}

/* Synchronous per-release writeback round: announce (+CTS+PUT for a remote
 * dirty round) then block until the invalidation round covering this
 * release closes (the ACK).  Mirrors the shared writeback_sync discipline,
 * including the heap rendezvous whose shutdown-escape LEAKS the block (a
 * late CTS/ACK may still write/post through the echoed cv — freeing it
 * would be a use-after-free; do not tighten). */
static void msi_writeback_sync(struct arts_db_cache_s *cache, uint64_t vnew,
                               uint32_t final_flag, const void *data,
                               uint64_t data_size) {
  unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
  struct arts_db_wb_rendezvous_s *wr =
      (struct arts_db_wb_rendezvous_s *)arts_malloc(sizeof(*wr));
  sem_init(&wr->sem, 0, 0);
  wr->landing = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  if (home_rank == arts_global_rank_id || data == NULL || data_size == 0) {
    arts_send_db_msi_writeback(cache, vnew, (uint64_t)(uintptr_t)wr,
                               final_flag, data, data_size,
                               /*rdzv_txid=*/0, /*rdzv_cookie=*/0);
    await_writeback_ack(&wr->sem);
    if (arts_atomic_read(&arts_node_info.shutdown_state) != 0) {
      return; /* shutdown escape: a late ACK may still post — leak */
    }
    sem_destroy(&wr->sem);
    arts_free(wr);
    return;
  }
  /* Remote dirty round: announce -> home landing (WRITEBACK_CTS) -> PUT ->
   * commit -> round-close ACK. */
  arts_send_db_msi_writeback(cache, vnew, (uint64_t)(uintptr_t)wr, final_flag,
                             /*data=*/NULL, data_size,
                             /*rdzv_txid=*/0, /*rdzv_cookie=*/0);
  await_writeback_ack(&wr->sem); /* CTS wake — or the shutdown escape */
  if (wr->landing.txid == 0) {
    /* Shutdown escape before the CTS: abandoned with the runtime.  This
     * unsynchronized read is exactly why wr must LEAK here. */
    return;
  }
  arts_net_put_payload((int)home_rank, wr->landing.addr, wr->landing.key,
                       wr->landing.txid, data, data_size,
                       /*on_local_done=*/NULL, NULL);
  arts_send_db_msi_writeback(cache, vnew, (uint64_t)(uintptr_t)wr, final_flag,
                             /*data=*/NULL, data_size, wr->landing.txid,
                             wr->landing.cookie);
  await_writeback_ack(&wr->sem); /* round-close ACK */
  if (arts_atomic_read(&arts_node_info.shutdown_state) != 0) {
    return;
  }
  sem_destroy(&wr->sem);
  arts_free(wr);
}

void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Writeback-axis numbering FIRST (fetch_add), then the word CAS carrying
   * {wc--, 0-edge demotion}, then the publication — the version label is
   * taken before the count so concurrent releases can never regress it. */
  uint64_t vnew =
      atomic_fetch_add_explicit(&cache->wb_next, 1u, memory_order_acq_rel);
  int op = MSI_CACHE_OP_REL_RW;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    next = msi_cache_compute_next(cur, op, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  uint32_t final_flag = (act == MSI_CACHE_ACT_WB_FINAL) ? 1u : 0u;
  if (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id) {
    /* Same-rank tenure: the home buffer IS the working buffer — publishing
     * must move no data and must NOT swap the install (a swap would orphan
     * the buffer this tenure's other writers keep writing).  The data-less
     * round still runs: it retires remote copies and advances the axis. */
    msi_writeback_sync(cache, vnew, final_flag, NULL, 0u);
    return;
  }
  /* Remote tenure: publish the working buffer.  The ref pins the bytes
   * across the one-sided PUT (the ACK follows the target-side write
   * completion, which implies the fabric drained the source). */
  arts_shared_ptr_t h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(h);
  msi_writeback_sync(cache, vnew, final_flag, buf ? buf->data : NULL,
                     buf ? cache->db_size : 0u);
  if (buf != NULL) {
    arts_db_buf_release(&h);
  }
}

/* arts_db_release_ro: the shared no-op body in coherence.c applies — an MSI
 * read release touches no protocol state (a valid copy persists; the EDT's
 * buffer ref is dropped by the dep-release path). */

/* ===== arts_handler_db_destroy =========================================
 * OOO_DB_DESTROY Cat-B body.  Fan-out DESTROY_NOTIFY to every rank in
 * cached_ranks, then detach the route-table slot. */
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
    const struct arts_rank_bitset_s *bs = &db->cached_ranks;
    for (unsigned int w = 0; w < bs->nwords; w++) {
      uint64_t snap = atomic_load_explicit(&bs->words[w], memory_order_acquire);
      while (snap) {
        unsigned int b = (unsigned int)__builtin_ctzll(snap);
        unsigned int rank = w * 64u + b;
        if (rank != self) {
          arts_send_db_cache_destroy(rank, a->db_guid);
        }
        snap &= snap - 1;
      }
    }
  }
  /* No rw_waiters drain: the request queue is single-consumer (the grant
   * claim's committer), and a legitimately destroyed DB has no outstanding
   * acquires — draining here would be a second, unsynchronized consumer. */
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* ===== home round engine ================================================
 * At most one invalidation round is open per DB (dir round_open).  A round:
 *   claim -> carry XCHG -> whole-batch queue drain -> single max-version
 *   install -> roster swap + ack arming -> INV multicast (+ the pending
 *   GRANT riding this round's send step) -> close on the ack 0-edge
 *   (WB_ACK every entry, retain-credit every final's releaser, one dir CAS
 *   folding {owner-final w--, owner clear, round_open drop, re-claim} — the
 *   counted final is the one matching the published owner; surplus creation
 *   finals are unit-neutral).
 * Every producer (writeback push, grant publish, close re-arm) calls
 * try_open; a claim that raced past the work it saw closes EMPTY (no
 * install, no swap, no INV).  The whole engine is a while-trampoline —
 * close chains (next grant, queued batch) loop instead of recursing. */

static void msi_home_grant_publish(struct arts_db_s *db);

static void msi_home_round_close(struct arts_db_s *db,
                                 struct arts_db_msi_wb_s *entries) {
  struct arts_db_cache_s *cache = &db->cache;
  /* A batch can hold MORE than one final: the (<=1) published owner's final
   * plus surplus creation-hold finals (concurrent creates of one GUID — the
   * directory counted only the creator it installed).  Only the final whose
   * releaser matches the published owner is the counted one; folding "last
   * final wins" into the close CAS would let a co-batched surplus final mask
   * the owner's decrement (w stuck > 0 — a permanent tenure).  The owner
   * field is stable across this close: it can only change GRANTING->rank
   * (a publish for a tenure whose final cannot be in an already-drained
   * batch), and NOBODY/GRANTING owners imply the prior final was already
   * counted, so a no-match batch is all-surplus by construction. */
  unsigned int owner_now = MSI_DIR_OWNER(
      atomic_load_explicit(&db->dir_state, memory_order_acquire));
  unsigned int final_rank = (unsigned int)MSI_OWNER_NOBODY;
  /* Credit step (commuting sends + the retain credit) BEFORE the close CAS:
   * the next round can open the instant round_open drops, and its roster
   * swap must already see the ex-owner's retained copy. */
  while (entries != NULL) {
    struct arts_db_msi_wb_s *e = entries;
    entries = (struct arts_db_msi_wb_s *)(uintptr_t)atomic_load_explicit(
        &e->link.next, memory_order_relaxed);
    if (e->final_flag != 0u) {
      if (e->releaser_rank == owner_now) {
        final_rank = e->releaser_rank;
      }
      if (e->releaser_rank != arts_global_rank_id) {
        (void)arts_rank_bitset_set(&db->roster, e->releaser_rank);
      }
    }
    if (e->cv != 0u) {
      arts_send_db_msi_writeback_ack(e->releaser_rank, cache->db_guid, e->cv);
    }
    arts_free(e);
  }
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    next = msi_dir_compute_next(cur, MSI_DIR_OP_ROUND_CLOSE, final_rank, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_DIR_ACT_GRANT_CLAIM) {
    msi_home_grant_publish(db); /* chain: grant the next queued writer */
  }
}

static void msi_home_round_try_open(struct arts_db_s *db) {
  struct arts_db_cache_s *cache = &db->cache;
  for (;;) {
    uint32_t act;
    uint64_t cur, next;
    do {
      cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
      next = msi_dir_compute_next(cur, MSI_DIR_OP_ROUND_CLAIM, 0u, &act);
      if (next == cur) {
        return; /* a round is already open; its closer re-arms */
      }
    } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                    memory_order_acq_rel,
                                                    memory_order_acquire));
    /* Claim held.  Take the pending-grant carry (exactly-once XCHG) and the
     * whole queued batch (Treiber head XCHG). */
    bool carry = atomic_exchange_explicit(&db->opening_pending, false,
                                          memory_order_acq_rel);
    arts_lf_link_t *chain = arts_lf_stack_drain(&db->wb_queue);
    struct arts_db_msi_wb_s *entries = (struct arts_db_msi_wb_s *)chain;
    if (!carry && entries == NULL) {
      /* Empty round (the claim raced past work another opener consumed):
       * close without install/swap/INV. */
      msi_home_round_close(db, NULL);
    } else {
      /* Single install point: only the batch's max-version entry publishes
       * (the conditional install rejects the rest); non-max landings
       * recycle at close (their entries' ACKs still fire). */
      struct arts_db_msi_wb_s *maxe = NULL;
      for (struct arts_db_msi_wb_s *e = entries; e != NULL;
           e = (struct arts_db_msi_wb_s *)(uintptr_t)atomic_load_explicit(
               &e->link.next, memory_order_relaxed)) {
        if (e->rdzv.addr != 0 && (maxe == NULL || e->vnew > maxe->vnew)) {
          maxe = e;
        }
      }
      uint64_t maxv = 0;
      for (struct arts_db_msi_wb_s *e = entries; e != NULL;
           e = (struct arts_db_msi_wb_s *)(uintptr_t)atomic_load_explicit(
               &e->link.next, memory_order_relaxed)) {
        if (e->vnew > maxv) {
          maxv = e->vnew;
        }
        if (e->rdzv.addr == 0) {
          continue; /* data-less entry (same-rank tenure / size-0 DB) */
        }
        struct arts_db_buffer_s *landing =
            (struct arts_db_buffer_s *)(uintptr_t)e->rdzv.cookie;
        if (e == maxe) {
          (void)arts_db_buf_install_landed(cache, e->vnew, landing,
                                           cache->db_size);
        } else {
          arts_db_buf_landing_recycle(cache, landing);
        }
        e->rdzv.addr = 0; /* consumed */
      }
      /* The protocol axis advances over the WHOLE batch — data-less
       * same-rank releases move no bytes (home buffer == the tenure's
       * working buffer) but their versions still retire older copies. */
      if (maxv > atomic_load_explicit(&db->hver, memory_order_acquire)) {
        atomic_store_explicit(&db->hver, maxv, memory_order_release);
      }
      /* First publication wakes the pre-publication read holds (readers
       * event-ordered after the creator's writes must see released bytes,
       * never the never-published zero state).  Guarded on an installed
       * buffer: the drain serves from the canonical copy. */
      {
        arts_shared_ptr_t pub_h = arts_db_buf_acquire(cache);
        bool pub = (arts_shared_get(pub_h) != NULL);
        arts_db_buf_release(&pub_h);
        if (pub) {
          arts_db_drain_pending_snapshot(cache);
        }
      }
      /* Roster swap (per-word XCHG: concurrent additions land in the NEXT
       * round) with the tenure owner self-excluded and the home's canonical
       * copy never a target; arm the ack count BEFORE the multicast. */
      uint64_t dirw = atomic_load_explicit(&db->dir_state, memory_order_acquire);
      unsigned int owner = MSI_DIR_OWNER(dirw);
      unsigned int self = arts_global_rank_id;
      unsigned int targets[64];
      unsigned int ntargets = 0;
      const struct arts_rank_bitset_s *bs = &db->roster;
      for (unsigned int wi = 0; wi < bs->nwords; wi++) {
        uint64_t snap = atomic_exchange_explicit(
            (_Atomic uint64_t *)&bs->words[wi], 0u, memory_order_acq_rel);
        while (snap) {
          unsigned int b = (unsigned int)__builtin_ctzll(snap);
          unsigned int rank = wi * 64u + b;
          snap &= snap - 1;
          if (rank == self || rank == owner) {
            continue;
          }
          if (ntargets <
              (unsigned int)(sizeof(targets) / sizeof(targets[0]))) {
            targets[ntargets++] = rank;
          } else {
            /* Rank counts beyond the stack window would need a heap list;
             * the rank field is 14-bit but deployments here are far
             * smaller — fail loudly rather than silently truncate. */
            (void)fprintf(stderr, "arts msi: round target overflow\n");
            abort();
          }
        }
      }
      uint32_t aact;
      do {
        cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
        next = msi_dir_compute_next(cur, MSI_DIR_OP_ACKS_ARM, ntargets, &aact);
      } while (!atomic_compare_exchange_weak_explicit(
          &db->dir_state, &cur, next, memory_order_acq_rel,
          memory_order_acquire));
      /* Publish the open round's batch for the closing ACK committer
       * BEFORE any INV departs. */
      db->round_entries = entries;
      for (unsigned int t = 0; t < ntargets; t++) {
        arts_send_db_msi_invalidate(targets[t], cache->db_guid);
      }
      if (carry) {
        /* The pending GRANT rides this round's send step: every earlier
         * round's INV to the grantee was ACKed — hence processed — before
         * the grant can be emitted.  Base = the canonical version this
         * round just advanced to. */
        uint64_t base = atomic_load_explicit(&db->hver, memory_order_acquire);
        uint64_t dnow = atomic_load_explicit(&db->dir_state,
                                             memory_order_acquire);
        arts_send_db_msi_grant(MSI_DIR_OWNER(dnow), db, base,
                               &db->grant_rdzv);
      }
      if (ntargets > 0u) {
        return; /* the ack 0-edge committer closes (and re-arms) */
      }
      db->round_entries = NULL;
      msi_home_round_close(db, entries);
    }
    /* Re-arm: work that arrived while we were closing. */
    if (!atomic_load_explicit(&db->opening_pending, memory_order_acquire) &&
        arts_lf_stack_empty(&db->wb_queue)) {
      return;
    }
  }
}

static void msi_home_grant_publish(struct arts_db_s *db) {
  unsigned int rank = 0;
  struct arts_rdzv_landing_s rdzv;
  bool ok = arts_home_lockreq_queue_pop(&db->rw_waiters, &rank, &rdzv);
  if (!ok) {
    /* The grant claim counted a writer whose queue push is mid-flight —
     * the push precedes the dir CAS, so this resolves within the pusher's
     * instruction window. */
    do {
      ok = arts_home_lockreq_queue_pop(&db->rw_waiters, &rank, &rdzv);
    } while (!ok);
  }
  db->grant_rdzv = rdzv;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    next = msi_dir_compute_next(cur, MSI_DIR_OP_OWNER_PUBLISH, rank, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  /* Arm the tenure-opening round (survives a busy round via the latch; the
   * closer re-arms it) and try to open.  The fence between the latch store
   * and try_open's round_open load is load-bearing (Dekker): without it the
   * load can execute ahead of the store draining, a concurrently-closing
   * round misses the latch in BOTH its carry exchange and its close
   * re-check, and this producer — the grant's only driver — has already
   * given up on the failed claim: the grant strands forever. */
  atomic_store_explicit(&db->opening_pending, true, memory_order_release);
  atomic_thread_fence(memory_order_seq_cst);
  msi_home_round_try_open(db);
}

/* ===== wire handler bodies (OoO Cat-B) ================================== */

/* Deferred pre-publication read serve: re-issued by the pending_snapshot
 * drain once the first install lands.  Same 3-step as the direct serve
 * (roster bit before the send). */
static void msi_pending_ro_serve(struct arts_db_cache_s *cache,
                                 struct arts_db_snapshot_waiter_s *w) {
  struct arts_db_s *db = (struct arts_db_s *)cache; /* cache is FIRST member */
  (void)arts_rank_bitset_set(&db->roster, w->requester);
  arts_send_db_msi_deliver(w->requester, db, &w->rdzv);
}

void arts_handler_db_msi_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_msi_request_s *a =
      (struct arts_ooo_args_db_msi_request_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  (void)arts_rank_bitset_set(&db->cached_ranks, a->requester);
  if (a->rdzv.txid == 0 && cache->db_size != 0 &&
      a->requester != arts_global_rank_id) {
    /* First touch: the request carried no landing (db_size unknown at the
     * requester).  It was NOT queued/served — answer the size and let the
     * requester re-issue with a landing. */
    arts_send_db_msi_cts(a->requester, cache->db_guid, cache->db_size,
                         a->mode);
    return;
  }
  if (a->mode == DB_MODE_RO) {
    if (cache->db_size != 0) {
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      bool published = (arts_shared_get(buf_h) != NULL);
      arts_db_buf_release(&buf_h);
      if (!published) {
        /* Pre-publication hold: no release has published this DB yet, and a
         * read that is event-ordered after the creator's writes must observe
         * the release-published bytes — serving now would hand out
         * never-published state.  Push-then-recheck: a concurrent first
         * install either sees our node in its drain or we drain ourselves. */
        struct arts_db_snapshot_waiter_s *w =
            (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
        w->edt_guid = NULL_GUID;
        w->slot = 0;
        w->target_version = 0;
        w->serve = msi_pending_ro_serve;
        w->requester = a->requester;
        w->rdzv = a->rdzv;
        arts_lf_stack_push(&cache->pending_snapshot, &w->link);
        buf_h = arts_db_buf_acquire(cache);
        published = (arts_shared_get(buf_h) != NULL);
        arts_db_buf_release(&buf_h);
        if (published) {
          arts_db_drain_pending_snapshot(cache);
        }
        return;
      }
    }
    /* Read serve: roster bit BEFORE the send — a serve whose copy departs
     * must already be a target of the next round. */
    (void)arts_rank_bitset_set(&db->roster, a->requester);
    arts_send_db_msi_deliver(a->requester, db, &a->rdzv);
    return;
  }
  /* Write request: queue push BEFORE the dir CAS (the claim's committer is
   * the sole popper, and the pop must always find the counted node). */
  arts_home_lockreq_queue_push(&db->rw_waiters, a->requester, &a->rdzv);
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    next = msi_dir_compute_next(cur, MSI_DIR_OP_REQ_RW, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_DIR_ACT_GRANT_CLAIM) {
    msi_home_grant_publish(db);
  }
}

/* Commit-leg pairing continuation: the dirty bytes are in the home landing;
 * queue the entry and drive the round. */
struct msi_wb_landed_ctx_s {
  arts_shared_ptr_t db_h;
  arts_guid_t db_guid;
  struct arts_db_msi_wb_s *entry;
};

static void msi_wb_landed_cb(void *arg) {
  struct msi_wb_landed_ctx_s *ctx = (struct msi_wb_landed_ctx_s *)arg;
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(ctx->db_h);
  if (db == NULL) {
    /* Destroyed mid-pairing: free the landing's storage, never strand the
     * blocked releaser. */
    struct arts_db_msi_wb_s *e = ctx->entry;
    if (e->rdzv.cookie != 0) {
      arts_regpool_free((void *)(uintptr_t)e->rdzv.cookie);
    }
    if (e->cv != 0) {
      arts_send_db_msi_writeback_ack(e->releaser_rank, ctx->db_guid, e->cv);
    }
    arts_free(e);
    arts_shared_release(&ctx->db_h);
    arts_free(ctx);
    return;
  }
  arts_lf_stack_push(&db->wb_queue, &ctx->entry->link);
  msi_home_round_try_open(db);
  arts_shared_release(&ctx->db_h);
  arts_free(ctx);
}

void arts_handler_db_msi_writeback(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_msi_writeback_s *a =
      (struct arts_ooo_args_db_msi_writeback_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  if (a->data_size != 0 && a->data_inline == 0 && a->rdzv_txid == 0) {
    /* Announce leg: hand the releaser a fresh home landing; nothing queues
     * yet — the commit leg (paired with the PUT completion) does. */
    struct arts_rdzv_landing_s landing;
    (void)arts_db_buf_landing_alloc(cache, a->data_size, &landing);
    arts_send_db_writeback_cts(a->releaser, a->db_guid, &landing, a->cv);
    return;
  }
  struct arts_db_msi_wb_s *e =
      (struct arts_db_msi_wb_s *)arts_malloc(sizeof(*e));
  e->vnew = a->vnew;
  e->releaser_rank = a->releaser;
  e->final_flag = a->final_flag;
  e->cv = a->cv;
  e->rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  if (a->data_inline != 0) {
    /* Same-rank publication: copy the inline payload into a pool buffer so
     * the round's single install point treats it like a landed one. */
    struct arts_db_buffer_s *b = arts_db_buf_alloc(cache, cache->db_size);
    memcpy(b->data, (const char *)a + sizeof(*a), a->data_size);
    e->rdzv.addr = 1u; /* data marker */
    e->rdzv.cookie = (uint64_t)(uintptr_t)b;
  } else if (a->rdzv_txid != 0) {
    /* Commit leg: the entry may only queue once the one-sided payload has
     * fully landed — pair {commit, PUT completion} (either arrival order)
     * and enqueue from the pairing continuation.  Queueing on the control
     * packet alone would let the round install a torn landing. */
    e->rdzv.addr = 1u;
    e->rdzv.cookie = a->rdzv_cookie;
    struct msi_wb_landed_ctx_s *ctx =
        (struct msi_wb_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = arts_route_table_lookup_db(a->db_guid);
    ctx->db_guid = a->db_guid;
    ctx->entry = e;
    arts_net_rdzv_expect(a->rdzv_txid, msi_wb_landed_cb, ctx);
    return;
  }
  arts_lf_stack_push(&db->wb_queue, &e->link);
  msi_home_round_try_open(db);
}

/* ===== Cat-C handler bodies ============================================= */

void arts_handler_db_msi_cts(struct arts_db_s *db,
                             struct arts_msg_msi_cts_packet_s *p) {
  struct arts_db_cache_s *cache = &db->cache;
  if (cache->db_size == 0) {
    cache->db_size = p->db_size;
  }
  arts_send_db_msi_request(cache, (arts_db_access_mode_t)p->mode);
}

/* Publish + whole-chain grab: install the landed payload (home-issued
 * install-lane version; a stale install retreats and recycles), then ONE
 * word CAS takes {REQ->VALID, chain head -> 0} together.  The kill-marked
 * variant serves its cohort once, then executes the reserved purge and the
 * owed ack — the ack LAST, so the round that killed this fetch cannot close
 * (and nothing newer can complete) until the doomed serve is done. */
static void msi_deliver_commit(arts_shared_ptr_t db_h, uint64_t version,
                               struct arts_db_buffer_s *landing,
                               uint64_t data_size) {
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  struct arts_db_cache_s *cache = &db->cache;
  if (landing != NULL) {
    (void)arts_db_buf_install_landed(cache, version, landing, data_size);
  }
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    next = msi_cache_compute_next(cur, MSI_CACHE_OP_DELIVER, 0u, &act);
    if (next == cur) {
      break; /* DROP: superseded (the grant's install-lane version won) */
    }
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_CACHE_ACT_PUBLISH || act == MSI_CACHE_ACT_PUBLISH_KILL) {
    msi_serve_chain(cache, MSI_CACHE_HEAD_RO(cur), /*serialized=*/false);
  }
  if (act == MSI_CACHE_ACT_PUBLISH_KILL) {
    uint32_t pact;
    do {
      cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
      /* The owed ack blocks the round chain, so nothing can retire the
       * transient valid copy before this purge. */
      if (MSI_CACHE_RO(cur) != MSI_RO_VALID) {
        (void)fprintf(stderr,
                      "arts msi: reserved purge found ro=%u (drifted)\n",
                      MSI_CACHE_RO(cur));
        abort();
      }
      next = msi_cache_compute_next(cur, MSI_CACHE_OP_KILL_PURGE, 0u, &pact);
    } while (!atomic_compare_exchange_weak_explicit(
        &cache->cache_state, &cur, next, memory_order_acq_rel,
        memory_order_acquire));
    arts_send_db_msi_invalidate_ack(arts_guid_get_rank(cache->db_guid),
                                    cache->db_guid); /* owed ack, LAST */
  }
  arts_shared_release(&db_h);
}

struct msi_landed_ctx_s {
  arts_shared_ptr_t db_h;
  struct arts_db_buffer_s *landing;
  uint64_t version;
  uint64_t data_size;
  bool is_grant;
};

static void msi_grant_commit(arts_shared_ptr_t db_h, uint64_t version,
                             struct arts_db_buffer_s *landing,
                             uint64_t data_size);

static void msi_landed_cb(void *arg) {
  struct msi_landed_ctx_s *ctx = (struct msi_landed_ctx_s *)arg;
  if (arts_shared_get(ctx->db_h) == NULL) {
    /* Destroyed while the pairing was outstanding (destroy during a pending
     * acquire is app UB): the cache's recycle pool is gone — return the
     * landing's storage straight to the registered pool and drop. */
    arts_regpool_free(ctx->landing);
    arts_shared_release(&ctx->db_h);
    arts_free(ctx);
    return;
  }
  if (ctx->is_grant) {
    msi_grant_commit(ctx->db_h, ctx->version, ctx->landing, ctx->data_size);
  } else {
    msi_deliver_commit(ctx->db_h, ctx->version, ctx->landing, ctx->data_size);
  }
  arts_free(ctx);
}

void arts_handler_db_msi_deliver(void *payload, size_t size) {
  (void)size;
  struct arts_msg_msi_deliver_packet_s *p =
      (struct arts_msg_msi_deliver_packet_s *)payload;
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_db_rdzv_discard_landing(p->rdzv_txid, p->rdzv_cookie);
    arts_shared_release(&db_h);
    return;
  }
  if (p->rdzv_txid != 0) {
    struct msi_landed_ctx_s *ctx =
        (struct msi_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->landing = (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie;
    ctx->version = p->version;
    ctx->data_size = p->data_size;
    ctx->is_grant = false;
    arts_net_rdzv_expect(p->rdzv_txid, msi_landed_cb, ctx);
    return;
  }
  /* Data-less reply: an advertised-but-unused landing recycles. */
  if (p->rdzv_cookie != 0) {
    arts_db_buf_landing_recycle(
        &db->cache, (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie);
  }
  msi_deliver_commit(db_h, p->version, NULL, 0u);
}

static void msi_grant_commit(arts_shared_ptr_t db_h, uint64_t version,
                             struct arts_db_buffer_s *landing,
                             uint64_t data_size) {
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  struct arts_db_cache_s *cache = &db->cache;
  if (landing != NULL) {
    (void)arts_db_buf_install_landed(cache, version, landing, data_size);
  }
  /* Seed the writeback axis BEFORE the publish lets local writers run: the
   * grantee's releases number monotonically from the grant's base. */
  atomic_store_explicit(&cache->wb_next, version + 1u, memory_order_release);
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    /* Contract: home grants only what was requested, and a grant can never
     * overtake a kill's owed ack (the un-acked round blocks the chain the
     * grant rides).  A violation silently absorbed here would drop that
     * owed ack and wedge the round chain — fail loudly instead. */
    if (MSI_CACHE_RW(cur) != MSI_RW_REQ ||
        MSI_CACHE_RO(cur) == MSI_RO_REQ_KILL) {
      (void)fprintf(stderr, "arts msi: grant against an impossible cache "
                            "state (rw=%u ro=%u)\n",
                    MSI_CACHE_RW(cur), MSI_CACHE_RO(cur));
      abort();
    }
    next = msi_cache_compute_next(cur, MSI_CACHE_OP_GRANT, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  /* Both chains were grabbed in the publish CAS: writers first (they were
   * counted at acquire and pin wc until served+released), then readers. */
  msi_serve_chain(cache, MSI_CACHE_HEAD_RW(cur), /*serialized=*/true);
  msi_serve_chain(cache, MSI_CACHE_HEAD_RO(cur), /*serialized=*/false);
  arts_shared_release(&db_h);
}

void arts_handler_db_msi_grant(void *payload, size_t size) {
  (void)size;
  struct arts_msg_msi_grant_packet_s *p =
      (struct arts_msg_msi_grant_packet_s *)payload;
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_db_rdzv_discard_landing(p->rdzv_txid, p->rdzv_cookie);
    arts_shared_release(&db_h);
    return;
  }
  if (p->rdzv_txid != 0) {
    struct msi_landed_ctx_s *ctx =
        (struct msi_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->landing = (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie;
    ctx->version = p->version;
    ctx->data_size = p->data_size;
    ctx->is_grant = true;
    arts_net_rdzv_expect(p->rdzv_txid, msi_landed_cb, ctx);
    return;
  }
  if (p->rdzv_cookie != 0) {
    arts_db_buf_landing_recycle(
        &db->cache, (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie);
  }
  msi_grant_commit(db_h, p->version, NULL, 0u);
}

void arts_handler_db_msi_invalidate(struct arts_db_s *db) {
  struct arts_db_cache_s *cache = &db->cache;
  unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    /* An invalidate can never hit an active write grant: the round snapshot
     * self-excludes the tenure owner and rounds serialize. */
    if (MSI_CACHE_RW(cur) == MSI_RW_GRANT) {
      (void)fprintf(stderr,
                    "arts msi: invalidate hit an active write grant\n");
      abort();
    }
    next = msi_cache_compute_next(cur, MSI_CACHE_OP_INVALIDATE, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_CACHE_ACT_PURGE_ACK || act == MSI_CACHE_ACT_NOOP_ACK) {
    arts_send_db_msi_invalidate_ack(home_rank, cache->db_guid);
  }
  /* KILL_MARKED: the ack is owed — the doomed DELIVER fires it after its
   * serve-once, keeping every pre-ack serve legally racy. */
}

void arts_handler_db_msi_invalidate_ack(struct arts_db_s *db,
                                        unsigned int sharer_rank) {
  (void)sharer_rank;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    next = msi_dir_compute_next(cur, MSI_DIR_OP_ACK_DEC, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_DIR_ACT_CLOSE) {
    struct arts_db_msi_wb_s *entries = db->round_entries;
    db->round_entries = NULL;
    msi_home_round_close(db, entries);
    /* Re-arm work that queued while the round was in flight. */
    if (atomic_load_explicit(&db->opening_pending, memory_order_acquire) ||
        !arts_lf_stack_empty(&db->wb_queue)) {
      msi_home_round_try_open(db);
    }
  }
}

/* ===== senders ========================================================== */
/* Control-only senders are plain packet fills; the payload-carrying trio
 * (deliver/grant/writeback) lands with the data-plane implementation. */

void arts_send_db_msi_cts(unsigned int requester_rank, arts_guid_t db_guid,
                          uint64_t db_size, arts_db_access_mode_t mode) {
  struct arts_msg_msi_cts_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_CTS);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.db_size = db_size;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_writeback_ack(unsigned int releaser_rank,
                                    arts_guid_t db_guid, uint64_t cv) {
  if (releaser_rank == arts_global_rank_id) {
    /* Home-local releaser: wake by pointer identity, no wire. */
    if (cv != 0) {
      sem_post((sem_t *)(uintptr_t)cv);
    }
    return;
  }
  struct arts_msg_msi_writeback_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_WRITEBACK_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.cv = cv;
  arts_transport_send_async((int)releaser_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_invalidate(unsigned int sharer_rank,
                                 arts_guid_t db_guid) {
  struct arts_msg_msi_invalidate_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_INVALIDATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)sharer_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_invalidate_ack(unsigned int home_rank,
                                     arts_guid_t db_guid) {
  struct arts_msg_msi_invalidate_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_INVALIDATE_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_request(struct arts_db_cache_s *cache,
                              arts_db_access_mode_t mode) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  struct arts_rdzv_landing_s rdzv = {0, 0, 0, 0};
  if (cache->db_size != 0 && arts_global_rank_count > 1 &&
      home_rank != arts_global_rank_id) {
    /* A FRESH landing per fetch: reception isolation when a read reply and
     * a grant are in flight together (each PUT lands in its own buffer,
     * installed or recycled at commit).  txid==0 = first touch (db_size
     * unknown): home answers MSI_CTS and the fetch re-issues. */
    (void)arts_db_buf_landing_alloc(cache, cache->db_size, &rdzv);
  }
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine so before-create reorders
     * defer exactly like the wire path. */
    struct arts_ooo_args_db_msi_request_s args = {
        .requester = arts_global_rank_id,
        .db_guid = db_guid,
        .mode = mode,
        .rdzv = rdzv,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_MSI_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  struct arts_msg_msi_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.rdzv.addr = rdzv.addr;
  p.rdzv.key = rdzv.key;
  p.rdzv.txid = rdzv.txid;
  p.rdzv.cookie = rdzv.cookie;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_deliver(unsigned int requester_rank,
                              struct arts_db_s *db,
                              const struct arts_rdzv_landing_s *rdzv) {
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_msg_msi_deliver_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_DELIVER);
  p.header.rank = arts_global_rank_id;
  p.db_guid = cache->db_guid;
  p.data_size = 0;
  p.rdzv_txid = 0;
  p.rdzv_cookie = (rdzv != NULL) ? rdzv->cookie : 0;
  /* Serve from the CURRENT install; the ref pins the bytes across the PUT
   * (released by the fabric's local completion).  The stamp is the
   * canonical axis (hver), which may run ahead of the installed buffer's
   * lane after same-rank (data-less) rounds. */
  arts_shared_ptr_t src_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  p.version = atomic_load_explicit(&arts_db_of_cache(cache)->hver,
                                   memory_order_acquire);
  if (src == NULL || rdzv == NULL || rdzv->txid == 0) {
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
    return;
  }
  p.data_size = cache->db_size;
  p.rdzv_txid = rdzv->txid;
  arts_net_put_payload((int)requester_rank, rdzv->addr, rdzv->key, rdzv->txid,
                       src->data, cache->db_size, arts_db_buf_ref_release_cb,
                       (void *)src_h);
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_grant(unsigned int requester_rank, struct arts_db_s *db,
                            uint64_t version,
                            const struct arts_rdzv_landing_s *rdzv) {
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_msg_msi_grant_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_GRANT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = cache->db_guid;
  p.version = version; /* writeback-axis base AND the install-lane stamp */
  p.data_size = 0;
  p.rdzv_txid = 0;
  p.rdzv_cookie = (rdzv != NULL) ? rdzv->cookie : 0;
  if (requester_rank == arts_global_rank_id) {
    /* Home-local grant: the canonical buffer already lives here; a data-less
     * self-grant runs the commit against it (the unused landing recycles in
     * the handler). */
    p.header.size = sizeof(p);
    arts_handler_db_msi_grant(&p, sizeof(p));
    return;
  }
  arts_shared_ptr_t src_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  if (src == NULL || rdzv == NULL || rdzv->txid == 0) {
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
    return;
  }
  p.data_size = cache->db_size;
  p.rdzv_txid = rdzv->txid;
  arts_net_put_payload((int)requester_rank, rdzv->addr, rdzv->key, rdzv->txid,
                       src->data, cache->db_size, arts_db_buf_ref_release_cb,
                       (void *)src_h);
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_writeback(struct arts_db_cache_s *cache, uint64_t vnew,
                                uint64_t cv, uint32_t final_flag,
                                const void *data, uint64_t data_size,
                                uint64_t rdzv_txid, uint64_t rdzv_cookie) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  if (home_rank == arts_global_rank_id) {
    /* Same-rank publication: the payload rides inline through the OoO args
     * copy (no wire, no RDMA); before-create reorders defer like RX. */
    uint64_t inline_size = (data != NULL) ? data_size : 0u;
    size_t asz = sizeof(struct arts_ooo_args_db_msi_writeback_s) +
                 (size_t)inline_size;
    char *abuf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_msi_writeback_s *args =
        (struct arts_ooo_args_db_msi_writeback_s *)abuf;
    args->releaser = arts_global_rank_id;
    args->db_guid = db_guid;
    args->vnew = vnew;
    args->cv = cv;
    args->final_flag = final_flag;
    args->data_inline = (inline_size > 0) ? 1u : 0u;
    args->data_size = data_size;
    args->rdzv_txid = 0;
    args->rdzv_cookie = 0;
    if (inline_size > 0) {
      memcpy(abuf + sizeof(*args), data, inline_size);
    }
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_MSI_WRITEBACK, abuf, asz);
    arts_free(abuf);
    return;
  }
  /* Remote: control-only in every leg — announce (data_size>0, txid 0) or
   * commit (txid set); the dirty payload travels one-sided. */
  struct arts_msg_msi_writeback_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_WRITEBACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.vnew = vnew;
  p.cv = cv;
  p.final_flag = final_flag;
  p.pad = 0;
  p.data_size = data_size;
  p.rdzv_txid = rdzv_txid;
  p.rdzv_cookie = rdzv_cookie;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}
