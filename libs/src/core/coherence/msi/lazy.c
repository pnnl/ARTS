/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol, LAZY timing: cache lifecycle + acquire/release/handler/sender
 * bodies.
 *
 * The canonical copy lives at the owner and moves owner → owner by migration
 * only; the home is a pure directory (owner field, migration claim, writer
 * FIFO, copy roster, invalidation-round engine) that never holds bytes.  There
 * is no writeback and no writeback ack — that is the only thing this timing
 * drops.  The invalidate ack it keeps: EVERY RW release asks the home for a
 * round, the home snapshots the roster, multicasts, collects every ack, and
 * only then does the release return.
 *
 * A read registers in the roster and is redirected to the owner, which serves
 * it immediately — whether or not a local writer is running there — and the
 * requester ALWAYS installs a durable copy.  Every later read on that rank is a
 * pure load: no message, no CAS, no version.  A copy dies exactly one way: an
 * INVALIDATE arrived.  A rank holding no bytes bounces the redirect back
 * through the home; that is a retry, not a park.
 *
 * Soundness rests on one order: the writes are in the owner's buffer BEFORE the
 * roster snapshot is taken, so a registration older than the snapshot is killed
 * by the round and a newer one is redirected to bytes that already carry the
 * write.  Neither leaves anything stale behind.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=MSI with LAZY timing.
 */
#include "arts/coherence/msi/types.h"

#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
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
#include "arts/system/print.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== pure arbiters (private copy — see arbiters.c) ==================== */
#include "arbiters.c"

/* ===== waiter pool (index-addressed chain nodes) ========================
 * Chain nodes live in per-DB chunked storage addressed by a 14-bit index
 * (0 = none) so a chain head fits the cache word.  Alloc = tagged free-list
 * pop (the tag defeats index ABA under concurrent alloc/free) with a
 * bump-allocator fallback; free = tagged push.  A node's fields are written
 * before the CAS that links it and read only by the committer that grabbed the
 * chain — single-owner after the grab, freed back to the pool after its
 * serve. */

#define MSI_WAITER_NCHUNKS                                                     \
  ((uint32_t)(((uint64_t)MSI_WAITER_IDX_MAX + 1u) / MSI_WAITER_CHUNK_CAP))
/* The tag takes EVERY bit the index leaves: it is what defeats index ABA, so
 * its wrap period is the only bound on how long a thread may be preempted
 * between reading the head and its compare-exchange. */
#define MSI_FREE_TAG_MASK                                                      \
  ((uint32_t)((1u << (32u - MSI_LAZY_CACHE_HEAD_BITS)) - 1u))
#define MSI_FREE_IDX(h) ((uint32_t)((h) & MSI_LAZY_CACHE_HEAD_MASK))
#define MSI_FREE_TAG(h) ((uint32_t)((h) >> MSI_LAZY_CACHE_HEAD_BITS))
#define MSI_FREE_MAKE(tag, idx)                                                \
  ((uint32_t)((((tag) & MSI_FREE_TAG_MASK) << MSI_LAZY_CACHE_HEAD_BITS) |      \
              ((idx) & MSI_LAZY_CACHE_HEAD_MASK)))

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
  uint32_t idx = atomic_fetch_add_explicit(&c->waiters.next_fresh, 1u,
                                           memory_order_acq_rel);
  if (idx > (uint32_t)MSI_WAITER_IDX_MAX) {
    ARTS_ERROR("arts msi: parked-waiter pool exhausted (index space)");
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
 * same CAS that published the transition, so it holds exactly that
 * transition's waiters — however late this loop runs, no other actor can reach
 * these nodes.  The secured signal (serialized-cursor advance) is reserved for
 * serialized deps: raising it for a non-serialized dep that happens to sit at
 * the cursor position falsely advances the cursor and enqueues a duplicate
 * resume, double-driving the next serialized dep.  A read acquire is never
 * serialized here, so every read serve gets the data-arrival signal only. */
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
  uint64_t seed = 0ULL; /* IDLE/IDLE: no copy, no ownership, no chains */
  if (kind == ARTS_DB_INIT_CREATOR_HOME ||
      kind == ARTS_DB_INIT_CREATOR_REMOTE) {
    /* The creator boots as the owner holding the DB (create defaults to an RW
     * acquire), so it may store at once: a brand-new GUID has no copies
     * anywhere.  Its publication release is an ORDINARY release and runs an
     * ordinary round — reads racing the create hold are served live bytes and
     * install durable copies that only that round retires. */
    seed = MSI_LAZY_CACHE_MAKE(MSI_RW_GRANT, MSI_RO_VALID, 0u, 0u, 0u, 1u, 0u,
                               0u, 0u);
  }
  atomic_store_explicit(&c->cache_state, seed, memory_order_relaxed);
  atomic_store_explicit(&c->waiters.chunks, (uintptr_t)0, memory_order_relaxed);
  atomic_store_explicit(&c->waiters.next_fresh, 1u, memory_order_relaxed);
  atomic_store_explicit(&c->waiters.free_head, 0u, memory_order_relaxed);
  atomic_store_explicit(&c->migrate_target, MSI_OWNER_NOBODY,
                        memory_order_relaxed);
  c->migrate_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache);
  /* A destroy that respects the contract finds every chain empty (nothing
   * blocks here, so no waiter can outlive the cache) and only the node-pool
   * storage to tear down. */
  struct msi_waiter_dir_s *dir =
      (struct msi_waiter_dir_s *)atomic_load_explicit(&cache->waiters.chunks,
                                                      memory_order_acquire);
  if (dir != NULL) {
    for (uint32_t i = 0; i < MSI_WAITER_NCHUNKS; i++) {
      struct arts_db_msi_waiter_s *chunk =
          atomic_load_explicit(&dir->chunk[i], memory_order_acquire);
      if (chunk != NULL) {
        arts_free(chunk);
      }
    }
    arts_free(dir);
    atomic_store_explicit(&cache->waiters.chunks, (uintptr_t)0,
                          memory_order_relaxed);
  }
  arts_db_cache_common_destroy_post(cache);
}

void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  /* The directory names the creator as owner from arts_db_home_init; there is
   * no separate holder field to publish at create. */
  (void)db;
  (void)creator_rank;
}

void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  /* The home holds no data in this timing: the creator is the owner and the
   * home is a pure directory. */
  (void)cache;
  (void)db_size;
}

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  /* A write acquire can wait on other ranks' releases (cross-DB cyclic hazard:
   * acquire-all must issue it in strict slot order).  A read acquire waits on
   * no EDT's release — it is served or bounced — so it carries no cycle risk
   * and may issue in parallel. */
  return mode == DB_MODE_RW;
}

/* ===== acquire / release ================================================ */

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
    /* Blind local hit: a covering copy is read with ZERO coherence-word CASes,
     * zero messages and — decisively — zero comparisons.  Holding a copy IS
     * the permission to read it; only an invalidate takes that away.  Reading
     * the owner's live buffer while a local writer runs there is a race the
     * memory model already permits (the two are unordered by construction). */
    uint64_t peek =
        atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_LAZY_CACHE_RW(peek) == MSI_RW_GRANT ||
        MSI_LAZY_CACHE_RO(peek) == MSI_RO_VALID) {
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

  int op = (mode == DB_MODE_RW) ? MSI_LAZY_CACHE_OP_ACQ_RW
                                : MSI_LAZY_CACHE_OP_ACQ_RO;
  uint32_t act = MSI_LAZY_CACHE_ACT_NONE;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    node->next = (mode == DB_MODE_RW) ? MSI_LAZY_CACHE_HEAD_RW(cur)
                                      : MSI_LAZY_CACHE_HEAD_RO(cur);
    next = msi_lazy_cache_compute_next(cur, op, idx, &act);
    if (next == cur) {
      break; /* covering copy: the word does not move */
    }
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));

  switch (act) {
  case MSI_LAZY_CACHE_ACT_SELF_SERVE:
    msi_waiter_free(cache, idx);
    if (mode == DB_MODE_RW) {
      mark_edt_secured_by_guid(edt->guid, slot);
    }
    mark_edt_ready_by_guid(edt->guid, slot);
    break;
  case MSI_LAZY_CACHE_ACT_SEND_RO:
    arts_send_db_msi_request(cache, DB_MODE_RO);
    break;
  case MSI_LAZY_CACHE_ACT_SEND_RW:
    arts_send_db_msi_request(cache, DB_MODE_RW);
    break;
  default: /* PARK: whatever lands serves us */
    break;
  }
}

/* Ship the canonical bytes to the target the consuming CAS named, then
 * re-request ownership when local writers are still queued.  Everything it
 * needs was captured BEFORE that CAS — the word cannot carry the off-word
 * target, and re-reading state the CAS already decided is how a second request
 * gets emitted for one migration. */
static void msi_lazy_ship(struct arts_db_cache_s *cache, unsigned int target,
                          const struct arts_rdzv_landing_s *rdzv,
                          uint64_t next_word) {
  arts_shared_ptr_t src_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  uint64_t base =
      (src != NULL) ? __atomic_load_n(&src->version, __ATOMIC_ACQUIRE) : 0u;
  uint64_t data_size = (src != NULL) ? cache->db_size : 0u;
  arts_send_db_msi_deliver_rw(target, cache->db_guid, base, rdzv, src_h,
                              data_size);
  if (MSI_LAZY_CACHE_RW(next_word) == MSI_RW_REQ) {
    arts_send_db_msi_request(cache, DB_MODE_RW);
  }
}

/* Re-attempt the ownership hand-off against the current word.  Used wherever a
 * hand-off may have become possible without a state change of its own (the
 * confirm gate opening).  ACT_NONE leaves the word untouched, so the CAS is
 * skipped entirely. */
static void msi_lazy_try_migrate(struct arts_db_cache_s *cache) {
  unsigned int target = MSI_OWNER_NOBODY;
  struct arts_rdzv_landing_s rdzv = {0, 0, 0, 0};
  uint32_t act = MSI_LAZY_CACHE_ACT_NONE;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_LAZY_CACHE_MTP(cur) != 0u) {
      /* Only read the off-word pair when the bit that publishes it is set:
       * with the bit clear they are dead and their writer is unsynchronised. */
      target =
          atomic_load_explicit(&cache->migrate_target, memory_order_acquire);
      rdzv = cache->migrate_rdzv;
    }
    next = msi_lazy_cache_compute_next(cur, MSI_LAZY_CACHE_OP_MIGRATE, 0u,
                                       &act);
    if (next == cur) {
      return; /* no hand-off is possible from this state */
    }
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_LAZY_CACHE_ACT_MIGRATE) {
    msi_lazy_ship(cache, target, &rdzv, next);
  }
}

/* Block until this release's invalidation round closes.  The ONLY blocking
 * point in the arm, and it follows the sanctioned rendezvous discipline: a
 * 1-second re-check cadence, an escape once teardown starts (the round can no
 * longer complete), and a HEAP block that is deliberately LEAKED on that escape
 * — a late ROUND_DONE still posts through the echoed address, so freeing it
 * would be a use-after-free.  Do not tighten. */
static void msi_lazy_await_round(struct arts_db_cache_s *cache) {
  struct arts_db_wb_rendezvous_s *wr =
      (struct arts_db_wb_rendezvous_s *)arts_malloc(sizeof(*wr));
  sem_init(&wr->sem, 0, 0);
  wr->landing = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  arts_send_db_msi_round_req(cache->db_guid, (uint64_t)(uintptr_t)wr);
  await_writeback_ack(&wr->sem);
  if (arts_atomic_read(&arts_node_info.shutdown_state) != 0) {
    return; /* teardown escape: a late wake may still post — leak */
  }
  sem_destroy(&wr->sem);
  arts_free(wr);
}

void arts_db_release_rw(struct arts_db_cache_s *cache) {
  {
    /* Nothing is held: a dep woken with a NULL pointer by a destroy never took
     * a count, so there is no publication and no round to run. */
    uint64_t peek =
        atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_LAZY_CACHE_RW(peek) != MSI_RW_GRANT ||
        MSI_LAZY_CACHE_WC(peek) == 0u) {
      return;
    }
  }
  /* The version fetch_add is the publication atom and comes FIRST: the round
   * this release asks for must snapshot the roster AFTER it, which is the one
   * order the soundness argument rests on.  It is also what a hand-off carries
   * as the receiving slot's install stamp. */
  {
    arts_shared_ptr_t h = arts_db_buf_acquire(cache);
    struct arts_db_buffer_s *buf =
        (struct arts_db_buffer_s *)arts_shared_get(h);
    if (buf != NULL) {
      (void)__atomic_add_fetch(&buf->version, 1u, __ATOMIC_ACQ_REL);
      arts_db_buf_release(&h);
    }
  }
  msi_lazy_await_round(cache);
  /* The count is dropped only once the round has closed.  This is an
   * execution-layer contract, not a protocol requirement: a release that has
   * not returned still holds its dep's descriptor pin, and handing ownership on
   * before it returns would split "released" from "no longer held". */
  unsigned int target = MSI_OWNER_NOBODY;
  struct arts_rdzv_landing_s rdzv = {0, 0, 0, 0};
  uint32_t act = MSI_LAZY_CACHE_ACT_NONE;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_LAZY_CACHE_MTP(cur) != 0u) {
      /* Only read the off-word pair when the bit that publishes it is set:
       * with the bit clear they are dead and their writer is unsynchronised. */
      target =
          atomic_load_explicit(&cache->migrate_target, memory_order_acquire);
      rdzv = cache->migrate_rdzv;
    }
    next =
        msi_lazy_cache_compute_next(cur, MSI_LAZY_CACHE_OP_REL_RW, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_LAZY_CACHE_ACT_MIGRATE) {
    msi_lazy_ship(cache, target, &rdzv, next);
  }
}

/* arts_db_release_ro: the shared no-op body in coherence.c applies — a read
 * release touches no protocol state (a valid copy persists across it, and the
 * EDT's buffer ref is dropped by the dep-release path). */

/* ===== arts_handler_db_destroy =========================================
 * OOO_DB_DESTROY Cat-B body.  Fan-out DESTROY_NOTIFY to every rank that ever
 * touched this DB (readers, the owner chain, queued writers), then detach the
 * route-table slot. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
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
  /* No rw_waiters drain: the request FIFO is single-consumer (the migration
   * claim's committer), and a legitimately destroyed DB has no outstanding
   * acquires — draining here would be a second, unsynchronized consumer. */
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* ===== home directory: migration chain ==================================
 * The claim that set `moving` elects exactly one forwarder, so the FIFO front
 * it reads is stable until the matching CONFIRM pops it.  Peek, never pop: the
 * entry must survive the whole round trip so the CONFIRM can assert against it,
 * and popping earlier is what lets a concurrent claim forward an entry to the
 * rank that is about to own it. */
static void msi_home_migrate_publish(struct arts_db_s *db) {
  unsigned int front = 0;
  struct arts_rdzv_landing_s rdzv = {0, 0, 0, 0};
  while (!arts_home_lockreq_queue_peek(&db->rw_waiters, &front, &rdzv)) {
    /* The push precedes the counting CAS, so a claim always has an entry; a
     * producer caught mid-link resolves within its own instruction window. */
  }
  unsigned int owner = MSI_LAZY_DIR_OWNER(
      atomic_load_explicit(&db->dir_state, memory_order_acquire));
  if (front == owner) {
    /* The owner's own entry is popped by the CONFIRM that made it the owner,
     * so it can never still be the front when a claim forwards. */
    ARTS_ERROR("arts msi: migration target %u is already the owner", front);
  }
  arts_send_db_msi_fwdm(owner, db->cache.db_guid, front, &rdzv);
}

/* ===== home directory: invalidation rounds ==============================
 * One round per RW release.  At most one is open per DB (the dir word's
 * round_open bit); requests that arrive while one runs wait in the queue and
 * the closer re-arms.  A round closes exactly ONE request: coalescing would be
 * sound only if the snapshot provably followed every coalesced release's
 * version bump, which the home cannot know.
 *
 * The snapshot excludes nobody — not the owner, not this rank.  The roster is
 * an over-approximation of who might hold a copy, and the one target that must
 * not act on its INVALIDATE (the owner, whose bit was earned in its reader era)
 * is disarmed by the arbiter instead, which costs one harmless ack and keeps
 * the snapshot honest.
 *
 * The whole engine is a trampoline.  A close can re-open, and on this rank the
 * INVALIDATE/ack legs run inline, so a close reached from inside the engine
 * must not re-enter it: the running frame's loop picks the work up instead.
 * A chain stays within one DB (nothing an INVALIDATE or a close does reaches
 * another), so one thread-local marker is enough to detect that. */

static _Thread_local struct arts_db_s *tl_round_frame;

static void msi_home_round_engine(struct arts_db_s *db);

/* Every rank of the 14-bit rank space fits this many bitset words, so a
 * snapshot never needs a heap list (which would have to be published before
 * the first INVALIDATE departs). */
#define MSI_LAZY_ROSTER_WORDS_MAX 256u

static struct arts_db_msi_round_req_s *
msi_home_round_take(struct arts_db_s *db) {
  for (;;) {
    uint32_t n = atomic_load_explicit(&db->round_pending, memory_order_acquire);
    if (n == 0u) {
      return NULL;
    }
    if (atomic_compare_exchange_weak_explicit(&db->round_pending, &n, n - 1u,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      break;
    }
  }
  arts_lf_link_t *node;
  while ((node = arts_mpsc_pop(&db->round_q)) == NULL) {
    /* claimed by the count above: a node exists, its producer is mid-link */
  }
  return ARTS_CONTAINER_OF(node, struct arts_db_msi_round_req_s, link);
}

/* Drop the claim and answer the request this round was opened for.  The request
 * is consumed BEFORE the claim drops: the next round republishes the slot the
 * instant round_open clears. */
static void msi_home_round_close(struct arts_db_s *db) {
  struct arts_db_msi_round_req_s *req = atomic_exchange_explicit(
      &db->cur_round, (struct arts_db_msi_round_req_s *)NULL,
      memory_order_acq_rel);
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    next =
        msi_lazy_dir_compute_next(cur, MSI_LAZY_DIR_OP_ROUND_CLOSE, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  if (req != NULL) {
    arts_send_db_msi_round_done(req->rank, db->cache.db_guid, req->cv);
    arts_free(req);
  }
  msi_home_round_engine(db);
}

static void msi_home_round_engine(struct arts_db_s *db) {
  if (tl_round_frame == db) {
    return; /* re-entrant: the running frame's loop takes the work */
  }
  tl_round_frame = db;
  unsigned int self = arts_global_rank_id;
  /* Store-load barrier for the two-sided handshake this loop closes: a producer
   * raises the queue count and then reads the claim, while a closer drops the
   * claim and then reads the count.  Both sides run a read-modify-write before
   * their load, but neither RMW forbids the load from being satisfied ahead of
   * it — and if BOTH loads went stale ("a round is open" / "nothing queued")
   * the queued request would never be opened and its releaser would never
   * wake.  The fence is what makes at least one side observe the other. */
  atomic_thread_fence(memory_order_seq_cst);
  for (;;) {
    if (atomic_load_explicit(&db->round_pending, memory_order_acquire) == 0u) {
      break;
    }
    uint32_t act;
    uint64_t cur, next;
    bool claimed = true;
    do {
      cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
      next = msi_lazy_dir_compute_next(cur, MSI_LAZY_DIR_OP_ROUND_CLAIM, 0u,
                                       &act);
      if (next == cur) {
        claimed = false; /* a round is already open; its closer re-arms */
        break;
      }
    } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                    memory_order_acq_rel,
                                                    memory_order_acquire));
    if (!claimed) {
      break;
    }
    struct arts_db_msi_round_req_s *req = msi_home_round_take(db);
    if (req == NULL) {
      /* The claim raced past work another opener already consumed: drop it
       * without a snapshot, a multicast or an answer. */
      do {
        cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
        next = msi_lazy_dir_compute_next(cur, MSI_LAZY_DIR_OP_ROUND_CLOSE, 0u,
                                         &act);
      } while (!atomic_compare_exchange_weak_explicit(
          &db->dir_state, &cur, next, memory_order_acq_rel,
          memory_order_acquire));
      continue;
    }
    /* Publish what the close must answer BEFORE arming the count: an ack can
     * reach zero the instant the count is armed. */
    atomic_store_explicit(&db->cur_round, req, memory_order_release);
    /* Snapshot the roster (per-word exchange: registrations racing this land
     * in the NEXT round, which is exactly the case the redirect covers). */
    uint64_t snap[MSI_LAZY_ROSTER_WORDS_MAX];
    const struct arts_rank_bitset_s *bs = &db->roster;
    if (bs->nwords > MSI_LAZY_ROSTER_WORDS_MAX) {
      ARTS_ERROR("arts msi: roster wider than the rank space");
    }
    unsigned int nt = 0;
    for (unsigned int wi = 0; wi < bs->nwords; wi++) {
      snap[wi] = atomic_exchange_explicit((_Atomic uint64_t *)&bs->words[wi],
                                          0u, memory_order_acq_rel);
      nt += (unsigned int)__builtin_popcountll(snap[wi]);
    }
    do {
      cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
      next = msi_lazy_dir_compute_next(cur, MSI_LAZY_DIR_OP_ACKS_ARM, nt, &act);
    } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                    memory_order_acq_rel,
                                                    memory_order_acquire));
    /* The local target goes LAST: its invalidate and its ack run inline, and
     * that chain can close the round, so every remote one must be on the wire
     * before it starts. */
    bool self_target = false;
    for (unsigned int wi = 0; wi < bs->nwords; wi++) {
      uint64_t bits = snap[wi];
      while (bits) {
        unsigned int b = (unsigned int)__builtin_ctzll(bits);
        bits &= bits - 1;
        unsigned int rank = wi * 64u + b;
        if (rank == self) {
          self_target = true;
          continue;
        }
        arts_send_db_msi_invalidate(rank, db->cache.db_guid);
      }
    }
    if (self_target) {
      /* Straight to the body on the descriptor this frame already pins: going
       * back through the route table would let a concurrent detach turn this
       * leg into a MISS, and a MISS whose synthesised ack cannot find the
       * directory either would leave the round short one ack forever. */
      arts_handler_db_msi_invalidate(db);
    }
    if (nt > 0u) {
      /* The ack 0-edge closes.  If the inline local chain already did, the
       * loop's next pass picks up whatever else is queued. */
      continue;
    }
    msi_home_round_close(db);
  }
  tl_round_frame = NULL;
}

/* ===== home handler bodies ============================================== */

void arts_handler_db_msi_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_msi_request_s *a =
      (struct arts_ooo_args_db_msi_request_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  (void)arts_rank_bitset_set(&db->cached_ranks, a->requester);
  if (a->rdzv.txid == 0 && cache->db_size != 0 &&
      a->requester != arts_global_rank_id) {
    /* First touch: the request carried no landing (db_size unknown at the
     * requester).  Nothing was registered or queued — answer the size and let
     * the requester re-issue with one. */
    arts_send_db_msi_cts(a->requester, cache->db_guid, cache->db_size, a->mode);
    return;
  }
  if (a->mode == DB_MODE_RO) {
    /* Register BEFORE reading the owner, then redirect unconditionally — no
     * branch, no gate, no queue.  That order is the whole reason the roster
     * over-approximates safely: any copy this redirect produces is necessarily
     * a target of the next round, because the bit was set before the directory
     * could even name who would mint it. */
    (void)arts_rank_bitset_set(&db->roster, a->requester);
    unsigned int owner = MSI_LAZY_DIR_OWNER(
        atomic_load_explicit(&db->dir_state, memory_order_acquire));
    arts_send_db_msi_redir(owner, cache->db_guid, a->requester, &a->rdzv);
    return;
  }
  /* Write request: the FIFO push precedes the counting CAS, so the claim that
   * CAS elects always finds the entry it counted. */
  arts_home_lockreq_queue_push(&db->rw_waiters, a->requester, &a->rdzv);
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    next = msi_lazy_dir_compute_next(cur, MSI_LAZY_DIR_OP_REQ_RW, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_LAZY_DIR_ACT_MIGRATE_CLAIM) {
    msi_home_migrate_publish(db);
  }
}

void arts_handler_db_msi_round_req(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_msi_round_req_s *a =
      (struct arts_ooo_args_db_msi_round_req_s *)args_v;
  struct arts_db_msi_round_req_s *r =
      (struct arts_db_msi_round_req_s *)arts_malloc(sizeof(*r));
  r->rank = a->rank;
  r->cv = a->cv;
  arts_mpsc_push(&db->round_q, &r->link);
  /* The count is raised only once the push has completed, so a claimant that
   * sees it can always reach the node (a pop that catches a mid-link producer
   * simply retries). */
  (void)atomic_fetch_add_explicit(&db->round_pending, 1u, memory_order_acq_rel);
  msi_home_round_engine(db);
}

void arts_handler_db_msi_confirm(struct arts_db_s *db, unsigned int new_owner) {
  /* The FIFO pop CANNOT join the directory CAS (the queue is off-word), and it
   * must precede it: publish the flip first and a concurrent claim sees an
   * entry that is about to be popped — and forwards the migration to the rank
   * that just became the owner.  Once, outside the retry loop: inside it, a
   * lost CAS would pop twice. */
  unsigned int front = 0;
  struct arts_rdzv_landing_s ignored = {0, 0, 0, 0};
  while (!arts_home_lockreq_queue_pop(&db->rw_waiters, &front, &ignored)) {
    /* the confirming rank's own entry is queued; a mid-link producer resolves
     * within its instruction window */
  }
  if (front != new_owner) {
    ARTS_ERROR("arts msi: confirm from rank %u but FIFO front is %u", new_owner,
               front);
  }
  unsigned int prev_owner = MSI_OWNER_NOBODY;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    if (MSI_LAZY_DIR_W(cur) == 0u || MSI_LAZY_DIR_MOVING(cur) == 0u) {
      ARTS_ERROR("arts msi: confirm with w=%u moving=%u", MSI_LAZY_DIR_W(cur),
                 MSI_LAZY_DIR_MOVING(cur));
    }
    prev_owner = MSI_LAZY_DIR_OWNER(cur);
    next = msi_lazy_dir_compute_next(cur, MSI_LAZY_DIR_OP_CONFIRM_FLIP,
                                     new_owner, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  (void)arts_rank_bitset_set(&db->cached_ranks, new_owner);
  /* Credit the ex-owner's retained copy back into the roster BEFORE the ack
   * opens this rank's store gate: a copy that outlives the hand-off must be a
   * target of the very first round the new owner can trigger. */
  if (prev_owner != new_owner && prev_owner != MSI_OWNER_NOBODY) {
    (void)arts_rank_bitset_set(&db->roster, prev_owner);
    (void)arts_rank_bitset_set(&db->cached_ranks, prev_owner);
  }
  if (act == MSI_LAZY_DIR_ACT_MIGRATE_CLAIM) {
    msi_home_migrate_publish(db); /* chain the next migration */
  }
  arts_send_db_msi_confirm_ack(new_owner, db->cache.db_guid);
}

void arts_handler_db_msi_invalidate_ack(struct arts_db_s *db,
                                        unsigned int sharer_rank) {
  (void)sharer_rank;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    if (MSI_LAZY_DIR_ACKS(cur) == 0u) {
      ARTS_ERROR("arts msi: invalidate ack with no outstanding acks");
    }
    next = msi_lazy_dir_compute_next(cur, MSI_LAZY_DIR_OP_ACK_DEC, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_LAZY_DIR_ACT_CLOSE) {
    msi_home_round_close(db);
  }
}

/* ===== owner-side bodies ================================================ */

/* The serve decision is one pure read of the cache word with two outcomes:
 * this rank holds bytes and serves them live, or it holds none and hands the
 * request back to the home.  Whether a local writer is running is NOT part of
 * it — withholding the copy in that case would move the correctness burden onto
 * the read side, which is the one thing this protocol does not do. */
void arts_handler_db_msi_redir(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_db_msi_redir_args_s *a =
      (struct arts_db_msi_redir_args_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  uint32_t act;
  uint64_t cur =
      atomic_load_explicit(&cache->cache_state, memory_order_acquire);
  (void)msi_lazy_cache_compute_next(cur, MSI_LAZY_CACHE_OP_SERVE_DECIDE, 0u,
                                    &act);
  if (act == MSI_LAZY_CACHE_ACT_BOUNCE) {
    arts_send_db_msi_ro_request(cache->db_guid, a->requester, &a->rdzv);
    return;
  }
  /* The install-lane stamp comes out of the very buffer being served, so a slot
   * swap racing this serve can never pair old bytes with a newer stamp or the
   * reverse.  It travels with the payload for one purpose only: ordering the
   * receiver's two install lanes. */
  arts_shared_ptr_t src_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  uint64_t stamp =
      (src != NULL) ? __atomic_load_n(&src->version, __ATOMIC_ACQUIRE) : 0u;
  uint64_t data_size = (src != NULL) ? cache->db_size : 0u;
  arts_send_db_msi_deliver(a->requester, cache->db_guid, stamp, &a->rdzv, src_h,
                           data_size);
}

void arts_handler_db_msi_fwdm(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_db_msi_fwdm_args_s *a = (struct arts_db_msi_fwdm_args_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  /* Target and landing are published BEFORE the bit that arms them; with the
   * bit clear they are dead, so this store needs no coordination beyond the
   * home's single-migration claim.  Reversed, a hand-off could fire with no
   * published target. */
  cache->migrate_rdzv = a->rdzv;
  atomic_store_explicit(&cache->migrate_target, a->target,
                        memory_order_release);
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_LAZY_CACHE_MTP(cur) != 0u) {
      ARTS_ERROR("arts msi: second migration order at one owner");
    }
    next = msi_lazy_cache_compute_next(cur, MSI_LAZY_CACHE_OP_FWDM, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  /* Arming and shipping are one atom, so an order that finds the owner already
   * idle needs no second event to move the copy.  This CAS is the one that
   * armed the pair, so the order's own fields are the published ones — no
   * re-read. */
  if (act == MSI_LAZY_CACHE_ACT_MIGRATE) {
    msi_lazy_ship(cache, a->target, &a->rdzv, next);
  }
}

/* ===== requester-side bodies ============================================ */

/* A read reply.  The bytes are installed (stamp-conditionally) BEFORE the CAS
 * that publishes the state, so whoever observes a valid copy provably observes
 * its data; the stamp decides only which of the two asynchronous install lanes
 * wins the slot, and a stale one retreats and recycles.  Nothing here compares
 * it to judge validity — the state alone does that. */
static void msi_lazy_deliver_commit(arts_shared_ptr_t db_h, uint64_t stamp,
                                    struct arts_db_buffer_s *landing,
                                    uint64_t data_size) {
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  struct arts_db_cache_s *cache = &db->cache;
  if (landing != NULL) {
    (void)arts_db_buf_install_landed(cache, stamp, landing, data_size);
  }
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_LAZY_CACHE_INFLIGHT(cur) == 0u) {
      ARTS_ERROR("arts msi: read reply with no outstanding fetch");
    }
    if (MSI_LAZY_CACHE_RO(cur) == MSI_RO_VALID &&
        MSI_LAZY_CACHE_HEAD_RO(cur) != 0u) {
      ARTS_ERROR("arts msi: readers parked on a valid copy");
    }
    next = msi_lazy_cache_compute_next(cur, MSI_LAZY_CACHE_OP_DELIVER, 0u,
                                       &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  switch (act) {
  case MSI_LAZY_CACHE_ACT_PUBLISH:
    msi_serve_chain(cache, MSI_LAZY_CACHE_HEAD_RO(cur), /*serialized=*/false);
    break;
  case MSI_LAZY_CACHE_ACT_PUBLISH_KILL:
    /* Serve first, THEN pay the ack this rank has owed since the invalidate
     * marked the fetch: while it is unpaid the round that marked it cannot
     * close, so nothing newer can complete before the doomed serve is done. */
    msi_serve_chain(cache, MSI_LAZY_CACHE_HEAD_RO(cur), /*serialized=*/false);
    arts_send_db_msi_invalidate_ack(arts_guid_get_rank(cache->db_guid),
                                    cache->db_guid);
    break;
  case MSI_LAZY_CACHE_ACT_DROP_REFETCH:
    /* The same CAS reopened the fetch for the readers that chained behind an
     * already-orphaned reply. */
    arts_send_db_msi_request(cache, DB_MODE_RO);
    break;
  default: /* DROP: an install or a purge already retired this fetch */
    break;
  }
  arts_shared_release(&db_h);
}

struct msi_lazy_landed_ctx_s {
  arts_shared_ptr_t db_h;
  struct arts_db_buffer_s *landing;
  uint64_t version;
  uint64_t data_size;
  bool is_rw;
};

static void msi_lazy_deliver_rw_commit(arts_shared_ptr_t db_h, uint64_t base,
                                       struct arts_db_buffer_s *landing,
                                       uint64_t data_size);

static void msi_lazy_landed_cb(void *arg) {
  struct msi_lazy_landed_ctx_s *ctx = (struct msi_lazy_landed_ctx_s *)arg;
  if (arts_shared_get(ctx->db_h) == NULL) {
    /* Destroyed while the pairing was outstanding (destroy during a pending
     * acquire is app UB): the cache's recycle pool is gone — return the
     * landing's storage straight to the registered pool. */
    arts_regpool_free(ctx->landing);
    arts_shared_release(&ctx->db_h);
    arts_free(ctx);
    return;
  }
  if (ctx->is_rw) {
    msi_lazy_deliver_rw_commit(ctx->db_h, ctx->version, ctx->landing,
                               ctx->data_size);
  } else {
    msi_lazy_deliver_commit(ctx->db_h, ctx->version, ctx->landing,
                            ctx->data_size);
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
    struct msi_lazy_landed_ctx_s *ctx =
        (struct msi_lazy_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->landing = (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie;
    ctx->version = p->version;
    ctx->data_size = p->data_size;
    ctx->is_rw = false;
    arts_net_rdzv_expect(p->rdzv_txid, msi_lazy_landed_cb, ctx);
    return;
  }
  /* Data-less reply (nothing published, or a serve that resolved back onto
   * this rank): the advertised landing went unused. */
  if (p->rdzv_cookie != 0) {
    arts_db_buf_landing_recycle(
        &db->cache, (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie);
  }
  msi_lazy_deliver_commit(db_h, p->version, NULL, 0u);
}

/* Ownership arrives.  The buffer swap precedes the publishing CAS (a rank that
 * observes ownership must observe the bytes), and that CAS grabs the read chain
 * and serves it IMMEDIATELY — a reader must not wait for the directory flip.
 * The writers stay queued behind the confirm gate: storing before the flip is
 * published would let a reader registered afterwards be redirected to the
 * previous owner's retained (older) copy and install it durably.
 * Taking the read plane to VALID here is what stops an orphan read reply from
 * putting older bytes over the canonical copy; the fetch flag is deliberately
 * left alone, since that reply is still coming. */
static void msi_lazy_deliver_rw_commit(arts_shared_ptr_t db_h, uint64_t base,
                                       struct arts_db_buffer_s *landing,
                                       uint64_t data_size) {
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  struct arts_db_cache_s *cache = &db->cache;
  if (landing != NULL) {
    /* The migrated canonical buffer always wins the slot: at most one migration
     * is in flight globally and its base is the highest published version, so
     * no read reply can carry more.  The install stays stamp-conditional
     * anyway — equal stamps mean equal bytes. */
    arts_shared_ptr_t slot_h = arts_db_buf_acquire(cache);
    struct arts_db_buffer_s *slot =
        (struct arts_db_buffer_s *)arts_shared_get(slot_h);
    if (slot != NULL) {
      uint64_t have = __atomic_load_n(&slot->version, __ATOMIC_ACQUIRE);
      if (base < have) {
        ARTS_ERROR("arts msi: ownership install is stale (base %llu < %llu)",
                   (unsigned long long)base, (unsigned long long)have);
      }
      arts_db_buf_release(&slot_h);
    }
    (void)arts_db_buf_install_landed(cache, base, landing, data_size);
  }
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_LAZY_CACHE_RW(cur) != MSI_RW_REQ) {
      ARTS_ERROR("arts msi: ownership delivered without a request (rw=%u)",
                 MSI_LAZY_CACHE_RW(cur));
    }
    next = msi_lazy_cache_compute_next(cur, MSI_LAZY_CACHE_OP_DELIVER_RW, 0u,
                                       &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  bool owed = (MSI_LAZY_CACHE_RO(cur) == MSI_RO_REQ_KILL);
  msi_serve_chain(cache, MSI_LAZY_CACHE_HEAD_RO(cur), /*serialized=*/false);
  if (owed) {
    /* The open fetch this rank owed an ack for is resolved by owning the copy
     * outright; pay it after the serve, as on the read path. */
    arts_send_db_msi_invalidate_ack(arts_guid_get_rank(cache->db_guid),
                                    cache->db_guid);
  }
  arts_send_db_msi_confirm(arts_guid_get_rank(cache->db_guid), cache->db_guid);
  arts_shared_release(&db_h);
}

void arts_handler_db_msi_deliver_rw(void *payload, size_t size) {
  (void)size;
  struct arts_msg_msi_deliver_rw_packet_s *p =
      (struct arts_msg_msi_deliver_rw_packet_s *)payload;
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_db_rdzv_discard_landing(p->rdzv.txid, p->rdzv.cookie);
    arts_shared_release(&db_h);
    return;
  }
  if (p->rdzv.txid != 0) {
    struct msi_lazy_landed_ctx_s *ctx =
        (struct msi_lazy_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->landing = (struct arts_db_buffer_s *)(uintptr_t)p->rdzv.cookie;
    ctx->version = p->version;
    ctx->data_size = p->data_size;
    ctx->is_rw = true;
    arts_net_rdzv_expect(p->rdzv.txid, msi_lazy_landed_cb, ctx);
    return;
  }
  if (p->rdzv.cookie != 0) {
    arts_db_buf_landing_recycle(
        &db->cache, (struct arts_db_buffer_s *)(uintptr_t)p->rdzv.cookie);
  }
  msi_lazy_deliver_rw_commit(db_h, p->version, NULL, 0u);
}

/* The directory flip is published: this rank may store, its queued writers may
 * run, and a migration order that arrived during the gate may finally be
 * honoured. */
void arts_handler_db_msi_confirm_ack(struct arts_db_s *db) {
  struct arts_db_cache_s *cache = &db->cache;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    next = msi_lazy_cache_compute_next(cur, MSI_LAZY_CACHE_OP_CONFIRM_ACK, 0u,
                                       &act);
    if (next == cur) {
      break; /* no gate standing: nothing to open, nobody to admit */
    }
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_LAZY_CACHE_ACT_UNGATE) {
    msi_serve_chain(cache, MSI_LAZY_CACHE_HEAD_RW(cur), /*serialized=*/true);
  }
  /* Unconditional: a migration order that arrived while the gate was shut has
   * no other event left to drive it. */
  msi_lazy_try_migrate(cache);
}

void arts_handler_db_msi_invalidate(struct arts_db_s *db) {
  struct arts_db_cache_s *cache = &db->cache;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (MSI_LAZY_CACHE_RW(cur) != MSI_RW_GRANT &&
        MSI_LAZY_CACHE_RO(cur) == MSI_RO_REQ_KILL) {
      /* Unreachable: the ack this rank owes keeps the round that marked the
       * fetch open, and rounds are mutually exclusive, so no second one can
       * target it. */
      ARTS_ERROR("arts msi: second invalidate on a kill-marked fetch");
    }
    next = msi_lazy_cache_compute_next(cur, MSI_LAZY_CACHE_OP_INV, 0u, &act);
    if (next == cur) {
      break; /* ownership supersedes it, or it is idempotent here */
    }
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act != MSI_LAZY_CACHE_ACT_KILL_OWED) {
    arts_send_db_msi_invalidate_ack(arts_guid_get_rank(cache->db_guid),
                                    cache->db_guid);
  }
}

void arts_handler_db_msi_cts(struct arts_db_s *db,
                             struct arts_msg_msi_cts_packet_s *p) {
  struct arts_db_cache_s *cache = &db->cache;
  if (cache->db_size == 0) {
    cache->db_size = p->db_size;
  }
  arts_send_db_msi_request(cache, (arts_db_access_mode_t)p->mode);
}

/* ===== senders ==========================================================
 * A self-send omits the wire, never a protocol step: the deferrable message
 * re-enters through the OoO engine (so a before-create reorder defers exactly
 * as the wire path would) and the direct ones mirror the dispatcher's
 * ref-pinned lookup, miss-action included. */

static struct arts_db_s *msi_lazy_pin(arts_guid_t guid,
                                      arts_shared_ptr_t *out_h) {
  *out_h = arts_route_table_lookup_db(guid);
  return (struct arts_db_s *)arts_shared_get(*out_h);
}

void arts_send_db_msi_cts(unsigned int requester_rank, arts_guid_t db_guid,
                          uint64_t db_size, arts_db_access_mode_t mode) {
  struct arts_msg_msi_cts_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_CTS);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.db_size = db_size;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  if (requester_rank == arts_global_rank_id) {
    arts_shared_ptr_t h;
    struct arts_db_s *db = msi_lazy_pin(db_guid, &h);
    if (db != NULL) {
      arts_handler_db_msi_cts(db, &p);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_invalidate(unsigned int sharer_rank,
                                 arts_guid_t db_guid) {
  if (sharer_rank == arts_global_rank_id) {
    /* MISS must still ack: the round's close is gated on every roster member
     * answering, and a torn-down cache is exactly as invalidated as a purged
     * one. */
    arts_shared_ptr_t h;
    struct arts_db_s *db = msi_lazy_pin(db_guid, &h);
    if (db != NULL) {
      arts_handler_db_msi_invalidate(db);
      arts_shared_release(&h);
      return;
    }
    arts_shared_release(&h);
    arts_send_db_msi_invalidate_ack(arts_guid_get_rank(db_guid), db_guid);
    return;
  }
  struct arts_msg_msi_invalidate_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_INVALIDATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)sharer_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_invalidate_ack(unsigned int home_rank,
                                     arts_guid_t db_guid) {
  if (home_rank == arts_global_rank_id) {
    arts_shared_ptr_t h;
    struct arts_db_s *db = msi_lazy_pin(db_guid, &h);
    if (db != NULL) {
      arts_handler_db_msi_invalidate_ack(db, arts_global_rank_id);
    }
    arts_shared_release(&h);
    return;
  }
  struct arts_msg_msi_invalidate_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_INVALIDATE_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* A FRESH landing per request: a read reply and an ownership deliver can be in
 * flight to one rank together, and each must land in its own buffer (installed
 * or recycled at commit) — that overlap is exactly what the install stamp
 * arbitrates.  txid == 0 = first touch (db_size unknown): the home answers with
 * the size and the request re-issues. */
static void msi_lazy_fetch_landing(struct arts_db_cache_s *cache,
                                   struct arts_rdzv_landing_s *out) {
  *out = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  if (cache->db_size != 0 && arts_global_rank_count > 1) {
    (void)arts_db_buf_landing_alloc(cache, cache->db_size, out);
  }
}

void arts_send_db_msi_request(struct arts_db_cache_s *cache,
                              arts_db_access_mode_t mode) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  struct arts_rdzv_landing_s rdzv;
  msi_lazy_fetch_landing(cache, &rdzv);
  if (home_rank == arts_global_rank_id) {
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
  p.requester = arts_global_rank_id;
  p.rdzv.addr = rdzv.addr;
  p.rdzv.key = rdzv.key;
  p.rdzv.txid = rdzv.txid;
  p.rdzv.cookie = rdzv.cookie;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* The bounce: a redirect target holding no bytes re-sends the read request to
 * the home ON THE READER'S BEHALF, carrying the reader's own landing.  It is a
 * retry, not a park — nothing is queued and nothing sleeps — and it terminates
 * because each pass re-resolves against a later ownership generation. */
void arts_send_db_msi_ro_request(arts_guid_t db_guid, unsigned int requester,
                                 const struct arts_rdzv_landing_s *rdzv) {
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  struct arts_rdzv_landing_s land =
      (rdzv != NULL) ? *rdzv : (struct arts_rdzv_landing_s){0, 0, 0, 0};
  if (home_rank == arts_global_rank_id) {
    struct arts_ooo_args_db_msi_request_s args = {
        .requester = requester,
        .db_guid = db_guid,
        .mode = DB_MODE_RO,
        .rdzv = land,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_MSI_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  struct arts_msg_msi_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)DB_MODE_RO;
  p.requester = requester;
  p.rdzv.addr = land.addr;
  p.rdzv.key = land.key;
  p.rdzv.txid = land.txid;
  p.rdzv.cookie = land.cookie;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_redir(unsigned int owner_rank, arts_guid_t db_guid,
                            unsigned int requester,
                            const struct arts_rdzv_landing_s *rdzv) {
  struct arts_db_msi_redir_args_s args = {
      .db_guid = db_guid,
      .requester = requester,
      .rdzv = (rdzv != NULL) ? *rdzv : (struct arts_rdzv_landing_s){0, 0, 0, 0},
  };
  if (owner_rank == arts_global_rank_id) {
    arts_shared_ptr_t h;
    struct arts_db_s *db = msi_lazy_pin(db_guid, &h);
    if (db != NULL) {
      arts_handler_db_msi_redir(db, &args);
      arts_shared_release(&h);
      return;
    }
    arts_shared_release(&h);
    /* MISS: this rank holds nothing to serve from, which is precisely the
     * bounce case — hand the request back so the directory re-resolves it. */
    arts_send_db_msi_ro_request(db_guid, requester, &args.rdzv);
    return;
  }
  struct arts_msg_msi_redir_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_REDIR);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.requester = requester;
  p.pad = 0;
  p.rdzv.addr = args.rdzv.addr;
  p.rdzv.key = args.rdzv.key;
  p.rdzv.txid = args.rdzv.txid;
  p.rdzv.cookie = args.rdzv.cookie;
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_fwdm(unsigned int owner_rank, arts_guid_t db_guid,
                           unsigned int target,
                           const struct arts_rdzv_landing_s *rdzv) {
  struct arts_db_msi_fwdm_args_s args = {
      .db_guid = db_guid,
      .target = target,
      .rdzv = (rdzv != NULL) ? *rdzv : (struct arts_rdzv_landing_s){0, 0, 0, 0},
  };
  if (owner_rank == arts_global_rank_id) {
    arts_shared_ptr_t h;
    struct arts_db_s *db = msi_lazy_pin(db_guid, &h);
    if (db != NULL) {
      arts_handler_db_msi_fwdm(db, &args);
    }
    /* MISS: the copy this order was to move no longer exists on this rank, so
     * there is nothing to ship and nothing that could stand in for it. */
    arts_shared_release(&h);
    return;
  }
  struct arts_msg_msi_fwdm_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_FWDM);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.target = target;
  p.pad = 0;
  p.rdzv.addr = args.rdzv.addr;
  p.rdzv.key = args.rdzv.key;
  p.rdzv.txid = args.rdzv.txid;
  p.rdzv.cookie = args.rdzv.cookie;
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_deliver(unsigned int requester_rank, arts_guid_t db_guid,
                              uint64_t version,
                              const struct arts_rdzv_landing_s *rdzv,
                              arts_shared_ptr_t src_h, uint64_t data_size) {
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  struct arts_msg_msi_deliver_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_DELIVER);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.data_size = 0;
  p.rdzv_txid = 0;
  p.rdzv_cookie = (rdzv != NULL) ? rdzv->cookie : 0;
  if (requester_rank == arts_global_rank_id || src == NULL || rdzv == NULL ||
      rdzv->txid == 0 || data_size == 0) {
    /* Nothing moves: the requester either IS the server (its own buffer is
     * already the source) or there is no payload to carry.  The advertised
     * landing goes back to the pool inside the handler. */
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    if (requester_rank == arts_global_rank_id) {
      arts_handler_db_msi_deliver(&p, sizeof(p));
      return;
    }
    arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
    return;
  }
  /* One-sided serve straight out of the served buffer; the strong ref moves to
   * the PUT's local completion, pinning those bytes until the fabric has
   * drained them. */
  p.data_size = data_size;
  p.rdzv_txid = rdzv->txid;
  arts_net_put_payload((int)requester_rank, rdzv->addr, rdzv->key, rdzv->txid,
                       src->data, data_size, arts_db_buf_ref_release_cb,
                       (void *)src_h);
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_deliver_rw(unsigned int target_rank, arts_guid_t db_guid,
                                 uint64_t version,
                                 const struct arts_rdzv_landing_s *rdzv,
                                 arts_shared_ptr_t src_h, uint64_t data_size) {
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  struct arts_msg_msi_deliver_rw_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_DELIVER_RW);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.data_size = 0;
  p.rdzv.addr = 0;
  p.rdzv.key = 0;
  p.rdzv.txid = 0;
  p.rdzv.cookie = (rdzv != NULL) ? rdzv->cookie : 0;
  if (target_rank == arts_global_rank_id || src == NULL || rdzv == NULL ||
      rdzv->txid == 0 || data_size == 0) {
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    if (target_rank == arts_global_rank_id) {
      arts_handler_db_msi_deliver_rw(&p, sizeof(p));
      return;
    }
    arts_transport_send_async((int)target_rank, (char *)&p, sizeof(p));
    return;
  }
  p.data_size = data_size;
  p.rdzv.addr = rdzv->addr;
  p.rdzv.key = rdzv->key;
  p.rdzv.txid = rdzv->txid;
  arts_net_put_payload((int)target_rank, rdzv->addr, rdzv->key, rdzv->txid,
                       src->data, data_size, arts_db_buf_ref_release_cb,
                       (void *)src_h);
  arts_transport_send_async((int)target_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_confirm(unsigned int home_rank, arts_guid_t db_guid) {
  if (home_rank == arts_global_rank_id) {
    arts_shared_ptr_t h;
    struct arts_db_s *db = msi_lazy_pin(db_guid, &h);
    if (db != NULL) {
      arts_handler_db_msi_confirm(db, arts_global_rank_id);
    }
    /* MISS: the directory this would advance is gone. */
    arts_shared_release(&h);
    return;
  }
  struct arts_msg_msi_confirm_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_CONFIRM);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_confirm_ack(unsigned int owner_rank,
                                  arts_guid_t db_guid) {
  if (owner_rank == arts_global_rank_id) {
    arts_shared_ptr_t h;
    struct arts_db_s *db = msi_lazy_pin(db_guid, &h);
    if (db != NULL) {
      arts_handler_db_msi_confirm_ack(db);
    }
    /* MISS: the gate it would open no longer exists. */
    arts_shared_release(&h);
    return;
  }
  struct arts_msg_msi_confirm_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_CONFIRM_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

/* A home-local release calls the home body DIRECTLY rather than deferring
 * through the OoO engine: a destroy that has already detached the slot would
 * defer the request forever, and the releaser is blocked on its answer. */
void arts_send_db_msi_round_req(arts_guid_t db_guid, uint64_t cv) {
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  if (home_rank == arts_global_rank_id) {
    struct arts_ooo_args_db_msi_round_req_s args = {
        .db_guid = db_guid,
        .rank = arts_global_rank_id,
        .cv = cv,
    };
    arts_shared_ptr_t h;
    struct arts_db_s *db = msi_lazy_pin(db_guid, &h);
    if (db != NULL) {
      arts_handler_db_msi_round_req(db, &args);
      arts_shared_release(&h);
      return;
    }
    arts_shared_release(&h);
    arts_send_db_msi_round_done(arts_global_rank_id, db_guid, cv);
    return;
  }
  struct arts_msg_msi_round_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_ROUND_REQ);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.cv = cv;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_msi_round_done(unsigned int releaser_rank,
                                 arts_guid_t db_guid, uint64_t cv) {
  if (releaser_rank == arts_global_rank_id) {
    /* Home-local releaser: wake by pointer identity, no wire and no lookup —
     * the wake must never depend on the cache still existing. */
    if (cv != 0) {
      sem_post((sem_t *)(uintptr_t)cv);
    }
    return;
  }
  struct arts_msg_msi_round_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_MSI_ROUND_DONE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.cv = cv;
  arts_transport_send_async((int)releaser_rank, (char *)&p, sizeof(p));
}
