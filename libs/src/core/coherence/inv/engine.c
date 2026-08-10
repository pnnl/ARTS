/* SPDX-License-Identifier: Apache-2.0
 *
 * The MSI engine — everything both placements run identically.
 *
 * The reader plane (waiter pool, chain serve, fetch install) and the
 * invalidation round (claim, roster snapshot, multicast, ack accounting,
 * close) live here, along with the cache lifecycle and the wire senders.  What
 * is left to the placement TUs is exactly three decisions, because those are
 * the only three that "where do the canonical bytes live" actually changes:
 *
 *   - an RO acquire: served locally (HOME, at the home rank) or redirected to
 *     the current grant holder (OWNER);
 *   - a release: publishes the payload to the home (HOME) or publishes control
 *     only (OWNER) — the round itself is identical either way;
 *   - a read request at the home: answered from the home's buffer (HOME) or
 *     forwarded to the holder (OWNER).
 *
 * Write ownership is not here at all: it is the migrating sentinel grant
 * (coherence/grant.c plus its placement half), shared with RCU.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=MSI.
 */
#include "arts/coherence/inv/types.h"

#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/directory.h"
#include "arts/coherence/handlers.h"
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
#include <semaphore.h>
#include <stdio.h>
#include <string.h>
#include "arts/coherence/buffer.h"
#include "arts/coherence/handlers.h"
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
#include "arts/counter/Preamble.h"

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

struct inv_waiter_dir_s {
  _Atomic(struct arts_db_inv_waiter_s *) chunk[MSI_WAITER_NCHUNKS];
};

static struct inv_waiter_dir_s *inv_waiter_dir(struct arts_db_cache_s *c) {
  uintptr_t d = atomic_load_explicit(&c->waiters.chunks, memory_order_acquire);
  if (d == 0) {
    struct inv_waiter_dir_s *fresh =
        (struct inv_waiter_dir_s *)arts_calloc(1, sizeof(*fresh));
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
  return (struct inv_waiter_dir_s *)d;
}

struct arts_db_inv_waiter_s *inv_waiter_ptr(struct arts_db_cache_s *c,
                                                   uint32_t idx) {
  struct inv_waiter_dir_s *dir = inv_waiter_dir(c);
  struct arts_db_inv_waiter_s *chunk = atomic_load_explicit(
      &dir->chunk[idx / MSI_WAITER_CHUNK_CAP], memory_order_acquire);
  return &chunk[idx % MSI_WAITER_CHUNK_CAP];
}

uint32_t inv_waiter_alloc(struct arts_db_cache_s *c) {
  for (;;) {
    uint32_t h =
        atomic_load_explicit(&c->waiters.free_head, memory_order_acquire);
    uint32_t idx = MSI_FREE_IDX(h);
    if (idx == 0u) {
      break; /* free list empty: bump-allocate */
    }
    uint32_t next = inv_waiter_ptr(c, idx)->next;
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
  struct inv_waiter_dir_s *dir = inv_waiter_dir(c);
  _Atomic(struct arts_db_inv_waiter_s *) *slot =
      &dir->chunk[idx / MSI_WAITER_CHUNK_CAP];
  struct arts_db_inv_waiter_s *chunk =
      atomic_load_explicit(slot, memory_order_acquire);
  if (chunk == NULL) {
    struct arts_db_inv_waiter_s *fresh = (struct arts_db_inv_waiter_s *)
        arts_calloc(MSI_WAITER_CHUNK_CAP, sizeof(*fresh));
    struct arts_db_inv_waiter_s *expect = NULL;
    if (!atomic_compare_exchange_strong_explicit(slot, &expect, fresh,
                                                 memory_order_acq_rel,
                                                 memory_order_acquire)) {
      arts_free(fresh);
    }
  }
  return idx;
}

void inv_waiter_free(struct arts_db_cache_s *c, uint32_t idx) {
  struct arts_db_inv_waiter_s *node = inv_waiter_ptr(c, idx);
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
 * duplicate resume, double-driving the next serialized dep, whose writer
 * count is then bumped twice against a single release and never reaches
 * zero.  A non-serialized waiter
 * gets the data-arrival signal only. */
void inv_serve_chain(struct arts_db_cache_s *cache, uint32_t head,
                            bool serialized) {
  while (head != 0u) {
    struct arts_db_inv_waiter_s *w = inv_waiter_ptr(cache, head);
    uint32_t next = w->next;
    arts_guid_t edt_guid = w->edt_guid;
    unsigned int slot = w->slot;
    inv_waiter_free(cache, head);
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
  /* Reader word: IDLE, no fetch open, chain empty.  A creator's copy is a
   * valid read copy — it holds the bytes it is about to write. */
  uint64_t seed = 0ULL;
  bool creator = (kind == ARTS_DB_INIT_CREATOR_HOME ||
                  kind == ARTS_DB_INIT_CREATOR_REMOTE);
  if (creator) {
    seed = MSI_CACHE_MAKE(MSI_RO_VALID, 0u, 0u);
  }
  atomic_store_explicit(&c->cache_state, seed, memory_order_relaxed);
  atomic_store_explicit(&c->waiters.chunks, (uintptr_t)0, memory_order_relaxed);
  atomic_store_explicit(&c->waiters.next_fresh, 1u, memory_order_relaxed);
  atomic_store_explicit(&c->waiters.free_head, 0u, memory_order_relaxed);
  /* Grant plane.  A creator boots holding the grant: the sentinel (+1) plus
   * the create-default RW acquire's own count (+1).  Everyone else starts at
   * zero, which IS "not the owner" — there is no separate ownership bit. */
  c->writer_count = creator ? 2u : 0u;
  arts_pending_rw_queue_init(&c->pending_rw);
  c->grant_req_in_flight = 0u;
  c->grant_unconfirmed = 0u;
  c->incoming_new_owner = ARTS_NO_PENDING_OWNER;
  c->incoming_new_owner_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  /* MSI's sharer plane carries no version ledger by construction; NULL selects
   * the shared transfer helper's empty-map branch. */
  c->cached_version = NULL;
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache); /* buffer-NULL FIRST */
  arts_pending_rw_queue_destroy(&cache->pending_rw);
  /* The reader chain lives in the cache word; with no outstanding acquirers a
   * legitimate destroy leaves the head at 0 (only fetch-open states park), so
   * only the node-pool storage itself is torn down here. */
  arts_db_cache_common_destroy_post(cache);
}

void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  /* The creator boots holding the grant; name it in the home directory. */
  atomic_store_explicit(&db->rw_holder, creator_rank, memory_order_release);
}

void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  /* Metadata-only home until the creator's first publish: a creator-held
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
  /* No pending_rw drain: the grant request queue is single-consumer (the
   * transfer baton's holder), and a legitimately destroyed DB has no
   * outstanding acquires — draining here would be a second, unsynchronized
   * consumer. */
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* ===== the invalidation round ===========================================
 * MSI's whole identity, and the only thing this arm adds to the shared grant.
 *
 * A release does not return until every copy of the data block it just wrote
 * is dead.  The home serializes that with one claim bit: claim -> drain the
 * queued releases -> install the newest payload -> snapshot the roster ->
 * arm the ack count -> multicast INVALIDATE -> close on the ack that reaches
 * zero.  The closer is unique, so it is the actor that wakes the releasers and
 * re-opens for whatever queued up meanwhile.
 *
 * A batch is order-insensitive: at most one install (the highest-version entry
 * that carries data) and a set of commuting wakes.  The roster snapshot is a
 * per-word XCHG, so a rank registered during the round lands in the NEXT
 * one — which is correct, because it will be served the bytes this round
 * installed.  Every producer (a publish push, a close re-arm) calls try_open;
 * a claim that raced past the work it saw simply closes empty.  The engine is
 * a while-trampoline: a close that finds more work loops instead of recursing.
 */

static void inv_home_round_close(struct arts_db_s *db,
                                 struct arts_db_inv_pub_s *entries) {
  struct arts_db_cache_s *cache = &db->cache;
  /* Wake every releaser this round covered, then drop the claim.  The wakes
   * commute and must land BEFORE the claim drops: the next round can open the
   * instant round_open clears. */
  while (entries != NULL) {
    struct arts_db_inv_pub_s *e = entries;
    entries = (struct arts_db_inv_pub_s *)(uintptr_t)atomic_load_explicit(
        &e->link.next, memory_order_relaxed);
    if (e->cv != 0u) {
      arts_send_db_publish_ack(e->releaser_rank, cache->db_guid, e->cv);
    }
    arts_free(e);
  }
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    next = inv_dir_compute_next(cur, MSI_DIR_OP_ROUND_CLOSE, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
}

void inv_home_round_try_open(struct arts_db_s *db) {
  struct arts_db_cache_s *cache = &db->cache;
  for (;;) {
    uint32_t act;
    uint64_t cur, next;
    do {
      cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
      next = inv_dir_compute_next(cur, MSI_DIR_OP_ROUND_CLAIM, 0u, &act);
      if (next == cur) {
        return; /* a round is already open; its closer re-arms */
      }
    } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                    memory_order_acq_rel,
                                                    memory_order_acquire));
    /* Claim held.  Take the whole queued batch (Treiber head XCHG). */
    arts_lf_link_t *chain = arts_lf_stack_drain(&db->pub_queue);
    struct arts_db_inv_pub_s *entries = (struct arts_db_inv_pub_s *)chain;
    /* One round per claim, counted where the claim is won so an empty round
     * (a claim that raced past consumed work) is counted too — it costs the
     * same directory transition. */
    INCREMENT_NUM_INVALIDATE_ROUND_BY(1);
    if (entries == NULL) {
      /* Empty round: the claim raced past work another opener consumed. */
      inv_home_round_close(db, NULL);
    } else {
      /* Single install point: only the batch's highest-version entry that
       * carries data publishes (the conditional install rejects the rest);
       * non-max landings recycle here, and their wakes still fire at close.
       * Under OWNER placement no entry carries data and this loop is a no-op —
       * the round is pure control, which is the only difference between the
       * two placements' releases. */
      struct arts_db_inv_pub_s *maxe = NULL;
      for (struct arts_db_inv_pub_s *e = entries; e != NULL;
           e = (struct arts_db_inv_pub_s *)(uintptr_t)atomic_load_explicit(
               &e->link.next, memory_order_relaxed)) {
        if (e->rdzv.addr != 0 && (maxe == NULL || e->vnew > maxe->vnew)) {
          maxe = e;
        }
      }
      uint64_t maxv = 0;
      for (struct arts_db_inv_pub_s *e = entries; e != NULL;
           e = (struct arts_db_inv_pub_s *)(uintptr_t)atomic_load_explicit(
               &e->link.next, memory_order_relaxed)) {
        if (e->vnew > maxv) {
          maxv = e->vnew;
        }
        if (e->rdzv.addr == 0) {
          continue; /* data-less entry (same-rank release / OWNER placement) */
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
      /* Advance the publication axis over the WHOLE batch, data-less entries
       * included: a same-rank release moves no bytes (the home buffer IS the
       * owner's working buffer) but its round still retires every remote copy,
       * so it is a publication and readers served after it must be given a
       * stamp that says so. */
      if (maxv > atomic_load_explicit(&db->hver, memory_order_acquire)) {
        atomic_store_explicit(&db->hver, maxv, memory_order_release);
      }
      /* First publication wakes the pre-publication read holds (readers
       * event-ordered after the creator's writes must see released bytes,
       * never the never-published zero state). */
      {
        arts_shared_ptr_t pub_h = arts_db_buf_acquire(cache);
        bool pub = (arts_shared_get(pub_h) != NULL);
        arts_db_buf_release(&pub_h);
        if (pub) {
          arts_db_drain_pending_snapshot(cache);
        }
      }
      /* Roster snapshot (per-word XCHG) with the grant holder excluded — it is
       * the writer, its copy is the newest by definition.  The home rank is
       * excluded ONLY where the home's own buffer is the canonical copy: under
       * HOME it is, and this round just installed into it; under OWNER the home
       * holds no canonical bytes, so a reader copy that happens to live on the
       * home rank is as stale as any other and must be invalidated like one.
       * Arm the count BEFORE the multicast. */
      /* Self-exclusion names the ranks that actually WROTE this round — the
       * batch's releasers — not whatever the directory currently points at.
       * The justification for skipping a rank is "its copy is the newest by
       * definition", and that is true of a releaser; `rw_holder` may still
       * name the previous generation, because the home flips it only at the
       * CONFIRM, well after the new owner has installed and (under HOME, which
       * has no confirm gate) already run and released.  Excluding by
       * `rw_holder` therefore skipped exactly the ex-holder that this round
       * exists to retire. */
      unsigned int writers[8];
      unsigned int nwriters = 0;
      for (struct arts_db_inv_pub_s *e = entries; e != NULL;
           e = (struct arts_db_inv_pub_s *)(uintptr_t)atomic_load_explicit(
               &e->link.next, memory_order_relaxed)) {
        bool seen = false;
        for (unsigned int k = 0; k < nwriters; k++) {
          if (writers[k] == e->releaser_rank) {
            seen = true;
            break;
          }
        }
        if (!seen &&
            nwriters < (unsigned int)(sizeof(writers) / sizeof(writers[0]))) {
          writers[nwriters++] = e->releaser_rank;
        }
      }
#ifdef ARTS_WRITE_POLICY_WT
      unsigned int self = arts_global_rank_id;
#else
      unsigned int self = (unsigned int)-1; /* no rank is exempt */
#endif
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
          if (rank == self) {
            continue;
          }
          bool is_writer = false;
          for (unsigned int k = 0; k < nwriters; k++) {
            if (writers[k] == rank) {
              is_writer = true;
              break;
            }
          }
          if (is_writer) {
            continue;
          }
          if (ntargets <
              (unsigned int)(sizeof(targets) / sizeof(targets[0]))) {
            targets[ntargets++] = rank;
          } else {
            /* Rank counts beyond the stack window would need a heap list; the
             * rank field is 14-bit but deployments here are far smaller — fail
             * loudly rather than silently truncate a round. */
            (void)fprintf(stderr, "arts msi: round target overflow\n");
            abort();
          }
        }
      }
      uint32_t aact;
      do {
        cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
        next = inv_dir_compute_next(cur, MSI_DIR_OP_ACKS_ARM, ntargets, &aact);
      } while (!atomic_compare_exchange_weak_explicit(
          &db->dir_state, &cur, next, memory_order_acq_rel,
          memory_order_acquire));
      /* Publish the open round's batch for the closing ack committer BEFORE
       * any INVALIDATE departs. */
      db->round_entries = entries;
      for (unsigned int t = 0; t < ntargets; t++) {
        arts_send_db_inv_invalidate(targets[t], cache->db_guid);
      }
      if (ntargets > 0u) {
        return; /* the ack 0-edge committer closes (and re-arms) */
      }
      db->round_entries = NULL;
      inv_home_round_close(db, entries);
    }
    /* Re-arm: work that arrived while we were closing. */
    if (arts_lf_stack_empty(&db->pub_queue)) {
      return;
    }
  }
}

/* ===== wire handler bodies (OoO Cat-B) ================================== */

/* Commit-leg pairing continuation: the dirty bytes are in the home landing;
 * queue the entry and drive the round. */
struct inv_pub_landed_ctx_s {
  arts_shared_ptr_t db_h;
  arts_guid_t db_guid;
  struct arts_db_inv_pub_s *entry;
};

static void inv_pub_landed_cb(void *arg) {
  struct inv_pub_landed_ctx_s *ctx = (struct inv_pub_landed_ctx_s *)arg;
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(ctx->db_h);
  if (db == NULL) {
    /* Destroyed mid-pairing: free the landing's storage, never strand the
     * blocked releaser. */
    struct arts_db_inv_pub_s *e = ctx->entry;
    if (e->rdzv.cookie != 0) {
      arts_regpool_free((void *)(uintptr_t)e->rdzv.cookie);
    }
    if (e->cv != 0) {
      arts_send_db_publish_ack(e->releaser_rank, ctx->db_guid, e->cv);
    }
    arts_free(e);
    arts_shared_release(&ctx->db_h);
    arts_free(ctx);
    return;
  }
  arts_lf_stack_push(&db->pub_queue, &ctx->entry->link);
  inv_home_round_try_open(db);
  arts_shared_release(&ctx->db_h);
  arts_free(ctx);
}

void arts_handler_db_publish(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_publish_s *a =
      (struct arts_ooo_args_db_publish_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  if (a->data_size != 0 && a->data_inline == 0 && a->rdzv_txid == 0) {
    /* Announce leg: hand the releaser a fresh home landing; nothing queues
     * yet — the commit leg (paired with the PUT completion) does. */
    struct arts_rdzv_landing_s landing;
    (void)arts_db_buf_landing_alloc(cache, a->data_size, &landing);
    arts_send_db_publish_cts(a->releaser, a->db_guid, &landing, a->cv);
    return;
  }
  struct arts_db_inv_pub_s *e =
      (struct arts_db_inv_pub_s *)arts_malloc(sizeof(*e));
  e->vnew = a->version;
  e->releaser_rank = a->releaser;
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
    struct inv_pub_landed_ctx_s *ctx =
        (struct inv_pub_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = arts_route_table_lookup_db(a->db_guid);
    ctx->db_guid = a->db_guid;
    ctx->entry = e;
    arts_net_rdzv_expect(a->rdzv_txid, inv_pub_landed_cb, ctx);
    return;
  }
  arts_lf_stack_push(&db->pub_queue, &e->link);
  inv_home_round_try_open(db);
}

/* ===== Cat-C handler bodies ============================================= */

void arts_handler_db_inv_cts(struct arts_db_s *db,
                             struct arts_msg_inv_cts_packet_s *p) {
  struct arts_db_cache_s *cache = &db->cache;
  if (cache->db_size == 0) {
    cache->db_size = p->db_size;
  }
  arts_send_db_inv_request(cache, (arts_db_access_mode_t)p->mode);
}

/* Publish + whole-chain grab: install the landed payload (home-issued
 * install-lane version; a stale install retreats and recycles), then ONE
 * word CAS takes {REQ->VALID, chain head -> 0} together.  The kill-marked
 * variant serves its cohort once, then executes the reserved purge and the
 * owed ack — the ack LAST, so the round that killed this fetch cannot close
 * (and nothing newer can complete) until the doomed serve is done. */
static void inv_deliver_commit(arts_shared_ptr_t db_h, uint64_t version,
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
    next = inv_cache_compute_next(cur, MSI_CACHE_OP_DELIVER, 0u, &act);
    if (next == cur) {
      break; /* DROP: superseded (an ownership install's version won) */
    }
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_CACHE_ACT_PUBLISH || act == MSI_CACHE_ACT_PUBLISH_KILL) {
    inv_serve_chain(cache, MSI_CACHE_HEAD_RO(cur), /*serialized=*/false);
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
      next = inv_cache_compute_next(cur, MSI_CACHE_OP_KILL_PURGE, 0u, &pact);
    } while (!atomic_compare_exchange_weak_explicit(
        &cache->cache_state, &cur, next, memory_order_acq_rel,
        memory_order_acquire));
    arts_send_db_inv_invalidate_ack(arts_guid_get_rank(cache->db_guid),
                                    cache->db_guid); /* owed ack, LAST */
  }
  arts_shared_release(&db_h);
}

struct inv_landed_ctx_s {
  arts_shared_ptr_t db_h;
  struct arts_db_buffer_s *landing;
  uint64_t version;
  uint64_t data_size;
};


static void inv_landed_cb(void *arg) {
  struct inv_landed_ctx_s *ctx = (struct inv_landed_ctx_s *)arg;
  if (arts_shared_get(ctx->db_h) == NULL) {
    /* Destroyed while the pairing was outstanding (destroy during a pending
     * acquire is app UB): the cache's recycle pool is gone — return the
     * landing's storage straight to the registered pool and drop. */
    arts_regpool_free(ctx->landing);
    arts_shared_release(&ctx->db_h);
    arts_free(ctx);
    return;
  }
  inv_deliver_commit(ctx->db_h, ctx->version, ctx->landing, ctx->data_size);
  arts_free(ctx);
}

void arts_handler_db_inv_deliver(void *payload, size_t size) {
  (void)size;
  struct arts_msg_inv_deliver_packet_s *p =
      (struct arts_msg_inv_deliver_packet_s *)payload;
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_db_rdzv_discard_landing(p->rdzv_txid, p->rdzv_cookie);
    arts_shared_release(&db_h);
    return;
  }
  if (p->rdzv_txid != 0) {
    struct inv_landed_ctx_s *ctx =
        (struct inv_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->landing = (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie;
    ctx->version = p->version;
    ctx->data_size = p->data_size;
    arts_net_rdzv_expect(p->rdzv_txid, inv_landed_cb, ctx);
    return;
  }
  /* Data-less reply: an advertised-but-unused landing recycles. */
  if (p->rdzv_cookie != 0) {
    arts_db_buf_landing_recycle(
        &db->cache, (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie);
  }
  inv_deliver_commit(db_h, p->version, NULL, 0u);
}

void arts_handler_db_inv_invalidate(struct arts_db_s *db) {
  struct arts_db_cache_s *cache = &db->cache;
  unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    /* An invalidate can never hit the grant holder: the round snapshot
     * self-excludes it and rounds serialize. */
    if ((int)arts_atomic_read(&cache->writer_count) > 0) {
      (void)fprintf(stderr, "arts msi: invalidate hit the grant holder\n");
      abort();
    }
    next = inv_cache_compute_next(cur, MSI_CACHE_OP_INVALIDATE, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_CACHE_ACT_PURGE_ACK || act == MSI_CACHE_ACT_NOOP_ACK) {
    arts_send_db_inv_invalidate_ack(home_rank, cache->db_guid);
  }
  /* KILL_MARKED: the ack is owed — the doomed DELIVER fires it after its
   * serve-once, keeping every pre-ack serve legally racy. */
}

void arts_handler_db_inv_invalidate_ack(struct arts_db_s *db,
                                        unsigned int sharer_rank) {
  (void)sharer_rank;
  INCREMENT_NUM_INVALIDATE_ACK_BY(1);
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->dir_state, memory_order_acquire);
    next = inv_dir_compute_next(cur, MSI_DIR_OP_ACK_DEC, 0u, &act);
  } while (!atomic_compare_exchange_weak_explicit(&db->dir_state, &cur, next,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == MSI_DIR_ACT_CLOSE) {
    struct arts_db_inv_pub_s *entries = db->round_entries;
    db->round_entries = NULL;
    inv_home_round_close(db, entries);
    /* Re-arm work that queued while the round was in flight. */
    if (!arts_lf_stack_empty(&db->pub_queue)) {
      inv_home_round_try_open(db);
    }
  }
}

/* ===== senders ========================================================== */
/* Control-only senders are plain packet fills; the payload-carrying trio
 * (deliver/publish) lands with the data-plane implementation. */

void arts_send_db_inv_cts(unsigned int requester_rank, arts_guid_t db_guid,
                          uint64_t db_size, arts_db_access_mode_t mode) {
  struct arts_msg_inv_cts_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_INV_CTS);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.db_size = db_size;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}


/* One message per sharer in the roster snapshot: the multicast whose width is
 * what the round costs. */
void arts_send_db_inv_invalidate(unsigned int sharer_rank,
                                 arts_guid_t db_guid) {
  INCREMENT_NUM_INVALIDATE_SENT_BY(1);
  if (sharer_rank == arts_global_rank_id) {
    /* Local hit: run the retirement inline.  Reachable under OWNER, where the
     * home holds no canonical copy and is therefore an ordinary sharer — its
     * own copy has to be retired like everyone else's.  A MISS still owes the
     * round its ack, which the handler body sends. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_inv_invalidate(db);
    } else {
      arts_send_db_inv_invalidate_ack(arts_guid_get_rank(db_guid), db_guid);
    }
    arts_shared_release(&h);
    return;
  }
  struct arts_msg_inv_invalidate_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_INV_INVALIDATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)sharer_rank, (char *)&p, sizeof(p));
}

void arts_send_db_inv_invalidate_ack(unsigned int home_rank,
                                     arts_guid_t db_guid) {
  if (home_rank == arts_global_rank_id) {
    /* Local hit: account the ack inline.  A MISS drops — the round's other
     * acks still close it, and a destroyed DB has no round to close. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_inv_invalidate_ack(db, arts_global_rank_id);
    }
    arts_shared_release(&h);
    return;
  }
  struct arts_msg_inv_invalidate_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_INV_INVALIDATE_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_inv_request(struct arts_db_cache_s *cache,
                              arts_db_access_mode_t mode) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  struct arts_rdzv_landing_s rdzv = {0, 0, 0, 0};
  /* A FRESH landing per fetch: reception isolation when a read reply and an
   * ownership transfer are in flight together (each PUT lands in its own
   * buffer, installed or recycled at commit).  txid==0 = first touch (db_size
   * unknown): the home answers MSI_CTS and the fetch re-issues.
   *
   * Being the home rank does NOT excuse a requester from advertising one.
   * Under HOME the home answers from its own canonical buffer and the landing
   * simply goes unused; under OWNER the home holds no bytes at all, so its own
   * reads are served by a remote holder and need somewhere to land.  Skipping
   * the allocation there silently produced a data-less reply, leaving the
   * reader on whatever stale copy it already had. */
  if (cache->db_size != 0 && arts_global_rank_count > 1) {
    (void)arts_db_buf_landing_alloc(cache, cache->db_size, &rdzv);
  }
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine so before-create reorders
     * defer exactly like the wire path. */
    struct arts_ooo_args_db_inv_request_s args = {
        .requester = arts_global_rank_id,
        .db_guid = db_guid,
        .mode = mode,
        .rdzv = rdzv,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_INV_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  struct arts_msg_inv_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_INV_REQUEST);
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

void arts_send_db_inv_deliver(unsigned int requester_rank,
                              struct arts_db_s *db,
                              const struct arts_rdzv_landing_s *rdzv) {
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_msg_inv_deliver_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_INV_DELIVER);
  p.header.rank = arts_global_rank_id;
  p.db_guid = cache->db_guid;
  p.data_size = 0;
  p.rdzv_txid = 0;
  p.rdzv_cookie = (rdzv != NULL) ? rdzv->cookie : 0;
  /* Serve from the CURRENT install; the ref pins the bytes across the PUT
   * (released by the fabric's local completion).  The stamp is the served
   * buffer's own version — it exists only to arbitrate the two asynchronous
   * install lanes at the requester, never to decide validity. */
  arts_shared_ptr_t src_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
#ifdef ARTS_WRITE_POLICY_WT
  /* Stamp on the home's publication axis, never the buffer's own lane — see
   * hver.  The stamp exists only to arbitrate the receiver's two asynchronous
   * install lanes; nothing compares it to decide whether a copy is valid. */
  p.version = atomic_load_explicit(&arts_db_of_cache(cache)->hver,
                                   memory_order_acquire);
#else
  /* Owner-canonical: the served buffer's own version IS the axis, because
   * ownership migration carries it from owner to owner as one chain. */
  p.version = (src != NULL) ? arts_atomic_read_u64(&src->version) : 0u;
#endif
  if (requester_rank == arts_global_rank_id) {
    /* Self-serve: the requester's cache IS this cache — the bytes are already
     * where they need to be, so no payload moves and the reply is dispatched
     * inline.  Reachable whenever the server and the reader are the same rank
     * (a single-rank run, or a home that also holds the grant). */
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    if (rdzv != NULL && rdzv->cookie != 0) {
      arts_db_buf_landing_recycle(cache,
                                  (struct arts_db_buffer_s *)(uintptr_t)
                                      rdzv->cookie);
      p.rdzv_cookie = 0;
    }
    arts_handler_db_inv_deliver(&p, sizeof(p));
    return;
  }
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



/* The ex-holder keeps the bytes it wrote, so from the flip onward it is a
 * sharer: the new owner's first release must invalidate it like any other.
 *
 * Under HOME the home rank is exempt — its buffer IS the canonical copy, kept
 * current by every release's publish, so it is never stale and never a target.
 * Under OWNER nothing is exempt: the home holds no canonical bytes, so a copy
 * that happens to sit on the home rank goes stale exactly like any other and
 * must be registered. */
void arts_db_grant_note_ex_holder(struct arts_db_s *db, unsigned int rank) {
#ifdef ARTS_WRITE_POLICY_WT
  if (rank == arts_global_rank_id) {
    return;
  }
#endif
  (void)arts_rank_bitset_set(&db->roster, rank);
}

/* Cat-C pure body (PUBLISH_ACK).  Cache-independent pointer-identity sem post
 * on a->cv (the releaser's heap rendezvous, valid on this rank — the ack always
 * returns to the sender), so item_v is unused.  The dispatcher posts on BOTH a
 * HIT and a MISS, so a torn-down home never strands a blocked releaser. */
void arts_handler_db_publish_ack(void *item_v, void *args_v) {
  (void)item_v;
  struct arts_db_publish_ack_args_s *a =
      (struct arts_db_publish_ack_args_s *)args_v;
  sem_t *s = (sem_t *)(uintptr_t)a->cv;
  if (s != NULL) {
    sem_post(s);
  }
}
