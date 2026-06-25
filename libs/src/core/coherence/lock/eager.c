/* SPDX-License-Identifier: Apache-2.0
 *
 * LOCK protocol EAGER timing translation unit.
 *
 * Defines: lock_compute_next, cache_compute_next,
 *          lock_home_grant (static), arts_send_db_lock_grant,
 *          arts_handler_db_lock_request, arts_handler_db_lock_release,
 *          arts_db_acquire_is_serialized, arts_db_cache_init,
 *          arts_db_cache_destructor, arts_db_create_publish_holder,
 *          arts_db_create_install_home_buffer,
 *          arts_send_db_lock_request, lock_drain_pending (static),
 *          arts_handler_db_acquire, arts_handler_db_lock_grant,
 *          arts_db_release_rw, arts_db_release_ro,
 *          lock_send_release_rw (static), lock_send_release_ro (static),
 *          arts_send_db_lock_release.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=LOCK + ARTS_PROTOCOL_TIMING=EAGER.
 * Contains NO timing preprocessor guards — the timing variant is selected at
 * link time by CMakeLists.txt (home.c eager.c vs home.c lazy.c).
 */

/* lock/types.h must precede all other coherence headers: it defines
 * arts_db_cache_s, arts_db_s, LOCK_HELD_*, arts_db_lock_waiter_s, and the
 * arts_home_lockreq_queue_s for the LOCK build (coherence.h and handlers.h
 * declare functions that take these types by pointer). */
#include "arts/coherence/lock/types.h"

#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
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
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/identity.h"
#include "arts/system/threads.h"
#include "arts/transport/outbox.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== RO waiter node (rank-granular RO queue entry) ===================
 * Mirrors the definition in home.c (both TUs include lock/types.h which does
 * not define this node — it is a file-local type used only by home-side grant
 * and teardown logic in home.c, and by arts_send_db_lock_grant here which
 * calls arts_handler_db_lock_grant as a local-hit self-send). */
struct arts_lock_ro_node_s {
  arts_lf_link_t link; /* FIRST */
  unsigned int rank;
};

/* ===== Pure state-transition arbiters ==================================
 * lock_compute_next and cache_compute_next are pure functions (no external
 * calls, no global state).  They live in arbiters.c so the unit test
 * (tests/unit/lock_compute_next.c) can #include just the arbiters without
 * pulling in the full handler bodies. */
#include "arbiters.c"

/* ===== Grant dispatcher (shared by request + release handlers) ========= */

/* Acquire the home buffer's data for a grant payload and fan-out grants.
 * After pop (rw) / drain (ro), for each target rank: self → direct handler
 * call (local hit, wire 0) / remote → MSG_DB_LOCK_GRANT. */
static void lock_home_grant(struct arts_db_s *db, struct arts_db_cache_s *cache,
                            uint32_t grant) {
  if (grant == LOCK_GRANT_NONE) {
    return;
  }
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  const void *data = buf ? buf->data : NULL;
  uint64_t data_size = buf ? cache->db_size : 0;
  /* The grant carries the home buffer's own version.  No separate counter is
   * needed: home commits (RW writeback installs) are sequentialized by the
   * protocol (RW single-owner inter-node + ACK-gated release), so buf->version
   * is monotone across rounds.  The requester's buf_install guard
   * (old.version >= new_version → reject) uses this version to accept a fresh
   * grant and discard a stale duplicate. */
  uint64_t version = buf ? buf->version : 0;
  if (grant == LOCK_GRANT_ONE_RW) {
    unsigned int rank;
    if (arts_home_lockreq_queue_pop(&db->rw_waiters, &rank)) {
      arts_send_db_lock_grant(rank, cache->db_guid, DB_MODE_RW, version, data,
                              data_size);
    }
  } else { /* LOCK_GRANT_ALL_RO: drain ro_waiters, fan-out to each rank. */
    arts_lf_link_t *node = arts_lf_stack_drain(&db->ro_waiters);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_lock_ro_node_s *rn =
          ARTS_CONTAINER_OF(node, struct arts_lock_ro_node_s, link);
      arts_send_db_lock_grant(rn->rank, cache->db_guid, DB_MODE_RO, version,
                              data, data_size);
      arts_free(rn);
      node = nx;
    }
  }
  arts_db_buf_release(&buf_h);
}

/* ===== arts_send_db_lock_grant ========================================= */

void arts_send_db_lock_grant(unsigned int requester_rank, arts_guid_t db_guid,
                             arts_db_access_mode_t mode, uint64_t version,
                             const void *data, uint64_t data_size) {
  struct arts_msg_lock_grant_packet_s p;
  uint64_t total = sizeof(p) + data_size;
  arts_fill_packet_header(&p.header, total, MSG_DB_LOCK_GRANT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.version = version;
  if (requester_rank == arts_global_rank_id) {
    /* Self-send (home == requester): build the contiguous buffer the handler
     * expects (header + data) and call the body directly (local hit, wire 0).
     */
    char *buf = (char *)arts_malloc(total);
    memcpy(buf, &p, sizeof(p));
    if (data_size > 0 && data != NULL) {
      memcpy(buf + sizeof(p), data, data_size);
    }
    arts_handler_db_lock_grant(buf, (size_t)total);
    arts_free(buf);
    return;
  }
  if (data == NULL || data_size == 0) {
    arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
    return;
  }
  /* Assemble header + payload into a single contiguous allocation and send
   * in one call.  arts_transport_send_payload_async stores only the payload
   * pointer, which points into the home buffer (buf->data).  That buffer can
   * be freed or overwritten by a concurrent writeback before the sender thread
   * reads it, producing a dangling-pointer read and stale data on the wire.
   * Copying into a fresh allocation here makes the transport packet
   * self-contained and owner-independent of the home buffer lifetime. */
  char *pkt = (char *)arts_malloc((size_t)total);
  memcpy(pkt, &p, sizeof(p));
  memcpy(pkt + sizeof(p), data, (size_t)data_size);
  arts_transport_send_async((int)requester_rank, pkt, (unsigned int)total);
  arts_free(pkt);
}

/* ===== arts_handler_db_lock_request ==================================== */

void arts_handler_db_lock_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_ooo_args_db_lock_request_s *a =
      (struct arts_ooo_args_db_lock_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)a->mode;

  /* (1) push-before-CAS: enqueue this requester rank in its mode's queue
   * BEFORE reading lock_state, so the CAS transition already sees this
   * participant counted. */
  if (mode == DB_MODE_RW) {
    arts_home_lockreq_queue_push(&db->rw_waiters, requester);
  } else {
    struct arts_lock_ro_node_s *n =
        (struct arts_lock_ro_node_s *)arts_malloc(sizeof(*n));
    n->rank = requester;
    arts_lf_stack_push(&db->ro_waiters, &n->link);
  }
  arts_rank_bitset_set(&db->cached_ranks,
                       requester); /* destroy fan-out roster */

  /* (2) read -> compute next -> CAS retry on contention. */
  int op = (mode == DB_MODE_RW) ? LOCK_OP_RW_ACQ : LOCK_OP_RO_ACQ;
  uint32_t grant;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    next = lock_compute_next(cur, op, &grant);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_acq_rel, memory_order_acquire));

  /* (3) grant per the transition case. */
  lock_home_grant(db, cache, grant);
}

/* ===== arts_handler_db_lock_release ==================================== */

void arts_handler_db_lock_release(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_ooo_args_db_lock_release_s *a =
      (struct arts_ooo_args_db_lock_release_s *)args_v;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)a->mode;

  /* (1) RW: write the writeback into home's stable buffer in place, then ACK.
   * Under exclusive-lock serialization the releaser held the sole RW grant and
   * home grants the next holder only after this writeback completes, so no
   * reader is touching the buffer here — the in-place overwrite is safe and the
   * buffer address stays fixed (preserving DBs with internal self-pointers).
   * No versioning: home commits are already sequentialized by the protocol (RW
   * single-owner + ACK-gated release). */
  if (mode == DB_MODE_RW && a->data_size > 0) {
    const void *data = (const char *)a + sizeof(*a);
    arts_db_buf_write_inplace(cache, data, a->data_size);
  }
  if (mode == DB_MODE_RW && a->cv != 0) {
    arts_send_db_lock_release_ack(a->releaser, a->db_guid, a->cv);
  }

  /* (2) transition CAS. */
  int op = (mode == DB_MODE_RW) ? LOCK_OP_RW_REL : LOCK_OP_RO_REL;
  uint32_t grant;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    next = lock_compute_next(cur, op, &grant);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_release, memory_order_acquire));

  /* (3) grant the next holder(s). */
  lock_home_grant(db, cache, grant);
}

/* ===== arts_db_acquire_is_serialized ===================================
 * LOCK: both RW and RO are blocking locks — both are GUID-serialized so
 * the engine acquires them in a global order (deadlock-free lock ordering). */
bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW || mode == DB_MODE_RO;
}

/* ===== arts_db_cache_init ==============================================
 * Initialize the per-rank cache for the LOCK protocol.  For LOCK, the home-rank
 * lock_state and the per-cache cache_state single word are the coherence state;
 * arts_db_cache_common_init handles the shared fields (db_guid, db_size,
 * pending_snapshot, home-directory init).
 *
 * Creator hold: arts_db_create defaults to an RW acquire (OCR contract; only
 * ARTS_DB_PROP_NO_ACQUIRE skips it), so the creator EDT must hold the lock RW
 * exactly like any granted writer — otherwise its matching arts_db_release (or
 * the EDT-epilogue auto-release) would run against a hold that was never taken.
 * For the creator/home kinds we therefore SEED the cache held in RW
 * (rw_state=GRANT, rw_count=1); the home lock_state is seeded w=1 in
 * arts_db_home_init.  The release path drives both back to 0 — the standard
 * acquire/release pair, no protocol-specific create bookkeeping.  (NO_ACQUIRE
 * resets these to idle in arts_db_create, mirroring the single-owner arms.) */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  uint64_t seed = 0ULL; /* rw_state=ro_state=IDLE, both counts 0 */
  if (kind == ARTS_DB_INIT_CREATOR_HOME ||
      kind == ARTS_DB_INIT_CREATOR_REMOTE) {
    /* Creator holds RW: GRANT + one writer. */
    seed = CACHE_MAKE(CACHE_ST_GRANT, CACHE_ST_IDLE, 1u, 0u);
  }
  atomic_store_explicit(&c->cache_state, seed, memory_order_relaxed);
  arts_lf_stack_init(&c->ro_pending);
  arts_lf_stack_init(&c->rw_pending);
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

/* ===== arts_db_cache_destructor =========================================
 * free parked nodes only — destroying an in-use DB is undefined (OCR), no
 * wake.
 *
 * Destructor runs at refcount 0 as the SOLE owner of the object; no concurrent
 * actor remains, so it is the single safe drainer of the parked-waiter stacks.
 * Destroy itself is just the route-slot detach (CAS value→NULL + drop the
 * install ref). */
void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache); /* buffer-NULL FIRST */
  /* Drain + free the parked-waiter stacks (empty in the normal case — a legit
   * destroy has no outstanding acquirers). */
  for (arts_lf_stack_t *q = &cache->ro_pending;; q = &cache->rw_pending) {
    arts_lf_link_t *node = arts_lf_stack_drain(q);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_db_lock_waiter_s *w =
          ARTS_CONTAINER_OF(node, struct arts_db_lock_waiter_s, link);
      arts_free(w);
      node = nx;
    }
    if (q == &cache->rw_pending) {
      break;
    }
  }
  arts_db_cache_common_destroy_post(cache); /* snapshot free → home teardown */
}

/* ===== arts_db_create_publish_holder ====================================
 * LOCK home init: nothing to publish for the holder field — LOCK tracks
 * mode via lock_state, not rw_holder.  The creator becomes the first RW
 * holder via the normal acquire/self-grant chain, so there is nothing to
 * publish at create. */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  (void)db;
  (void)creator_rank;
}

/* ===== arts_db_create_install_home_buffer ================================
 * LOCK home init: install the zero-init home buffer at creation time so the
 * home always holds the canonical backing store.  The first GRANT carries
 * this data (empty / zero at first) to the requester; the requester's first
 * RW release sends back the updated contents via LOCK_RELEASE writeback. */
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  if (db_size > 0) {
    arts_db_buf_write_inplace(cache, /*data=*/NULL, db_size); /* zero-init */
  }
}

/* ===== arts_send_db_lock_request ========================================
 * Send MSG_DB_LOCK_REQUEST to the home rank.  Self-send (home == this rank)
 * dispatches through the OoO engine (HIT runs inline; MISS defers until the
 * home db_s is installed).  Remote send goes via the transport. */
void arts_send_db_lock_request(unsigned int home_rank, arts_guid_t db_guid,
                               arts_db_access_mode_t mode) {
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine so before-create reorders are
     * handled correctly (the engine defers when the slot is absent). */
    struct arts_ooo_args_db_lock_request_s args = {
        .requester = arts_global_rank_id,
        .db_guid = db_guid,
        .mode = (uint32_t)mode,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_LOCK_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  struct arts_msg_lock_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_LOCK_REQUEST);
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* Serve every waiter currently in `q`: a single atomic-exchange drain claims
 * the whole stack, so each node is taken by exactly one drainer (any node a
 * drainer "misses" is taken by another).  Serving = re-derive dep->ptr +
 * cursor-advance + account (mark_edt_secured + mark_edt_ready); it does NOT
 * touch the count (the count was bumped at the waiter's acquire cas1, before
 * it was pushed — "queued ⟹ counted"). */
static void lock_drain_pending(arts_lf_stack_t *q) {
  arts_lf_link_t *node = arts_lf_stack_drain(q);
  while (node != NULL) {
    arts_lf_link_t *nx =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    struct arts_db_lock_waiter_s *w =
        ARTS_CONTAINER_OF(node, struct arts_db_lock_waiter_s, link);
    /* Position-idempotent cursor advance (fires next serialized dep), then
     * re-derive dep->ptr from the installed buffer + account (may schedule the
     * EDT when acquire_remaining reaches 0). */
    mark_edt_secured_by_guid(w->edt_guid, w->slot);
    mark_edt_ready_by_guid(w->edt_guid, w->slot);
    arts_free(w);
    node = nx;
  }
}

/* ===== arts_handler_db_acquire =========================================
 * OOO_DB_ACQUIRE Cat-B body — protocol-agnostic signature.
 *
 * item is the pre-pinned home db_s; args is {edt, db_guid, slot}; mode is read
 * from depv[slot].mode.
 *
 * Two-CAS acquire (see the single-word design doc):
 *   cas1  count++ ONLY (fetch_add), BEFORE the push, so a waiter present in a
 *         queue is already counted → a drained waiter cannot be released out
 *         from under by a concurrent holder (no premature grant release).
 *   push  the waiter (now drainable; already counted).
 *   cas2  state decision on the FRESH (post-push) state (no count change) →
 *         no orphaned REQUEST; a waiter served before cas2 just reads GRANT and
 *         DRAINs (idempotent).
 *   act   SEND a REQUEST (cas2 opened IDLE→REQ) or DRAIN (state already GRANT).
 *         NONE = coalesced/covered; the pending grant's drain will serve us. */
void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  arts_db_access_mode_t mode = depv[slot].mode;
  arts_guid_t db_guid = cache->db_guid;

  /* cas1: count++ before push. */
  atomic_fetch_add_explicit(
      &cache->cache_state, (mode == DB_MODE_RW) ? CACHE_RW_UNIT : CACHE_RO_UNIT,
      memory_order_acq_rel);

  /* push: the waiter is now visible to any drainer (and already counted). */
  struct arts_db_lock_waiter_s *w =
      (struct arts_db_lock_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt->guid;
  w->slot = slot;
  arts_lf_stack_push(
      (mode == DB_MODE_RW) ? &cache->rw_pending : &cache->ro_pending, &w->link);

  /* cas2: state decision on fresh state. */
  int op = (mode == DB_MODE_RW) ? CACHE_OP_ACQ_RW : CACHE_OP_ACQ_RO;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    next = cache_compute_next(cur, op, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));

  switch (act) {
  case CACHE_ACT_SEND_RW:
    arts_send_db_lock_request(arts_guid_get_rank(db_guid), db_guid, DB_MODE_RW);
    break;
  case CACHE_ACT_SEND_RO:
    arts_send_db_lock_request(arts_guid_get_rank(db_guid), db_guid, DB_MODE_RO);
    break;
  case CACHE_ACT_DRAIN_RW:
    lock_drain_pending(&cache->rw_pending);
    break;
  case CACHE_ACT_DRAIN_RO:
    lock_drain_pending(&cache->ro_pending);
    break;
  default: /* NONE: parked; a grant / another acquire's drain serves us. */
    break;
  }
}

/* ===== arts_handler_db_lock_grant =======================================
 * Cat-C pure body — grant arrived at the requester rank.  payload is the full
 * contiguous wire buffer (header + db_size data bytes); size is the byte count.
 *
 *   (1) Cat-C route-table lookup (NULL ⇒ DB destroyed concurrently ⇒ drop).
 *   (2) Install the grant data BEFORE the CAS so drained waiters observe it.
 *   (3) CAS the single word: RW grant Q→GRANT (action DRAIN_BOTH); RO grant
 *       Q→GRANT (DRAIN_RO) or, when rc==0 (the RO waiters were already served
 *       by an RW grant — RW⊇RO), a phantom Q→IDLE that is returned to home at
 *       once (REL_RO).
 *   (4) Run the action after the CAS commits (mirrors the home grant path:
 *       compute decides, the pop/drain/send runs after).
 *
 * Grant is one drainer among several (the acquire DRAIN action drains too); the
 * whole-stack atomic-exchange makes serving idempotent and exactly-once. */
void arts_handler_db_lock_grant(void *payload, size_t size) {
  struct arts_msg_lock_grant_packet_s *p =
      (struct arts_msg_lock_grant_packet_s *)payload;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)p->mode;
  const void *data = (const char *)p + sizeof(*p);
  uint64_t data_size = (uint64_t)size - (uint64_t)sizeof(*p);

  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_shared_release(&db_h);
    return;
  }
  struct arts_db_cache_s *cache = &db->cache;

  /* (2) Write grant data into the cache's stable buffer in place.  Exclusive-
   * lock serialization means no EDT on this rank holds the buffer when the
   * grant lands, and a duplicate grant carries identical bytes, so the in-place
   * overwrite is safe and idempotent — no versioning needed.  Keeping the
   * buffer address fixed preserves DBs that hold internal self-pointers. */
  if (data_size > 0u) {
    arts_db_buf_write_inplace(cache, data, data_size);
  }

  /* (3) CAS the state word. */
  int op = (mode == DB_MODE_RW) ? CACHE_OP_GRANT_RW : CACHE_OP_GRANT_RO;
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    next = cache_compute_next(cur, op, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));

  /* (4) Action after the CAS. */
  switch (act) {
  case CACHE_ACT_DRAIN_BOTH: /* RW grant serves this rank's RW + RO cohort */
    lock_drain_pending(&cache->rw_pending);
    lock_drain_pending(&cache->ro_pending);
    break;
  case CACHE_ACT_DRAIN_RO:
    lock_drain_pending(&cache->ro_pending);
    break;
  case CACHE_ACT_REL_RO: /* phantom RO grant: nothing to serve, return to home
                          */
    arts_send_db_lock_release(arts_guid_get_rank(p->db_guid), p->db_guid,
                              DB_MODE_RO, /*version=*/0u, /*cv=*/0u, NULL, 0u);
    break;
  default:
    break;
  }

  arts_shared_release(&db_h);
}

/* ===== arts_send_db_lock_release ========================================
 * Send LOCK_RELEASE to the home.  RW carries writeback data, version, and cv
 * (the releaser's stack-local sem_t address so home can echo it in the ACK);
 * RO carries none and passes version=0 / cv=0.
 * Self-send (home == this rank) routes through the OoO engine so reordering
 * against DB_CREATE is handled identically to the wire path. */
void arts_send_db_lock_release(unsigned int home_rank, arts_guid_t db_guid,
                               arts_db_access_mode_t mode, uint64_t version,
                               uint64_t cv, const void *data,
                               uint64_t data_size) {
  uint64_t ds = (mode == DB_MODE_RW && data != NULL) ? data_size : 0u;

  if (home_rank == arts_global_rank_id) {
    /* Self-send: build the contiguous args buffer (header + inline data) and
     * route through the OoO engine exactly as the wire RX dispatcher does.
     * HIT runs arts_handler_db_lock_release inline (which posts cv); MISS
     * defers the args until the home db_s is installed. */
    uint32_t asz =
        (uint32_t)(sizeof(struct arts_ooo_args_db_lock_release_s) + ds);
    char *abuf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_lock_release_s *args =
        (struct arts_ooo_args_db_lock_release_s *)abuf;
    args->releaser = arts_global_rank_id;
    args->db_guid = db_guid;
    args->mode = (uint32_t)mode;
    args->data_size = ds;
    args->cv = cv;
    args->version = version;
    if (ds > 0u) {
      memcpy(abuf + sizeof(*args), data, (size_t)ds);
    }
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_LOCK_RELEASE, abuf, asz);
    arts_free(abuf);
    return;
  }

  /* Remote send: build the wire packet header + optional inline payload. */
  struct arts_msg_lock_release_packet_s p;
  uint64_t total = sizeof(p) + ds;
  arts_fill_packet_header(&p.header, total, MSG_DB_LOCK_RELEASE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.version = version;
  p.cv = cv;
  if (ds == 0u) {
    arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
    return;
  }
  /* Assemble header + payload into a single contiguous allocation.
   * arts_transport_send_payload_async stores only the payload pointer, which
   * points into the caller's buffer (buf->data).  That buffer is released by
   * lock_send_release_rw immediately after this call, which can free or
   * reuse the data before the sender thread reads it — producing stale bytes
   * on the wire.  Copying into a fresh allocation makes the packet
   * self-contained and independent of the caller's buffer lifetime. */
  char *pkt = (char *)arts_malloc((size_t)total);
  memcpy(pkt, &p, sizeof(p));
  memcpy(pkt + sizeof(p), data, (size_t)ds);
  arts_transport_send_async((int)home_rank, pkt, (unsigned int)total);
  arts_free(pkt);
}

/* ===== release-edge senders ============================================
 * Called AFTER the release CAS has already moved the state word to IDLE for
 * this phase (so for a self-send, the inline grant handler that follows will
 * CAS the next phase onto an already-IDLE word — no overwrite).
 *
 * RW → synchronous writeback: ship buf->data with a stack-local sem_t cv
 *      token; home echoes it in LOCK_RELEASE_ACK and await_writeback_ack
 *      returns only after the post (no lost update across TCP; self-send posts
 *      inline and returns at once).
 * RO → data-less fire-and-forget notify. */
static void lock_send_release_rw(struct arts_db_cache_s *cache) {
  unsigned int home = (unsigned int)arts_guid_get_rank(cache->db_guid);
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  const void *data = (buf != NULL) ? buf->data : NULL;
  uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
  sem_t cv;
  sem_init(&cv, 0, 0);
  uint64_t cv_token = (uint64_t)(uintptr_t)&cv;
  /* version=0: home is the sole version authority (used only for home's
   * buf_install; the requester does not set it). */
  arts_send_db_lock_release(home, cache->db_guid, DB_MODE_RW, /*version=*/0u,
                            cv_token, data, ds);
  await_writeback_ack(&cv);
  sem_destroy(&cv);
  arts_db_buf_release(&buf_h);
}

static void lock_send_release_ro(struct arts_db_cache_s *cache) {
  unsigned int home = (unsigned int)arts_guid_get_rank(cache->db_guid);
  arts_send_db_lock_release(home, cache->db_guid, DB_MODE_RO, /*version=*/0u,
                            /*cv=*/0u, NULL, 0u);
}

/* ===== arts_db_release_rw / arts_db_release_ro =========================
 * Single-word release: CAS the count down (+ the 0-edge state transition,
 * atomic together) via cache_compute_next, then run the release action it
 * returns.  CACHE_ACT_REL_RW → writeback (the last holder of an RW grant, incl.
 * the last RO joiner under it); CACHE_ACT_REL_RO → notify (last RO holder).
 *
 * Destroyed-guard: a dep NULL-woken at destroy never really held the lock; a
 * count of 0 means there is nothing to release — skip rather than underflow.
 * (Full destroy reconciliation under the acquire-time count is a separate
 * subtask.) */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_RW_CNT(cur) == 0u) {
      return; /* destroyed-guard / already released */
    }
    next = cache_compute_next(cur, CACHE_OP_REL_RW, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_REL_RW) {
    lock_send_release_rw(cache);
  }
}

void arts_db_release_ro(struct arts_db_cache_s *cache) {
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_RO_CNT(cur) == 0u) {
      return; /* destroyed-guard / already released */
    }
    next = cache_compute_next(cur, CACHE_OP_REL_RO, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_REL_RW) {
    lock_send_release_rw(
        cache); /* last RO joiner under an RW grant → writeback */
  } else if (act == CACHE_ACT_REL_RO) {
    lock_send_release_ro(cache);
  }
}
