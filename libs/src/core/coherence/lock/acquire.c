/* SPDX-License-Identifier: Apache-2.0
 *
 * LOCK protocol acquire translation unit.
 *
 * Defines: arts_db_acquire_is_serialized, arts_handler_db_acquire,
 *          arts_send_db_lock_request, arts_handler_db_lock_grant,
 *          arts_db_cache_init, arts_db_cache_destructor,
 *          arts_db_create_publish_holder, arts_db_create_install_home_buffer.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=LOCK (selected by
 * libs/src/core/CMakeLists.txt).  Contains NO timing preprocessor guards.
 */

/* lock/types.h must precede all other coherence headers: it defines
 * arts_db_cache_s, arts_db_s, LOCK_HELD_*, arts_db_lock_waiter_s, and the
 * arts_home_lockreq_queue_s for the LOCK build (coherence.h and handlers.h
 * declare functions that take these types by pointer). */
#include "arts/coherence/lock/types.h"

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
#include "arts/system/threads.h"
#include "arts/transport/outbox.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

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

/* ===== cache_compute_next ==============================================
 * Cache-side analogue of the home lock_compute_next: pure function of the
 * current cache_state word, run inside a CAS-retry loop.  Returns the next
 * word and (via out_action) what the caller must do AFTER the CAS commits.
 *
 * ACQ_* ops decide request/join state ONLY — the count was bumped by the
 * separate fetch_add cas1 before the push.  REL_* ops carry the count-- inside
 * the CAS, since the decrement and the 0-edge state change must be atomic.
 * See docs/superpowers/specs/2026-06-20-lock-cache-single-word-design.md. */
uint64_t cache_compute_next(uint64_t cur, int op, uint32_t *out_action) {
  uint32_t rws = CACHE_RW_ST(cur);
  uint32_t ros = CACHE_RO_ST(cur);
  uint32_t wc = CACHE_RW_CNT(cur);
  uint32_t rc = CACHE_RO_CNT(cur);
  uint32_t act = CACHE_ACT_NONE;
  switch (op) {
  case CACHE_OP_ACQ_RW:
    if (rws == CACHE_ST_IDLE) {
      rws = CACHE_ST_REQ;
      act =
          CACHE_ACT_SEND_RW; /* first writer on this rank: request from home */
    } else if (rws == CACHE_ST_REQ) {
      act = CACHE_ACT_NONE; /* coalesce onto the in-flight RW request */
    } else {                /* GRANT: join the held RW phase */
      act = CACHE_ACT_DRAIN_RW;
    }
    break;
  case CACHE_OP_ACQ_RO:
    if (rws == CACHE_ST_GRANT || ros == CACHE_ST_GRANT) {
      act = CACHE_ACT_DRAIN_RO; /* RW⊇RO local join, or RO phase held */
    } else if (rws == CACHE_ST_IDLE && ros == CACHE_ST_IDLE) {
      ros = CACHE_ST_REQ;
      act = CACHE_ACT_SEND_RO;
    } else {
      act = CACHE_ACT_NONE; /* covered by an in-flight RW/RO request; park */
    }
    break;
  case CACHE_OP_GRANT_RW: /* precond: rws==REQ, ros!=GRANT */
    rws = CACHE_ST_GRANT;
    act = CACHE_ACT_DRAIN_BOTH; /* RW grant serves this rank's RW + RO cohort */
    break;
  case CACHE_OP_GRANT_RO: /* precond: ros==REQ, rws!=GRANT */
    if (rc > 0) {
      ros = CACHE_ST_GRANT;
      act = CACHE_ACT_DRAIN_RO;
    } else {
      /* phantom: the RO waiters were already served by an RW grant (RW⊇RO);
       * this grant has nothing to serve — return it to home at once. */
      ros = CACHE_ST_IDLE;
      act = CACHE_ACT_REL_RO;
    }
    break;
  case CACHE_OP_REL_RW: /* precond: rws==GRANT */
    wc -= 1;
    if (wc == 0 && rc == 0) {
      rws = CACHE_ST_IDLE;
      act = CACHE_ACT_REL_RW; /* last holder of the RW grant → writeback */
    }
    break;
  case CACHE_OP_REL_RO:
    rc -= 1;
    if (rws == CACHE_ST_GRANT && wc == 0 && rc == 0) {
      rws = CACHE_ST_IDLE;
      act = CACHE_ACT_REL_RW; /* last RO joiner under an RW grant → writeback */
    } else if (ros == CACHE_ST_GRANT && rc == 0) {
      ros = CACHE_ST_IDLE;
      act = CACHE_ACT_REL_RO; /* last holder of the RO grant → notify */
    }
    break;
  default:
    break;
  }
  *out_action = act;
  return CACHE_MAKE(rws, ros, wc, rc);
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
