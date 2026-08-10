/* SPDX-License-Identifier: Apache-2.0
 *
 * RWLOCK protocol, OWNER placement.
 *
 * The home holds no bytes — only a directory (current owner, phase, waiter
 * queues).  The payload stays with the last read--write holder and moves
 * owner->owner on demand; a release with nothing pending moves nothing at all.
 * Remote requests all go to the home, while an acquire on the rank that already
 * owns the data is served from its own buffer with no home round-trip.
 *
 * What makes that tractable is the protocol's own phase exclusion.  Read--write
 * and read-only phases never overlap, so an ownership migration (read--write
 * phase only) and a reader serve (read-only phase only) can never be in flight
 * together: the migration-versus-in-transit-reader race that a migrating
 * protocol with a concurrent reader plane must solve cannot arise here.  The
 * price is that a remote read waits out the write phase at the home instead of
 * being served from a second copy — which is what phase exclusion means.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=EXCL + ARTS_RELEASE_POLICY=RETAIN;
 * the placement variant is a link-time file choice, so this TU contains no
 * placement preprocessor guards.
 */

/* lock/types.h must precede all other coherence headers: it defines
 * arts_db_cache_s, arts_db_s, the CACHE_ and LOCK_ macros, the OWNER arbiter
 * prototypes, and arts_db_excl_waiter_s for the RWLOCK build. */
#include "arts/counter/object_counter.h"
#include "arts/coherence/excl/types.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/directory.h"
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/gas/route_table.h"
#include "arts/memory/regpool.h" /* arts_regpool_free (orphaned landing) */
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/identity.h"
#include "arts/system/threads.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/counter/Preamble.h"

/* ===== RO-serve node (reader rank the owner must serve in the RO phase) =
 * The owner's ro_serve is an arts_lf_stack_t (Treiber); each node carries one
 * reader rank that the home asked the owner to serve via FORWARD(serve).  File-
 * local: only the owner-side FORWARD handler (push) and owner_try_execute /
 * the destructor (drain) touch it. */
struct arts_lock_ro_serve_node_s {
  arts_lf_link_t link; /* FIRST — required by arts_lf_stack_t */
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv; /* reader's serve landing (fresh buffer) */
};

/* fetch_add units for the cas1 count bump (the OWNER cache_state has different
 * field shifts than HOME, which defines CACHE_RW_UNIT/CACHE_RO_UNIT itself).
 * Adding these never touches the higher state fields. */
#define CACHE_OWNER_RW_UNIT ((uint64_t)1 << CACHE_OWNER_WC_SHIFT)
#define CACHE_OWNER_RO_UNIT ((uint64_t)1 << CACHE_OWNER_RC_SHIFT)

/* ===== cache_owner_compute_next =========================================
 * OWNER owner-cache arbiter — pure function of the single cache_state word, run
 * inside a CAS-retry loop.  The cache_state packs owner-bit + rw_st/ro_st +
 * migrate_target + wc + rc into one word, so every transition (including the
 * RW-release 0-edge migration decision) is a single CAS.
 *
 * The ACQ ops carry an owner fast-path: when this rank already holds the data
 * and is otherwise eligible, the acquire is granted LOCALLY (DRAIN) with no
 * REQUEST to home.  Eligibility is read entirely from the word — RW needs
 * owner-bit && rc==0 && migrate_target==NO_TARGET, RO needs owner-bit && wc==0.
 * A non-empty ro_serve need not be checked for the RW path: ro_serve is filled
 * only by a home FORWARD(RO), which arrives in the read-only phase, where this
 * rank cannot be holding a write grant.
 *
 * FORWARD-migrate and FORWARD-serve do NOT pass through this arbiter — their
 * target rank is not derivable from the word, so the FORWARD handler installs
 * migrate_target / pushes ro_serve itself.
 *
 * ACQ_* ops decide request/join/local-grant state ONLY; the count was bumped by
 * a separate fetch_add beforehand.  REL_* ops carry the count-- inside the CAS,
 * because the decrement and the 0-edge transition must be atomic together.
 */
uint64_t cache_owner_compute_next(uint64_t cur, int op, uint32_t *out_action) {
  uint32_t own = CACHE_OWNER(cur);
  uint32_t rws = CACHE_RW_ST(cur);
  uint32_t ros = CACHE_RO_ST(cur);
  uint32_t mt = CACHE_MIGRATE_TARGET(cur);
  uint32_t wc = CACHE_RW_CNT(cur);
  uint32_t rc = CACHE_RO_CNT(cur);
  uint32_t act = CACHE_ACT_NONE;
  switch (op) {
  case CACHE_OP_ACQ_RW:
    if (rws == CACHE_ST_GRANT) {
      act = CACHE_ACT_DRAIN_RW; /* join the held RW grant */
    } else if (rws == CACHE_ST_REQ) {
      act = CACHE_ACT_NONE; /* coalesce onto the in-flight RW request */
    } else if (own == 1u) {
      /* owner fast-path: this rank holds the data → grant locally, no home
       * round-trip.  OCR intra-node model: once a node holds the grant, ALL
       * local RW/RO acquires are immediate — there is NO intra-node exclusion
       * (the RWLOCK serializes only inter-node, at the home), and a pending
       * migration does not block local serving.  The migration ships once wc
       * and rc both reach 0 (the release 0-edge). */
      rws = CACHE_ST_GRANT;
      act = CACHE_ACT_DRAIN_RW;
    } else {
      rws = CACHE_ST_REQ;
      act = CACHE_ACT_SEND_RW; /* request migration from home */
    }
    break;
  case CACHE_OP_ACQ_RO:
    /* A write grant covers reads (RW⊇RO), so an in-flight RW request already
     * covers this reader and it must NOT send an RO request of its own: the home
     * would count an r that is later served through the RW grant and therefore
     * never RO_RETURNed, stranding r above zero and deadlocking the RO phase. */
    if (rws == CACHE_ST_GRANT || ros == CACHE_ST_GRANT) {
      act = CACHE_ACT_DRAIN_RO; /* RW⊇RO local join, or RO copy held */
    } else if (own == 1u) {
      /* A local hit keys off the owner-bit ALONE and must not touch ro_st.
       * ro_st tracks this rank's home-visible RO request; promoting it to GRANT
       * here would erase an outstanding REQ, so a later REL_RO would clear the
       * word and a second RO request would go out — double-counting the home's r
       * against a single RO_RETURN.  It would also leave ro_st==GRANT behind
       * after this rank migrates away, and the branch above would then read
       * stale local bytes. */
      act = CACHE_ACT_DRAIN_RO;
    } else if (rws == CACHE_ST_IDLE && ros == CACHE_ST_IDLE) {
      ros = CACHE_ST_REQ;
      act = CACHE_ACT_SEND_RO; /* fresh RO REQUEST to home */
    } else {
      /* rw_st==REQ (covered by the in-flight RW request, served via RW⊇RO) or
       * ro_st==REQ (coalesce onto the in-flight RO request) → park. */
      act = CACHE_ACT_NONE;
    }
    break;
  case CACHE_OP_REL_RW: /* precond: rws==GRANT, own==1 */
    wc -= 1;
    /* The grant ends only when BOTH counts reach zero: readers that joined
     * under it (RW⊇RO) may outlive the last writer, so wc==0 with rc>0 keeps the
     * grant and the migration ships at the rc 0-edge instead. */
    if (wc == 0u && rc == 0u) {
      rws = CACHE_ST_IDLE;
      if (mt != ARTS_LOCK_NO_TARGET) {
        /* 0-edge with a pending migration: clear owner-bit + migrate_target in
         * the SAME next-state (single CAS) and signal the ship.  This is the
         * RWLOCK-OWNER analogue of RCU's 0-edge transfer, but a single CAS — no
         * split-decrement, no separate in-flight flag. */
        own = 0u;
        mt = ARTS_LOCK_NO_TARGET;
        act = CACHE_ACT_MIGRATE;
      } else {
        /* sticky: keep ownership + data, no wire traffic. */
        act = CACHE_ACT_NONE;
      }
    }
    break;
  case CACHE_OP_REL_RO:
    rc -= 1;
    if (rc == 0u) {
      /* These readers held either a standalone RO grant or a write grant they
       * joined (RW⊇RO), so clear whichever is live — the last joiner under a
       * write grant completes that grant. */
      if (rws == CACHE_ST_GRANT && wc == 0u) {
        rws = CACHE_ST_IDLE;
      } else if (ros == CACHE_ST_GRANT) {
        ros = CACHE_ST_IDLE;
      }
      act = CACHE_ACT_REL_RO;
    }
    break;
  default:
    break;
  }
  *out_action = act;
  /* Preserve the ro_granted flag (spare bit 63 — see CACHE_LEASED) across every
   * transition EXCEPT the RO release 0-edge, which fully relinquishes the RO
   * grant and clears it.  The flag is SET by the DELIVER(RO) handler when a
   * lent RO copy is actually received (a borrower owes exactly one RO_RETURN) —
   * NOT at request time, and NEVER for an owner serving its own readers locally
   * (RW⊇RO / fast-path: the owner holds the data, it borrowed nothing).
   * CACHE_MAKE_FULL produces bit63==0, so clearing is the default; we OR the
   * bit back only while the grant is still (partially) held. */
  uint64_t next = CACHE_MAKE_FULL(own, rws, ros, mt, wc, rc);
  if (!(op == CACHE_OP_REL_RO && rc == 0u)) {
    next |= (cur & CACHE_OWNER_LEASED_MASK);
  }
  return next;
}

/* ===== arts_db_acquire_is_serialized ===================================
 * RWLOCK: both RW and RO are blocking locks — both are GUID-serialized so the
 * engine acquires them in a global order (deadlock-free lock ordering). */
bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW || mode == DB_MODE_RO;
}

/* ===== arts_db_cache_init ==============================================
 * Initialize the per-rank cache for the RWLOCK-OWNER protocol.
 *
 * Creator hold: arts_db_create defaults to an RW acquire (OCR contract; only
 * ARTS_DB_PROP_NO_ACQUIRE skips it).  The creator is the first data owner, so
 * we seed cache_state with owner-bit=1, rw_st=GRANT, wc=1, rc=0,
 * migrate_target=NO_TARGET.  The matching arts_db_release drives wc back to 0
 * (a sticky release — the creator stays the owner).  Non-creator (OWNER or
 * HOME_RECV) stubs start with all fields zero/idle (not owner). */
void arts_db_cache_init(struct arts_db_cache_s *c, arts_guid_t db_guid,
                        uint64_t db_size, arts_db_init_kind_t kind,
                        unsigned int creator_rank) {
  uint64_t seed;
  if (kind == ARTS_DB_INIT_CREATOR_HOME ||
      kind == ARTS_DB_INIT_CREATOR_REMOTE) {
    /* Creator holds RW and is the data owner: owner=1, rw_st=GRANT, wc=1,
     * ro_st=IDLE, rc=0, migrate_target=NO_TARGET (no pending migration). */
    seed = CACHE_MAKE_FULL(1u, CACHE_ST_GRANT, CACHE_ST_IDLE,
                           ARTS_LOCK_NO_TARGET, 1u, 0u);
  } else {
    /* Non-creator cache stub: not owner, all state IDLE, counts 0,
     * migrate_target=NO_TARGET. */
    seed = CACHE_MAKE_FULL(0u, CACHE_ST_IDLE, CACHE_ST_IDLE,
                           ARTS_LOCK_NO_TARGET, 0u, 0u);
  }
  atomic_store_explicit(&c->cache_state, seed, memory_order_relaxed);
  arts_lf_stack_init(&c->ro_pending);
  arts_lf_stack_init(&c->rw_pending);
  arts_lf_stack_init(&c->ro_serve);
  c->migrate_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  arts_db_cache_common_init(c, db_guid, db_size, kind, creator_rank);
}

/* ===== arts_db_cache_destructor ========================================
 * Runs at refcount 0 as the SOLE owner of the object; no concurrent actor
 * remains, so it is the single safe drainer of the parked-waiter stacks.
 * Destroying an in-use DB is undefined (OCR) — no waiter wake.  Destroy itself
 * is just the route-slot detach (CAS value→NULL + drop the install ref). */
void arts_db_cache_destructor(struct arts_db_cache_s *cache) {
  if (cache == NULL) {
    return;
  }
  arts_db_cache_common_destroy_pre(cache); /* buffer-NULL FIRST */
  /* Drain + free the parked-waiter stacks (ro_pending, rw_pending) and the
   * RO-phase serve list (ro_serve).  All empty in the normal case — a legit
   * destroy has no outstanding acquirers and no pending RO serve. */
  for (arts_lf_stack_t *q = &cache->ro_pending;; q = &cache->rw_pending) {
    arts_lf_link_t *node = arts_lf_stack_drain(q);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_db_excl_waiter_s *w =
          ARTS_CONTAINER_OF(node, struct arts_db_excl_waiter_s, link);
      arts_free(w);
      node = nx;
    }
    if (q == &cache->rw_pending) {
      break;
    }
  }
  {
    arts_lf_link_t *node = arts_lf_stack_drain(&cache->ro_serve);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_lock_ro_serve_node_s *rn =
          ARTS_CONTAINER_OF(node, struct arts_lock_ro_serve_node_s, link);
      arts_free(rn);
      node = nx;
    }
  }
  arts_db_cache_common_destroy_post(cache); /* snapshot free → home teardown */
}

/* ===== arts_db_create_publish_holder ====================================
 * OWNER home init: the creator is the first data owner.  arts_db_home_init
 * already seeds lock_state with the idle directory (LOCK_MAKE(PH_IDLE,
 * creator, 0, 0)); this coalesce-path leaf re-publishes owner=creator into the
 * home word (the home_initialized==true branch of the create handler). */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  /* Re-seed the home directory to phase=IDLE, owner=creator, w=0, r=0 — the
   * same idle-directory state arts_db_home_init installs under OWNER (the
   * creator's RW hold/release are cache-local/sticky and never reach home, so
   * home must not count it; owner names the migrate/serve FORWARD recipient).
   * Single coalesce-path store (the route slot is freshly promoted), no CAS
   * needed. */
  atomic_store_explicit(&db->lock_state,
                        LOCK_MAKE(LOCK_PHASE_IDLE, creator_rank, 0u, 0u),
                        memory_order_release);
}

/* ===== arts_db_create_install_home_buffer ===============================
 * OWNER: data stays with the owner (the creator); the home holds no canonical
 * buffer.  Nothing to install at create — the creator's own cache buffer is
 * installed by db_create_in_place / the DB_CREATE handler. */
void arts_db_create_install_home_buffer(struct arts_db_cache_s *cache,
                                        uint64_t db_size) {
  (void)cache;
  (void)db_size;
}

/* ===== lock_drain_pending ==============================================
 * Serve every waiter currently in `q`: a single atomic-exchange drain claims
 * the whole stack, so each node is taken by exactly one drainer (any node a
 * drainer "misses" is taken by another).  Serving = position-idempotent cursor
 * advance (fires the next serialized dep) + re-derive dep->ptr from the
 * installed buffer + account (may schedule the EDT when acquire_remaining
 * reaches 0).  It does NOT touch the count (the count was bumped at the
 * waiter's acquire cas1, before it was pushed — "queued ⟹ counted"). */
static void lock_drain_pending(arts_lf_stack_t *q) {
  arts_lf_link_t *node = arts_lf_stack_drain(q);
  while (node != NULL) {
    arts_lf_link_t *nx =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    struct arts_db_excl_waiter_s *w =
        ARTS_CONTAINER_OF(node, struct arts_db_excl_waiter_s, link);
    mark_edt_secured_by_guid(w->edt_guid, w->slot);
    mark_edt_ready_by_guid(w->edt_guid, w->slot);
    arts_free(w);
    node = nx;
  }
}

/* ===== owner_try_execute ===============================================
 * The serve-RO + RW-migration driver.  Called whenever an event may have made
 * this rank ready to ship: a FORWARD installing a pending action, a DELIVER
 * making this rank the owner, or a release 0-edge.
 *
 * The release path decides its own 0-edge inside one CAS; this routine exists
 * for the case where the FORWARD lands when there is no live holder left to
 * release, so nothing else would re-evaluate the pending action. */
static void owner_try_execute(struct arts_db_cache_s *cache) {
  /* Callers publish their work — a Treiber push onto ro_serve, or a cache_state
   * CAS — before calling in, and this routine then acquire-loads cache_state to
   * decide whether to drain.  A release-store followed by an acquire-load to a
   * DIFFERENT location is unordered, so without a full fence the gate-load may
   * hoist above the publishing store and the two parties miss each other: the
   * pusher reads a stale live-writer state and skips the drain, while the
   * releaser's drain runs before the pushed node is visible.  The reader strands
   * and the read-only phase never completes. */
  atomic_thread_fence(memory_order_seq_cst);
  /* (1) A pending migrate_target ships the buffer, in one of two shapes.
   *
   * target != self is a real migration: the bytes leave, so it waits for no live
   * local holder of either kind, clears owner-bit + migrate_target, and DELIVERs
   * to the target, which CONFIRMs the home.
   *
   * target == self means the home granted the write phase to the rank that
   * already holds the data — an owner that asked for RW while its own readers
   * blocked its local fast path.  Nothing may move; once the readers drain, the
   * parked writers are granted locally and the home is CONFIRMed so it advances
   * the phase.  Here wc is the writer cohort BEING granted, not a blocker, so
   * this shape must NOT wait for wc==0 — that writer's own count can never reach
   * zero on its own, and waiting would deadlock. */
  for (;;) {
    uint64_t cur =
        atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    uint32_t mt = CACHE_MIGRATE_TARGET(cur);
    /* The standalone-RO clause is the reader guard for the self-migration shape,
     * which has no wc/rc gate of its own: a self-grant must not overwrite a
     * live, separately held RO grant.  A pending RO request, or an orphan grant
     * left by a reader that self-served through a write grant and released, is
     * not a live reader and must not block.  The real-ship arm needs none of
     * this — its wc==0 && rc==0 gate is strictly stronger, and additionally
     * catches a reader that joined under a write grant, which is counted in rc
     * while ro_st stays IDLE. */
    if (mt == ARTS_LOCK_NO_TARGET || CACHE_OWNER(cur) == 0u ||
        (CACHE_RO_ST(cur) == CACHE_ST_GRANT && CACHE_RO_CNT(cur) != 0u)) {
      break; /* no eligible migration (or a live standalone RO grant) */
    }
    if (mt == arts_global_rank_id) {
      /* Self-migration → local RW grant.  Keep owner-bit + counts; set
       * rw_st=GRANT, clear migrate_target in one CAS, then drain rw_pending and
       * CONFIRM home.  (No data ship.)  Does NOT require wc==0. */
      uint64_t next = CACHE_MAKE_FULL(1u, CACHE_ST_GRANT, CACHE_RO_ST(cur),
                                      ARTS_LOCK_NO_TARGET, CACHE_RW_CNT(cur),
                                      CACHE_RO_CNT(cur)) |
                      (cur & CACHE_OWNER_LEASED_MASK); /* preserve ro_granted */
      if (atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur, next,
                                                memory_order_acq_rel,
                                                memory_order_acquire)) {
        lock_drain_pending(&cache->rw_pending);
        arts_send_db_excl_confirm(arts_guid_get_rank(cache->db_guid),
                                  cache->db_guid);
        break;
      }
      continue; /* CAS lost — re-snapshot. */
    }
    /* Real migration to another rank: the buffer LEAVES this rank by one-sided
     * PUT.  Its safety rests on global-lock exclusion — NO local holder may be
     * touching the stable buffer when it ships, because a later migration back
     * to this rank installs IN PLACE and would overwrite the buffer under a
     * still-live holder.  So ship only at the SAME 0-edge REL_RW commits on:
     * wc==0 && rc==0 — no live writer AND no live reader of ANY grant form.
     * Gating on the RW/RO STATE alone is unsound: an RW⊇RO reader is counted in
     * rc while ro_st stays IDLE (it joined under rw_st==GRANT), so a state-only
     * gate would ship out from under it.  When own==1 (guaranteed on this arm)
     * wc/rc count ONLY live granted holders — a non-owner's pending REQ never
     * reaches here — so wc==0 && rc==0 is exactly "no live local holder".  No
     * deadlock: whichever flavor releases LAST re-runs owner_try_execute
     * (REL_RW on its wc 0-edge, REL_RO on its rc 0-edge), so a migration parked
     * here ships the instant the final holder releases.  Preserve ro_st/rc/
     * ro_granted; clear owner-bit + migrate_target and zero wc. */
    if (CACHE_RW_CNT(cur) != 0u || CACHE_RO_CNT(cur) != 0u) {
      break;
    }
    uint64_t next =
        CACHE_MAKE_FULL(0u, CACHE_ST_IDLE, CACHE_RO_ST(cur),
                        ARTS_LOCK_NO_TARGET, 0u, CACHE_RO_CNT(cur)) |
        (cur & CACHE_OWNER_LEASED_MASK); /* preserve ro_granted */
    if (atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur, next,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      /* owner-bit + migrate_target cleared (published) BEFORE the DELIVER
       * ships — a concurrent acquire now sees not-owner and routes via home. */
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      struct arts_db_buffer_s *buf =
          (struct arts_db_buffer_s *)arts_shared_get(buf_h);
      uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
      /* Consume the pending migrate target's landing (single in-flight
       * migration; the FORWARD handler wrote it before publishing the
       * target).  buf_h transfers into the deliver sender (PUT source pin). */
      struct arts_rdzv_landing_s mt_rdzv = cache->migrate_rdzv;
      cache->migrate_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
      arts_send_db_excl_deliver(mt, cache->db_guid, (uint32_t)DB_MODE_RW,
                                &mt_rdzv, buf_h, ds);
      break;
    }
    /* CAS lost — re-snapshot. */
  }

  /* (2) Ship an RO copy to each reader the home asked this rank to serve.  The
   * gate is a live GRANTED writer, not wc>0: wc also counts a merely PENDING
   * write request, which must not hold readers back.  A reader pushed after this
   * drain is picked up by the pusher's own recheck or by the next release. */
  uint64_t cur =
      atomic_load_explicit(&cache->cache_state, memory_order_acquire);
  if (CACHE_OWNER(cur) == 1u && CACHE_RW_ST(cur) != CACHE_ST_GRANT) {
    arts_lf_link_t *node = arts_lf_stack_drain(&cache->ro_serve);
    if (node != NULL) {
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      struct arts_db_buffer_s *buf =
          (struct arts_db_buffer_s *)arts_shared_get(buf_h);
      uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
      while (node != NULL) {
        arts_lf_link_t *nx =
            atomic_load_explicit(&node->next, memory_order_relaxed);
        struct arts_lock_ro_serve_node_s *rn =
            ARTS_CONTAINER_OF(node, struct arts_lock_ro_serve_node_s, link);
        /* Each fan-out PUT pins the source with its own strong ref. */
        arts_send_db_excl_deliver(
            rn->rank, cache->db_guid, (uint32_t)DB_MODE_RO, &rn->rdzv,
            (buf != NULL) ? arts_shared_copy(buf_h) : NULL, ds);
        arts_free(rn);
        node = nx;
      }
      arts_db_buf_release(&buf_h);
    }
  }
}

/* ===== arts_send_db_excl_request ========================================
 * Send MSG_DB_EXCL_REQUEST to the home rank.  Self-send (home == this rank)
 * dispatches through the OoO engine (HIT runs inline; MISS defers until the
 * home db_s is installed) — a REQUEST can race the home CREATE on a cross-rank
 * stub-install path.  Remote send goes via the transport. */
/* Advertise this rank's deliver landing.  RW: the stable buffer, installed
 * IN PLACE by the migration DELIVER (safe — the global RW lock excludes every
 * holder while the migration flies; fixed address preserved).  RO: a FRESH
 * landing buffer — a stale RO grant can race a newer ownership install on
 * this rank (the deliver handler's ro_return arm), and an in-place PUT could
 * not be un-written, so RO serves land aside and install (one copy) only when
 * the state machine accepts them. */
static bool lock_owner_request_landing(struct arts_db_cache_s *cache,
                                      arts_db_access_mode_t mode,
                                      struct arts_rdzv_landing_s *out) {
  *out = (struct arts_rdzv_landing_s){0, 0, 0, 0};
  if (cache->db_size == 0 || arts_global_rank_count <= 1) {
    return false;
  }
  if (mode == DB_MODE_RO) {
    /* Mirror the RW branch's explicit failure handling: on a landing-alloc
     * failure do not advertise a partially-filled landing — re-zero it and
     * report no landing, exactly as the RW path does on rdzv-registration
     * failure. */
    if (arts_db_buf_landing_alloc(cache, cache->db_size, out) == NULL) {
      *out = (struct arts_rdzv_landing_s){0, 0, 0, 0};
      return false;
    }
    return true;
  }
  arts_shared_ptr_t h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf = (struct arts_db_buffer_s *)arts_shared_get(h);
  if (buf == NULL) {
    /* First touch: allocate the one stable buffer (zero-filled; the migration
     * PUT fully overwrites it before any drained waiter reads). */
    arts_db_buf_write_inplace(cache, NULL, cache->db_size);
    h = arts_db_buf_acquire(cache);
    buf = (struct arts_db_buffer_s *)arts_shared_get(h);
  }
  if (buf == NULL ||
      !arts_net_rdzv_local(buf->data, cache->db_size, &out->addr, &out->key)) {
    arts_db_buf_release(&h);
    *out = (struct arts_rdzv_landing_s){0, 0, 0, 0};
    return false;
  }
  out->txid = arts_net_rdzv_txid_next();
  out->cookie = 0; /* in-place: this rank finds the buffer via its cache */
  arts_db_buf_release(&h);
  return true;
}

void arts_send_db_excl_request(struct arts_db_cache_s *cache,
                               arts_db_access_mode_t mode) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  struct arts_rdzv_landing_s rdzv;
  (void)lock_owner_request_landing(cache, mode, &rdzv);
  if (home_rank == arts_global_rank_id) {
    struct arts_ooo_args_db_excl_request_s args = {
        .requester = arts_global_rank_id,
        .db_guid = db_guid,
        .mode = mode,
        .rdzv = rdzv,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_EXCL_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  struct arts_msg_excl_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_REQUEST);
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.rdzv.addr = rdzv.addr;
  p.rdzv.key = rdzv.key;
  p.rdzv.txid = rdzv.txid;
  p.rdzv.cookie = rdzv.cookie;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* LOCK_CTS sender (home → first-touch requester) + requester-side body. */
void arts_send_db_excl_cts(unsigned int requester_rank, arts_guid_t db_guid,
                           uint64_t db_size, uint32_t mode) {
  struct arts_msg_excl_cts_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_CTS);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.db_size = db_size;
  p.mode = mode;
  p.pad = 0;
  if (requester_rank == arts_global_rank_id) {
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_excl_cts(db, &p);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}

void arts_handler_db_excl_cts(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_msg_excl_cts_packet_s *p =
      (struct arts_msg_excl_cts_packet_s *)args_v;
  if (cache->db_size == 0) {
    cache->db_size = p->db_size;
  }
  arts_send_db_excl_request(cache, (arts_db_access_mode_t)p->mode);
}

/* ===== arts_send_db_excl_forward ========================================
 * home → current owner.  Direct send (Cat-C): the owner's cache is always
 * installed (it became owner via DELIVER, or is the creator).  Self-send (owner
 * == home) calls the handler body directly (local hit, no wire). */
void arts_send_db_excl_forward(unsigned int owner_rank, arts_guid_t db_guid,
                               uint32_t mode, uint32_t target,
                               const struct arts_rdzv_landing_s *target_rdzv) {
  struct arts_msg_excl_forward_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_FORWARD);
  p.db_guid = db_guid;
  p.mode = mode;
  p.target = target;
  if (target_rdzv != NULL) {
    p.rdzv.addr = target_rdzv->addr;
    p.rdzv.key = target_rdzv->key;
    p.rdzv.txid = target_rdzv->txid;
    p.rdzv.cookie = target_rdzv->cookie;
  } else {
    p.rdzv = (struct arts_msg_rdzv_landing_s){0, 0, 0, 0};
  }
  if (owner_rank == arts_global_rank_id) {
    arts_handler_db_excl_forward(&p);
    return;
  }
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

/* ===== arts_send_db_excl_deliver ========================================
 * owner → target.  Direct send (Cat-C): the target's cache is always
 * installed (it sent its own REQUEST first, or is the migration target named
 * by home).  Versionless — exclusive-lock serialization guarantees no stale
 * write can race.  The payload travels one-sided into the target's forwarded
 * landing; a self-send builds the contiguous (header + data) buffer and calls
 * the handler body directly (recycling an unused fresh landing). */
void arts_send_db_excl_deliver(unsigned int target_rank, arts_guid_t db_guid,
                               uint32_t mode,
                               const struct arts_rdzv_landing_s *rdzv,
                               arts_shared_ptr_t src_h, uint64_t data_size) {
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  struct arts_msg_excl_deliver_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_DELIVER);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = mode;
  p.pad = 0;
  p.data_size = 0;
  p.rdzv_txid = 0;
  p.rdzv_cookie = (rdzv != NULL) ? rdzv->cookie : 0;
  if (target_rank == arts_global_rank_id) {
    /* Self-serve: inline (header + data) — the target's advertised landing
     * goes unused; recycle a fresh RO landing (cookie names it; RW landings
     * are the stable buffer itself, cookie 0). */
    if (rdzv != NULL && rdzv->cookie != 0) {
      arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
      struct arts_db_s *own = (struct arts_db_s *)arts_shared_get(h);
      if (own != NULL) {
        arts_db_buf_landing_recycle(
            &own->cache, (struct arts_db_buffer_s *)(uintptr_t)rdzv->cookie);
      }
      arts_shared_release(&h);
    }
    uint64_t ds = (src != NULL) ? data_size : 0;
    uint64_t total = sizeof(p) + ds;
    p.header.size = total;
    p.data_size = ds;
    char *buf = (char *)arts_malloc(total);
    memcpy(buf, &p, sizeof(p));
    if (ds > 0) {
      memcpy(buf + sizeof(p), src->data, (size_t)ds);
    }
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    arts_handler_db_excl_deliver(buf, (size_t)total);
    arts_free(buf);
    return;
  }
  if (src == NULL || data_size == 0 || rdzv == NULL || rdzv->txid == 0) {
    /* Data-less deliver (sentinel DB / nothing published). */
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    arts_transport_send_async((int)target_rank, (char *)&p, sizeof(p));
    return;
  }
  /* One-sided deliver: PUT straight from the owner buffer into the target's
   * forwarded landing; the strong ref transfers to the PUT's local
   * completion, pinning the source bytes until the fabric drains them. */
  p.data_size = data_size;
  p.rdzv_txid = rdzv->txid;
  arts_net_put_payload((int)target_rank, rdzv->addr, rdzv->key, rdzv->txid,
                       src->data, data_size, arts_db_buf_ref_release_cb,
                       (void *)src_h);
  arts_transport_send_async((int)target_rank, (char *)&p, sizeof(p));
}

/* ===== arts_send_db_excl_confirm ========================================
 * new owner → home: migration complete.  Direct send (Cat-C): home db_s is
 * always installed (home sent the FORWARD that triggered the migration).
 * Self-send (new owner == home) calls the home handler body directly. */
void arts_send_db_excl_confirm(unsigned int home_rank, arts_guid_t db_guid) {
  struct arts_msg_excl_confirm_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_CONFIRM);
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_excl_confirm(&p);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* ===== arts_send_db_excl_roret ==========================================
 * reader → home: RO release (data-less).  Direct send (Cat-C): home db_s is
 * always installed (home sent the FORWARD(serve) that granted this reader its
 * copy).  Self-send (reader == home) calls the home handler body directly. */
void arts_send_db_excl_roret(unsigned int home_rank, arts_guid_t db_guid) {
  struct arts_msg_excl_confirm_packet_s p; /* CONFIRM/RORET share the struct */
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_RORET);
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_excl_roret(&p);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* ===== arts_handler_db_acquire =========================================
 * OOO_DB_ACQUIRE Cat-B body — protocol-agnostic signature.  item is the
 * pre-pinned home db_s; args is {edt, db_guid, slot}; mode is read from
 * depv[slot].mode.
 *
 * Two-CAS acquire (the HOME RWLOCK discipline, OWNER arbiter):
 *   cas1  count++ ONLY (fetch_add), BEFORE the push, so a waiter present in a
 *         queue is already counted → a drained waiter cannot be released out
 *         from under by a concurrent holder (no premature grant release).
 *   push  the waiter (now drainable; already counted).
 *   cas2  state decision on the FRESH (post-push) state via
 *         cache_owner_compute_next.  Owner fast-path → local GRANT (DRAIN);
 *         non-owner → IDLE→REQ (SEND REQUEST to home); covered → park.
 *   act   SEND a REQUEST, DRAIN (serve parked locally), or NONE (covered). */
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
  atomic_fetch_add_explicit(&cache->cache_state,
                            (mode == DB_MODE_RW) ? CACHE_OWNER_RW_UNIT
                                                 : CACHE_OWNER_RO_UNIT,
                            memory_order_acq_rel);

  /* push: the waiter is now visible to any drainer (and already counted). */
  struct arts_db_excl_waiter_s *w =
      (struct arts_db_excl_waiter_s *)arts_malloc(sizeof(*w));
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
    next = cache_owner_compute_next(cur, op, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));

  switch (act) {
  case CACHE_ACT_SEND_RW:
    INCREMENT_NUM_DB_ACQUIRE_REMOTE_BY(1);
    arts_object_acquire(true);
    arts_send_db_excl_request(cache, DB_MODE_RW);
    break;
  case CACHE_ACT_SEND_RO:
    INCREMENT_NUM_DB_ACQUIRE_REMOTE_BY(1);
    arts_object_acquire(true);
    arts_send_db_excl_request(cache, DB_MODE_RO);
    break;
  case CACHE_ACT_DRAIN_RW:
    /* The permission was already held here, so the waiter is served without
     * asking anyone — the retained grant's whole point. */
    INCREMENT_NUM_DB_ACQUIRE_LOCAL_HIT_BY(1);
    arts_object_acquire(false);
    lock_drain_pending(&cache->rw_pending);
    break;
  case CACHE_ACT_DRAIN_RO:
    INCREMENT_NUM_DB_ACQUIRE_LOCAL_HIT_BY(1);
    arts_object_acquire(false);
    lock_drain_pending(&cache->ro_pending);
    break;
  default: /* NONE: parked; a DELIVER / another acquire's drain serves us. */
    break;
  }
}

/* ===== arts_handler_db_excl_forward ====================================
 * Cat-C @owner.  item_v is the full wire packet (arts_msg_excl_forward_packet_s
 * *).  RW(mode) → install migrate_target (CAS into cache_state); RO(mode) →
 * push the reader rank onto ro_serve.  Then owner_try_execute drives the ship
 * (immediate if wc==0; otherwise the eventual release 0-edge does it).
 *
 * Cat-C route-table lookup: a NULL lookup means the DB was destroyed
 * concurrently → drop. */
void arts_handler_db_excl_forward(void *item_v) {
  struct arts_msg_excl_forward_packet_s *p =
      (struct arts_msg_excl_forward_packet_s *)item_v;
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_shared_release(&db_h);
    return;
  }
  struct arts_db_cache_s *cache = &db->cache;

  if ((arts_db_access_mode_t)p->mode == DB_MODE_RW) {
    /* Stash the migrate target's landing BEFORE the CAS that publishes the
     * target (single in-flight migration per owner — home serializes by
     * CONFIRM — so this plain store has no concurrent writer; the shipping
     * actor reads it only after observing the published target). */
    cache->migrate_rdzv.addr = p->rdzv.addr;
    cache->migrate_rdzv.key = p->rdzv.key;
    cache->migrate_rdzv.txid = p->rdzv.txid;
    cache->migrate_rdzv.cookie = p->rdzv.cookie;
    /* Install the migrate target into the word (single CAS preserving the
     * rest). push-before-decision is not needed here — there is exactly one
     * in-flight migrate target per owner (home serializes FORWARD-migrate by
     * CONFIRM). */
    uint64_t cur, next;
    do {
      cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
      next =
          CACHE_MAKE_FULL(CACHE_OWNER(cur), CACHE_RW_ST(cur), CACHE_RO_ST(cur),
                          p->target, CACHE_RW_CNT(cur), CACHE_RO_CNT(cur)) |
          (cur & CACHE_OWNER_LEASED_MASK); /* preserve ro_granted */
    } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                    next, memory_order_acq_rel,
                                                    memory_order_acquire));
  } else {
    /* Push the reader rank onto ro_serve, THEN owner_try_execute — push-then-
     * recheck (owner_try_execute re-reads wc==0) so a reader pushed just after
     * a drain is not stranded. */
    struct arts_lock_ro_serve_node_s *rn =
        (struct arts_lock_ro_serve_node_s *)arts_malloc(sizeof(*rn));
    rn->rank = p->target;
    rn->rdzv.addr = p->rdzv.addr;
    rn->rdzv.key = p->rdzv.key;
    rn->rdzv.txid = p->rdzv.txid;
    rn->rdzv.cookie = p->rdzv.cookie;
    arts_lf_stack_push(&cache->ro_serve, &rn->link);
  }

  owner_try_execute(cache);
  arts_shared_release(&db_h);
}

/* ===== arts_handler_db_excl_deliver ====================================
 * Cat-C @target.  payload is the full contiguous wire buffer (header + db_size
 * data bytes); size is the byte count.
 *
 *   RW → become owner: install the data, set owner-bit + rw_st=GRANT (CAS),
 *        drain rw_pending (the parked RW writers are secured + ready), send
 *        CONFIRM to home, then owner_try_execute (in case a migrate_target /
 *        ro_serve was already pending — RW chain / RW→RO flip).
 *   RO → install the copy, set ro_st=GRANT (CAS), drain ro_pending (the parked
 *        RO readers are secured + ready).
 *
 * Cat-C route-table lookup: NULL ⇒ DB destroyed concurrently ⇒ drop.
 * Data is installed BEFORE the CAS so drained waiters observe it. */
/* Post-arrival tail of the DELIVER handler: the state transition + drains /
 * CONFIRM / return, run once the payload bytes are available.  `landing` is
 * non-NULL only for a one-sided RO serve (the fresh aside buffer the bytes
 * landed in; installed or recycled per the ro_return decision).  `data` is
 * the inline same-rank payload (NULL on the one-sided path).  Consumes db_h. */
static void lock_deliver_commit(arts_shared_ptr_t db_h, arts_guid_t db_guid,
                                arts_db_access_mode_t mode, const void *data,
                                uint64_t data_size,
                                struct arts_db_buffer_s *landing);

/* Rendezvous continuation: the deliver payload fully landed (RW: in place in
 * the stable buffer; RO: in the fresh aside landing).  Runs the commit. */
struct lock_deliver_landed_ctx_s {
  arts_shared_ptr_t db_h;
  arts_guid_t db_guid;
  arts_db_access_mode_t mode;
  uint64_t data_size;
  struct arts_db_buffer_s *landing; /* RO aside landing; NULL for RW */
};

static void lock_deliver_landed_cb(void *arg) {
  struct lock_deliver_landed_ctx_s *ctx =
      (struct lock_deliver_landed_ctx_s *)arg;
  if (arts_shared_get(ctx->db_h) != NULL) {
    lock_deliver_commit(ctx->db_h, ctx->db_guid, ctx->mode, NULL,
                        ctx->data_size, ctx->landing);
  } else {
    if (ctx->landing != NULL) {
      arts_regpool_free(ctx->landing); /* cache gone — free the aside bytes */
    }
    arts_shared_release(&ctx->db_h);
  }
  arts_free(ctx);
}

void arts_handler_db_excl_deliver(void *payload, size_t size) {
  struct arts_msg_excl_deliver_packet_s *p =
      (struct arts_msg_excl_deliver_packet_s *)payload;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)p->mode;
  const void *data = (const char *)p + sizeof(*p);
  uint64_t data_size = (uint64_t)size - (uint64_t)sizeof(*p);

  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    /* Destroyed mid-flight: consume any pairing (frees an RO aside landing;
     * an RW in-place landing has cookie 0 — nothing to free). */
    arts_db_rdzv_discard_landing(p->rdzv_txid, p->rdzv_cookie);
    arts_shared_release(&db_h);
    return;
  }

  if (p->rdzv_txid != 0) {
    /* One-sided deliver: pair this packet with the write completion (either
     * order); the commit runs once both are in. */
    struct lock_deliver_landed_ctx_s *ctx =
        (struct lock_deliver_landed_ctx_s *)arts_malloc(sizeof(*ctx));
    ctx->db_h = db_h;
    ctx->db_guid = p->db_guid;
    ctx->mode = mode;
    ctx->data_size = p->data_size;
    ctx->landing = (struct arts_db_buffer_s *)(uintptr_t)p->rdzv_cookie;
    arts_net_rdzv_expect(p->rdzv_txid, lock_deliver_landed_cb, ctx);
    return;
  }

  /* Same-rank / data-less deliver: inline payload (if any), no landing. */
  lock_deliver_commit(db_h, p->db_guid, mode, (data_size > 0u) ? data : NULL,
                      data_size, NULL); /* consumes db_h */
}

static void lock_deliver_commit(arts_shared_ptr_t db_h, arts_guid_t db_guid,
                                arts_db_access_mode_t mode, const void *data,
                                uint64_t data_size,
                                struct arts_db_buffer_s *landing) {
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  struct arts_db_cache_s *cache = &db->cache;

  if (mode == DB_MODE_RW) {
    /* Migration is the ONLY thing that moves buffer data (owner→owner).  A
     * one-sided migration already landed IN PLACE in the stable buffer ("imm
     * seen => buffer valid"; no concurrent holder — migration is
     * CONFIRM-gated, single owner, global-lock excluded); a same-rank inline
     * payload installs here.  Then become owner: owner-bit + rw_st=GRANT in
     * one CAS (preserve counts + ro_st; migrate_target stays NO_TARGET — a
     * chained migrate is installed later by a fresh FORWARD). */
    if (data != NULL && data_size > 0u) {
      arts_db_buf_write_inplace(cache, data, data_size);
    }
    uint64_t cur, next;
    do {
      cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
      next = CACHE_MAKE_FULL(1u, CACHE_ST_GRANT, CACHE_RO_ST(cur),
                             CACHE_MIGRATE_TARGET(cur), CACHE_RW_CNT(cur),
                             CACHE_RO_CNT(cur)) |
             (cur & CACHE_OWNER_LEASED_MASK); /* preserve ro_granted */
    } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                    next, memory_order_acq_rel,
                                                    memory_order_acquire));
    /* RW grant serves this rank's RW + RO cohort (RW⊇RO): drain BOTH the parked
     * writers and any parked readers (a reader that requested RO before this
     * rank became the RW owner is parked in ro_pending and joins the RW grant
     * here).  Without the ro_pending drain those readers strand, rc never
     * reaches zero, and the migration — which waits for it — deadlocks. */
    lock_drain_pending(&cache->rw_pending);
    lock_drain_pending(&cache->ro_pending);
    /* Tell home the migration is complete so it can advance (RW chain / flip).
     */
    arts_send_db_excl_confirm(arts_guid_get_rank(db_guid), db_guid);
    /* A migrate_target / ro_serve may already be pending (home chained the next
     * action); drive it. */
    owner_try_execute(cache);
  } else {
    /* RO grant: a borrower may receive a read-only copy.  Decide on the
     * snapshot, install BEFORE publishing, then CAS:
     *   ro_return (rw_st==GRANT || rc==0): nothing to serve here.  rw_st==GRANT
     *       means this rank already holds the data as the RW owner (RW⊇RO
     * covers its readers); rc==0 means its readers were already served (e.g.
     * via an RW⊇RO join while it was a past owner).  In BOTH cases do NOT drain
     *       and do NOT install (installing would override valid owner data), do
     *       NOT take ros/ro_granted; set ros=IDLE and immediately return the
     *       unneeded grant (RO_RETURN) to balance the home's r.  This is the
     * one uniform handler for current-owner / past-owner / phantom — the home
     *       sends RO grants unconditionally and needs no owner tracking.
     *   serve (rw_st!=GRANT && rc>0): the rank held stale data, so install the
     *       granted copy (override is correct, like HOME's RO-grant memcpy),
     * set ros=GRANT + ro_granted (owes one RO_RETURN when its readers drain),
     *       then serve them. */
    uint64_t cur, next;
    bool ro_return;
    bool installed = false;
    do {
      cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
      ro_return =
          (CACHE_RW_ST(cur) == CACHE_ST_GRANT) || (CACHE_RO_CNT(cur) == 0u);
      if (!ro_return && !installed) {
        /* Deferred commit (universal invariant): a grant may become observable
         * only AFTER its bytes are installed.  Install the granted copy into
         * the stable buffer (fixed address) BEFORE the CAS that publishes
         * ros=GRANT: the moment any acquirer observes GRANT (its acquire-load
         * of cache_state pairing with this CAS's release) it can be granted
         * locally and read the buffer, so publishing first would hand out the
         * not-yet-installed (stale) bytes.  Installing pre-publish is safe:
         * while this rank's RO grant is outstanding the home cannot start an
         * RW phase (its r-count includes this grant), so no ownership DELIVER
         * can race this commit; and with ros still REQ every local RO acquire
         * parks, so no reader touches the buffer until the publish. */
        if (landing != NULL) {
          arts_db_buf_write_inplace(cache, landing->data, data_size);
        } else if (data != NULL && data_size > 0u) {
          arts_db_buf_write_inplace(cache, data, data_size);
        }
        installed = true;
      }
      next = CACHE_MAKE_FULL(CACHE_OWNER(cur), CACHE_RW_ST(cur),
                             ro_return ? CACHE_ST_IDLE : CACHE_ST_GRANT,
                             CACHE_MIGRATE_TARGET(cur), CACHE_RW_CNT(cur),
                             CACHE_RO_CNT(cur)) |
             (cur & CACHE_OWNER_LEASED_MASK);
      if (!ro_return) {
        next |= CACHE_OWNER_LEASED_MASK;
      }
    } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                    next, memory_order_acq_rel,
                                                    memory_order_acquire));
    if (ro_return) {
      /* Nothing to serve — and the granted bytes must NOT touch the stable
       * buffer (this rank may hold newer owner data).  This is exactly why an
       * RO serve lands ASIDE: recycle the untouched landing and return the
       * grant. */
      if (landing != NULL) {
        arts_db_buf_landing_recycle(cache, landing);
      }
      arts_send_db_excl_roret(arts_guid_get_rank(db_guid), db_guid);
    } else {
      /* Bytes were installed before the publish above; the landing is spent —
       * recycle it and wake the parked readers. */
      if (landing != NULL) {
        arts_db_buf_landing_recycle(cache, landing);
      }
      lock_drain_pending(&cache->ro_pending);
    }
  }

  arts_shared_release(&db_h);
}

/* ===== arts_db_release_rw / arts_db_release_ro =========================
 * OWNER release — single-word CAS via cache_owner_compute_next, then dispatch the
 * returned action.
 *
 * RW: wc-- (+ 0-edge migration decision) in ONE CAS.  CACHE_ACT_MIGRATE means
 *     the 0-edge cleared owner-bit + migrate_target in the same next-state
 *     (publish-before-send); ship DELIVER(RW) to the target.  No home
 * publish, no ACK (the HOME ACK path is not linked in OWNER) — data is sticky
 * on the owner.  No pending target → sticky (no-op). RO: rc--; on the 0-edge —
 * if this rank is still the owner, owner_try_execute (drive any pending RO
 * serve / migration now that readers are gone); else RORET to home (a remote RO
 * copy holder returning its grant).
 *
 * Destroyed-guard: a dep NULL-woken at destroy never really held the lock; a
 * count of 0 means there is nothing to release — skip rather than underflow. */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  uint32_t act;
  uint64_t cur, next;
  uint32_t target = ARTS_LOCK_NO_TARGET;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_RW_CNT(cur) == 0u) {
      return; /* destroyed-guard / already released */
    }
    target = CACHE_MIGRATE_TARGET(cur); /* captured pre-CAS for the ship */
    next = cache_owner_compute_next(cur, CACHE_OP_REL_RW, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_MIGRATE) {
    /* owner-bit + migrate_target already cleared (published) inside the CAS;
     * ship the buffer to the target now. */
    arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
    struct arts_db_buffer_s *buf =
        (struct arts_db_buffer_s *)arts_shared_get(buf_h);
    uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
    /* Consume the pending migrate target's landing; buf_h transfers into the
     * deliver sender (PUT source pin). */
    struct arts_rdzv_landing_s mt_rdzv = cache->migrate_rdzv;
    cache->migrate_rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
    arts_send_db_excl_deliver(target, cache->db_guid, (uint32_t)DB_MODE_RW,
                              &mt_rdzv, buf_h, ds);
  } else if (CACHE_RW_CNT(next) == 0u && CACHE_OWNER(next) == 1u) {
    /* Sticky 0-edge (no pending migration): the writer phase just ended and we
     * remain the owner.  Drive any RO readers the home queued via
     * FORWARD(serve) during the write phase: they sit in ro_serve until the last
     * writer leaves.  No-op on a single node, where ro_serve is always empty. */
    owner_try_execute(cache);
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
    next = cache_owner_compute_next(cur, CACHE_OP_REL_RO, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_REL_RO) {
    /* rc 0-edge.  The RO_RETURN decision keys off ro_granted (CACHE_LEASED of
     * the pre-CAS state that committed), NOT the owner-bit: a home-COUNTED RO
     * grant (one that arrived via DELIVER(RO)) owes the home a RO_RETURN to
     * balance its r, even if this rank is now the owner (a held reader that was
     * promoted to owner at the RW→RO flip self-served its own data — without
     * this RO_RETURN the home's r would never reach 0 and the RO→RW flip would
     * deadlock).  An owner-fast-path local RO (never granted, never counted at
     * home) sends none. Independently, if this rank is still the owner, drive
     * any pending RO serve / migration now that the local readers have drained.
     */
    bool granted = CACHE_LEASED(cur);
    if (granted) {
      arts_send_db_excl_roret(arts_guid_get_rank(cache->db_guid),
                              cache->db_guid);
    }
    uint64_t now =
        atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_OWNER(now) == 1u) {
      owner_try_execute(cache);
    }
  }
}

/* ===== lock_owner_compute_next ==========================================
 * OWNER home arbiter — pure function of the single lock_state word
 * [phase:2 | owner:14 | w:24 | r:24], run inside a CAS-retry loop.  This IS the
 * home phase-transition table.
 *
 * The transitions:
 *   RW_REQ  : w++; phase==IDLE → phase=RW, action=MIGRATE.  (phase==RW: just
 *             queue, next migrate fires at CONFIRM; phase==RO: hold, r
 * untouched) RO_REQ  : r++; phase==RO → SERVE_ONE; phase==IDLE → phase=RO +
 * SERVE_ONE. (phase==RW: held in ro_waiters by the caller, no action) CONFIRM :
 * w--, owner=new_owner (ONE CAS); w>0 → MIGRATE (RW chain); elif r>0 → phase=RO
 * + SERVE_ALL (RW→RO flip); else phase=IDLE. RO_RET  : r--; on r==0: w>0 →
 * phase=RW + MIGRATE (RO→RW flip); else phase=IDLE.
 *
 * Equivalences: mig_inflight == (phase==RW), serving_ro == (phase==RO),
 * r == the old rc — no separate flags; everything lives in the word.
 *
 * The FORWARD recipient is OWNER() of the returned word.  The migrate target /
 * served reader(s) are resolved by the caller from rw_waiters/ro_waiters. */
uint64_t lock_owner_compute_next(uint64_t cur, int op, unsigned int new_owner,
                                uint32_t *out_action) {
  uint32_t phase = LOCK_PHASE(cur);
  uint32_t owner = LOCK_OWNER(cur);
  uint32_t w = LOCK_W(cur);
  uint32_t r = LOCK_R(cur);
  uint32_t action = LOCK_ACTION_NONE;
  switch (op) {
  case LOCK_OP_RW_ACQ:
    w += 1;
    if (phase == LOCK_PHASE_IDLE) {
      phase = LOCK_PHASE_RW;
      action = LOCK_ACTION_FORWARD_MIGRATE;
    }
    /* phase==RW: queue only (next migrate at CONFIRM).
     * phase==RO: hold (r unaffected here — the RW parks in rw_waiters). */
    break;
  case LOCK_OP_RO_ACQ:
    r += 1;
    if (phase == LOCK_PHASE_RO) {
      action = LOCK_ACTION_FORWARD_SERVE_ONE;
    } else if (phase == LOCK_PHASE_IDLE) {
      phase = LOCK_PHASE_RO;
      action = LOCK_ACTION_FORWARD_SERVE_ONE;
    }
    /* phase==RW: held in ro_waiters by the caller; no action. */
    break;
  case LOCK_OP_CONFIRM:
    w -= 1;
    owner = new_owner;
    if (w > 0u) {
      action = LOCK_ACTION_FORWARD_MIGRATE; /* RW chain */
    } else if (r > 0u) {
      phase = LOCK_PHASE_RO;
      action = LOCK_ACTION_FORWARD_SERVE_ALL; /* RW → RO flip */
    } else {
      phase = LOCK_PHASE_IDLE;
    }
    break;
  case LOCK_OP_RO_RET:
    r -= 1;
    if (r == 0u) {
      if (w > 0u) {
        phase = LOCK_PHASE_RW;
        action = LOCK_ACTION_FORWARD_MIGRATE; /* RO → RW flip */
      } else {
        phase = LOCK_PHASE_IDLE;
      }
    }
    break;
  default:
    break;
  }
  *out_action = action;
  return LOCK_MAKE(phase, owner, w, r);
}

/* ===== Home-side handlers (Task 6) =====================================
 * The home directory arbiter + action dispatch.  Each handler runs the single-
 * word lock_state CAS via lock_owner_compute_next, then dispatches the returned
 * FORWARD action to the current owner (OWNER() of the freshly-CAS'd word),
 * resolving the migrate target / served reader(s) from the lock-free queues.
 *
 * They run CONCURRENTLY across receiver threads with no mutex; correctness
 * rests entirely on the single-word CAS plus the lock-free queues.  The protocol
 * tolerates message reorder because the transport guarantees no per-peer FIFO:
 * r counts OUTSTANDING RO grants, so it may transiently exceed the rank count,
 * and the write flip is gated on r reaching zero rather than on any arrival
 * order. */

/* ===== RO waiter node (home-side held-RO queue entry) ==================
 * Held-RO readers (an RO REQUEST that arrived during the RW phase) wait in
 * db->ro_waiters until the RW→RO flip drains them.  Same shape as the
 * file-local node in home.c / home.c (ro_waiters is the Treiber db->ro_waiters
 * those TUs also use); the SERVE_ONE path serves the requester directly and
 * never touches this queue, so the only producers/consumers are the home
 * REQUEST hold-push and the CONFIRM SERVE_ALL drain. */
struct arts_lock_ro_node_s {
  arts_lf_link_t link; /* FIRST */
  unsigned int rank;
  struct arts_rdzv_landing_s rdzv; /* reader's serve landing, forwarded */
};

/* Dispatch a MIGRATE or SERVE_ALL FORWARD action after the lock_state CAS
 * committed.  `word` is the freshly-CAS'd lock_state, whose OWNER() names the
 * rank to FORWARD to.  SERVE_ONE is not handled here: its target is the
 * requesting rank, which the word does not carry. */
static void lock_owner_home_forward(struct arts_db_s *db, uint32_t action,
                                   uint64_t word) {
  unsigned int owner = LOCK_OWNER(word);
  if (action == LOCK_ACTION_FORWARD_MIGRATE) {
    /* peek, not pop: the requester stays queued until its own CONFIRM pops it,
     * which is what makes "queue front == the rank that will CONFIRM" hold. */
    unsigned int target;
    struct arts_rdzv_landing_s target_rdzv;
    if (arts_home_grantreq_queue_peek(&db->rw_waiters, &target, &target_rdzv)) {
      arts_send_db_excl_forward(owner, db->cache.db_guid, (uint32_t)DB_MODE_RW,
                                target, &target_rdzv);
    }
  } else if (action == LOCK_ACTION_FORWARD_SERVE_ALL) {
    /* RW → RO flip: drain every held RO waiter and FORWARD-serve each
     * UNCONDITIONALLY (the fixed single-target packet, same shape as
     * SERVE_ONE). No owner identity tracking here: a rank that is (or was) the
     * owner gets the grant like any other, and its cache returns it immediately
     * (RO_RETURN) when it already holds the data (rw_st==GRANT) or has nothing
     * to serve (rc==0) — see arts_handler_db_excl_deliver's RO arm.  Keeping
     * the home side unconditional avoids tracking owner identity across
     * migrations. */
    arts_lf_link_t *node = arts_lf_stack_drain(&db->ro_waiters);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_lock_ro_node_s *rn =
          ARTS_CONTAINER_OF(node, struct arts_lock_ro_node_s, link);
      arts_send_db_excl_forward(owner, db->cache.db_guid, (uint32_t)DB_MODE_RO,
                                rn->rank, &rn->rdzv);
      arts_free(rn);
      node = nx;
    }
  }
}

/* ===== arts_handler_db_excl_request ====================================
 * Cat-B @home (OoO OOO_DB_EXCL_REQUEST).  item_v is the pre-pinned home db_s;
 * args_v is {requester, db_guid, mode}.
 *
 * Both modes push the requester onto their waiter queue BEFORE the CAS, so
 * "queued ⟹ counted" holds and any transition that observes the count also finds
 * the node.  A write request that finds the lock idle FORWARDs the owner to
 * migrate to the queue front; a read request is served at once in the read-only
 * or idle phase and otherwise held for the eventual flip. */
void arts_handler_db_excl_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_excl_request_s *a =
      (struct arts_ooo_args_db_excl_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)a->mode;

  if (a->rdzv.txid == 0 && db->cache.db_size > 0 &&
      arts_global_rank_count > 1) {
    /* First-touch request without a landing: the requester did not know
     * db_size.  Answer with the size (CTS) and do NOT enqueue — the deliver
     * plane requires a landing.  The requester re-issues with one. */
    arts_send_db_excl_cts(requester, db->cache.db_guid, db->cache.db_size,
                          (uint32_t)mode);
    return;
  }

  arts_rank_bitset_set(&db->cached_ranks,
                       requester); /* destroy fan-out roster */

  uint32_t action;
  uint64_t cur, next;
  if (mode == DB_MODE_RW) {
    /* push-before-CAS: the requester is queued (and thus counted by w) before
     * the transition reads w, so a concurrent CONFIRM/peek sees it. */
    arts_home_grantreq_queue_push(&db->rw_waiters, requester, &a->rdzv);
    do {
      cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
      next = lock_owner_compute_next(cur, LOCK_OP_RW_ACQ, /*new_owner=*/0u,
                                    &action);
    } while (!atomic_compare_exchange_weak_explicit(&db->lock_state, &cur, next,
                                                    memory_order_acq_rel,
                                                    memory_order_acquire));
    lock_owner_home_forward(db, action, next);
  } else {
    /* Enqueue BEFORE the r++ CAS, so whichever transition observes that r++ —
     * this request's own serve, or a concurrent flip's SERVE_ALL — finds the node
     * already queued and serves it from the freshly-CAS'd owner.  Serving is
     * ALWAYS a queue drain; there is deliberately no path that serves the
     * requester directly from a snapshot, because such a snapshot can name an
     * owner that has since migrated, stranding both the reader and the home's
     * r. */
    struct arts_lock_ro_node_s *n =
        (struct arts_lock_ro_node_s *)arts_malloc(sizeof(*n));
    n->rank = requester;
    n->rdzv = a->rdzv;
    arts_lf_stack_push(&db->ro_waiters, &n->link);
    do {
      cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
      next = lock_owner_compute_next(cur, LOCK_OP_RO_ACQ, /*new_owner=*/0u,
                                    &action);
    } while (!atomic_compare_exchange_weak_explicit(&db->lock_state, &cur, next,
                                                    memory_order_acq_rel,
                                                    memory_order_acquire));
    /* RO/idle phase (SERVE_ONE): drain ro_waiters and FORWARD each reader to
     * the freshly-CAS'd owner — this serves the just-pushed node (plus any
     * reader a concurrent request pushed; XCHG drain partitions are disjoint).
     * RW phase (NONE): leave the node held; the RW→RO flip's SERVE_ALL drains
     * it. */
    if (action == LOCK_ACTION_FORWARD_SERVE_ONE) {
      lock_owner_home_forward(db, LOCK_ACTION_FORWARD_SERVE_ALL, next);
    }
  }
}

/* ===== arts_handler_db_excl_confirm ====================================
 * Cat-C @home (direct dispatch).  item_v is the full wire packet
 * (arts_msg_excl_confirm_packet_s *); the new owner is packet->header.rank (the
 * sender — both wire and self-send set it via arts_fill_packet_header).
 *
 * Cat-C route-table lookup: NULL ⇒ DB destroyed concurrently ⇒ drop.  pop the
 * RW waiter (assert front == src), then CAS lock_owner_compute_next(CONFIRM,
 * new_owner=src) — this sets owner=src + w-- + next phase + action in ONE CAS.
 * Dispatch: MIGRATE → next RW migrate; SERVE_ALL → RW→RO flip drain. */
void arts_handler_db_excl_confirm(void *item_v) {
  struct arts_msg_excl_confirm_packet_s *p =
      (struct arts_msg_excl_confirm_packet_s *)item_v;
  unsigned int new_owner = p->header.rank;
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_shared_release(&db_h);
    return;
  }

  /* pop the confirming requester off the RW queue (it was the migrate target).
   * front must equal new_owner (the home FORWARDed migrate to the queue front,
   * and CONFIRM comes only from that owner).
   *
   * Stale cross-generation guard: a labeled GUID can be destroyed and
   * re-created (route slot re-promoted to a FRESH db_s with an empty rw_waiters
   * / w==0); a CONFIRM from the OLD generation, delayed on the wire, can then
   * land on the NEW generation's home.  Such a stale CONFIRM finds the RW queue
   * empty (the legit migrate target was pushed BEFORE the FORWARD that triggers
   * any real CONFIRM, so a real CONFIRM always pops successfully).  An empty
   * pop therefore means a stale/duplicate CONFIRM — drop it WITHOUT the w--
   * transition, else it underflows w (0 → 0xFFFFFF) and dispatches a spurious
   * MIGRATE. */
  unsigned int front;
  if (!arts_home_grantreq_queue_pop(&db->rw_waiters, &front, NULL)) {
    arts_shared_release(&db_h);
    return; /* stale / duplicate CONFIRM (empty RW queue) — drop */
  }
  /* front == new_owner under the protocol; the pop just dequeued it. */
  (void)front;

  uint32_t action;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    next = lock_owner_compute_next(cur, LOCK_OP_CONFIRM, new_owner, &action);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_acq_rel, memory_order_acquire));
  lock_owner_home_forward(db, action, next);
  arts_shared_release(&db_h);
}

/* ===== arts_handler_db_excl_roret ======================================
 * Cat-C @home (direct dispatch).  item_v is the full wire packet
 * (arts_msg_excl_confirm_packet_s *, shared with CONFIRM); the returning reader
 * is packet->header.rank (informational — RO accounting is rank-agnostic).
 *
 * CAS lock_owner_compute_next(RO_RET) — r--; on r→0 with RW waiting, flip phase
 * to RW and MIGRATE to the next RW waiter (RO→RW flip).  Dispatch the action.
 */
void arts_handler_db_excl_roret(void *item_v) {
  struct arts_msg_excl_confirm_packet_s *p =
      (struct arts_msg_excl_confirm_packet_s *)item_v;
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_shared_release(&db_h);
    return;
  }

  uint32_t action;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    /* Stale cross-generation guard (symmetric to CONFIRM): a RO_RETURN from an
     * OLD generation, delayed on the wire, can land on a re-created labeled
     * GUID's fresh home (r==0).  r counts outstanding RO grants; a RO_RETURN
     * can only follow its own grant, so r==0 means a stale/duplicate RO_RETURN
     * — drop it, else r underflows (0 → 0xFFFFFF) and dispatches a spurious
     * flip. */
    if (LOCK_R(cur) == 0u) {
      arts_shared_release(&db_h);
      return;
    }
    next =
        lock_owner_compute_next(cur, LOCK_OP_RO_RET, /*new_owner=*/0u, &action);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_acq_rel, memory_order_acquire));
  lock_owner_home_forward(db, action, next);
  arts_shared_release(&db_h);
}
