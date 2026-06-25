/* SPDX-License-Identifier: Apache-2.0
 *
 * LOCK protocol LAZY timing translation unit.
 *
 * Defines: cache_lazy_compute_next (the LAZY owner-cache arbiter),
 *          owner_try_execute (static), lock_drain_pending (static),
 *          arts_db_acquire_is_serialized, arts_db_cache_init,
 *          arts_db_cache_destructor, arts_db_create_publish_holder,
 *          arts_db_create_install_home_buffer,
 *          arts_send_db_lock_request, arts_handler_db_acquire,
 *          arts_db_release_rw, arts_db_release_ro,
 *          arts_send_db_lock_forward/deliver/confirm/roret,
 *          arts_handler_db_lock_forward/deliver,
 *          and (Task-6 stubs) arts_handler_db_lock_request/confirm/roret.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=LOCK + ARTS_PROTOCOL_TIMING=LAZY.
 * Contains NO timing preprocessor guards — the timing variant is selected at
 * link time by CMakeLists.txt (home.c eager.c vs home.c lazy.c).
 *
 * Design: docs/superpowers/specs/2026-06-24-lock-lazy-coherence-design.md.
 * The owner/cache side mirrors the verified model checker
 * (2026-06-24-lock-lazy-modelcheck.py) functions owner_try_execute,
 * cache_on_FORWARD, cache_on_DELIVER, issue_acquire, release_rw, release_ro.
 *
 * Data is LAZY-sticky on the owner (the last RW holder); the home holds only a
 * directory (owner rank + phase + waiter queues), no canonical buffer.  All
 * remote requests go to the home; intra-node (owner-local) RW/RO are served
 * directly from the owner's buffer without a home round-trip.
 */

/* lock/types.h must precede all other coherence headers: it defines
 * arts_db_cache_s, arts_db_s, the CACHE_ and LOCK_ macros, the LAZY arbiter
 * prototypes, and arts_db_lock_waiter_s for the LOCK build. */
#include "arts/coherence/lock/types.h"

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

/* ===== RO-serve node (reader rank the owner must serve in the RO phase) =
 * The owner's ro_serve is an arts_lf_stack_t (Treiber); each node carries one
 * reader rank that the home asked the owner to serve via FORWARD(serve).  File-
 * local: only the owner-side FORWARD handler (push) and owner_try_execute /
 * the destructor (drain) touch it. */
struct arts_lock_ro_serve_node_s {
  arts_lf_link_t link; /* FIRST — required by arts_lf_stack_t */
  unsigned int rank;
};

/* fetch_add units for the cas1 count bump (the LAZY cache_state has different
 * field shifts than EAGER, which defines CACHE_RW_UNIT/CACHE_RO_UNIT itself).
 * Adding these never touches the higher state fields. */
#define CACHE_LAZY_RW_UNIT ((uint64_t)1 << CACHE_LAZY_WC_SHIFT)
#define CACHE_LAZY_RO_UNIT ((uint64_t)1 << CACHE_LAZY_RC_SHIFT)

/* ===== cache_lazy_compute_next =========================================
 * LAZY owner-cache arbiter — pure function of the single cache_state word, run
 * inside a CAS-retry loop.  The cache_state packs owner-bit + rw_st/ro_st +
 * migrate_target + wc + rc into one word, so every transition (including the
 * RW-release 0-edge migration decision) is a single CAS.
 *
 * Mirrors the EAGER cache_compute_next op set (ACQ_RW/ACQ_RO/REL_RW/REL_RO)
 * but with the LAZY actions, plus the owner fast-path on the ACQ ops (model
 * issue_acquire): when this rank is the data owner and capable, the acquire is
 * granted LOCALLY (DRAIN) with no REQUEST to home.  FORWARD-migrate / FORWARD-
 * serve do NOT go through this arbiter: arts_handler_db_lock_forward CASes
 * migrate_target / pushes ro_serve directly (the target/reader cannot be a
 * pure-function output), so there are no FORWARD ops here.
 *
 * ACQ_* ops decide request/join/local-grant state ONLY — the count was bumped
 * by the separate fetch_add cas1 BEFORE the push.  REL_* ops carry the count--
 * inside the CAS (the decrement and the 0-edge transition must be atomic).
 *
 * Owner fast-path eligibility is read entirely from the word:
 *   RW: owner-bit && rc==0 && migrate_target==NO_TARGET   (model issue_acquire
 *       also gates on ro_serve being empty; ro_serve is only ever populated by
 *       a home FORWARD(RO), which arrives during the RO phase — in which the
 *       owner is not holding RW — so an RW owner-fast-path with a non-empty
 *       ro_serve is not reachable on the single arbiter path).
 *   RO: owner-bit && wc==0
 */
uint64_t cache_lazy_compute_next(uint64_t cur, int op, uint32_t *out_action) {
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
       * (the LOCK serializes only inter-node, at the home), and a pending
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
    /* RO decision (mirrors EAGER cache_compute_next ACQ_RO):
     *   rw_st==GRANT  → this rank holds RW (owner) → read locally (RW⊇RO).
     *   ro_st==GRANT  → an RO copy is already held → join it.
     *   owner         → local hit: this rank holds the data → read locally.
     *   rw_st==IDLE && ro_st==IDLE → send a fresh RO REQUEST to home.
     *   otherwise (rw_st==REQ, or ro_st==REQ) → park (NONE).  A pending RW
     *     request already covers this reader (RW⊇RO — the RW grant's DRAIN_BOTH
     *     serves it), and a pending RO request coalesces it.  Sending a
     * SEPARATE RO request when an RW request is in flight would home-count an r
     * that is then served via RW⊇RO and never RO_RETURNed (it is not a leased
     *     copy) — stranding the home's r and deadlocking the RO phase. */
    if (rws == CACHE_ST_GRANT || ros == CACHE_ST_GRANT) {
      act = CACHE_ACT_DRAIN_RO; /* RW⊇RO local join, or RO copy held */
    } else if (own == 1u) {
      /* local hit: this rank owns the data → read locally.  The decision keys
       * ONLY off the owner-bit; it must NOT touch ro_st.  Overwriting an
       * outstanding ro_st==REQ (a home RO request already in flight) to GRANT
       * would (1) let a later reader send a SECOND RO request once REL_RO
       * clears it, double-counting the home's r, and (2) leave a stale
       * ro_st==GRANT behind after this rank migrates away, so a later reader
       * reads stale local data via the ro_st==GRANT branch above. */
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
    /* Mirror EAGER REL_RW: the grant ends only when BOTH wc==0 AND rc==0 — RO
     * joiners (RW⊇RO) may still hold it after the last writer leaves.  wc==0
     * but rc>0 leaves rws==GRANT; the migrate ships at the rc 0-edge (REL_RO).
     */
    if (wc == 0u && rc == 0u) {
      rws = CACHE_ST_IDLE;
      if (mt != ARTS_LOCK_NO_TARGET) {
        /* 0-edge with a pending migration: clear owner-bit + migrate_target in
         * the SAME next-state (single CAS) and signal the ship.  This is the
         * LOCK-LAZY analogue of MRNEW's 0-edge transfer, but a single CAS — no
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
      /* Mirror EAGER REL_RO: the readers may have held a standalone RO grant
       * (ros==GRANT) OR joined an RW grant (rws==GRANT, RW⊇RO).  Clear
       * whichever is held; the last RO joiner under an RW grant completes that
       * grant. The action drives owner_try_execute (migrate if mt && wc==0 &&
       * rc==0) and, if this rank owed a home-counted RO_RETURN (ro_leased),
       * sends it. */
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
  /* Preserve the ro_leased flag (spare bit 63 — see CACHE_LEASED) across every
   * transition EXCEPT the RO release 0-edge, which fully relinquishes the RO
   * grant and clears it.  The flag is SET by the DELIVER(RO) handler when a
   * lent RO copy is actually received (a borrower owes exactly one RO_RETURN) —
   * NOT at request time, and NEVER for an owner serving its own readers locally
   * (RW⊇RO / fast-path: the owner holds the data, it borrowed nothing).
   * CACHE_MAKE_FULL produces bit63==0, so clearing is the default; we OR the
   * bit back only while the grant is still (partially) held. */
  uint64_t next = CACHE_MAKE_FULL(own, rws, ros, mt, wc, rc);
  if (!(op == CACHE_OP_REL_RO && rc == 0u)) {
    next |= (cur & CACHE_LAZY_LEASED_MASK);
  }
  return next;
}

/* ===== arts_db_acquire_is_serialized ===================================
 * LOCK: both RW and RO are blocking locks — both are GUID-serialized so the
 * engine acquires them in a global order (deadlock-free lock ordering). */
bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  return mode == DB_MODE_RW || mode == DB_MODE_RO;
}

/* ===== arts_db_cache_init ==============================================
 * Initialize the per-rank cache for the LOCK-LAZY protocol.
 *
 * Creator hold: arts_db_create defaults to an RW acquire (OCR contract; only
 * ARTS_DB_PROP_NO_ACQUIRE skips it).  The creator is the first data owner, so
 * we seed cache_state with owner-bit=1, rw_st=GRANT, wc=1, rc=0,
 * migrate_target=NO_TARGET.  The matching arts_db_release drives wc back to 0
 * (a sticky release — the creator stays the owner).  Non-creator (LAZY or
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
      struct arts_db_lock_waiter_s *w =
          ARTS_CONTAINER_OF(node, struct arts_db_lock_waiter_s, link);
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
 * LAZY home init: the creator is the first data owner.  arts_db_home_init
 * already seeds lock_state with the idle directory (LOCK_MAKE(PH_IDLE,
 * creator, 0, 0)); this coalesce-path leaf re-publishes owner=creator into the
 * home word (the home_initialized==true branch of the create handler). */
void arts_db_create_publish_holder(struct arts_db_s *db,
                                   unsigned int creator_rank) {
  /* Re-seed the home directory to phase=IDLE, owner=creator, w=0, r=0 — the
   * same idle-directory state arts_db_home_init installs under LAZY (the
   * creator's RW hold/release are cache-local/sticky and never reach home, so
   * home must not count it; owner names the migrate/serve FORWARD recipient).
   * Single coalesce-path store (the route slot is freshly promoted), no CAS
   * needed. */
  atomic_store_explicit(&db->lock_state,
                        LOCK_MAKE(LOCK_PHASE_IDLE, creator_rank, 0u, 0u),
                        memory_order_release);
}

/* ===== arts_db_create_install_home_buffer ===============================
 * LAZY: data stays with the owner (the creator); the home holds no canonical
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
    struct arts_db_lock_waiter_s *w =
        ARTS_CONTAINER_OF(node, struct arts_db_lock_waiter_s, link);
    mark_edt_secured_by_guid(w->edt_guid, w->slot);
    mark_edt_ready_by_guid(w->edt_guid, w->slot);
    arts_free(w);
    node = nx;
  }
}

/* ===== owner_try_execute ===============================================
 * The serve-RO + RW-migration driver, mirroring the model owner_try_execute.
 * Runs whenever an event may have made the owner ready to ship: after a
 * FORWARD installs a pending action, after a DELIVER makes this rank the owner,
 * or at a release 0-edge.
 *
 *   - RW migration: a pending migrate_target with wc==0 (and rc==0) ships the
 *     buffer to the target via DELIVER(RW).  Decided as the single-CAS REL_RW
 *     0-edge (arts_db_release_rw) — owner_try_execute handles the case where
 * the FORWARD lands while wc is already 0 (no live writer to release), by
 *     re-running the REL_RW-style 0-edge decision here.
 *   - RO serve: drain ro_serve and DELIVER(RO copy) to each reader, gated on
 *     wc==0 (no live writer).  push-then-recheck (in the handler) closes the
 *     stranded-reader window.
 *
 * Reads a single snapshot of cache_state; the migrate ship clears owner-bit +
 * migrate_target via one CAS (publish-before-send).  The RO drain is gated on
 * wc==0 read from the same word. */
static void owner_try_execute(struct arts_db_cache_s *cache) {
  /* StoreLoad fence: every caller publishes its work (a Treiber push onto
   * ro_serve, or a cache_state CAS) BEFORE calling here, then this routine
   * re-reads cache_state with a plain acquire-load to decide whether to drain.
   * On a weak-memory model (e.g. ARMv8.1+) a release-store followed by an
   * acquire-load to a DIFFERENT location is not ordered, so the gate-load could
   * hoist above the publishing store: a reader pushed onto ro_serve just after
   * a releaser flipped rw_st could be seen by neither (the pushing thread reads
   * a stale GRANT and skips the drain; the releasing thread's drain misses the
   * not-yet-visible node) — the reader strands and the RO phase deadlocks.  A
   * full fence closes that window.  Same hazard, same fix as the OoO defer
   * push-then-recheck (ooo.c).  No-op on x86-TSO, where the publishing
   * lock-prefixed RMW already fences. */
  atomic_thread_fence(memory_order_seq_cst);
  /* (1) RW migration: a pending migrate_target ships the buffer to the target
   * once this rank has no live readers.  Two cases:
   *
   *   target != self (a real migration): also gate on wc==0 (a local writer
   *     must finish before the data leaves), clear owner-bit + migrate_target,
   *     DELIVER(RW) to the target.  The new owner CONFIRMs home.
   *
   *   target == self (self-migration / owner self-RW-after-RO): home granted
   * the RW phase to the rank that already holds the data — the owner requested
   * RW while blocked by its own readers (cache RW fast-path needs rc==0).  The
   *     data must NOT move; instead, once the readers drain (rc==0), grant the
   *     parked RW writers LOCALLY (keep owner-bit, set rw_st=GRANT, drain
   *     rw_pending) and CONFIRM home so it advances the phase.  Here the local
   *     wc is the writer COHORT being granted, NOT a blocker — so this case
   * does NOT gate on wc==0 (gating on it would deadlock: the parked writer's
   *     acquire-time wc never reaches 0 on its own).  Mirrors the model
   *     issue_acquire→REQUEST→FORWARD(migrate→self)→owner_try_execute path with
   *     grant-time counting, made deadlock-free under acquire-time counting. */
  for (;;) {
    uint64_t cur =
        atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    uint32_t mt = CACHE_MIGRATE_TARGET(cur);
    /* Reader gate = "a LIVE granted reader is holding the buffer" =
     * ro_st==GRANT && rc>0, NOT rc!=0 alone.  The rc field also counts a
     * PENDING (not-yet-granted) RO request (ro_st==REQ, bumped at the acquire
     * cas1 before any DELIVER) — that must NOT block migration (it is held at
     * the home for the eventual RO phase).  And ro_st==GRANT with rc==0 is an
     * ORPHAN grant (a late DELIVER(RO) whose RO was already served+released via
     * an RW⊇RO local join) — no live reader, so it must NOT block migration
     * either.  Only a granted reader actually reading (ro_st==GRANT && rc>0)
     * blocks the ship. */
    if (mt == ARTS_LOCK_NO_TARGET || CACHE_OWNER(cur) == 0u ||
        (CACHE_RO_ST(cur) == CACHE_ST_GRANT && CACHE_RO_CNT(cur) != 0u)) {
      break; /* no eligible migration (or a live granted reader still holds it)
              */
    }
    if (mt == arts_global_rank_id) {
      /* Self-migration → local RW grant.  Keep owner-bit + counts; set
       * rw_st=GRANT, clear migrate_target in one CAS, then drain rw_pending and
       * CONFIRM home.  (No data ship.)  Does NOT require wc==0. */
      uint64_t next = CACHE_MAKE_FULL(1u, CACHE_ST_GRANT, CACHE_RO_ST(cur),
                                      ARTS_LOCK_NO_TARGET, CACHE_RW_CNT(cur),
                                      CACHE_RO_CNT(cur)) |
                      (cur & CACHE_LAZY_LEASED_MASK); /* preserve ro_leased */
      if (atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur, next,
                                                memory_order_acq_rel,
                                                memory_order_acquire)) {
        lock_drain_pending(&cache->rw_pending);
        arts_send_db_lock_confirm(arts_guid_get_rank(cache->db_guid),
                                  cache->db_guid);
        break;
      }
      continue; /* CAS lost — re-snapshot. */
    }
    /* Real migration to another rank: gate on "no live GRANTED writer"
     * (rw_st!=GRANT), NOT wc!=0.  A GRANTED writer is mid-write and must finish
     * before the data leaves (that ship is driven by the REL_RW 0-edge, not
     * here).  But wc also counts PENDING RW requests (rw_st==REQ, bumped at the
     * acquire cas1) — e.g. a sticky owner with a pending migrate_target that
     * takes new local RW acquires routes them to the home as REQ and bumps wc.
     * Those pending writers are NOT using the buffer and must not block the
     * ship; they stay parked in rw_pending (their wc is preserved across the
     * migration, NOT zeroed) and are re-granted when this rank re-acquires
     * ownership via a later DELIVER(RW).  Gating on wc!=0 instead would
     * deadlock the chain (the owner can never reach wc==0 while it keeps
     * accepting queued RW acquires). Preserve rw_st/wc/rc and ro_leased; clear
     * only owner-bit + migrate_target.
     */
    if (CACHE_RW_CNT(cur) != 0u) { /* AB-TEST: revert to wc==0 gate */
      break;
    }
    uint64_t next =
        CACHE_MAKE_FULL(0u, CACHE_ST_IDLE, CACHE_RO_ST(cur),
                        ARTS_LOCK_NO_TARGET, 0u, CACHE_RO_CNT(cur)) |
        (cur & CACHE_LAZY_LEASED_MASK); /* preserve ro_leased */
    if (atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur, next,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      /* owner-bit + migrate_target cleared (published) BEFORE the DELIVER
       * ships — a concurrent acquire now sees not-owner and routes via home. */
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      struct arts_db_buffer_s *buf =
          (struct arts_db_buffer_s *)arts_shared_get(buf_h);
      const void *data = (buf != NULL) ? buf->data : NULL;
      uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
      arts_send_db_lock_deliver(mt, cache->db_guid, (uint32_t)DB_MODE_RW, data,
                                ds);
      arts_db_buf_release(&buf_h);
      break;
    }
    /* CAS lost — re-snapshot. */
  }

  /* (2) RO serve: drain ro_serve and ship an RO copy to each reader, gated on
   * "no live GRANTED writer" (rw_st!=GRANT).  As with the reader gate above,
   * the wc field counts a *pending* RW request (rw_st==REQ, bumped at acquire
   * cas1) as well as a granted one; the model gates RO serve on local_wc
   * (GRANTED writers only), so the faithful C analogue is rw_st!=GRANT.  A
   * reader pushed after this drain (during an active RW phase) is served by the
   * handler's push-then-recheck or a later owner_try_execute. */
  uint64_t cur =
      atomic_load_explicit(&cache->cache_state, memory_order_acquire);
  if (CACHE_OWNER(cur) == 1u && CACHE_RW_ST(cur) != CACHE_ST_GRANT) {
    arts_lf_link_t *node = arts_lf_stack_drain(&cache->ro_serve);
    if (node != NULL) {
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      struct arts_db_buffer_s *buf =
          (struct arts_db_buffer_s *)arts_shared_get(buf_h);
      const void *data = (buf != NULL) ? buf->data : NULL;
      uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
      while (node != NULL) {
        arts_lf_link_t *nx =
            atomic_load_explicit(&node->next, memory_order_relaxed);
        struct arts_lock_ro_serve_node_s *rn =
            ARTS_CONTAINER_OF(node, struct arts_lock_ro_serve_node_s, link);
        arts_send_db_lock_deliver(rn->rank, cache->db_guid,
                                  (uint32_t)DB_MODE_RO, data, ds);
        arts_free(rn);
        node = nx;
      }
      arts_db_buf_release(&buf_h);
    }
  }
}

/* ===== arts_send_db_lock_request ========================================
 * Send MSG_DB_LOCK_REQUEST to the home rank.  Self-send (home == this rank)
 * dispatches through the OoO engine (HIT runs inline; MISS defers until the
 * home db_s is installed) — a REQUEST can race the home CREATE on a cross-rank
 * lazy-install path.  Remote send goes via the transport. */
void arts_send_db_lock_request(unsigned int home_rank, arts_guid_t db_guid,
                               arts_db_access_mode_t mode) {
  if (home_rank == arts_global_rank_id) {
    struct arts_ooo_args_db_lock_request_s args = {
        .requester = arts_global_rank_id,
        .db_guid = db_guid,
        .mode = mode,
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

/* ===== arts_send_db_lock_forward ========================================
 * home → current owner.  Direct send (Cat-C): the owner's cache is always
 * installed (it became owner via DELIVER, or is the creator).  Self-send (owner
 * == home) calls the handler body directly (local hit, no wire). */
void arts_send_db_lock_forward(unsigned int owner_rank, arts_guid_t db_guid,
                               uint32_t mode, uint32_t target) {
  struct arts_msg_lock_forward_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_LOCK_FORWARD);
  p.db_guid = db_guid;
  p.mode = mode;
  p.target = target;
  if (owner_rank == arts_global_rank_id) {
    arts_handler_db_lock_forward(&p);
    return;
  }
  arts_transport_send_async((int)owner_rank, (char *)&p, sizeof(p));
}

/* ===== arts_send_db_lock_deliver ========================================
 * owner → target (+ inline data).  Direct send (Cat-C): the target's cache is
 * always installed (it sent its own REQUEST first, or is the migration target
 * named by home).  Versionless — exclusive-lock serialization guarantees no
 * stale write can race.  Self-send builds the contiguous (header + data) buffer
 * and calls the handler body directly. */
void arts_send_db_lock_deliver(unsigned int target_rank, arts_guid_t db_guid,
                               uint32_t mode, const void *data,
                               uint64_t data_size) {
  struct arts_msg_lock_deliver_packet_s p;
  uint64_t total = sizeof(p) + data_size;
  arts_fill_packet_header(&p.header, total, MSG_DB_LOCK_DELIVER);
  p.db_guid = db_guid;
  p.mode = mode;
  p.pad = 0;
  if (target_rank == arts_global_rank_id) {
    /* Self-send: build header + data contiguously and call the body directly.
     */
    char *buf = (char *)arts_malloc((size_t)total);
    memcpy(buf, &p, sizeof(p));
    if (data_size > 0 && data != NULL) {
      memcpy(buf + sizeof(p), data, (size_t)data_size);
    }
    arts_handler_db_lock_deliver(buf, (size_t)total);
    arts_free(buf);
    return;
  }
  if (data == NULL || data_size == 0) {
    arts_transport_send_async((int)target_rank, (char *)&p, sizeof(p));
    return;
  }
  /* Assemble header + payload into one fresh allocation (the transport stores
   * only the payload pointer; copying makes the packet self-contained and
   * independent of the owner buffer's lifetime). */
  char *pkt = (char *)arts_malloc((size_t)total);
  memcpy(pkt, &p, sizeof(p));
  memcpy(pkt + sizeof(p), data, (size_t)data_size);
  arts_transport_send_async((int)target_rank, pkt, (unsigned int)total);
  arts_free(pkt);
}

/* ===== arts_send_db_lock_confirm ========================================
 * new owner → home: migration complete.  Direct send (Cat-C): home db_s is
 * always installed (home sent the FORWARD that triggered the migration).
 * Self-send (new owner == home) calls the home handler body directly. */
void arts_send_db_lock_confirm(unsigned int home_rank, arts_guid_t db_guid) {
  struct arts_msg_lock_confirm_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_LOCK_CONFIRM);
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_lock_confirm(&p);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* ===== arts_send_db_lock_roret ==========================================
 * reader → home: RO release (data-less).  Direct send (Cat-C): home db_s is
 * always installed (home sent the FORWARD(serve) that granted this reader its
 * copy).  Self-send (reader == home) calls the home handler body directly. */
void arts_send_db_lock_roret(unsigned int home_rank, arts_guid_t db_guid) {
  struct arts_msg_lock_confirm_packet_s p; /* CONFIRM/RORET share the struct */
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_LOCK_RORET);
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_lock_roret(&p);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* ===== arts_handler_db_acquire =========================================
 * OOO_DB_ACQUIRE Cat-B body — protocol-agnostic signature.  item is the
 * pre-pinned home db_s; args is {edt, db_guid, slot}; mode is read from
 * depv[slot].mode.
 *
 * Two-CAS acquire (the EAGER LOCK discipline, LAZY arbiter):
 *   cas1  count++ ONLY (fetch_add), BEFORE the push, so a waiter present in a
 *         queue is already counted → a drained waiter cannot be released out
 *         from under by a concurrent holder (no premature grant release).
 *   push  the waiter (now drainable; already counted).
 *   cas2  state decision on the FRESH (post-push) state via
 *         cache_lazy_compute_next.  Owner fast-path → local GRANT (DRAIN);
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
                            (mode == DB_MODE_RW) ? CACHE_LAZY_RW_UNIT
                                                 : CACHE_LAZY_RO_UNIT,
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
    next = cache_lazy_compute_next(cur, op, &act);
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
  default: /* NONE: parked; a DELIVER / another acquire's drain serves us. */
    break;
  }
}

/* ===== arts_handler_db_lock_forward ====================================
 * Cat-C @owner.  item_v is the full wire packet (arts_msg_lock_forward_packet_s
 * *).  RW(mode) → install migrate_target (CAS into cache_state); RO(mode) →
 * push the reader rank onto ro_serve.  Then owner_try_execute drives the ship
 * (immediate if wc==0; otherwise the eventual release 0-edge does it).
 *
 * Cat-C route-table lookup: a NULL lookup means the DB was destroyed
 * concurrently → drop. */
void arts_handler_db_lock_forward(void *item_v) {
  struct arts_msg_lock_forward_packet_s *p =
      (struct arts_msg_lock_forward_packet_s *)item_v;
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(p->db_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (db == NULL) {
    arts_shared_release(&db_h);
    return;
  }
  struct arts_db_cache_s *cache = &db->cache;

  if ((arts_db_access_mode_t)p->mode == DB_MODE_RW) {
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
          (cur & CACHE_LAZY_LEASED_MASK); /* preserve ro_leased */
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
    arts_lf_stack_push(&cache->ro_serve, &rn->link);
  }

  owner_try_execute(cache);
  arts_shared_release(&db_h);
}

/* ===== arts_handler_db_lock_deliver ====================================
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
void arts_handler_db_lock_deliver(void *payload, size_t size) {
  struct arts_msg_lock_deliver_packet_s *p =
      (struct arts_msg_lock_deliver_packet_s *)payload;
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

  if (mode == DB_MODE_RW) {
    /* Migration is the ONLY thing that moves buffer data (owner→owner). Install
     * the migrated data BEFORE the CAS (no concurrent holder — migration is
     * CONFIRM-gated, single owner), then become owner: owner-bit + rw_st=GRANT
     * in one CAS (preserve counts + ro_st; migrate_target stays NO_TARGET — a
     * chained migrate is installed later by a fresh FORWARD). */
    if (data_size > 0u) {
      arts_db_buf_write_inplace(cache, data, data_size);
    }
    uint64_t cur, next;
    do {
      cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
      next = CACHE_MAKE_FULL(1u, CACHE_ST_GRANT, CACHE_RO_ST(cur),
                             CACHE_MIGRATE_TARGET(cur), CACHE_RW_CNT(cur),
                             CACHE_RO_CNT(cur)) |
             (cur & CACHE_LAZY_LEASED_MASK); /* preserve ro_leased */
    } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                    next, memory_order_acq_rel,
                                                    memory_order_acquire));
    /* RW grant serves this rank's RW + RO cohort (RW⊇RO): drain BOTH the parked
     * writers and any parked readers (a reader that requested RO before this
     * rank became the RW owner is parked in ro_pending and joins the RW grant
     * here).  Mirrors EAGER's GRANT_RW DRAIN_BOTH; without the ro_pending drain
     * those readers are stranded, rc never reaches 0, and the migration (gated
     * wc==0 && rc==0) deadlocks.  Counts were bumped at each acquire's cas1
     * ("queued ⟹ counted"). */
    lock_drain_pending(&cache->rw_pending);
    lock_drain_pending(&cache->ro_pending);
    /* Tell home the migration is complete so it can advance (RW chain / flip).
     */
    arts_send_db_lock_confirm(arts_guid_get_rank(p->db_guid), p->db_guid);
    /* A migrate_target / ro_serve may already be pending (home chained the next
     * action); drive it. */
    owner_try_execute(cache);
  } else {
    /* RO grant: a borrower may receive a read-only copy.  Decide on the CAS'd
     * state (calculate first, CAS, then act only on success):
     *   ro_return (rw_st==GRANT || rc==0): nothing to serve here.  rw_st==GRANT
     *       means this rank already holds the data as the RW owner (RW⊇RO
     * covers its readers); rc==0 means its readers were already served (e.g.
     * via an RW⊇RO join while it was a past owner).  In BOTH cases do NOT drain
     *       and do NOT install (installing would override valid owner data), do
     *       NOT take ros/ro_leased; set ros=IDLE and immediately return the
     *       unneeded grant (RO_RETURN) to balance the home's r.  This is the
     * one uniform handler for current-owner / past-owner / phantom — the home
     *       sends RO grants unconditionally and needs no owner tracking.
     *   serve (rw_st!=GRANT && rc>0): the rank held stale data, so install the
     *       granted copy (override is correct, like EAGER's RO-grant memcpy),
     * set ros=GRANT + ro_leased (owes one RO_RETURN when its readers drain),
     *       then serve them. */
    uint64_t cur, next;
    bool ro_return;
    do {
      cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
      ro_return =
          (CACHE_RW_ST(cur) == CACHE_ST_GRANT) || (CACHE_RO_CNT(cur) == 0u);
      next = CACHE_MAKE_FULL(CACHE_OWNER(cur), CACHE_RW_ST(cur),
                             ro_return ? CACHE_ST_IDLE : CACHE_ST_GRANT,
                             CACHE_MIGRATE_TARGET(cur), CACHE_RW_CNT(cur),
                             CACHE_RO_CNT(cur)) |
             (cur & CACHE_LAZY_LEASED_MASK);
      if (!ro_return) {
        next |= CACHE_LAZY_LEASED_MASK;
      }
    } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                    next, memory_order_acq_rel,
                                                    memory_order_acquire));
    if (ro_return) {
      arts_send_db_lock_roret(arts_guid_get_rank(p->db_guid), p->db_guid);
    } else {
      if (data_size > 0u) {
        arts_db_buf_write_inplace(cache, data, data_size);
      }
      lock_drain_pending(&cache->ro_pending);
    }
  }

  arts_shared_release(&db_h);
}

/* ===== arts_db_release_rw / arts_db_release_ro =========================
 * LAZY release — single-word CAS via cache_lazy_compute_next, then dispatch the
 * returned action.
 *
 * RW: wc-- (+ 0-edge migration decision) in ONE CAS.  CACHE_ACT_MIGRATE means
 *     the 0-edge cleared owner-bit + migrate_target in the same next-state
 *     (publish-before-send); ship DELIVER(RW) to the target.  No home
 * writeback, no ACK (the EAGER ACK path is not linked in LAZY) — data is sticky
 * on the owner.  No pending target → sticky (no-op). RO: rc--; on the 0-edge —
 * if this rank is still the owner, owner_try_execute (drive any pending RO
 * serve / migration now that readers are gone); else RORET to home (a remote RO
 * copy holder returning its lease).
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
    next = cache_lazy_compute_next(cur, CACHE_OP_REL_RW, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_MIGRATE) {
    /* owner-bit + migrate_target already cleared (published) inside the CAS;
     * ship the buffer to the target now. */
    arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
    struct arts_db_buffer_s *buf =
        (struct arts_db_buffer_s *)arts_shared_get(buf_h);
    const void *data = (buf != NULL) ? buf->data : NULL;
    uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
    arts_send_db_lock_deliver(target, cache->db_guid, (uint32_t)DB_MODE_RW,
                              data, ds);
    arts_db_buf_release(&buf_h);
  } else if (CACHE_RW_CNT(next) == 0u && CACHE_OWNER(next) == 1u) {
    /* Sticky 0-edge (no pending migration): the writer phase just ended and we
     * remain the owner.  Drive any RO readers the home queued via
     * FORWARD(serve) during the RW phase (the RW→RO flip): they sit in ro_serve
     * until wc==0. Mirrors the model release_rw (owner_try_execute on the wc==0
     * edge while still owner).  No-op single-node (ro_serve empty). */
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
    next = cache_lazy_compute_next(cur, CACHE_OP_REL_RO, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_REL_RO) {
    /* rc 0-edge.  The RO_RETURN decision keys off ro_leased (CACHE_LEASED of
     * the pre-CAS state that committed), NOT the owner-bit: a home-COUNTED RO
     * grant (one that arrived via DELIVER(RO)) owes the home a RO_RETURN to
     * balance its r, even if this rank is now the owner (a held reader that was
     * promoted to owner at the RW→RO flip self-served its own data — without
     * this RO_RETURN the home's r would never reach 0 and the RO→RW flip would
     * deadlock).  An owner-fast-path local RO (never leased, never counted at
     * home) sends none. Independently, if this rank is still the owner, drive
     * any pending RO serve / migration now that the local readers have drained.
     */
    bool leased = CACHE_LEASED(cur);
    if (leased) {
      arts_send_db_lock_roret(arts_guid_get_rank(cache->db_guid),
                              cache->db_guid);
    }
    uint64_t now =
        atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_OWNER(now) == 1u) {
      owner_try_execute(cache);
    }
  }
}

/* ===== lock_lazy_compute_next ==========================================
 * LAZY home arbiter — pure function of the single lock_state word
 * [phase:2 | owner:14 | w:24 | r:24], run inside a CAS-retry loop.  This IS the
 * design §3.6 phase-transition table; it mirrors the verified model checker
 * (2026-06-24-lock-lazy-modelcheck.py) function lock_lazy_compute_next 1:1.
 *
 * The transitions (model verbatim):
 *   RW_REQ  : w++; phase==IDLE → phase=RW, action=MIGRATE.  (phase==RW: just
 *             queue, next migrate fires at CONFIRM; phase==RO: hold, r
 * untouched) RO_REQ  : r++; phase==RO → SERVE_ONE; phase==IDLE → phase=RO +
 * SERVE_ONE. (phase==RW: held in ro_waiters by the caller, no action) CONFIRM :
 * w--, owner=new_owner (ONE CAS); w>0 → MIGRATE (RW chain); elif r>0 → phase=RO
 * + SERVE_ALL (RW→RO flip); else phase=IDLE. RO_RET  : r--; on r==0: w>0 →
 * phase=RW + MIGRATE (RO→RW flip); else phase=IDLE.
 *
 * Equivalences (§3.2): mig_inflight == (phase==RW), serving_ro == (phase==RO),
 * r == the old rc — no separate flags; everything lives in the word.
 *
 * The FORWARD recipient is OWNER() of the returned word.  The migrate target /
 * served reader(s) are resolved by the caller from rw_waiters/ro_waiters. */
uint64_t lock_lazy_compute_next(uint64_t cur, int op, unsigned int new_owner,
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
 * word lock_state CAS via lock_lazy_compute_next, then dispatches the returned
 * FORWARD action to the current owner (OWNER() of the freshly-CAS'd word),
 * resolving the migrate target / served reader(s) from the lock-free queues.
 *
 * These mirror the verified model functions home_on_REQUEST / home_on_CONFIRM /
 * home_on_RO_RETURN.  They run CONCURRENTLY (multiple receiver threads);
 * correctness rests on the single-word CAS + the lock-free queues, exactly like
 * EAGER — no mutex.  The protocol is reorder-tolerant (no per-(src,dst) FIFO):
 * r counts outstanding RO grants and may transiently exceed the rank count
 * under reorder; r==0 (not RO_RETURN ordering) gates the RW flip. */

/* ===== RO waiter node (home-side held-RO queue entry) ==================
 * Held-RO readers (an RO REQUEST that arrived during the RW phase) wait in
 * db->ro_waiters until the RW→RO flip drains them.  Same shape as the
 * file-local node in home.c / eager.c (ro_waiters is the Treiber db->ro_waiters
 * those TUs also use); the SERVE_ONE path serves the requester directly and
 * never touches this queue, so the only producers/consumers are the home
 * REQUEST hold-push and the CONFIRM SERVE_ALL drain. */
struct arts_lock_ro_node_s {
  arts_lf_link_t link; /* FIRST */
  unsigned int rank;
};

/* Dispatch a MIGRATE or SERVE_ALL FORWARD action after the lock_state CAS
 * committed.  `word` is the freshly-CAS'd lock_state (OWNER() = the current
 * owner to FORWARD to).  SERVE_ONE is NOT handled here — it serves the
 * requesting rank directly (the rank is not in the word; see the RO arm of
 * arts_handler_db_lock_request, model home_on_REQUEST). */
static void lock_lazy_home_forward(struct arts_db_s *db, uint32_t action,
                                   uint64_t word) {
  unsigned int owner = LOCK_OWNER(word);
  if (action == LOCK_ACTION_FORWARD_MIGRATE) {
    /* Migrate RW ownership to the front of the RW waiter queue.  peek (not
     * pop): the requester stays queued until its CONFIRM pops it (model
     * invariant "CONFIRM front == new owner"). */
    unsigned int target;
    if (arts_home_lockreq_queue_peek(&db->rw_waiters, &target)) {
      arts_send_db_lock_forward(owner, db->cache.db_guid, (uint32_t)DB_MODE_RW,
                                target);
    }
  } else if (action == LOCK_ACTION_FORWARD_SERVE_ALL) {
    /* RW → RO flip: drain every held RO waiter and FORWARD-serve each
     * UNCONDITIONALLY (the fixed single-target packet, same shape as
     * SERVE_ONE). No owner identity tracking here: a rank that is (or was) the
     * owner gets the grant like any other, and its cache returns it immediately
     * (RO_RETURN) when it already holds the data (rw_st==GRANT) or has nothing
     * to serve (rc==0) — see arts_handler_db_lock_deliver's RO arm.  Keeping
     * the home side unconditional avoids tracking owner identity across
     * migrations. */
    arts_lf_link_t *node = arts_lf_stack_drain(&db->ro_waiters);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_lock_ro_node_s *rn =
          ARTS_CONTAINER_OF(node, struct arts_lock_ro_node_s, link);
      arts_send_db_lock_forward(owner, db->cache.db_guid, (uint32_t)DB_MODE_RO,
                                rn->rank);
      arts_free(rn);
      node = nx;
    }
  }
}

/* ===== arts_handler_db_lock_request ====================================
 * Cat-B @home (OoO OOO_DB_LOCK_REQUEST).  item_v is the pre-pinned home db_s;
 * args_v is {requester, db_guid, mode}.  Mirrors the model home_on_REQUEST.
 *
 * RW: push the requester onto rw_waiters (push-before-CAS: "queued ⟹ counted"),
 *     then CAS lock_lazy_compute_next(RW_REQ).  MIGRATE (idle→RW) → FORWARD the
 *     owner to migrate to the queue front (model: w.rw_waiters[0]).
 * RO: CAS lock_lazy_compute_next(RO_REQ) first; on SERVE_ONE (RO/idle phase)
 *     push the reader THEN FORWARD-serve it; on NONE (RW phase) hold it in
 *     ro_waiters for the eventual RW→RO flip (model home_on_REQUEST RO arm).
 *
 * The owner FORWARD recipient is OWNER() of the freshly-CAS'd word. */
void arts_handler_db_lock_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_lock_request_s *a =
      (struct arts_ooo_args_db_lock_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)a->mode;

  arts_rank_bitset_set(&db->cached_ranks,
                       requester); /* destroy fan-out roster */

  uint32_t action;
  uint64_t cur, next;
  if (mode == DB_MODE_RW) {
    /* push-before-CAS: the requester is queued (and thus counted by w) before
     * the transition reads w, so a concurrent CONFIRM/peek sees it. */
    arts_home_lockreq_queue_push(&db->rw_waiters, requester);
    do {
      cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
      next = lock_lazy_compute_next(cur, LOCK_OP_RW_ACQ, /*new_owner=*/0u,
                                    &action);
    } while (!atomic_compare_exchange_weak_explicit(&db->lock_state, &cur, next,
                                                    memory_order_acq_rel,
                                                    memory_order_acquire));
    lock_lazy_home_forward(db, action, next);
  } else {
    /* RO: push-before-CAS — mirror the RW arm (and EAGER's home request, which
     * pushes BOTH modes before the CAS).  Enqueue the reader in ro_waiters
     * BEFORE the r++ CAS, so any transition that observes this r++ (a
     * concurrent CONFIRM RW→RO flip's SERVE_ALL, or this request's own serve)
     * also finds the node already queued and serves it from the FRESHLY-CAS'd
     * owner. Serving is always a drain of ro_waiters; there is no direct "serve
     * the requester" path, hence no stale-owner snapshot and no post-push
     * recheck (the old CAS-then-push + recheck band-aid could FORWARD to an
     * owner that had since migrated, stranding the reader and the home's r). */
    struct arts_lock_ro_node_s *n =
        (struct arts_lock_ro_node_s *)arts_malloc(sizeof(*n));
    n->rank = requester;
    arts_lf_stack_push(&db->ro_waiters, &n->link);
    do {
      cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
      next = lock_lazy_compute_next(cur, LOCK_OP_RO_ACQ, /*new_owner=*/0u,
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
      lock_lazy_home_forward(db, LOCK_ACTION_FORWARD_SERVE_ALL, next);
    }
  }
}

/* ===== arts_handler_db_lock_confirm ====================================
 * Cat-C @home (direct dispatch).  item_v is the full wire packet
 * (arts_msg_lock_confirm_packet_s *); the new owner is packet->header.rank (the
 * sender — both wire and self-send set it via arts_fill_packet_header). Mirrors
 * the model home_on_CONFIRM.
 *
 * Cat-C route-table lookup: NULL ⇒ DB destroyed concurrently ⇒ drop.  pop the
 * RW waiter (assert front == src), then CAS lock_lazy_compute_next(CONFIRM,
 * new_owner=src) — this sets owner=src + w-- + next phase + action in ONE CAS.
 * Dispatch: MIGRATE → next RW migrate; SERVE_ALL → RW→RO flip drain. */
void arts_handler_db_lock_confirm(void *item_v) {
  struct arts_msg_lock_confirm_packet_s *p =
      (struct arts_msg_lock_confirm_packet_s *)item_v;
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
  if (!arts_home_lockreq_queue_pop(&db->rw_waiters, &front)) {
    arts_shared_release(&db_h);
    return; /* stale / duplicate CONFIRM (empty RW queue) — drop */
  }
  /* front == new_owner under the protocol; the pop just dequeued it. */
  (void)front;

  uint32_t action;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    next = lock_lazy_compute_next(cur, LOCK_OP_CONFIRM, new_owner, &action);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_acq_rel, memory_order_acquire));
  lock_lazy_home_forward(db, action, next);
  arts_shared_release(&db_h);
}

/* ===== arts_handler_db_lock_roret ======================================
 * Cat-C @home (direct dispatch).  item_v is the full wire packet
 * (arts_msg_lock_confirm_packet_s *, shared with CONFIRM); the returning reader
 * is packet->header.rank (informational — RO accounting is rank-agnostic).
 * Mirrors the model home_on_RO_RETURN.
 *
 * CAS lock_lazy_compute_next(RO_RET) — r--; on r→0 with RW waiting, flip phase
 * to RW and MIGRATE to the next RW waiter (RO→RW flip).  Dispatch the action.
 */
void arts_handler_db_lock_roret(void *item_v) {
  struct arts_msg_lock_confirm_packet_s *p =
      (struct arts_msg_lock_confirm_packet_s *)item_v;
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
        lock_lazy_compute_next(cur, LOCK_OP_RO_RET, /*new_owner=*/0u, &action);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_acq_rel, memory_order_acquire));
  lock_lazy_home_forward(db, action, next);
  arts_shared_release(&db_h);
}
