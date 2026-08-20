/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_shared_ptr_t — atomic_shared_ptr via SPLIT REFERENCE COUNTING.
 * Williams "C++ Concurrency in Action" Sec. 7.2.4, with two refinements that
 * the textbook stack version handles structurally but a single atomic pointer
 * over a recycling allocator does not:
 *
 *   (1) GENERATION TAG (ABA).  The 16-byte DWCAS slot is {cb*, ext}, where
 *       ext = (generation << 16) | claims.  `generation` is bumped on every
 *       store/exchange/compare_exchange.  A control block freed (back to
 *       mimalloc) and recycled to the SAME address is reinstalled under a NEW
 *       generation, so a stalled load holding a stale {cb, ext} can never
 *       DWCAS-claim the recycled cb — what the old type-stable cb pool used to
 *       guarantee structurally.  `claims` is the transient in-flight LOAD count
 *       (≪ 2^16): a load DWCAS-bumps claims (no cb dereference — immune to a
 *       concurrent free), pinning the cb, then dereferences it to fold the
 *       claim into the cb's count and hand back a plain owned cb pointer.
 *
 *   (2) TWO-FIELD COUNT (transient-zero free).  The cb count packs an `in_slot`
 *       presence bit (bit 0) under the reference count: count = (refs << 1) |
 *       in_slot.  The cb is freed ONLY when count == 0 (refs == 0 AND not in a
 *       slot).  This is essential: when a store removes the cb it must fold the
 *       in-flight claims into refs AND drop the slot hold, but those folds race
 *       the in-flight loads' own fold/undo.  Without the presence bit, refs
 *       could transiently hit 0 mid-reconcile and free a still-referenced cb
 *       (a real heap-use-after-free).  The in_slot bit keeps count odd (≠ 0)
 *       for as long as the cb sits in a slot, so the free can only fire after
 *       the reconcile atomically clears in_slot and folds the claims in one
 *       fetch_add.  Reference deltas are therefore scaled by 2 (SHARED_REF);
 *       installing sets the hold (fetch_add -1: one ref → in_slot); reconcile
 *       clears it (fetch_add 2*claims - 1).
 *
 * No type-stable pool ⇒ per-thread (NUMA-local) cb alloc/free scaling instead
 * of a single global pool head.
 */

#include "arts/utils/shared.h"

#include "arts/utils/malloc.h" /* arts_calloc / arts_free */

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* count = (refs << 1) | in_slot.  One reference unit is 2; the in_slot
 * presence bit is 1.  Freed iff count == 0 (no refs, not in a slot). */
#define SHARED_REF 2

struct arts_shared_s {
  _Atomic(int64_t) count; /* (refs << 1) | in_slot.  Signed: refs may dip
                             negative transiently mid-reconcile; only the exact
                             0 (both fields zero) frees. */
  void *object;
  void (*deleter)(void *);
  _Atomic(uint64_t) tag; /* identity stamp — see shared.h */
};

/* ext = (generation << 16) | claims. */
#define SHARED_CLAIMS_BITS 16
static inline uint64_t shared_claims(uint64_t ext) { return ext & 0xFFFFu; }
static inline uint64_t shared_gen(uint64_t ext) {
  return ext >> SHARED_CLAIMS_BITS;
}
/* The ext to install on a store: bump the generation, reset claims to 0. */
static inline uint64_t shared_next_gen_ext(uint64_t ext) {
  return (shared_gen(ext) + 1u) << SHARED_CLAIMS_BITS;
}

/* fetch_add `delta` onto count; free on the transition to exactly 0. */
static void shared_count_add(struct arts_shared_s *cb, int64_t delta) {
  int64_t prev =
      atomic_fetch_add_explicit(&cb->count, delta, memory_order_acq_rel);
  if (prev + delta == 0) {
    if (cb->deleter) {
      cb->deleter(cb->object);
    }
    arts_free(cb);
  }
}

/* ── Local API (single-owner) ──────────────────────────────────────────── */

arts_shared_ptr_t arts_shared_make(void *object, void (*deleter)(void *)) {
  /* arts_calloc (not arts_malloc): the standalone shared-ptr unit tests are
   * library-independent and shim arts_calloc/arts_free; the zero-init is also a
   * cheap safety net before the fields are set below. */
  struct arts_shared_s *cb =
      (struct arts_shared_s *)arts_calloc(1, sizeof(struct arts_shared_s));
  atomic_store_explicit(&cb->count, SHARED_REF,
                        memory_order_relaxed); /* refs=1 */
  cb->object = object;
  cb->deleter = deleter;
  return cb;
}

arts_shared_ptr_t arts_shared_copy(arts_shared_ptr_t b) {
  if (!b) {
    return NULL;
  }
  /* Caller already holds b ⇒ refs ≥ 1 ⇒ cb alive; a plain add is safe. */
  atomic_fetch_add_explicit(&b->count, SHARED_REF, memory_order_relaxed);
  return b;
}

void arts_shared_release(arts_shared_ptr_t *p) {
  arts_shared_ptr_t cb = *p;
  if (!cb) {
    return;
  }
  *p = NULL;
  shared_count_add(cb, -SHARED_REF);
}

void arts_shared_abandon(arts_shared_ptr_t *p) {
  arts_shared_ptr_t cb = *p;
  if (!cb) {
    return;
  }
  *p = NULL;
  /* Unpublished cb (refs == 1, never stored/copied): free WITHOUT running the
   * deleter — the wrapped object stays owned by the caller. */
  arts_free(cb);
}

void *arts_shared_get(arts_shared_ptr_t p) { return p ? p->object : NULL; }

/* Relaxed on both sides: publication order rides the slot.  The stamp is
 * sequenced before the release-CAS that publishes the cb into a slot, and a
 * reader's tag load is sequenced after the acquire load that pinned the cb
 * out of that slot, so the pairing on the slot carries the stamp across. */
void arts_shared_set_tag(arts_shared_ptr_t p, uint64_t tag) {
  atomic_store_explicit(&p->tag, tag, memory_order_relaxed);
}

uint64_t arts_shared_tag(arts_shared_ptr_t p) {
  return atomic_load_explicit(&p->tag, memory_order_relaxed);
}

/* ── Atomic slot API (multi-thread shared) ─────────────────────────────── */

arts_shared_ptr_t arts_atomic_shared_load(arts_atomic_shared_ptr_t *slot) {
  arts_shared_slot_t cur = atomic_load_explicit(slot, memory_order_acquire);
  /* Claim: DWCAS claims++ (pure slot CAS, no cb dereference). */
  for (;;) {
    if (cur.cb == NULL) {
      return NULL;
    }
    arts_shared_slot_t claimed = {cur.cb, cur.ext + 1u};
    if (atomic_compare_exchange_weak_explicit(
            slot, &cur, claimed, memory_order_acq_rel, memory_order_acquire)) {
      cur = claimed;
      break;
    }
    /* cur reloaded by the failed CAS — retry. */
  }
  struct arts_shared_s *cb = cur.cb; /* pinned: claims ≥ 1 keeps it alive. */
  uint64_t claim_gen = shared_gen(cur.ext);
  /* Fold the claim into a stable owned ref (cb is safe to dereference now). */
  atomic_fetch_add_explicit(&cb->count, SHARED_REF, memory_order_relaxed);
  /* Release the claim while the slot still holds cb at the SAME generation.  If
   * a store has bumped the generation (and possibly recycled cb), it already
   * folded our claim into refs — so undo our extra fold instead. */
  for (;;) {
    if (cur.cb != cb || shared_gen(cur.ext) != claim_gen) {
      /* The reconcile already folded our claim as our owned ref (count holds
       * it), so this undo never frees; never check for 0. */
      atomic_fetch_sub_explicit(&cb->count, SHARED_REF, memory_order_relaxed);
      break;
    }
    arts_shared_slot_t released = {cb, cur.ext - 1u};
    if (atomic_compare_exchange_weak_explicit(
            slot, &cur, released, memory_order_acq_rel, memory_order_acquire)) {
      break; /* released our claim; we own a folded ref. */
    }
    /* cur reloaded — claims or generation changed; retry. */
  }
  return cb;
}

/* Install new_val into the slot (one of new_val's refs becomes the slot hold,
 * bumping the generation) and return the displaced value.  Caller has already
 * set new_val's in_slot bit (or new_val is NULL).  Shared by store/exchange. */
static arts_shared_slot_t shared_slot_install(arts_atomic_shared_ptr_t *slot,
                                              arts_shared_ptr_t new_val) {
  arts_shared_slot_t cur = atomic_load_explicit(slot, memory_order_acquire);
  for (;;) {
    arts_shared_slot_t desired = {new_val, shared_next_gen_ext(cur.ext)};
    if (atomic_compare_exchange_weak_explicit(
            slot, &cur, desired, memory_order_acq_rel, memory_order_acquire)) {
      return cur; /* the displaced value. */
    }
    /* cur reloaded — retry. */
  }
}

void arts_atomic_shared_store(arts_atomic_shared_ptr_t *slot,
                              arts_shared_ptr_t new_val) {
  if (new_val != NULL) {
    /* One of new_val's refs becomes the slot hold (refs-- , in_slot=1). */
    atomic_fetch_add_explicit(&new_val->count, -1, memory_order_relaxed);
  }
  arts_shared_slot_t old = shared_slot_install(slot, new_val);
  if (old.cb != NULL) {
    /* old leaves the slot: clear in_slot (-1) and fold in-flight claims
     * (+2*claims); free if that was the last reference. */
    shared_count_add(old.cb, 2 * (int64_t)shared_claims(old.ext) - 1);
  }
}

arts_shared_ptr_t arts_atomic_shared_exchange(arts_atomic_shared_ptr_t *slot,
                                              arts_shared_ptr_t new_val) {
  if (new_val != NULL) {
    atomic_fetch_add_explicit(&new_val->count, -1, memory_order_relaxed);
  }
  arts_shared_slot_t old = shared_slot_install(slot, new_val);
  if (old.cb != NULL) {
    /* Clear in_slot and fold claims, but KEEP the hold as a ref handed to the
     * caller (refs += claims + 1): clear (-1) + caller ref (+2) + claims
     * (+2*claims).  Never frees (the caller's ref survives). */
    atomic_fetch_add_explicit(&old.cb->count,
                              2 * (int64_t)shared_claims(old.ext) + 1,
                              memory_order_relaxed);
  }
  return old.cb;
}

bool arts_atomic_shared_compare_exchange(arts_atomic_shared_ptr_t *slot,
                                         arts_shared_ptr_t expected,
                                         arts_shared_ptr_t new_val) {
  arts_shared_slot_t cur = atomic_load_explicit(slot, memory_order_acquire);
  bool hold_set = false;
  for (;;) {
    if (cur.cb != expected) {
      if (hold_set) {
        /* We tentatively set new_val's hold but the slot drifted off expected;
         * undo so new_val stays a plain caller handle. */
        atomic_fetch_add_explicit(&new_val->count, 1, memory_order_relaxed);
      }
      return false; /* mismatch — leave slot + both refs untouched. */
    }
    if (new_val != NULL && !hold_set) {
      atomic_fetch_add_explicit(&new_val->count, -1, memory_order_relaxed);
      hold_set = true;
    }
    arts_shared_slot_t desired = {new_val, shared_next_gen_ext(cur.ext)};
    if (atomic_compare_exchange_weak_explicit(
            slot, &cur, desired, memory_order_acq_rel, memory_order_acquire)) {
      /* expected left the slot: clear in_slot (-1) + fold claims.  The caller
       * keeps its OWN ref on expected (pinned it against ABA), released
       * separately, so this does not free expected from under the caller.
       * expected == NULL is install-into-empty (CAS NULL→new_val): nothing left
       * the slot, so there is no hold to drop. */
      if (expected != NULL) {
        shared_count_add(expected, 2 * (int64_t)shared_claims(cur.ext) - 1);
      }
      return true;
    }
    /* CAS failed; cur reloaded.  If only claims/generation changed (cb still
     * expected) the loop retries the install; if cb changed it bails above. */
  }
}
