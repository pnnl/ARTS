/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_shared_ptr_t — atomic_shared_ptr pattern (ad-hoc lock-free).
 *
 * Semantic equivalent of std::atomic<std::shared_ptr<T>>: a control block
 * (cb) carries a reference count, the managed object pointer, and a per-object
 * deleter.  Slots that publish a cb are declared by the caller as
 * arts_atomic_shared_ptr_t (a 16-byte DWCAS slot).  Concurrent readers use the
 * split-reference-counting load below (Williams "C++ Concurrency in Action"
 * §7.2.4): it DWCAS-bumps a per-slot in-flight counter WITHOUT dereferencing
 * the cb (so it never touches a cb a concurrent store may be freeing), which
 * pins the cb, then folds the claim into the cb's count and returns a plain
 * owned cb pointer.  Control blocks are allocated/freed per-op by the allocator
 * (no type-stable pool); a per-slot generation tag defeats ABA on a recycled cb
 * address, and a two-field count (refs + in_slot presence bit) makes the free
 * decision immune to the reconcile/load transient-zero race.  Full protocol and
 * the invariants it rests on are documented in shared.c.
 *
 * Local API (single-owner):
 *   make    — allocate a cb (ref = 1) wrapping object + deleter.
 *   copy    — ref++ on a cb the caller already holds (never fails).
 *   release — ref--; last drop runs deleter(object) + frees the cb.
 *   get     — raw object pointer (valid while the caller holds a ref).
 *
 * Atomic slot API (multi-thread shared):
 *   load     — split-count acquire (claim + fold); returns a caller-owned ref.
 *   store    — slot takes ownership of new_val; old slot value is released.
 *   exchange — atomic swap; returns the old value (caller releases it).
 */

#ifndef ARTS_UTILS_SHARED_H
#define ARTS_UTILS_SHARED_H
#ifdef __cplusplus
extern "C" {
#endif

/* Opaque control-block pointer.  The struct definition lives in shared.c;
 * callers only ever hold the pointer. */
typedef struct arts_shared_s *arts_shared_ptr_t;

/* Atomic slot type — SPLIT REFERENCE COUNTING.  The slot packs the cb pointer
 * with a 64-bit external counter `ext` into a 16-byte DWCAS word: a load
 * DWCAS-increments `ext` WITHOUT dereferencing the cb (so it never touches a cb
 * a concurrent store may be freeing), which pins the cb; only then does it
 * dereference to fold the claim into the cb's internal count.  16-byte aligned
 * for cmpxchg16b (x86-64) / casp (ARM64).  The C++/nvcc translation units
 * (which pull this header transitively via runtime_types.h) see the same
 * 16-byte struct layout (no _Atomic) so structs embedding a slot match
 * byte-for-byte; the slot API below is C-only. */
#include <stdint.h>
typedef struct {
  arts_shared_ptr_t cb;
  uint64_t ext;
} __attribute__((aligned(16))) arts_shared_slot_t;
#ifdef __cplusplus
typedef arts_shared_slot_t arts_atomic_shared_ptr_t;
#else
#include <stdatomic.h>
#include <stdbool.h>
typedef _Atomic(arts_shared_slot_t) arts_atomic_shared_ptr_t;
#endif

/* ── Local API (single-owner) ──────────────────────────────────────────── */

/* Allocate a cb (from the allocator) with ref = 1, wrapping `object` and
 * `deleter`.  `deleter` may be NULL (object is unmanaged). */
arts_shared_ptr_t arts_shared_make(void *object, void (*deleter)(void *));

/* Ref++ on a cb the caller already holds.  Returns b (or NULL if b is NULL).
 * Always succeeds: the caller's ref keeps the cb alive. */
arts_shared_ptr_t arts_shared_copy(arts_shared_ptr_t b);

/* Ref--; on the last drop runs deleter(object) and frees the cb.  Sets
 * *p = NULL.  No-op when *p is NULL. */
void arts_shared_release(arts_shared_ptr_t *p);

/* Cancel an UNPUBLISHED cb (just returned by arts_shared_make, never stored
 * into a slot and never copied): free the control block WITHOUT running the
 * deleter, so the wrapped object stays owned by the caller.  Use on the losing
 * side of an install race ("insert-or-fail": the object is still mine").
 * Precondition: ref == 1 (no other holder).  Sets *p = NULL; no-op when *p is
 * NULL. */
void arts_shared_abandon(arts_shared_ptr_t *p);

/* Raw managed-object pointer; valid while the caller holds a ref. */
void *arts_shared_get(arts_shared_ptr_t p);

/* Identity tag: the key a cb is published under, stamped by the publisher
 * BEFORE the cb first enters any slot (re-stamped only while the cb is
 * detached from every slot).  A reader that pinned a cb out of a keyed slot
 * can verify the value it HOLDS against the key it ASKED for by comparing
 * this field: the pinned cb cannot be recycled beneath the reader, so the
 * comparison cannot suffer the ABA that re-reading the table's own key word
 * can — a freed and re-claimed slot may return to a previously observed key
 * while holding a different object.  Atomic because stale handles may still
 * be read across a re-stamp; a racing reader observes one of the two
 * identities, each of which it must treat correctly.  0 = untagged. */
void arts_shared_set_tag(arts_shared_ptr_t p, uint64_t tag);
uint64_t arts_shared_tag(arts_shared_ptr_t p);

/* ── Atomic slot API (multi-thread shared) ─────────────────────────────── */
#ifndef __cplusplus

/* Acquire-and-validate load: returns a caller-owned ref (strong already
 * incremented) or NULL if the slot is empty / the cb is dying.  The caller
 * must arts_shared_release the result. */
arts_shared_ptr_t arts_atomic_shared_load(arts_atomic_shared_ptr_t *slot);

/* Publish new_val into the slot (slot takes ownership of new_val's ref);
 * the previous slot value, if any, is released. */
void arts_atomic_shared_store(arts_atomic_shared_ptr_t *slot,
                              arts_shared_ptr_t new_val);

/* Atomically swap new_val into the slot and return the previous value.  The
 * caller owns the returned ref and must release it. */
arts_shared_ptr_t arts_atomic_shared_exchange(arts_atomic_shared_ptr_t *slot,
                                              arts_shared_ptr_t new_val);

/* Conditional publish (std::atomic<shared_ptr>::compare_exchange_strong).  If
 * the slot still holds `expected`, replace it with `new_val` (slot takes
 * new_val's ref, drops the ref it held on `expected`) and return true.  On
 * mismatch return false and leave the slot + both refs untouched.  Caller must
 * keep its own ref on `expected` alive across the call (pinning it against cb
 * recycle, so the raw-pointer compare cannot ABA). */
bool arts_atomic_shared_compare_exchange(arts_atomic_shared_ptr_t *slot,
                                         arts_shared_ptr_t expected,
                                         arts_shared_ptr_t new_val);

#endif /* !__cplusplus */

#ifdef __cplusplus
}
#endif
#endif /* ARTS_UTILS_SHARED_H */
