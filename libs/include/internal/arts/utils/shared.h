/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_shared_ptr_t — atomic_shared_ptr pattern (ad-hoc lock-free).
 *
 * Semantic equivalent of std::atomic<std::shared_ptr<T>>: a control block
 * (cb) carries a strong refcount, the managed object pointer, and a per-
 * object deleter.  Slots that publish a cb are declared by the caller as
 * arts_atomic_shared_ptr_t (an _Atomic cb pointer); concurrent readers use
 * the acquire-and-validate load below, which is immune to use-after-free
 * because cb memory is drawn from a global pool that is never returned to
 * the allocator (last strong drop recycles the cb into the pool, so the
 * pointer always addresses valid memory — possibly a different live object,
 * which the slot revalidation step rejects).
 *
 * Local API (single-owner):
 *   make    — allocate a cb (strong = 1) wrapping object + deleter.
 *   copy    — strong++ on a cb the caller already holds (never fails).
 *   release — strong--; last drop runs deleter(object) + recycles the cb.
 *   get     — raw object pointer (valid while the caller holds a ref).
 *
 * Atomic slot API (multi-thread shared):
 *   load     — acquire-and-validate: atomic load + strong-inc, retry on ABA.
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

/* Portable atomic slot type.  C uses C11 _Atomic; the C++/nvcc translation
 * units (which pull this header transitively via runtime_types.h) drop the
 * _Atomic qualifier for layout-only visibility — the slot API itself is
 * C-only.  Both see a single pointer-width slot, so the layout matches. */
#ifdef __cplusplus
typedef arts_shared_ptr_t arts_atomic_shared_ptr_t;
#else
#include <stdatomic.h>
#include <stdbool.h>
typedef _Atomic(arts_shared_ptr_t) arts_atomic_shared_ptr_t;
#endif

/* ── Local API (single-owner) ──────────────────────────────────────────── */

/* Allocate a cb (from the global pool, or fresh) with strong = 1, wrapping
 * `object` and `deleter`.  `deleter` may be NULL (object is unmanaged). */
arts_shared_ptr_t arts_shared_make(void *object, void (*deleter)(void *));

/* Strong++ on a cb the caller already holds.  Returns b (or NULL if b is
 * NULL).  Always succeeds: the caller's ref keeps the cb alive. */
arts_shared_ptr_t arts_shared_copy(arts_shared_ptr_t b);

/* Strong--; on the last drop runs deleter(object) and recycles the cb into
 * the global pool.  Sets *p = NULL.  No-op when *p is NULL. */
void arts_shared_release(arts_shared_ptr_t *p);

/* Cancel an UNPUBLISHED cb (just returned by arts_shared_make, never stored
 * into a slot and never copied): recycle the control block into the pool
 * WITHOUT running the deleter, so the wrapped object stays owned by the
 * caller.  Use on the losing side of an install race ("insert-or-fail":
 * the object is still mine").  Precondition: strong == 1 (no other holder).
 * Sets *p = NULL; no-op when *p is NULL. */
void arts_shared_abandon(arts_shared_ptr_t *p);

/* Raw managed-object pointer; valid while the caller holds a ref. */
void *arts_shared_get(arts_shared_ptr_t p);

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
