/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_shared_ptr_t implementation — see arts/utils/shared.h.
 *
 * The control block embeds an arts_lf_link_t as its first member so a
 * recycled cb can ride the global DWCAS pool.  The pool is never drained
 * back to the allocator during the run, which is what makes the
 * acquire-and-validate load immune to use-after-free: a stale cb pointer
 * always addresses valid (possibly reused) memory, and the slot
 * revalidation step rejects a cb that has since been reinstalled.
 */

#include "arts/utils/shared.h"

#include "arts/utils/lockfree_lifo.h" /* arts_lf_link_t */
#include "arts/utils/lockfree_pool.h" /* arts_lockfree_pool_t */

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct arts_shared_s {
  arts_lf_link_t link;      /* first member — rides the cb pool */
  _Atomic(uint64_t) strong; /* strong refcount; 0 ⇒ dying/recycled */
  void *object;             /* managed object pointer */
  void (*deleter)(void *);  /* run once on the last strong drop */
};

/* Global cb pool — never freed for the lifetime of the process.  Initialized
 * before main() via the constructor below so the very first arts_shared_make
 * (which may run during static init in unit tests) sees a valid node_size. */
static arts_lockfree_pool_t g_shared_cb_pool;

__attribute__((constructor)) static void arts_shared_pool_ctor(void) {
  arts_lf_pool_init(&g_shared_cb_pool, sizeof(struct arts_shared_s));
}

arts_shared_ptr_t arts_shared_make(void *object, void (*deleter)(void *)) {
  struct arts_shared_s *cb = arts_lf_pool_alloc(&g_shared_cb_pool);
  atomic_store_explicit(&cb->strong, 1u, memory_order_relaxed);
  cb->object = object;
  cb->deleter = deleter;
  return cb;
}

arts_shared_ptr_t arts_shared_copy(arts_shared_ptr_t b) {
  if (!b)
    return NULL;
  /* Caller already holds b ⇒ strong ≥ 1 ⇒ cb alive; a plain add is safe. */
  atomic_fetch_add_explicit(&b->strong, 1u, memory_order_relaxed);
  return b;
}

void arts_shared_release(arts_shared_ptr_t *p) {
  arts_shared_ptr_t cb = *p;
  if (!cb)
    return;
  *p = NULL;
  uint64_t prev =
      atomic_fetch_sub_explicit(&cb->strong, 1u, memory_order_acq_rel);
  if (prev == 1u) {
    /* Last drop: run the deleter, then recycle the cb (never freed). */
    if (cb->deleter)
      cb->deleter(cb->object);
    arts_lf_pool_release(&g_shared_cb_pool, cb);
  }
}

void arts_shared_abandon(arts_shared_ptr_t *p) {
  arts_shared_ptr_t cb = *p;
  if (!cb)
    return;
  *p = NULL;
  /* Unpublished cb (strong == 1, never shared): recycle the control block
   * without running the deleter — the wrapped object stays the caller's. */
  atomic_store_explicit(&cb->strong, 0u, memory_order_relaxed);
  arts_lf_pool_release(&g_shared_cb_pool, cb);
}

void *arts_shared_get(arts_shared_ptr_t p) { return p ? p->object : NULL; }

arts_shared_ptr_t arts_atomic_shared_load(arts_atomic_shared_ptr_t *slot) {
  for (;;) {
    arts_shared_ptr_t cb = atomic_load_explicit(slot, memory_order_acquire);
    if (!cb)
      return NULL;
    /* CAS strong-inc, but only while strong > 0 (cb not yet dying). */
    uint64_t s = atomic_load_explicit(&cb->strong, memory_order_relaxed);
    bool got = false;
    while (s != 0u) {
      if (atomic_compare_exchange_weak_explicit(&cb->strong, &s, s + 1u,
                                                memory_order_acq_rel,
                                                memory_order_relaxed)) {
        got = true;
        break;
      }
    }
    if (!got)
      continue; /* cb was dying — reload the slot (will see NULL/new cb). */
    /* Revalidate: if the slot still points at cb, our ref is good.  ABA on
     * a recycled cb pointer is caught here — a different install swaps the
     * slot to a different cb pointer, so we release and retry. */
    if (atomic_load_explicit(slot, memory_order_acquire) == cb)
      return cb;
    arts_shared_release(&cb);
  }
}

void arts_atomic_shared_store(arts_atomic_shared_ptr_t *slot,
                              arts_shared_ptr_t new_val) {
  arts_shared_ptr_t old =
      atomic_exchange_explicit(slot, new_val, memory_order_acq_rel);
  if (old)
    arts_shared_release(&old);
}

arts_shared_ptr_t arts_atomic_shared_exchange(arts_atomic_shared_ptr_t *slot,
                                              arts_shared_ptr_t new_val) {
  return atomic_exchange_explicit(slot, new_val, memory_order_acq_rel);
}

bool arts_atomic_shared_compare_exchange(arts_atomic_shared_ptr_t *slot,
                                         arts_shared_ptr_t expected,
                                         arts_shared_ptr_t new_val) {
  arts_shared_ptr_t e = expected;
  if (atomic_compare_exchange_strong_explicit(
          slot, &e, new_val, memory_order_acq_rel, memory_order_acquire)) {
    /* Slot ownership moved expected → new_val.  Drop the ref the slot held on
     * the old value (the caller keeps its own ref on `expected`, which pinned
     * it against ABA across this call and which the caller releases itself). */
    if (expected) {
      arts_shared_ptr_t old = expected;
      arts_shared_release(&old);
    }
    return true;
  }
  return false;
}
