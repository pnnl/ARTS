/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_lockfree_pool_t — DWCAS tagged-head Treiber-stack object pool.
 *
 * Layout: 16-byte head = (arts_lf_link_t *ptr, uintptr_t tag).  Each push
 * increments the tag, so even if a popped node is re-pushed (ABA), the
 * DWCAS observes the tag mismatch and retries.  Lifetime invariant: nodes
 * live until pool destroy — alloc/release transfer ownership only.  This
 * makes the batch_fetch chain-walk safe (no use-after-free on `next`).
 *
 * `count` is a best-effort approximate value used by the tiered-pool
 * watermark heuristics.  Updates are relaxed fetch_add /
 * fetch_sub — never used for correctness.
 *
 * Header-only inline — no separate .c file.  All callers compile with
 * -mcx16 on x86_64 (set globally in the top-level CMakeLists.txt).
 */

#ifndef ARTS_UTILS_LOCKFREE_POOL_H
#define ARTS_UTILS_LOCKFREE_POOL_H

#include "arts/utils/lockfree_lifo.h" /* arts_lf_link_t */
#include "arts/utils/malloc.h"        /* arts_calloc / arts_free */

#include <stdalign.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** 16-byte DWCAS head (ptr + tag).  Must be 16-byte aligned for cmpxchg16b
 *  / ldaxp+stxp.  GCC and Clang issue lock-free 128-bit atomics for this
 *  layout when -mcx16 is set.
 *
 *  Note: GCC C rejects `typedef struct alignas(16) {...}` (placement of
 *  the alignment specifier between the `struct` keyword and `{` is not
 *  accepted).  We use the GNU/C11 attribute on the typedef name instead. */
typedef struct {
  arts_lf_link_t *ptr;
  uintptr_t tag;
} __attribute__((aligned(16))) arts_lf_pool_head_t;
_Static_assert(sizeof(arts_lf_pool_head_t) == 16, "DWCAS layout");
_Static_assert(alignof(arts_lf_pool_head_t) == 16, "DWCAS alignment");

typedef struct {
  _Atomic(arts_lf_pool_head_t) head;
  _Atomic(uint32_t) count; /* approx — best-effort */
  size_t node_size;        /* fallback alloc size */
} arts_lockfree_pool_t;

/* ── Initialization / teardown ─────────────────────────────────────────── */

static inline void arts_lf_pool_init(arts_lockfree_pool_t *p,
                                     size_t node_size) {
  arts_lf_pool_head_t empty = {.ptr = NULL, .tag = 0};
  atomic_store_explicit(&p->head, empty, memory_order_relaxed);
  atomic_store_explicit(&p->count, 0u, memory_order_relaxed);
  p->node_size = node_size;
}

/** Drain everything and free each node via `free_fn`.  Caller is
 *  responsible for ensuring no other thread touches the pool during destroy,
 *  and that `free_fn` is the matching free for whatever allocator produced
 *  the pool's nodes — a pool seeded from a non-default allocator (e.g. a
 *  registered/pinned pool) must be destroyed with that allocator's free, not
 *  the default arts_free, or the mismatched free is a correctness bug (double
 *  bookkeeping, or freeing memory the wrong allocator does not own). */
static inline void arts_lf_pool_destroy_with(arts_lockfree_pool_t *p,
                                             void (*free_fn)(void *)) {
  arts_lf_pool_head_t cur =
      atomic_load_explicit(&p->head, memory_order_acquire);
  arts_lf_link_t *node = cur.ptr;
  while (node) {
    arts_lf_link_t *next =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    free_fn(node);
    node = next;
  }
  arts_lf_pool_head_t empty = {.ptr = NULL, .tag = cur.tag};
  atomic_store_explicit(&p->head, empty, memory_order_relaxed);
  atomic_store_explicit(&p->count, 0u, memory_order_relaxed);
}

/** Drain everything and free each node via arts_free.  Caller is
 *  responsible for ensuring no other thread touches the pool during
 *  destroy. */
static inline void arts_lf_pool_destroy(arts_lockfree_pool_t *p) {
  arts_lf_pool_destroy_with(p, arts_free);
}

/* ── Single-node alloc / release ──────────────────────────────────────── */

/** DWCAS pop.  On empty pool, falls back to arts_calloc(1, node_size).
 *  Returned object's first sizeof(arts_lf_link_t) bytes are uninitialized
 *  after pop — caller is responsible for re-init. */
static inline void *arts_lf_pool_alloc(arts_lockfree_pool_t *p) {
  arts_lf_pool_head_t cur =
      atomic_load_explicit(&p->head, memory_order_acquire);
  for (;;) {
    if (!cur.ptr) {
      /* Pool empty — heap fallback.  arts_calloc zero-fills, so
       * subsequent atomic_init on link.next is well-defined. */
      return arts_calloc(1, p->node_size);
    }
    arts_lf_link_t *next =
        atomic_load_explicit(&cur.ptr->next, memory_order_relaxed);
    arts_lf_pool_head_t newh = {.ptr = next, .tag = cur.tag + 1};
    if (atomic_compare_exchange_weak_explicit(
            &p->head, &cur, newh, memory_order_acquire, memory_order_acquire)) {
      atomic_fetch_sub_explicit(&p->count, 1u, memory_order_relaxed);
      return cur.ptr;
    }
    /* cur reloaded by CAS — retry */
  }
}

/** DWCAS push.  Tag is incremented by the writer (CAS install); reader
 *  side increments tag on pop too — both contribute to ABA defense. */
static inline void arts_lf_pool_release(arts_lockfree_pool_t *p, void *node) {
  arts_lf_link_t *n = (arts_lf_link_t *)node;
  arts_lf_pool_head_t cur =
      atomic_load_explicit(&p->head, memory_order_relaxed);
  for (;;) {
    atomic_store_explicit(&n->next, cur.ptr, memory_order_relaxed);
    arts_lf_pool_head_t newh = {.ptr = n, .tag = cur.tag + 1};
    if (atomic_compare_exchange_weak_explicit(
            &p->head, &cur, newh, memory_order_release, memory_order_relaxed)) {
      atomic_fetch_add_explicit(&p->count, 1u, memory_order_relaxed);
      return;
    }
    /* cur reloaded by CAS — retry */
  }
}

/* ── Batch primitives ───────────────────────────────────────────────────── */

/** Try to detach up to `want` nodes from the top of the pool.  Walks the
 *  chain to determine the tail (or end), then DWCAS the new head.  On
 *  success, returns chain head and writes the actual count to *got.
 *  Returns NULL when the pool is empty (sets *got = 0).
 *
 *  Caller becomes the owner of the returned chain — must either re-push
 *  it, install it elsewhere, or free each node before pool destroy. */
static inline arts_lf_link_t *arts_lf_pool_batch_fetch(arts_lockfree_pool_t *p,
                                                       uint32_t want,
                                                       uint32_t *got) {
  if (got) {
    *got = 0;
  }
  if (want == 0) {
    return NULL;
  }
  arts_lf_pool_head_t cur =
      atomic_load_explicit(&p->head, memory_order_acquire);
  for (;;) {
    if (!cur.ptr) {
      if (got) {
        *got = 0;
      }
      return NULL;
    }
    /* Walk up to `want` nodes; safe due to lifetime invariant (pool
     * nodes live until pool destroy). */
    arts_lf_link_t *tail = cur.ptr;
    uint32_t taken = 1;
    arts_lf_link_t *next_after =
        atomic_load_explicit(&tail->next, memory_order_relaxed);
    while (taken < want && next_after) {
      tail = next_after;
      next_after = atomic_load_explicit(&tail->next, memory_order_relaxed);
      taken++;
    }
    arts_lf_pool_head_t newh = {.ptr = next_after, .tag = cur.tag + 1};
    if (atomic_compare_exchange_weak_explicit(
            &p->head, &cur, newh, memory_order_acquire, memory_order_acquire)) {
      /* Detach the chain (so caller doesn't accidentally walk into the
       * remaining pool nodes). */
      atomic_store_explicit(&tail->next, NULL, memory_order_relaxed);
      atomic_fetch_sub_explicit(&p->count, taken, memory_order_relaxed);
      if (got) {
        *got = taken;
      }
      return cur.ptr;
    }
    /* cur reloaded — retry walk from new top. */
  }
}

/** Pop exactly one recycled node, or NULL if the pool is empty.  Unlike
 *  arts_lf_pool_alloc this does NOT fall back to the allocator on an empty
 *  pool — the caller (e.g. the per-DB buffer pool, which needs a 64-byte
 *  aligned allocation) does its own allocation on a miss.  Built on
 *  batch_fetch(1) so it shares the DWCAS ABA-safe pop. */
static inline arts_lf_link_t *
arts_lf_pool_pop_or_null(arts_lockfree_pool_t *p) {
  uint32_t got = 0;
  return arts_lf_pool_batch_fetch(p, 1, &got); /* NULL on empty; no calloc */
}

/** Prepend a caller-owned chain of `batch_n` nodes [head .. tail] onto the
 *  pool.  Caller must have already terminated batch_tail->next = NULL is
 *  NOT required here — we overwrite it with the previous head. */
static inline void arts_lf_pool_batch_release(arts_lockfree_pool_t *p,
                                              arts_lf_link_t *batch_head,
                                              arts_lf_link_t *batch_tail,
                                              uint32_t batch_n) {
  if (!batch_head || !batch_tail || batch_n == 0) {
    return;
  }
  arts_lf_pool_head_t cur =
      atomic_load_explicit(&p->head, memory_order_relaxed);
  for (;;) {
    atomic_store_explicit(&batch_tail->next, cur.ptr, memory_order_relaxed);
    arts_lf_pool_head_t newh = {.ptr = batch_head, .tag = cur.tag + 1};
    if (atomic_compare_exchange_weak_explicit(
            &p->head, &cur, newh, memory_order_release, memory_order_relaxed)) {
      atomic_fetch_add_explicit(&p->count, batch_n, memory_order_relaxed);
      return;
    }
    /* cur reloaded — retry. */
  }
}

/** Detach up to `want` nodes from the top of the pool, returning chain
 *  head/tail via out-params.  Used by the tiered-pool spill path
 *  (NUMA → global).  Sets both *out_head and *out_tail to NULL when the
 *  pool is empty. */
static inline void arts_lf_pool_batch_drain(arts_lockfree_pool_t *p,
                                            uint32_t want,
                                            arts_lf_link_t **out_head,
                                            arts_lf_link_t **out_tail) {
  if (out_head) {
    *out_head = NULL;
  }
  if (out_tail) {
    *out_tail = NULL;
  }
  if (want == 0) {
    return;
  }
  arts_lf_pool_head_t cur =
      atomic_load_explicit(&p->head, memory_order_acquire);
  for (;;) {
    if (!cur.ptr) {
      return;
    }
    arts_lf_link_t *tail = cur.ptr;
    uint32_t taken = 1;
    arts_lf_link_t *next_after =
        atomic_load_explicit(&tail->next, memory_order_relaxed);
    while (taken < want && next_after) {
      tail = next_after;
      next_after = atomic_load_explicit(&tail->next, memory_order_relaxed);
      taken++;
    }
    arts_lf_pool_head_t newh = {.ptr = next_after, .tag = cur.tag + 1};
    if (atomic_compare_exchange_weak_explicit(
            &p->head, &cur, newh, memory_order_acquire, memory_order_acquire)) {
      atomic_store_explicit(&tail->next, NULL, memory_order_relaxed);
      atomic_fetch_sub_explicit(&p->count, taken, memory_order_relaxed);
      if (out_head) {
        *out_head = cur.ptr;
      }
      if (out_tail) {
        *out_tail = tail;
      }
      return;
    }
  }
}

#ifdef __cplusplus
}
#endif

#endif /* ARTS_UTILS_LOCKFREE_POOL_H */
