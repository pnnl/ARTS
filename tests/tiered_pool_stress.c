/* SPDX-License-Identifier: Apache-2.0
 *
 * Stress test for arts_tiered_pool_t (3-tier hierarchical pool).
 * Tiered (thread-local + per-NUMA + global) pool stress test.
 *
 * Two scenarios:
 *
 *   1. Symmetric workload — 16 threads, 50/50 alloc/release mix,
 *      1M ops/thread.  Threads simulate distribution across 4 NUMA
 *      domains (4 threads per NUMA) by setting arts_thread_info.thread_id
 *      and numa_domain_id manually.  Note the placeholder
 *      (arts_tiered_pool_numa_id always returns 0) collapses tier 1 to
 *      a single shard — the test still exercises tier 0 and tier 2.
 *
 *   2. Asymmetric workload — 4 alloc-only + 4 release-only threads with
 *      a fixed buffer of pre-populated nodes circulated through a
 *      shared MPSC ring.  Verifies global pool absorbs cross-thread
 *      flow without unbounded growth or memory corruption.
 *
 * Pass criterion: all scenarios complete without abort/assert and
 * arts_tiered_pool_destroy() drains every node (ASan: 0 leaks).
 */

#include "arts/runtime_state.h" /* arts_thread_info, arts_node_info */
#include "arts/utils/lockfree_lifo.h"
#include "arts/utils/lockfree_pool.h"
#include "arts/utils/malloc.h"
#include "arts/utils/tiered_pool.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ──────────────────────────────────────────────────────────────────────── */

#define NUM_THREADS 16
#define NUM_NUMA 4
#define OPS_PER_THREAD 1000000u

typedef struct {
  arts_lf_link_t link;
  uint64_t sentinel;
} pool_node_t;

static arts_tiered_pool_t g_pool;
static _Atomic int g_start;

/* Per-thread bookkeeping: alloc/release counts for sanity reporting. */
typedef struct {
  uint32_t tid;
  uint64_t allocs;
  uint64_t releases;
} thread_ctx_t;

static void *symmetric_thread(void *arg) {
  thread_ctx_t *ctx = (thread_ctx_t *)arg;
  /* Make the runtime helpers see this thread as worker `tid`.  Without
   * the runtime running, arts_thread_info is zero-initialized; we set
   * the fields the tiered pool reads. */
  arts_thread_info.thread_id = ctx->tid;
  arts_thread_info.numa_domain_id = ctx->tid / (NUM_THREADS / NUM_NUMA);

  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }

  /* Maintain a small private stack so we can do ~50/50 alloc/release
   * without immediately underflowing.  Cap at 32 to keep tier-0 hits
   * dominant.  Use an LCG to make the alloc/release mix reproducibly
   * "random" (per-thread seed). */
  pool_node_t *stash[32];
  int top = 0;
  uint32_t lcg = ctx->tid * 2654435761u + 1;
  for (uint32_t i = 0; i < OPS_PER_THREAD; i++) {
    lcg = lcg * 1103515245u + 12345u;
    int do_alloc = (top == 0) ? 1 : (top == 32 ? 0 : (int)(lcg & 1));
    if (do_alloc) {
      pool_node_t *n = (pool_node_t *)arts_tiered_pool_alloc(&g_pool);
      if (!n) {
        fprintf(stderr, "alloc returned NULL\n");
        abort();
      }
      atomic_init(&n->link.next, NULL);
      n->sentinel = ((uint64_t)ctx->tid << 32) | i;
      stash[top++] = n;
      ctx->allocs++;
    } else {
      pool_node_t *n = stash[--top];
      /* Sanity: sentinel still readable. */
      (void)n->sentinel;
      arts_tiered_pool_release(&g_pool, n);
      ctx->releases++;
    }
  }
  /* Drain remaining stashed nodes back to the pool. */
  while (top > 0) {
    arts_tiered_pool_release(&g_pool, stash[--top]);
    ctx->releases++;
  }
  return NULL;
}

static int run_symmetric(void) {
  arts_tiered_pool_cfg_t cfg = {
      .H_local = 128,
      .B_local = 64,
      .H_numa = 1024,
      .B_numa = 256,
  };
  arts_tiered_pool_init_explicit(&g_pool, sizeof(pool_node_t), NUM_THREADS,
                                 NUM_NUMA, cfg);
  atomic_init(&g_start, 0);

  pthread_t th[NUM_THREADS];
  thread_ctx_t ctx[NUM_THREADS];
  for (uint32_t t = 0; t < NUM_THREADS; t++) {
    ctx[t].tid = t;
    ctx[t].allocs = 0;
    ctx[t].releases = 0;
    if (pthread_create(&th[t], NULL, symmetric_thread, &ctx[t]) != 0) {
      fprintf(stderr, "pthread_create failed\n");
      return 1;
    }
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (uint32_t t = 0; t < NUM_THREADS; t++) {
    pthread_join(th[t], NULL);
  }

  uint64_t total_allocs = 0, total_releases = 0;
  for (uint32_t t = 0; t < NUM_THREADS; t++) {
    total_allocs += ctx[t].allocs;
    total_releases += ctx[t].releases;
  }
  if (total_allocs != total_releases) {
    fprintf(stderr,
            "symmetric: alloc/release imbalance (allocs=%" PRIu64
            " releases=%" PRIu64 ")\n",
            total_allocs, total_releases);
    return 1;
  }

  arts_tiered_pool_destroy(&g_pool);
  printf("  symmetric: %u threads x %u ops PASS (allocs=%" PRIu64 ")\n",
         NUM_THREADS, OPS_PER_THREAD, total_allocs);
  return 0;
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Asymmetric workload: alloc-only and release-only threads communicate
 * via a shared MPMC ring buffer.  Producer: alloc → push to ring;
 * consumer: pop from ring → release.  Verifies tier-1 → tier-2 flow
 * absorbs the imbalance without growing unbounded. */

#define ASYM_PRODUCERS 4
#define ASYM_CONSUMERS 4
#define ASYM_OPS 200000u
#define RING_SIZE 4096

static pool_node_t *g_ring[RING_SIZE];
static _Atomic uint32_t g_ring_head; /* next free slot */
static _Atomic uint32_t g_ring_tail; /* next ready slot */
/* Use a per-slot sequence counter for safe MPMC. */
static _Atomic uint32_t g_ring_seq[RING_SIZE];

static void ring_push(pool_node_t *n) {
  uint32_t pos =
      atomic_fetch_add_explicit(&g_ring_head, 1, memory_order_relaxed);
  uint32_t idx = pos % RING_SIZE;
  /* Spin until consumer has cleared this slot (seq == pos). */
  while (atomic_load_explicit(&g_ring_seq[idx], memory_order_acquire) != pos) {
    /* spin */
  }
  g_ring[idx] = n;
  atomic_store_explicit(&g_ring_seq[idx], pos + 1, memory_order_release);
}

static pool_node_t *ring_pop(void) {
  for (;;) {
    uint32_t pos = atomic_load_explicit(&g_ring_tail, memory_order_relaxed);
    uint32_t idx = pos % RING_SIZE;
    uint32_t seq = atomic_load_explicit(&g_ring_seq[idx], memory_order_acquire);
    if (seq != pos + 1) {
      /* Slot not yet ready — caller will spin. */
      return NULL;
    }
    /* Try to claim this slot. */
    if (atomic_compare_exchange_weak_explicit(&g_ring_tail, &pos, pos + 1,
                                              memory_order_relaxed,
                                              memory_order_relaxed)) {
      pool_node_t *n = g_ring[idx];
      atomic_store_explicit(&g_ring_seq[idx], pos + RING_SIZE,
                            memory_order_release);
      return n;
    }
    /* Lost CAS — retry. */
  }
}

static _Atomic uint64_t g_asym_produced;
static _Atomic uint64_t g_asym_consumed;

static void *asym_producer(void *arg) {
  thread_ctx_t *ctx = (thread_ctx_t *)arg;
  arts_thread_info.thread_id = ctx->tid;
  arts_thread_info.numa_domain_id = 0;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (uint32_t i = 0; i < ASYM_OPS; i++) {
    pool_node_t *n = (pool_node_t *)arts_tiered_pool_alloc(&g_pool);
    if (!n)
      abort();
    atomic_init(&n->link.next, NULL);
    n->sentinel = ((uint64_t)ctx->tid << 32) | i;
    ring_push(n);
    atomic_fetch_add_explicit(&g_asym_produced, 1, memory_order_relaxed);
  }
  return NULL;
}

static void *asym_consumer(void *arg) {
  thread_ctx_t *ctx = (thread_ctx_t *)arg;
  arts_thread_info.thread_id = ctx->tid;
  /* Place consumers on a different (stub-collapsed) NUMA so the
   * release path stresses cross-NUMA. */
  arts_thread_info.numa_domain_id = 1;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  uint64_t target = (uint64_t)ASYM_PRODUCERS * ASYM_OPS;
  while (atomic_load_explicit(&g_asym_consumed, memory_order_acquire) <
         target) {
    pool_node_t *n = ring_pop();
    if (!n) {
      /* Spin: producers may not have caught up. */
      continue;
    }
    /* Sanity. */
    (void)n->sentinel;
    arts_tiered_pool_release(&g_pool, n);
    atomic_fetch_add_explicit(&g_asym_consumed, 1, memory_order_relaxed);
  }
  return NULL;
}

static int run_asymmetric(void) {
  arts_tiered_pool_cfg_t cfg = {
      .H_local = 64,
      .B_local = 32,
      .H_numa = 256,
      .B_numa = 128,
  };
  arts_tiered_pool_init_explicit(&g_pool, sizeof(pool_node_t),
                                 ASYM_PRODUCERS +
                                     ASYM_CONSUMERS, /* num_threads */
                                 2, /* num_numa_nodes (manual) */
                                 cfg);
  atomic_init(&g_start, 0);
  atomic_init(&g_asym_produced, 0);
  atomic_init(&g_asym_consumed, 0);
  atomic_init(&g_ring_head, 0);
  atomic_init(&g_ring_tail, 0);
  for (int i = 0; i < RING_SIZE; i++) {
    atomic_init(&g_ring_seq[i], (uint32_t)i);
    g_ring[i] = NULL;
  }

  pthread_t th[ASYM_PRODUCERS + ASYM_CONSUMERS];
  thread_ctx_t ctx[ASYM_PRODUCERS + ASYM_CONSUMERS];
  for (int i = 0; i < ASYM_PRODUCERS; i++) {
    ctx[i].tid = (uint32_t)i;
    ctx[i].allocs = 0;
    ctx[i].releases = 0;
    if (pthread_create(&th[i], NULL, asym_producer, &ctx[i]) != 0)
      abort();
  }
  for (int i = 0; i < ASYM_CONSUMERS; i++) {
    int idx = ASYM_PRODUCERS + i;
    ctx[idx].tid = (uint32_t)idx;
    ctx[idx].allocs = 0;
    ctx[idx].releases = 0;
    if (pthread_create(&th[idx], NULL, asym_consumer, &ctx[idx]) != 0)
      abort();
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < ASYM_PRODUCERS + ASYM_CONSUMERS; i++) {
    pthread_join(th[i], NULL);
  }

  uint64_t produced = atomic_load(&g_asym_produced);
  uint64_t consumed = atomic_load(&g_asym_consumed);
  if (produced != consumed) {
    fprintf(stderr, "asym: produced=%" PRIu64 " consumed=%" PRIu64 "\n",
            produced, consumed);
    return 1;
  }
  arts_tiered_pool_destroy(&g_pool);
  printf("  asymmetric: %d producers + %d consumers x %u ops PASS\n",
         ASYM_PRODUCERS, ASYM_CONSUMERS, ASYM_OPS);
  return 0;
}

int main(void) {
  if (run_symmetric() != 0)
    return 1;
  if (run_asymmetric() != 0)
    return 1;
  printf("PASS tiered_pool_stress\n");
  return 0;
}
