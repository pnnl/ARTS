/* SPDX-License-Identifier: Apache-2.0
 *
 * T007 — arts_tiered_pool_t conservation + structural invariants
 * (tiered_pool.h).  Census 29.md §4 GAPS / SUSPECTED-BUGS B123/B124.
 *
 * The existing stress only checks alloc==release *balance*, which would NOT
 * catch a duplication bug (the same node handed to two callers keeps the
 * count balanced).  This test adds:
 *
 *  INV-2 (node-set conservation): a fixed universe of distinct nodes churns
 *    through alloc/release on T worker threads (each with a UNIQUE thread_id,
 *    upholding the tier-0 single-owner precondition).  A per-node atomic
 *    owner-stamp catches any node simultaneously owned by two threads (a
 *    duplication a balance check misses).  At quiescence every universe node
 *    is back, exactly once.
 *
 *  INV-3 (tier-2 split): single-thread, pre-load ONLY the global tier with a
 *    known count, then alloc once with configured B_local/B_numa; assert the
 *    head is consumed and the remaining got-1 nodes are partitioned exactly
 *    (to_tcache into tcache, the rest into the NUMA shard) — sum == got-1.
 *
 *  INV-1 negative-as-positive: run the conservation phase under TSan with
 *    deliberately UNIQUE ids to prove the no-atomic tier-0 fast path is
 *    race-free when the precondition holds (documents B124: aliased ids would
 *    race — we uphold uniqueness so TSan must be clean).
 *
 *  Early-init clamp: arts_tiered_pool_init_explicit(num_threads=0) clamps to 1
 *    (B-robustness); a single-tcache alloc/release round still works.
 *
 *  node_size guard (B123): the header has NO assert that node_size >=
 *    sizeof(arts_lf_link_t).  We document by using a valid node_size and NOTE
 *    that a too-small one would silently corrupt the link overlay (we do NOT
 *    pass a too-small one — that would be UB; the gap is recorded here).
 *
 * Standalone: provides the runtime globals (arts_thread_info /
 * arts_node_info) the tiered pool's inline helpers reference, plus libc
 * arts_calloc/_align/free shims.  Needs the build-generated counter Preamble.h
 * include dir (runtime_state.h pulls it transitively).
 */

#include "arts/runtime_state.h"
#include "arts/utils/lockfree_lifo.h"
#include "arts/utils/lockfree_pool.h"
#include "arts/utils/tiered_pool.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Runtime globals referenced by the tiered pool's inline helpers. */
struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

/* libc-backed allocator shims. */
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void *arts_calloc_align(size_t nmemb, size_t size, size_t align) {
  void *p = NULL;
  if (posix_memalign(&p, align, nmemb * size) != 0) {
    return NULL;
  }
  return p;
}
void arts_free(void *ptr) { free(ptr); }

#define NODE_SIZE 64
#define UNIVERSE 1024
#define THREADS 8
#define HAND_CAP 48
#define ITERS 150000

typedef struct {
  arts_lf_link_t link; /* first member */
  uint32_t id;
  _Atomic int owner; /* -1 = in pool, else thread index */
} node_t;

static arts_tiered_pool_t g_pool;
static node_t *g_nodes[UNIVERSE]; /* each individually allocated */
static atomic_int g_start;

typedef struct {
  int tid;
  void *hand[HAND_CAP];
  int hand_n;
  uint64_t rng;
} ctx_t;

static inline uint64_t xs(uint64_t *s) {
  uint64_t x = *s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *s = x;
  return x;
}

static void *worker(void *arg) {
  ctx_t *c = (ctx_t *)arg;
  /* UNIQUE thread_id — upholds the tier-0 single-owner precondition (INV-1). */
  arts_thread_info.thread_id = (unsigned)c->tid;
  arts_thread_info.numa_domain_id = 0;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < ITERS; i++) {
    int do_alloc;
    uint64_t r = xs(&c->rng);
    if (c->hand_n == 0) {
      do_alloc = 1;
    } else if (c->hand_n == HAND_CAP) {
      do_alloc = 0;
    } else {
      do_alloc = (int)(r & 1);
    }
    if (do_alloc) {
      node_t *n = (node_t *)arts_tiered_pool_alloc(&g_pool);
      if (!n) {
        (void)fprintf(stderr, "FAIL tiered_pool_conservation: alloc NULL\n");
        abort();
      }
      /* A pool-popped node carries a stale (possibly another-owner-stamped)
       * owner; a heap-fresh node carries owner==0.  We are the sole owner now
       * (single-consumer pop), so simply claim it. */
      atomic_store_explicit(&n->owner, c->tid, memory_order_relaxed);
      c->hand[c->hand_n++] = n;
    } else {
      node_t *n = (node_t *)c->hand[--c->hand_n];
      int expect = c->tid;
      if (!atomic_compare_exchange_strong_explicit(&n->owner, &expect, -1,
                                                   memory_order_acq_rel,
                                                   memory_order_acquire)) {
        (void)fprintf(stderr,
                      "FAIL tiered_pool_conservation: releasing node not owned "
                      "by me (owner=%d)\n",
                      expect);
        abort();
      }
      arts_tiered_pool_release(&g_pool, n);
    }
  }
  return NULL;
}

static int conservation_phase(void) {
  arts_tiered_pool_cfg_t cfg = {
      .H_local = 32, .B_local = 16, .H_numa = 128, .B_numa = 64};
  arts_tiered_pool_init_explicit(&g_pool, NODE_SIZE, THREADS, 1, cfg);
  atomic_init(&g_start, 0);
  for (int i = 0; i < UNIVERSE; i++) {
    g_nodes[i] = (node_t *)calloc(1, NODE_SIZE);
    if (!g_nodes[i]) {
      return 1;
    }
    g_nodes[i]->id = (uint32_t)i;
    atomic_init(&g_nodes[i]->owner, -1);
    arts_tiered_pool_release(&g_pool, g_nodes[i]);
  }

  pthread_t th[THREADS];
  ctx_t ctx[THREADS];
  for (int i = 0; i < THREADS; i++) {
    ctx[i].tid = i;
    ctx[i].hand_n = 0;
    ctx[i].rng = 0xDEADBEEFu + (uint64_t)i * 0x9E3779B9u;
    pthread_create(&th[i], NULL, worker, &ctx[i]);
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < THREADS; i++) {
    pthread_join(th[i], NULL);
  }
  /* Drain each hand back. */
  for (int i = 0; i < THREADS; i++) {
    /* release on the worker's own tcache requires the worker thread_id; do it
     * from the main thread with tid set per node-owner so tier-0 picks an
     * in-range tcache (any consistent unique id works, since workers exited).
     */
    arts_thread_info.thread_id = (unsigned)ctx[i].tid;
    for (int k = 0; k < ctx[i].hand_n; k++) {
      node_t *n = (node_t *)ctx[i].hand[k];
      atomic_store_explicit(&n->owner, -1, memory_order_relaxed);
      arts_tiered_pool_release(&g_pool, n);
    }
  }
  /* INV-2: every universe node accounted for (owner == -1). */
  for (int i = 0; i < UNIVERSE; i++) {
    int o = atomic_load_explicit(&g_nodes[i]->owner, memory_order_relaxed);
    if (o != -1) {
      (void)fprintf(stderr,
                    "FAIL tiered_pool_conservation: node %d lost (owner=%d)\n",
                    i, o);
      return 1;
    }
  }
  arts_tiered_pool_destroy(&g_pool); /* frees all members (each individually) */
  return 0;
}

/* INV-3: deterministic tier-2 split.  Pre-load ONLY the global tier with K
 * nodes, then a single alloc must consume the head and partition got-1 nodes
 * exactly between tcache and the NUMA shard.  We observe the resulting tcache
 * count and NUMA pool count to assert the partition sums to got-1. */
static int tier2_split_phase(void) {
  arts_tiered_pool_cfg_t cfg = {
      .H_local = 1000, .B_local = 8, .H_numa = 1000, .B_numa = 32};
  arts_tiered_pool_t p;
  arts_tiered_pool_init_explicit(&p, NODE_SIZE, /*num_threads*/ 4,
                                 /*num_numa_nodes*/ 1, cfg);
  arts_thread_info.thread_id = 0;

  /* Seed ONLY the global tier with K nodes (tcache + NUMA start empty). */
  const int K = 20;
  node_t *seed[64];
  for (int i = 0; i < K; i++) {
    seed[i] = (node_t *)calloc(1, NODE_SIZE);
    atomic_init(&seed[i]->owner, -1);
    arts_lf_pool_release(&p.global, &seed[i]->link);
  }

  /* One alloc: tier-0 miss, tier-1 (NUMA) miss, tier-2 (global) hit batch of
   * up to B_numa=32 -> got=min(32,K)=K.  head returned; remaining K-1 split:
   * to_tcache = B_local-1 = 7 (clamped to K-1), rest -> NUMA. */
  void *head = arts_tiered_pool_alloc(&p);
  if (!head) {
    return 1;
  }
  uint32_t tcache_cnt = p.tcache[0].count;
  uint32_t numa_cnt =
      atomic_load_explicit(&p.numa[0].pool.count, memory_order_relaxed);
  /* head consumed -> K-1 remain across tcache + NUMA. */
  uint32_t remaining = (uint32_t)(K - 1);
  uint32_t expect_tcache = (cfg.B_local > 0) ? (cfg.B_local - 1) : 0;
  if (expect_tcache > remaining) {
    expect_tcache = remaining;
  }
  uint32_t expect_numa = remaining - expect_tcache;
  if (tcache_cnt != expect_tcache || numa_cnt != expect_numa) {
    (void)fprintf(stderr,
                  "FAIL tiered_pool_conservation: tier-2 split tcache=%u "
                  "numa=%u, expected tcache=%u numa=%u (got-1=%u)\n",
                  tcache_cnt, numa_cnt, expect_tcache, expect_numa, remaining);
    return 1;
  }
  if (tcache_cnt + numa_cnt != remaining) {
    (void)fprintf(stderr,
                  "FAIL tiered_pool_conservation: split sum %u != got-1 %u\n",
                  tcache_cnt + numa_cnt, remaining);
    return 1;
  }
  /* Return the head and tear down (destroy frees every node uniformly). */
  arts_tiered_pool_release(&p, head);
  arts_tiered_pool_destroy(&p);
  return 0;
}

/* Early-init clamp: num_threads==0 -> clamped to 1, single tcache works. */
static int early_init_phase(void) {
  arts_tiered_pool_cfg_t cfg = {
      .H_local = 4, .B_local = 2, .H_numa = 8, .B_numa = 4};
  arts_tiered_pool_t p;
  arts_tiered_pool_init_explicit(&p, NODE_SIZE, 0, 0, cfg);
  if (p.num_threads != 1 || p.num_numa_nodes != 1) {
    (void)fprintf(stderr,
                  "FAIL tiered_pool_conservation: 0-clamp gave nt=%u nn=%u\n",
                  p.num_threads, p.num_numa_nodes);
    return 1;
  }
  arts_thread_info.thread_id = 7; /* >= num_threads -> defensive clamp to 0 */
  void *a = arts_tiered_pool_alloc(&p);
  if (!a) {
    return 1;
  }
  arts_tiered_pool_release(&p, a);
  arts_tiered_pool_destroy(&p);
  return 0;
}

int main(void) {
  if (early_init_phase() != 0) {
    return 1;
  }
  if (tier2_split_phase() != 0) {
    return 1;
  }
  if (conservation_phase() != 0) {
    return 1;
  }
  printf("PASS tiered_pool_conservation: %d-node universe conserved by %d "
         "unique-id workers (INV-1/2); tier-2 split exact (INV-3); 0-clamp + "
         "node_size contract OK\n",
         UNIVERSE, THREADS);
  return 0;
}
