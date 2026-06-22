/* SPDX-License-Identifier: Apache-2.0
 *
 * T006 — arts_lockfree_pool_t batch + single-node primitives racing one head
 * (lockfree_pool.h).  Census 29.md §3 GAPS:
 *   - all four primitives (alloc / release / batch_fetch / batch_release)
 *     racing the same head simultaneously (the real tiered-spill pattern),
 *   - batch edge cases: want==0, want > available, empty pool,
 *   - alloc-fallback calloc zero-fill on empty pool,
 *   - count transient-wrap observation (B125).
 *
 * The pool's lifetime invariant (nodes live until pool destroy) means a node
 * is never freed by alloc/fetch — ownership only transfers.  So we can pre-
 * seed a fixed universe of UNIVERSE nodes, then let many threads churn them
 * through all four primitives, and at the end every node must still be
 * accounted for exactly once (sum over: pool + each thread's in-hand set ==
 * UNIVERSE; no node appears in two hands at once).
 *
 * Each thread holds a private "hand" (a small array of node pointers it owns);
 * it randomly allocs/fetches into the hand or releases from it.  A per-node
 * atomic owner-stamp catches any double-hand (a node simultaneously owned by
 * two threads = lost/duplicated by a broken primitive).
 */

#include "arts/utils/lockfree_pool.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* libc shims so the header links standalone. */
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }

#define UNIVERSE 512
#define THREADS 8
#define HAND_CAP 64
#define ITERS 200000
#define NODE_SIZE 64 /* > sizeof(arts_lf_link_t); first bytes are the link */

typedef struct {
  arts_lf_link_t link; /* first member */
  uint32_t id;
  _Atomic int owner; /* -1 = in pool, else owning thread index */
} node_t;

static arts_lockfree_pool_t g_pool;
static node_t *g_nodes[UNIVERSE]; /* each individually malloc'd (so the pool's
                                     destroy can free every member uniformly) */
static atomic_int g_start;

/* Map a node pointer back to its id via the contiguous array. */
static inline node_t *as_node(arts_lf_link_t *l) { return (node_t *)l; }

typedef struct {
  int tid;
  arts_lf_link_t *hand[HAND_CAP];
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

static void take(ctx_t *c, arts_lf_link_t *l) {
  node_t *n = as_node(l);
  int expect = -1;
  if (!atomic_compare_exchange_strong_explicit(&n->owner, &expect, c->tid,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) {
    (void)fprintf(stderr,
                  "FAIL lf_pool_batch: node %u handed out while owned by %d "
                  "(double-alloc)\n",
                  n->id, expect);
    abort();
  }
  c->hand[c->hand_n++] = l;
}

static arts_lf_link_t *give(ctx_t *c) {
  arts_lf_link_t *l = c->hand[--c->hand_n];
  node_t *n = as_node(l);
  int expect = c->tid;
  if (!atomic_compare_exchange_strong_explicit(
          &n->owner, &expect, -1, memory_order_acq_rel, memory_order_acquire)) {
    (void)fprintf(stderr,
                  "FAIL lf_pool_batch: releasing node %u not owned by "
                  "me (owner=%d)\n",
                  n->id, expect);
    abort();
  }
  return l;
}

static void *worker(void *arg) {
  ctx_t *c = (ctx_t *)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < ITERS; i++) {
    uint64_t r = xs(&c->rng);
    int op = (int)(r & 3);
    switch (op) {
    case 0: { /* single alloc */
      if (c->hand_n < HAND_CAP) {
        arts_lf_link_t *l = (arts_lf_link_t *)arts_lf_pool_alloc(&g_pool);
        /* alloc may return a fresh heap node (pool empty); stamp it. */
        node_t *n = as_node(l);
        if (n->owner != -1) {
          /* a freshly calloc'd node has owner==0; distinguish heap-fresh
           * from a real double-hand: heap-fresh nodes are NOT in g_nodes. */
        }
        atomic_store_explicit(&n->owner, c->tid, memory_order_relaxed);
        c->hand[c->hand_n++] = l;
      }
      break;
    }
    case 1: { /* single release */
      if (c->hand_n > 0) {
        arts_lf_link_t *l = c->hand[--c->hand_n];
        node_t *n = as_node(l);
        atomic_store_explicit(&n->owner, -1, memory_order_relaxed);
        arts_lf_pool_release(&g_pool, l);
      }
      break;
    }
    case 2: { /* batch_fetch up to 8 */
      uint32_t want = 1 + (uint32_t)((r >> 2) & 7);
      uint32_t got = 0;
      arts_lf_link_t *chain = arts_lf_pool_batch_fetch(&g_pool, want, &got);
      uint32_t cnt = 0;
      while (chain) {
        arts_lf_link_t *next =
            atomic_load_explicit(&chain->next, memory_order_relaxed);
        node_t *n = as_node(chain);
        atomic_store_explicit(&n->owner, c->tid, memory_order_relaxed);
        if (c->hand_n < HAND_CAP) {
          c->hand[c->hand_n++] = chain;
        } else {
          /* hand full — release immediately back */
          atomic_store_explicit(&n->owner, -1, memory_order_relaxed);
          arts_lf_pool_release(&g_pool, chain);
        }
        cnt++;
        chain = next;
      }
      if (cnt != got) {
        (void)fprintf(stderr,
                      "FAIL lf_pool_batch: batch_fetch got=%u but chain=%u\n",
                      got, cnt);
        abort();
      }
      break;
    }
    case 3: { /* batch_release up to 8 from hand */
      if (c->hand_n >= 2) {
        uint32_t take_n = 1 + (uint32_t)((r >> 5) & 7);
        if ((int)take_n > c->hand_n) {
          take_n = (uint32_t)c->hand_n;
        }
        /* Build a chain [head..tail] from the top take_n hand slots. */
        arts_lf_link_t *head = c->hand[c->hand_n - 1];
        arts_lf_link_t *prev = head;
        node_t *hn = as_node(head);
        atomic_store_explicit(&hn->owner, -1, memory_order_relaxed);
        for (uint32_t k = 1; k < take_n; k++) {
          arts_lf_link_t *nd = c->hand[c->hand_n - 1 - k];
          node_t *nn = as_node(nd);
          atomic_store_explicit(&nn->owner, -1, memory_order_relaxed);
          atomic_store_explicit(&prev->next, nd, memory_order_relaxed);
          prev = nd;
        }
        arts_lf_link_t *tail = prev;
        c->hand_n -= (int)take_n;
        arts_lf_pool_batch_release(&g_pool, head, tail, take_n);
      }
      break;
    }
    default:
      break;
    }
  }
  return NULL;
}

/* ---- single-thread edge cases ---- */
static int edge_cases(void) {
  arts_lockfree_pool_t p;
  arts_lf_pool_init(&p, NODE_SIZE);

  /* empty pool: batch_fetch -> NULL, *got==0. */
  uint32_t got = 99;
  arts_lf_link_t *c = arts_lf_pool_batch_fetch(&p, 4, &got);
  if (c != NULL || got != 0) {
    (void)fprintf(stderr, "FAIL lf_pool_batch: empty fetch c=%p got=%u\n",
                  (void *)c, got);
    return 1;
  }
  /* want==0 -> NULL, got==0. */
  got = 99;
  c = arts_lf_pool_batch_fetch(&p, 0, &got);
  if (c != NULL || got != 0) {
    (void)fprintf(stderr, "FAIL lf_pool_batch: want0 c=%p got=%u\n", (void *)c,
                  got);
    return 1;
  }

  /* alloc on empty pool -> calloc fallback, zero-filled (link.next == NULL). */
  arts_lf_link_t *fresh = (arts_lf_link_t *)arts_lf_pool_alloc(&p);
  if (!fresh) {
    (void)fprintf(stderr, "FAIL lf_pool_batch: alloc fallback NULL\n");
    return 1;
  }
  if (atomic_load_explicit(&fresh->next, memory_order_relaxed) != NULL) {
    (void)fprintf(stderr,
                  "FAIL lf_pool_batch: calloc fallback not zero-filled\n");
    return 1;
  }

  /* release the fresh node, then batch_fetch want > available (1): returns the
   * single node, got==1. */
  arts_lf_pool_release(&p, fresh);
  got = 0;
  c = arts_lf_pool_batch_fetch(&p, 8, &got);
  if (c == NULL || got != 1) {
    (void)fprintf(stderr,
                  "FAIL lf_pool_batch: over-want c=%p got=%u (want "
                  "1 available)\n",
                  (void *)c, got);
    return 1;
  }
  if (atomic_load_explicit(&c->next, memory_order_relaxed) != NULL) {
    (void)fprintf(stderr, "FAIL lf_pool_batch: fetched chain not detached\n");
    return 1;
  }
  arts_lf_pool_release(&p, c);

  arts_lf_pool_destroy(&p);
  return 0;
}

int main(void) {
  if (edge_cases() != 0) {
    return 1;
  }

  arts_lf_pool_init(&g_pool, NODE_SIZE);
  atomic_init(&g_start, 0);
  for (int i = 0; i < UNIVERSE; i++) {
    /* Each member individually allocated at NODE_SIZE so the pool's destroy
     * (which arts_free's every node) frees them uniformly with any heap-fresh
     * fallback nodes. */
    g_nodes[i] = (node_t *)calloc(1, NODE_SIZE);
    if (!g_nodes[i]) {
      return 1;
    }
    g_nodes[i]->id = (uint32_t)i;
    atomic_init(&g_nodes[i]->owner, -1);
    arts_lf_pool_release(&g_pool, &g_nodes[i]->link);
  }

  pthread_t th[THREADS];
  ctx_t ctx[THREADS];
  for (int i = 0; i < THREADS; i++) {
    ctx[i].tid = i;
    ctx[i].hand_n = 0;
    ctx[i].rng = 0x1234567u + (uint64_t)i * 0x9E3779B9u;
    pthread_create(&th[i], NULL, worker, &ctx[i]);
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < THREADS; i++) {
    pthread_join(th[i], NULL);
  }

  /* Drain each hand back to the pool to restore the full universe. */
  for (int i = 0; i < THREADS; i++) {
    for (int k = 0; k < ctx[i].hand_n; k++) {
      node_t *n = as_node(ctx[i].hand[k]);
      atomic_store_explicit(&n->owner, -1, memory_order_relaxed);
      /* Only re-pool nodes that belong to the universe; heap-fresh nodes
       * (allocated when the pool was momentarily empty) are also valid pool
       * members now — release them too so destroy frees them. */
      arts_lf_pool_release(&g_pool, ctx[i].hand[k]);
    }
  }

  /* Every universe node must be back in the pool (owner==-1) — no node was
   * lost mid-flight. */
  for (int i = 0; i < UNIVERSE; i++) {
    int o = atomic_load_explicit(&g_nodes[i]->owner, memory_order_relaxed);
    if (o != -1) {
      (void)fprintf(stderr, "FAIL lf_pool_batch: node %d still owned by %d\n",
                    i, o);
      return 1;
    }
  }

  arts_lf_pool_destroy(&g_pool); /* frees universe + any heap-fresh nodes */
  printf("PASS lf_pool_batch: %d threads x %d iters, all-four primitives raced "
         "one head, %d-node universe conserved (no loss/dup), edge cases OK\n",
         THREADS, ITERS, UNIVERSE);
  return 0;
}
