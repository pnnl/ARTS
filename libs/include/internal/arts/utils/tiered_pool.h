/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_tiered_pool_t — 3-tier hierarchical object pool.
 *
 * Tier 0  : thread-private cache (zero atomics on hit, indexed by
 *           arts_get_worker_id())
 * Tier 1  : per-NUMA shard (one arts_lockfree_pool_t per NUMA domain;
 *           cache-line-padded)
 * Tier 2  : global arts_lockfree_pool_t (cross-NUMA spillover)
 * Tier 3  : heap fallback via arts_calloc (only when all tiers empty)
 *
 * Each pool instance picks watermarks (H_local, B_local, H_numa, B_numa)
 * from the known call pattern of its caller — unlike jemalloc/tcmalloc
 * which tune by generic size class, we tune by call site.
 *
 * arts_get_numa_id() always returns 0 and num_numa_nodes
 * is forced to 1.  Tier 1 thus collapses to a single shard for now —
 * the tier-0 fast path and tier-2 spillover still exercise.  Replace
 * with hwloc lookup when migrating event_dep_pool to the actual
 * workload.
 */

#ifndef ARTS_UTILS_TIERED_POOL_H
#define ARTS_UTILS_TIERED_POOL_H

#include "arts/defs.h"                /* ARTS_ALIGNED */
#include "arts/runtime_state.h"       /* arts_thread_info, arts_node_info */
#include "arts/utils/lockfree_lifo.h" /* arts_lf_link_t */
#include "arts/utils/lockfree_pool.h" /* arts_lockfree_pool_t + batch ops */
#include "arts/utils/malloc.h"        /* arts_calloc / arts_free */

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Tier-0 thread-private cache.  Plain pointers — no atomics, no locks.
 *  Owner is uniquely determined by arts_get_worker_id() at access time. */
typedef struct {
  arts_lf_link_t *head;
  uint32_t count;
} arts_pool_tcache_t;

/** Tier-1 NUMA shard.  Cache-line padded to avoid false-sharing across
 *  shards (different NUMA domains touch disjoint shards but the array of
 *  shards lives in contiguous memory). */
typedef struct {
  arts_lockfree_pool_t pool;
  /* Pad to a 64-byte cache line — relax if pool grows. */
  char _pad[64 - (sizeof(arts_lockfree_pool_t) % 64)];
} ARTS_ALIGNED(64) arts_pool_numa_shard_t;

typedef struct {
  uint32_t H_local; /**< tcache → NUMA spill threshold (in node count). */
  uint32_t B_local; /**< Batch size for tcache ↔ NUMA transfer. */
  uint32_t H_numa;  /**< NUMA → global spill threshold. */
  uint32_t B_numa;  /**< Batch size for NUMA ↔ global transfer. */
} arts_tiered_pool_cfg_t;

typedef struct arts_tiered_pool_s {
  arts_pool_tcache_t *tcache;   /**< [num_threads] */
  arts_pool_numa_shard_t *numa; /**< [num_numa_nodes] */
  uint32_t num_threads;         /**< len(tcache) */
  uint32_t num_numa_nodes;      /**< len(numa) */
  arts_lockfree_pool_t global;
  size_t node_size;
  arts_tiered_pool_cfg_t cfg;
} arts_tiered_pool_t;

/* ── Worker / NUMA id resolution ───────────────────────────────────────── */

/** Caller's worker thread id.  Falls back to 0 if the runtime has not
 *  initialized the per-thread state (e.g., very early init). */
static inline uint32_t arts_tiered_pool_worker_id(void) {
  uint32_t tid = (uint32_t)arts_thread_info.thread_id;
  return tid;
}

/** collapse to one NUMA shard.  Replace with
 *  hwloc-based lookup (`arts_thread_info.numa_domain_id`) when the
 *  consumer (event_dep_pool) is wired up. */
static inline uint32_t arts_tiered_pool_numa_id(uint32_t tid) {
  (void)tid;
  return 0;
}

/* ── Tier-0 helpers (thread-private, no atomics) ──────────────────────── */

static inline void arts_tiered_pool_tcache_push(arts_pool_tcache_t *c,
                                                arts_lf_link_t *node) {
  atomic_store_explicit(&node->next, c->head, memory_order_relaxed);
  c->head = node;
  c->count++;
}

static inline arts_lf_link_t *
arts_tiered_pool_tcache_pop(arts_pool_tcache_t *c) {
  arts_lf_link_t *n = c->head;
  if (!n) {
    return NULL;
  }
  c->head = atomic_load_explicit(&n->next, memory_order_relaxed);
  c->count--;
  return n;
}

/** Detach the top `n` nodes from the tcache, returning chain head/tail
 *  via out-params.  Reduces c->count by the actual number detached. */
static inline void arts_tiered_pool_tcache_detach(arts_pool_tcache_t *c,
                                                  uint32_t n,
                                                  arts_lf_link_t **out_head,
                                                  arts_lf_link_t **out_tail,
                                                  uint32_t *out_n) {
  *out_head = NULL;
  *out_tail = NULL;
  if (out_n) {
    *out_n = 0;
  }
  if (n == 0 || !c->head) {
    return;
  }
  arts_lf_link_t *head = c->head;
  arts_lf_link_t *tail = head;
  uint32_t taken = 1;
  arts_lf_link_t *next =
      atomic_load_explicit(&tail->next, memory_order_relaxed);
  while (taken < n && next) {
    tail = next;
    next = atomic_load_explicit(&tail->next, memory_order_relaxed);
    taken++;
  }
  /* Disconnect the detached chain from the tcache. */
  c->head = next;
  c->count -= taken;
  atomic_store_explicit(&tail->next, NULL, memory_order_relaxed);
  *out_head = head;
  *out_tail = tail;
  if (out_n) {
    *out_n = taken;
  }
}

/** Install a chain of `n` nodes [head .. ?] into the tcache by walking
 *  forward from `head` to find the tail and stitching.  Used after
 *  batch_fetch from NUMA / global. */
static inline void arts_tiered_pool_tcache_install(arts_pool_tcache_t *c,
                                                   arts_lf_link_t *head,
                                                   uint32_t n) {
  if (!head || n == 0) {
    return;
  }
  arts_lf_link_t *tail = head;
  uint32_t walked = 1;
  while (walked < n) {
    arts_lf_link_t *next =
        atomic_load_explicit(&tail->next, memory_order_relaxed);
    if (!next) {
      break;
    }
    tail = next;
    walked++;
  }
  atomic_store_explicit(&tail->next, c->head, memory_order_relaxed);
  c->head = head;
  c->count += walked;
}

/* ── Initialization / teardown ─────────────────────────────────────────── */

/** Initialize a tiered pool.  `num_threads` must equal the runtime's
 *  total thread count (workers + senders + receivers); pass it
 *  explicitly so the pool works during early init when arts_node_info
 *  may not yet be populated. */
static inline void arts_tiered_pool_init_explicit(arts_tiered_pool_t *p,
                                                  size_t node_size,
                                                  uint32_t num_threads,
                                                  uint32_t num_numa_nodes,
                                                  arts_tiered_pool_cfg_t cfg) {
  if (num_threads == 0) {
    num_threads = 1;
  }
  if (num_numa_nodes == 0) {
    num_numa_nodes = 1;
  }
  p->num_threads = num_threads;
  p->num_numa_nodes = num_numa_nodes;
  p->node_size = node_size;
  p->cfg = cfg;
  p->tcache =
      (arts_pool_tcache_t *)arts_calloc(num_threads, sizeof(*p->tcache));
  p->numa = (arts_pool_numa_shard_t *)arts_calloc_align(num_numa_nodes,
                                                        sizeof(*p->numa), 64);
  for (uint32_t i = 0; i < num_numa_nodes; i++) {
    arts_lf_pool_init(&p->numa[i].pool, node_size);
  }
  arts_lf_pool_init(&p->global, node_size);
}

/** Convenience init for the common case: pulls num_threads /
 *  num_numa_nodes from runtime state.  Stub forces
 *  num_numa_nodes=1 (see arts_tiered_pool_numa_id). */
static inline void arts_tiered_pool_init(arts_tiered_pool_t *p,
                                         size_t node_size,
                                         arts_tiered_pool_cfg_t cfg) {
  uint32_t nt = (uint32_t)arts_node_info.total_thread_count;
  if (nt == 0) {
    nt = 1;
  }
  /* single NUMA shard.  Replace with hwloc lookup
   * (arts_node_info-cached num_numa_nodes) when migrating the consumer. */
  uint32_t nn = 1;
  arts_tiered_pool_init_explicit(p, node_size, nt, nn, cfg);
}

/** Drain all tiers and free every node.  Caller is responsible for
 *  ensuring no other thread touches the pool during destroy. */
static inline void arts_tiered_pool_destroy(arts_tiered_pool_t *p) {
  /* Drain tier-0 tcaches. */
  if (p->tcache) {
    for (uint32_t t = 0; t < p->num_threads; t++) {
      arts_lf_link_t *n = p->tcache[t].head;
      while (n) {
        arts_lf_link_t *next =
            atomic_load_explicit(&n->next, memory_order_relaxed);
        arts_free(n);
        n = next;
      }
      p->tcache[t].head = NULL;
      p->tcache[t].count = 0;
    }
    arts_free(p->tcache);
    p->tcache = NULL;
  }
  /* Drain tier-1 NUMA shards. */
  if (p->numa) {
    for (uint32_t i = 0; i < p->num_numa_nodes; i++) {
      arts_lf_pool_destroy(&p->numa[i].pool);
    }
    arts_free(p->numa);
    p->numa = NULL;
  }
  /* Drain tier-2 global. */
  arts_lf_pool_destroy(&p->global);
}

/* ── Alloc ─────────────────────────────────────────────────────────────── */

static inline void *arts_tiered_pool_alloc(arts_tiered_pool_t *p) {
  uint32_t tid = arts_tiered_pool_worker_id();
  if (tid >= p->num_threads) {
    tid = 0; /* defensive */
  }
  arts_pool_tcache_t *c = &p->tcache[tid];

  /* Tier 0 — thread-private cache, zero atomics on hit. */
  arts_lf_link_t *n = arts_tiered_pool_tcache_pop(c);
  if (n) {
    return n;
  }

  uint32_t numa_id = arts_tiered_pool_numa_id(tid);
  if (numa_id >= p->num_numa_nodes) {
    numa_id = 0;
  }

  /* Tier 1 — NUMA shard, batch-fetch B_local nodes. */
  uint32_t got = 0;
  arts_lf_link_t *batch =
      arts_lf_pool_batch_fetch(&p->numa[numa_id].pool, p->cfg.B_local, &got);
  if (batch && got > 0) {
    /* Return the head; install the rest in tcache. */
    arts_lf_link_t *rest =
        atomic_load_explicit(&batch->next, memory_order_relaxed);
    atomic_store_explicit(&batch->next, NULL, memory_order_relaxed);
    if (rest && got > 1) {
      arts_tiered_pool_tcache_install(c, rest, got - 1);
    }
    return batch;
  }

  /* Tier 2 — global, larger batch (refills NUMA shard then tcache). */
  batch = arts_lf_pool_batch_fetch(&p->global, p->cfg.B_numa, &got);
  if (batch && got > 0) {
    /* Return head; route B_local-1 to tcache, rest (if any) to NUMA. */
    arts_lf_link_t *rest =
        atomic_load_explicit(&batch->next, memory_order_relaxed);
    atomic_store_explicit(&batch->next, NULL, memory_order_relaxed);
    uint32_t remaining = got - 1; /* head consumed */

    /* Move up to (B_local - 1) into tcache. */
    uint32_t to_tcache = (p->cfg.B_local > 0) ? (p->cfg.B_local - 1) : 0;
    if (to_tcache > remaining) {
      to_tcache = remaining;
    }

    if (rest && to_tcache > 0) {
      /* Walk to find the split tail. */
      arts_lf_link_t *split_tail = rest;
      uint32_t walked = 1;
      while (walked < to_tcache) {
        arts_lf_link_t *next =
            atomic_load_explicit(&split_tail->next, memory_order_relaxed);
        if (!next) {
          break;
        }
        split_tail = next;
        walked++;
      }
      arts_lf_link_t *to_numa_head =
          atomic_load_explicit(&split_tail->next, memory_order_relaxed);
      atomic_store_explicit(&split_tail->next, NULL, memory_order_relaxed);

      /* Install [rest .. split_tail] (walked nodes) into tcache. */
      atomic_store_explicit(&split_tail->next, c->head, memory_order_relaxed);
      c->head = rest;
      c->count += walked;
      remaining -= walked;

      /* Push remaining tail into NUMA shard. */
      if (to_numa_head && remaining > 0) {
        arts_lf_link_t *numa_tail = to_numa_head;
        uint32_t numa_walked = 1;
        while (numa_walked < remaining) {
          arts_lf_link_t *next =
              atomic_load_explicit(&numa_tail->next, memory_order_relaxed);
          if (!next) {
            break;
          }
          numa_tail = next;
          numa_walked++;
        }
        arts_lf_pool_batch_release(&p->numa[numa_id].pool, to_numa_head,
                                   numa_tail, numa_walked);
      }
    } else if (rest && remaining > 0) {
      /* No tcache route — push everything to NUMA. */
      arts_lf_link_t *numa_tail = rest;
      uint32_t numa_walked = 1;
      while (numa_walked < remaining) {
        arts_lf_link_t *next =
            atomic_load_explicit(&numa_tail->next, memory_order_relaxed);
        if (!next) {
          break;
        }
        numa_tail = next;
        numa_walked++;
      }
      arts_lf_pool_batch_release(&p->numa[numa_id].pool, rest, numa_tail,
                                 numa_walked);
    }
    return batch;
  }

  /* Tier 3 — heap fallback. */
  return arts_calloc(1, p->node_size);
}

/* ── Release ───────────────────────────────────────────────────────────── */

static inline void arts_tiered_pool_release(arts_tiered_pool_t *p, void *node) {
  /* Use current-thread cache (not alloc-thread cache).  Cross-thread
   * alloc/release cycles balance through tier 1/2. */
  uint32_t tid = arts_tiered_pool_worker_id();
  if (tid >= p->num_threads) {
    tid = 0;
  }
  arts_pool_tcache_t *c = &p->tcache[tid];
  arts_lf_link_t *n = (arts_lf_link_t *)node;

  /* Tier 0 fast path. */
  if (c->count < p->cfg.H_local) {
    arts_tiered_pool_tcache_push(c, n);
    return;
  }

  /* Tier 0 full → spill B_local to NUMA, then push the new node. */
  uint32_t numa_id = arts_tiered_pool_numa_id(tid);
  if (numa_id >= p->num_numa_nodes) {
    numa_id = 0;
  }

  arts_lf_link_t *spill_head = NULL;
  arts_lf_link_t *spill_tail = NULL;
  uint32_t spill_n = 0;
  arts_tiered_pool_tcache_detach(c, p->cfg.B_local, &spill_head, &spill_tail,
                                 &spill_n);
  if (spill_head && spill_n > 0) {
    arts_lf_pool_batch_release(&p->numa[numa_id].pool, spill_head, spill_tail,
                               spill_n);
  }

  /* If NUMA is at high-water, spill B_numa to global. */
  uint32_t numa_count =
      atomic_load_explicit(&p->numa[numa_id].pool.count, memory_order_relaxed);
  if (numa_count > p->cfg.H_numa) {
    arts_lf_link_t *gh = NULL;
    arts_lf_link_t *gt = NULL;
    arts_lf_pool_batch_drain(&p->numa[numa_id].pool, p->cfg.B_numa, &gh, &gt);
    if (gh && gt) {
      /* batch_drain doesn't report count, walk to compute (chain tail
       * already terminated by batch_drain). */
      uint32_t cnt = 0;
      arts_lf_link_t *cur = gh;
      while (cur) {
        cnt++;
        cur = atomic_load_explicit(&cur->next, memory_order_relaxed);
      }
      arts_lf_pool_batch_release(&p->global, gh, gt, cnt);
    }
  }

  /* Push the freshly released node onto the (now-spilled) tcache. */
  arts_tiered_pool_tcache_push(c, n);
}

#ifdef __cplusplus
}
#endif

#endif /* ARTS_UTILS_TIERED_POOL_H */
