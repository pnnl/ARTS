/* SPDX-License-Identifier: Apache-2.0
 *
 * T003 — arts_lf_stack_drain / arts_lf_stack_reverse_drain (lockfree_lifo.h).
 *
 * Three properties (census 29.md §2 GAPS):
 *
 *  (1) ORDERING (single-thread, deterministic): push 0..K-1 in order; drain
 *      must return LIFO (K-1 .. 0); on a fresh fill reverse_drain must return
 *      FIFO (0 .. K-1).  The event subsystem relies on reverse_drain being
 *      FIFO, so both orderings are pinned exactly.
 *
 *  (2) DISJOINT PARTITIONS (concurrent): one stack, P producers pushing N
 *      nodes each while D drainer threads repeatedly drain().  drain() is a
 *      single atomic_exchange(NULL): each node is detached by exactly one
 *      drainer.  Across all drainers + the final cleanup drain, every pushed
 *      node appears exactly once (no loss / no dup / no double-detach).
 *
 *  (3) PUBLISH/CONSUME HAPPENS-BEFORE: each node's payload is written before
 *      push (release CAS); a drainer reads it after drain (acquire exchange).
 *      A torn read would mean the release/acquire pairing is broken.
 *
 * The no-reentry invariant is respected: drained nodes are NEVER re-pushed to
 * the same stack — each drainer collects them into a private list and we tally
 * at the end.
 */

#include "arts/utils/lockfree_lifo.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ORD_K 1000
#define PRODUCERS 6
#define PER_PRODUCER 100000
#define DRAINERS 4
#define TOTAL ((size_t)PRODUCERS * PER_PRODUCER)
#define MAGIC 0xABCDu

typedef struct {
  arts_lf_link_t link;
  uint32_t id;
  uint32_t magic;
} node_t;

/* ---- Part 1: deterministic ordering ---- */
static int ordering_check(void) {
  arts_lf_stack_t s;
  arts_lf_stack_init(&s);
  node_t *nodes = (node_t *)calloc(ORD_K, sizeof(node_t));
  if (!nodes) {
    return 1;
  }
  for (int i = 0; i < ORD_K; i++) {
    nodes[i].id = (uint32_t)i;
    arts_lf_stack_push(&s, &nodes[i].link);
  }
  /* drain == LIFO: expect ORD_K-1 down to 0. */
  arts_lf_link_t *l = arts_lf_stack_drain(&s);
  for (int expect = ORD_K - 1; expect >= 0; expect--) {
    if (!l) {
      (void)fprintf(stderr, "FAIL lf_lifo_drain: LIFO short at %d\n", expect);
      return 1;
    }
    node_t *n = (node_t *)l;
    if ((int)n->id != expect) {
      (void)fprintf(stderr, "FAIL lf_lifo_drain: LIFO got %u expect %d\n",
                    n->id, expect);
      return 1;
    }
    l = atomic_load_explicit(&l->next, memory_order_relaxed);
  }
  if (l != NULL) {
    (void)fprintf(stderr, "FAIL lf_lifo_drain: LIFO trailing nodes\n");
    return 1;
  }

  /* re-fill, reverse_drain == FIFO: expect 0 up to ORD_K-1. */
  for (int i = 0; i < ORD_K; i++) {
    arts_lf_stack_push(&s, &nodes[i].link);
  }
  l = arts_lf_stack_reverse_drain(&s);
  for (int expect = 0; expect < ORD_K; expect++) {
    if (!l) {
      (void)fprintf(stderr, "FAIL lf_lifo_drain: FIFO short at %d\n", expect);
      return 1;
    }
    node_t *n = (node_t *)l;
    if ((int)n->id != expect) {
      (void)fprintf(stderr, "FAIL lf_lifo_drain: FIFO got %u expect %d\n",
                    n->id, expect);
      return 1;
    }
    l = atomic_load_explicit(&l->next, memory_order_relaxed);
  }
  if (l != NULL) {
    (void)fprintf(stderr, "FAIL lf_lifo_drain: FIFO trailing nodes\n");
    return 1;
  }
  free(nodes);
  return 0;
}

/* ---- Part 2/3: concurrent disjoint partitions + happens-before ---- */
static arts_lf_stack_t g_s;
static node_t *g_nodes;
static atomic_int g_start;
static _Atomic uint64_t g_prod_done;

/* Per-node "seen" tally accumulated by drainers; atomic to detect any
 * double-detach across drainers. */
static _Atomic uint8_t *g_seen;
static _Atomic uint64_t g_total_seen;

static void *producer(void *arg) {
  uint32_t base = (uint32_t)(uintptr_t)arg * PER_PRODUCER;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (uint32_t i = 0; i < PER_PRODUCER; i++) {
    node_t *n = &g_nodes[base + i];
    n->id = base + i;
    n->magic = MAGIC; /* publish before push */
    arts_lf_stack_push(&g_s, &n->link);
  }
  atomic_fetch_add_explicit(&g_prod_done, 1, memory_order_release);
  return NULL;
}

static void consume_chain(arts_lf_link_t *l) {
  while (l) {
    arts_lf_link_t *next = atomic_load_explicit(&l->next, memory_order_relaxed);
    node_t *n = (node_t *)l;
    if (n->magic != MAGIC || n->id >= TOTAL) {
      (void)fprintf(stderr, "FAIL lf_lifo_drain: torn drained node id=%u\n",
                    n->id);
      abort();
    }
    uint8_t prev =
        atomic_fetch_add_explicit(&g_seen[n->id], 1, memory_order_relaxed);
    if (prev != 0) {
      (void)fprintf(stderr,
                    "FAIL lf_lifo_drain: node %u detached twice (disjoint "
                    "partition violated)\n",
                    n->id);
      abort();
    }
    atomic_fetch_add_explicit(&g_total_seen, 1, memory_order_relaxed);
    l = next;
  }
}

static void *drainer(void *arg) {
  (void)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (;;) {
    arts_lf_link_t *chain = arts_lf_stack_drain(&g_s);
    consume_chain(chain);
    if (atomic_load_explicit(&g_prod_done, memory_order_acquire) == PRODUCERS &&
        atomic_load_explicit(&g_total_seen, memory_order_relaxed) >= TOTAL) {
      break;
    }
  }
  return NULL;
}

static int concurrent_check(void) {
  arts_lf_stack_init(&g_s);
  atomic_init(&g_start, 0);
  atomic_init(&g_prod_done, 0);
  atomic_init(&g_total_seen, 0);
  g_nodes = (node_t *)calloc(TOTAL, sizeof(node_t));
  g_seen = (_Atomic uint8_t *)calloc(TOTAL, sizeof(_Atomic uint8_t));
  if (!g_nodes || !g_seen) {
    return 1;
  }

  pthread_t prod[PRODUCERS];
  pthread_t drn[DRAINERS];
  for (int i = 0; i < PRODUCERS; i++) {
    pthread_create(&prod[i], NULL, producer, (void *)(uintptr_t)i);
  }
  for (int i = 0; i < DRAINERS; i++) {
    pthread_create(&drn[i], NULL, drainer, NULL);
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < PRODUCERS; i++) {
    pthread_join(prod[i], NULL);
  }
  for (int i = 0; i < DRAINERS; i++) {
    pthread_join(drn[i], NULL);
  }
  /* Final cleanup drain (a drainer may have exited just before the last
   * push landed). */
  consume_chain(arts_lf_stack_drain(&g_s));

  if (atomic_load_explicit(&g_total_seen, memory_order_relaxed) != TOTAL) {
    (void)fprintf(stderr, "FAIL lf_lifo_drain: total %" PRIu64 " != %zu\n",
                  atomic_load_explicit(&g_total_seen, memory_order_relaxed),
                  TOTAL);
    return 1;
  }
  for (size_t i = 0; i < TOTAL; i++) {
    if (atomic_load_explicit(&g_seen[i], memory_order_relaxed) != 1) {
      (void)fprintf(stderr, "FAIL lf_lifo_drain: node %zu seen %u times\n", i,
                    atomic_load_explicit(&g_seen[i], memory_order_relaxed));
      return 1;
    }
  }
  free(g_nodes);
  free((void *)g_seen);
  return 0;
}

int main(void) {
  if (ordering_check() != 0) {
    return 1;
  }
  if (concurrent_check() != 0) {
    return 1;
  }
  printf("PASS lf_lifo_drain: LIFO/FIFO ordering pinned; %zu nodes drained "
         "disjointly by %d drainers, no loss/dup/torn\n",
         TOTAL, DRAINERS);
  return 0;
}
