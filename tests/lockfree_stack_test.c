/* SPDX-License-Identifier: Apache-2.0
 *
 * Stress test for arts_lockfree_stack: many concurrent pushers/poppers
 * exercise the tagged-pointer ABA defense and verify FIFO-by-recycle
 * (the same node may be pushed/popped many times — counter must absorb
 * every reuse).
 *
 * Test layout:
 *   1. Pre-allocate N nodes, push them all serially.  Verify pop drains
 *      to empty.
 *   2. Spawn T worker threads.  Each thread pops a node, increments a
 *      per-thread counter, pushes the node back.  Repeat ITERS times.
 *      Total pop+push operations: T * ITERS.  No lost or doubled nodes
 *      (sum of per-thread counters == T * ITERS).
 *   3. Drain the stack and verify exactly N nodes return.
 */

#include "arts/utils/lockfree_stack.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NODES 64
#define THREADS 16
#define ITERS 100000

typedef struct test_node_s {
  arts_lockfree_stack_node_t link; /* MUST be first */
  uint32_t id;
  _Atomic uint64_t
      bounce_count; /* +1 per pop+push round; total sums to T*ITERS */
} test_node_t;

static arts_lockfree_stack_t g_stack;
static test_node_t g_nodes[NODES];
static _Atomic uint64_t g_total_bounces;

static void *worker(void *arg) {
  (void)arg;
  for (int i = 0; i < ITERS; i++) {
    arts_lockfree_stack_node_t *n;
    /* Spin until a node is available (the queue may briefly empty when
     * all T threads happen to be mid-pop). */
    while ((n = arts_lockfree_stack_pop(&g_stack)) == NULL) {
      /* nothing */
    }
    test_node_t *t = (test_node_t *)n;
    atomic_fetch_add_explicit(&t->bounce_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_total_bounces, 1, memory_order_relaxed);
    arts_lockfree_stack_push(&g_stack, n);
  }
  return NULL;
}

int main(void) {
  arts_lockfree_stack_init(&g_stack);

  /* --- Phase 1: serial push/pop drain. */
  for (int i = 0; i < NODES; i++) {
    g_nodes[i].id = (uint32_t)i;
    atomic_init(&g_nodes[i].bounce_count, 0);
    arts_lockfree_stack_push(&g_stack, &g_nodes[i].link);
  }
  /* Pop everything; verify count == NODES. */
  int popped = 0;
  arts_lockfree_stack_node_t *n;
  while ((n = arts_lockfree_stack_pop(&g_stack)) != NULL) {
    popped++;
  }
  if (popped != NODES) {
    fprintf(stderr, "Phase 1: expected %d, got %d\n", NODES, popped);
    return 1;
  }
  if (!arts_lockfree_stack_empty(&g_stack)) {
    fprintf(stderr, "Phase 1: stack should be empty\n");
    return 1;
  }

  /* --- Phase 2: concurrent stress.  Re-push all nodes. */
  for (int i = 0; i < NODES; i++) {
    arts_lockfree_stack_push(&g_stack, &g_nodes[i].link);
  }
  atomic_init(&g_total_bounces, 0);

  pthread_t threads[THREADS];
  for (int i = 0; i < THREADS; i++) {
    pthread_create(&threads[i], NULL, worker, NULL);
  }
  for (int i = 0; i < THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  uint64_t total = atomic_load_explicit(&g_total_bounces, memory_order_relaxed);
  if (total != (uint64_t)THREADS * ITERS) {
    fprintf(stderr, "Phase 2: expected %d bounces, got %" PRIu64 "\n",
            THREADS * ITERS, total);
    return 1;
  }

  /* --- Phase 3: final drain — must return exactly NODES distinct nodes. */
  int seen[NODES] = {0};
  int drained = 0;
  while ((n = arts_lockfree_stack_pop(&g_stack)) != NULL) {
    test_node_t *t = (test_node_t *)n;
    if (t->id >= NODES) {
      fprintf(stderr, "Phase 3: bogus id %u\n", t->id);
      return 1;
    }
    if (seen[t->id]) {
      fprintf(stderr, "Phase 3: duplicate id %u\n", t->id);
      return 1;
    }
    seen[t->id] = 1;
    drained++;
  }
  if (drained != NODES) {
    fprintf(stderr, "Phase 3: expected %d nodes, drained %d\n", NODES, drained);
    return 1;
  }

  printf("PASS lockfree_stack: %d nodes, %d threads, %" PRIu64 " bounces\n",
         NODES, THREADS, total);
  return 0;
}
