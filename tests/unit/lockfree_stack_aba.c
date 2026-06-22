/* SPDX-License-Identifier: Apache-2.0
 *
 * T001 — arts_lockfree_stack (legacy tagged-pointer Treiber stack) ABA stress.
 *
 * Property under test: the 16-bit ABA counter in the packed top word
 * ([counter:16 | ptr:48]) must absorb node recycling without ever losing or
 * duplicating a node, even when the per-`top` CAS counter wraps every 65536
 * successful updates.  The census (29.md §1, suspected bug B122) flags that
 * ABA is only *deterministically* defended within a window of 65535
 * intervening updates; a node popped and re-pushed after EXACTLY 2^16 (mod
 * 2^16) intervening successful CASes could in principle fool a stale popper.
 *
 * This test drives an aggregate update count FAR exceeding 2^16 per node and,
 * in addition to the existing bounce stress, deliberately constructs a small
 * "stride" pattern (few nodes, very many cheap pop/push rounds) so the counter
 * sweeps through its full 16-bit range many thousands of times.  Correctness
 * criterion (the only thing that is genuinely observable from the outside):
 * after all threads finish, every original node is still present exactly once,
 * none is lost, none is duplicated, and no payload is corrupted.
 *
 * Interleaving: T worker threads, started on a release start-gate, each loop
 * ITERS times performing pop -> bump per-node + global counter -> push.  A
 * small NODES count maximizes contention on `top` (so the 16-bit counter wraps
 * fast) and maximizes the chance any latent ABA defect surfaces as a lost or
 * duplicated node.
 */

#include "arts/utils/lockfree_stack.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Small node count -> high contention -> fast 16-bit counter wrap.
 * THREADS * ITERS total pop+push rounds: 8 * 2_000_000 = 16M successful CASes
 * in aggregate, sweeping the 16-bit counter ~244 times. */
#define NODES 4
#define THREADS 8
#define ITERS 2000000

typedef struct test_node_s {
  arts_lockfree_stack_node_t link; /* MUST be first */
  uint32_t id;
  uint32_t magic; /* corruption canary */
  _Atomic uint64_t bounce_count;
} test_node_t;

static arts_lockfree_stack_t g_stack;
static test_node_t g_nodes[NODES];
static _Atomic uint64_t g_total_bounces;
static atomic_int g_start;

#define MAGIC 0xC0FFEEu

static void *worker(void *arg) {
  (void)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
    /* spin on start gate */
  }
  for (int i = 0; i < ITERS; i++) {
    arts_lockfree_stack_node_t *n;
    while ((n = arts_lockfree_stack_pop(&g_stack)) == NULL) {
      /* stack briefly empty when all threads are mid-pop */
    }
    test_node_t *t = (test_node_t *)n;
    if (t->magic != MAGIC || t->id >= NODES) {
      (void)fprintf(stderr,
                    "FAIL lockfree_stack_aba: corrupted node magic=%x id=%u\n",
                    t->magic, t->id);
      abort();
    }
    atomic_fetch_add_explicit(&t->bounce_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_total_bounces, 1, memory_order_relaxed);
    arts_lockfree_stack_push(&g_stack, n);
  }
  return NULL;
}

int main(void) {
  arts_lockfree_stack_init(&g_stack);
  atomic_init(&g_start, 0);
  atomic_init(&g_total_bounces, 0);

  for (int i = 0; i < NODES; i++) {
    g_nodes[i].id = (uint32_t)i;
    g_nodes[i].magic = MAGIC;
    atomic_init(&g_nodes[i].bounce_count, 0);
    arts_lockfree_stack_push(&g_stack, &g_nodes[i].link);
  }

  pthread_t threads[THREADS];
  for (int i = 0; i < THREADS; i++) {
    if (pthread_create(&threads[i], NULL, worker, NULL) != 0) {
      (void)fprintf(stderr, "FAIL lockfree_stack_aba: pthread_create %d\n", i);
      return 1;
    }
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  uint64_t total = atomic_load_explicit(&g_total_bounces, memory_order_relaxed);
  if (total != (uint64_t)THREADS * ITERS) {
    (void)fprintf(stderr,
                  "FAIL lockfree_stack_aba: expected %" PRIu64
                  " bounces, got %" PRIu64 "\n",
                  (uint64_t)THREADS * ITERS, total);
    return 1;
  }

  /* Final drain: exactly NODES distinct, uncorrupted nodes must return. */
  int seen[NODES] = {0};
  int drained = 0;
  arts_lockfree_stack_node_t *n;
  while ((n = arts_lockfree_stack_pop(&g_stack)) != NULL) {
    test_node_t *t = (test_node_t *)n;
    if (t->id >= NODES) {
      (void)fprintf(stderr, "FAIL lockfree_stack_aba: bogus id %u\n", t->id);
      return 1;
    }
    if (t->magic != MAGIC) {
      (void)fprintf(stderr, "FAIL lockfree_stack_aba: corrupt magic id=%u\n",
                    t->id);
      return 1;
    }
    if (seen[t->id]) {
      (void)fprintf(stderr, "FAIL lockfree_stack_aba: duplicate id %u\n",
                    t->id);
      return 1;
    }
    seen[t->id] = 1;
    drained++;
  }
  if (drained != NODES) {
    (void)fprintf(stderr,
                  "FAIL lockfree_stack_aba: expected %d nodes, drained %d\n",
                  NODES, drained);
    return 1;
  }

  printf("PASS lockfree_stack_aba: %d nodes, %d threads, %" PRIu64
         " bounces (16-bit counter swept ~%" PRIu64 "x), no loss/dup/corrupt\n",
         NODES, THREADS, total, total >> 16);
  return 0;
}
