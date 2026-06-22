/* SPDX-License-Identifier: Apache-2.0
 *
 * T009 — arts_mpsc_drain_remaining (mpsc.h) teardown helper.
 *
 * Census 29.md §5 GAP: drain_remaining (the single-consumer object-teardown
 * helper) was entirely untested.  Contract: on a QUIESCENT queue (all
 * producers stopped) a single drain_remaining returns a NULL-terminated chain
 * of EVERY linked node, re-threading each popped node's `next` to the
 * following popped node, and the embedded stub is NEVER returned.
 *
 * Scenario:
 *   - Concurrently push K nodes from P producers (so the internal mid-link
 *     windows really occur), join all producers (queue now quiescent), then
 *     call arts_mpsc_drain_remaining ONCE.
 *   - Assert: returned chain length == K (no loss), every pushed node appears
 *     exactly once (no dup), the chain is NULL-terminated, and the stub
 *     (&q.stub) never appears in the chain.
 *
 * Also a second deterministic single-thread case (push 3, drain) so the
 * chain-rethreading + stub-exclusion is pinned without any race noise.
 */

#include "arts/utils/mpsc.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PRODUCERS 6
#define PER_PRODUCER 100000
#define K ((size_t)PRODUCERS * PER_PRODUCER)

typedef struct {
  arts_lf_link_t link; /* first member */
  uint32_t id;
} node_t;

static arts_mpsc_t g_q;
static node_t *g_nodes;
static atomic_int g_start;

static void *producer(void *arg) {
  uint32_t base = (uint32_t)(uintptr_t)arg * PER_PRODUCER;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (uint32_t i = 0; i < PER_PRODUCER; i++) {
    node_t *n = &g_nodes[base + i];
    n->id = base + i;
    arts_mpsc_push(&g_q, &n->link);
  }
  return NULL;
}

static int deterministic_case(void) {
  arts_mpsc_t q;
  arts_mpsc_init(&q);
  node_t a = {0}, b = {0}, c = {0};
  a.id = 1;
  b.id = 2;
  c.id = 3;
  arts_mpsc_push(&q, &a.link);
  arts_mpsc_push(&q, &b.link);
  arts_mpsc_push(&q, &c.link);

  arts_lf_link_t *chain = arts_mpsc_drain_remaining(&q);
  uint32_t order[3];
  int n = 0;
  for (arts_lf_link_t *l = chain; l;) {
    if (l == &q.stub) {
      (void)fprintf(stderr,
                    "FAIL mpsc_drain_remaining: stub returned in chain\n");
      return 1;
    }
    if (n >= 3) {
      (void)fprintf(stderr, "FAIL mpsc_drain_remaining: chain longer than 3\n");
      return 1;
    }
    order[n++] = ((node_t *)l)->id;
    l = atomic_load_explicit(&l->next, memory_order_relaxed);
  }
  if (n != 3) {
    (void)fprintf(stderr, "FAIL mpsc_drain_remaining: got %d expected 3\n", n);
    return 1;
  }
  /* Vyukov is FIFO: drain order must be push order 1,2,3. */
  if (order[0] != 1 || order[1] != 2 || order[2] != 3) {
    (void)fprintf(stderr,
                  "FAIL mpsc_drain_remaining: order %u,%u,%u (want 1,2,3)\n",
                  order[0], order[1], order[2]);
    return 1;
  }
  return 0;
}

int main(void) {
  if (deterministic_case() != 0) {
    return 1;
  }

  arts_mpsc_init(&g_q);
  atomic_init(&g_start, 0);
  g_nodes = (node_t *)calloc(K, sizeof(node_t));
  if (!g_nodes) {
    return 1;
  }

  pthread_t prod[PRODUCERS];
  for (int i = 0; i < PRODUCERS; i++) {
    pthread_create(&prod[i], NULL, producer, (void *)(uintptr_t)i);
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < PRODUCERS; i++) {
    pthread_join(prod[i], NULL); /* queue is now quiescent */
  }

  /* Single drain of a quiescent queue must return ALL K nodes. */
  arts_lf_link_t *chain = arts_mpsc_drain_remaining(&g_q);
  uint8_t *seen = (uint8_t *)calloc(K, 1);
  if (!seen) {
    return 1;
  }
  size_t count = 0;
  for (arts_lf_link_t *l = chain; l;) {
    if (l == &g_q.stub) {
      (void)fprintf(stderr,
                    "FAIL mpsc_drain_remaining: stub in concurrent chain\n");
      return 1;
    }
    node_t *nd = (node_t *)l;
    if (nd->id >= K) {
      (void)fprintf(stderr, "FAIL mpsc_drain_remaining: bogus id %u\n", nd->id);
      return 1;
    }
    if (seen[nd->id]) {
      (void)fprintf(stderr, "FAIL mpsc_drain_remaining: dup id %u\n", nd->id);
      return 1;
    }
    seen[nd->id] = 1;
    count++;
    l = atomic_load_explicit(&l->next, memory_order_relaxed);
  }
  if (count != K) {
    (void)fprintf(stderr,
                  "FAIL mpsc_drain_remaining: drained %zu of %zu (single drain "
                  "of quiescent queue must get all)\n",
                  count, K);
    return 1;
  }
  /* A second drain must yield nothing. */
  if (arts_mpsc_drain_remaining(&g_q) != NULL) {
    (void)fprintf(stderr,
                  "FAIL mpsc_drain_remaining: second drain non-empty\n");
    return 1;
  }

  free(seen);
  free(g_nodes);
  printf("PASS mpsc_drain_remaining: %zu nodes drained in one quiescent pass, "
         "stub excluded, FIFO order\n",
         K);
  return 0;
}
