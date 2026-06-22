/* SPDX-License-Identifier: Apache-2.0
 *
 * event_drain_simple_idempotent — C15 / T162 (pure_unit).
 *
 * Target: drain_simple_chain (event.c) reverse-drain idempotency + multi-caller
 * self-serialize.  drain_simple_chain is static, so this test reconstructs its
 * EXACT control structure over the public lock-free primitive it relies on
 * (arts/utils/lockfree_lifo.h, arts_lf_stack_reverse_drain) and pins the
 * properties drain_simple_chain depends on:
 *
 *   drain_simple_chain():
 *     for (;;) {
 *       fifo = arts_lf_stack_reverse_drain(&stack);   // atomic_exchange
 * head,NULL if (!fifo) return;                            // drain complete
 *       while (fifo) { deliver(fifo); free?; fifo = next; }
 *     }
 *
 * Properties:
 *  (1) SELF-SERIALIZE: arts_lf_stack_reverse_drain is a single
 *      atomic_exchange(head, NULL).  When D callers drain one stack
 *      concurrently, each pushed node is detached by EXACTLY ONE caller — no
 *      node delivered twice, none lost (the "only one caller per chain, others
 *      see NULL and exit" invariant).
 *  (2) IDEMPOTENT: once the stack is empty, every further drain returns NULL
 *      (a second drain is a no-op — no double-deliver).
 *  (3) OUTER-LOOP CATCHES MID-ITERATION PUSHES: a push that lands AFTER a
 *      caller's reverse_drain snapshot but while other producers are still
 *      active is caught by a subsequent drain pass.  Across the whole run every
 *      pushed node is delivered exactly once and the stack ends empty.
 *  (4) FIFO within one drained chain (reverse_drain returns oldest-push-first),
 *      which is the order drain_simple_chain delivers waiters in.
 *
 * The no-reentry invariant (a node belongs to at most one stack at a time) is
 * respected: a detached node is delivered and never re-pushed onto the stack.
 *
 * Standalone: header-only primitive, libc, pthreads — no ARTS runtime link.
 */

#include "arts/utils/lockfree_lifo.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  arts_lf_link_t link;
  uint32_t id;
} dep_t;

/* ---- Part 1: FIFO order within one reverse_drain (the delivery order) ---- */
#define ORD_K 512
static int fifo_order_check(void) {
  arts_lf_stack_t s;
  arts_lf_stack_init(&s);
  dep_t *nodes = (dep_t *)calloc(ORD_K, sizeof(dep_t));
  if (!nodes) {
    return 1;
  }
  for (int i = 0; i < ORD_K; i++) {
    nodes[i].id = (uint32_t)i;
    arts_lf_stack_push(&s, &nodes[i].link);
  }
  arts_lf_link_t *fifo = arts_lf_stack_reverse_drain(&s);
  for (int expect = 0; expect < ORD_K; expect++) {
    if (!fifo) {
      (void)fprintf(stderr, "FAIL: FIFO short at %d\n", expect);
      free(nodes);
      return 1;
    }
    if ((int)((dep_t *)fifo)->id != expect) {
      (void)fprintf(stderr, "FAIL: FIFO got %u expect %d\n",
                    ((dep_t *)fifo)->id, expect);
      free(nodes);
      return 1;
    }
    fifo = atomic_load_explicit(&fifo->next, memory_order_relaxed);
  }
  if (fifo != NULL) {
    (void)fprintf(stderr, "FAIL: FIFO trailing nodes\n");
    free(nodes);
    return 1;
  }
  /* (2) IDEMPOTENT: stack now empty -> further drains return NULL. */
  for (int i = 0; i < 5; i++) {
    if (arts_lf_stack_reverse_drain(&s) != NULL) {
      (void)fprintf(stderr, "FAIL: drain of empty stack returned non-NULL\n");
      free(nodes);
      return 1;
    }
  }
  free(nodes);
  return 0;
}

/* ---- Part 2/3: concurrent multi-caller self-serialize + outer-loop catch ----
 */
#define PRODUCERS 6
#define PER_PRODUCER 60000
#define DRAINERS 4
#define TOTAL ((size_t)PRODUCERS * PER_PRODUCER)

static arts_lf_stack_t g_s;
static dep_t *g_nodes;
static atomic_int g_start;
static _Atomic uint64_t g_prod_done;
static _Atomic uint8_t *g_delivered; /* per-node delivery tally */
static _Atomic uint64_t g_total_delivered;

static void *producer(void *arg) {
  uint32_t base = (uint32_t)(uintptr_t)arg * PER_PRODUCER;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (uint32_t i = 0; i < PER_PRODUCER; i++) {
    dep_t *n = &g_nodes[base + i];
    n->id = base + i;
    arts_lf_stack_push(&g_s, &n->link);
  }
  atomic_fetch_add_explicit(&g_prod_done, 1, memory_order_release);
  return NULL;
}

/* Mirror drain_simple_chain's body: reverse_drain then walk + "deliver" each
 * node exactly once.  A node delivered twice across drainers => self-serialize
 * violated. */
static void drain_simple_chain_like(void) {
  for (;;) {
    arts_lf_link_t *fifo = arts_lf_stack_reverse_drain(&g_s);
    if (!fifo) {
      return; /* drain complete (idempotent: empty -> NULL) */
    }
    while (fifo) {
      arts_lf_link_t *next =
          atomic_load_explicit(&fifo->next, memory_order_relaxed);
      dep_t *d = (dep_t *)fifo;
      if (d->id >= TOTAL) {
        (void)fprintf(stderr, "FAIL: corrupt node id=%u\n", d->id);
        abort();
      }
      uint8_t prev = atomic_fetch_add_explicit(&g_delivered[d->id], 1,
                                               memory_order_relaxed);
      if (prev != 0) {
        (void)fprintf(stderr,
                      "FAIL: node %u delivered twice (self-serialize "
                      "violated)\n",
                      d->id);
        abort();
      }
      atomic_fetch_add_explicit(&g_total_delivered, 1, memory_order_relaxed);
      fifo = next;
    }
  }
}

static void *drainer(void *arg) {
  (void)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  /* Loop draining until all producers finished AND the outer loop has caught
   * the last mid-iteration pushes (g_total_delivered reaches TOTAL). */
  for (;;) {
    drain_simple_chain_like();
    if (atomic_load_explicit(&g_prod_done, memory_order_acquire) == PRODUCERS &&
        atomic_load_explicit(&g_total_delivered, memory_order_relaxed) >=
            TOTAL) {
      break;
    }
  }
  return NULL;
}

static int concurrent_check(void) {
  arts_lf_stack_init(&g_s);
  atomic_init(&g_start, 0);
  atomic_init(&g_prod_done, 0);
  atomic_init(&g_total_delivered, 0);
  g_nodes = (dep_t *)calloc(TOTAL, sizeof(dep_t));
  g_delivered = (_Atomic uint8_t *)calloc(TOTAL, sizeof(_Atomic uint8_t));
  if (!g_nodes || !g_delivered) {
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
  /* Final outer-loop catch in case a drainer exited just before the last push
   * landed (mirrors the firing thread's final pass). */
  drain_simple_chain_like();

  if (atomic_load_explicit(&g_total_delivered, memory_order_relaxed) != TOTAL) {
    (void)fprintf(
        stderr, "FAIL: delivered %" PRIu64 " != %zu\n",
        atomic_load_explicit(&g_total_delivered, memory_order_relaxed), TOTAL);
    return 1;
  }
  for (size_t i = 0; i < TOTAL; i++) {
    if (atomic_load_explicit(&g_delivered[i], memory_order_relaxed) != 1) {
      (void)fprintf(
          stderr, "FAIL: node %zu delivered %u times\n", i,
          atomic_load_explicit(&g_delivered[i], memory_order_relaxed));
      return 1;
    }
  }
  /* Stack must be empty + drain idempotent at the end. */
  if (arts_lf_stack_reverse_drain(&g_s) != NULL) {
    (void)fprintf(stderr, "FAIL: stack not empty after full drain\n");
    return 1;
  }
  free(g_nodes);
  free((void *)g_delivered);
  return 0;
}

int main(void) {
  if (fifo_order_check() != 0) {
    return 1;
  }
  if (concurrent_check() != 0) {
    return 1;
  }
  printf(
      "PASS event_drain_simple_idempotent: FIFO delivery + idempotent empty "
      "drain; %zu nodes drained exactly-once by %d self-serialized callers\n",
      TOTAL, DRAINERS);
  return 0;
}
