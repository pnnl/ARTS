/* SPDX-License-Identifier: Apache-2.0
 *
 * T002 — arts_lf_stack_pop_one (lockfree_lifo.h) in isolation.
 *
 * arts_lf_stack_pop_one is a CAS-loop pop on the 8-byte-head Treiber LIFO that
 * has NO tag/counter; its ABA-freedom rests entirely on the header's
 * documented "no re-entry into the same stack" invariant (a popped node must
 * NEVER be pushed back onto the SAME stack — it must be freed or pushed to a
 * DIFFERENT stack).  The existing stress only exercises reverse_drain, so
 * pop_one (a distinct function with its own acquire/release semantics) was
 * entirely untested (census 29.md §2 GAP).
 *
 * Design that RESPECTS the invariant while still racing pop_one hard:
 *   - SRC stack is pre-seeded with N distinct nodes.
 *   - C consumer threads each loop: pop_one(SRC); if non-NULL, push the node
 *     to a SECOND stack (SINK, a different stack) and count it.  Nodes thus
 *     migrate SRC -> SINK exactly once; they never return to SRC, so the
 *     no-reentry invariant holds and any pop_one ABA defect would surface as a
 *     lost node, a duplicated node, or a corrupted chain.
 *   - A separate publish/consume happens-before check: each node carries a
 *     payload written before its (one and only) push to SRC; every consumer
 *     re-reads it after pop_one — pop_one's acquire load of head must observe
 *     the producer-side release publication.
 *
 * Termination: producers are done up-front (all N seeded before consumers
 * start), so consumers stop once the global migrated count reaches N.
 * Assertion: SINK ends with exactly N distinct, uncorrupted nodes.
 */

#include "arts/utils/lockfree_lifo.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NODES 200000
#define CONSUMERS 8
#define MAGIC 0x5A5Au

typedef struct {
  arts_lf_link_t link; /* first member */
  uint32_t id;
  uint32_t magic; /* published before push; checked after pop */
} node_t;

static arts_lf_stack_t g_src;
static arts_lf_stack_t g_sink;
static node_t *g_nodes;
static _Atomic uint64_t g_migrated;
static atomic_int g_start;

static void *consumer(void *arg) {
  (void)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
    /* start gate */
  }
  for (;;) {
    if (atomic_load_explicit(&g_migrated, memory_order_relaxed) >=
        (uint64_t)NODES) {
      break;
    }
    arts_lf_link_t *l = arts_lf_stack_pop_one(&g_src);
    if (!l) {
      continue; /* lost the CAS or source momentarily empty */
    }
    node_t *n = (node_t *)l;
    /* happens-before: producer published magic BEFORE push; pop_one's
     * acquire on head must make it visible. */
    if (n->magic != MAGIC || n->id >= NODES) {
      (void)fprintf(stderr,
                    "FAIL lf_lifo_pop_one: torn payload magic=%x id=%u\n",
                    n->magic, n->id);
      abort();
    }
    arts_lf_stack_push(&g_sink, l); /* DIFFERENT stack — invariant upheld */
    atomic_fetch_add_explicit(&g_migrated, 1, memory_order_relaxed);
  }
  return NULL;
}

int main(void) {
  arts_lf_stack_init(&g_src);
  arts_lf_stack_init(&g_sink);
  atomic_init(&g_migrated, 0);
  atomic_init(&g_start, 0);

  g_nodes = (node_t *)calloc(NODES, sizeof(node_t));
  if (!g_nodes) {
    (void)fprintf(stderr, "FAIL lf_lifo_pop_one: calloc\n");
    return 1;
  }
  for (int i = 0; i < NODES; i++) {
    g_nodes[i].id = (uint32_t)i;
    g_nodes[i].magic = MAGIC; /* publish payload before push */
    arts_lf_stack_push(&g_src, &g_nodes[i].link);
  }

  pthread_t th[CONSUMERS];
  for (int i = 0; i < CONSUMERS; i++) {
    if (pthread_create(&th[i], NULL, consumer, NULL) != 0) {
      (void)fprintf(stderr, "FAIL lf_lifo_pop_one: pthread_create %d\n", i);
      return 1;
    }
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < CONSUMERS; i++) {
    pthread_join(th[i], NULL);
  }

  /* SRC must now be empty; SINK must hold exactly NODES distinct nodes. */
  if (arts_lf_stack_pop_one(&g_src) != NULL) {
    (void)fprintf(stderr, "FAIL lf_lifo_pop_one: SRC not drained\n");
    return 1;
  }
  uint8_t *seen = (uint8_t *)calloc(NODES, 1);
  if (!seen) {
    (void)fprintf(stderr, "FAIL lf_lifo_pop_one: calloc seen\n");
    return 1;
  }
  int drained = 0;
  arts_lf_link_t *l;
  while ((l = arts_lf_stack_pop_one(&g_sink)) != NULL) {
    node_t *n = (node_t *)l;
    if (n->id >= NODES || n->magic != MAGIC) {
      (void)fprintf(stderr, "FAIL lf_lifo_pop_one: bad sink node id=%u\n",
                    n->id);
      return 1;
    }
    if (seen[n->id]) {
      (void)fprintf(stderr, "FAIL lf_lifo_pop_one: duplicate id=%u\n", n->id);
      return 1;
    }
    seen[n->id] = 1;
    drained++;
  }
  if (drained != NODES) {
    (void)fprintf(stderr, "FAIL lf_lifo_pop_one: expected %d in sink, got %d\n",
                  NODES, drained);
    return 1;
  }

  free(seen);
  free(g_nodes);
  printf("PASS lf_lifo_pop_one: %d nodes migrated SRC->SINK by %d consumers, "
         "no loss/dup/torn\n",
         NODES, CONSUMERS);
  return 0;
}
