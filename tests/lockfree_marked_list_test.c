/* SPDX-License-Identifier: Apache-2.0
 *
 * Stress test for arts_lockfree_marked_list.
 *
 * Phase 1 (sanity): single-thread push, traverse, mark, traverse-and-
 *   help-unlink, recycle.  Verifies the basic invariants.
 *
 * Phase 2 (concurrent push): T threads each push N nodes — total T*N
 *   pushes; traverse must see exactly T*N unmarked nodes (in some
 *   order); no losses, no duplicates.
 *
 * Phase 3 (concurrent push + mark): half the threads push, the other
 *   half traverse-and-mark.  Each successful mark increments a counter;
 *   final mark count == push count, and every node is eventually
 *   physically unlinked + recycled (verifiable via traversal seeing
 *   nothing).
 */

#include "arts/utils/marked_list.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct test_node_s {
  arts_marked_list_node_t link; /* MUST be first */
  uint64_t value;
  _Atomic int visited; /* set during traversal — sanity */
} test_node_t;

static arts_marked_list_t g_list;
static _Atomic uint64_t g_visit_count;
static _Atomic uint64_t g_mark_count;

static void count_visit(arts_marked_list_node_t *n, void *ctx) {
  (void)ctx;
  test_node_t *t = (test_node_t *)n;
  atomic_fetch_add_explicit(&g_visit_count, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&t->visited, 1, memory_order_relaxed);
}

static void mark_visit(arts_marked_list_node_t *n, void *ctx) {
  (void)ctx;
  /* Read payload BEFORE mark — the module may recycle the node any
   * time after mark returns true. */
  test_node_t *t = (test_node_t *)n;
  uint64_t snapshot = t->value;
  (void)snapshot;
  if (arts_marked_list_mark(n)) {
    atomic_fetch_add_explicit(&g_mark_count, 1, memory_order_relaxed);
  }
}

#define PHASE1_NODES 8

static int phase1_sanity(void) {
  arts_marked_list_init(&g_list, sizeof(test_node_t));

  /* Push 8 nodes. */
  for (int i = 0; i < PHASE1_NODES; i++) {
    arts_marked_list_node_t *n = arts_marked_list_alloc(&g_list);
    test_node_t *t = (test_node_t *)n;
    t->value = (uint64_t)(i + 100);
    atomic_init(&t->visited, 0);
    arts_marked_list_push(&g_list, n);
  }

  /* Traverse — should visit all 8. */
  atomic_init(&g_visit_count, 0);
  arts_marked_list_traverse(&g_list, count_visit, NULL);
  uint64_t v = atomic_load_explicit(&g_visit_count, memory_order_relaxed);
  if (v != PHASE1_NODES) {
    fprintf(stderr, "Phase 1 traverse: expected %d, got %" PRIu64 "\n",
            PHASE1_NODES, v);
    return 1;
  }

  /* Mark every node, then traverse to help-unlink. */
  atomic_init(&g_mark_count, 0);
  arts_marked_list_traverse(&g_list, mark_visit, NULL);
  uint64_t m = atomic_load_explicit(&g_mark_count, memory_order_relaxed);
  if (m != PHASE1_NODES) {
    fprintf(stderr, "Phase 1 mark: expected %d, got %" PRIu64 "\n",
            PHASE1_NODES, m);
    return 1;
  }

  /* Now traverse again — every node should have been physically
   * unlinked, traversal sees 0. */
  atomic_init(&g_visit_count, 0);
  arts_marked_list_traverse(&g_list, count_visit, NULL);
  v = atomic_load_explicit(&g_visit_count, memory_order_relaxed);
  if (v != 0) {
    fprintf(stderr, "Phase 1 post-unlink: expected 0, got %" PRIu64 "\n", v);
    return 1;
  }

  /* Recycle pool should have at least one node — alloc and re-use. */
  arts_marked_list_node_t *n = arts_marked_list_alloc(&g_list);
  if (n == NULL) {
    fprintf(stderr, "Phase 1 recycle: alloc returned NULL\n");
    return 1;
  }
  test_node_t *t = (test_node_t *)n;
  t->value = 999;
  arts_marked_list_push(&g_list, n);
  atomic_init(&g_visit_count, 0);
  arts_marked_list_traverse(&g_list, count_visit, NULL);
  v = atomic_load_explicit(&g_visit_count, memory_order_relaxed);
  if (v != 1) {
    fprintf(stderr, "Phase 1 recycle visit: expected 1, got %" PRIu64 "\n", v);
    return 1;
  }

  arts_marked_list_destroy(&g_list);
  printf("PASS phase1_sanity\n");
  return 0;
}

#define PHASE2_THREADS 8
#define PHASE2_PER_THREAD 1000

static void *phase2_pusher(void *arg) {
  uint64_t base = (uint64_t)(uintptr_t)arg;
  for (int i = 0; i < PHASE2_PER_THREAD; i++) {
    arts_marked_list_node_t *n = arts_marked_list_alloc(&g_list);
    test_node_t *t = (test_node_t *)n;
    t->value = base * 1000 + (uint64_t)i;
    atomic_init(&t->visited, 0);
    arts_marked_list_push(&g_list, n);
  }
  return NULL;
}

static int phase2_concurrent_push(void) {
  arts_marked_list_init(&g_list, sizeof(test_node_t));
  atomic_init(&g_visit_count, 0);

  pthread_t threads[PHASE2_THREADS];
  for (int i = 0; i < PHASE2_THREADS; i++) {
    pthread_create(&threads[i], NULL, phase2_pusher,
                   (void *)(uintptr_t)(i + 1));
  }
  for (int i = 0; i < PHASE2_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  arts_marked_list_traverse(&g_list, count_visit, NULL);
  uint64_t v = atomic_load_explicit(&g_visit_count, memory_order_relaxed);
  uint64_t expected = (uint64_t)PHASE2_THREADS * PHASE2_PER_THREAD;
  if (v != expected) {
    fprintf(stderr, "Phase 2 traverse: expected %" PRIu64 ", got %" PRIu64 "\n",
            expected, v);
    return 1;
  }

  arts_marked_list_destroy(&g_list);
  printf("PASS phase2_concurrent_push: %" PRIu64 " nodes\n", v);
  return 0;
}

/* PHASE3_THREADS = 1 pusher + (PHASE3_THREADS - 1) markers.  Single
 * pusher matches ARTS's real-world acquire pattern (a per-EDT push of
 * the waiter from one acquire context); the multi-marker case
 * exercises traversal/unlink races.
 *
 * Modest per-run sizing: this stress isolates the primitive in
 * isolation; full sustained stress comes via ARTS's existing
 * cdag-stress harness once the primitive is wired into the protocol. */
#define PHASE3_THREADS 2
#define PHASE3_PER_PUSHER 100
#define PHASE3_TRAVERSAL_ROUNDS 200

static _Atomic int g_phase3_done;

static void *phase3_pusher(void *arg) {
  uint64_t base = (uint64_t)(uintptr_t)arg;
  for (int i = 0; i < PHASE3_PER_PUSHER; i++) {
    arts_marked_list_node_t *n = arts_marked_list_alloc(&g_list);
    test_node_t *t = (test_node_t *)n;
    t->value = base * 100000 + (uint64_t)i;
    atomic_init(&t->visited, 0);
    arts_marked_list_push(&g_list, n);
  }
  return NULL;
}

static void *phase3_marker(void *arg) {
  (void)arg;
  while (!atomic_load_explicit(&g_phase3_done, memory_order_acquire)) {
    arts_marked_list_traverse(&g_list, mark_visit, NULL);
  }
  arts_marked_list_traverse(&g_list, mark_visit, NULL);
  return NULL;
}

static int phase3_concurrent_push_and_mark(void) {
  arts_marked_list_init(&g_list, sizeof(test_node_t));
  atomic_init(&g_mark_count, 0);
  atomic_init(&g_phase3_done, 0);

  /* 1 pusher + (THREADS-1) markers — matches the ARTS protocol's
   * usage pattern (one acquire pushes the waiter; many traversal
   * helpers may concurrently mark + unlink). */
  int pushers = 1;
  int markers = PHASE3_THREADS - pushers;
  pthread_t pusher_threads[PHASE3_THREADS / 2];
  pthread_t marker_threads[PHASE3_THREADS / 2];

  for (int i = 0; i < markers; i++) {
    pthread_create(&marker_threads[i], NULL, phase3_marker, NULL);
  }
  for (int i = 0; i < pushers; i++) {
    pthread_create(&pusher_threads[i], NULL, phase3_pusher,
                   (void *)(uintptr_t)(i + 1));
  }

  for (int i = 0; i < pushers; i++) {
    pthread_join(pusher_threads[i], NULL);
  }
  fprintf(stderr, "pushers done\n");
  /* Pushers done — give markers a moment to drain, then signal stop. */
  atomic_store_explicit(&g_phase3_done, 1, memory_order_release);
  fprintf(stderr, "done flag set\n");
  for (int i = 0; i < markers; i++) {
    pthread_join(marker_threads[i], NULL);
    fprintf(stderr, "marker %d joined\n", i);
  }

  uint64_t marks = atomic_load_explicit(&g_mark_count, memory_order_relaxed);
  uint64_t expected_pushes = (uint64_t)pushers * PHASE3_PER_PUSHER;
  if (marks != expected_pushes) {
    fprintf(stderr, "Phase 3 marks: expected %" PRIu64 ", got %" PRIu64 "\n",
            expected_pushes, marks);
    return 1;
  }

  /* All marked, traversal should help-unlink all and end with empty
   * chain. */
  atomic_init(&g_visit_count, 0);
  arts_marked_list_traverse(&g_list, count_visit, NULL);
  uint64_t v = atomic_load_explicit(&g_visit_count, memory_order_relaxed);
  if (v != 0) {
    fprintf(stderr, "Phase 3 final traverse: expected 0, got %" PRIu64 "\n", v);
    return 1;
  }

  arts_marked_list_destroy(&g_list);
  printf("PASS phase3_concurrent_push_and_mark: %" PRIu64
         " marks across %d pushers + %d markers\n",
         marks, pushers, markers);
  return 0;
}

int main(void) {
  if (phase1_sanity() != 0)
    return 1;
  if (phase2_concurrent_push() != 0)
    return 1;
  if (phase3_concurrent_push_and_mark() != 0)
    return 1;
  printf("ALL PASS\n");
  return 0;
}
