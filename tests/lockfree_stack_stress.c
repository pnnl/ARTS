/* SPDX-License-Identifier: Apache-2.0
 *
 * Stress test for arts_lf_stack_t (8-byte head CAS Treiber LIFO)
 * defined in arts/utils/lockfree_lifo.h.
 *
 * Layout:
 *   - 8 producer threads, each pushes 1M nodes carrying sequence numbers
 *     (producer_id * 1M + iter).
 *   - 1 consumer thread invokes reverse_drain() in a loop until 8M nodes
 *     have been observed.
 *
 * Asserts:
 *   1. Every sequence number in [0, 8M) is observed exactly once
 *      (no losses, no duplicates).
 *   2. Per-producer FIFO ordering is preserved by reverse_drain (within
 *      the chain returned by a single drain, each producer's seq numbers
 *      are monotonically increasing — across drains as well).
 *   3. Every node is freed before exit (ASan: 0 leaks).
 *
 * The "node never re-enters the same stack" invariant is upheld trivially
 * here: the consumer frees each node after observing it.
 */

#include "arts/utils/lockfree_lifo.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_PRODUCERS 8
#define NODES_PER_PRODUCER (1u << 20) /* 1M */
#define TOTAL_NODES ((uint64_t)NUM_PRODUCERS * NODES_PER_PRODUCER)

typedef struct stress_node_s {
  arts_lf_link_t link; /* MUST be first */
  uint32_t producer_id;
  uint32_t seq_in_producer;
} stress_node_t;

static arts_lf_stack_t g_stack;

/* Producer barrier: wait until consumer is ready before flooding. */
static _Atomic int g_start_flag;

/* Consumer counts. */
static _Atomic uint64_t g_consumed;

static void *producer_thread(void *arg) {
  uint32_t pid = (uint32_t)(uintptr_t)arg;
  /* Wait for go-signal so all producers ramp up together. */
  while (atomic_load_explicit(&g_start_flag, memory_order_acquire) == 0) {
    /* spin */
  }
  for (uint32_t i = 0; i < NODES_PER_PRODUCER; i++) {
    stress_node_t *n = (stress_node_t *)malloc(sizeof(*n));
    if (!n) {
      (void)fprintf(stderr, "malloc failed\n");
      abort();
    }
    atomic_init(&n->link.next, NULL);
    n->producer_id = pid;
    n->seq_in_producer = i;
    arts_lf_stack_push(&g_stack, &n->link);
  }
  return NULL;
}

int main(void) {
  arts_lf_stack_init(&g_stack);
  atomic_init(&g_start_flag, 0);
  atomic_init(&g_consumed, 0);

  pthread_t producers[NUM_PRODUCERS];
  for (uint32_t p = 0; p < NUM_PRODUCERS; p++) {
    if (pthread_create(&producers[p], NULL, producer_thread,
                       (void *)(uintptr_t)p) != 0) {
      (void)fprintf(stderr, "pthread_create failed for producer %u\n", p);
      return 1;
    }
  }

  /* Per-producer "next expected seq" counter for FIFO-order verification.
   * We accumulate observations from concurrent reverse_drain calls; as
   * long as each chain we drain is in push-order (FIFO per producer due
   * to the program-order of pushes within a single producer thread),
   * across-drain ordering is also maintained. */
  uint32_t *expected_next = calloc(NUM_PRODUCERS, sizeof(uint32_t));
  /* Bit-vector for "seen" — TOTAL_NODES bits == 1MB / 8M / 1MB. */
  size_t bv_words = (size_t)((TOTAL_NODES + 63) / 64);
  uint64_t *seen = calloc(bv_words, sizeof(uint64_t));
  if (!expected_next || !seen) {
    (void)fprintf(stderr, "calloc failed\n");
    return 1;
  }

  /* Release producers. */
  atomic_store_explicit(&g_start_flag, 1, memory_order_release);

  /* Consumer loop runs in this (main) thread.  We loop draining until we
   * have observed all TOTAL_NODES nodes.  This is the single-consumer
   * point — only this thread ever calls drain. */
  uint64_t consumed = 0;
  uint64_t empty_drains = 0;
  while (consumed < TOTAL_NODES) {
    arts_lf_link_t *chain = arts_lf_stack_reverse_drain(&g_stack);
    if (!chain) {
      empty_drains++;
      /* Producers may not have started yet, or they're between pushes.
       * Yield-spin. */
      continue;
    }
    while (chain) {
      stress_node_t *n = (stress_node_t *)chain;
      arts_lf_link_t *next =
          atomic_load_explicit(&chain->next, memory_order_relaxed);

      if (n->producer_id >= NUM_PRODUCERS) {
        (void)fprintf(stderr, "Bogus producer_id=%u\n", n->producer_id);
        return 1;
      }
      if (n->seq_in_producer >= NODES_PER_PRODUCER) {
        (void)fprintf(stderr, "Bogus seq_in_producer=%u (producer %u)\n",
                      n->seq_in_producer, n->producer_id);
        return 1;
      }

      /* Per-producer FIFO check: this seq must be exactly the next
       * expected one for this producer (since all the producer's pushes
       * are program-order inside the producer thread, reverse_drain
       * preserves FIFO across drains). */
      if (n->seq_in_producer != expected_next[n->producer_id]) {
        (void)fprintf(
            stderr, "FIFO violation: producer=%u expected seq=%u got seq=%u\n",
            n->producer_id, expected_next[n->producer_id], n->seq_in_producer);
        return 1;
      }
      expected_next[n->producer_id]++;

      /* Uniqueness check via bit-vector. */
      uint64_t flat =
          ((uint64_t)n->producer_id * NODES_PER_PRODUCER) + n->seq_in_producer;
      uint64_t word = flat / 64;
      uint64_t bit = (uint64_t)1 << (flat % 64);
      if (seen[word] & bit) {
        (void)fprintf(stderr, "Duplicate: producer=%u seq=%u\n", n->producer_id,
                      n->seq_in_producer);
        return 1;
      }
      seen[word] |= bit;

      consumed++;
      free(n);
      chain = next;
    }
  }
  atomic_store_explicit(&g_consumed, consumed, memory_order_relaxed);

  for (uint32_t p = 0; p < NUM_PRODUCERS; p++) {
    pthread_join(producers[p], NULL);
  }

  /* Final sanity drain: the stack must now be empty (producers all joined,
   * consumer received TOTAL_NODES). */
  arts_lf_link_t *leftover = arts_lf_stack_reverse_drain(&g_stack);
  if (leftover) {
    (void)fprintf(stderr, "Stack not empty after producers joined\n");
    return 1;
  }

  /* Verify completeness: every bit set, every per-producer count ==
   * NODES_PER_PRODUCER. */
  for (uint32_t p = 0; p < NUM_PRODUCERS; p++) {
    if (expected_next[p] != NODES_PER_PRODUCER) {
      (void)fprintf(stderr, "Producer %u: only %u / %u nodes observed\n", p,
                    expected_next[p], NODES_PER_PRODUCER);
      return 1;
    }
  }
  for (size_t w = 0; w < bv_words; w++) {
    /* Last word may have unused trailing bits — but TOTAL_NODES is a
     * multiple of 64 (8 * 1M = 8M, divisible by 64), so all bits in all
     * words must be set. */
    if (seen[w] != ~(uint64_t)0) {
      (void)fprintf(stderr,
                    "Bit-vector word %zu = 0x%" PRIx64 " (incomplete)\n", w,
                    seen[w]);
      return 1;
    }
  }

  free(seen);
  free(expected_next);

  printf("PASS lockfree_stack_stress: %u producers x %u nodes = %" PRIu64
         " total, %" PRIu64 " empty drains\n",
         NUM_PRODUCERS, NODES_PER_PRODUCER, consumed, empty_drains);
  return 0;
}
