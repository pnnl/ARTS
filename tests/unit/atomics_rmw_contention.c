/* SPDX-License-Identifier: Apache-2.0
 *
 * T242 — contention correctness of arts_atomic_fetch_add_u64 (and the 32-bit
 * arts_atomic_fetch_add) in libs/src/core/utils/atomics.c.
 *
 * Property under test: N threads each perform M fetch_add(+1) on one shared
 * counter.  Two invariants must hold:
 *   (1) Final value == N*M  (no lost updates — full-barrier RMW).
 *   (2) The OLD values returned across all N*M operations form a *gap-free
 *       partition* of [0, N*M):  each integer in that range is returned by
 *       exactly one operation.  This is the strong linearizability witness for
 *       fetch_add — it proves both that each op returned the genuine pre-value
 *       (the convention) AND that no two ops observed the same pre-value
 *       (atomicity).  A non-atomic increment would produce duplicate or
 *       missing tickets even when the final total happened to look right.
 *
 * Each thread records the tickets it drew into a disjoint slice of a shared
 * witness array; main verifies every slot in [0, N*M) was hit exactly once.
 *
 * A start-gate (spin until released) maximises the overlap window.
 *
 * Pure-unit: links only against atomics.c + libc.  Run repeatedly under TSan.
 */

#include "arts/utils/atomics.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS 12
#define ITERS 50000
#define TOTAL ((uint64_t)THREADS * ITERS)

static volatile uint64_t g_counter;       /* the contended u64 */
static volatile unsigned int g_counter32; /* the contended u32 */
static atomic_int g_gate;                 /* start gate */

/* Each ticket index records how many ops drew it (must be exactly 1). */
static uint8_t *g_ticket_hits;   /* size TOTAL */
static uint8_t *g_ticket_hits32; /* size TOTAL */

typedef struct {
  int tid;
} arg_t;

static void *worker(void *p) {
  arg_t *a = (arg_t *)p;
  (void)a;
  while (atomic_load_explicit(&g_gate, memory_order_acquire) == 0) {
    /* spin until released */
  }
  for (int i = 0; i < ITERS; i++) {
    uint64_t old = arts_atomic_fetch_add_u64(&g_counter, 1);
    /* old must be a valid ticket in range; record the hit. */
    if (old < TOTAL) {
      /* relaxed atomic increment of the per-ticket hit count. */
      __atomic_fetch_add(&g_ticket_hits[old], 1, __ATOMIC_RELAXED);
    } else {
      /* out of range == lost-update / over-count: flag with a sentinel. */
      g_ticket_hits[0] = 0xFF;
    }

    unsigned int old32 = arts_atomic_fetch_add(&g_counter32, 1);
    if (old32 < TOTAL) {
      __atomic_fetch_add(&g_ticket_hits32[old32], 1, __ATOMIC_RELAXED);
    } else {
      g_ticket_hits32[0] = 0xFF;
    }
  }
  return NULL;
}

int main(void) {
  g_ticket_hits = (uint8_t *)calloc(TOTAL, 1);
  g_ticket_hits32 = (uint8_t *)calloc(TOTAL, 1);
  if (!g_ticket_hits || !g_ticket_hits32) {
    (void)fprintf(stderr, "FAIL atomics_rmw_contention: calloc\n");
    return 1;
  }
  g_counter = 0;
  g_counter32 = 0;
  atomic_init(&g_gate, 0);

  pthread_t th[THREADS];
  arg_t args[THREADS];
  for (int i = 0; i < THREADS; i++) {
    args[i].tid = i;
    if (pthread_create(&th[i], NULL, worker, &args[i]) != 0) {
      (void)fprintf(stderr, "FAIL atomics_rmw_contention: pthread_create\n");
      return 1;
    }
  }
  atomic_store_explicit(&g_gate, 1, memory_order_release);
  for (int i = 0; i < THREADS; i++) {
    pthread_join(th[i], NULL);
  }

  int rc = 0;

  /* (1) Final totals == N*M. */
  if (g_counter != TOTAL) {
    (void)fprintf(stderr,
                  "FAIL atomics_rmw_contention: u64 final %" PRIu64
                  " != %" PRIu64 " (lost updates)\n",
                  g_counter, TOTAL);
    rc = 1;
  }
  if ((uint64_t)g_counter32 != TOTAL) {
    (void)fprintf(stderr,
                  "FAIL atomics_rmw_contention: u32 final %u != %" PRIu64 "\n",
                  g_counter32, TOTAL);
    rc = 1;
  }

  /* (2) Every ticket in [0, TOTAL) was returned exactly once. */
  uint64_t bad64 = 0, bad32 = 0;
  for (uint64_t t = 0; t < TOTAL; t++) {
    if (g_ticket_hits[t] != 1) {
      bad64++;
    }
    if (g_ticket_hits32[t] != 1) {
      bad32++;
    }
  }
  if (bad64) {
    (void)fprintf(stderr,
                  "FAIL atomics_rmw_contention: u64 %" PRIu64
                  " tickets not drawn exactly once (dup/missing pre-values)\n",
                  bad64);
    rc = 1;
  }
  if (bad32) {
    (void)fprintf(stderr,
                  "FAIL atomics_rmw_contention: u32 %" PRIu64
                  " tickets not drawn exactly once\n",
                  bad32);
    rc = 1;
  }

  free(g_ticket_hits);
  free(g_ticket_hits32);

  if (rc) {
    return 1;
  }
  printf("PASS atomics_rmw_contention: %d threads x %d = %" PRIu64
         " ops, gap-free ticket partition (u64+u32)\n",
         THREADS, ITERS, TOTAL);
  return 0;
}
