/* SPDX-License-Identifier: Apache-2.0
 *
 * T055 — rank_u64_map advance / set / get contract + concurrency (B033: set is
 * non-monotonic and can clobber advance; B034: create/get/set no NULL/OOB
 * guard).
 *
 * arts_rank_u64_map_advance(m, rank, v) = monotonic-max via CAS-loop; returns
 * true IFF the slot strictly advanced (v > old).  arts_rank_u64_map_set is
 * NON-monotonic: it overwrites unconditionally and can move a slot backward.
 *
 * Parts:
 *   1. Single-thread contract:
 *      - advance from 0: advance(5)->true, advance(5)->false (equal),
 *        advance(4)->false (lower), advance(6)->true.  Slot ends 6.
 *      - set CAN go backward: set(2) after slot==6 leaves slot==2 (documents
 *        B033 — set is non-monotonic).  advance never retreats.
 *      - rank >= nranks: advance/set are no-ops (false / silent); get OOB → 0.
 *   2. Concurrency (stress): K threads each advance the SAME slot with strictly
 *      increasing DISTINCT proposals.  Invariants after join:
 *      - final slot == global max proposal,
 *      - the number of advance()==true returns is >= 1 and each true caller's
 *        proposed value was, at the moment it won, the new max — so the set of
 *        winning values is strictly increasing and the LAST winner equals the
 *        global max.  No lost update: the max always lands.
 *      - exactly-once for a single value: among many threads proposing the SAME
 *        value V on a fresh slot, exactly ONE gets true.
 *   3. create(0) yields a usable empty map; destroy(NULL) is a no-op.
 *
 * Standalone: links rank_u64_map.c (uses libc malloc directly).
 */

#include "arts/coherence/directory.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NRANKS 4
#define ADVANCERS 8
#define ADV_ITERS 5000

static struct arts_rank_to_u64_map_s *g_map;
static _Atomic int g_start;
static _Atomic uint64_t g_true_count;     /* total advance()==true returns */
static _Atomic uint64_t g_max_true_value; /* highest value that won a CAS */

static int part1_contract(void) {
  struct arts_rank_to_u64_map_s *m = arts_rank_u64_map_create(NRANKS);
  int rc = 0;

  if (!arts_rank_u64_map_advance(m, 0, 5)) {
    (void)fprintf(stderr, "FAIL advance(5) from 0 should be true\n");
    rc = 1;
  }
  if (arts_rank_u64_map_advance(m, 0, 5)) {
    (void)fprintf(stderr, "FAIL advance(5)==5 should be false (equal)\n");
    rc = 1;
  }
  if (arts_rank_u64_map_advance(m, 0, 4)) {
    (void)fprintf(stderr, "FAIL advance(4)<5 should be false\n");
    rc = 1;
  }
  if (!arts_rank_u64_map_advance(m, 0, 6)) {
    (void)fprintf(stderr, "FAIL advance(6)>5 should be true\n");
    rc = 1;
  }
  if (arts_rank_u64_map_get(m, 0) != 6) {
    (void)fprintf(stderr, "FAIL slot should be 6\n");
    rc = 1;
  }
  /* set CAN go backward (B033). */
  arts_rank_u64_map_set(m, 0, 2);
  if (arts_rank_u64_map_get(m, 0) != 2) {
    (void)fprintf(stderr, "FAIL set(2) should overwrite backward to 2\n");
    rc = 1;
  }
  /* advance still monotone from the new (lower) value. */
  if (arts_rank_u64_map_advance(m, 0, 1)) {
    (void)fprintf(stderr, "FAIL advance(1)<2 should be false\n");
    rc = 1;
  }
  if (!arts_rank_u64_map_advance(m, 0, 3)) {
    (void)fprintf(stderr, "FAIL advance(3)>2 should be true\n");
    rc = 1;
  }

  /* OOB rank no-ops. */
  if (arts_rank_u64_map_advance(m, NRANKS + 1, 99)) {
    (void)fprintf(stderr, "FAIL advance on OOB rank should be false\n");
    rc = 1;
  }
  arts_rank_u64_map_set(m, NRANKS + 1, 99); /* silent no-op */
  if (arts_rank_u64_map_get(m, NRANKS + 1) != 0) {
    (void)fprintf(stderr, "FAIL get on OOB rank should be 0\n");
    rc = 1;
  }

  arts_rank_u64_map_destroy(m);
  return rc;
}

static void *advancer(void *arg) {
  uint64_t base = (uint64_t)(uintptr_t)arg;
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  /* Strictly increasing distinct proposals: base, base+ADVANCERS, ... */
  for (int i = 0; i < ADV_ITERS; i++) {
    uint64_t v = base + (uint64_t)i * ADVANCERS;
    if (arts_rank_u64_map_advance(g_map, 0, v)) {
      atomic_fetch_add_explicit(&g_true_count, 1, memory_order_relaxed);
      /* Track the highest value that ever won a CAS. */
      uint64_t cur =
          atomic_load_explicit(&g_max_true_value, memory_order_relaxed);
      while (v > cur) {
        if (atomic_compare_exchange_weak_explicit(&g_max_true_value, &cur, v,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
          break;
        }
      }
    }
  }
  return NULL;
}

static int part2_stress(void) {
  g_map = arts_rank_u64_map_create(NRANKS);
  atomic_store_explicit(&g_start, 0, memory_order_release);
  atomic_store_explicit(&g_true_count, 0, memory_order_release);
  atomic_store_explicit(&g_max_true_value, 0, memory_order_release);

  pthread_t th[ADVANCERS];
  for (int i = 0; i < ADVANCERS; i++) {
    /* base in [1..ADVANCERS] so proposals are distinct across threads. */
    pthread_create(&th[i], NULL, advancer, (void *)(uintptr_t)(i + 1));
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < ADVANCERS; i++) {
    pthread_join(th[i], NULL);
  }

  /* Global max proposal = (ADVANCERS) + (ADV_ITERS-1)*ADVANCERS. */
  uint64_t global_max =
      (uint64_t)ADVANCERS + (uint64_t)(ADV_ITERS - 1) * ADVANCERS;
  uint64_t slot = arts_rank_u64_map_get(g_map, 0);
  int rc = 0;
  if (slot != global_max) {
    (void)fprintf(stderr,
                  "FAIL rank_u64_map_advance: final slot %" PRIu64
                  " != global max %" PRIu64 " (lost update)\n",
                  slot, global_max);
    rc = 1;
  }
  uint64_t tc = atomic_load_explicit(&g_true_count, memory_order_acquire);
  uint64_t maxtv =
      atomic_load_explicit(&g_max_true_value, memory_order_acquire);
  if (tc < 1) {
    (void)fprintf(stderr, "FAIL rank_u64_map_advance: no advance won\n");
    rc = 1;
  }
  if (maxtv != global_max) {
    (void)fprintf(stderr,
                  "FAIL rank_u64_map_advance: max winning value %" PRIu64
                  " != global max %" PRIu64 "\n",
                  maxtv, global_max);
    rc = 1;
  }
  arts_rank_u64_map_destroy(g_map);
  g_map = NULL;
  return rc;
}

/* Exactly-once: many threads advance the SAME value on a fresh slot. */
static _Atomic uint64_t g_same_true;
static struct arts_rank_to_u64_map_s *g_same_map;
static _Atomic int g_same_start;
static void *same_advancer(void *arg) {
  (void)arg;
  while (!atomic_load_explicit(&g_same_start, memory_order_acquire)) {
  }
  if (arts_rank_u64_map_advance(g_same_map, 0, 42)) {
    atomic_fetch_add_explicit(&g_same_true, 1, memory_order_relaxed);
  }
  return NULL;
}

static int part2b_exactly_once(void) {
  int rc = 0;
  for (int round = 0; round < 200; round++) {
    g_same_map = arts_rank_u64_map_create(NRANKS);
    atomic_store_explicit(&g_same_true, 0, memory_order_release);
    atomic_store_explicit(&g_same_start, 0, memory_order_release);
    pthread_t th[ADVANCERS];
    for (int i = 0; i < ADVANCERS; i++) {
      pthread_create(&th[i], NULL, same_advancer, NULL);
    }
    atomic_store_explicit(&g_same_start, 1, memory_order_release);
    for (int i = 0; i < ADVANCERS; i++) {
      pthread_join(th[i], NULL);
    }
    uint64_t t = atomic_load_explicit(&g_same_true, memory_order_acquire);
    if (t != 1) {
      (void)fprintf(stderr,
                    "FAIL rank_u64_map_advance: same-value exactly-once got "
                    "%" PRIu64 " (round %d)\n",
                    t, round);
      rc = 1;
    }
    if (arts_rank_u64_map_get(g_same_map, 0) != 42) {
      rc = 1;
    }
    arts_rank_u64_map_destroy(g_same_map);
    if (rc) {
      break;
    }
  }
  return rc;
}

static int part3_edge(void) {
  /* create(0): usable empty map; advance/get OOB on it are no-ops. */
  struct arts_rank_to_u64_map_s *m0 = arts_rank_u64_map_create(0);
  int rc = 0;
  if (arts_rank_u64_map_advance(m0, 0, 1)) {
    (void)fprintf(stderr, "FAIL advance on nranks=0 map should be false\n");
    rc = 1;
  }
  if (arts_rank_u64_map_get(m0, 0) != 0) {
    rc = 1;
  }
  arts_rank_u64_map_destroy(m0);
  /* destroy(NULL) is a no-op. */
  arts_rank_u64_map_destroy(NULL);
  return rc;
}

int main(void) {
  if (part1_contract() != 0) {
    return 1;
  }
  if (part2_stress() != 0) {
    return 1;
  }
  if (part2b_exactly_once() != 0) {
    return 1;
  }
  if (part3_edge() != 0) {
    return 1;
  }
  printf("PASS rank_u64_map_advance: monotone contract, %d-thread max-converge "
         "no-lost-update, same-value exactly-once, set-backward, OOB no-op, "
         "create(0)/destroy(NULL)\n",
         ADVANCERS);
  return 0;
}
