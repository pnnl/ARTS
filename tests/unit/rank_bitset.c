/* SPDX-License-Identifier: Apache-2.0
 *
 * T056 — rank_bitset set / for_each / init / destroy (B035: for_each snapshot
 * vs concurrent set; first-set-wins).  Built VAL+OWNER (the bitset is the OWNER
 * destroy fan-out roster; HOME reuses the version map, so this is self-skipped
 * under WRF_VAL and only meaningful where cached_ranks exists).
 *
 * Each word covers 64 ranks; nwords == (nranks+63)/64.  arts_rank_bitset_set
 * returns true IFF the bit was previously clear (first-time set).
 * arts_rank_bitset_for_each invokes cb(rank,ctx) once per set bit (per-word
 * acquire snapshot — the contract requires no concurrent set DURING iteration,
 * satisfied at destroy fan-out by the slot-absent invariant).
 *
 * Parts:
 *   1. init sizes: nranks 1/63/64/65/127/128/130 → nwords correct.
 *   2. Word-boundary set + for_each: set ranks {0,1,62,63,64,65,126,127} on a
 *      130-rank set; for_each visits EXACTLY those ranks, each once, no
 *      duplicates, no spurious ranks (catches off-by-one across the 63/64 and
 *      127/128 word boundaries).
 *   3. First-set-wins single-thread: set(r)->true, set(r)->false.
 *   4. OOB reject: set(rank >= nranks) -> false, never crashes, never sets.
 *   5. Concurrency: K threads all set the SAME rank → exactly ONE true return
 *      (atomic_fetch_or first-setter).  Repeated across rounds + ranks on both
 *      sides of a word boundary.
 *   6. Concurrency: K threads each set a DISTINCT rank (spanning word
 *      boundaries) → every set returns true once, for_each (after join) visits
 *      every rank exactly once.
 *
 * Standalone: links val/home.c (carries the bitset fns) with libc-backed
 * alloc shims.  Built with -DARTS_PROTOCOL_VAL=1 -DARTS_WRITE_POLICY_WB=1.
 */

#include "arts/rank_bitset.h"
#include "arts/coherence/directory.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(ARTS_PROTOCOL_WRF_VAL)
/* WRF_VAL carries no per-DB rank bit-set (no cached_ranks roster; its home arm is
 * the version map).  The bitset functions are not even compiled for WRF_VAL, so
 * this test self-skips there. */
int main(void) {
  printf("PASS rank_bitset: skipped under WRF_VAL (no rank bit-set in this "
         "protocol)\n");
  return 0;
}
#else

#define NRANKS 130
#define THREADS 8

struct visit_ctx {
  int seen[NRANKS + 8];
  int count;
  int bad;
};
static void visit_cb(unsigned int rank, void *ctx) {
  struct visit_ctx *v = (struct visit_ctx *)ctx;
  if (rank >= (unsigned int)(NRANKS + 8)) {
    v->bad = 1;
    return;
  }
  v->seen[rank]++;
  v->count++;
}

static int part1_sizes(void) {
  struct {
    unsigned int nranks, nwords;
  } cases[] = {{1, 1},   {63, 1},  {64, 1},  {65, 2},
               {127, 2}, {128, 2}, {129, 3}, {130, 3}};
  int rc = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    struct arts_rank_bitset_s r;
    arts_rank_bitset_init(&r, cases[i].nranks);
    if (r.nwords != cases[i].nwords || r.nranks != cases[i].nranks) {
      (void)fprintf(stderr, "FAIL rank_bitset: init(%u) nwords %u != %u\n",
                    cases[i].nranks, r.nwords, cases[i].nwords);
      rc = 1;
    }
    arts_rank_bitset_destroy(&r);
  }
  return rc;
}

static int part2_boundaries(void) {
  struct arts_rank_bitset_s r;
  arts_rank_bitset_init(&r, NRANKS);
  unsigned int ranks[] = {0, 1, 62, 63, 64, 65, 126, 127};
  const int n = (int)(sizeof(ranks) / sizeof(ranks[0]));
  int rc = 0;
  for (int i = 0; i < n; i++) {
    if (!arts_rank_bitset_set(&r, ranks[i])) {
      (void)fprintf(stderr, "FAIL rank_bitset: first set(%u) not true\n",
                    ranks[i]);
      rc = 1;
    }
  }
  struct visit_ctx v;
  memset(&v, 0, sizeof(v));
  arts_rank_bitset_for_each(&r, visit_cb, &v);
  if (v.bad || v.count != n) {
    (void)fprintf(
        stderr, "FAIL rank_bitset: for_each visited %d (expected %d) bad=%d\n",
        v.count, n, v.bad);
    rc = 1;
  }
  for (int i = 0; i < n; i++) {
    if (v.seen[ranks[i]] != 1) {
      (void)fprintf(stderr, "FAIL rank_bitset: rank %u visited %d times\n",
                    ranks[i], v.seen[ranks[i]]);
      rc = 1;
    }
  }
  /* No spurious ranks visited. */
  for (int i = 0; i < NRANKS; i++) {
    int expected = 0;
    for (int j = 0; j < n; j++) {
      if (ranks[j] == (unsigned int)i) {
        expected = 1;
      }
    }
    if (v.seen[i] != expected) {
      (void)fprintf(stderr, "FAIL rank_bitset: rank %d seen=%d expected=%d\n",
                    i, v.seen[i], expected);
      rc = 1;
    }
  }
  /* Idempotent set: re-set returns false. */
  for (int i = 0; i < n; i++) {
    if (arts_rank_bitset_set(&r, ranks[i])) {
      (void)fprintf(stderr, "FAIL rank_bitset: re-set(%u) should be false\n",
                    ranks[i]);
      rc = 1;
    }
  }
  arts_rank_bitset_destroy(&r);
  return rc;
}

static int part4_oob(void) {
  struct arts_rank_bitset_s r;
  arts_rank_bitset_init(&r, 64);
  int rc = 0;
  if (arts_rank_bitset_set(&r, 64)) { /* == nranks, OOB */
    (void)fprintf(stderr, "FAIL rank_bitset: set(64) on nranks=64 OOB\n");
    rc = 1;
  }
  if (arts_rank_bitset_set(&r, 1000)) {
    (void)fprintf(stderr, "FAIL rank_bitset: set(1000) OOB\n");
    rc = 1;
  }
  /* No bits set by OOB calls. */
  struct visit_ctx v;
  memset(&v, 0, sizeof(v));
  arts_rank_bitset_for_each(&r, visit_cb, &v);
  if (v.count != 0) {
    (void)fprintf(stderr, "FAIL rank_bitset: OOB set added %d bits\n", v.count);
    rc = 1;
  }
  arts_rank_bitset_destroy(&r);
  return rc;
}

/* --- Concurrency: same rank → exactly one true. --- */
static struct arts_rank_bitset_s g_r;
static _Atomic int g_start;
static _Atomic int g_true_count;
static unsigned int g_target_rank;

static void *same_setter(void *arg) {
  (void)arg;
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  if (arts_rank_bitset_set(&g_r, g_target_rank)) {
    atomic_fetch_add_explicit(&g_true_count, 1, memory_order_relaxed);
  }
  return NULL;
}

static int part5_same_rank(void) {
  unsigned int targets[] = {0, 63, 64, 127, 65}; /* both sides of boundaries */
  int rc = 0;
  for (size_t ti = 0; ti < sizeof(targets) / sizeof(targets[0]); ti++) {
    for (int round = 0; round < 50; round++) {
      arts_rank_bitset_init(&g_r, NRANKS);
      g_target_rank = targets[ti];
      atomic_store_explicit(&g_start, 0, memory_order_release);
      atomic_store_explicit(&g_true_count, 0, memory_order_release);
      pthread_t th[THREADS];
      for (int i = 0; i < THREADS; i++) {
        pthread_create(&th[i], NULL, same_setter, NULL);
      }
      atomic_store_explicit(&g_start, 1, memory_order_release);
      for (int i = 0; i < THREADS; i++) {
        pthread_join(th[i], NULL);
      }
      int t = atomic_load_explicit(&g_true_count, memory_order_acquire);
      if (t != 1) {
        (void)fprintf(stderr,
                      "FAIL rank_bitset: same-rank %u exactly-once got %d "
                      "(round %d)\n",
                      targets[ti], t, round);
        rc = 1;
      }
      arts_rank_bitset_destroy(&g_r);
      if (rc) {
        return rc;
      }
    }
  }
  return rc;
}

/* --- Concurrency: distinct ranks → all set, for_each visits each once. --- */
static _Atomic int g_distinct_true;
static void *distinct_setter(void *arg) {
  unsigned int rank = (unsigned int)(uintptr_t)arg;
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  if (arts_rank_bitset_set(&g_r, rank)) {
    atomic_fetch_add_explicit(&g_distinct_true, 1, memory_order_relaxed);
  }
  return NULL;
}

static int part6_distinct(void) {
  /* Spread ranks across all 3 words of a 130-rank set. */
  unsigned int ranks[THREADS] = {3, 60, 63, 64, 70, 120, 127, 129};
  int rc = 0;
  for (int round = 0; round < 50; round++) {
    arts_rank_bitset_init(&g_r, NRANKS);
    atomic_store_explicit(&g_start, 0, memory_order_release);
    atomic_store_explicit(&g_distinct_true, 0, memory_order_release);
    pthread_t th[THREADS];
    for (int i = 0; i < THREADS; i++) {
      pthread_create(&th[i], NULL, distinct_setter,
                     (void *)(uintptr_t)ranks[i]);
    }
    atomic_store_explicit(&g_start, 1, memory_order_release);
    for (int i = 0; i < THREADS; i++) {
      pthread_join(th[i], NULL);
    }
    int t = atomic_load_explicit(&g_distinct_true, memory_order_acquire);
    if (t != THREADS) {
      (void)fprintf(stderr,
                    "FAIL rank_bitset: distinct-rank true count %d != %d\n", t,
                    THREADS);
      rc = 1;
    }
    struct visit_ctx v;
    memset(&v, 0, sizeof(v));
    arts_rank_bitset_for_each(&g_r, visit_cb, &v);
    if (v.count != THREADS || v.bad) {
      (void)fprintf(stderr, "FAIL rank_bitset: distinct for_each count %d\n",
                    v.count);
      rc = 1;
    }
    for (int i = 0; i < THREADS; i++) {
      if (v.seen[ranks[i]] != 1) {
        rc = 1;
      }
    }
    arts_rank_bitset_destroy(&g_r);
    if (rc) {
      return rc;
    }
  }
  return rc;
}

int main(void) {
  if (part1_sizes() != 0) {
    return 1;
  }
  if (part2_boundaries() != 0) {
    return 1;
  }
  if (part4_oob() != 0) {
    return 1;
  }
  if (part5_same_rank() != 0) {
    return 1;
  }
  if (part6_distinct() != 0) {
    return 1;
  }
  printf("PASS rank_bitset: sizes, word-boundary for_each single-visit, "
         "first-set-wins, OOB reject, %d-thread same-rank exactly-once + "
         "distinct-rank all-set\n",
         THREADS);
  return 0;
}

/* ── libc-backed alloc shims so the test links standalone against just the
 * protocol's home.c (VAL).  When linked against the full ARTS library
 * (the EXCL path, which pulls in a monolithic lock/home.c), libarts already
 * provides these, so the shims are compiled out to avoid multiple-definition.
 */
#ifdef ARTS_UNIT_STANDALONE_SHIMS
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
void *arts_malloc(size_t size) { return malloc(size); }
#endif

#endif /* !ARTS_PROTOCOL_WRF_VAL */
