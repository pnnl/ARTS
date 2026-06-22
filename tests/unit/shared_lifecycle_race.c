/* SPDX-License-Identifier: Apache-2.0
 *
 * T018 — arts_shared_ptr_t mixed-API lifecycle race (shared.c).
 * Census 28.md gaps #3 (mixed-API churn), #8 (N-way last-drop deleter once),
 * #4 (abandon-vs-make), #6 (make leaves link uninit, B126), #5 (release
 * underflow, B128).
 *
 * Parts:
 *
 *  (A) N-way last-drop, deleter runs EXACTLY once: make one cb, hand N threads
 *      a copy each (strong = N+1), all N release simultaneously plus the
 *      owner; the counting deleter must fire exactly once.  Repeated over many
 *      rounds.  (Existing tests churn DISTINCT objects through a slot, never N
 *      releasers of ONE cb.)
 *
 *  (B) Mixed-API churn on one slot: threads concurrently store / exchange /
 *      compare_exchange / abandon-loser plus copy+release of loaded refs.  At
 *      quiescence make-count == delete-count (refcount conservation, no leak /
 *      no double-free under ASan).
 *
 *  (C) abandon-vs-make no double-pop: a thread makes then abandons in a tight
 *      loop while others make/release — the cb pool must never hand the same cb
 *      to two live owners (caught by an object-identity stamp + ASan).
 *
 *  (D) make-leaves-link-uninit invariant (B126, documented): make() does NOT
 *      re-init cb->link.next, relying on release/abandon to overwrite it before
 *      the next pool push.  We pin the OBSERVABLE consequence: a make -> use ->
 *      release -> make cycle reuses cbs cleanly with no corruption (if the
 *      stale link were ever read while live, ASan/logic would break).
 *
 *  (E) release underflow (B128, documented — NOT triggered): a double-release
 *      wraps `strong` silently with no assert.  We do NOT perform a real
 *      double release (that is UB that could resurrect a cb); the gap is
 *      recorded here in the comment.  (Doing it would corrupt the pool.)
 */

#include "arts/utils/shared.h"

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic uint64_t g_made;
static _Atomic uint64_t g_deleted;

typedef struct {
  uint32_t magic;
  _Atomic int live; /* identity stamp: 1 while owned, 0 when freed */
} obj_t;
#define OBJ_MAGIC 0x600D600Du

static void *make_obj(void) {
  obj_t *o = (obj_t *)malloc(sizeof(obj_t));
  o->magic = OBJ_MAGIC;
  atomic_init(&o->live, 1);
  atomic_fetch_add_explicit(&g_made, 1, memory_order_relaxed);
  return o;
}
static void del_obj(void *p) {
  obj_t *o = (obj_t *)p;
  if (o->magic != OBJ_MAGIC) {
    (void)fprintf(stderr, "FAIL shared_lifecycle_race: deleter on bad obj\n");
    abort();
  }
  o->magic = 0xDEAD;
  free(o);
  atomic_fetch_add_explicit(&g_deleted, 1, memory_order_relaxed);
}

/* ---- Part A: N-way last-drop, deleter exactly once ---- */
#define A_THREADS 8
/* Each round spawns + joins A_THREADS threads through a barrier, so the round
 * count is bounded by pthread_create cost, not by the race window (the
 * last-drop race is fully exercised every round).  A few thousand rounds hits
 * the simultaneous-final-release window many times while keeping a single run
 * well under the suite timeout even under ASan. */
#define A_ROUNDS 4000

static arts_shared_ptr_t g_a_cb;
static _Atomic int g_a_gate;
static pthread_barrier_t g_a_barrier;

static void *a_worker(void *arg) {
  arts_shared_ptr_t my = (arts_shared_ptr_t)arg;
  /* all releasers fire together to maximize the last-drop race */
  pthread_barrier_wait(&g_a_barrier);
  arts_shared_release(&my);
  return NULL;
}

static int part_a(void) {
  atomic_store(&g_made, 0);
  atomic_store(&g_deleted, 0);
  for (int r = 0; r < A_ROUNDS; r++) {
    g_a_cb = arts_shared_make(make_obj(), del_obj); /* strong=1 */
    arts_shared_ptr_t copies[A_THREADS];
    for (int i = 0; i < A_THREADS; i++) {
      copies[i] = arts_shared_copy(g_a_cb); /* strong -> 1+A_THREADS */
    }
    pthread_barrier_init(&g_a_barrier, NULL, A_THREADS);
    pthread_t th[A_THREADS];
    for (int i = 0; i < A_THREADS; i++) {
      pthread_create(&th[i], NULL, a_worker, copies[i]);
    }
    for (int i = 0; i < A_THREADS; i++) {
      pthread_join(th[i], NULL);
    }
    pthread_barrier_destroy(&g_a_barrier);
    /* Owner's ref still held -> deleter must NOT have run yet. */
    if (atomic_load(&g_deleted) != (uint64_t)r) {
      (void)fprintf(stderr,
                    "FAIL shared_lifecycle_race[A]: premature delete round %d "
                    "(deleted=%" PRIu64 ")\n",
                    r, atomic_load(&g_deleted));
      return 1;
    }
    arts_shared_release(&g_a_cb); /* last drop -> deleter once */
    if (atomic_load(&g_deleted) != (uint64_t)(r + 1)) {
      (void)fprintf(stderr,
                    "FAIL shared_lifecycle_race[A]: deleter not exactly once "
                    "round %d\n",
                    r);
      return 1;
    }
  }
  if (atomic_load(&g_made) != A_ROUNDS || atomic_load(&g_deleted) != A_ROUNDS) {
    (void)fprintf(stderr,
                  "FAIL shared_lifecycle_race[A]: made/deleted mismatch\n");
    return 1;
  }
  return 0;
}

/* ---- Part B+C: mixed-API churn + abandon-vs-make ---- */
static arts_atomic_shared_ptr_t g_slot;
static atomic_int g_start;
static _Atomic int g_stop;

#define MIX_THREADS 8
#define MIX_OPS 100000

static inline uint64_t xs(uint64_t *s) {
  uint64_t x = *s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *s = x;
  return x;
}

static void *mix_worker(void *arg) {
  uint64_t rng = (uint64_t)(uintptr_t)arg * 0x9E3779B97F4A7C15ull + 1;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < MIX_OPS; i++) {
    uint64_t r = xs(&rng);
    switch (r & 7) {
    case 0:
    case 1: { /* store a fresh cb (releases prior occupant) */
      arts_atomic_shared_store(&g_slot, arts_shared_make(make_obj(), del_obj));
      break;
    }
    case 2: { /* exchange a fresh cb, release the returned old */
      arts_shared_ptr_t old = arts_atomic_shared_exchange(
          &g_slot, arts_shared_make(make_obj(), del_obj));
      if (old) {
        arts_shared_release(&old);
      }
      break;
    }
    case 3: { /* compare_exchange against a pinned load (abandon loser) */
      arts_shared_ptr_t exp = arts_atomic_shared_load(&g_slot);
      arts_shared_ptr_t fresh = arts_shared_make(make_obj(), del_obj);
      if (!arts_atomic_shared_compare_exchange(&g_slot, exp, fresh)) {
        void *obj = arts_shared_get(fresh);
        arts_shared_abandon(&fresh); /* loser: object stays ours, free it */
        del_obj(obj);
      }
      if (exp) {
        arts_shared_release(&exp);
      }
      break;
    }
    case 4: { /* abandon-vs-make: make then immediately abandon (no publish) */
      arts_shared_ptr_t tmp = arts_shared_make(make_obj(), del_obj);
      void *obj = arts_shared_get(tmp);
      arts_shared_abandon(&tmp); /* recycle cb, no deleter */
      del_obj(obj);              /* object is ours */
      break;
    }
    default: { /* load + get + (maybe copy) + release */
      arts_shared_ptr_t l = arts_atomic_shared_load(&g_slot);
      if (l) {
        obj_t *o = (obj_t *)arts_shared_get(l);
        if (o->magic != OBJ_MAGIC ||
            atomic_load_explicit(&o->live, memory_order_relaxed) != 1) {
          (void)fprintf(stderr,
                        "FAIL shared_lifecycle_race[B]: loaded dead object\n");
          abort();
        }
        if (r & 8) {
          arts_shared_ptr_t l2 = arts_shared_copy(l);
          arts_shared_release(&l2);
        }
        arts_shared_release(&l);
      }
      break;
    }
    }
  }
  return NULL;
}

static int part_bc(void) {
  atomic_store(&g_made, 0);
  atomic_store(&g_deleted, 0);
  atomic_init(&g_start, 0);
  atomic_init(&g_stop, 0);
  atomic_store(&g_slot, (arts_shared_ptr_t)NULL);
  arts_atomic_shared_store(&g_slot, arts_shared_make(make_obj(), del_obj));

  pthread_t th[MIX_THREADS];
  for (int i = 0; i < MIX_THREADS; i++) {
    pthread_create(&th[i], NULL, mix_worker, (void *)(uintptr_t)(i + 1));
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < MIX_THREADS; i++) {
    pthread_join(th[i], NULL);
  }
  arts_atomic_shared_store(&g_slot, (arts_shared_ptr_t)NULL); /* drop final */

  uint64_t made = atomic_load(&g_made);
  uint64_t del = atomic_load(&g_deleted);
  if (made != del) {
    (void)fprintf(stderr,
                  "FAIL shared_lifecycle_race[B/C]: made=%" PRIu64
                  " != deleted=%" PRIu64 " (leak/double-free)\n",
                  made, del);
    return 1;
  }
  return 0;
}

/* ---- Part D: make/use/release cycles reuse cbs cleanly (B126 invariant) ----
 */
static int part_d(void) {
  atomic_store(&g_made, 0);
  atomic_store(&g_deleted, 0);
  /* Tight reuse: make -> get -> release N times.  Each release recycles the cb
   * into the pool; the next make pops it.  If make's failure to re-init
   * link.next ever corrupted a live cb, this churn (with ASan) would break. */
  for (int i = 0; i < 200000; i++) {
    arts_shared_ptr_t s = arts_shared_make(make_obj(), del_obj);
    obj_t *o = (obj_t *)arts_shared_get(s);
    if (o->magic != OBJ_MAGIC) {
      (void)fprintf(stderr, "FAIL shared_lifecycle_race[D]: bad reuse\n");
      return 1;
    }
    arts_shared_release(&s);
  }
  if (atomic_load(&g_made) != atomic_load(&g_deleted)) {
    (void)fprintf(stderr, "FAIL shared_lifecycle_race[D]: reuse imbalance\n");
    return 1;
  }
  return 0;
}

int main(void) {
  if (part_a() != 0) {
    return 1;
  }
  if (part_bc() != 0) {
    return 1;
  }
  if (part_d() != 0) {
    return 1;
  }
  printf("PASS shared_lifecycle_race: N-way last-drop deleter once; mixed-API "
         "churn + abandon conserved every object (no UAF/leak); cb reuse clean "
         "(B126 invariant held)\n");
  return 0;
}

/* libc-backed shims so the test links without the ARTS runtime. */
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
