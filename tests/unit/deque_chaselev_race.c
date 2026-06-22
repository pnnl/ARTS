/* SPDX-License-Identifier: Apache-2.0
 *
 * T020 — Chase-Lev work-stealing deque race (libs/src/core/utils/deque.c).
 * THE core contract: one OWNER thread that push_front/pop_front, and N
 * STEALER threads that pop_back, must between them observe every pushed item
 * exactly once — no loss, no duplicate, no corruption — while the array grows
 * under concurrent steals.  (Suspected bug B052: missing store-store fence
 * before the activeArray publish in push_front; masked on x86 TSO, surfaced
 * here under TSan and on weak-mem hardware.)
 *
 * Design (single-producer, multi-consumer = the exact Chase-Lev pattern):
 *   - Owner pushes items 0..NITEMS-1 (each encoded as a unique non-NULL ptr),
 *     interleaving its OWN pop_front with the pushes so the b==t last-element
 *     tie-break CAS path is hit alongside stealers.
 *   - Stealers spin on pop_back until the owner is done AND the deque is
 *     drained.
 *   - Every popped item (by owner or stealer) is recorded in a per-item
 *     atomic "seen" counter.  At the end EVERY item must have been seen
 *     EXACTLY once.  Total pop count must equal NITEMS.
 *   - The initial deque capacity is tiny so push_front grows it many times
 *     while stealers are actively reading activeArray/top (grow-under-steal).
 *
 * Why exactly-once is the right invariant: pop_front and pop_back race only on
 * the single remaining element, resolved by the CAS on `top`; a correct
 * deque guarantees exactly one winner.  A dropped item => some seen==0 (and
 * total<NITEMS); a duplicated item => some seen>1 (and total>NITEMS).
 */

#include "arts/utils/deque.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NSTEALERS 6
#define NITEMS (200 * 1000)
#define INIT_CAP 2 /* force many grows under steal */

static struct arts_deque_s *g_deque;
static atomic_int g_start;            /* start gate */
static atomic_int g_owner_done;       /* owner finished pushing+self-popping */
static atomic_uint_least32_t *g_seen; /* per-item pop count */
static _Atomic uint64_t g_total_popped;

static inline void *enc(uint64_t v) { return (void *)(uintptr_t)(v + 1); }
static inline uint64_t dec(void *p) { return (uint64_t)(uintptr_t)p - 1; }

static int record(void *o) {
  uint64_t v = dec(o);
  if (v >= NITEMS) {
    (void)fprintf(stderr, "FAIL deque_chaselev_race: bogus value %" PRIu64 "\n",
                  v);
    abort();
  }
  unsigned prev =
      atomic_fetch_add_explicit(&g_seen[v], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_total_popped, 1, memory_order_relaxed);
  if (prev != 0) {
    (void)fprintf(stderr,
                  "FAIL deque_chaselev_race: DUPLICATE pop of item %" PRIu64
                  " (now seen %u)\n",
                  v, prev + 1);
    abort();
  }
  return 0;
}

static void *stealer(void *arg) {
  (void)arg;
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  for (;;) {
    void *o = arts_deque_pop_back(g_deque);
    if (o) {
      record(o);
      continue;
    }
    /* Empty right now.  Only terminate once the owner is done AND a couple of
     * follow-up pop_backs still find nothing (avoid quitting during a
     * transient mid-grow empty window). */
    if (atomic_load_explicit(&g_owner_done, memory_order_acquire)) {
      void *a = arts_deque_pop_back(g_deque);
      if (a) {
        record(a);
        continue;
      }
      void *b = arts_deque_pop_back(g_deque);
      if (b) {
        record(b);
        continue;
      }
      break;
    }
  }
  return NULL;
}

static void *owner(void *arg) {
  (void)arg;
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  uint64_t next = 0;
  /* Push all items, periodically self-popping from the front to hit the
   * owner-vs-stealer last-element CAS path. */
  while (next < NITEMS) {
    arts_deque_push_front(g_deque, enc(next), 0);
    next++;
    /* Every so often, try to reclaim our own most-recent push (LIFO). */
    if ((next & 0x3) == 0) {
      void *o = arts_deque_pop_front(g_deque);
      if (o)
        record(o);
    }
  }
  /* Drain whatever the owner can still reclaim from the front. */
  for (;;) {
    void *o = arts_deque_pop_front(g_deque);
    if (!o)
      break;
    record(o);
  }
  atomic_store_explicit(&g_owner_done, 1, memory_order_release);
  return NULL;
}

int main(void) {
  g_deque = arts_deque_new(INIT_CAP);
  g_seen = calloc(NITEMS, sizeof(*g_seen));
  if (!g_seen) {
    (void)fprintf(stderr, "FAIL deque_chaselev_race: OOM seen[]\n");
    return 1;
  }
  atomic_init(&g_start, 0);
  atomic_init(&g_owner_done, 0);
  atomic_init(&g_total_popped, 0);

  pthread_t stealers[NSTEALERS];
  pthread_t owner_t;
  for (int i = 0; i < NSTEALERS; i++) {
    pthread_create(&stealers[i], NULL, stealer, NULL);
  }
  pthread_create(&owner_t, NULL, owner, NULL);

  atomic_store_explicit(&g_start, 1, memory_order_release);

  pthread_join(owner_t, NULL);
  for (int i = 0; i < NSTEALERS; i++) {
    pthread_join(stealers[i], NULL);
  }

  /* Final accounting: every item seen exactly once. */
  uint64_t missing = 0, dup = 0;
  for (uint64_t v = 0; v < NITEMS; v++) {
    unsigned s = atomic_load_explicit(&g_seen[v], memory_order_relaxed);
    if (s == 0)
      missing++;
    else if (s > 1)
      dup++;
  }
  uint64_t total = atomic_load_explicit(&g_total_popped, memory_order_relaxed);
  free(g_seen);
  arts_deque_delete(g_deque);

  if (missing || dup) {
    (void)fprintf(stderr,
                  "FAIL deque_chaselev_race: missing=%" PRIu64 " dup=%" PRIu64
                  " total=%" PRIu64 " (want %d)\n",
                  missing, dup, total, NITEMS);
    return 1;
  }
  if (total != (uint64_t)NITEMS) {
    (void)fprintf(stderr,
                  "FAIL deque_chaselev_race: total=%" PRIu64 " want %d\n",
                  total, NITEMS);
    return 1;
  }

  printf("PASS deque_chaselev_race: %d items, 1 owner + %d stealers, "
         "exactly-once\n",
         NITEMS, NSTEALERS);
  return 0;
}

/* ── libc-backed shims so the test links deque.c without the ARTS runtime ── */
void *arts_calloc_align(size_t nmemb, size_t size, size_t align) {
  void *p = NULL;
  size_t total = nmemb * size;
  if (align < sizeof(void *))
    align = sizeof(void *);
  size_t rem = total % align;
  if (rem)
    total += align - rem;
  if (posix_memalign(&p, align, total) != 0)
    return NULL;
  memset(p, 0, total);
  return p;
}
void arts_free(void *ptr) { free(ptr); }
uint64_t arts_atomic_cswap_u64(volatile uint64_t *destination, uint64_t old_val,
                               uint64_t swap_in) {
  __atomic_compare_exchange_n(destination, &old_val, swap_in, false,
                              __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return old_val;
}
