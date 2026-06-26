/* SPDX-License-Identifier: Apache-2.0
 *
 * T022 — grow-during-steal value-correctness + segment-reclamation safety
 * (libs/src/core/utils/deque.c).  Targets suspected bug B052: push_front grows
 * the array (grow_circular_array copies the live window [t,b) into a 2x
 * segment) and publishes `deque->activeArray = a` with NO store-store fence
 * before the publish.  A stealer's pop_back reads activeArray, then reads
 * segment[t]; if it observes the new activeArray before the copied element
 * stores (weak-mem), it reads a stale/garbage slot.  Also: the old segment is
 * deliberately never freed until arts_deque_delete, so a stealer that captured
 * a stale activeArray must still read VALID memory (no UAF).
 *
 * The value-identity check is the teeth here: each item's value encodes the
 * logical index at which it was pushed (value == index).  pop_back returns
 * items, and EACH returned value must equal a distinct, in-range index — and
 * EACH index must be returned exactly once.  A grow that copies the wrong
 * window, or a stealer that reads a not-yet-copied slot, yields a value that
 * is out of range or duplicated => detected.  ASan additionally proves the
 * old segments are not used-after-free (they're freed only at delete).
 *
 * To MAXIMIZE grow-under-steal: tiny initial capacity (1 -> grows on the very
 * first push), and the owner pushes in tight bursts (no self-pop) so the
 * deque fills and grows repeatedly while stealers hammer pop_back.
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
#define NITEMS (300 * 1000)
#define INIT_CAP 1 /* grows on the first push */

static struct arts_deque_s *g_deque;
static atomic_int g_start;
static atomic_int g_owner_done;
static atomic_uint_least32_t *g_seen;
static _Atomic uint64_t g_total;

static inline void *enc(uint64_t v) { return (void *)(uintptr_t)(v + 1); }
static inline uint64_t dec(void *p) { return (uint64_t)(uintptr_t)p - 1; }

static void record(void *o) {
  uint64_t v = dec(o);
  if (v >= NITEMS) { /* a stale/garbage slot read => out-of-range value */
    (void)fprintf(stderr,
                  "FAIL deque_grow_during_steal: out-of-range value %" PRIu64
                  " (stale slot across grow?)\n",
                  v);
    abort();
  }
  unsigned prev =
      atomic_fetch_add_explicit(&g_seen[v], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_total, 1, memory_order_relaxed);
  if (prev != 0) {
    (void)fprintf(stderr,
                  "FAIL deque_grow_during_steal: DUPLICATE value %" PRIu64 "\n",
                  v);
    abort();
  }
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
    if (atomic_load_explicit(&g_owner_done, memory_order_acquire)) {
      /* settle: a couple of confirming empty pops past owner-done. */
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
  /* Pure producer: push in bursts so the deque repeatedly fills and grows
   * while stealers are mid-pop_back.  No owner-side pop here — we want the
   * grow path stressed against steals, not the last-element CAS (that is
   * T021). */
  for (uint64_t i = 0; i < NITEMS; i++) {
    arts_deque_push_front(g_deque, enc(i), 0);
  }
  /* Owner reclaims any leftover from the front so the run terminates even if
   * stealers fall behind. */
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
    (void)fprintf(stderr, "FAIL deque_grow_during_steal: OOM\n");
    return 1;
  }
  atomic_init(&g_start, 0);
  atomic_init(&g_owner_done, 0);
  atomic_init(&g_total, 0);

  pthread_t st[NSTEALERS];
  pthread_t ow;
  for (int i = 0; i < NSTEALERS; i++) {
    pthread_create(&st[i], NULL, stealer, NULL);
  }
  pthread_create(&ow, NULL, owner, NULL);
  atomic_store_explicit(&g_start, 1, memory_order_release);
  pthread_join(ow, NULL);
  for (int i = 0; i < NSTEALERS; i++) {
    pthread_join(st[i], NULL);
  }

  uint64_t missing = 0, dup = 0;
  for (uint64_t v = 0; v < NITEMS; v++) {
    unsigned s = atomic_load_explicit(&g_seen[v], memory_order_relaxed);
    if (s == 0)
      missing++;
    else if (s > 1)
      dup++;
  }
  uint64_t total = atomic_load_explicit(&g_total, memory_order_relaxed);
  free(g_seen);
  arts_deque_delete(g_deque);

  if (missing || dup || total != (uint64_t)NITEMS) {
    (void)fprintf(stderr,
                  "FAIL deque_grow_during_steal: missing=%" PRIu64
                  " dup=%" PRIu64 " total=%" PRIu64 " (want %d)\n",
                  missing, dup, total, NITEMS);
    return 1;
  }

  printf("PASS deque_grow_during_steal: %d items, %d stealers, every value "
         "exactly-once across grows\n",
         NITEMS, NSTEALERS);
  return 0;
}

/* ── libc-backed shims so the test links deque.c without the ARTS runtime ── */
void *arts_calloc_aligned(size_t nmemb, size_t size, size_t align) {
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
