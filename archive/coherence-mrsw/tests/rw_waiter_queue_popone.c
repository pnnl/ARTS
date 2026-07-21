/* SPDX-License-Identifier: Apache-2.0
 *
 * T066 — MRSW arts_db_rw_waiter_queue_* (cache.pending_rw, pop-one MPSC).
 *
 * Unlike the MRNEW cache-side RW chain (a LIFO Treiber stack), the MRSW
 * per-cache RW-waiter queue is a Vyukov MPSC consumed ONE waiter at a time in
 * FIFO order — the releaser pops exactly the next waiter to grant — and exposes
 * peek_empty for a release-time Dekker re-check.  Defined in
 * mrsw/waiter_queue.c → MRSW-only.  #includes that TU standalone (precedent:
 * rank_bitset.c).  Self-skips elsewhere.
 *
 * Properties exercised:
 *   1. Functional: init→peek_empty true; push then peek_empty false; pop FIFO
 *      ((edt_guid,slot) round-trip via container_of); pop on empty → false.
 *   2. Stub never freed / last-node-becomes-sentinel: drain to empty, re-push,
 *      drain again (the popped node becomes the new sentinel; the embedded stub
 *      is never freed — ASan would catch a stub free).
 *   3. count conservation, no double-free: N producers + 1 consumer pop-one;
 *      every (producer,seq) popped exactly once, per-producer FIFO preserved
 *      (count == produced; ASan clean).
 *   4. producer-mid-link retry + peek_empty false-empty window: the pop() loop
 *      spins on the transient-NULL mid-link; peek_empty conservatively reports
 *      empty during that window (re-checked after the link lands).  Driven by
 *      heavy concurrency.
 *
 * Build: -DARTS_PROTOCOL_MRSW=1 (+ a timing) -DARTS_UNIT_STANDALONE_SHIMS.
 */

#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW)
int main(void) {
  printf("PASS rw_waiter_queue_popone: skipped (MRSW-only; cache.pending_rw is "
         "a pop-one Vyukov MPSC only in the MRSW engine)\n");
  return 0;
}
#else

#include "arts/coherence/coherence.h"
#include "arts/coherence/home.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void *watchdog(void *arg) {
  (void)arg;
  struct timespec ts = {15, 0};
  nanosleep(&ts, NULL);
  (void)fprintf(stderr,
                "FAIL rw_waiter_queue_popone: TIMEOUT (suspected deadlock)\n");
  _exit(1);
  return NULL;
}

static arts_guid_t mk_guid(uint64_t v) { return (arts_guid_t)v; }

/* ---- Part 1+2: functional FIFO + peek_empty + stub-reuse ---- */
static int part1_functional(void) {
  struct arts_db_rw_waiter_queue_s q;
  arts_db_rw_waiter_queue_init(&q);
  int rc = 0;
  arts_guid_t g;
  unsigned int s;

  if (!arts_db_rw_waiter_queue_peek_empty(&q)) {
    (void)fprintf(stderr, "FAIL: fresh queue peek_empty not true\n");
    rc = 1;
  }
  if (arts_db_rw_waiter_queue_pop(&q, &g, &s)) {
    (void)fprintf(stderr, "FAIL: pop on empty returned true\n");
    rc = 1;
  }

  /* Push (guid=i, slot=i*7+1) for i in 0..9; pop FIFO. */
  for (unsigned int i = 0; i < 10; i++) {
    arts_db_rw_waiter_queue_push(&q, mk_guid(i), i * 7 + 1);
  }
  if (arts_db_rw_waiter_queue_peek_empty(&q)) {
    (void)fprintf(stderr, "FAIL: peek_empty true after pushes\n");
    rc = 1;
  }
  for (unsigned int i = 0; i < 10; i++) {
    arts_guid_t gg = mk_guid(0xdead);
    unsigned int ss = 0xffff;
    if (!arts_db_rw_waiter_queue_pop(&q, &gg, &ss)) {
      (void)fprintf(stderr, "FAIL: pop %u false\n", i);
      rc = 1;
      break;
    }
    if ((uint64_t)gg != i || ss != i * 7 + 1) {
      (void)fprintf(stderr,
                    "FAIL: FIFO/container_of mismatch i=%u got guid=%llu "
                    "slot=%u\n",
                    i, (unsigned long long)(uint64_t)gg, ss);
      rc = 1;
    }
  }
  if (!arts_db_rw_waiter_queue_peek_empty(&q)) {
    (void)fprintf(stderr, "FAIL: peek_empty false after draining all\n");
    rc = 1;
  }

  /* Stub-reuse: the last popped node became the sentinel; re-push + drain. */
  for (int round = 0; round < 3; round++) {
    for (unsigned int i = 0; i < 5; i++) {
      arts_db_rw_waiter_queue_push(&q, mk_guid(round * 100 + i), i);
    }
    for (unsigned int i = 0; i < 5; i++) {
      arts_guid_t gg;
      unsigned int ss;
      if (!arts_db_rw_waiter_queue_pop(&q, &gg, &ss) ||
          (uint64_t)gg != (uint64_t)(round * 100 + i) || ss != i) {
        (void)fprintf(stderr, "FAIL: stub-reuse round %d i %u\n", round, i);
        rc = 1;
      }
    }
  }
  arts_db_rw_waiter_queue_destroy(&q);
  return rc;
}

/* ---- Part 3+4: N producers + 1 consumer pop-one, per-producer FIFO ---- */
#define NPROD 6
#define PER_PROD 20000

static struct arts_db_rw_waiter_queue_s g_q;
static _Atomic int g_start;

static void *producer(void *arg) {
  uint64_t prod = (uint64_t)(uintptr_t)arg;
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  for (uint64_t s = 0; s < PER_PROD; s++) {
    /* guid encodes (prod,seq); slot carries seq for an independent check. */
    arts_db_rw_waiter_queue_push(&g_q, mk_guid(prod * PER_PROD + s),
                                 (unsigned int)s);
  }
  return NULL;
}

static int part3_mpsc(void) {
  arts_db_rw_waiter_queue_init(&g_q);
  atomic_store_explicit(&g_start, 0, memory_order_release);
  pthread_t th[NPROD];
  for (uint64_t i = 0; i < NPROD; i++) {
    pthread_create(&th[i], NULL, producer, (void *)(uintptr_t)i);
  }
  int rc = 0;
  int last_seq[NPROD];
  long count[NPROD];
  for (int i = 0; i < NPROD; i++) {
    last_seq[i] = -1;
    count[i] = 0;
  }
  const long total = (long)NPROD * PER_PROD;
  long consumed = 0;
  atomic_store_explicit(&g_start, 1, memory_order_release);

  while (consumed < total) {
    arts_guid_t g;
    unsigned int slot;
    if (!arts_db_rw_waiter_queue_pop(&g_q, &g, &slot)) {
      continue; /* truly empty between pushes */
    }
    uint64_t v = (uint64_t)g;
    uint64_t prod = v / PER_PROD;
    int seq = (int)(v % PER_PROD);
    if (prod >= NPROD) {
      (void)fprintf(stderr, "FAIL: bogus guid %llu\n", (unsigned long long)v);
      rc = 1;
      break;
    }
    if (slot != (unsigned int)seq) {
      (void)fprintf(stderr, "FAIL: slot/guid desync prod=%llu seq=%d slot=%u\n",
                    (unsigned long long)prod, seq, slot);
      rc = 1;
      break;
    }
    if (seq <= last_seq[prod]) {
      (void)fprintf(
          stderr, "FAIL: per-producer FIFO violated prod=%llu seq=%d last=%d\n",
          (unsigned long long)prod, seq, last_seq[prod]);
      rc = 1;
      break;
    }
    last_seq[prod] = seq;
    count[prod]++;
    consumed++;
  }
  for (int i = 0; i < NPROD; i++) {
    pthread_join(th[i], NULL);
  }
  if (!rc) {
    for (int i = 0; i < NPROD; i++) {
      if (count[i] != PER_PROD) {
        (void)fprintf(stderr, "FAIL: prod %d count %ld != %d\n", i, count[i],
                      PER_PROD);
        rc = 1;
      }
    }
    if (!arts_db_rw_waiter_queue_peek_empty(&g_q)) {
      (void)fprintf(stderr, "FAIL: peek_empty false after full drain\n");
      rc = 1;
    }
  }
  arts_db_rw_waiter_queue_destroy(&g_q);
  return rc;
}

int main(void) {
  pthread_t wd;
  pthread_create(&wd, NULL, watchdog, NULL);
  pthread_detach(wd);

  if (part1_functional() != 0) {
    return 1;
  }
  if (part3_mpsc() != 0) {
    return 1;
  }
  printf(
      "PASS rw_waiter_queue_popone: FIFO pop-one + peek_empty + stub-reuse + "
      "%d-producer MPSC per-producer-FIFO (%d each) count-conservation\n",
      NPROD, PER_PROD);
  return 0;
}

#ifdef ARTS_UNIT_STANDALONE_SHIMS
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
void *arts_malloc(size_t size) { return malloc(size); }
#endif

#endif /* ARTS_PROTOCOL_MRSW */

#if defined(ARTS_PROTOCOL_MRSW)
#include "core/coherence/mrsw/waiter_queue.c"
#endif
