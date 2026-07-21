/* SPDX-License-Identifier: Apache-2.0
 *
 * T065 — RCU arts_pending_rw_queue_* (cache-side RW waiter chain).
 *
 * A Treiber stack (arts_lf_stack_t) of arts_db_rw_waiter_s nodes (link FIRST).
 * Producers (foreign-rank acquire_remote_rw) prepend via release-CAS; the
 * single home-side consumer either drain-alls (GRANT / fail / destroy) — woken
 * + freed once each — or non-destructively for_each-walks (PROCEED, never
 * frees).  Order is immaterial (every waiter is woken regardless), which is why
 * a LIFO stack suffices vs the home lockreq Vyukov FIFO.
 *
 * Defined in rcu/home.c → RCU-only.  #includes the TU standalone
 * (precedent: rank_bitset.c).  Self-skips elsewhere.
 *
 * Properties exercised:
 *   1. drain-all conservation: push K waiters, one drain wakes + frees EXACTLY
 *      K, each once (counted via a per-waiter id seen-array; ASan/LSan catch a
 *      missed free or double-free).
 *   2. container_of: each woken cb receives the (edt_guid, slot) the producer
 *      pushed — proves link-is-first ARTS_CONTAINER_OF recovery.
 *   3. for_each never frees: a for_each walk leaves the chain intact, so a
 *      following drain still wakes every node (no premature free).
 *   4. concurrency: N producers push concurrently with repeated drains; a push
 *      racing a drain forms a fresh stack picked up by the NEXT drain — so the
 *      union over all drains wakes EXACTLY the pushed set, no loss, no dup.
 *
 * Build: -DARTS_PROTOCOL_RCU=1 (+ a timing) -DARTS_UNIT_STANDALONE_SHIMS.
 */

#include <stdio.h>

#if !defined(ARTS_PROTOCOL_RCU)
int main(void) {
  printf("PASS pending_rw_treiber: skipped (RCU-only; cache.pending_rw is a "
         "Treiber stack only in the RCU engine)\n");
  return 0;
}
#else

#include "arts/coherence/coherence.h"
#include "arts/coherence/home.h"
#include "arts/utils/lockfree_lifo.h"

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
                "FAIL pending_rw_treiber: TIMEOUT (suspected deadlock)\n");
  _exit(1);
  return NULL;
}

/* Build a GUID-shaped value that round-trips through the cb without
 * interpretation — the queue only stores/forwards it. */
static arts_guid_t mk_guid(uint64_t v) { return (arts_guid_t)v; }

/* ---- drain cb: records (guid,slot) per woken waiter ---- */
struct drain_ctx {
  int *seen;    /* seen[id]++ */
  int cap;      /* capacity of seen */
  int count;    /* total woken */
  int bad_guid; /* count of guid!=encoded(slot) container_of mismatches */
  int oob;      /* out-of-range ids */
};

/* We encode id into BOTH guid (id) and slot (id ^ 0x5555) so the cb can verify
 * the (guid,slot) pair was delivered together (container_of integrity). */
static void drain_cb(arts_guid_t edt_guid, unsigned int slot, void *ctxv) {
  struct drain_ctx *c = (struct drain_ctx *)ctxv;
  long id = (long)(uint64_t)edt_guid;
  if (id < 0 || id >= c->cap) {
    c->oob++;
    return;
  }
  if (slot != (unsigned int)(id ^ 0x5555)) {
    c->bad_guid++;
  }
  c->seen[id]++;
  c->count++;
}

static struct arts_db_rw_waiter_s *mk_waiter(long id) {
  struct arts_db_rw_waiter_s *w =
      (struct arts_db_rw_waiter_s *)malloc(sizeof(*w));
  w->edt_guid = mk_guid((uint64_t)id);
  w->slot = (unsigned int)(id ^ 0x5555);
  return w;
}

/* ---- Part 1+2: drain-all conservation + container_of ---- */
static int part1_drain_all(void) {
  arts_lf_stack_t q;
  arts_pending_rw_queue_init(&q);
  const int K = 1000;
  for (long i = 0; i < K; i++) {
    arts_pending_rw_queue_push(&q, mk_waiter(i));
  }
  int seen[1000];
  memset(seen, 0, sizeof(seen));
  struct drain_ctx c = {seen, K, 0, 0, 0};
  arts_pending_rw_queue_drain(&q, drain_cb, &c);
  int rc = 0;
  if (c.count != K || c.bad_guid || c.oob) {
    (void)fprintf(stderr, "FAIL: drain woke %d (want %d) bad_guid=%d oob=%d\n",
                  c.count, K, c.bad_guid, c.oob);
    rc = 1;
  }
  for (int i = 0; i < K; i++) {
    if (seen[i] != 1) {
      (void)fprintf(stderr, "FAIL: waiter %d woken %d times\n", i, seen[i]);
      rc = 1;
    }
  }
  /* Drain again on the now-empty stack: zero wakes, no crash. */
  struct drain_ctx c2 = {seen, K, 0, 0, 0};
  arts_pending_rw_queue_drain(&q, drain_cb, &c2);
  if (c2.count != 0) {
    (void)fprintf(stderr, "FAIL: second drain woke %d (want 0)\n", c2.count);
    rc = 1;
  }
  arts_pending_rw_queue_destroy(&q);
  return rc;
}

/* ---- Part 3: for_each never frees ---- */
static int part3_for_each_no_free(void) {
  arts_lf_stack_t q;
  arts_pending_rw_queue_init(&q);
  const int K = 200;
  for (long i = 0; i < K; i++) {
    arts_pending_rw_queue_push(&q, mk_waiter(i));
  }
  int seen[200];
  int rc = 0;
  /* Two for_each walks: both must visit all K, none freed (so the second walk
   * still sees them).  A premature free → ASan UAF or a short count. */
  for (int pass = 0; pass < 2; pass++) {
    memset(seen, 0, sizeof(seen));
    struct drain_ctx c = {seen, K, 0, 0, 0};
    arts_pending_rw_queue_for_each(&q, drain_cb, &c);
    if (c.count != K || c.bad_guid || c.oob) {
      (void)fprintf(stderr, "FAIL: for_each pass %d count %d (want %d)\n", pass,
                    c.count, K);
      rc = 1;
    }
    for (int i = 0; i < K; i++) {
      if (seen[i] != 1) {
        rc = 1;
      }
    }
  }
  /* A subsequent drain still frees every node exactly once (proves for_each
   * left them all on the chain). */
  memset(seen, 0, sizeof(seen));
  struct drain_ctx c = {seen, K, 0, 0, 0};
  arts_pending_rw_queue_drain(&q, drain_cb, &c);
  if (c.count != K) {
    (void)fprintf(stderr, "FAIL: drain after for_each woke %d (want %d)\n",
                  c.count, K);
    rc = 1;
  }
  arts_pending_rw_queue_destroy(&q);
  return rc;
}

/* ---- Part 4: push concurrent with drain — survives to next drain ---- */
#define NPROD 6
#define PER_PROD 5000
#define TOTAL (NPROD * PER_PROD)

static arts_lf_stack_t g_q;
static _Atomic int g_start;
static _Atomic int g_prod_done;
static _Atomic int g_seen[TOTAL];
static _Atomic long g_woken;

static void conc_cb(arts_guid_t edt_guid, unsigned int slot, void *ctxv) {
  (void)slot;
  (void)ctxv;
  long id = (long)(uint64_t)edt_guid;
  if (id >= 0 && id < TOTAL) {
    atomic_fetch_add_explicit(&g_seen[id], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_woken, 1, memory_order_relaxed);
  }
}

static void *conc_producer(void *arg) {
  long base = (long)(intptr_t)arg * PER_PROD;
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  for (long s = 0; s < PER_PROD; s++) {
    long id = base + s;
    struct arts_db_rw_waiter_s *w =
        (struct arts_db_rw_waiter_s *)malloc(sizeof(*w));
    w->edt_guid = mk_guid((uint64_t)id);
    w->slot = (unsigned int)id;
    arts_pending_rw_queue_push(&g_q, w);
  }
  atomic_fetch_add_explicit(&g_prod_done, 1, memory_order_release);
  return NULL;
}

static int part4_concurrent(void) {
  arts_pending_rw_queue_init(&g_q);
  atomic_store_explicit(&g_start, 0, memory_order_release);
  atomic_store_explicit(&g_prod_done, 0, memory_order_release);
  atomic_store_explicit(&g_woken, 0, memory_order_release);
  for (int i = 0; i < TOTAL; i++) {
    atomic_store_explicit(&g_seen[i], 0, memory_order_relaxed);
  }
  pthread_t th[NPROD];
  for (int i = 0; i < NPROD; i++) {
    pthread_create(&th[i], NULL, conc_producer, (void *)(intptr_t)i);
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);

  /* Single consumer: keep draining while producers run, then drain once more
   * after they finish to absorb any fresh-stack remainder.  A push racing the
   * drain's atomic-exchange forms a new stack the next drain claims. */
  while (atomic_load_explicit(&g_prod_done, memory_order_acquire) < NPROD) {
    arts_pending_rw_queue_drain(&g_q, conc_cb, NULL);
  }
  /* Final settling drains (loop until the woken total stabilises at TOTAL). */
  for (int spin = 0; spin < 1000; spin++) {
    arts_pending_rw_queue_drain(&g_q, conc_cb, NULL);
    if (atomic_load_explicit(&g_woken, memory_order_acquire) == TOTAL) {
      break;
    }
  }
  for (int i = 0; i < NPROD; i++) {
    pthread_join(th[i], NULL);
  }
  /* One more drain after join: guarantees nothing is left on the chain. */
  arts_pending_rw_queue_drain(&g_q, conc_cb, NULL);

  int rc = 0;
  long w = atomic_load_explicit(&g_woken, memory_order_acquire);
  if (w != TOTAL) {
    (void)fprintf(stderr, "FAIL: concurrent woke %ld (want %d)\n", w, TOTAL);
    rc = 1;
  }
  for (int i = 0; i < TOTAL; i++) {
    int s = atomic_load_explicit(&g_seen[i], memory_order_relaxed);
    if (s != 1) {
      (void)fprintf(stderr, "FAIL: waiter %d woken %d times (loss/dup)\n", i,
                    s);
      rc = 1;
      break;
    }
  }
  arts_pending_rw_queue_destroy(&g_q);
  return rc;
}

int main(void) {
  pthread_t wd;
  pthread_create(&wd, NULL, watchdog, NULL);
  pthread_detach(wd);

  if (part1_drain_all() != 0) {
    return 1;
  }
  if (part3_for_each_no_free() != 0) {
    return 1;
  }
  if (part4_concurrent() != 0) {
    return 1;
  }
  printf("PASS pending_rw_treiber: drain-all conservation + container_of + "
         "for_each-no-free + %d-producer concurrent (%d each) no loss/dup\n",
         NPROD, PER_PROD);
  return 0;
}

#ifdef ARTS_UNIT_STANDALONE_SHIMS
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
void *arts_malloc(size_t size) { return malloc(size); }
#endif

#endif /* ARTS_PROTOCOL_RCU */

#if defined(ARTS_PROTOCOL_RCU)
#include "core/coherence/rcu/home.c"
#endif
