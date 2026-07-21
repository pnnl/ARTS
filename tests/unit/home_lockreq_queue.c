/* SPDX-License-Identifier: Apache-2.0
 *
 * T064 — arts_home_lockreq_queue_{init,push,pop,peek,empty,destroy}
 *        (the home OWNERSHIP-REQUEST FIFO; a Vyukov MPSC, drain-one).
 *
 * This queue type is byte-identical in rcu/home.c and
 * lock/home.c — whichever home.c the build configures defines the bodies.  The
 * test #includes the matching protocol's home.c into a standalone TU (no ARTS
 * runtime; precedent: tests/unit/rank_bitset.c, edt_gpu.cu->edt.c) and is
 * validated under all three protocol defines.  Self-skips under WRF_RCU (no
 * ownership lease → no home lockreq queue compiled).
 *
 * Properties exercised:
 *   1. Functional single-thread: empty on init, peek/pop false on empty,
 *      FIFO order (oldest popped first = next ownership target), peek
 *      non-destructive, empty() tracks pop draining.
 *   2. Stub never freed: drain-to-empty then re-push then drain again works
 *      (the embedded sentinel stays usable; ASan would catch a stub free).
 *   3. Concurrent N producers + 1 consumer: no loss, no dup, and per-producer
 *      FIFO preserved (each producer pushes a strictly increasing sequence;
 *      the consumer must observe each producer's values in increasing order).
 *      This also exercises the transient-NULL mid-link spin retry in pop().
 *
 * Build: -DARTS_PROTOCOL_RCU=1 (+ -DARTS_TIMING_LAZY=1 |
 *        -DARTS_PROTOCOL_RWLOCK=1
 *   with -DARTS_UNIT_STANDALONE_SHIMS for the libc alloc shims.
 */

#include <stdio.h>

#if defined(ARTS_PROTOCOL_WRF_RCU)
/* WRF_RCU carries no ownership lease → coherence.h does not even define the home
 * lockreq queue struct under WRF_RCU, so the rest of this TU would not compile.
 * Self-skip cleanly BEFORE pulling in any coherence header. */
int main(void) {
  printf(
      "PASS home_lockreq_queue: skipped under WRF_RCU (no home lockreq queue in "
      "this protocol)\n");
  return 0;
}
#else

#include "arts/coherence/home.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The pop() transient-NULL retry is lock-free but unbounded under a stalled
 * producer; a hung consumer would otherwise wedge ctest, so arm a watchdog. */
#include <unistd.h>
static void *watchdog(void *arg) {
  (void)arg;
  struct timespec ts = {15, 0};
  nanosleep(&ts, NULL);
  (void)fprintf(stderr,
                "FAIL home_lockreq_queue: TIMEOUT (suspected deadlock)\n");
  _exit(1);
  return NULL;
}

/* ---- Part 1: single-thread functional + FIFO + peek + empty ---- */
static int part1_functional(void) {
  struct arts_home_lockreq_queue_s q;
  arts_home_lockreq_queue_init(&q);
  int rc = 0;
  unsigned int r = 0xdeadbeef;

  if (!arts_home_lockreq_queue_empty(&q)) {
    (void)fprintf(stderr, "FAIL: fresh queue not empty\n");
    rc = 1;
  }
  if (arts_home_lockreq_queue_peek(&q, &r, NULL)) {
    (void)fprintf(stderr, "FAIL: peek on empty returned true\n");
    rc = 1;
  }
  if (arts_home_lockreq_queue_pop(&q, &r, NULL)) {
    (void)fprintf(stderr, "FAIL: pop on empty returned true\n");
    rc = 1;
  }

  /* Push 0..9 — FIFO: pop must yield 0,1,...,9. */
  for (unsigned int i = 0; i < 10; i++) {
    arts_home_lockreq_queue_push(&q, i, NULL);
  }
  if (arts_home_lockreq_queue_empty(&q)) {
    (void)fprintf(stderr, "FAIL: queue empty after 10 pushes\n");
    rc = 1;
  }
  /* peek is non-destructive: two peeks return the same front (oldest). */
  unsigned int p1 = 0, p2 = 0;
  if (!arts_home_lockreq_queue_peek(&q, &p1, NULL) ||
      !arts_home_lockreq_queue_peek(&q, &p2, NULL) || p1 != 0 || p2 != 0) {
    (void)fprintf(stderr,
                  "FAIL: peek not non-destructive / not FIFO front "
                  "(p1=%u p2=%u)\n",
                  p1, p2);
    rc = 1;
  }
  for (unsigned int i = 0; i < 10; i++) {
    unsigned int got = 0xffffffff;
    if (!arts_home_lockreq_queue_pop(&q, &got, NULL)) {
      (void)fprintf(stderr, "FAIL: pop %u returned false\n", i);
      rc = 1;
      break;
    }
    if (got != i) {
      (void)fprintf(stderr, "FAIL: FIFO violated, expected %u got %u\n", i,
                    got);
      rc = 1;
    }
  }
  if (!arts_home_lockreq_queue_empty(&q)) {
    (void)fprintf(stderr, "FAIL: queue not empty after draining all\n");
    rc = 1;
  }
  if (arts_home_lockreq_queue_pop(&q, &r, NULL)) {
    (void)fprintf(stderr, "FAIL: pop after drain returned true\n");
    rc = 1;
  }
  arts_home_lockreq_queue_destroy(&q);
  return rc;
}

/* ---- Part 2: stub-never-freed — drain, re-push, drain again ---- */
static int part2_stub_reuse(void) {
  struct arts_home_lockreq_queue_s q;
  arts_home_lockreq_queue_init(&q);
  int rc = 0;
  for (int round = 0; round < 3; round++) {
    for (unsigned int i = 0; i < 5; i++) {
      arts_home_lockreq_queue_push(&q, round * 100 + i, NULL);
    }
    for (unsigned int i = 0; i < 5; i++) {
      unsigned int got = 0;
      if (!arts_home_lockreq_queue_pop(&q, &got, NULL) ||
          got != (unsigned int)(round * 100 + i)) {
        (void)fprintf(stderr, "FAIL: stub-reuse round %d idx %u got %u\n",
                      round, i, got);
        rc = 1;
      }
    }
    if (!arts_home_lockreq_queue_empty(&q)) {
      (void)fprintf(stderr, "FAIL: stub-reuse not empty after round %d\n",
                    round);
      rc = 1;
    }
  }
  arts_home_lockreq_queue_destroy(&q);
  return rc;
}

/* ---- Part 3: N producers + 1 consumer, per-producer FIFO ---- */
#define NPROD 6
#define PER_PROD 20000
/* Encode (producer, seq) into a single rank value: rank = prod*PER_PROD + seq.
 * Consumer recovers producer = rank / PER_PROD, seq = rank % PER_PROD and
 * checks per-producer monotonicity (FIFO) + counts every value exactly once. */

static struct arts_home_lockreq_queue_s g_q;
static _Atomic int g_start;
static _Atomic int g_produced;

static void *producer(void *arg) {
  unsigned int prod = (unsigned int)(uintptr_t)arg;
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  for (unsigned int s = 0; s < PER_PROD; s++) {
    arts_home_lockreq_queue_push(&g_q, prod * PER_PROD + s, NULL);
    atomic_fetch_add_explicit(&g_produced, 1, memory_order_relaxed);
  }
  return NULL;
}

static int part3_mpsc(void) {
  arts_home_lockreq_queue_init(&g_q);
  atomic_store_explicit(&g_start, 0, memory_order_release);
  atomic_store_explicit(&g_produced, 0, memory_order_release);

  pthread_t th[NPROD];
  for (unsigned int i = 0; i < NPROD; i++) {
    pthread_create(&th[i], NULL, producer, (void *)(uintptr_t)i);
  }

  int rc = 0;
  int *last_seq = (int *)malloc(sizeof(int) * NPROD);
  int *count = (int *)calloc(NPROD, sizeof(int));
  for (int i = 0; i < NPROD; i++) {
    last_seq[i] = -1;
  }
  const long total = (long)NPROD * PER_PROD;
  long consumed = 0;

  atomic_store_explicit(&g_start, 1, memory_order_release);

  /* Single consumer drains until it has seen every value.  pop() exercises the
   * transient-NULL mid-link spin while producers are still mid-push. */
  while (consumed < total) {
    unsigned int v;
    if (!arts_home_lockreq_queue_pop(&g_q, &v, NULL)) {
      /* Truly empty (producers between pushes) — retry. */
      continue;
    }
    unsigned int prod = v / PER_PROD;
    int seq = (int)(v % PER_PROD);
    if (prod >= NPROD) {
      (void)fprintf(stderr, "FAIL: bogus value %u (prod %u)\n", v, prod);
      rc = 1;
      break;
    }
    if (seq <= last_seq[prod]) {
      (void)fprintf(stderr,
                    "FAIL: per-producer FIFO violated prod=%u seq=%d last=%d\n",
                    prod, seq, last_seq[prod]);
      rc = 1;
      break;
    }
    last_seq[prod] = seq;
    count[prod]++;
    consumed++;
  }

  for (unsigned int i = 0; i < NPROD; i++) {
    pthread_join(th[i], NULL);
  }
  if (!rc) {
    for (int i = 0; i < NPROD; i++) {
      if (count[i] != PER_PROD) {
        (void)fprintf(stderr, "FAIL: prod %d count %d != %d (loss/dup)\n", i,
                      count[i], PER_PROD);
        rc = 1;
      }
    }
    if (!arts_home_lockreq_queue_empty(&g_q)) {
      (void)fprintf(stderr, "FAIL: queue not empty after full drain\n");
      rc = 1;
    }
  }
  free(last_seq);
  free(count);
  arts_home_lockreq_queue_destroy(&g_q);
  return rc;
}

int main(void) {
  pthread_t wd;
  pthread_create(&wd, NULL, watchdog, NULL);
  pthread_detach(wd);

  if (part1_functional() != 0) {
    return 1;
  }
  if (part2_stub_reuse() != 0) {
    return 1;
  }
  if (part3_mpsc() != 0) {
    return 1;
  }
  printf("PASS home_lockreq_queue: FIFO + peek-nondestructive + empty + "
         "stub-reuse + %d-producer MPSC per-producer-FIFO (%d each)\n",
         NPROD, PER_PROD);
  return 0;
}

/* libc-backed alloc shims (the home.c bodies only use bare malloc/free for the
 * queue nodes; the bitset uses arts_calloc/arts_free, pulled in by the same
 * home.c TU).  Provided only in the standalone build. */
#ifdef ARTS_UNIT_STANDALONE_SHIMS
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
void *arts_malloc(size_t size) { return malloc(size); }

/* RWLOCK's home.c is monolithic: the same TU that defines the queue/bitset bodies
 * we test also defines the RWLOCK request/release/destroy HANDLERS, which
 * reference the broader runtime (transport, route table, buffer cb, identity).
 * T064 never invokes any of those — it only drives the queue API — so satisfy
 * the linker with inert stubs.  None are reachable from the test's call graph.
 */
#if defined(ARTS_PROTOCOL_RWLOCK)
#include "arts/coherence/buffer.h"
#include "arts/utils/shared.h"
unsigned int arts_global_rank_id = 0;
void mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot) {
  (void)edt_guid;
  (void)slot;
}
void arts_transport_send_async(int rank, char *message, unsigned int length) {
  (void)rank;
  (void)message;
  (void)length;
}
void arts_transport_loopback_post(const void *packet, unsigned int size) {
  (void)packet;
  (void)size;
}
void arts_handler_db_lock_grant(void *payload, size_t size) {
  (void)payload;
  (void)size;
}
arts_shared_ptr_t arts_db_buf_acquire(struct arts_db_cache_s *cache) {
  (void)cache;
  return NULL;
}
void arts_db_buf_release(arts_shared_ptr_t *h) { (void)h; }
struct arts_db_buffer_s *arts_db_buf_install(struct arts_db_cache_s *cache,
                                             uint64_t new_version,
                                             const void *data_payload,
                                             uint64_t db_size) {
  (void)cache;
  (void)new_version;
  (void)data_payload;
  (void)db_size;
  return NULL;
}
void arts_db_buf_write_inplace(struct arts_db_cache_s *cache, const void *data,
                               uint64_t db_size) {
  (void)cache;
  (void)data;
  (void)db_size;
}
void *arts_shared_get(arts_shared_ptr_t p) {
  (void)p;
  return NULL;
}
void arts_send_db_lock_release_ack(unsigned int releaser_rank,
                                   arts_guid_t db_guid, uint64_t cv) {
  (void)releaser_rank;
  (void)db_guid;
  (void)cv;
}
void arts_send_db_cache_destroy(unsigned int sharer_rank, arts_guid_t db_guid) {
  (void)sharer_rank;
  (void)db_guid;
}
bool arts_route_table_set_destroyed(arts_guid_t key) {
  (void)key;
  return false;
}
#endif /* ARTS_PROTOCOL_RWLOCK */
#endif

#endif /* !ARTS_PROTOCOL_WRF_RCU */

/* Pull in the protocol's home.c (defines the queue bodies).  Done at the end so
 * the test's own includes/decls are in scope first. */
#if defined(ARTS_PROTOCOL_RCU)
#include "core/coherence/rcu/home.c"
#elif defined(ARTS_PROTOCOL_RWLOCK)
#include "core/coherence/rwlock/home.c"
#endif
