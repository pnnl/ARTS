/* SPDX-License-Identifier: Apache-2.0
 *
 * T243 — spin-lock correctness in libs/src/core/utils/atomics.c:
 *   arts_lock / arts_unlock / arts_try_lock  (mutual-exclusion spin lock)
 *   arts_reader_lock / arts_reader_unlock / arts_writer_lock /
 *   arts_writer_try_lock / arts_writer_unlock  (RW spin lock)
 *
 * Properties:
 *   A. Mutual exclusion / no lost updates:  N threads each do M
 *      lock; nonatomic_counter++; unlock.  Final == N*M.  Because the lock's
 *      acquire (cswap, __sync full barrier) and release (__atomic_store
 *      RELEASE) establish happens-before, the *plain* (non-atomic) counter
 *      update inside the critical section is data-race-free — this is exactly
 *      the property arts_lock must provide for outbox.c's seq_num_lock[].
 *   B. try_lock semantics:  on an unheld lock try_lock succeeds (0->1) and a
 *      second try_lock fails; after unlock it succeeds again.
 *   C. RW exclusion invariant:  a live reader census and a writer-presence
 *      flag must never be simultaneously positive — i.e. arts_writer_lock's
 *      critical section never overlaps any arts_reader_lock critical section,
 *      and concurrent readers may coexist.  We track an atomic active-reader
 *      count and an atomic in-writer flag and assert the forbidden overlap
 *      (writer sees readers>0, or reader sees writer set) never occurs.
 *
 * DOCUMENTED MISMATCH (census bug #2, not a defect we fix here):
 *   arts_writer_try_lock is a "try" only w.r.t. writer-vs-writer contention.
 *   On the success path (it won the write_lock CAS) it then *blocks*
 *   spinning `while (*read_lock)` until all readers drain — an unbounded
 *   wait.  Part D documents this by timing a writer_try_lock issued while a
 *   reader is parked in its critical section: the call does NOT return until
 *   the reader releases.  This is a naming/contract mismatch, reported not
 *   masked.
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

/* ----- Part A: mutual exclusion ----- */
#define A_THREADS 12
#define A_ITERS 40000

static volatile unsigned int g_lock; /* 0 = free */
static uint64_t g_plain_counter;     /* updated only inside the CS */
static atomic_int g_gate;

static void *mutex_worker(void *p) {
  (void)p;
  while (atomic_load_explicit(&g_gate, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < A_ITERS; i++) {
    (void)arts_lock(&g_lock);
    /* critical section: plain non-atomic RMW, protected by the lock. */
    g_plain_counter++;
    arts_unlock(&g_lock);
  }
  return NULL;
}

/* ----- Part C: RW exclusion ----- */
#define C_THREADS_R 8
#define C_THREADS_W 3
#define C_ITERS 8000

static volatile unsigned int g_read_lock;  /* reader census in the lock */
static volatile unsigned int g_write_lock; /* writer flag in the lock */

static atomic_int g_active_readers; /* witness: readers in their CS */
static atomic_int g_active_writer;  /* witness: writer in its CS (0/1) */
static atomic_int g_violation;      /* set if forbidden overlap observed */
static atomic_int g_gate_c;

static void check_reader_cs(void) {
  /* Inside a reader CS no writer may be present. */
  if (atomic_load_explicit(&g_active_writer, memory_order_acquire) != 0) {
    atomic_store_explicit(&g_violation, 1, memory_order_release);
  }
}

static void check_writer_cs(void) {
  /* Inside the writer CS no reader may be present, and writer is exclusive. */
  if (atomic_load_explicit(&g_active_readers, memory_order_acquire) != 0) {
    atomic_store_explicit(&g_violation, 2, memory_order_release);
  }
  if (atomic_load_explicit(&g_active_writer, memory_order_acquire) != 1) {
    atomic_store_explicit(&g_violation, 3, memory_order_release);
  }
}

static void *reader_worker(void *p) {
  (void)p;
  while (atomic_load_explicit(&g_gate_c, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < C_ITERS; i++) {
    arts_reader_lock(&g_read_lock, &g_write_lock);
    atomic_fetch_add_explicit(&g_active_readers, 1, memory_order_acq_rel);
    check_reader_cs();
    /* tiny dwell to widen the overlap window */
    for (volatile int s = 0; s < 4; s++) {
    }
    check_reader_cs();
    atomic_fetch_sub_explicit(&g_active_readers, 1, memory_order_acq_rel);
    arts_reader_unlock(&g_read_lock);
  }
  return NULL;
}

static void *writer_worker(void *p) {
  (void)p;
  while (atomic_load_explicit(&g_gate_c, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < C_ITERS; i++) {
    arts_writer_lock(&g_read_lock, &g_write_lock);
    atomic_store_explicit(&g_active_writer, 1, memory_order_release);
    check_writer_cs();
    for (volatile int s = 0; s < 4; s++) {
    }
    check_writer_cs();
    atomic_store_explicit(&g_active_writer, 0, memory_order_release);
    arts_writer_unlock(&g_write_lock);
  }
  return NULL;
}

/* ----- Part D: writer_try_lock blocks on reader drain (documentation) ----- */
static volatile unsigned int d_read_lock;
static volatile unsigned int d_write_lock;
static atomic_int d_reader_in_cs;   /* reader is holding the read lock */
static atomic_int d_reader_release; /* main tells reader to leave */

/* writer_try_lock helper, defined at file scope below. */
static void *d_try_thread(void *arg);

static void *d_reader(void *p) {
  (void)p;
  arts_reader_lock(&d_read_lock, &d_write_lock);
  atomic_store_explicit(&d_reader_in_cs, 1, memory_order_release);
  /* hold the read lock until told to release */
  while (atomic_load_explicit(&d_reader_release, memory_order_acquire) == 0) {
  }
  arts_reader_unlock(&d_read_lock);
  return NULL;
}

int main(void) {
  int rc = 0;

  /* ===== Part B: try_lock state machine (single thread). ===== */
  {
    volatile unsigned int lk = 0;
    if (!arts_try_lock(&lk)) {
      (void)fprintf(stderr, "FAIL atomics_locks: try_lock on free lock\n");
      rc = 1;
    }
    if (arts_try_lock(&lk)) {
      (void)fprintf(stderr, "FAIL atomics_locks: try_lock on held lock\n");
      rc = 1;
    }
    arts_unlock(&lk);
    if (!arts_try_lock(&lk)) {
      (void)fprintf(stderr,
                    "FAIL atomics_locks: try_lock after unlock failed\n");
      rc = 1;
    }
    arts_unlock(&lk);
  }

  /* ===== Part A: mutual exclusion under contention. ===== */
  {
    g_lock = 0;
    g_plain_counter = 0;
    atomic_init(&g_gate, 0);
    pthread_t th[A_THREADS];
    for (int i = 0; i < A_THREADS; i++) {
      pthread_create(&th[i], NULL, mutex_worker, NULL);
    }
    atomic_store_explicit(&g_gate, 1, memory_order_release);
    for (int i = 0; i < A_THREADS; i++) {
      pthread_join(th[i], NULL);
    }
    uint64_t want = (uint64_t)A_THREADS * A_ITERS;
    if (g_plain_counter != want) {
      (void)fprintf(stderr,
                    "FAIL atomics_locks: lost updates %" PRIu64 " != %" PRIu64
                    "\n",
                    g_plain_counter, want);
      rc = 1;
    }
  }

  /* ===== Part C: reader/writer exclusion invariant. ===== */
  {
    g_read_lock = 0;
    g_write_lock = 0;
    atomic_init(&g_active_readers, 0);
    atomic_init(&g_active_writer, 0);
    atomic_init(&g_violation, 0);
    atomic_init(&g_gate_c, 0);
    pthread_t rth[C_THREADS_R];
    pthread_t wth[C_THREADS_W];
    for (int i = 0; i < C_THREADS_R; i++) {
      pthread_create(&rth[i], NULL, reader_worker, NULL);
    }
    for (int i = 0; i < C_THREADS_W; i++) {
      pthread_create(&wth[i], NULL, writer_worker, NULL);
    }
    atomic_store_explicit(&g_gate_c, 1, memory_order_release);
    for (int i = 0; i < C_THREADS_R; i++) {
      pthread_join(rth[i], NULL);
    }
    for (int i = 0; i < C_THREADS_W; i++) {
      pthread_join(wth[i], NULL);
    }
    int v = atomic_load_explicit(&g_violation, memory_order_acquire);
    if (v != 0) {
      (void)fprintf(stderr,
                    "FAIL atomics_locks: RW exclusion violated (code %d)\n", v);
      rc = 1;
    }
    /* both witnesses must have drained to zero */
    if (atomic_load_explicit(&g_active_readers, memory_order_acquire) != 0 ||
        atomic_load_explicit(&g_active_writer, memory_order_acquire) != 0) {
      (void)fprintf(stderr, "FAIL atomics_locks: RW witness not drained\n");
      rc = 1;
    }
  }

  /* ===== Part D: document writer_try_lock blocks on reader drain. ===== */
  {
    d_read_lock = 0;
    d_write_lock = 0;
    atomic_init(&d_reader_in_cs, 0);
    atomic_init(&d_reader_release, 0);

    pthread_t rd;
    pthread_create(&rd, NULL, d_reader, NULL);
    /* wait until the reader is parked inside its critical section */
    while (atomic_load_explicit(&d_reader_in_cs, memory_order_acquire) == 0) {
    }
    /* Issue writer_try_lock from a helper thread so we can observe that it does
     * NOT return while the reader is still in its CS (the contract mismatch).
     */
    atomic_int try_done;
    atomic_init(&try_done, 0);
    pthread_t wr;
    pthread_create(&wr, NULL, d_try_thread, &try_done);

    /* Give the writer_try_lock time to run; it must STILL be blocked because
     * a reader holds the read lock.  Poll briefly. */
    int observed_blocked = 0;
    for (int spin = 0; spin < 2000000; spin++) {
      if (atomic_load_explicit(&try_done, memory_order_acquire) == 0) {
        observed_blocked = 1; /* at least once seen still-blocked */
      } else {
        break;
      }
    }
    if (!observed_blocked) {
      /* Extremely unlikely; would mean the try returned instantly even with a
       * reader present.  Not a hard failure of the runtime, but unexpected. */
      (void)fprintf(stderr,
                    "NOTE atomics_locks: writer_try_lock returned before "
                    "reader drained (unexpected)\n");
    }
    /* Now release the reader; the writer_try_lock must then complete. */
    atomic_store_explicit(&d_reader_release, 1, memory_order_release);
    pthread_join(rd, NULL);
    pthread_join(wr, NULL);
    if (atomic_load_explicit(&try_done, memory_order_acquire) != 1) {
      (void)fprintf(stderr,
                    "FAIL atomics_locks: writer_try_lock never completed\n");
      rc = 1;
    }
  }

  if (rc) {
    return 1;
  }
  printf("PASS atomics_locks: mutual-exclusion (no lost updates), try_lock "
         "state machine, RW no reader/writer overlap; writer_try_lock "
         "blocks-on-reader-drain documented\n");
  return 0;
}

/* writer_try_lock helper: must run AFTER the reader is parked; will block on
 * reader drain (the documented mismatch), then succeed once the reader leaves.
 * Sets *done=1 only after writer_try_lock returns true. */
static void *d_try_thread(void *arg) {
  atomic_int *done = (atomic_int *)arg;
  bool got = arts_writer_try_lock(&d_read_lock, &d_write_lock);
  if (got) {
    arts_writer_unlock(&d_write_lock);
  }
  atomic_store_explicit(done, got ? 1 : -1, memory_order_release);
  return NULL;
}
