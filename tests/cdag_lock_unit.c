/*
 * cdag_lock_unit.c — Standalone unit tests for cdag_lock.
 *
 * This test does NOT start the ARTS runtime. It exercises the cdag_lock
 * algorithm directly by passing dummy (non-NULL) pointers for struct
 * arts_edt_s* — the lock never dereferences them, it only stores them in
 * the waiter records for later callback delivery.
 */

#include "arts/memory/cdag_lock.h"
#include "arts/memory/db.h"
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static atomic_int g_callbacks = 0;
static atomic_int g_live_ro = 0;
static atomic_int g_live_ew = 0;
static atomic_int g_safety_bug = 0;

static void reset_counters(void) {
  atomic_store(&g_callbacks, 0);
  atomic_store(&g_live_ro, 0);
  atomic_store(&g_live_ew, 0);
  atomic_store(&g_safety_bug, 0);
}

static void count_cb(const struct cdag_lock_request_s *req, void *ctx) {
  (void)req;
  (void)ctx;
  atomic_fetch_add(&g_callbacks, 1);
}

struct record_ctx {
  unsigned int count;
  unsigned int cap;
  uintptr_t *entries;
};
static void record_cb(const struct cdag_lock_request_s *req, void *ctx) {
  struct record_ctx *r = (struct record_ctx *)ctx;
  if (r->count < r->cap) {
    r->entries[r->count++] = (uintptr_t)req->edt;
  }
}

static struct cdag_lock_request_s make_req(arts_db_access_mode_t mode,
                                           uintptr_t id) {
  struct cdag_lock_request_s r = {0};
  r.edt = (struct arts_edt_s *)id;
  r.edt_guid = (arts_guid_t)id;
  r.origin_rank = 0;
  r.slot = 0;
  r.mode = mode;
  return r;
}

/* ====================== T1 single RO ====================== */
static void t1_single_ro(void) {
  reset_counters();
  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  struct cdag_lock_request_s r = make_req(DB_MODE_RO, 1);
  enum cdag_submit_result res = cdag_lock_submit(&lock, &r);
  assert(res == CDAG_SUBMIT_HEAD_IMMEDIATE);

  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 0);
  assert(!cdag_lock_has_head(&lock));

  cdag_lock_destroy(&lock);
  printf("T1 single RO: PASS\n");
}

/* ====================== T2 single EW ====================== */
static void t2_single_ew(void) {
  reset_counters();
  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  struct cdag_lock_request_s r = make_req(DB_MODE_EW, 1);
  enum cdag_submit_result res = cdag_lock_submit(&lock, &r);
  assert(res == CDAG_SUBMIT_HEAD_IMMEDIATE);

  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 0);

  cdag_lock_destroy(&lock);
  printf("T2 single EW: PASS\n");
}

/* ====================== T3 RO batch ====================== */
static void t3_ro_batch(void) {
  reset_counters();
  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  for (int i = 0; i < 3; i++) {
    struct cdag_lock_request_s r = make_req(DB_MODE_RO, (uintptr_t)(i + 1));
    assert(cdag_lock_submit(&lock, &r) == CDAG_SUBMIT_HEAD_IMMEDIATE);
  }

  for (int i = 0; i < 3; i++) {
    cdag_lock_release(&lock, count_cb, NULL);
  }
  assert(atomic_load(&g_callbacks) == 0);
  assert(!cdag_lock_has_head(&lock));

  cdag_lock_destroy(&lock);
  printf("T3 RO batch: PASS\n");
}

/* ====================== T4 phase-fair ====================== */
static void t4_phase_fair(void) {
  reset_counters();
  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  /* 2 RO in head */
  struct cdag_lock_request_s r = make_req(DB_MODE_RO, 1);
  assert(cdag_lock_submit(&lock, &r) == CDAG_SUBMIT_HEAD_IMMEDIATE);
  r = make_req(DB_MODE_RO, 2);
  assert(cdag_lock_submit(&lock, &r) == CDAG_SUBMIT_HEAD_IMMEDIATE);

  /* EW arrives → queued in new generation after head */
  r = make_req(DB_MODE_EW, 3);
  assert(cdag_lock_submit(&lock, &r) == CDAG_SUBMIT_QUEUED);

  /* Third RO after writer — phase fair: cannot join original RO head */
  r = make_req(DB_MODE_RO, 4);
  assert(cdag_lock_submit(&lock, &r) == CDAG_SUBMIT_QUEUED);

  /* Drain original RO: the last one advances to EW generation, dispatches 1. */
  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 0);
  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 1); /* EW dispatched */

  /* Release EW → advances to ro3, dispatches 1. */
  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 2);

  /* Release ro3 → head empty. */
  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 2);
  assert(!cdag_lock_has_head(&lock));

  cdag_lock_destroy(&lock);
  printf("T4 phase-fair: PASS\n");
}

/* ====================== T5 EW chain ====================== */
static void t5_ew_chain(void) {
  reset_counters();
  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  for (int i = 0; i < 3; i++) {
    struct cdag_lock_request_s r = make_req(DB_MODE_EW, (uintptr_t)(i + 1));
    enum cdag_submit_result res = cdag_lock_submit(&lock, &r);
    if (i == 0) {
      assert(res == CDAG_SUBMIT_HEAD_IMMEDIATE);
    } else {
      assert(res == CDAG_SUBMIT_QUEUED);
    }
  }

  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 1);
  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 2);
  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 2);
  assert(!cdag_lock_has_head(&lock));

  cdag_lock_destroy(&lock);
  printf("T5 EW chain: PASS\n");
}

/* ====================== T6 RO batch release order ====================== */
static void t6_ro_release_order(void) {
  reset_counters();
  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  for (int i = 0; i < 3; i++) {
    struct cdag_lock_request_s r = make_req(DB_MODE_RO, (uintptr_t)(i + 1));
    assert(cdag_lock_submit(&lock, &r) == CDAG_SUBMIT_HEAD_IMMEDIATE);
  }

  /* Release-order within an RO batch doesn't matter — only the last
   * release (any order) advances. */
  cdag_lock_release(&lock, count_cb, NULL);
  cdag_lock_release(&lock, count_cb, NULL);
  cdag_lock_release(&lock, count_cb, NULL);
  assert(atomic_load(&g_callbacks) == 0);
  assert(!cdag_lock_has_head(&lock));

  cdag_lock_destroy(&lock);
  printf("T6 RO release order: PASS\n");
}

/* ====================== T7 callback FIFO order ====================== */
static void t7_callback_order(void) {
  reset_counters();
  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  /* 1 EW at head, then 3 RO queued behind. */
  struct cdag_lock_request_s r = make_req(DB_MODE_EW, 0xEE);
  assert(cdag_lock_submit(&lock, &r) == CDAG_SUBMIT_HEAD_IMMEDIATE);

  for (int i = 0; i < 3; i++) {
    r = make_req(DB_MODE_RO, (uintptr_t)(0x10 + i));
    assert(cdag_lock_submit(&lock, &r) == CDAG_SUBMIT_QUEUED);
  }

  /* Release EW → should dispatch 3 callbacks in FIFO order. */
  uintptr_t slots[8] = {0};
  struct record_ctx rc = {.count = 0, .cap = 8, .entries = slots};
  cdag_lock_release(&lock, record_cb, &rc);
  assert(rc.count == 3);
  assert(rc.entries[0] == 0x10);
  assert(rc.entries[1] == 0x11);
  assert(rc.entries[2] == 0x12);

  /* Drain the RO generation. */
  for (int i = 0; i < 3; i++) {
    cdag_lock_release(&lock, count_cb, NULL);
  }
  cdag_lock_destroy(&lock);
  printf("T7 callback FIFO order: PASS\n");
}

/* ====================== T8 multi-threaded RO stress ====================== */
struct stress_ctx_s {
  struct cdag_lock_s *lock;
  int iters;
};

static void *ro_stress_thread(void *arg) {
  struct stress_ctx_s *s = (struct stress_ctx_s *)arg;
  for (int i = 0; i < s->iters; i++) {
    struct cdag_lock_request_s r = make_req(DB_MODE_RO, (uintptr_t)(i + 1));
    enum cdag_submit_result res = cdag_lock_submit(s->lock, &r);
    assert(res == CDAG_SUBMIT_HEAD_IMMEDIATE);
    atomic_fetch_add(&g_live_ro, 1);
    if (atomic_load(&g_live_ew) > 0) {
      atomic_store(&g_safety_bug, 1);
    }
    for (volatile int k = 0; k < 20; k++) {
    }
    atomic_fetch_sub(&g_live_ro, 1);
    cdag_lock_release(s->lock, count_cb, NULL);
  }
  return NULL;
}

static void t8_ro_stress(int nthreads, int iters) {
  reset_counters();
  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  pthread_t *tids = (pthread_t *)calloc((size_t)nthreads, sizeof(pthread_t));
  struct stress_ctx_s *ctxs = (struct stress_ctx_s *)calloc(
      (size_t)nthreads, sizeof(struct stress_ctx_s));
  for (int i = 0; i < nthreads; i++) {
    ctxs[i].lock = &lock;
    ctxs[i].iters = iters;
    pthread_create(&tids[i], NULL, ro_stress_thread, &ctxs[i]);
  }
  for (int i = 0; i < nthreads; i++) {
    pthread_join(tids[i], NULL);
  }
  int le = atomic_load(&g_live_ew);
  int lr = atomic_load(&g_live_ro);
  int bug = atomic_load(&g_safety_bug);
  printf("T8 RO stress (%d threads × %d iters): live_ew=%d live_ro=%d bug=%d\n",
         nthreads, iters, le, lr, bug);
  assert(le == 0 && lr == 0 && bug == 0);
  assert(!cdag_lock_has_head(&lock));

  cdag_lock_destroy(&lock);
  free(tids);
  free(ctxs);
  printf("T8 RO stress: PASS\n");
}

/* ====================== T9 single-thread mixed dispatch ====================== */
/* A scheduler-model test: submit a series of mixed RO/EW requests, track
 * how many are currently dispatched (live), and release them in FIFO
 * order. After each release, on_advance fires for any newly-runnable
 * requests, which we treat as "new live" by incrementing a counter.
 *
 * Because we release exactly one slot per call, the total number of
 * releases must equal the total number of submits. */

struct t9_state_s {
  int n_live; /* requests dispatched but not yet released */
  int n_total_dispatched;
  int max_ro_concurrent;
  int max_ew_concurrent;
  int current_ro;
  int current_ew;
  int bug;
};

static struct t9_state_s g_t9_state;

static void t9_cb(const struct cdag_lock_request_s *req, void *ctx) {
  (void)ctx;
  g_t9_state.n_live++;
  g_t9_state.n_total_dispatched++;
  if (cdag_lock_mode_of(req->mode) == CDAG_LOCK_MODE_EW) {
    g_t9_state.current_ew++;
    if (g_t9_state.current_ew > 1 || g_t9_state.current_ro > 0) {
      g_t9_state.bug = 1;
    }
    if (g_t9_state.current_ew > g_t9_state.max_ew_concurrent) {
      g_t9_state.max_ew_concurrent = g_t9_state.current_ew;
    }
  } else {
    g_t9_state.current_ro++;
    if (g_t9_state.current_ew > 0) {
      g_t9_state.bug = 1;
    }
    if (g_t9_state.current_ro > g_t9_state.max_ro_concurrent) {
      g_t9_state.max_ro_concurrent = g_t9_state.current_ro;
    }
  }
}

/* Release one slot, but because gen-less release decrements head's
 * active, we need to know what mode the head currently is. We keep a
 * parallel queue of "dispatched modes" that tracks the FIFO of
 * dispatched requests across all generations. When we release, we pop
 * the front and decrement the appropriate current_* counter. */
static arts_db_access_mode_t g_t9_dispatch_fifo[512];
static int g_t9_fifo_head = 0;
static int g_t9_fifo_tail = 0;

static void t9_cb_fifo(const struct cdag_lock_request_s *req, void *ctx) {
  (void)ctx;
  g_t9_state.n_total_dispatched++;
  g_t9_dispatch_fifo[g_t9_fifo_tail++] = req->mode;
  if (cdag_lock_mode_of(req->mode) == CDAG_LOCK_MODE_EW) {
    g_t9_state.current_ew++;
    if (g_t9_state.current_ew > 1 || g_t9_state.current_ro > 0) {
      g_t9_state.bug = 1;
    }
  } else {
    g_t9_state.current_ro++;
    if (g_t9_state.current_ew > 0) {
      g_t9_state.bug = 1;
    }
  }
}

static void t9_do_release(struct cdag_lock_s *lock) {
  /* Pop one dispatched entry from the FIFO. */
  if (g_t9_fifo_head == g_t9_fifo_tail) {
    return;
  }
  arts_db_access_mode_t m = g_t9_dispatch_fifo[g_t9_fifo_head++];
  if (cdag_lock_mode_of(m) == CDAG_LOCK_MODE_EW) {
    g_t9_state.current_ew--;
  } else {
    g_t9_state.current_ro--;
  }
  cdag_lock_release(lock, t9_cb_fifo, NULL);
}

static void t9_mixed_dispatch(int n_requests, unsigned int seed) {
  reset_counters();
  memset(&g_t9_state, 0, sizeof(g_t9_state));
  g_t9_fifo_head = 0;
  g_t9_fifo_tail = 0;

  struct cdag_lock_s lock;
  cdag_lock_init(&lock);

  int total_submits = 0;
  for (int i = 0; i < n_requests; i++) {
    int want_ew = (rand_r(&seed) & 3) == 0;
    arts_db_access_mode_t mode = want_ew ? DB_MODE_EW : DB_MODE_RO;

    struct cdag_lock_request_s r = {0};
    r.edt = (struct arts_edt_s *)(uintptr_t)(i + 1);
    r.edt_guid = (arts_guid_t)i;
    r.mode = mode;
    enum cdag_submit_result res = cdag_lock_submit(&lock, &r);
    total_submits++;

    if (res == CDAG_SUBMIT_HEAD_IMMEDIATE) {
      /* Equivalent to a dispatch happening via the submit itself. */
      g_t9_dispatch_fifo[g_t9_fifo_tail++] = mode;
      if (cdag_lock_mode_of(mode) == CDAG_LOCK_MODE_EW) {
        g_t9_state.current_ew++;
        if (g_t9_state.current_ew > 1 || g_t9_state.current_ro > 0) {
          g_t9_state.bug = 1;
        }
      } else {
        g_t9_state.current_ro++;
        if (g_t9_state.current_ew > 0) {
          g_t9_state.bug = 1;
        }
      }
      g_t9_state.n_total_dispatched++;
    }

    /* Half the time, release an already-dispatched request to exercise
     * advance chains. */
    if ((rand_r(&seed) & 1) == 0) {
      t9_do_release(&lock);
    }
  }

  /* Drain everything that's still dispatched, which will cascade through
   * advance() and dispatch the queued generations. */
  while (g_t9_fifo_head != g_t9_fifo_tail) {
    t9_do_release(&lock);
  }

  printf("T9 mixed dispatch (%d reqs): total_dispatched=%d current_ew=%d "
         "current_ro=%d bug=%d\n",
         n_requests, g_t9_state.n_total_dispatched, g_t9_state.current_ew,
         g_t9_state.current_ro, g_t9_state.bug);
  assert(g_t9_state.bug == 0);
  assert(g_t9_state.current_ew == 0);
  assert(g_t9_state.current_ro == 0);
  assert(g_t9_state.n_total_dispatched == total_submits);
  assert(!cdag_lock_has_head(&lock));

  cdag_lock_destroy(&lock);
  printf("T9 mixed dispatch: PASS\n");
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  t1_single_ro();
  t2_single_ew();
  t3_ro_batch();
  t4_phase_fair();
  t5_ew_chain();
  t6_ro_release_order();
  t7_callback_order();
  t8_ro_stress(4, 10000);
  t8_ro_stress(8, 5000);
  t8_ro_stress(16, 2000);
  t9_mixed_dispatch(64, 1);
  t9_mixed_dispatch(128, 42);
  t9_mixed_dispatch(200, 2026);
  printf("ALL UNIT TESTS PASSED\n");
  return 0;
}
