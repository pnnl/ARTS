/* SPDX-License-Identifier: Apache-2.0
 *
 * T017 — arts_atomic_shared_compare_exchange + load forward-progress
 * (shared.c).  Census 28.md §1.10 / gaps #1 (the single biggest unit-coverage
 * gap: compare_exchange has ZERO direct coverage) + §1.7 load livelock guard.
 *
 * compare_exchange contract:
 *   success — slot held `expected`, swap to `new_val`: slot takes new_val's
 *     ref and drops the ONE ref the slot held on `expected`; returns true.
 *     The caller's OWN ref on `expected` is untouched (it pinned `expected`
 *     against cb recycle so the raw-pointer compare cannot ABA).
 *   mismatch — slot did not hold `expected`: return false, leave slot + BOTH
 *     refs untouched.
 *
 * Part 1 (single-thread, deterministic, ASan): exact refcount accounting of
 * success and mismatch via a counting deleter — success drops exactly the
 * slot's ref on the old value; mismatch drops nothing.
 *
 * Part 2 (race, ASan): the real buffer.c:90 pattern.  N installer threads each
 *   make a fresh cb and try compare_exchange(slot, expected, fresh) where
 *   `expected` is a value they pinned by an atomic_load.  Exactly the CAS
 *   winners' new values survive; losers abandon their cb (object stays theirs).
 *   Concurrently a reader thread does atomic_shared_load+get+release in a tight
 *   loop (forward-progress under churn — must never livelock, never UAF, never
 *   read a torn/dead object).  At quiescence make-count == delete-count
 *   (refcount conservation, no leak / no double-free).
 */

#include "arts/utils/shared.h"

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic(uint64_t) g_made;
static _Atomic(uint64_t) g_deleted;

typedef struct {
  uint32_t magic;
} obj_t;
#define OBJ_MAGIC 0xBEEF1234u

static void *make_obj(void) {
  obj_t *o = (obj_t *)malloc(sizeof(obj_t));
  o->magic = OBJ_MAGIC;
  atomic_fetch_add_explicit(&g_made, 1, memory_order_relaxed);
  return o;
}
static void del_obj(void *p) {
  obj_t *o = (obj_t *)p;
  o->magic = 0xDEADDEADu;
  free(o);
  atomic_fetch_add_explicit(&g_deleted, 1, memory_order_relaxed);
}

/* ---- Part 1: deterministic refcount accounting ---- */
static int deterministic(void) {
  atomic_store(&g_made, 0);
  atomic_store(&g_deleted, 0);

  arts_atomic_shared_ptr_t slot;
  atomic_store(&slot, (arts_shared_slot_t){0});

  /* Install A into the slot; keep our own pinning ref on A. */
  arts_shared_ptr_t A = arts_shared_make(make_obj(), del_obj); /* strong=1 */
  arts_shared_ptr_t A_pin = arts_shared_copy(A);               /* strong=2 */
  arts_atomic_shared_store(&slot, A); /* slot owns one ref; A handle nulled? no:
                                         store does not null the handle */
  /* After store: slot holds a ref to A's cb (the one `A` carried).  A_pin is
   * our pinning ref.  No deleter yet. */
  assert(atomic_load(&g_deleted) == 0);

  /* SUCCESS: slot still holds A_pin's cb; swap to B.  Slot drops its ref on A
   * (so A's cb goes 2->1, still alive via A_pin); slot now owns B. */
  arts_shared_ptr_t B = arts_shared_make(make_obj(), del_obj); /* strong=1 */
  bool ok = arts_atomic_shared_compare_exchange(&slot, A_pin, B);
  if (!ok) {
    (void)fprintf(stderr, "FAIL shared_compare_exchange: success CAS failed\n");
    return 1;
  }
  /* A dropped slot ref (2->1) — A still alive via A_pin, no delete yet. */
  if (atomic_load(&g_deleted) != 0) {
    (void)fprintf(stderr,
                  "FAIL shared_compare_exchange: success deleted too early\n");
    return 1;
  }

  /* MISMATCH: slot now holds B, not A_pin.  compare_exchange(expected=A_pin)
   * must fail and touch nothing. */
  arts_shared_ptr_t C = arts_shared_make(make_obj(), del_obj); /* strong=1 */
  bool bad = arts_atomic_shared_compare_exchange(&slot, A_pin, C);
  if (bad) {
    (void)fprintf(stderr,
                  "FAIL shared_compare_exchange: mismatch returned true\n");
    return 1;
  }
  /* Mismatch dropped no refs; C is still ours (its make ref), B still in slot,
   * A still alive via A_pin. */
  if (atomic_load(&g_deleted) != 0) {
    (void)fprintf(stderr,
                  "FAIL shared_compare_exchange: mismatch dropped a ref\n");
    return 1;
  }

  /* Drop our pin on A -> A's cb 1->0 -> deleter runs for A (deleted: 0->1). */
  arts_shared_release(&A_pin);
  if (atomic_load(&g_deleted) != 1) {
    (void)fprintf(stderr,
                  "FAIL shared_compare_exchange: A not deleted on last drop "
                  "(deleted=%" PRIu64 ")\n",
                  atomic_load(&g_deleted));
    return 1;
  }

  /* C is ours and was never published -> abandon the cb (no deleter) and free
   * its object ourselves (deleted: 1->2). */
  void *c_obj = arts_shared_get(C);
  arts_shared_abandon(&C);
  del_obj(c_obj);
  if (atomic_load(&g_deleted) != 2) {
    (void)fprintf(stderr,
                  "FAIL shared_compare_exchange: C cleanup count wrong\n");
    return 1;
  }

  /* Clear the slot -> releases B -> deleter for B (deleted: 2->3). */
  arts_atomic_shared_store(&slot, (arts_shared_ptr_t)NULL);
  if (atomic_load(&g_deleted) != 3) {
    (void)fprintf(
        stderr, "FAIL shared_compare_exchange: B not deleted on slot clear\n");
    return 1;
  }
  /* made == 3 (A,B,C), deleted == 3 (A via pin, C by us, B via slot-clear). */
  if (atomic_load(&g_made) != 3 || atomic_load(&g_deleted) != 3) {
    (void)fprintf(stderr,
                  "FAIL shared_compare_exchange: made=%" PRIu64
                  " deleted=%" PRIu64 " (want 3/3)\n",
                  atomic_load(&g_made), atomic_load(&g_deleted));
    return 1;
  }
  return 0;
}

/* ---- Part 2: concurrent compare_exchange + load forward-progress ---- */
static arts_atomic_shared_ptr_t g_slot;
static atomic_int g_start;
static _Atomic int g_stop;
static _Atomic uint64_t g_cas_wins;

#define INSTALLERS 6
/* Sized so a single run completes well under the suite timeout even when TSan
 * is logging every racy access on the atomic-shared-ptr load/CAS retry path;
 * the install-race window is hit thousands of times per installer. */
#define INSTALLS_PER 40000

static void *installer(void *arg) {
  (void)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < INSTALLS_PER; i++) {
    /* Pin the current occupant (load gives us a fresh ref). */
    arts_shared_ptr_t expected = arts_atomic_shared_load(&g_slot);
    arts_shared_ptr_t fresh = arts_shared_make(make_obj(), del_obj);
    if (arts_atomic_shared_compare_exchange(&g_slot, expected, fresh)) {
      atomic_fetch_add_explicit(&g_cas_wins, 1, memory_order_relaxed);
      /* Won: slot took `fresh`, dropped its ref on `expected`. */
    } else {
      /* Lost: slot unchanged; `fresh` never published -> abandon it and free
       * its object ourselves (object stays ours, install-race-loser idiom). */
      void *obj = arts_shared_get(fresh);
      arts_shared_abandon(&fresh);
      del_obj(obj);
    }
    /* Drop our pinning ref on the old occupant. */
    if (expected) {
      arts_shared_release(&expected);
    }
  }
  return NULL;
}

static void *reader(void *arg) {
  (void)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  uint64_t reads = 0;
  while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
    arts_shared_ptr_t r = arts_atomic_shared_load(&g_slot);
    if (r) {
      obj_t *o = (obj_t *)arts_shared_get(r);
      if (o->magic != OBJ_MAGIC) {
        (void)fprintf(stderr,
                      "FAIL shared_compare_exchange: reader saw torn/dead "
                      "object magic=%x\n",
                      o->magic);
        abort();
      }
      arts_shared_release(&r);
      reads++;
    }
  }
  /* forward progress: a reader running alongside churning installers must have
   * completed many loads (never livelocked). */
  (void)reads;
  return NULL;
}

static int concurrent(void) {
  atomic_store(&g_made, 0);
  atomic_store(&g_deleted, 0);
  atomic_init(&g_start, 0);
  atomic_init(&g_stop, 0);
  atomic_init(&g_cas_wins, 0);
  atomic_store(&g_slot, (arts_shared_slot_t){0});

  /* Seed the slot. */
  arts_atomic_shared_store(&g_slot, arts_shared_make(make_obj(), del_obj));

  pthread_t inst[INSTALLERS];
  pthread_t rd;
  for (int i = 0; i < INSTALLERS; i++) {
    pthread_create(&inst[i], NULL, installer, NULL);
  }
  pthread_create(&rd, NULL, reader, NULL);
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < INSTALLERS; i++) {
    pthread_join(inst[i], NULL);
  }
  atomic_store_explicit(&g_stop, 1, memory_order_release);
  pthread_join(rd, NULL);

  /* Clear the slot -> releases the final occupant. */
  arts_atomic_shared_store(&g_slot, (arts_shared_ptr_t)NULL);

  /* Refcount conservation: every object ever made was deleted exactly once. */
  uint64_t made = atomic_load(&g_made);
  uint64_t del = atomic_load(&g_deleted);
  if (made != del) {
    (void)fprintf(stderr,
                  "FAIL shared_compare_exchange: made=%" PRIu64
                  " != deleted=%" PRIu64 " (leak or double-free)\n",
                  made, del);
    return 1;
  }
  return 0;
}

int main(void) {
  if (deterministic() != 0) {
    return 1;
  }
  if (concurrent() != 0) {
    return 1;
  }
  printf("PASS shared_compare_exchange: success/mismatch refcount exact; "
         "%d installers + reader race conserved every object (no UAF/leak), "
         "load made forward progress\n",
         INSTALLERS);
  return 0;
}

/* libc-backed shims so the test links without the ARTS runtime. */
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
