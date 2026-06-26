/* SPDX-License-Identifier: Apache-2.0
 *
 * ABA stress for arts_shared_ptr_t: a churner thread rapidly installs and
 * tears down control blocks on a single slot (NULL → cb → NULL), forcing
 * the global cb pool to recycle the same cb pointers back into the same
 * slot.  Concurrent readers load + validate + release.  The load's
 * acquire-and-validate (strong-inc then slot revalidation) must never hand
 * back a recycled cb that no longer belongs to the slot — AddressSanitizer
 * catches any resulting use-after-free, and the make/delete tally must
 * balance.
 *
 * Standalone — links shared.c directly with libc-backed alloc shims.
 */

#include "arts/utils/shared.h"

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define READERS 6
#define CHURN_ITERS 500000
#define READ_ITERS 500000
#define OBJ_MAGIC 0x5A5A5A5AU

typedef struct {
  uint32_t magic;
} obj_t;

static arts_atomic_shared_ptr_t g_slot;
static _Atomic(uint64_t) g_makes;
static _Atomic(uint64_t) g_deletes;
static _Atomic(uint64_t) g_hits;
static _Atomic(int) g_churn_done;

static void obj_deleter(void *p) {
  obj_t *o = (obj_t *)p;
  assert(o->magic == OBJ_MAGIC);
  o->magic = 0xDEADBEEFU;
  atomic_fetch_add_explicit(&g_deletes, 1, memory_order_relaxed);
  free(o);
}

static void *churner(void *arg) {
  (void)arg;
  for (int i = 0; i < CHURN_ITERS; ++i) {
    obj_t *o = (obj_t *)malloc(sizeof(obj_t));
    o->magic = OBJ_MAGIC;
    atomic_fetch_add_explicit(&g_makes, 1, memory_order_relaxed);
    arts_shared_ptr_t cb = arts_shared_make(o, obj_deleter);
    arts_atomic_shared_store(&g_slot, cb);
    /* Immediately tear it down so the cb recycles into the pool while
     * readers may still be mid-load on the same slot/pointer. */
    arts_shared_ptr_t old =
        arts_atomic_shared_exchange(&g_slot, (arts_shared_ptr_t)NULL);
    if (old) {
      arts_shared_release(&old);
    }
  }
  atomic_store_explicit(&g_churn_done, 1, memory_order_release);
  return NULL;
}

static void *reader(void *arg) {
  (void)arg;
  for (int i = 0; i < READ_ITERS; ++i) {
    arts_shared_ptr_t local = arts_atomic_shared_load(&g_slot);
    if (local) {
      obj_t *o = (obj_t *)arts_shared_get(local);
      assert(o->magic == OBJ_MAGIC); /* recycled-but-stale would trip here */
      atomic_fetch_add_explicit(&g_hits, 1, memory_order_relaxed);
      arts_shared_release(&local);
    }
    if (atomic_load_explicit(&g_churn_done, memory_order_acquire) &&
        i > READ_ITERS / 4) {
      break; /* churner finished; readers can stop early */
    }
  }
  return NULL;
}

int main(void) {
  atomic_store(&g_slot, (arts_shared_slot_t){0}); /* empty slot {NULL, ext=0} */

  pthread_t ch;
  pthread_t rd[READERS];
  for (int i = 0; i < READERS; ++i) {
    pthread_create(&rd[i], NULL, reader, NULL);
  }
  pthread_create(&ch, NULL, churner, NULL);

  pthread_join(ch, NULL);
  for (int i = 0; i < READERS; ++i) {
    pthread_join(rd[i], NULL);
  }

  arts_atomic_shared_store(&g_slot, (arts_shared_ptr_t)NULL);
  assert(arts_atomic_shared_load(&g_slot) == NULL);

  uint64_t makes = atomic_load(&g_makes);
  uint64_t deletes = atomic_load(&g_deletes);
  printf("shared_aba: makes=%llu deletes=%llu reader_hits=%llu\n",
         (unsigned long long)makes, (unsigned long long)deletes,
         (unsigned long long)atomic_load(&g_hits));
  assert(makes == (uint64_t)CHURN_ITERS);
  assert(deletes == makes); /* exactly one destroy per object */
  printf("shared_aba: OK\n");
  return 0;
}

void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
