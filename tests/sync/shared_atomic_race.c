/* SPDX-License-Identifier: Apache-2.0
 *
 * Concurrency invariants for the arts_shared_ptr_t atomic slot API:
 * N producers repeatedly publish fresh control blocks into one shared slot
 * while M consumers repeatedly load + dereference + release.  The managed
 * object is heap-allocated and freed by the deleter, so AddressSanitizer
 * flags any use-after-free or double-free, and the make/delete tallies must
 * match exactly at quiescence (no leak, no over-release).
 *
 * Standalone — links shared.c directly with libc-backed alloc shims.
 */

#include "arts/sync/shared.h"

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PRODUCERS 4
#define CONSUMERS 4
#define ITERS 200000

#define OBJ_MAGIC 0xC0FFEEU

typedef struct {
  uint32_t magic;
  uint32_t payload;
} obj_t;

static arts_atomic_shared_ptr_t g_slot;
static _Atomic(uint64_t) g_makes;
static _Atomic(uint64_t) g_deletes;
static _Atomic(uint64_t) g_sink; /* keeps consumer loads observable */

static void obj_deleter(void *p) {
  obj_t *o = (obj_t *)p;
  assert(o->magic == OBJ_MAGIC); /* not freed/torn */
  o->magic = 0xDEADU;
  atomic_fetch_add_explicit(&g_deletes, 1, memory_order_relaxed);
  free(o);
}

static void *producer(void *arg) {
  (void)arg;
  for (int i = 0; i < ITERS; ++i) {
    obj_t *o = (obj_t *)malloc(sizeof(obj_t));
    o->magic = OBJ_MAGIC;
    o->payload = (uint32_t)i;
    atomic_fetch_add_explicit(&g_makes, 1, memory_order_relaxed);
    arts_shared_ptr_t cb = arts_shared_make(o, obj_deleter);
    arts_atomic_shared_store(&g_slot, cb); /* releases prior occupant */
  }
  return NULL;
}

static void *consumer(void *arg) {
  (void)arg;
  uint64_t seen = 0;
  for (int i = 0; i < ITERS; ++i) {
    arts_shared_ptr_t local = arts_atomic_shared_load(&g_slot);
    if (local) {
      obj_t *o = (obj_t *)arts_shared_get(local);
      /* Holding a ref keeps the object alive: the read must be valid. */
      assert(o->magic == OBJ_MAGIC);
      seen += o->payload;
      arts_shared_release(&local);
    }
  }
  /* Publish seen so the compiler can't elide the loads. */
  atomic_fetch_add_explicit(&g_sink, seen, memory_order_relaxed);
  return NULL;
}

int main(void) {
  atomic_store(&g_slot, (arts_shared_ptr_t)NULL);

  pthread_t prod[PRODUCERS], cons[CONSUMERS];
  for (int i = 0; i < CONSUMERS; ++i)
    pthread_create(&cons[i], NULL, consumer, NULL);
  for (int i = 0; i < PRODUCERS; ++i)
    pthread_create(&prod[i], NULL, producer, NULL);

  for (int i = 0; i < PRODUCERS; ++i)
    pthread_join(prod[i], NULL);
  for (int i = 0; i < CONSUMERS; ++i)
    pthread_join(cons[i], NULL);

  /* Drop the final occupant. */
  arts_atomic_shared_store(&g_slot, (arts_shared_ptr_t)NULL);
  assert(arts_atomic_shared_load(&g_slot) == NULL);

  uint64_t makes = atomic_load(&g_makes);
  uint64_t deletes = atomic_load(&g_deletes);
  printf("shared_atomic_race: makes=%llu deletes=%llu\n",
         (unsigned long long)makes, (unsigned long long)deletes);
  assert(makes == (uint64_t)PRODUCERS * ITERS);
  assert(deletes == makes); /* every object destroyed exactly once */
  printf("shared_atomic_race: OK\n");
  return 0;
}

void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
