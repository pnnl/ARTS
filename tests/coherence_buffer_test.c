/* SPDX-License-Identifier: Apache-2.0
 *
 * Stress + correctness test for the buffer lifecycle primitives:
 *   arts_coh_buffer_alloc / install_buffer / acquire_buf / release_buf.
 *
 * The buffer is managed as an arts_shared_ptr_t (cache.buffer is the atomic
 * slot; each acquirer holds a strong ref; the cb deleter frees on last drop),
 * so these phases exercise the shared-ptr lifecycle behaviorally — identity,
 * version monotonicity, and crash/leak-freedom under concurrency — rather than
 * poking a raw refcount.  Run under ASan/TSan to catch UAF / leaks / races.
 *
 * Phase 1 (basic): single-thread install + acquire + release + version-
 *   conditional re-install (stale install must retreat).
 * Phase 2 (acquire/release race): N threads churning acquire+release on a
 *   single installed buffer; every acquire must return a live buffer.
 * Phase 3 (install race): N installers each push monotonically-increasing
 *   versions; final cache.buffer.version must equal the global max.
 * Phase 4 (mixed): installers + readers concurrently; final version == max.
 */

#include "arts/memory/coherence_buffer.h"
#include "arts/utils/shared.h"
#include "arts/utils/malloc.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DB_SIZE 64
#define READERS 8
#define READER_ITERS 50000
#define INSTALLERS 4
#define INSTALL_ITERS 200

static struct arts_db_cache_s g_cache;
static _Atomic int g_phase4_stop;

static void cache_init(struct arts_db_cache_s *cache) {
  memset(cache, 0, sizeof(*cache));
  cache->db_size = DB_SIZE;
}

/* Release the cache-hold (slot sentinel) — frees the buffer once no acquirer
 * holds a ref, matching arts_coh_cache_destructor's teardown. */
static void cache_teardown(struct arts_db_cache_s *cache) {
  arts_atomic_shared_store(&cache->buffer, NULL);
}

static int phase1_basic(void) {
  cache_init(&g_cache);
  /* Install version 1, zero-init. */
  struct arts_db_buffer_s *b1 =
      arts_coh_install_buffer(&g_cache, 1, NULL, DB_SIZE);
  if (b1 == NULL || b1->version != 1 ||
      arts_coh_buffer_peek(&g_cache) != b1) {
    fprintf(stderr, "Phase 1 install: bad state\n");
    return 1;
  }
  /* Acquire — returns a live ref to the same buffer. */
  arts_shared_ptr_t h = arts_coh_acquire_buf(&g_cache);
  struct arts_db_buffer_s *got = (struct arts_db_buffer_s *)arts_shared_get(h);
  if (got != b1) {
    fprintf(stderr, "Phase 1 acquire: got %p (expected %p)\n", (void *)got,
            (void *)b1);
    return 1;
  }
  /* Release the acquire ref; the slot still holds the cache-hold. */
  arts_coh_release_buf(&h);
  if (arts_coh_buffer_peek(&g_cache) != b1) {
    fprintf(stderr, "Phase 1 release: buffer changed\n");
    return 1;
  }

  /* Install a newer version 2, displacing v1.  v1's cache-hold drops; its cb
   * deleter frees it once no acquirer holds a ref. */
  struct arts_db_buffer_s *b2 =
      arts_coh_install_buffer(&g_cache, 2, NULL, DB_SIZE);
  if (b2 == b1 || b2->version != 2 ||
      arts_coh_buffer_peek(&g_cache) != b2) {
    fprintf(stderr, "Phase 1 v2 install: bad state\n");
    return 1;
  }
  /* Stale install (version 1) must retreat and leave b2 in place. */
  struct arts_db_buffer_s *retreat =
      arts_coh_install_buffer(&g_cache, 1, NULL, DB_SIZE);
  if (retreat != b2 || arts_coh_buffer_peek(&g_cache) != b2) {
    fprintf(stderr, "Phase 1 stale install: did not retreat\n");
    return 1;
  }
  cache_teardown(&g_cache);
  printf("PASS phase1_basic\n");
  return 0;
}

static void *phase2_reader(void *arg) {
  (void)arg;
  for (int i = 0; i < READER_ITERS; i++) {
    arts_shared_ptr_t h = arts_coh_acquire_buf(&g_cache);
    struct arts_db_buffer_s *b = (struct arts_db_buffer_s *)arts_shared_get(h);
    if (b == NULL) {
      fprintf(stderr, "Phase 2 reader: acquire returned NULL\n");
      _Exit(2);
    }
    /* Touch payload to ensure the buffer is real + alive. */
    volatile uint64_t v = b->version;
    (void)v;
    arts_coh_release_buf(&h);
  }
  return NULL;
}

static int phase2_acquire_release(void) {
  cache_init(&g_cache);
  arts_coh_install_buffer(&g_cache, 1, NULL, DB_SIZE);

  pthread_t threads[READERS];
  for (int i = 0; i < READERS; i++) {
    pthread_create(&threads[i], NULL, phase2_reader, NULL);
  }
  for (int i = 0; i < READERS; i++) {
    pthread_join(threads[i], NULL);
  }
  /* All readers finished; the installed buffer is still live (sentinel). */
  struct arts_db_buffer_s *cur = arts_coh_buffer_peek(&g_cache);
  if (cur == NULL || cur->version != 1) {
    fprintf(stderr, "Phase 2 final: cur=%p\n", (void *)cur);
    return 1;
  }
  cache_teardown(&g_cache);
  printf("PASS phase2_acquire_release: %d threads × %d iters\n", READERS,
         READER_ITERS);
  return 0;
}

static void *phase3_installer(void *arg) {
  uint64_t base = (uint64_t)(uintptr_t)arg;
  for (int i = 0; i < INSTALL_ITERS; i++) {
    uint64_t v = base * (uint64_t)INSTALL_ITERS + (uint64_t)(i + 1);
    arts_coh_install_buffer(&g_cache, v, NULL, DB_SIZE);
  }
  return NULL;
}

static int phase3_install_race(void) {
  cache_init(&g_cache);
  arts_coh_install_buffer(&g_cache, 0, NULL, DB_SIZE);

  pthread_t threads[INSTALLERS];
  for (int i = 0; i < INSTALLERS; i++) {
    pthread_create(&threads[i], NULL, phase3_installer,
                   (void *)(uintptr_t)(i + 1));
  }
  for (int i = 0; i < INSTALLERS; i++) {
    pthread_join(threads[i], NULL);
  }

  struct arts_db_buffer_s *cur = arts_coh_buffer_peek(&g_cache);
  /* Each installer i ∈ [1, INSTALLERS] pushes versions
   * (i*INSTALL_ITERS + 1) … (i*INSTALL_ITERS + INSTALL_ITERS); global max is
   * (INSTALLERS+1)*INSTALL_ITERS. */
  uint64_t expected_max = (uint64_t)(INSTALLERS + 1) * (uint64_t)INSTALL_ITERS;
  if (cur == NULL || cur->version != expected_max) {
    fprintf(stderr,
            "Phase 3 install race: expected version %" PRIu64 ", got %" PRIu64
            "\n",
            expected_max, cur ? cur->version : 0);
    return 1;
  }
  cache_teardown(&g_cache);
  printf("PASS phase3_install_race: %d installers × %d iters\n", INSTALLERS,
         INSTALL_ITERS);
  return 0;
}

static void *phase4_reader(void *arg) {
  (void)arg;
  while (!atomic_load_explicit(&g_phase4_stop, memory_order_acquire)) {
    arts_shared_ptr_t h = arts_coh_acquire_buf(&g_cache);
    struct arts_db_buffer_s *b = (struct arts_db_buffer_s *)arts_shared_get(h);
    if (b == NULL) {
      continue; /* transient — installer mid-publish. */
    }
    volatile uint64_t v = b->version;
    (void)v;
    arts_coh_release_buf(&h);
  }
  return NULL;
}

static int phase4_mixed(void) {
  cache_init(&g_cache);
  arts_coh_install_buffer(&g_cache, 0, NULL, DB_SIZE);
  atomic_init(&g_phase4_stop, 0);

  pthread_t readers[READERS];
  pthread_t installers[INSTALLERS];
  for (int i = 0; i < READERS; i++) {
    pthread_create(&readers[i], NULL, phase4_reader, NULL);
  }
  for (int i = 0; i < INSTALLERS; i++) {
    pthread_create(&installers[i], NULL, phase3_installer,
                   (void *)(uintptr_t)(i + 1));
  }
  for (int i = 0; i < INSTALLERS; i++) {
    pthread_join(installers[i], NULL);
  }
  atomic_store_explicit(&g_phase4_stop, 1, memory_order_release);
  for (int i = 0; i < READERS; i++) {
    pthread_join(readers[i], NULL);
  }

  struct arts_db_buffer_s *cur = arts_coh_buffer_peek(&g_cache);
  uint64_t expected_max = (uint64_t)(INSTALLERS + 1) * (uint64_t)INSTALL_ITERS;
  if (cur == NULL || cur->version != expected_max) {
    fprintf(stderr, "Phase 4 mixed final: cur=%p version=%" PRIu64 "\n",
            (void *)cur, cur ? cur->version : 0);
    return 1;
  }
  cache_teardown(&g_cache);
  printf("PASS phase4_mixed: %d readers + %d installers concurrently\n",
         READERS, INSTALLERS);
  return 0;
}

int main(void) {
  if (phase1_basic() != 0)
    return 1;
  if (phase2_acquire_release() != 0)
    return 1;
  if (phase3_install_race() != 0)
    return 1;
  if (phase4_mixed() != 0)
    return 1;
  printf("ALL PASS\n");
  return 0;
}
