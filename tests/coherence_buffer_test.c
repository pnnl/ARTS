/* SPDX-License-Identifier: Apache-2.0
 *
 * Stress + correctness test for the buffer lifecycle primitives:
 *   arts_coherence_buffer_alloc / install_buffer / acquire_buf / release_buf.
 *
 * Phase 1 (basic): single-thread install+acquire+release.  Verifies
 *   ref_count walks 1→2→1→0 and the buffer ends up recycled.
 *
 * Phase 2 (acquire/release race): N threads churning acquire+release
 *   on a single installed buffer; ref_count must never go negative
 *   and must return to 1 (sentinel) when all threads finish.
 *
 * Phase 3 (install race): N installer threads each push a buffer at
 *   a unique higher version; final cache.buffer.version must equal
 *   the maximum installed version.  Stale installs (lower version)
 *   must retreat without corrupting state.
 *
 * Phase 4 (mixed): installers + readers concurrently.  Final state:
 *   one buffer with sentinel ref_count == 1, version == max installed.
 */

#include "arts/memory/coherence_buffer.h"
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
  arts_lockfree_stack_init(&cache->buffer_pool);
  cache->db_size = DB_SIZE;
}

static int phase1_basic(void) {
  cache_init(&g_cache);
  /* Install version 1, zero-init. */
  struct arts_db_buffer_s *b1 =
      arts_coherence_install_buffer(&g_cache, 1, NULL, DB_SIZE);
  if (b1 == NULL || b1->version != 1 || b1->ref_count != 1) {
    fprintf(stderr, "Phase 1 install: bad state\n");
    return 1;
  }
  /* Acquire — bumps ref_count to 2. */
  struct arts_db_buffer_s *got = arts_coherence_acquire_buf(&g_cache);
  if (got != b1 || b1->ref_count != 2) {
    fprintf(stderr, "Phase 1 acquire: ref_count=%u (expected 2)\n",
            b1->ref_count);
    return 1;
  }
  /* Release — back to 1 (sentinel only). */
  arts_coherence_release_buf(&g_cache, got);
  if (b1->ref_count != 1) {
    fprintf(stderr, "Phase 1 release: ref_count=%u (expected 1)\n",
            b1->ref_count);
    return 1;
  }

  /* Install a newer version 2, displacing v1.  v1's sentinel goes
   * away; ref_count → 0; buffer pushed to pool.  v2 should now be
   * cache.buffer with ref_count = 1. */
  struct arts_db_buffer_s *b2 =
      arts_coherence_install_buffer(&g_cache, 2, NULL, DB_SIZE);
  if (b2 == b1 || b2->version != 2 || b2->ref_count != 1) {
    fprintf(stderr, "Phase 1 v2 install: bad state\n");
    return 1;
  }
  /* Stale install (version 1) should retreat. */
  struct arts_db_buffer_s *retreat =
      arts_coherence_install_buffer(&g_cache, 1, NULL, DB_SIZE);
  if (retreat != b2) {
    fprintf(stderr, "Phase 1 stale install: did not retreat\n");
    return 1;
  }
  if (g_cache.buffer != (volatile struct arts_db_buffer_s *)b2) {
    fprintf(stderr, "Phase 1 stale install: cache.buffer changed\n");
    return 1;
  }
  /* Drain pool / current buffer manually for cleanup. */
  arts_lockfree_stack_node_t *l;
  arts_coherence_release_buf(&g_cache, b2); /* sentinel → 0 → recycle */
  while ((l = arts_lockfree_stack_pop(&g_cache.buffer_pool)) != NULL) {
    arts_free((struct arts_db_buffer_s *)l);
  }

  printf("PASS phase1_basic\n");
  return 0;
}

static void *phase2_reader(void *arg) {
  (void)arg;
  for (int i = 0; i < READER_ITERS; i++) {
    struct arts_db_buffer_s *b = arts_coherence_acquire_buf(&g_cache);
    if (b == NULL) {
      fprintf(stderr, "Phase 2 reader: acquire returned NULL\n");
      _Exit(2);
    }
    /* Touch payload to ensure the buffer is real. */
    volatile uint64_t v = b->version;
    (void)v;
    arts_coherence_release_buf(&g_cache, b);
  }
  return NULL;
}

static int phase2_acquire_release(void) {
  cache_init(&g_cache);
  arts_coherence_install_buffer(&g_cache, 1, NULL, DB_SIZE);

  pthread_t threads[READERS];
  for (int i = 0; i < READERS; i++) {
    pthread_create(&threads[i], NULL, phase2_reader, NULL);
  }
  for (int i = 0; i < READERS; i++) {
    pthread_join(threads[i], NULL);
  }
  /* All readers finished; ref_count must be back to 1 (sentinel). */
  struct arts_db_buffer_s *cur = (struct arts_db_buffer_s *)g_cache.buffer;
  if (cur == NULL || cur->ref_count != 1) {
    fprintf(stderr, "Phase 2 final: cur=%p ref_count=%u\n", (void *)cur,
            cur ? cur->ref_count : 0);
    return 1;
  }
  arts_coherence_release_buf(&g_cache, cur);
  arts_lockfree_stack_node_t *l;
  while ((l = arts_lockfree_stack_pop(&g_cache.buffer_pool)) != NULL) {
    arts_free((struct arts_db_buffer_s *)l);
  }
  printf("PASS phase2_acquire_release: %d threads × %d iters\n", READERS,
         READER_ITERS);
  return 0;
}

static void *phase3_installer(void *arg) {
  uint64_t base = (uint64_t)(uintptr_t)arg;
  for (int i = 0; i < INSTALL_ITERS; i++) {
    /* Each installer pushes monotonically-increasing versions
     * within its own range, so the global max is across all
     * installers' last attempts. */
    uint64_t v = base * (uint64_t)INSTALL_ITERS + (uint64_t)(i + 1);
    arts_coherence_install_buffer(&g_cache, v, NULL, DB_SIZE);
  }
  return NULL;
}

static int phase3_install_race(void) {
  cache_init(&g_cache);
  /* Seed with version 0 so first install always wins. */
  arts_coherence_install_buffer(&g_cache, 0, NULL, DB_SIZE);

  pthread_t threads[INSTALLERS];
  for (int i = 0; i < INSTALLERS; i++) {
    pthread_create(&threads[i], NULL, phase3_installer,
                   (void *)(uintptr_t)(i + 1));
  }
  for (int i = 0; i < INSTALLERS; i++) {
    pthread_join(threads[i], NULL);
  }

  struct arts_db_buffer_s *cur = (struct arts_db_buffer_s *)g_cache.buffer;
  /* Expected max version: max over all (base * INSTALL_ITERS + INSTALL_ITERS)
   * for base in 1..INSTALLERS = INSTALLERS * INSTALL_ITERS. */
  /* Each installer i ∈ [1, INSTALLERS] pushes versions
   * (i*INSTALL_ITERS + 1) … (i*INSTALL_ITERS + INSTALL_ITERS).  The
   * global max is INSTALLERS*INSTALL_ITERS + INSTALL_ITERS. */
  uint64_t expected_max = (uint64_t)(INSTALLERS + 1) * (uint64_t)INSTALL_ITERS;
  if (cur == NULL || cur->version != expected_max) {
    fprintf(stderr,
            "Phase 3 install race: expected version %" PRIu64 ", got %" PRIu64
            "\n",
            expected_max, cur ? cur->version : 0);
    return 1;
  }
  if (cur->ref_count != 1) {
    fprintf(stderr, "Phase 3 install race: ref_count=%u (expected 1)\n",
            cur->ref_count);
    return 1;
  }
  arts_coherence_release_buf(&g_cache, cur);
  arts_lockfree_stack_node_t *l;
  while ((l = arts_lockfree_stack_pop(&g_cache.buffer_pool)) != NULL) {
    arts_free((struct arts_db_buffer_s *)l);
  }
  printf("PASS phase3_install_race: %d installers × %d iters\n", INSTALLERS,
         INSTALL_ITERS);
  return 0;
}

static void *phase4_reader(void *arg) {
  (void)arg;
  while (!atomic_load_explicit(&g_phase4_stop, memory_order_acquire)) {
    struct arts_db_buffer_s *b = arts_coherence_acquire_buf(&g_cache);
    if (b == NULL) {
      continue; /* transient — installer between exchange and CAS. */
    }
    volatile uint64_t v = b->version;
    (void)v;
    arts_coherence_release_buf(&g_cache, b);
  }
  return NULL;
}

static int phase4_mixed(void) {
  cache_init(&g_cache);
  arts_coherence_install_buffer(&g_cache, 0, NULL, DB_SIZE);
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

  struct arts_db_buffer_s *cur = (struct arts_db_buffer_s *)g_cache.buffer;
  /* Each installer i ∈ [1, INSTALLERS] pushes versions
   * (i*INSTALL_ITERS + 1) … (i*INSTALL_ITERS + INSTALL_ITERS).  The
   * global max is INSTALLERS*INSTALL_ITERS + INSTALL_ITERS. */
  uint64_t expected_max = (uint64_t)(INSTALLERS + 1) * (uint64_t)INSTALL_ITERS;
  if (cur == NULL || cur->version != expected_max || cur->ref_count != 1) {
    fprintf(stderr,
            "Phase 4 mixed final: cur=%p version=%" PRIu64 " ref_count=%u\n",
            (void *)cur, cur ? cur->version : 0, cur ? cur->ref_count : 0);
    return 1;
  }
  arts_coherence_release_buf(&g_cache, cur);
  arts_lockfree_stack_node_t *l;
  while ((l = arts_lockfree_stack_pop(&g_cache.buffer_pool)) != NULL) {
    arts_free((struct arts_db_buffer_s *)l);
  }
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
