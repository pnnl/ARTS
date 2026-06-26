/* SPDX-License-Identifier: Apache-2.0
 *
 * T052 — lazy db_size learn (B030: cache->db_size non-atomic lazy learn,
 * buffer.c:63-65).
 *
 * A lazily-installed cache starts with db_size == 0; the first install with a
 * non-zero db_size "learns" the real size by writing cache->db_size = db_size
 * (only while it is still 0).  Two facets:
 *
 *  Part 1 (single-thread semantics):
 *    - First install (db_size=N) sets cache->db_size = N.
 *    - A second install (same OR different version) must NOT re-learn: it
 * leaves cache->db_size at N even if a different size is passed (the learn
 * branch is gated on db_size==0, which is now false).
 *
 *  Part 2 (concurrency — targets the SUSPECTED non-atomic learn):
 *    - K threads each perform the *first* install on a freshly-zeroed cache,
 *      all carrying the SAME db_size N.  After the join: cache->db_size == N,
 *      exactly one buffer survives in the slot, version == max, no torn write.
 *    The write `cache->db_size = db_size` is a PLAIN (non-atomic) store with no
 *    ordering vs. concurrent readers (db.c's total_size).  Because all writers
 *    write the SAME value here, the result is value-correct, but the store is a
 *    data race on the bytes.  This test is built/run under ThreadSanitizer:
 *    if TSan reports a write/write (or read/write) data race on cache->db_size,
 *    that is the genuine B030 defect surfacing — it is recorded, NOT masked.
 *
 * Standalone: links buffer.c + shared.c with libc-backed alloc shims.
 */

#include "arts/coherence/buffer.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DB_SIZE 128
#define LEARNERS 8

static struct arts_db_cache_s g_cache;
static _Atomic int g_start;

static int part1_single(void) {
  memset(&g_cache, 0, sizeof(g_cache)); /* db_size == 0 (lazy) */
  unsigned char payload[DB_SIZE];
  memset(payload, 0x11, DB_SIZE);

  /* First install learns N. */
  arts_db_buf_install(&g_cache, 1, payload, DB_SIZE);
  if (g_cache.db_size != DB_SIZE) {
    (void)fprintf(stderr,
                  "FAIL buffer_lazy_db_size_learn: first install did not learn "
                  "size (db_size=%llu)\n",
                  (unsigned long long)g_cache.db_size);
    return 1;
  }
  /* Second install with a DIFFERENT db_size must NOT re-learn (still N). The
   * buffer payload uses the new size, but cache->db_size is a one-shot learn.
   */
  unsigned char payload2[DB_SIZE * 2];
  memset(payload2, 0x22, sizeof(payload2));
  arts_db_buf_install(&g_cache, 2, payload2, DB_SIZE * 2);
  if (g_cache.db_size != DB_SIZE) {
    (void)fprintf(stderr,
                  "FAIL buffer_lazy_db_size_learn: second install re-learned "
                  "size (db_size=%llu, expected %d)\n",
                  (unsigned long long)g_cache.db_size, DB_SIZE);
    return 1;
  }
  arts_atomic_shared_store(&g_cache.buffer, NULL);
  return 0;
}

static void *learner(void *arg) {
  uint64_t base = (uint64_t)(uintptr_t)arg;
  unsigned char payload[DB_SIZE];
  memset(payload, (int)(base & 0xFF), DB_SIZE);
  while (!atomic_load_explicit(&g_start, memory_order_acquire)) {
  }
  /* A SINGLE install per thread, all racing on the very first install of a
   * freshly-zeroed cache, so each one evaluates the (cache->db_size == 0) learn
   * branch concurrently — the minimal interleaving that surfaces the non-atomic
   * learn without unbounded buffer churn. base == this thread's version, so the
   * global max version is LEARNERS. */
  arts_db_buf_install(&g_cache, base, payload, DB_SIZE);
  return NULL;
}

static int part2_race(void) {
  memset(&g_cache, 0, sizeof(g_cache)); /* db_size == 0 (lazy) */
  atomic_store_explicit(&g_start, 0, memory_order_release);

  pthread_t th[LEARNERS];
  for (int i = 0; i < LEARNERS; i++) {
    pthread_create(&th[i], NULL, learner, (void *)(uintptr_t)(i + 1));
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < LEARNERS; i++) {
    pthread_join(th[i], NULL);
  }

  if (g_cache.db_size != DB_SIZE) {
    (void)fprintf(stderr,
                  "FAIL buffer_lazy_db_size_learn: concurrent learn yielded "
                  "db_size=%llu (expected %d) — torn write?\n",
                  (unsigned long long)g_cache.db_size, DB_SIZE);
    return 1;
  }
  /* Slot holds the global max version, exactly one live buffer. */
  arts_shared_ptr_t h = arts_db_buf_acquire(&g_cache);
  struct arts_db_buffer_s *cur = (struct arts_db_buffer_s *)arts_shared_get(h);
  /* learner version == its base ∈ [1..LEARNERS]; the highest published version
   * is therefore LEARNERS (the version guard publishes strictly increasing
   * versions, so the slot ends at the global max). */
  uint64_t expected_max = (uint64_t)LEARNERS;
  int rc = (cur == NULL || cur->version != expected_max);
  if (rc) {
    (void)fprintf(stderr,
                  "FAIL buffer_lazy_db_size_learn: final version %" PRIu64
                  " != %" PRIu64 "\n",
                  cur ? cur->version : 0, expected_max);
  }
  arts_db_buf_release(&h);
  arts_atomic_shared_store(&g_cache.buffer, NULL);
  return rc;
}

int main(void) {
  if (part1_single() != 0) {
    return 1;
  }
  if (part2_race() != 0) {
    return 1;
  }
  printf(
      "PASS buffer_lazy_db_size_learn: one-shot learn + %d-thread concurrent "
      "first-install (run under TSan to surface the non-atomic learn)\n",
      LEARNERS);
  return 0;
}

/* ── libc-backed alloc shims so the test links without the ARTS runtime ── */
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
void *arts_malloc_aligned(size_t size, size_t align) {
  void *p = NULL;
  size_t a = align < sizeof(void *) ? sizeof(void *) : align;
  if (posix_memalign(&p, a, size) != 0) {
    return NULL;
  }
  return p;
}
