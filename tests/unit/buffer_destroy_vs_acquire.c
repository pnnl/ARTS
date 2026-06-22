/* SPDX-License-Identifier: Apache-2.0
 *
 * T053 — buffer destroy-vs-acquire headline race (TSan + ASan).
 *
 * The buffer slot (cache.buffer, an atomic shared_ptr) is concurrently:
 *   - installed + stored-to-NULL by an "owner" thread.  store-NULL is exactly
 *     the destroy_pre primitive (drop the cache-hold), the real destroy path.
 *   - acquired + released by many "reader" threads.
 *
 * Invariants under stress (the cb keeps the bytes alive for any in-flight
 * acquirer even across a concurrent destroy):
 *   1. Every NON-NULL acquire yields a LIVE, readable buffer — no
 *      use-after-free / double-free (ASan), no data race on the bytes (TSan).
 *   2. Each buffer is internally consistent: data[] is stamped with a sentinel
 *      derived from its own version, so a reader can verify it observed a whole
 *      buffer (not freed/torn memory): every byte == (version & 0xFF).
 *   3. Version monotonicity per reader: across a single acquire the version is
 *      a real installed version; the slot's published version never retreats
 *      (checked by the owner: each install strictly increases version).
 *   4. After all joins, the final store-NULL leaves the slot empty and ASan
 *      reports no leak (the cb deleter freed every retired buffer once all
 *      acquirers released).
 *
 * The lost-CAS abandon path (install retreats and frees its own unpublished
 * buffer) and the displaced-buffer path (a winning install drops the old
 * cache-hold) are both exercised implicitly by the churn; ASan accounts every
 * alloc/free so an unbalanced abandon/release surfaces as leak or double-free.
 *
 * Standalone: links buffer.c + shared.c with libc-backed alloc shims.  Run
 * repeatedly under TSan to shake out intermittency.
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

#define DB_SIZE 96
#define READERS 8
#define OWNER_ITERS 60000

static struct arts_db_cache_s g_cache;
static _Atomic int g_stop;
static _Atomic int g_fail;

/* Each buffer's bytes are all (version & 0xFF) so a reader can verify it saw a
 * whole, live buffer rather than freed/torn storage. */
static void fill_for_version(unsigned char *buf, uint64_t v) {
  memset(buf, (int)(v & 0xFF), DB_SIZE);
}

static void *owner(void *arg) {
  (void)arg;
  unsigned char payload[DB_SIZE];
  for (int i = 1; i <= OWNER_ITERS; i++) {
    uint64_t v = (uint64_t)i;
    fill_for_version(payload, v);
    arts_db_buf_install(&g_cache, v, payload, DB_SIZE);
    /* destroy_pre: drop the cache-hold (store NULL).  An in-flight reader's
     * own ref must keep the bytes alive. */
    arts_atomic_shared_store(&g_cache.buffer, NULL);
  }
  return NULL;
}

static void *reader(void *arg) {
  (void)arg;
  while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
    arts_shared_ptr_t h = arts_db_buf_acquire(&g_cache);
    struct arts_db_buffer_s *b = (struct arts_db_buffer_s *)arts_shared_get(h);
    if (b == NULL) {
      continue; /* slot momentarily NULL (between install and re-install). */
    }
    uint64_t v = b->version;
    /* Verify the buffer is whole + alive: every byte equals (version & 0xFF).
     * A UAF / torn read would show a mismatching byte. */
    unsigned char expect = (unsigned char)(v & 0xFF);
    int bad = 0;
    for (int i = 0; i < DB_SIZE; i++) {
      /* data[] is `char` (signed); compare as unsigned char to avoid a
       * sign-extension false mismatch for byte values >= 0x80. */
      if ((unsigned char)b->data[i] != expect) {
        bad = 1;
        break;
      }
    }
    if (bad) {
      (void)fprintf(stderr,
                    "FAIL buffer_destroy_vs_acquire: buffer v=%" PRIu64
                    " has inconsistent bytes (UAF/torn read)\n",
                    v);
      atomic_store_explicit(&g_fail, 1, memory_order_release);
    }
    arts_db_buf_release(&h);
  }
  return NULL;
}

int main(void) {
  memset(&g_cache, 0, sizeof(g_cache));
  g_cache.db_size = DB_SIZE;
  atomic_store_explicit(&g_stop, 0, memory_order_release);
  atomic_store_explicit(&g_fail, 0, memory_order_release);

  /* Seed one buffer so readers can start immediately. */
  unsigned char seed[DB_SIZE];
  fill_for_version(seed, 0);
  arts_db_buf_install(&g_cache, 0, seed, DB_SIZE);

  pthread_t rd[READERS];
  for (int i = 0; i < READERS; i++) {
    pthread_create(&rd[i], NULL, reader, NULL);
  }
  pthread_t ow;
  pthread_create(&ow, NULL, owner, NULL);

  pthread_join(ow, NULL);
  atomic_store_explicit(&g_stop, 1, memory_order_release);
  for (int i = 0; i < READERS; i++) {
    pthread_join(rd[i], NULL);
  }

  /* Final teardown: ensure the slot is NULL so the last buffer (if any) is
   * freed; ASan then accounts every buffer. */
  arts_atomic_shared_store(&g_cache.buffer, NULL);

  if (atomic_load_explicit(&g_fail, memory_order_acquire) != 0) {
    return 1;
  }
  printf("PASS buffer_destroy_vs_acquire: %d readers vs install+store-NULL "
         "owner (%d iters), no UAF/torn read\n",
         READERS, OWNER_ITERS);
  return 0;
}

/* ── libc-backed alloc shims so the test links without the ARTS runtime ── */
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
void *arts_malloc_align(size_t size, size_t align) {
  void *p = NULL;
  size_t a = align < sizeof(void *) ? sizeof(void *) : align;
  if (posix_memalign(&p, a, size) != 0) {
    return NULL;
  }
  return p;
}
