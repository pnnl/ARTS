/* SPDX-License-Identifier: Apache-2.0
 *
 * T050 — zero-size DB buffer install (ASan-focused).
 *
 * A DB may legitimately carry db_size == 0 (e.g. an empty / placeholder DB,
 * the coherence_zero_size_xfer integration path).  arts_db_buf_install with
 * db_size == 0 must:
 *   - return a NON-NULL buffer (the header is always allocated even with no
 *     FAM payload),
 *   - stamp the requested version,
 *   - never touch data[] (no memcpy / memset of 0 bytes — but more importantly,
 *     never read/write past the header),
 *   - work for BOTH a NULL payload and a non-NULL payload (with db_size 0 the
 *     payload pointer must be ignored, not dereferenced).
 *
 * The acquire/release/teardown path must be clean (no leak, no double free, no
 * OOB) under ASan.  buf_from_data on a zero-size buffer's data[] must still
 * recover the buffer.
 *
 * Also checks: when the cache starts db_size==0 and a zero-size install runs,
 * the lazy db_size-learn branch is NOT taken (it is gated on db_size > 0), so
 * cache->db_size stays 0 — documents that a zero-size install does not falsely
 * "learn" a size.
 *
 * Standalone: links buffer.c + shared.c with libc-backed alloc shims.
 */

#include "arts/coherence/buffer.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct arts_db_cache_s g_cache;

static void cache_init(uint64_t db_size) {
  memset(&g_cache, 0, sizeof(g_cache));
  g_cache.db_size = db_size;
}
static void cache_teardown(void) {
  arts_atomic_shared_store(&g_cache.buffer, NULL);
}

static int check_zero_install(uint64_t v, const void *payload,
                              const char *label) {
  struct arts_db_buffer_s *b = arts_db_buf_install(&g_cache, v, payload, 0);
  if (b == NULL) {
    (void)fprintf(stderr, "FAIL buffer_zero_size: %s install returned NULL\n",
                  label);
    return 1;
  }
  if (b->version != v) {
    (void)fprintf(stderr, "FAIL buffer_zero_size: %s version %llu != %llu\n",
                  label, (unsigned long long)b->version, (unsigned long long)v);
    return 1;
  }
  arts_shared_ptr_t h = arts_db_buf_acquire(&g_cache);
  struct arts_db_buffer_s *got = (struct arts_db_buffer_s *)arts_shared_get(h);
  int rc = 0;
  if (got != b) {
    (void)fprintf(stderr, "FAIL buffer_zero_size: %s acquire != install\n",
                  label);
    rc = 1;
  } else if (arts_db_buf_from_data(got->data) != got) {
    (void)fprintf(stderr, "FAIL buffer_zero_size: %s from_data(data) != buf\n",
                  label);
    rc = 1;
  }
  arts_db_buf_release(&h);
  return rc;
}

int main(void) {
  /* Case A: cache db_size 0, NULL payload, zero-size install. */
  cache_init(0);
  if (check_zero_install(1, NULL, "NULL-payload") != 0) {
    return 1;
  }
  if (g_cache.db_size != 0) {
    (void)fprintf(stderr,
                  "FAIL buffer_zero_size: zero install spuriously learned "
                  "db_size=%llu (expected 0)\n",
                  (unsigned long long)g_cache.db_size);
    return 1;
  }
  cache_teardown();

  /* Case B: cache db_size 0, NON-NULL payload but db_size 0 — payload must be
   * ignored, never dereferenced. */
  cache_init(0);
  unsigned char dummy[16];
  memset(dummy, 0x5A, sizeof(dummy));
  if (check_zero_install(2, dummy, "nonNULL-payload") != 0) {
    return 1;
  }
  cache_teardown();

  /* Case C: a few zero-size installs in a row at increasing versions — every
   * acquire is clean. */
  cache_init(0);
  for (uint64_t v = 1; v <= 5; v++) {
    if (check_zero_install(v, (v & 1) ? NULL : dummy, "seq") != 0) {
      return 1;
    }
  }
  arts_shared_ptr_t h = arts_db_buf_acquire(&g_cache);
  struct arts_db_buffer_s *cur = (struct arts_db_buffer_s *)arts_shared_get(h);
  int rc = (cur == NULL || cur->version != 5);
  if (rc) {
    (void)fprintf(stderr, "FAIL buffer_zero_size: seq final version %llu\n",
                  cur ? (unsigned long long)cur->version : 0ULL);
  }
  arts_db_buf_release(&h);
  cache_teardown();
  if (rc) {
    return 1;
  }

  printf("PASS buffer_zero_size: NULL + non-NULL payload, version stamp, "
         "no false size-learn, clean acquire/teardown\n");
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
