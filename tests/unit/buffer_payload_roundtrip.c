/* SPDX-License-Identifier: Apache-2.0
 *
 * T049 — buffer install/acquire byte-exact round-trip + from_data identity.
 *
 * Property: arts_db_buf_install(cache, v, payload, db_size) publishes a buffer
 * whose data[] holds an exact copy of `payload` for db_size bytes, stamped with
 * version v; a subsequent arts_db_buf_acquire returns that same buffer and the
 * bytes survive verbatim.  Several distinct byte patterns are installed at
 * strictly increasing versions; after each, an acquire must observe the new
 * pattern (not a stale one) and the matching version.
 *
 * Also exercises arts_db_buf_from_data:
 *   - from_data(buf->data) == buf for an installed buffer (pointer arithmetic
 *     recovers the enclosing buffer).
 *   - from_data(NULL) == NULL.
 *
 * Single-threaded; the value is the byte-exactness of the memcpy publish path
 * and the FAM offset, which the existing coherence_buffer_test does not assert.
 * Run under ASan to catch any over/under-copy of the payload.
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

#define DB_SIZE 200

static struct arts_db_cache_s g_cache;

static void cache_init(uint64_t db_size) {
  memset(&g_cache, 0, sizeof(g_cache));
  g_cache.db_size = db_size;
}
static void cache_teardown(void) {
  arts_atomic_shared_store(&g_cache.buffer, NULL);
}

/* Install `payload` at version v, then acquire and assert byte-exact + version.
 * Returns 0 on success. */
static int install_check(uint64_t v, const unsigned char *payload) {
  struct arts_db_buffer_s *b =
      arts_db_buf_install(&g_cache, v, payload, DB_SIZE);
  if (b == NULL) {
    (void)fprintf(stderr,
                  "FAIL buffer_payload_roundtrip: install v=%llu NULL\n",
                  (unsigned long long)v);
    return 1;
  }
  if (b->version != v) {
    (void)fprintf(stderr,
                  "FAIL buffer_payload_roundtrip: version %llu != %llu\n",
                  (unsigned long long)b->version, (unsigned long long)v);
    return 1;
  }
  arts_shared_ptr_t h = arts_db_buf_acquire(&g_cache);
  struct arts_db_buffer_s *got = (struct arts_db_buffer_s *)arts_shared_get(h);
  int rc = 0;
  if (got != b) {
    (void)fprintf(stderr,
                  "FAIL buffer_payload_roundtrip: acquire != install\n");
    rc = 1;
  } else if (memcmp(got->data, payload, DB_SIZE) != 0) {
    (void)fprintf(stderr,
                  "FAIL buffer_payload_roundtrip: bytes differ at v=%llu\n",
                  (unsigned long long)v);
    rc = 1;
  }
  /* from_data identity from the live data pointer. */
  if (rc == 0 && arts_db_buf_from_data(got->data) != got) {
    (void)fprintf(stderr,
                  "FAIL buffer_payload_roundtrip: from_data(data) != buf\n");
    rc = 1;
  }
  arts_db_buf_release(&h);
  return rc;
}

int main(void) {
  cache_init(DB_SIZE);

  /* from_data(NULL) == NULL (before any install). */
  if (arts_db_buf_from_data(NULL) != NULL) {
    (void)fprintf(stderr, "FAIL buffer_payload_roundtrip: from_data(NULL)\n");
    return 1;
  }

  unsigned char pat[4][DB_SIZE];
  for (int p = 0; p < 4; p++) {
    for (int i = 0; i < DB_SIZE; i++) {
      pat[p][i] = (unsigned char)((i * 31 + p * 97 + 1) & 0xFF);
    }
  }

  /* Install at strictly increasing versions; each must publish its own bytes.
   */
  uint64_t versions[4] = {1, 2, 5, 100};
  for (int p = 0; p < 4; p++) {
    if (install_check(versions[p], pat[p]) != 0) {
      return 1;
    }
  }

  /* A stale install (lower version) must retreat and NOT overwrite the bytes:
   * acquire still sees the last (highest-version) pattern. */
  unsigned char other[DB_SIZE];
  memset(other, 0xAB, DB_SIZE);
  struct arts_db_buffer_s *retreat =
      arts_db_buf_install(&g_cache, 3, other, DB_SIZE);
  arts_shared_ptr_t h = arts_db_buf_acquire(&g_cache);
  struct arts_db_buffer_s *cur = (struct arts_db_buffer_s *)arts_shared_get(h);
  int rc = 0;
  if (cur->version != 100 || memcmp(cur->data, pat[3], DB_SIZE) != 0 ||
      retreat != cur) {
    (void)fprintf(stderr,
                  "FAIL buffer_payload_roundtrip: stale install corrupted "
                  "bytes/version (v=%llu)\n",
                  (unsigned long long)cur->version);
    rc = 1;
  }
  arts_db_buf_release(&h);
  cache_teardown();
  if (rc != 0) {
    return 1;
  }

  printf("PASS buffer_payload_roundtrip: 4 patterns, byte-exact, "
         "from_data identity, stale retreat preserves bytes\n");
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
