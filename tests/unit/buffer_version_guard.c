/* SPDX-License-Identifier: Apache-2.0
 *
 * T051 — buffer install version guard ('>=' retreat) monotonicity (B031:
 * same-version distinct-bytes silent loss, documented as by-design).
 *
 * arts_db_buf_install publishes a new buffer only when its version is strictly
 * greater than the currently-installed one (the guard is old->version >=
 * new_version ⇒ retreat).  Consequences this test pins:
 *
 *   1. Install v=5 (bytes A).  Re-install v=5 (bytes B, DISTINCT).  The second
 *      install must RETREAT: it returns the FIRST buffer, the slot keeps the
 *      first buffer, and the first bytes (A) are preserved — bytes B are
 *      silently dropped.  This documents the same-version semantics: equal
 *      version ⇒ no publish.  (The protocol must guarantee same-version ⇒
 *      same-bytes; this test encodes that the runtime relies on it.)
 *   2. The retreating install must free its own (unpublished) buffer — under
 *      ASan there must be NO leak and NO double-free.  The slot's buffer must
 *      remain valid and readable afterwards.
 *   3. Version never retreats: after v=5 then v=5 then v=4, the slot version is
 *      still 5; a v=6 install does publish and advances to 6.
 *
 * Standalone: links buffer.c + shared.c with libc-backed alloc shims.  Run
 * under ASan for the leak/double-free obligation.
 */

#include "arts/coherence/buffer.h"
#include "arts/memory/regpool.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DB_SIZE 64

static struct arts_db_cache_s g_cache;

static void cache_init(void) {
  memset(&g_cache, 0, sizeof(g_cache));
  g_cache.db_size = DB_SIZE;
}
static void cache_teardown(void) {
  arts_atomic_shared_store(&g_cache.buffer, NULL);
}

static uint64_t slot_version(void) {
  arts_shared_ptr_t h = arts_db_buf_acquire(&g_cache);
  struct arts_db_buffer_s *b = (struct arts_db_buffer_s *)arts_shared_get(h);
  uint64_t v = b ? b->version : (uint64_t)-1;
  arts_db_buf_release(&h);
  return v;
}

int main(void) {
  cache_init();

  unsigned char a[DB_SIZE], b[DB_SIZE];
  memset(a, 0xA1, DB_SIZE);
  memset(b, 0xB2, DB_SIZE);

  /* (1) Install v=5 bytes A. */
  struct arts_db_buffer_s *buf_a = arts_db_buf_install(&g_cache, 5, a, DB_SIZE);
  if (buf_a == NULL || buf_a->version != 5) {
    (void)fprintf(stderr, "FAIL buffer_version_guard: first install bad\n");
    return 1;
  }

  /* Re-install v=5 bytes B — MUST retreat, returning the first buffer. */
  struct arts_db_buffer_s *ret = arts_db_buf_install(&g_cache, 5, b, DB_SIZE);
  if (ret != buf_a) {
    (void)fprintf(stderr,
                  "FAIL buffer_version_guard: equal-version install did not "
                  "retreat (returned %p, expected %p)\n",
                  (void *)ret, (void *)buf_a);
    return 1;
  }

  /* (2) Slot still holds buf_a; bytes A preserved (B dropped). */
  arts_shared_ptr_t h = arts_db_buf_acquire(&g_cache);
  struct arts_db_buffer_s *cur = (struct arts_db_buffer_s *)arts_shared_get(h);
  int rc = 0;
  if (cur != buf_a || cur->version != 5 || memcmp(cur->data, a, DB_SIZE) != 0) {
    (void)fprintf(stderr,
                  "FAIL buffer_version_guard: equal-version retreat altered "
                  "slot/bytes\n");
    rc = 1;
  }
  arts_db_buf_release(&h);
  if (rc) {
    return 1;
  }

  /* (3) Lower version retreats too; version never goes backward. */
  unsigned char c[DB_SIZE];
  memset(c, 0xCC, DB_SIZE);
  struct arts_db_buffer_s *ret4 = arts_db_buf_install(&g_cache, 4, c, DB_SIZE);
  if (ret4 != buf_a || slot_version() != 5) {
    (void)fprintf(stderr,
                  "FAIL buffer_version_guard: lower-version install retreated "
                  "version (now %llu)\n",
                  (unsigned long long)slot_version());
    return 1;
  }

  /* A strictly higher version DOES publish and advances. */
  struct arts_db_buffer_s *buf6 = arts_db_buf_install(&g_cache, 6, c, DB_SIZE);
  if (buf6 == buf_a || buf6->version != 6 || slot_version() != 6) {
    (void)fprintf(
        stderr, "FAIL buffer_version_guard: higher version did not advance\n");
    return 1;
  }
  arts_shared_ptr_t h6 = arts_db_buf_acquire(&g_cache);
  struct arts_db_buffer_s *cur6 =
      (struct arts_db_buffer_s *)arts_shared_get(h6);
  if (memcmp(cur6->data, c, DB_SIZE) != 0) {
    (void)fprintf(stderr, "FAIL buffer_version_guard: v6 bytes wrong\n");
    arts_db_buf_release(&h6);
    return 1;
  }
  arts_db_buf_release(&h6);

  cache_teardown();
  printf("PASS buffer_version_guard: equal/lower retreat preserves first "
         "buffer+bytes, monotone version, higher publishes\n");
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
void *arts_regpool_alloc_aligned(size_t size, size_t align) {
  return arts_malloc_aligned(size, align);
}
void arts_regpool_free(void *p) { free(p); }

/* Rendezvous-plane stubs: buffer.c's landing helpers reference the net core's
 * advertisement/txid primitives and the runtime's fatal-print externs; a pure
 * unit run never allocates a landing, so inert stubs satisfy the link. */
#include <stdint.h>
#include "arts/runtime_state.h"
#include "arts/system/threads.h"
unsigned int arts_global_rank_id = 0;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;
void arts_abort(uint8_t code) { exit(code ? code : 1); }
bool arts_net_rdzv_local(const void *p, uint64_t len, uint64_t *raddr,
                         uint64_t *rkey) {
  (void)p;
  (void)len;
  (void)raddr;
  (void)rkey;
  return false;
}
uint64_t arts_net_rdzv_txid_next(void) { return 1; }

