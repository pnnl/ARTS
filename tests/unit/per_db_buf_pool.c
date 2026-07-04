/* SPDX-License-Identifier: Apache-2.0 — per-DB buffer pool recycle test */
#include "arts/coherence/buffer.h"
#include "arts/coherence/types_common.h"
#include "arts/memory/regpool.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Minimal cache init: zero it, init the free-list + buffer slot.  The real
 * runtime does this in arts_db_cache_common_init; here we inline the two
 * fields the pool needs so the unit test is self-contained. */
static void mini_cache_init(struct arts_db_cache_s *c, uint64_t db_size) {
  memset(c, 0, sizeof(*c));
  c->db_size = db_size;
  arts_lf_pool_init(&c->buf_freelist, 0); /* DWCAS recycle pool; node_size 0 =
                                             pop-only, alloc is external */
  /* buffer slot starts NULL */
}

int main(void) {
  /* No ARTS runtime bootstrap here — arts_db_buf_install draws its buffers
   * from the registered pool, so this standalone test must init it itself,
   * the same precondition arts_runtime_node_init establishes before any
   * real worker/receiver thread runs. */
  assert(arts_regpool_init(NULL, (size_t)64 * 1024 * 1024, 1));

  struct arts_db_cache_s c;
  const uint64_t sz = 128;
  mini_cache_init(&c, sz);

  /* install v1, capture address, release it (refcount -> 0 -> recycled) */
  char payload[128];
  memset(payload, 0xAB, sz);
  struct arts_db_buffer_s *b1 = arts_db_buf_install(&c, 1, payload, sz);
  assert(b1 != NULL);
  void *a1 = (void *)b1;
  /* drop the cache-hold + our ref by storing NULL (release path frees->recycle)
   */
  arts_atomic_shared_store(&c.buffer, NULL);

  /* install v2: should PULL the recycled b1 (same address) */
  struct arts_db_buffer_s *b2 = arts_db_buf_install(&c, 2, payload, sz);
  assert(b2 != NULL);
  assert((void *)b2 == a1 && "v2 must reuse the recycled v1 buffer");
  printf("RECYCLE_OK reused=%p\n", (void *)b2);
  arts_atomic_shared_store(&c.buffer, NULL);
  arts_regpool_cleanup();
  return 0;
}
