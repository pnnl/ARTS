/* SPDX-License-Identifier: Apache-2.0
 *
 * Buffer lifecycle implementation.  See coherence_buffer.h for the
 * full contract; this file just realizes the three primitives.
 */

#include "arts/memory/coherence_buffer.h"

#include <stdlib.h>
#include <string.h>

#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* Helper: cast pool_link back to the enclosing buffer.  pool_link is
 * the FIRST field, so offsetof == 0 and the cast is a plain pointer
 * cast.  We still spell it out for clarity. */
static inline struct arts_db_buffer_s *
buf_from_pool_link(arts_lockfree_stack_node_t *link) {
  return (struct arts_db_buffer_s *)link;
}

static inline arts_lockfree_stack_node_t *
buf_to_pool_link(struct arts_db_buffer_s *buf) {
  return &buf->pool_link;
}

struct arts_db_buffer_s *
arts_coherence_buffer_alloc(struct arts_db_cache_s *cache, uint64_t db_size) {
  /* Hot path: pop from per-DB recycle pool.  All pool buffers share
   * the cache's fixed db_size so reuse is size-safe. */
  arts_lockfree_stack_node_t *link =
      arts_lockfree_stack_pop(&cache->buffer_pool);
  if (link != NULL) {
    return buf_from_pool_link(link);
  }
  /* Cold-start miss: alloc handle + FAM payload.  64-byte aligned via
   * arts_malloc_align so buf->data lands on a CXL-compatible boundary
   * (see cxl-64-byte-alignment invariant). */
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_malloc_align(sizeof(*buf) + db_size, 64);
  return buf;
}

struct arts_db_buffer_s *
arts_coherence_acquire_buf(struct arts_db_cache_s *cache) {
  /* Race-safe load + ref_count++.  See coherence_buffer.h header for
   * why the "rc > 0" CAS guard plus the never-free recycle pool
   * eliminates UAF without SMR / hazard pointers / mutex. */
  while (1) {
    /* Volatile load of cache.buffer; the void *-typed swap helper
     * elsewhere takes a `volatile void **`, so the cast through
     * `void *` keeps the type discipline consistent. */
    struct arts_db_buffer_s *buf = (struct arts_db_buffer_s *)cache->buffer;
    if (buf == NULL) {
      /* Either pre-publication (transient at creation, callers don't
       * reach here legitimately) or post-destroy NULL-swap by
       * try_finalize_destroy.  Caller treats NULL as
       * ARTS_DB_DESTROYED. */
      return NULL;
    }
    /* CAS-loop "increment if positive": never bump from 0.  rc == 0
     * means the buffer is already in (or being pushed to) the
     * recycle pool — refusing to claim it preserves the never-free
     * recycle invariant.  We re-load cache.buffer on each retry
     * because the buffer pointer may have been replaced.  Atomic
     * acquire-load on ref_count avoids a TSan race against the
     * atomic_sub in release_buf. */
    unsigned int rc = arts_atomic_read(&buf->ref_count);
    while (rc > 0) {
      if (arts_atomic_cswap(&buf->ref_count, rc, rc + 1) == rc) {
        /* Successfully bumped.  Re-validate the buffer is still the
         * installed cache.buffer — if a concurrent install replaced
         * it, drop our (orphan) ref and retry. */
        if (buf == (struct arts_db_buffer_s *)cache->buffer) {
          return buf;
        }
        /* Buffer was replaced; drop our spurious ref. */
        if (arts_atomic_sub(&buf->ref_count, 1) == 0) {
          arts_lockfree_stack_push(&cache->buffer_pool, buf_to_pool_link(buf));
        }
        rc = 0; /* break inner; outer retry. */
        break;
      }
      /* CAS lost; reload rc to its actual current value. */
      rc = arts_atomic_read(&buf->ref_count);
    }
    /* rc == 0 here either because the inner CAS-loop exited via the
     * "buffer was replaced" path or because some other thread raced
     * us down to 0.  Either way the outer loop reloads cache.buffer
     * and retries. */
  }
}

void arts_coherence_release_buf(struct arts_db_cache_s *cache,
                                struct arts_db_buffer_s *buf) {
  /* arts_atomic_sub returns the post-decrement value; recycle when
   * the decrement brought us to 0. */
  if (arts_atomic_sub(&buf->ref_count, 1) == 0) {
    arts_lockfree_stack_push(&cache->buffer_pool, buf_to_pool_link(buf));
  }
}

struct arts_db_buffer_s *
arts_coherence_install_buffer(struct arts_db_cache_s *cache,
                              uint64_t new_version, const void *data_payload,
                              uint64_t db_size) {
  /* Allocate fresh (or recycle).  Initialize private to this caller —
   * the buffer is not yet visible to any reader because cache.buffer
   * doesn't point at it. */
  struct arts_db_buffer_s *new_buf =
      arts_coherence_buffer_alloc(cache, db_size);
  if (new_buf == NULL) {
    return NULL; /* OOM — caller decides how to surface. */
  }
  new_buf->version = new_version;
  /* Publish bytes into buf->data (FAM, canonical user-visible storage).
   * data_payload == NULL ⇒ initial install at create-time: zero-init
   * data so subsequent reads see deterministic state. */
  if (db_size > 0) {
    if (data_payload != NULL) {
      memcpy(new_buf->data, data_payload, (size_t)db_size);
    } else {
      memset(new_buf->data, 0, (size_t)db_size);
    }
    /* Update cache->db_size: lazy-installed cache_s starts with
     * db_size=0; first install_buffer learns the real size from the
     * wire payload.  Without this, send_writeback later passes
     * data_size=0 → zero-payload fallback → corrupt raw-heap data. */
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
  }
  /* Publish the sentinel ref last — concurrent acquire_buf's "rc > 0"
   * CAS-loop sees the buffer only after init is fully visible. */
  new_buf->ref_count = 1;

  while (1) {
    struct arts_db_buffer_s *old = (struct arts_db_buffer_s *)cache->buffer;
    if (old != NULL && old->version >= new_version) {
      /* Stale install: retreat.  We must NOT do `ref_count = 0`
       * directly — if new_buf came from the recycle pool, a racing
       * acquire_buf might have CAS-bumped our sentinel from 1 to 2.
       * Plain store would clobber that increment, leaving the reader
       * with a phantom ref and eventually underflowing on release.
       * Use fetch_sub so the sentinel is dropped correctly; if we
       * brought ref_count to 0 here, we're the sole holder and
       * recycle.  Otherwise, a racing reader holds a ref and their
       * eventual release recycles. */
      if (arts_atomic_sub(&new_buf->ref_count, 1) == 0) {
        arts_lockfree_stack_push(&cache->buffer_pool,
                                 buf_to_pool_link(new_buf));
      }
      return old;
    }
    /* CAS install: from old → new_buf at cache.buffer. */
    if (arts_atomic_cswap_ptr((volatile void **)&cache->buffer, old, new_buf) ==
        old) {
      /* Success.  Retire old's sentinel ref. */
      if (old != NULL && arts_atomic_sub(&old->ref_count, 1) == 0) {
        arts_lockfree_stack_push(&cache->buffer_pool, buf_to_pool_link(old));
      }
      return new_buf;
    }
    /* CAS lost; cache.buffer was concurrently replaced.  Loop and
     * re-evaluate against the new state — possibly we'll retreat
     * (incoming version no longer ahead) or possibly we'll install. */
  }
}
