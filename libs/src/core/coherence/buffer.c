/* SPDX-License-Identifier: Apache-2.0
 *
 * Buffer lifecycle implementation.  See coherence_buffer.h for the contract.
 *
 * The buffer is managed as an arts_shared_ptr_t: cache.buffer is the atomic
 * slot, the slot holds the "cache-hold" ref, and every acquirer holds one more.
 * The cb deleter frees the buffer on the last drop.  Because the split-
 * reference-counting load pins the cb without dereferencing it (and the buffer
 * carries no back-pointer to its cache), an in-flight acquire is immune to a
 * concurrent destroy: the bytes live until the final holder releases, and the
 * release path touches only the (always-valid) cb handle, never a possibly-
 * freed buffer.
 */

#include "arts/coherence/buffer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

/* cb deleter — runs once, on the last strong drop.  When owner_cache is set,
 * recycles the buffer onto the per-DB free-list (unbounded) instead of freeing
 * so that subsequent installs can reuse it without hitting the allocator. */
static void buffer_deleter(void *obj) {
  struct arts_db_buffer_s *b = (struct arts_db_buffer_s *)obj;
  b->cb = NULL;
  struct arts_db_cache_s *cache = b->owner_cache;
  if (cache != NULL) {
    arts_lf_pool_release(&cache->buf_freelist, &b->pool_link);
    return;
  }
  arts_free(b);
}

struct arts_db_buffer_s *arts_db_buf_alloc(struct arts_db_cache_s *cache,
                                           uint64_t db_size) {
  /* Pull a recycled buffer from the per-DB free-list when available. */
  arts_lf_link_t *node = arts_lf_pool_pop_or_null(&cache->buf_freelist);
  if (node != NULL) {
    return (
        struct arts_db_buffer_s *)node; /* recycled; caller re-inits fields */
  }
  /* 64-byte aligned so buf->data (offset 64) lands on a cache-line / CXL
   * boundary.  Recycled by the cb deleter via the free-list; freed only when
   * owner_cache is NULL (shouldn't happen in normal operation). */
  return (struct arts_db_buffer_s *)arts_malloc_aligned(
      sizeof(struct arts_db_buffer_s) + db_size, 64);
}

arts_shared_ptr_t arts_db_buf_acquire(struct arts_db_cache_s *cache) {
  /* Acquire-and-validate load: returns a caller-owned strong ref (keeps the
   * buffer alive) or NULL if no buffer is installed.  Caller releases via
   * arts_db_buf_release. */
  return arts_atomic_shared_load(&cache->buffer);
}

void arts_db_buf_release(arts_shared_ptr_t *h) { arts_shared_release(h); }

struct arts_db_buffer_s *arts_db_buf_install(struct arts_db_cache_s *cache,
                                             uint64_t new_version,
                                             const void *data_payload,
                                             uint64_t db_size) {
  struct arts_db_buffer_s *new_buf = arts_db_buf_alloc(cache, db_size);
  if (new_buf == NULL) {
    return NULL; /* OOM — caller decides how to surface. */
  }
  new_buf->owner_cache = cache;
  new_buf->version = new_version;
  /* Publish bytes into buf->data (FAM, canonical user-visible storage).
   * data_payload == NULL ⇒ initial install at create-time: zero-init so
   * subsequent reads see deterministic state. */
  if (db_size > 0) {
    if (data_payload != NULL) {
      memcpy(new_buf->data, data_payload, (size_t)db_size);
    } else {
      memset(new_buf->data, 0, (size_t)db_size);
    }
    /* Lazy-installed caches start with db_size==0; the first install learns
     * the real size from the wire payload. */
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
  }
  /* Wrap the buffer in a fresh control block (strong = 1).  The slot will
   * take this ref as the cache-hold on a successful publish.  Stash the cb in
   * the buffer so a holder can recover it (buf->cb) to release without
   * threading the handle through the acquire call chain. */
  arts_shared_ptr_t new_cb = arts_shared_make(new_buf, buffer_deleter);
  new_buf->cb = new_cb;

  for (;;) {
    arts_shared_ptr_t old_h = arts_atomic_shared_load(&cache->buffer);
    struct arts_db_buffer_s *old =
        (struct arts_db_buffer_s *)arts_shared_get(old_h);
    if (old != NULL && old->version >= new_version) {
      /* Stale install: a newer (or equal) buffer is already published.
       * Drop our load ref, abandon the unpublished cb (keeps new_buf ours)
       * and free new_buf.  old stays alive via the slot's sentinel ref. */
      arts_shared_release(&old_h);
      arts_shared_abandon(&new_cb);
      buffer_deleter(new_buf); /* recycle the just-allocated buffer */
      return old;
    }
    /* Conditional publish: install new_cb only while the slot still holds
     * old_h.  old_h pins old's cb (strong >= 1) so the raw-pointer compare
     * cannot ABA.  On success the slot drops its ref on the old value. */
    if (arts_atomic_shared_compare_exchange(&cache->buffer, old_h, new_cb)) {
      if (old_h != NULL) {
        arts_shared_release(&old_h); /* our load ref on old */
      }
      return new_buf;
    }
    /* CAS lost — cache.buffer changed concurrently; re-evaluate. */
    if (old_h != NULL) {
      arts_shared_release(&old_h);
    }
  }
}

void arts_db_buf_write_inplace(struct arts_db_cache_s *cache, const void *data,
                               uint64_t db_size) {
  arts_shared_ptr_t h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf = (struct arts_db_buffer_s *)arts_shared_get(h);
  if (buf != NULL) {
    /* Established stable buffer: overwrite in place.  The address is fixed, so
     * a DB holding internal self-pointers stays valid.  Safe because the
     * caller's protocol guarantees no concurrent reader during the write. */
    if (db_size > 0 && data != NULL) {
      memcpy(buf->data, data, (size_t)db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    arts_shared_release(&h);
    return;
  }
  /* First touch: allocate the one stable buffer (no further realloc).  Single
   * producer here — the caller's serialization makes the first write to a given
   * cache unique — so a plain store publishes it. */
  struct arts_db_buffer_s *nb = arts_db_buf_alloc(cache, db_size);
  if (nb == NULL) {
    return; /* OOM — caller decides how to surface. */
  }
  nb->owner_cache = cache;
  nb->version = 0;
  if (db_size > 0) {
    if (data != NULL) {
      memcpy(nb->data, data, (size_t)db_size);
    } else {
      memset(nb->data, 0, (size_t)db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
  }
  arts_shared_ptr_t cb = arts_shared_make(nb, buffer_deleter);
  nb->cb = cb;
  arts_atomic_shared_store(&cache->buffer, cb);
}
