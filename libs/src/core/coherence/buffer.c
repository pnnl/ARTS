/* SPDX-License-Identifier: Apache-2.0
 *
 * Buffer lifecycle implementation.  See coherence_buffer.h for the contract.
 *
 * The buffer is managed as an arts_shared_ptr_t: cache.buffer is the atomic
 * slot, the slot holds the "cache-hold" strong ref, and every acquirer holds
 * one more.  The cb deleter frees the buffer on the last drop.  Because the
 * control block is drawn from a pool that is never returned to the allocator
 * and the buffer carries no back-pointer to its cache, an in-flight acquire is
 * immune to a concurrent destroy: the bytes live until the final holder
 * releases, and the release path touches only the (always-valid) cb handle,
 * never a possibly-freed buffer.
 */

#include "arts/coherence/buffer.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

/* cb deleter — runs once, on the last strong drop. */
static void buffer_deleter(void *obj) { arts_free(obj); }

struct arts_db_buffer_s *arts_db_buf_alloc(uint64_t db_size) {
  /* 64-byte aligned so buf->data (offset 64) lands on a cache-line / CXL
   * boundary.  Freed by the cb deleter (arts_free), never recycled. */
  return (struct arts_db_buffer_s *)arts_malloc_align(
      sizeof(struct arts_db_buffer_s) + db_size, 64);
}

arts_shared_ptr_t arts_db_buf_acquire(struct arts_db_cache_s *cache) {
  /* Acquire-and-validate load: returns a caller-owned strong ref (keeps the
   * buffer alive) or NULL if no buffer is installed.  Caller releases via
   * arts_db_buf_release. */
  return arts_atomic_shared_load(&cache->buffer);
}

void arts_db_buf_release(arts_shared_ptr_t *h) { arts_shared_release(h); }

struct arts_db_buffer_s *arts_db_buf_peek(struct arts_db_cache_s *cache) {
  /* Unsafe non-refcounted peek — valid only in create-time / single-owner
   * windows where no concurrent destroy can free the buffer.  Used to read
   * the freshly-installed payload pointer at DB create. */
  arts_shared_ptr_t cb =
      atomic_load_explicit(&cache->buffer, memory_order_acquire);
  return (struct arts_db_buffer_s *)arts_shared_get(cb);
}

struct arts_db_buffer_s *arts_db_buf_install(struct arts_db_cache_s *cache,
                                             uint64_t new_version,
                                             const void *data_payload,
                                             uint64_t db_size) {
  struct arts_db_buffer_s *new_buf = arts_db_buf_alloc(db_size);
  if (new_buf == NULL) {
    return NULL; /* OOM — caller decides how to surface. */
  }
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
      arts_free(new_buf);
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
