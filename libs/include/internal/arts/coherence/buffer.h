/* SPDX-License-Identifier: Apache-2.0
 *
 * Buffer lifecycle for the coherence protocol.
 *
 * Three primitives manage every arts_db_buffer_s instance during a
 * DB's lifetime:
 *
 *   acquire_buf   — race-safe buffer acquire.  Increments ref_count
 *                   via a CAS-loop guard ("only bump if rc > 0"),
 *                   re-validates the buffer pointer is still
 *                   cache.buffer after the bump, and retries
 *                   otherwise.  Returns NULL only on destroy NULL-
 *                   swap or pre-population (transient at creation).
 *
 *   release_buf   — drops a single ref; whoever brings ref_count to
 *                   0 pushes the buffer back to cache.buffer_pool.
 *                   Pool is per-DB and never `free`d at runtime.
 *
 *   install_buffer — CAS-loop install of a new buffer at cache.buffer.
 *                   Stale installs (incoming version <= current) are
 *                   rejected via "old->version >= new_version" check.
 *                   Pops a buffer from cache.buffer_pool first; mallocs
 *                   only on cold-start miss.  The retired old buffer's
 *                   sentinel ref is fetch_sub'd; whoever brings it to
 *                   0 recycles.
 *
 * Together these eliminate any use-after-free without SMR, hazard
 * pointers, or mutexes:
 *
 *   - never-free recycle: memory stays valid for the DB's lifetime.
 *   - "rc > 0" CAS guard: a reader can never bump from 0 → 1, so a
 *     freshly-popped buffer's mid-init state (version/data being
 *     filled) is unobservable until install_buffer's
 *     ref_count.store(1) publishes.
 *   - version monotonicity across recycles: ABA on the buffer
 *     pointer is benign because the version stamp inside the buffer
 *     is monotonic per DB.  A reader observing a recycled pointer
 *     necessarily observes a version >= its acquire-time version,
 *     which is exactly what RC promises.
 */

#ifndef ARTS_MEMORY_COHERENCE_BUFFER_H
#define ARTS_MEMORY_COHERENCE_BUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "arts/coherence/coherence.h"
#include "arts/utils/shared.h" /* arts_shared_ptr_t */

/* Allocate a fresh, uninitialized buffer (header + db_size payload), 64-byte
 * aligned.  install_buffer fills + wraps it in a control block; before that
 * the buffer is private to the caller. */
struct arts_db_buffer_s *arts_db_buf_alloc(uint64_t db_size);

/* Race-safe acquire: returns a caller-owned strong ref to the installed
 * buffer (keeping it alive against a concurrent destroy), or NULL if no
 * buffer is currently installed.  Recover the buffer via arts_shared_get;
 * release via arts_db_buf_release when done. */
arts_shared_ptr_t arts_db_buf_acquire(struct arts_db_cache_s *cache);

/* Drop a strong ref taken via acquire_buf.  On the last drop the cb deleter
 * frees the buffer.  Sets *h = NULL. */
void arts_db_buf_release(arts_shared_ptr_t *h);

/* Unsafe non-refcounted peek of the installed buffer — valid only in
 * create-time / single-owner windows where no concurrent destroy can free
 * it.  Returns NULL if no buffer is installed. */
struct arts_db_buffer_s *arts_db_buf_peek(struct arts_db_cache_s *cache);

/* Recover the enclosing arts_db_buffer_s from a data pointer (which aliases
 * buf->data, the FAM canonical payload).  Pointer arithmetic only — does NOT
 * touch the buffer, so it is safe even if the buffer has since been freed
 * (the caller must already hold a ref or know the buffer is alive). */
static inline struct arts_db_buffer_s *arts_db_buf_from_data(void *data) {
  if (data == NULL) {
    return NULL;
  }
  return (struct arts_db_buffer_s *)((char *)data -
                                     offsetof(struct arts_db_buffer_s, data));
}

/* Install a new buffer carrying `new_version` at cache.buffer.
 *
 * data_payload semantics:
 *   NULL  ⇒ zero-init the new buffer (used by DB_CREATE so the
 *           backing store starts predictable).
 *   non-NULL ⇒ memcpy db_size bytes from data_payload.
 *
 * Returns the buffer that became (or remains) cache.buffer.  Stale installs
 * (new_version <= old->version) retreat and return the old buffer unchanged.
 * Publishes via a version-conditional shared-ptr compare-exchange; the slot
 * takes the cache-hold ref and the retired buffer's ref is dropped (its cb
 * deleter frees it once the last in-flight acquirer releases). */
struct arts_db_buffer_s *arts_db_buf_install(struct arts_db_cache_s *cache,
                                             uint64_t new_version,
                                             const void *data_payload,
                                             uint64_t db_size);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_BUFFER_H */
