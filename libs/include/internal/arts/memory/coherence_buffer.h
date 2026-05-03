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

#include "arts/memory/coherence.h"

/* Allocate a fresh buffer (sized for db_size payload + header) and
 * initialize it.  Caller fills ref_count via the install path —
 * arts_coherence_install_buffer publishes the buffer to cache.buffer
 * via CAS; before that, the buffer is private to the caller. */
struct arts_db_buffer_s *
arts_coherence_buffer_alloc(struct arts_db_cache_s *cache, uint64_t db_size);

/* Race-safe acquire.  Returns the live buffer with ref_count
 * incremented, or NULL if no buffer is currently installed (only
 * possible during pre-publication or post-destroy NULL-swap). */
struct arts_db_buffer_s *
arts_coherence_acquire_buf(struct arts_db_cache_s *cache);

/* Drop a single ref previously taken via acquire_buf / install (the
 * sentinel).  When the decrement returns 1 (i.e. brings the count to
 * 0), the buffer is pushed to cache.buffer_pool — recycle, NOT free. */
void arts_coherence_release_buf(struct arts_db_cache_s *cache,
                                struct arts_db_buffer_s *buf);

/* Recover the enclosing arts_db_buffer_s from a data pointer (which
 * aliases buf->data, the FAM canonical payload).  Used by
 * release_one_dep to drop the EDT's buf ref without having to track
 * the buf pointer separately — the EDT only sees buf->data via
 * depv[slot].ptr, and cache.buffer may have been replaced since
 * acquire time, so we recover the original buf via container_of. */
static inline struct arts_db_buffer_s *
arts_coherence_buf_from_data(void *data) {
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
 * Returns the buffer that became (or remains) cache.buffer.  Stale
 * installs (new_version <= old->version) retreat and return the old
 * buffer unchanged.  Callers typically ignore the return — what
 * matters is that cache.buffer afterward holds a buffer at >= new_version.
 *
 * This call may pop from cache.buffer_pool on the hot path; on a cold-
 * start miss it falls through to malloc. */
struct arts_db_buffer_s *
arts_coherence_install_buffer(struct arts_db_cache_s *cache,
                              uint64_t new_version, const void *data_payload,
                              uint64_t db_size);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_BUFFER_H */
