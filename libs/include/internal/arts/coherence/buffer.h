/* SPDX-License-Identifier: Apache-2.0
 *
 * Buffer lifecycle for the coherence protocol.
 *
 * Each DB version is a separate arts_db_buffer_s wrapped in an
 * arts_shared_ptr_t (cb).  cache.buffer is the atomic slot; the slot holds the
 * "cache-hold" strong ref and every acquirer holds one more.  There is NO
 * per-DB buffer free-list: arts_db_buf_alloc pulls a recycled buffer from
 * cache->buf_freelist before falling back to arts_regpool_alloc_aligned; buffer_deleter
 * pushes the buffer back onto the free-list (unbounded) on the last strong
 * drop so version buffers are reused across the DB's lifetime.
 *
 * Three primitives manage every buffer during a DB's lifetime:
 *
 *   arts_db_buf_acquire  — race-safe acquire.  arts_atomic_shared_load returns
 *                          a caller-owned strong ref (keeping the buffer alive
 *                          against a concurrent destroy) or NULL if none is
 *                          installed.
 *   arts_db_buf_release  — drops one strong ref; the cb deleter frees the
 *                          buffer on the last drop.
 *   arts_db_buf_install  — version-conditional shared-ptr compare-exchange of a
 *                          new buffer at cache.buffer.  Stale installs
 *                          (old->version >= new_version) retreat and free the
 *                          new buffer; the slot drops its ref on the retired
 *                          buffer (its cb deleter frees it once the last
 *                          in-flight acquirer releases).
 *
 * Together these eliminate use-after-free without SMR / hazard pointers /
 * mutexes:
 *   - the cb keeps the buffer bytes valid for any in-flight acquirer even
 *     across a concurrent destroy (the buffer carries no back-pointer to its
 *     cache, so the release path never touches a possibly-freed cache).
 *   - version monotonicity: the version stamp inside the buffer is monotonic
 *     per DB, so a reader observing the installed buffer necessarily observes a
 *     version >= its acquire-time version — exactly what the protocol promises.
 *
 * Convention: all shared-object access is via caller-owned cb handles
 * (lookup_* / _acquire → handle; release required). No raw no-ref peeks.
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

/* Allocate a buffer (header + db_size payload), 64-byte aligned.  Pulls a
 * recycled buffer from cache->buf_freelist when one is available; falls back
 * to arts_regpool_alloc_aligned.  install_buffer fills + wraps it in a
 * control block; before that the buffer is private to the caller. */
struct arts_db_buffer_s *arts_db_buf_alloc(struct arts_db_cache_s *cache,
                                           uint64_t db_size);

/* As arts_db_buf_alloc, but buf->data reads as zero (recycled buffers are
 * cleared; fresh pool memory arrives zeroed untouched). */
struct arts_db_buffer_s *arts_db_buf_alloc_zeroed(struct arts_db_cache_s *cache,
                                                  uint64_t db_size);

/* Race-safe acquire: returns a caller-owned strong ref to the installed
 * buffer (keeping it alive against a concurrent destroy), or NULL if no
 * buffer is currently installed.  Recover the buffer via arts_shared_get;
 * release via arts_db_buf_release when done. */
arts_shared_ptr_t arts_db_buf_acquire(struct arts_db_cache_s *cache);

/* Drop a strong ref taken via acquire_buf.  On the last drop the cb deleter
 * frees the buffer.  Sets *h = NULL. */
void arts_db_buf_release(arts_shared_ptr_t *h);

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

/* ===== Rendezvous landing lifecycle =========================================
 * A landing is a FRESH buffer allocated by the RECEIVER of a bulk payload and
 * advertised (addr/key/txid/cookie) to the sender, which fi_writedata's the
 * payload straight into landing->data.  On {metadata, write-completion}
 * pairing the landing becomes the installed buffer via
 * arts_db_buf_install_landed — today's install logic with the memcpy deleted —
 * preserving the fresh-buffer-per-install invariant (concurrent readers of the
 * old buffer keep a valid snapshot) with zero copies at both ends. */

/* Allocate a fresh landing for `db_size` payload bytes and fill its wire
 * advertisement (addr per negotiated mr_mode, MR key, fresh txid, cookie =
 * the landing pointer).  Fails loudly if the buffer cannot be advertised
 * (payloads cannot move one-sided without a fabric-registered pool). */
struct arts_db_buffer_s *
arts_db_buf_landing_alloc(struct arts_db_cache_s *cache, uint64_t db_size,
                          struct arts_rdzv_landing_s *out);

/* Return a never-installed landing to the per-DB free-list (the payload did
 * not move: dedup'd response, unused advertisement, self-transfer). */
void arts_db_buf_landing_recycle(struct arts_db_cache_s *cache,
                                 struct arts_db_buffer_s *b);

/* Install a PUT-landed buffer carrying `new_version` at cache.buffer — the
 * no-copy twin of arts_db_buf_install (the payload bytes are already in
 * landing->data).  Stale installs retreat, recycle the landing, and return
 * the newer installed buffer unchanged. */
struct arts_db_buffer_s *
arts_db_buf_install_landed(struct arts_db_cache_s *cache, uint64_t new_version,
                           struct arts_db_buffer_s *landing, uint64_t db_size);

/* One-shot completion callback releasing the strong buffer ref passed as arg —
 * the PUT source-lifetime gate (the ref keeps the source buffer's bytes valid
 * until the fabric's local completion says it no longer reads them). */
void arts_db_buf_ref_release_cb(void *arg);

/* Write into the cache's single stable buffer in place.
 *
 * Unlike arts_db_buf_install (versioned realloc-on-write — a fresh buffer is
 * allocated and swapped on every call, so concurrent readers of the old buffer
 * keep a valid snapshot), this keeps the buffer at a FIXED address from its
 * first allocation until destroy.  DBs that store internal absolute pointers
 * into their own backing store therefore stay valid across writes.
 *
 * The first call (no buffer yet) allocates the one stable buffer; every call
 * memcpy's `data` (or zero-fills when data == NULL) into the existing buffer's
 * data in place.  Never reallocates or version-swaps once established.
 *
 * Correct ONLY when no concurrent reader holds the buffer while it is written:
 * the caller's protocol must serialize access so an overwrite never races a
 * read (an exclusive-lock protocol guarantees this; snapshot protocols do not
 * and MUST use arts_db_buf_install instead). */
void arts_db_buf_write_inplace(struct arts_db_cache_s *cache, const void *data,
                               uint64_t db_size);

/* In-place publish commit: version-stamp the stable buffer (bytes already
 * landed).  Hard-errors on a non-increasing version — the single-flight
 * publish discipline makes that unreachable in a legal run. */
void arts_db_buf_bump_inplace(struct arts_db_cache_s *cache, uint64_t version);

/* First-touch stable-buffer materialization from a size BOUND: allocates and
 * zero-fills at `capacity` WITHOUT recording `cache->db_size` — the size is
 * only ever declared by a wire-carried exact value, never by a locally
 * decoded bound (a bound recorded as the size would later be sent as an
 * exact wire length).  Established buffers are left untouched. */
void arts_db_buf_prepare_inplace(struct arts_db_cache_s *cache,
                                 uint64_t capacity);

/* Size to allocate/advertise for this rank's FIRST fetch of a DB: the known
 * exact size, else the GUID's szhint bound, else 0 (= no landing; the
 * size-CTS round remains as the sentinel fallback). */
uint64_t arts_db_first_fetch_size(const struct arts_db_cache_s *cache);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_BUFFER_H */
