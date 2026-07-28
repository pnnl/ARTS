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

#include "arts/memory/regpool.h"
#include "arts/system/print.h"  /* ARTS_ERROR (unadvertisable landing) */
#include "arts/transport/net.h" /* arts_net_rdzv_local / _txid_next */
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
  arts_regpool_free(b);
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
   * boundary.  Drawn from the registered pool so every DB payload buffer
   * falls inside a slab that is (or will be) pinned and pre-registered with
   * the fabric, resolvable by arts_regpool_lookup for one-sided RDMA.
   * Recycled by the cb deleter via the free-list; freed only when
   * owner_cache is NULL (shouldn't happen in normal operation). */
  return (struct arts_db_buffer_s *)arts_regpool_alloc_aligned(
      sizeof(struct arts_db_buffer_s) + db_size, 64);
}

/* As arts_db_buf_alloc, but buf->data reads as zero.  A recycled buffer is
 * cleared here (its previous contents are arbitrary); a fresh pool
 * allocation arrives zeroed without being touched. */
struct arts_db_buffer_s *arts_db_buf_alloc_zeroed(struct arts_db_cache_s *cache,
                                                  uint64_t db_size) {
  arts_lf_link_t *node = arts_lf_pool_pop_or_null(&cache->buf_freelist);
  if (node != NULL) {
    struct arts_db_buffer_s *b = (struct arts_db_buffer_s *)node;
    memset(b->data, 0, (size_t)db_size);
    return b;
  }
  return (struct arts_db_buffer_s *)arts_regpool_zalloc_aligned(
      sizeof(struct arts_db_buffer_s) + db_size, 64);
}

arts_shared_ptr_t arts_db_buf_acquire(struct arts_db_cache_s *cache) {
  /* Acquire-and-validate load: returns a caller-owned strong ref (keeps the
   * buffer alive) or NULL if no buffer is installed.  Caller releases via
   * arts_db_buf_release. */
  return arts_atomic_shared_load(&cache->buffer);
}

void arts_db_buf_release(arts_shared_ptr_t *h) { arts_shared_release(h); }

/* Version-conditional publish of a fully-initialized private buffer (fields +
 * payload bytes already set; no cb yet).  Shared by the copy install
 * (arts_db_buf_install) and the rendezvous landed install
 * (arts_db_buf_install_landed).  On a stale loss the private buffer is
 * recycled and the newer installed buffer returned. */
static struct arts_db_buffer_s *buf_publish(struct arts_db_cache_s *cache,
                                            struct arts_db_buffer_s *new_buf,
                                            uint64_t new_version) {
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
    /* The publish decision reads a version another rank's release may be
     * bumping in place; an acquire load pairs with that read-modify-write so
     * the comparison never observes a torn or reordered value. */
    if (old != NULL &&
        __atomic_load_n(&old->version, __ATOMIC_ACQUIRE) >= new_version) {
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

struct arts_db_buffer_s *arts_db_buf_install(struct arts_db_cache_s *cache,
                                             uint64_t new_version,
                                             const void *data_payload,
                                             uint64_t db_size) {
  /* data_payload == NULL ⇒ initial install at create-time: the payload must
   * read as zero (deterministic state).  Take the zeroed allocation path so
   * only a recycled buffer is actually cleared — fresh pool memory is
   * kernel-zeroed already, and skipping the redundant full-payload memset
   * keeps the touch (and its page faults) off the creator's critical path. */
  struct arts_db_buffer_s *new_buf =
      (data_payload == NULL && db_size > 0)
          ? arts_db_buf_alloc_zeroed(cache, db_size)
          : arts_db_buf_alloc(cache, db_size);
  if (new_buf == NULL) {
    return NULL; /* OOM — caller decides how to surface. */
  }
  new_buf->owner_cache = cache;
  new_buf->version = new_version;
  /* Publish bytes into buf->data (FAM, canonical user-visible storage). */
  if (db_size > 0) {
    if (data_payload != NULL) {
      memcpy(new_buf->data, data_payload, (size_t)db_size);
    }
    /* Lazy-installed caches start with db_size==0; the first install learns
     * the real size from the wire payload. */
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
  }
  return buf_publish(cache, new_buf, new_version);
}

/* ===== Rendezvous landing lifecycle (see buffer.h) ======================= */

struct arts_db_buffer_s *
arts_db_buf_landing_alloc(struct arts_db_cache_s *cache, uint64_t db_size,
                          struct arts_rdzv_landing_s *out) {
  struct arts_db_buffer_s *b = arts_db_buf_alloc(cache, db_size);
  if (b == NULL) {
    ARTS_ERROR("coherence: rendezvous landing alloc failed (%llu bytes)",
               (unsigned long long)db_size);
  }
  b->owner_cache = cache;
  if (!arts_net_rdzv_local(b->data, db_size, &out->addr, &out->key)) {
    /* The buffer lies in no fabric-registered slab, so no peer can PUT into
     * it.  One-sided bulk transfer requires the registered arena pool. */
    ARTS_ERROR("coherence: landing buffer is not fabric-registered — "
               "one-sided payloads require the registered pool "
               "(ARTS_MALLOC=mimalloc)");
  }
  out->txid = arts_net_rdzv_txid_next();
  out->cookie = (uint64_t)(uintptr_t)b;
  return b;
}

void arts_db_buf_landing_recycle(struct arts_db_cache_s *cache,
                                 struct arts_db_buffer_s *b) {
  if (b == NULL) {
    return;
  }
  b->cb = NULL;
  b->owner_cache = cache;
  arts_lf_pool_release(&cache->buf_freelist, &b->pool_link);
}

struct arts_db_buffer_s *
arts_db_buf_install_landed(struct arts_db_cache_s *cache, uint64_t new_version,
                           struct arts_db_buffer_s *landing,
                           uint64_t db_size) {
  landing->owner_cache = cache;
  landing->version = new_version;
  if (db_size > 0 && cache->db_size == 0) {
    cache->db_size = db_size;
  }
  return buf_publish(cache, landing, new_version);
}

void arts_db_buf_ref_release_cb(void *arg) {
  arts_shared_ptr_t h = (arts_shared_ptr_t)arg;
  arts_shared_release(&h);
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
  /* First touch: allocate the one stable buffer (no further realloc).
   *
   * First touches CAN race: independent acquisition paths materialize the
   * same cache's buffer concurrently (e.g. two request rounds whose replies
   * arrive on different progress threads).  Publish with an
   * install-if-absent CAS, never an unconditional store — a losing store
   * would REPLACE the winner's established buffer, detaching any
   * fixed-address landing already advertised on it (silently discarding
   * data already landed there) while later readers re-derive their pointers
   * from the fresh, still-zero instance. */
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
  if (!arts_atomic_shared_compare_exchange(&cache->buffer, NULL, cb)) {
    /* Lost the install race.  Racing first touches publish the same zero
     * first image, so adopting the winner is value-identical; a
     * data-carrying caller (whose write the coherence protocol serializes
     * against every reader) writes its bytes through the established buffer
     * instead. */
    arts_shared_release(&cb);
    if (db_size > 0 && data != NULL) {
      arts_shared_ptr_t wh = arts_db_buf_acquire(cache);
      struct arts_db_buffer_s *wbuf =
          (struct arts_db_buffer_s *)arts_shared_get(wh);
      if (wbuf != NULL) {
        memcpy(wbuf->data, data, (size_t)db_size);
      }
      arts_shared_release(&wh);
    }
  }
}
