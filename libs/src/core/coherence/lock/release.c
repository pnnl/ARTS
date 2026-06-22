/* SPDX-License-Identifier: Apache-2.0
 *
 * LOCK protocol release translation unit.
 *
 * Defines: arts_db_release_rw, arts_db_release_ro, arts_send_db_lock_release.
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=LOCK.
 */

/* lock/types.h must precede all other coherence headers: it defines
 * arts_db_cache_s, arts_db_s, LOCK_HELD_*, and the home-directory types for
 * the LOCK build. */
#include "arts/coherence/lock/types.h"

#include <semaphore.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/ooo.h"
#include "arts/system/identity.h"
#include "arts/transport/outbox.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== arts_send_db_lock_release ========================================
 * Send LOCK_RELEASE to the home.  RW carries writeback data, version, and cv
 * (the releaser's stack-local sem_t address so home can echo it in the ACK);
 * RO carries none and passes version=0 / cv=0.
 * Self-send (home == this rank) routes through the OoO engine so reordering
 * against DB_CREATE is handled identically to the wire path. */
void arts_send_db_lock_release(unsigned int home_rank, arts_guid_t db_guid,
                               arts_db_access_mode_t mode, uint64_t version,
                               uint64_t cv, const void *data,
                               uint64_t data_size) {
  uint64_t ds = (mode == DB_MODE_RW && data != NULL) ? data_size : 0u;

  if (home_rank == arts_global_rank_id) {
    /* Self-send: build the contiguous args buffer (header + inline data) and
     * route through the OoO engine exactly as the wire RX dispatcher does.
     * HIT runs arts_handler_db_lock_release inline (which posts cv); MISS
     * defers the args until the home db_s is installed. */
    uint32_t asz =
        (uint32_t)(sizeof(struct arts_ooo_args_db_lock_release_s) + ds);
    char *abuf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_lock_release_s *args =
        (struct arts_ooo_args_db_lock_release_s *)abuf;
    args->releaser = arts_global_rank_id;
    args->db_guid = db_guid;
    args->mode = (uint32_t)mode;
    args->data_size = ds;
    args->cv = cv;
    args->version = version;
    if (ds > 0u) {
      memcpy(abuf + sizeof(*args), data, (size_t)ds);
    }
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_LOCK_RELEASE, abuf, asz);
    arts_free(abuf);
    return;
  }

  /* Remote send: build the wire packet header + optional inline payload. */
  struct arts_msg_lock_release_packet_s p;
  uint64_t total = sizeof(p) + ds;
  arts_fill_packet_header(&p.header, total, MSG_DB_LOCK_RELEASE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.version = version;
  p.cv = cv;
  if (ds == 0u) {
    arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
    return;
  }
  /* Assemble header + payload into a single contiguous allocation.
   * arts_transport_send_payload_async stores only the payload pointer, which
   * points into the caller's buffer (buf->data).  That buffer is released by
   * lock_send_release_rw immediately after this call, which can free or
   * reuse the data before the sender thread reads it — producing stale bytes
   * on the wire.  Copying into a fresh allocation makes the packet
   * self-contained and independent of the caller's buffer lifetime. */
  char *pkt = (char *)arts_malloc((size_t)total);
  memcpy(pkt, &p, sizeof(p));
  memcpy(pkt + sizeof(p), data, (size_t)ds);
  arts_transport_send_async((int)home_rank, pkt, (unsigned int)total);
  arts_free(pkt);
}

/* ===== release-edge senders ============================================
 * Called AFTER the release CAS has already moved the state word to IDLE for
 * this phase (so for a self-send, the inline grant handler that follows will
 * CAS the next phase onto an already-IDLE word — no overwrite).
 *
 * RW → synchronous writeback: ship buf->data with a stack-local sem_t cv
 *      token; home echoes it in LOCK_RELEASE_ACK and await_writeback_ack
 *      returns only after the post (no lost update across TCP; self-send posts
 *      inline and returns at once).
 * RO → data-less fire-and-forget notify. */
static void lock_send_release_rw(struct arts_db_cache_s *cache) {
  unsigned int home = (unsigned int)arts_guid_get_rank(cache->db_guid);
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  const void *data = (buf != NULL) ? buf->data : NULL;
  uint64_t ds = (buf != NULL) ? cache->db_size : 0u;
  sem_t cv;
  sem_init(&cv, 0, 0);
  uint64_t cv_token = (uint64_t)(uintptr_t)&cv;
  /* version=0: home is the sole version authority (used only for home's
   * buf_install; the requester does not set it). */
  arts_send_db_lock_release(home, cache->db_guid, DB_MODE_RW, /*version=*/0u,
                            cv_token, data, ds);
  await_writeback_ack(&cv);
  sem_destroy(&cv);
  arts_db_buf_release(&buf_h);
}

static void lock_send_release_ro(struct arts_db_cache_s *cache) {
  unsigned int home = (unsigned int)arts_guid_get_rank(cache->db_guid);
  arts_send_db_lock_release(home, cache->db_guid, DB_MODE_RO, /*version=*/0u,
                            /*cv=*/0u, NULL, 0u);
}

/* ===== arts_db_release_rw / arts_db_release_ro =========================
 * Single-word release: CAS the count down (+ the 0-edge state transition,
 * atomic together) via cache_compute_next, then run the release action it
 * returns.  CACHE_ACT_REL_RW → writeback (the last holder of an RW grant, incl.
 * the last RO joiner under it); CACHE_ACT_REL_RO → notify (last RO holder).
 *
 * Destroyed-guard: a dep NULL-woken at destroy never really held the lock; a
 * count of 0 means there is nothing to release — skip rather than underflow.
 * (Full destroy reconciliation under the acquire-time count is a separate
 * subtask.) */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_RW_CNT(cur) == 0u) {
      return; /* destroyed-guard / already released */
    }
    next = cache_compute_next(cur, CACHE_OP_REL_RW, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_REL_RW) {
    lock_send_release_rw(cache);
  }
}

void arts_db_release_ro(struct arts_db_cache_s *cache) {
  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    if (CACHE_RO_CNT(cur) == 0u) {
      return; /* destroyed-guard / already released */
    }
    next = cache_compute_next(cur, CACHE_OP_REL_RO, &act);
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));
  if (act == CACHE_ACT_REL_RW) {
    lock_send_release_rw(
        cache); /* last RO joiner under an RW grant → writeback */
  } else if (act == CACHE_ACT_REL_RO) {
    lock_send_release_ro(cache);
  }
}
