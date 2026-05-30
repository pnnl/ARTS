/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence release path implementation.  See the design plan for
 * the four release cases (R1-R4) and the WRITEBACK_ACK rendezvous.
 *
 * Wait/wake mechanism: a stack-local binary semaphore.  release_rw
 * sem_init's a sem_t on its stack, embeds its address in the WRITEBACK
 * packet, and sem_waits on it.  The home echoes that address verbatim in
 * the WRITEBACK_ACK; arts_handler_db_writeback_ack sem_posts it.  Matching
 * is by pointer identity (the address is valid only on the releaser rank,
 * where the post runs) — no per-cache seq state, no busy-wait.
 */

#include "arts/memory/coherence_release.h"

#include <errno.h>
#include <semaphore.h>
#include <stdint.h>
#include <time.h>

#include "arts/gas/route_table.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/coherence_home.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"

/* ===== local_transfer / invalidate_transfer (RC/LRC only) =========
 *
 * LC has no LOCK_REQ / INVALIDATE / GRANT round; non-home writers push
 * data back to home via synchronous WRITEBACK and then release their
 * writer_count.  Home is always the canonical data holder, so no
 * local_transfer_now or invalidate_transfer is needed in LC builds. */
#ifndef ARTS_MEMORY_MODEL_LC

void arts_coh_local_transfer_now(struct arts_db_cache_s *cache) {
  unsigned int new_owner;
  if (!arts_home_lockreq_queue_pop(&cache->home.pending_rw, &new_owner)) {
    /* No queued waiter — home keeps the buffer.  Restore ownership
     * sentinel so the next local-rank acquire hits the fast path
     * (is_owner=true) and any foreign LOCK_REQ correctly drives an
     * INVALIDATE_NOTICE → invalidate_transfer round-trip.  Round
     * complete with no successor — clear the in-flight gate so future
     * LOCK_REQs can re-arm. */
    cache->writer_count = 1;
    atomic_store_explicit(&cache->home.rw_holder, arts_global_rank_id,
                          memory_order_release);
    atomic_store_explicit(&cache->home.invalidate_in_flight, 0,
                          memory_order_release);
    return;
  }
  atomic_store_explicit(&cache->home.rw_holder, new_owner,
                        memory_order_release);
  bool has_next = !arts_home_lockreq_queue_empty(&cache->home.pending_rw);
  arts_shared_ptr_t master_h = arts_coherence_acquire_buf(cache);
  struct arts_db_buffer_s *master =
      (struct arts_db_buffer_s *)arts_shared_get(master_h);
  /* master == NULL means a sentinel DB (db_size==0) that never installs
   * a buffer.  Treat it as a no-data GRANT so the new owner's parked
   * waiter still fires. */
  if (master == NULL) {
    /* Sentinel DB: send GRANT with no data; receiver's drain still
     * fires its waiters (no buffer to install). */
    arts_send_db_ownership_response(new_owner, cache->db_guid, /*version=*/0,
                                    has_next, NULL, 0);
  } else {
#ifndef ARTS_MEMORY_MODEL_LRC
    /* RC: monotonic dedup via home->last_sent_version watermark. */
    uint64_t cur =
        arts_rank_u64_map_get(cache->home.last_sent_version, new_owner);
    if (cur >= master->version) {
      arts_send_db_ownership_response(new_owner, cache->db_guid,
                                      master->version, has_next, NULL, 0);
    } else {
      arts_rank_u64_map_set(cache->home.last_sent_version, new_owner,
                            master->version);
      arts_send_db_ownership_response(new_owner, cache->db_guid,
                                      master->version, has_next, master->data,
                                      cache->db_size);
    }
#else
    /* LRC: grant without home-side dedup; owner-side dedup via
     * cache->last_sent_version handles redundant sends. */
    arts_send_db_ownership_response(new_owner, cache->db_guid, master->version,
                                    has_next, master->data, cache->db_size);
#endif
    arts_coherence_release_buf(&master_h);
  }
  /* Round complete: rw_holder advanced, GRANT dispatched.  Clear the
   * in-flight gate.  When has_next was true the GRANT itself encodes
   * the relay instruction — the new owner's drain will withdraw the
   * sentinel and forward ownership to the next queued requester via
   * its own release path (local_transfer_now on home, RELEASE_OWNERSHIP
   * on a remote owner).  We do NOT send a fresh INVALIDATE_NOTICE
   * here: the chain is self-driving from this point. */
  atomic_store_explicit(&cache->home.invalidate_in_flight, 0,
                        memory_order_release);
}

void arts_coh_invalidate_transfer(struct arts_db_cache_s *cache) {
  bool is_home =
      ((unsigned int)arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  if (is_home) {
    arts_coh_local_transfer_now(cache);
  } else {
    unsigned int home_rank = (unsigned int)arts_guid_get_rank(cache->db_guid);
    arts_send_db_ownership_return(home_rank, cache->db_guid);
  }
}

#endif /* !ARTS_MEMORY_MODEL_LC — end of local_transfer_now +                \
          invalidate_transfer */

/* ===== writeback ACK wait (RC and LC) ================================
 *
 * RC and LC use synchronous WRITEBACK with a stack-local semaphore matched
 * by pointer identity.  LRC uses TRANSFER_OWNERSHIP instead and sends no
 * WRITEBACK_ACK, so this helper is excluded from LRC builds. */
#ifndef ARTS_MEMORY_MODEL_LRC

static void await_writeback_ack(sem_t *cv) {
  /* Block on the stack-local semaphore until arts_handler_db_writeback_ack
   * posts it.  No busy-wait: sem_timedwait sleeps the worker.  We re-arm on a
   * coarse cadence only to re-check the shutdown flag — once teardown starts
   * the network receiver stops draining and the ACK never arrives, so the EDT
   * epilogue must not block forever (returning lets the worker exit). */
  for (;;) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 1; /* shutdown re-check cadence, not a timeout on the ACK */
    if (sem_timedwait(cv, &ts) == 0) {
      return; /* ACK arrived (pointer-identity post) */
    }
    if (errno == ETIMEDOUT &&
        arts_atomic_read(&arts_node_info.shutdown_state) != 0) {
      return; /* teardown: ACK will never come */
    }
    /* ETIMEDOUT (not shutting down) or EINTR: re-arm the blocking wait. */
  }
}
#endif /* !ARTS_MEMORY_MODEL_LRC */

/* ===== release_rw =================================================== */

void arts_coh_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped
   * ownership (e.g. an RC-style call against a cache that's already
   * been torn down by a destroy fan-out).  Decrementing would
   * underflow; bail.  Atomic acquire-load avoids a TSan race against
   * concurrent writer_count writes. */
  if (arts_atomic_read(&cache->writer_count) == 0) {
    return;
  }

  /* Acquire current buffer for version bump + WRITEBACK send.  This is
   * a local ref scoped to release_rw — the EDT's own ref (from acquire)
   * is dropped separately by release_one_dep. */
  arts_shared_ptr_t buf_h = arts_coherence_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t new_v = 0;
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
    new_v = arts_atomic_read_u64(&buf->version);
  }

#ifdef ARTS_MEMORY_MODEL_LRC
  /* LRC: drop the buffer ref BEFORE decrementing writer_count, so the slot's
   * cache-hold is the only ref that can keep the buffer alive past
   * writer_count==0 (a concurrent teardown then frees it via the cb deleter
   * with no dangling local ref). */
  if (buf != NULL) {
    arts_coherence_release_buf(&buf_h);
    buf = NULL;
  }
#endif

  unsigned int rest =
      arts_atomic_sub(&cache->writer_count, 1); /* post-decrement value */

  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);

#if defined(ARTS_MEMORY_MODEL_LC)
  /* LC release: every non-home write must be pushed back to home
   * synchronously so home remains canonical before any subsequent
   * acquire can see fresh data.  Home itself needs no WRITEBACK.
   *
   * R3: intermediate release (rest > 0, non-home) — same sync writeback.
   * R4: last release (rest == 0, non-home) — sync writeback to home. */
  if (!is_home && buf != NULL) {
    sem_t cv;
    sem_init(&cv, 0, 0);
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_writeback(home_rank, cache->db_guid, new_v,
                           (uint64_t)(uintptr_t)&cv, ARTS_WB_NORMAL, buf->data,
                           cache->db_size);
    await_writeback_ack(&cv);
    sem_destroy(&cv);
  }
  /* Release the buffer ref held for the version-bump and WRITEBACK read. */
  if (buf != NULL) {
    arts_coherence_release_buf(&buf_h);
  }
#elif defined(ARTS_MEMORY_MODEL_LRC)
  /* LRC: drop our buffer ref BEFORE decrementing writer_count.
   *
   * Invariant: when writer_count reaches 0, no thread may hold an
   * outstanding buffer ref acquired in this call, because a concurrent
   * deferred-free teardown (triggered once refs drain) will free
   * cache->buffer_pool.  Any subsequent release_buf write to that pool
   * would corrupt freed memory.
   *
   * In RC this window does not exist because local_transfer_now restores
   * the sentinel (writer_count = 1) when no pending waiter is queued,
   * keeping writer_count above 0 until the next proper acquire.  LRC has
   * no such sentinel restoration, so we must close the window here by
   * releasing the ref before exposing writer_count == 0.
   * (buf was already released before the writer_count decrement.) */
  /* buf was dropped before decrement; skip further release below. */
  if (rest == 0) {
    if (is_home) {
      /* LRC home-owner release: check if an INVALIDATE_NOTICE arrived while
       * local writers were active and set transfer_pending.  If so, ship
       * TRANSFER_OWNERSHIP now that we are the last releaser. */
      if (atomic_load_explicit(&cache->transfer_pending,
                               memory_order_acquire) == 1u) {
        atomic_store_explicit(&cache->transfer_pending, 0u,
                              memory_order_release);
        arts_coh_lrc_ship_transfer(cache);
      }
      /* If transfer_pending == 0: no INVALIDATE arrived yet; home retains
       * ownership (rw_holder stays self) until a future LOCK_REQ arrives. */
    } else {
      /* LRC R4 (non-home owner, rest == 0): ship TRANSFER_OWNERSHIP if
       * transfer_pending was set by INVALIDATE_NOTICE handler. */
      if (atomic_load_explicit(&cache->transfer_pending,
                               memory_order_acquire) == 1u) {
        atomic_store_explicit(&cache->transfer_pending, 0u,
                              memory_order_release);
        arts_coh_lrc_ship_transfer(cache);
      }
    }
  }
  /* buf was dropped before decrement; new_v not used in LRC. */
  (void)new_v;
#else
  /* RC */
  if (rest == 0) {
    if (is_home) {
      arts_coh_local_transfer_now(cache);
    } else {
      if (buf != NULL) {
        /* RC R4: WRITEBACK_AND_TRANSFER + await ACK (stack-local sem). */
        sem_t cv;
        sem_init(&cv, 0, 0);
        unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
        arts_send_db_writeback(home_rank, cache->db_guid, new_v,
                               (uint64_t)(uintptr_t)&cv, ARTS_WB_AND_TRANSFER,
                               buf->data, cache->db_size);
        await_writeback_ack(&cv);
        sem_destroy(&cv);
      }
    }
  } else if (!is_home && buf != NULL) {
    /* RC R3: intermediate writeback so remote ROs see fresh data + await ACK.
     */
    sem_t cv;
    sem_init(&cv, 0, 0);
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_send_db_writeback(home_rank, cache->db_guid, new_v,
                           (uint64_t)(uintptr_t)&cv, ARTS_WB_NORMAL, buf->data,
                           cache->db_size);
    await_writeback_ack(&cv);
    sem_destroy(&cv);
  }
  /* RC: release buffer ref after the writeback (which reads buf->data). */
  if (buf != NULL) {
    arts_coherence_release_buf(&buf_h);
  }
#endif /* model dispatch */
}

void arts_coh_release_ro(struct arts_db_cache_s *cache) {
  /* RO release is also no-op here — the EDT's buf ref is dropped by
   * release_one_dep's DIST branch via release_buf (matching the
   * acquire_buf in mark_edt_ready_by_guid / acquire_local). */
  (void)cache;
}
