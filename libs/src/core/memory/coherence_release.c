/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence release path implementation.  See the design plan for
 * the four release cases (R1-R4) and the WRITEBACK_ACK seq
 * rendezvous protocol.
 *
 * Wait/wake mechanism: per-cache (writeback_seq, writeback_acked_seq)
 * pair.  release_rw atomically claims a fresh seq by bumping
 * writeback_seq, embeds it in the WRITEBACK packet, and spin-waits
 * on writeback_acked_seq >= my_seq.  ACK handler does an atomic
 * monotonic-max on writeback_acked_seq.  Multiple releases on the
 * same cache rendezvous independently because each holds a unique
 * seq.
 */

#include "arts/memory/coherence_release.h"

#include <sched.h>

#include "arts/gas/route_table.h"
#include "arts/memory/coherence_buffer.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/coherence_home.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"

/* ===== local_transfer / invalidate_transfer (strong overrides) ===== */

void arts_coh_local_transfer_now(struct arts_db_cache_s *cache) {
  unsigned int new_owner;
  if (!arts_pending_rw_dequeue(cache->home->pending_rw, &new_owner)) {
    /* No queued waiter — home keeps the buffer.  Restore ownership
     * sentinel so the next local-rank acquire hits the fast path
     * (is_owner=true) and any foreign LOCK_REQ correctly drives an
     * INVALIDATE_NOTICE → invalidate_transfer round-trip.  Round
     * complete with no successor — clear the in-flight gate so future
     * LOCK_REQs can re-arm. */
    cache->writer_count = 1;
    cache->home->rw_holder = arts_global_rank_id;
    arts_atomic_swap(&cache->home->invalidate_in_flight, 0);
    return;
  }
  cache->home->rw_holder = new_owner;
  bool has_next = !arts_pending_rw_empty(cache->home->pending_rw);
  struct arts_db_buffer_s *master = arts_coherence_acquire_buf(cache);
  /* master == NULL has two distinct causes: (a) a destroy raced past
   * the precheck and detached the buffer, or (b) sentinel DBs
   * (db_size==0) never install a buffer in the first place.  Treat the
   * latter as a no-data GRANT so the new owner's parked waiter still
   * fires; only the destroy case bails via DESTROY_NOTIFY. */
  if (master == NULL) {
    if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
      arts_coh_send_destroy_notify(new_owner, cache->db_guid);
      /* Round terminated by destroy — clear gate; no successor to arm. */
      arts_atomic_swap(&cache->home->invalidate_in_flight, 0);
      return;
    }
    /* Sentinel DB: send GRANT with no data; receiver's drain still
     * fires its waiters (no buffer to install). */
    arts_coh_send_grant(new_owner, cache->db_guid, /*version=*/0, has_next,
                        NULL, 0);
  } else {
    /* Design plan §local_transfer (inline grant_to): monotonic dedup.
     * Source bytes from master->data (canonical FAM payload). */
    uint64_t cur =
        arts_rank_u64_map_get(cache->home->last_sent_version, new_owner);
    if (cur >= master->version) {
      arts_coh_send_grant(new_owner, cache->db_guid, master->version, has_next,
                          NULL, 0);
    } else {
      arts_rank_u64_map_set(cache->home->last_sent_version, new_owner,
                            master->version);
      arts_coh_send_grant(new_owner, cache->db_guid, master->version, has_next,
                          master->data, cache->db_size);
    }
    arts_coherence_release_buf(cache, master);
  }
  /* Round complete: rw_holder advanced, GRANT dispatched.  Clear the
   * in-flight gate.  When has_next was true the GRANT itself encodes
   * the relay instruction — the new owner's drain will withdraw the
   * sentinel and forward ownership to the next queued requester via
   * its own release path (local_transfer_now on home, RELEASE_OWNERSHIP
   * on a remote owner).  We do NOT send a fresh INVALIDATE_NOTICE
   * here: the chain is self-driving from this point. */
  arts_atomic_swap(&cache->home->invalidate_in_flight, 0);
}

void arts_coh_invalidate_transfer(struct arts_db_cache_s *cache) {
  bool is_home =
      ((unsigned int)arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  if (is_home) {
    arts_coh_local_transfer_now(cache);
  } else {
    unsigned int home_rank = (unsigned int)arts_guid_get_rank(cache->db_guid);
    arts_coh_send_release_ownership(home_rank, cache->db_guid);
  }
}

/* ===== writeback_ack_signal: monotonic-max on acked_seq =========== */

void arts_coh_writeback_ack_signal(arts_guid_t db_guid, uint64_t seq) {
  /* The ACK handler runs on the network thread; the spinner is on a
   * worker thread.  Both touch cache.writeback_acked_seq via
   * arts_atomic_*; the atomic CAS serializes the publish. */
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(db_guid);
  if (cache == NULL) {
    return;
  }
  /* Monotonic-max via CAS-loop: only advance if our seq is higher. */
  while (1) {
    uint64_t cur = arts_atomic_read_u64(&cache->writeback_acked_seq);
    if (cur >= seq) {
      break;
    }
    if (arts_atomic_cswap_u64(&cache->writeback_acked_seq, cur, seq) == cur) {
      break;
    }
  }
}

static void await_writeback_ack(struct arts_db_cache_s *cache, uint64_t seq) {
  /* Spin-wait for ACK.  release_rw runs in the EDT epilogue (after
   * TIME_EDT_EXEC_STOP) so arts_yield's TIME_EDT_EXEC_STOP/_START
   * pairing breaks here — use plain sched_yield() instead.  Forward
   * progress: the network receiver thread runs handle_writeback_ack
   * independently and bumps writeback_acked_seq via writeback_ack_signal;
   * this worker thread just polls. */
  while (arts_atomic_read_u64(&cache->writeback_acked_seq) < seq) {
    sched_yield();
    /* Short-circuit on destroy: handle_writeback's destroy precheck
     * sends WRITEBACK_ACK back already, but if cache was torn down on
     * this rank too, no one signals — break out for shutdown. */
    if (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE) {
      return;
    }
    /* Shutdown bail-out: once the runtime starts tearing down, network
     * threads stop draining and the ACK will never arrive.  Returning
     * here leaks the seq slot but lets the EDT epilogue finish so the
     * worker thread can exit instead of being cancelled mid-spin
     * (which would leave the cache->buffer in a UAF window when the
     * route_table tears down concurrently). */
    if (arts_atomic_read(&arts_node_info.shutdown_state) != 0) {
      return;
    }
  }
}

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
   * is dropped separately by release_one_dep via release_buf. */
  struct arts_db_buffer_s *buf = arts_coherence_acquire_buf(cache);
  uint64_t new_v = 0;
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
    new_v = arts_atomic_read_u64(&buf->version);
  }

  unsigned int rest =
      arts_atomic_sub(&cache->writer_count, 1); /* post-decrement value */

  bool skip_network =
      (arts_atomic_read(&cache->destroy_state) != ARTS_DB_DESTROY_NONE);
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);

  if (rest == 0) {
    if (is_home) {
      if (!skip_network) {
        arts_coh_local_transfer_now(cache);
      }
    } else if (!skip_network && buf != NULL) {
      /* R4: WRITEBACK_AND_TRANSFER + await ACK (design plan §release_rw). */
      uint64_t my_seq = arts_atomic_add_u64(&cache->writeback_seq, 1);
      unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
      arts_coh_send_writeback(home_rank, cache->db_guid, new_v, my_seq,
                              ARTS_WB_AND_TRANSFER, buf->data, cache->db_size);
      await_writeback_ack(cache, my_seq);
    }
  } else if (!is_home && !skip_network && buf != NULL) {
    /* R3: intermediate writeback so remote ROs see fresh data + await ACK. */
    uint64_t my_seq = arts_atomic_add_u64(&cache->writeback_seq, 1);
    unsigned int home_rank = arts_guid_get_rank(cache->db_guid);
    arts_coh_send_writeback(home_rank, cache->db_guid, new_v, my_seq,
                            ARTS_WB_NORMAL, buf->data, cache->db_size);
    await_writeback_ack(cache, my_seq);
  }

  if (buf != NULL) {
    arts_coherence_release_buf(cache, buf);
  }

  if (skip_network) {
    extern void arts_coh_try_finalize_destroy(struct arts_db_cache_s * cache);
    arts_coh_try_finalize_destroy(cache);
  }
}

void arts_coh_release_ro(struct arts_db_cache_s *cache) {
  /* RO release is also no-op here — the EDT's buf ref is dropped by
   * release_one_dep's DIST branch via release_buf (matching the
   * acquire_buf in mark_edt_ready_by_guid / acquire_local). */
  (void)cache;
}
