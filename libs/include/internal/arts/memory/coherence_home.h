/* SPDX-License-Identifier: Apache-2.0
 *
 * Home-side metadata helpers for the coherence protocol.
 *
 * Concurrency model:
 *   - pending_rw queue: Vyukov MPSC.  Multi-producer (any handler thread
 * enqueues on LOCK_REQ); single-consumer in time (the unique actor holding the
 *     invalidate_in_flight = 1 baton).  Lock-free queue ops.
 *   - last_sent_version map: per-slot atomic.  Each rank slot is an independent
 *     _Atomic(uint64_t) accessed via atomic load/store and CAS-loop
 * monotonic-max for advance.  No cross-slot invariant.
 *   - rw_holder, invalidate_in_flight, destroy_in_flight: _Atomic.  rw_holder
 *     uses acquire/release; gates use acq_rel CAS for the baton.
 */

#ifndef ARTS_MEMORY_COHERENCE_HOME_H
#define ARTS_MEMORY_COHERENCE_HOME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arts/memory/coherence.h"

/* arts_home_lockreq_node_s and arts_home_lockreq_queue_s are defined in
 * coherence.h (included above), where arts_db_home_s embeds the queue. */

void arts_home_lockreq_queue_init(struct arts_home_lockreq_queue_s *q);
void arts_home_lockreq_queue_destroy(struct arts_home_lockreq_queue_s *q);
void arts_home_lockreq_queue_push(struct arts_home_lockreq_queue_s *q,
                                  unsigned int rank);
/* Pop the front rank (single consumer).  Returns true and sets *out_rank
 * on success; returns false when the queue is empty. */
bool arts_home_lockreq_queue_pop(struct arts_home_lockreq_queue_s *q,
                                 unsigned int *out_rank);
bool arts_home_lockreq_queue_empty(const struct arts_home_lockreq_queue_s *q);

/*--- pending_ro_forwards queue ------------------------------------------
 *
 * Deferred RO_REQ (GET_DATA) messages parked while invalidate_in_flight
 * is set.  The queue is Vyukov MPSC in LRC builds; in RC builds the
 * init/destroy are no-ops and push/pop are never called. */
void arts_home_pending_ro_queue_init(struct arts_home_pending_ro_queue_s *q);
void arts_home_pending_ro_queue_destroy(struct arts_home_pending_ro_queue_s *q);
#ifdef ARTS_MEMORY_MODEL_LRC
void arts_home_pending_ro_queue_push(struct arts_home_pending_ro_queue_s *q,
                                     unsigned int requester_rank,
                                     void *waiter_addr);
/* Pop the front entry (single consumer).  Returns true and sets *out_rank /
 * *out_waiter_addr on success; returns false when the queue is empty. */
bool arts_home_pending_ro_queue_pop(struct arts_home_pending_ro_queue_s *q,
                                    unsigned int *out_rank,
                                    void **out_waiter_addr);
#endif /* ARTS_MEMORY_MODEL_LRC */

/*--- last_sent_version dense map ----------------------------------------*/

struct arts_rank_to_u64_map_s {
  _Atomic(uint64_t)
      *slots; /* sized to nranks; each slot is independent atomic */
  unsigned int nranks;
};

struct arts_rank_to_u64_map_s *arts_rank_u64_map_create(unsigned int nranks);
void arts_rank_u64_map_destroy(struct arts_rank_to_u64_map_s *m);
uint64_t arts_rank_u64_map_get(const struct arts_rank_to_u64_map_s *m,
                               unsigned int rank);
void arts_rank_u64_map_set(struct arts_rank_to_u64_map_s *m, unsigned int rank,
                           uint64_t value);
/* Monotonic max update — sets m[rank] = max(m[rank], value) via CAS-loop;
 * returns true if the slot advanced (i.e. value was strictly greater than the
 * old slot).  Safe for concurrent callers on the same slot: loses are
 * harmless because a higher value will have won the CAS. */
bool arts_rank_u64_map_advance(struct arts_rank_to_u64_map_s *m,
                               unsigned int rank, uint64_t value);

/*--- arts_db_home_s lifecycle -------------------------------------------*/

/* Initialize a home metadata block in place (embedded by value in the
 * cache, no separate allocation).  rw_holder is set by the caller
 * (typically creator_rank under PROP_NONE, or self_rank under NO_ACQUIRE). */
void arts_db_home_init(struct arts_db_home_s *home, unsigned int rw_holder,
                       unsigned int nranks);
/* Release a home block's owned sub-resources (queues / maps) in place.
 * Does NOT free the block itself (it lives inside the cache). */
void arts_db_home_teardown(struct arts_db_home_s *home);

#ifdef ARTS_MEMORY_MODEL_LRC
/*--- last_sent_version map serialization (LRC only) ---------------------
 *
 * Used to piggyback the owner-side dedup map onto TRANSFER_OWNERSHIP
 * messages so the new owner can continue skipping redundant DATA_RESPONSE
 * sends without re-learning which ranks already hold a fresh copy.
 *
 * Wire layout (in out buffer, starting at byte 0):
 *   uint32_t count;        number of non-zero (rank, version) pairs
 *   uint32_t pad;          alignment pad
 *   arts_remote_rank_version_pair_s pairs[count];
 *
 * Caller must allocate at least:
 *   sizeof(uint32_t) * 2 + nranks * sizeof(arts_remote_rank_version_pair_s)
 * bytes for the output buffer.
 *
 * NOT thread-safe with concurrent arts_rank_u64_map_advance calls on the
 * same map.  Caller must establish exclusion (writer_count == 0 + handler
 * serialization) before invoking. */
size_t arts_rank_u64_map_serialize(const struct arts_rank_to_u64_map_s *m,
                                   void *out);

/* Build a fresh map from the wire-format buffer produced by
 * arts_rank_u64_map_serialize.  `nranks` sizes the new map's slot array.
 * `size` is the byte length of the buffer (used for bounds assertions in
 * debug builds only; pass the actual received length). */
struct arts_rank_to_u64_map_s *
arts_rank_u64_map_deserialize(const void *in, size_t size, unsigned int nranks);
#endif /* ARTS_MEMORY_MODEL_LRC */

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_HOME_H */
