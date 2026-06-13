/* SPDX-License-Identifier: Apache-2.0
 *
 * Home-side metadata helpers for the coherence protocol.
 *
 * Concurrency model:
 *   - pending_rw queue: Vyukov MPSC.  Multi-producer (any handler thread
 * enqueues on OWNERSHIP_REQUEST); single-consumer in time (the unique actor
 * holding the invalidate_in_flight = 1 baton).  Lock-free queue ops.
 *   - last_sent_version map: per-slot atomic.  Each rank slot is an independent
 *     _Atomic(uint64_t) accessed via atomic load/store and CAS-loop
 * monotonic-max for advance.  No cross-slot invariant.
 *   - rw_holder, invalidate_in_flight: _Atomic.  rw_holder uses
 *     acquire/release; the invalidate_in_flight gate uses acq_rel CAS for the
 *     baton.
 */

#ifndef ARTS_MEMORY_COHERENCE_HOME_H
#define ARTS_MEMORY_COHERENCE_HOME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arts/coherence/coherence.h"
/* struct arts_rank_bitset_s definition (embedded by value in arts_db_s).
 * The rank-bitset FUNCTION declarations are folded into this header below;
 * the STRUCT lives in rank_bitset.h because coherence_types.h includes
 * it directly to lay out arts_db_s and must not depend on this header. */
#include "arts/rank_bitset.h"

/* arts_home_lockreq_node_s and arts_home_lockreq_queue_s are defined in
 * coherence.h (included above), where arts_db_s embeds the queue. */

void arts_home_lockreq_queue_init(struct arts_home_lockreq_queue_s *q);
void arts_home_lockreq_queue_destroy(struct arts_home_lockreq_queue_s *q);
void arts_home_lockreq_queue_push(struct arts_home_lockreq_queue_s *q,
                                  unsigned int rank);
/* Pop the front rank (single consumer).  Returns true and sets *out_rank
 * on success; returns false when the queue is empty. */
bool arts_home_lockreq_queue_pop(struct arts_home_lockreq_queue_s *q,
                                 unsigned int *out_rank);
/* Peek the front (oldest) requester rank without popping. Single consumer (the
 * baton holder). Returns true + sets *out_rank when non-empty. */
bool arts_home_lockreq_queue_peek(const struct arts_home_lockreq_queue_s *q,
                                  unsigned int *out_rank);
bool arts_home_lockreq_queue_empty(const struct arts_home_lockreq_queue_s *q);

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

/*--- rank bit-set ----------------------------------------------------
 *
 * Bit-packed atomic rank bit-set, sized to the cluster's rank count.  Used
 * only in lazy builds — the eager protocol reuses the per-rank version map
 * for the same purpose (set membership = nonzero entry).  The struct
 * arts_rank_bitset_s definition lives in rank_bitset.h (included
 * directly by coherence_types.h, which embeds it by value in
 * arts_db_s). */

void arts_rank_bitset_init(struct arts_rank_bitset_s *r, unsigned int nranks);
void arts_rank_bitset_destroy(struct arts_rank_bitset_s *r);
/* Set bit for rank; returns true if the bit was previously clear (first-time
 * set), false if already set.  Safe for concurrent callers. */
bool arts_rank_bitset_set(struct arts_rank_bitset_s *r, unsigned int rank);
/* Iterate over all set bits, invoking cb(rank, ctx) for each.  The snapshot
 * is acquired per-word; callers must ensure no concurrent set() during
 * iteration.  The destroy fan-out satisfies this as the home-side single
 * actor: by the time it scans, the DB's route-table slot is already absent,
 * so no later acquire can register a new reader. */
void arts_rank_bitset_for_each(const struct arts_rank_bitset_s *r,
                               void (*cb)(unsigned int rank, void *ctx),
                               void *ctx);

/*--- home-directory lifecycle (inlined in arts_db_s) --------------------*/

/* Initialize the home-directory fields inlined in struct arts_db_s (no separate
 * allocation).  rw_holder is set by the caller (typically creator_rank under
 * PROP_NONE, or self_rank under NO_ACQUIRE). */
void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks);
/* Release the home-directory owned sub-resources (queues / maps) in place.
 * Does NOT free the descriptor (the fields live inside the arts_db_s). */
void arts_db_home_teardown(struct arts_db_s *db);

/*--- last_sent_version map serialization (defined in coherence/lazy.c) -
 *
 * Used to piggyback the owner-side dedup map onto TRANSFER_OWNERSHIP
 * messages so the new owner can continue skipping redundant DATA_RESPONSE
 * sends without re-learning which ranks already hold a fresh copy.  Only the
 * lazy protocol TU defines these; the declarations are unconditional so this
 * header carries no coherence-model preprocessor logic.
 *
 * Wire layout (in out buffer, starting at byte 0):
 *   uint32_t count;        number of non-zero (rank, version) pairs
 *   uint32_t pad;          alignment pad
 *   arts_msg_rank_version_pair_s pairs[count];
 *
 * Caller must allocate at least:
 *   sizeof(uint32_t) * 2 + nranks * sizeof(arts_msg_rank_version_pair_s)
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

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_HOME_H */
