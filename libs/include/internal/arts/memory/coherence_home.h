/* SPDX-License-Identifier: Apache-2.0
 *
 * Home-side metadata helpers for the coherence protocol.
 *
 * The two helpers backing arts_db_home_s.pending_rw and
 * .last_sent_version are deliberately simple — under the pinned
 * single-threaded handler dispatch model (one network thread per
 * rank), home.* is touched by exactly one writer.  We therefore
 * implement:
 *
 *   pending_rw queue:
 *     A trivially-allocated FIFO (head/tail linked list) holding
 *     foreign LOCK_REQ requester ranks awaiting transfer.  Single-
 *     producer, single-consumer in the strict sense — the same
 *     handler thread enqueues on LOCK_REQ arrival and dequeues at
 *     WRITEBACK_AND_TRANSFER / RELEASE_OWNERSHIP / DESTROY_REQ
 *     handler entry.  No locks needed.
 *
 *   last_sent_version map [rank → uint64]:
 *     Dense array sized to arts_global_rank_count.  Per-rank slot
 *     accessed only by the home handler thread.  Cheap, O(1) read
 *     and write, no synchronization needed.  Memory cost per DB =
 *     8 bytes × ranks (≈ 80 KB at 10 K ranks).
 *
 * If we ever lift the single-threaded handler restriction (open
 * question in the plan), both helpers will need an atomic upgrade —
 * the queue to MPMC (Treiber-style or Michael-Scott) and the map to
 * either CAS-loop monotonic update per slot (the per-rank slot
 * already maps to a single 8-byte word) or a hashmap with per-bucket
 * locks.  Today we keep the trivial path. */

#ifndef ARTS_MEMORY_COHERENCE_HOME_H
#define ARTS_MEMORY_COHERENCE_HOME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#include "arts/memory/coherence.h"

/*--- pending_rw FIFO -----------------------------------------------------*/

struct arts_pending_rank_node_s {
  struct arts_pending_rank_node_s *next;
  unsigned int rank;
};

struct arts_lockfree_mpsc_s {
  struct arts_pending_rank_node_s *head;
  struct arts_pending_rank_node_s *tail;
};

struct arts_lockfree_mpsc_s *arts_pending_rw_create(void);
void arts_pending_rw_destroy(struct arts_lockfree_mpsc_s *q);
void arts_pending_rw_enqueue(struct arts_lockfree_mpsc_s *q, unsigned int rank);
/* Dequeue.  Returns true and writes *out_rank on success; false on empty. */
bool arts_pending_rw_dequeue(struct arts_lockfree_mpsc_s *q,
                             unsigned int *out_rank);
bool arts_pending_rw_empty(const struct arts_lockfree_mpsc_s *q);

/*--- last_sent_version dense map ----------------------------------------*/

struct arts_rank_to_u64_map_s {
  uint64_t *slots; /* sized to nranks */
  unsigned int nranks;
};

struct arts_rank_to_u64_map_s *arts_rank_u64_map_create(unsigned int nranks);
void arts_rank_u64_map_destroy(struct arts_rank_to_u64_map_s *m);
uint64_t arts_rank_u64_map_get(const struct arts_rank_to_u64_map_s *m,
                               unsigned int rank);
void arts_rank_u64_map_set(struct arts_rank_to_u64_map_s *m, unsigned int rank,
                           uint64_t value);
/* Monotonic max update — sets m[rank] = max(m[rank], value); returns true if
 * the slot advanced (i.e. value was strictly greater than the old slot).
 * Under single-threaded dispatch this is just `if (v > m[r]) m[r]=v` but
 * keeping the contract tight makes the upgrade-to-CAS path one-line. */
bool arts_rank_u64_map_advance(struct arts_rank_to_u64_map_s *m,
                               unsigned int rank, uint64_t value);

/*--- arts_db_home_s lifecycle -------------------------------------------*/

/* Allocate + initialize a home metadata block.  rw_holder is set by
 * the caller (typically creator_rank under PROP_NONE, or self_rank
 * under NO_ACQUIRE). */
struct arts_db_home_s *arts_db_home_create(unsigned int rw_holder,
                                           unsigned int nranks);
void arts_db_home_destroy(struct arts_db_home_s *home);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_HOME_H */
