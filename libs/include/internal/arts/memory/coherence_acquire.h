/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence-protocol acquire path.
 *
 * Implements the 8-case dispatcher (HOME × OWNER × {RO, RW}), the
 * remote acquire helpers (LOCK_REQ for RW, GET_DATA for RO), the
 * GRANT/DATA_RESPONSE-side drain routines, and the lazy first-touch
 * cache_s allocation for foreign ranks.
 *
 * Return contract:
 *   On success the call returns ARTS_DB_ACQUIRE_OK and writes the
 *   buffer-data pointer into *out_data.  Cases 1/3/5/6/2-fast-path
 *   take the synchronous path and finish before returning.
 *   ARTS_DB_ACQUIRE_PARK indicates the EDT was parked (cases 4/7/8 +
 *   case 6 fall-through).  The caller leaves the dep slot in the
 *   "depc_needed not yet decremented" state; the protocol's
 *   trigger path will fill the slot and decrement when the
 *   ownership/data lands.  ARTS_DB_ACQUIRE_DESTROYED tells the
 *   caller to surface destroy semantics to the EDT (caller's
 *   problem to decide). */

#ifndef ARTS_MEMORY_COHERENCE_ACQUIRE_H
#define ARTS_MEMORY_COHERENCE_ACQUIRE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arts/memory/coherence.h"

typedef enum {
  ARTS_DB_ACQUIRE_OK = 0,
  ARTS_DB_ACQUIRE_PARK,
  ARTS_DB_ACQUIRE_DESTROYED,
} arts_db_acquire_result_t;

/* Lazy first-touch: allocate cache_s on this rank if absent.  Used
 * by the dispatcher when arts_route_table_lookup_db_safe returns NULL.
 * Returns NULL only if a concurrent destroy raced ahead. */
struct arts_db_cache_s *arts_coh_lazy_install_cache_s(arts_guid_t db_guid,
                                                      uint64_t db_size);

/* 8-case acquire dispatcher.  edt_guid and slot identify the parked
 * EDT's dep slot if the path requires parking.  The caller is
 * responsible for the route_table ref on the underlying DB entry —
 * this function neither acquires nor releases it. */
arts_db_acquire_result_t arts_coh_db_acquire(struct arts_db_cache_s *cache,
                                             arts_guid_t edt_guid,
                                             unsigned int slot,
                                             arts_db_access_mode_t mode,
                                             void **out_data);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_ACQUIRE_H */
