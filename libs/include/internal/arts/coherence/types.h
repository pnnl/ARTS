/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#ifndef ARTS_MEMORY_COHERENCE_TYPES_H
#define ARTS_MEMORY_COHERENCE_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file coherence/types.h
 * @brief DB-coherence layout dispatcher — selects the per-protocol cache/db.
 *
 * Includes the protocol-agnostic layout (types_common.h), then the
 * protocol-specific cache/db (coherence/<proto>/types.h, chosen by
 * ARTS_PROTOCOL_WRF_VAL), then the of_cache / total_size / stub_size helpers that
 * need the complete cache + db_s types.  struct arts_db_s embeds the per-rank
 * cache (struct arts_db_cache_s) by value as its FIRST member, so each protocol
 * header defines the whole cache + db_s chain together.
 * arts/coherence/coherence.h keeps the protocol function declarations and
 * includes this header for the layouts.
 *
 * @note This is an internal header.  User code should include @c arts.h.
 */

#include "arts/coherence/types_common.h"

/* Protocol-specific cache/db layout.  WRF_VAL carries no ownership grant or RW
 * waiter queue (writer_count is a pure ref count); VAL carries the
 * ownership-cache shape, with the WT/WB write-policy split made inside the
 * header via ARTS_WRITE_POLICY_WB. */
#if defined(ARTS_PROTOCOL_EXCL)
#include "arts/coherence/excl/types.h"
#elif defined(ARTS_PROTOCOL_INV)
#include "arts/coherence/inv/types.h"
#elif defined(ARTS_PROTOCOL_WRF_VAL)
#include "arts/coherence/wrf_val/types.h"
#else
#include "arts/coherence/val/types.h"
#endif

/* ========================================================================= */
/** @addtogroup internal_db_structs
 *  Helpers below need the complete cache + db_s types from the protocol header.
 *  @{ */

/* Recover the wrapping struct arts_db_s from a coherence cache pointer.  cache
 * is the FIRST member of arts_db_s; container_of degenerates to the cache
 * address but is written as container_of for correctness-by-construction (and
 * to match the master plan's home-access idiom).  NULL-safe. */
static inline struct arts_db_s *arts_db_of_cache(struct arts_db_cache_s *c) {
  if (c == NULL) {
    return NULL;
  }
  return ARTS_CONTAINER_OF(c, struct arts_db_s, cache);
}

/* Total allocation size of a DB = wrapping struct + user payload.  A DB always
 * carries its own length in the cache (cache.db_size, set for every db_type at
 * create); there is no separate object header storing it. */
static inline uint64_t arts_db_total_size(const struct arts_db_s *db) {
  return sizeof(struct arts_db_s) + db->cache.db_size;
}

/* Footprint of a non-home / WB / creator-remote DB stub: the cache prefix +
 * db_type + home_initialized, stopping before the home-directory queues/maps
 * (which only the GUID home rank ever touches).  home_initialized MUST be in
 * bounds: the cache destructor reads it on EVERY free to decide whether to tear
 * down the home directory — on a stub it reads (zeroed) false and skips the
 * teardown.  Allocating this much keeps db_type + home_initialized in bounds
 * while shedding the bulky home directory (MPSC queue + rank->u64 map +
 * cached_ranks bitset).  The home rank allocates the full sizeof(struct
 * arts_db_s) instead.
 * The stub ends at the first home-directory field after home_initialized
 * (protocol-dependent: rw_holder for the ownership protocols, cached_version
 * for WRF_VAL). */
static inline uint64_t arts_db_cache_stub_size(void) {
#if defined(ARTS_PROTOCOL_WRF_VAL)
  return offsetof(struct arts_db_s, cached_version);
#elif defined(ARTS_PROTOCOL_EXCL)
  return offsetof(struct arts_db_s, lock_state);
#elif defined(ARTS_PROTOCOL_INV)
  return offsetof(struct arts_db_s, dir_state);
#else
  return offsetof(struct arts_db_s, rw_holder);
#endif
}

/** @} */ /* end internal_db_structs */

/* ========================================================================= */
#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_TYPES_H */
