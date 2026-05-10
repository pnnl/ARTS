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
#ifndef ARTS_GAS_GUID_H
#define ARTS_GAS_GUID_H
#include "arts.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file guid.h
 * @brief GUID layout and low-level GUID manipulation functions.
 *
 * A GUID is a 64-bit value composed of three packed fields:
 *
 * | Bits 63–62 | Bits 61–48 | Bits 47–0  |
 * |:----------:|:----------:|:----------:|
 * | type  (2)  | rank (14)  | key  (48)  |
 *
 * - **type** — @ref arts_guid_kind_t tag identifying the object kind.
 * - **rank** — Node rank that owns the object (up to 16 384 nodes).
 * - **key**  — Node-local key (up to ~281 trillion unique objects per node).
 *
 * Key occupies the least-significant bits so that GUID-range arithmetic
 * reduces to plain integer addition: @c start_guid+i yields the i-th GUID.
 *
 * All field access uses portable shift/mask macros — no C bitfield structs,
 * no endianness dependence.
 *
 * @note This is an internal header.  User code should include @c arts.h.
 */

/* ── GUID field dimensions ─────────────────────────────────────────────── */

#define ARTS_GUID_KEY_BITS 48
#define ARTS_GUID_RANK_BITS 14
#define ARTS_GUID_TYPE_BITS 2

/* ── Field positions (bit offset from LSB) ─────────────────────────────── */

#define ARTS_GUID_KEY_SHIFT 0
#define ARTS_GUID_RANK_SHIFT ARTS_GUID_KEY_BITS /* 48 */
#define ARTS_GUID_TYPE_SHIFT                                                   \
  (ARTS_GUID_KEY_BITS + ARTS_GUID_RANK_BITS) /* 62                             \
                                              */

/* ── Per-field masks (in field-local position) ─────────────────────────── */

#define ARTS_GUID_KEY_MASK (((uint64_t)1 << ARTS_GUID_KEY_BITS) - 1)
#define ARTS_GUID_RANK_MASK (((uint64_t)1 << ARTS_GUID_RANK_BITS) - 1)
#define ARTS_GUID_TYPE_MASK (((uint64_t)1 << ARTS_GUID_TYPE_BITS) - 1)

/* ── Extraction macros ─────────────────────────────────────────────────── */

/** Extract the 48-bit key from a GUID (bits 47–0). */
#define ARTS_GUID_GET_KEY(g) ((uint64_t)(g) & ARTS_GUID_KEY_MASK)

/** Extract the 14-bit rank from a GUID (bits 61–48). */
#define ARTS_GUID_GET_RANK(g)                                                  \
  (((uint64_t)(g) >> ARTS_GUID_RANK_SHIFT) & ARTS_GUID_RANK_MASK)

/** Extract the 2-bit type tag from a GUID (bits 63–62). */
#define ARTS_GUID_GET_TYPE(g)                                                  \
  (((uint64_t)(g) >> ARTS_GUID_TYPE_SHIFT) & ARTS_GUID_TYPE_MASK)

/* ── Construction macro ────────────────────────────────────────────────── */

/** Build a GUID from its three components. */
#define ARTS_GUID_MAKE(type, rank, key)                                        \
  ((arts_guid_t)(((uint64_t)(key) & ARTS_GUID_KEY_MASK) |                      \
                 (((uint64_t)(rank) & ARTS_GUID_RANK_MASK)                     \
                  << ARTS_GUID_RANK_SHIFT) |                                   \
                 (((uint64_t)(type) & ARTS_GUID_TYPE_MASK)                     \
                  << ARTS_GUID_TYPE_SHIFT)))

/**
 * @brief Create a GUID for a given node rank and type.
 *
 * Allocates a new key on @p route and packs it with @p type.
 *
 * @param route Target node rank.
 * @param type  Object kind tag (@ref arts_guid_kind_t), passed as unsigned int
 *              for internal flexibility (callers may supply pre-cast values).
 * @return A new GUID.
 */
arts_guid_t arts_guid_create_for_rank(unsigned int rank, unsigned int type);

/** Initialize the per-node GUID key generator. */
void arts_guid_key_generator_init();

/** Enable the global (post-init) GUID allocation path. */
void set_global_guid_on();

/** Switch GUID generator to the post-parallel-start mode. */
void set_guid_generator_after_parallel_start();

/**
 * @brief Extract the 48-bit key portion of a GUID.
 *
 * @param guid GUID to query.
 * @return The node-local key value.
 */
uint64_t arts_guid_get_key(arts_guid_t guid);

/**
 * @brief Hash a GUID's key for routing-table lookups.
 *
 * @param guid GUID to hash.
 * @return Hash value derived from the key field.
 */
uint64_t arts_guid_hash_key(arts_guid_t guid);

/**
 * @brief Reserve a contiguous range of hash-aligned GUIDs (internal use).
 *
 * Over-allocates by @p hash_size to find a hash-aligned start GUID.
 *
 * @param kind      Kind tag for every GUID in the range.
 * @param size      Number of GUIDs to allocate.
 * @param rank      Target node rank.
 * @param hash_size Hash-table bucket count for alignment.
 * @return The hash-aligned start GUID, or @c NULL_GUID on failure.
 */
arts_guid_t arts_guid_reserve_range_hash(arts_guid_kind_t kind,
                                         unsigned int size, unsigned int rank,
                                         unsigned int hash_size);

/** Sentinel rank value marking a "distributed" GUID range.  Stored in the
 *  rank field of a range GUID returned by
 *  @c arts_guid_reserve_range(kind, size, ARTS_HINT_ROUND_ROBIN).
 *  @c arts_guid_from_index / @c arts_guid_index_from detect this marker
 *  and place individual GUIDs round-robin across ranks (home = idx % nrank).
 *  Must not collide with @ref ARTS_CXL_RANK (0x3FFF).
 *  14-bit rank field => max real rank = 16382 (0x3FFE - 1). */
#define ARTS_DISTRIBUTED_RANK 0x3FFE

/* ── CXL GUID helpers ───────────────────────────────────────────────────────
 */

#ifdef ARTS_USE_CXL

#include <stdbool.h>

/** Base virtual address of the CXL FAM mapping. */
#define ARTS_CXL_BASE_ADDR 0x200000000000ULL

/** Sentinel rank value that identifies CXL-encoded GUIDs.
 *  14-bit rank field => use top value (0x3FFF). */
#define ARTS_CXL_RANK 0x3FFF

/** Extract a CXL pointer from a CXL-encoded GUID. */
static inline void *arts_cxl_get_ptr(arts_guid_t guid) {
  return (void *)(ARTS_CXL_BASE_ADDR + ARTS_GUID_GET_KEY(guid));
}

/** Build a CXL-encoded GUID from a CXL pointer (kind = ARTS_GUID_DB). */
static inline arts_guid_t arts_cxl_make_guid(void *ptr) {
  uint64_t offset = (uint64_t)(uintptr_t)ptr - ARTS_CXL_BASE_ADDR;
  return ARTS_GUID_MAKE(ARTS_GUID_DB, ARTS_CXL_RANK, offset);
}

/** Return true if the GUID uses CXL pointer encoding. */
static inline bool arts_guid_is_cxl(arts_guid_t guid) {
  return ARTS_GUID_GET_RANK(guid) == ARTS_CXL_RANK;
}

#endif /* ARTS_USE_CXL */

#ifdef __cplusplus
}
#endif

#endif
