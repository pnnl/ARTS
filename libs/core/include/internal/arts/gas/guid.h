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
 * | Bits 63–56 | Bits 55–40 | Bits 39–0  |
 * |:----------:|:----------:|:----------:|
 * | type  (8)  | rank (16)  | key  (40)  |
 *
 * - **type** — @ref arts_type_t tag identifying the object kind.
 * - **rank** — Node rank that owns the object (up to 65 535 nodes).
 * - **key**  — Node-local key (up to ~1 trillion unique objects per node).
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

#define ARTS_GUID_KEY_BITS  40
#define ARTS_GUID_RANK_BITS 16
#define ARTS_GUID_TYPE_BITS 8

/* ── Field positions (bit offset from LSB) ─────────────────────────────── */

#define ARTS_GUID_KEY_SHIFT  0
#define ARTS_GUID_RANK_SHIFT ARTS_GUID_KEY_BITS /* 40 */
#define ARTS_GUID_TYPE_SHIFT                       \
  (ARTS_GUID_KEY_BITS + ARTS_GUID_RANK_BITS) /* 56 \
                                              */

/* ── Per-field masks (in field-local position) ─────────────────────────── */

#define ARTS_GUID_KEY_MASK  (((uint64_t)1 << ARTS_GUID_KEY_BITS) - 1)
#define ARTS_GUID_RANK_MASK (((uint64_t)1 << ARTS_GUID_RANK_BITS) - 1)
#define ARTS_GUID_TYPE_MASK (((uint64_t)1 << ARTS_GUID_TYPE_BITS) - 1)

/* ── Extraction macros ─────────────────────────────────────────────────── */

/** Extract the 40-bit key from a GUID (bits 39–0). */
#define ARTS_GUID_GET_KEY(g) ((uint64_t)(g) & ARTS_GUID_KEY_MASK)

/** Extract the 16-bit rank from a GUID (bits 55–40). */
#define ARTS_GUID_GET_RANK(g) \
  (((uint64_t)(g) >> ARTS_GUID_RANK_SHIFT) & ARTS_GUID_RANK_MASK)

/** Extract the 8-bit type tag from a GUID (bits 63–56). */
#define ARTS_GUID_GET_TYPE(g) \
  (((uint64_t)(g) >> ARTS_GUID_TYPE_SHIFT) & ARTS_GUID_TYPE_MASK)

/* ── Construction macro ────────────────────────────────────────────────── */

/** Build a GUID from its three components. */
#define ARTS_GUID_MAKE(type, rank, key)                    \
  ((arts_guid_t)(((uint64_t)(key) & ARTS_GUID_KEY_MASK) |  \
                 (((uint64_t)(rank) & ARTS_GUID_RANK_MASK) \
                  << ARTS_GUID_RANK_SHIFT) |               \
                 (((uint64_t)(type) & ARTS_GUID_TYPE_MASK) \
                  << ARTS_GUID_TYPE_SHIFT)))

/**
 * @brief Create a GUID for a given node rank and type.
 *
 * Allocates a new key on @p route and packs it with @p type.
 *
 * @param route Target node rank.
 * @param type  Object type tag (@ref arts_type_t).
 * @return A new GUID.
 */
arts_guid_t arts_guid_create_for_rank(unsigned int route, unsigned int type);

/** Initialize the per-node GUID key generator. */
void arts_guid_key_generator_init();

/** Enable the global (post-init) GUID allocation path. */
void set_global_guid_on();

/** Switch GUID generator to the post-parallel-start mode. */
void set_guid_generator_after_parallel_start();

/**
 * @brief Extract the 40-bit key portion of a GUID.
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
 * @param type      Type tag for every GUID in the range.
 * @param size      Number of GUIDs to allocate.
 * @param route     Target node rank.
 * @param hash_size Hash-table bucket count for alignment.
 * @return The hash-aligned start GUID, or @c NULL_GUID on failure.
 */
arts_guid_t arts_guid_reserve_range_hash(arts_type_t type, unsigned int size,
                                         unsigned int route,
                                         unsigned int hash_size);

#ifdef __cplusplus
}
#endif

#endif
