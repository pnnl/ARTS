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
#include "arts/introspection/ArtsIdCounter.h"
#include "arts/arts.h"
#include "arts/introspection/Counter.h"
#include "arts/introspection/Preamble.h"
#include "arts/runtime/Globals.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Simple hash function (FNV-1a) for distributing arts_id values
static inline uint32_t artsIdHash(uint64_t arts_id) {
  uint64_t hash = 14695981039346656037ULL; // FNV offset basis
  hash ^= arts_id;
  hash *= 1099511628211ULL; // FNV prime
  return (uint32_t)(hash & (ARTS_ID_HASH_SIZE - 1));
}

// Get current time in nanoseconds
static inline uint64_t artsGetTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Find or create slot in hash table using linear probing
static inline artsIdMetrics *
artsIdFindSlot(artsIdMetrics *table, uint64_t arts_id, uint64_t *collisions) {
  uint32_t idx = artsIdHash(arts_id);
  uint32_t start_idx = idx;

  // Linear probing - good cache locality
  while (table[idx].valid && table[idx].arts_id != arts_id) {
    idx = (idx + 1) & (ARTS_ID_HASH_SIZE - 1);
    if (idx == start_idx) {
      // Hash table full - critical error
      fprintf(stderr,
              "WARNING: arts_id hash table full! Increase ARTS_ID_HASH_SIZE\n");
      return NULL;
    }
    (*collisions)++;
  }

  // Initialize slot if new
  if (!table[idx].valid) {
    memset(&table[idx], 0, sizeof(artsIdMetrics));
    table[idx].arts_id = arts_id;
    table[idx].valid = true;
  }

  return &table[idx];
}

// Record EDT execution metrics (aggregate mode)
void artsIdRecordEdtMetrics(uint64_t arts_id, uint64_t exec_ns,
                            uint64_t stall_ns, artsIdHashTable *hash_table) {
  if (arts_id == 0)
    return; // Skip if no arts_id set

  artsIdMetrics *slot = artsIdFindSlot(hash_table->edt_metrics, arts_id,
                                       &hash_table->edt_collisions);

  if (slot) {
    // Use atomic operations for thread safety (in case of shared access)
    __sync_fetch_and_add(&slot->invocations, 1);
    __sync_fetch_and_add(&slot->total_exec_ns, exec_ns);
    __sync_fetch_and_add(&slot->total_stall_ns, stall_ns);
  }
}

// Record DB access metrics (aggregate mode)
void artsIdRecordDbMetrics(uint64_t arts_id, uint64_t bytes_local,
                           uint64_t bytes_remote, uint64_t cache_misses,
                           artsIdHashTable *hash_table) {
  // Skip if no arts_id set
  if (arts_id == 0)
    return;

  artsIdMetrics *slot = artsIdFindSlot(hash_table->db_metrics, arts_id,
                                       &hash_table->db_collisions);

  if (slot) {
    __sync_fetch_and_add(&slot->invocations, 1);
    __sync_fetch_and_add(&slot->bytes_local, bytes_local);
    __sync_fetch_and_add(&slot->bytes_remote, bytes_remote);
    __sync_fetch_and_add(&slot->cache_misses, cache_misses);
  }
}

// Capture individual EDT execution (detailed mode)
void artsIdCaptureEdtExecution(uint64_t arts_id, uint64_t exec_ns,
                               uint64_t stall_ns, artsArrayList *captures) {
  // Skip if no arts_id set
  if (arts_id == 0)
    return;

  // Safety check
  if (captures == NULL)
    return;

  artsIdEdtCapture capture = {.arts_id = arts_id,
                              .timestamp_ns = artsGetTimeNs(),
                              .exec_ns = exec_ns,
                              .stall_ns = stall_ns,
                              .node = artsGetCurrentNode(),
                              .thread = artsThreadInfo.threadId};

  artsPushToArrayList(captures, &capture);
}

// Capture individual DB access (detailed mode)
void artsIdCaptureDbAccess(uint64_t arts_id, uint64_t bytes_accessed,
                           uint8_t access_type, artsArrayList *captures) {

  // Skip if no arts_id set
  if (arts_id == 0)
    return;
  // Safety check
  if (captures == NULL)
    return;

  artsIdDbCapture capture = {.arts_id = arts_id,
                             .timestamp_ns = artsGetTimeNs(),
                             .bytes_accessed = bytes_accessed,
                             .node = artsGetCurrentNode(),
                             .access_type = access_type};

  artsPushToArrayList(captures, &capture);
}

// Initialize a hash table to empty state
void artsIdInitHashTable(artsIdHashTable *table) {
  if (!table)
    return;
  memset(table, 0, sizeof(artsIdHashTable));
}

// Reduce (merge) src hash table into dest hash table
// Used for NODE mode reduction across threads
void artsIdReduceHashTables(artsIdHashTable *dest, const artsIdHashTable *src) {
  if (!dest || !src)
    return;

  // Merge EDT metrics
  for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
    if (!src->edt_metrics[i].valid)
      continue;

    uint64_t arts_id = src->edt_metrics[i].arts_id;
    artsIdMetrics *dest_slot =
        artsIdFindSlot(dest->edt_metrics, arts_id, &dest->edt_collisions);

    if (dest_slot) {
      // Accumulate metrics (no need for atomics during reduction - single
      // thread)
      dest_slot->invocations += src->edt_metrics[i].invocations;
      dest_slot->total_exec_ns += src->edt_metrics[i].total_exec_ns;
      dest_slot->total_stall_ns += src->edt_metrics[i].total_stall_ns;
    }
  }

  // Merge DB metrics
  for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
    if (!src->db_metrics[i].valid)
      continue;

    uint64_t arts_id = src->db_metrics[i].arts_id;
    artsIdMetrics *dest_slot =
        artsIdFindSlot(dest->db_metrics, arts_id, &dest->db_collisions);

    if (dest_slot) {
      // Accumulate metrics
      dest_slot->invocations += src->db_metrics[i].invocations;
      dest_slot->bytes_local += src->db_metrics[i].bytes_local;
      dest_slot->bytes_remote += src->db_metrics[i].bytes_remote;
      dest_slot->cache_misses += src->db_metrics[i].cache_misses;
    }
  }
}
