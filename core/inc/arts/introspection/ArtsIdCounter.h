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
#ifndef ARTS_ID_COUNTER_H
#define ARTS_ID_COUNTER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arts/utils/ArrayList.h"
#include <stdbool.h>
#include <stdint.h>

// Forward declaration
struct artsRuntimePrivate;

// Hash table size (must be power of 2 for fast modulo)
#ifndef ARTS_ID_HASH_SIZE
#define ARTS_ID_HASH_SIZE 256
#endif

// Per-arts_id aggregate metrics
typedef struct {
  uint64_t arts_id;        // Key
  uint64_t invocations;    // Number of invocations
  uint64_t total_exec_ns;  // Total execution time (nanoseconds)
  uint64_t total_stall_ns; // Total stall time (nanoseconds)
  uint64_t bytes_local;    // For DBs: local bytes accessed
  uint64_t bytes_remote;   // For DBs: remote bytes accessed
  uint64_t cache_misses;   // For DBs: cache misses
  bool valid;              // Slot occupied
} artsIdMetrics;

// Per-thread hash table for aggregate metrics
typedef struct {
  artsIdMetrics edt_metrics[ARTS_ID_HASH_SIZE]; // EDT metrics by arts_id
  artsIdMetrics db_metrics[ARTS_ID_HASH_SIZE];  // DB metrics by arts_id
  uint64_t edt_collisions; // Stats: hash collisions for EDTs
  uint64_t db_collisions;  // Stats: hash collisions for DBs
} artsIdHashTable;

// Per-invocation capture structure for detailed EDT tracking
typedef struct {
  uint64_t arts_id;      // Which arts_id
  uint64_t timestamp_ns; // When it executed
  uint64_t exec_ns;      // Execution time
  uint64_t stall_ns;     // Stall time
  uint32_t node;         // Which node
  uint32_t thread;       // Which thread
} artsIdEdtCapture;

// Per-invocation capture structure for detailed DB tracking
typedef struct {
  uint64_t arts_id;        // Which arts_id
  uint64_t timestamp_ns;   // When accessed
  uint64_t bytes_accessed; // How much data
  uint32_t node;           // Which node
  uint8_t access_type;     // READ (0) or WRITE (1)
} artsIdDbCapture;

// Function declarations

// Aggregate metrics recording (hash table based)
void artsIdRecordEdtMetrics(uint64_t arts_id, uint64_t exec_ns,
                            uint64_t stall_ns, artsIdHashTable *hash_table);
void artsIdRecordDbMetrics(uint64_t arts_id, uint64_t bytes_local,
                           uint64_t bytes_remote, uint64_t cache_misses,
                           artsIdHashTable *hash_table);

// Detailed per-invocation captures (ArrayList based)
void artsIdCaptureEdtExecution(uint64_t arts_id, uint64_t exec_ns,
                               uint64_t stall_ns, artsArrayList *captures);
void artsIdCaptureDbAccess(uint64_t arts_id, uint64_t bytes_accessed,
                           uint8_t access_type, artsArrayList *captures);

// Reduction functions for NODE mode (merge multiple hash tables)
void artsIdReduceHashTables(artsIdHashTable *dest, const artsIdHashTable *src);
void artsIdInitHashTable(artsIdHashTable *table);

#ifdef __cplusplus
}
#endif

#endif // ARTS_ID_COUNTER_H
