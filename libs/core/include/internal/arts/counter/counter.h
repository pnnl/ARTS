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
#ifndef ARTS_COUNTER_COUNTER_H
#define ARTS_COUNTER_COUNTER_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#include "arts/arts_defs.h"
#include "arts/counter/Preamble.h"

// X-macro: Define all counter types in one place.
// Format: X(counterName)
// Both the enum and string array are generated from this single list.
#define ARTS_COUNTER_LIST                                                      \
  X(EDT_COUNTER)                                                               \
  X(SLEEP_COUNTER)                                                             \
  X(SIGNAL_EVENT_COUNTER)                                                      \
  X(SIGNAL_PERSISTENT_EVENT_COUNTER)                                           \
  X(SIGNAL_EDT_COUNTER)                                                        \
  X(EDT_CREATE_COUNTER)                                                        \
  X(EVENT_CREATE_COUNTER)                                                      \
  X(PERSISTENT_EVENT_CREATE_COUNTER)                                           \
  X(DB_CREATE_COUNTER)                                                         \
  X(SMART_DB_CREATE_COUNTER)                                                   \
  X(MALLOC_MEMORY)                                                             \
  X(CALLOC_MEMORY)                                                             \
  X(FREE_MEMORY)                                                               \
  X(GUID_ALLOC_COUNTER)                                                        \
  X(GUID_LOOKUP_COUNTER)                                                       \
  X(GET_DB_COUNTER)                                                            \
  X(PUT_DB_COUNTER)                                                            \
  X(CONTEXT_SWITCH)                                                            \
  X(YIELD)                                                                     \
  X(REMOTE_MEMORY_MOVE)                                                        \
  X(MEMORY_FOOTPRINT)                                                          \
  X(EDT_RUNNING_TIME)                                                          \
  X(NUM_EDTS_CREATED)                                                          \
  X(NUM_EDTS_ACQUIRED)                                                         \
  X(NUM_EDTS_FINISHED)                                                         \
  X(REMOTE_BYTES_SENT)                                                         \
  X(REMOTE_BYTES_RECEIVED)                                                     \
  X(NUM_DBS_CREATED)                                                           \
  /* Acquire-Mode counters */                                                  \
  X(ACQUIRE_READ_MODE)                                                         \
  X(ACQUIRE_WRITE_MODE)                                                        \
  X(OWNER_UPDATES_SAVED)                                                       \
  X(OWNER_UPDATES_PERFORMED)                                                   \
  /* arts_id tracking counters */                                              \
  X(ARTS_ID_EDT_METRICS)                                                       \
  X(ARTS_ID_DB_METRICS)                                                        \
  X(ARTS_ID_EDT_CAPTURES)                                                      \
  X(ARTS_ID_DB_CAPTURES)                                                       \
  /* Per-node timing counters (CLUSTER level; master-measured) */              \
  X(INITIALIZATION_TIME)                                                       \
  X(END_TO_END_TIME)

// Generate enum from X-macro
typedef enum arts_counter_type_t {
#define X(name) name,
  ARTS_COUNTER_LIST
#undef X
      NUM_COUNTER_TYPES,
} arts_counter_type_t;

// Generate string array from X-macro (using stringification operator #)
static const char *const arts_counter_names[] = {
#define X(name) #name,
    ARTS_COUNTER_LIST
#undef X
};

typedef enum arts_counter_reduce_method_t {
  ARTS_COUNTER_REDUCE_SUM = 0,
  ARTS_COUNTER_REDUCE_MAX,
  ARTS_COUNTER_REDUCE_MIN,
  ARTS_COUNTER_REDUCE_MASTER, // Use master node's value only (no reduction)
} arts_counter_reduce_method_t;

// Counter mode: determines when/how often counters are captured
typedef enum arts_counter_mode_t {
  ARTS_COUNTER_MODE_OFF = 0,  // Counter disabled
  ARTS_COUNTER_MODE_ONCE = 1, // Single value at the end (no periodic capture)
  ARTS_COUNTER_MODE_PERIODIC = 2, // Periodic capture during execution
} arts_counter_mode_t;

// Counter level: determines the aggregation level for output
typedef enum arts_counter_level_t {
  ARTS_COUNTER_LEVEL_THREAD = 0,  // Per-thread output (no reduction)
  ARTS_COUNTER_LEVEL_NODE = 1,    // Per-node output (reduce across threads)
  ARTS_COUNTER_LEVEL_CLUSTER = 2, // Cluster output (reduce across all nodes)
} arts_counter_level_t;

typedef struct {
  uint64_t count;
  uint64_t start;
} arts_counter_t;

// Captured counter value with epoch information for PERIODIC mode.
// Epoch is the interval number (e.g., 1 for first interval, 100 for 100th).
// This allows proper reduction even when some intervals are skipped.
typedef struct {
  uint64_t epoch; // Interval number when captured
  uint64_t value; // Counter value at this epoch
} arts_counter_capture_t;

// Thread-local counter storage - simple array of counters only.
// Each thread updates these directly. No captures here.
// Defined in Counter.c, each thread has its own copy.
extern ARTS_THREAD_LOCAL arts_counter_t
    arts_thread_local_counters[NUM_COUNTER_TYPES];

// arts_id tracking stored separately per-thread (compile-time conditional)
#if ENABLE_ARTS_ID_EDT_METRICS || ENABLE_ARTS_ID_DB_METRICS
extern ARTS_THREAD_LOCAL arts_id_hash_table_t arts_thread_local_arts_id_metrics;
#endif
#if ENABLE_ARTS_ID_EDT_CAPTURES
extern ARTS_THREAD_LOCAL arts_array_list_t *arts_thread_local_edt_capture_list;
#endif
#if ENABLE_ARTS_ID_DB_CAPTURES
extern ARTS_THREAD_LOCAL arts_array_list_t *arts_thread_local_db_capture_list;
#endif

// Note: Saved counter data is stored directly in arts_node_info:
// - saved_counters[thread_id][counter_index]: final counter values
// - capture_arrays[thread_id][counter_index]: capture history (PERIODIC)
// - arts_id tracking data stored in __thread variables during runtime,
//   then merged at output time

// We do not implement system-wide counters due to the overhead of
// synchronization and network communication
// Also, we exclude most of the calculation part to reduce the impact on
// performance

// Counter initialization is handled internally by runtime init/cleanup.

void arts_counter_capture_start();
void arts_counter_capture_stop();
void arts_counter_increment_by(arts_counter_t *counter, uint64_t num);
void arts_counter_decrement_by(arts_counter_t *counter, uint64_t num);
void arts_counter_timer_start(arts_counter_t *counter);
void arts_counter_timer_end(arts_counter_t *counter);
void arts_counter_write(const char *output_folder, unsigned int node_id,
                        unsigned int thread_id);
void arts_counter_write_cluster(const char *output_folder,
                                unsigned int node_count);

// arts_id tracking wrapper functions (integrated with counter infrastructure)
void arts_counter_record_arts_id_edt(uint64_t arts_id, uint64_t exec_ns,
                                     uint64_t stall_ns);
void arts_counter_record_arts_id_db(uint64_t arts_id, uint64_t bytes_local,
                                    uint64_t bytes_remote,
                                    uint64_t cache_misses);
void arts_counter_capture_arts_id_edt(uint64_t arts_id, uint64_t exec_ns,
                                      uint64_t stall_ns);
void arts_counter_capture_arts_id_db(uint64_t arts_id, uint64_t bytes_accessed,
                                     uint8_t access_type);

#ifdef __cplusplus
}
#endif
#endif /* ARTS_COUNTER_COUNTER_H */
