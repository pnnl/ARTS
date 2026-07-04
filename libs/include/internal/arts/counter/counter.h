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

#include "arts/counter/Preamble.h"
#include "arts/defs.h"

// X-macro: Define all counter types in one place.
// Format: X(counterName)
// Both the enum and string array are generated from this single list.
#define ARTS_COUNTER_LIST                                                      \
  /* Time: EDT lifecycle */                                                    \
  X(TIME_EDT_EXEC)                                                             \
  X(TIME_EDT_CREATE)                                                           \
  X(TIME_EDT_SIGNAL)                                                           \
  X(TIME_CONTEXT_SWITCH)                                                       \
  /* Num: EDT lifecycle */                                                     \
  X(NUM_EDT_CREATE)                                                            \
  X(NUM_EDT_ACQUIRE)                                                           \
  X(NUM_EDT_FINISH)                                                            \
  X(NUM_EDT_SIGNAL)                                                            \
  X(NUM_YIELD)                                                                 \
  /* Time: DB lifecycle */                                                     \
  X(TIME_DB_CREATE)                                                            \
  X(TIME_DB_GET)                                                               \
  X(TIME_DB_PUT)                                                               \
  /* Num: DB lifecycle */                                                      \
  X(NUM_DB_CREATE)                                                             \
  X(NUM_DB_GET)                                                                \
  X(NUM_DB_PUT)                                                                \
  X(NUM_DB_DESTROY)                                                            \
  X(NUM_DB_ACQUIRE_READ)                                                       \
  X(NUM_DB_ACQUIRE_WRITE)                                                      \
  X(NUM_OWNER_UPDATE_SAVED)                                                    \
  X(NUM_OWNER_UPDATE_PERFORMED)                                                \
  /* Bytes: DB data */                                                         \
  X(BYTES_DB_CREATE)                                                           \
  X(BYTES_DB_PUT)                                                              \
  /* Bytes: memory */                                                          \
  X(BYTES_MEMORY_FOOTPRINT)                                                    \
  /* Bytes: network */                                                         \
  X(BYTES_REMOTE_SENT)                                                         \
  X(BYTES_REMOTE_RECEIVED)                                                     \
  /* Num: network */                                                           \
  X(NUM_REMOTE_SEND)                                                           \
  X(NUM_REMOTE_RECEIVE)                                                        \
  /* Num: network — outbound message size census, bucketed by total wire     \
   * size (header + payload) at the async send entry points, before         \
   * fragmentation/retry. NET_MSG_TOTAL is the message population count     \
   * standing in for a per-wire-type breakdown (no array-counter infra to   \
   * key by message type cheaply). */                                       \
  X(NET_MSG_LE64)                                                              \
  X(NET_MSG_LE512)                                                             \
  X(NET_MSG_LE4K)                                                              \
  X(NET_MSG_LE64K)                                                             \
  X(NET_MSG_GT64K)                                                             \
  X(NET_MSG_TOTAL)                                                             \
  /* Time: network */                                                          \
  X(TIME_REMOTE_MOVE)                                                          \
  /* Time: events */                                                           \
  X(TIME_EVENT_CREATE)                                                         \
  X(TIME_EVENT_SIGNAL)                                                         \
  /* Num: events */                                                            \
  X(NUM_EVENT_CREATE)                                                          \
  X(NUM_EVENT_SIGNAL)                                                          \
  /* Num: scheduling */                                                        \
  X(NUM_STEAL_ATTEMPT)                                                         \
  X(NUM_STEAL_SUCCESS)                                                         \
  /* Time: scheduling */                                                       \
  X(TIME_YIELD)                                                                \
  /* Num: epoch */                                                             \
  X(NUM_EPOCH_CREATE)                                                          \
  /* Num: out-of-order */                                                      \
  X(NUM_OO_ENQUEUE)                                                            \
  /* Object counters — per arts_id tracking */                                 \
  X(OBJ_NUM_EDT)                                                               \
  X(OBJ_TIME_EDT_EXEC)                                                         \
  X(OBJ_TIME_EDT_STALL)                                                        \
  X(OBJ_NUM_DB)                                                                \
  X(OBJ_BYTES_DB_LOCAL)                                                        \
  X(OBJ_BYTES_DB_REMOTE)                                                       \
  X(OBJ_NUM_DB_CACHE_MISS)                                                     \
  X(OBJ_TRACE_EDT)                                                             \
  X(OBJ_TRACE_DB)
/* Note: end-to-end / init wall time is no longer a counter — it is the
 * env-gated "[E2E] <ns>" stderr marker (rank 0; see runtime.c / shutdown.c /
 * threads.c), identical across arts/xsocr/ocr-vx. */

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

// Note: Saved counter data is stored directly in arts_node_info:
// - saved_counters[thread_id][counter_index]: final counter values
// - capture_arrays[thread_id][counter_index]: capture history (PERIODIC)
// Object counter (per-arts_id) data is managed by object_counter.h

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

// RTT-based time synchronization for precise epoch alignment.
// Worker initiates sync request, master responds, worker calculates offset.
// The time-offset state lives in counter.c (co-located with these handlers).
void arts_send_time_sync_request(void); // Worker sends request to master
void arts_handler_time_sync_request(void *pack);  // Master handles request
void arts_handler_time_sync_response(void *pack); // Worker handles response

#ifdef __cplusplus
}
#endif
#endif /* ARTS_COUNTER_COUNTER_H */
