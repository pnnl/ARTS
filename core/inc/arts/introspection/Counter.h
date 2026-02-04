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
#ifndef ARTS_INTROSPECTION_COUNTER_H
#define ARTS_INTROSPECTION_COUNTER_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#include "arts/introspection/ArtsIdCounter.h"
#include "arts/introspection/Preamble.h"
#include "arts/utils/ArrayList.h"

// X-macro: Define all counter types in one place.
// Format: X(counterName)
// Both the enum and string array are generated from this single list.
#define ARTS_COUNTER_LIST                                                      \
  X(edtCounter)                                                                \
  X(sleepCounter)                                                              \
  X(signalEventCounter)                                                        \
  X(signalPersistentEventCounter)                                              \
  X(signalEdtCounter)                                                          \
  X(edtCreateCounter)                                                          \
  X(eventCreateCounter)                                                        \
  X(persistentEventCreateCounter)                                              \
  X(dbCreateCounter)                                                           \
  X(smartDbCreateCounter)                                                      \
  X(mallocMemory)                                                              \
  X(callocMemory)                                                              \
  X(freeMemory)                                                                \
  X(guidAllocCounter)                                                          \
  X(guidLookupCounter)                                                         \
  X(getDbCounter)                                                              \
  X(putDbCounter)                                                              \
  X(contextSwitch)                                                             \
  X(yield)                                                                     \
  X(remoteMemoryMove)                                                          \
  X(memoryFootprint)                                                           \
  X(edtRunningTime)                                                            \
  X(numEdtsCreated)                                                            \
  X(numEdtsAcquired)                                                           \
  X(numEdtsFinished)                                                           \
  X(remoteBytesSent)                                                           \
  X(remoteBytesReceived)                                                       \
  X(numDbsCreated)                                                             \
  /* Acquire-Mode counters */                                                  \
  X(acquireReadMode)                                                           \
  X(acquireWriteMode)                                                          \
  X(ownerUpdatesSaved)                                                         \
  X(ownerUpdatesPerformed)                                                     \
  /* arts_id tracking counters */                                              \
  X(artsIdEdtMetrics)                                                          \
  X(artsIdDbMetrics)                                                           \
  X(artsIdEdtCaptures)                                                         \
  X(artsIdDbCaptures)                                                          \
  /* Per-node timing counters (CLUSTER level; master-measured) */              \
  X(initializationTime)                                                        \
  X(endToEndTime)

// Generate enum from X-macro
typedef enum artsCounterType {
#define X(name) name,
  ARTS_COUNTER_LIST
#undef X
      NUM_COUNTER_TYPES,
} artsCounterType;

// Generate string array from X-macro (using stringification operator #)
static const char *const artsCounterNames[] = {
#define X(name) #name,
    ARTS_COUNTER_LIST
#undef X
};

typedef enum artsCounterReduceMethod {
  artsCounterReduceSum = 0,
  artsCounterReduceMax,
  artsCounterReduceMin,
  artsCounterReduceMaster, // Use master node's value only (no reduction)
} artsCounterReduceMethod;

// Counter mode: determines when/how often counters are captured
typedef enum artsCounterMode {
  artsCounterModeOff = 0,      // Counter disabled
  artsCounterModeOnce = 1,     // Single value at the end (no periodic capture)
  artsCounterModePeriodic = 2, // Periodic capture during execution
} artsCounterMode;

// Counter level: determines the aggregation level for output
typedef enum artsCounterLevel {
  artsCounterLevelThread = 0,  // Per-thread output (no reduction)
  artsCounterLevelNode = 1,    // Per-node output (reduce across threads)
  artsCounterLevelCluster = 2, // Cluster output (reduce across all nodes)
} artsCounterLevel;

typedef struct {
  uint64_t count;
  uint64_t start;
} artsCounter;

// Captured counter value with epoch information for PERIODIC mode.
// Epoch is the interval number (e.g., 1 for first interval, 100 for 100th).
// This allows proper reduction even when some intervals are skipped.
typedef struct {
  uint64_t epoch; // Interval number when captured
  uint64_t value; // Counter value at this epoch
} artsCounterCapture;

// Thread-local counter storage - simple array of counters only.
// Each thread updates these directly. No captures here.
// Defined in Counter.c, each thread has its own copy.
extern __thread artsCounter artsThreadLocalCounters[NUM_COUNTER_TYPES];

// arts_id tracking stored separately per-thread (compile-time conditional)
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
extern __thread artsIdHashTable artsThreadLocalArtsIdMetrics;
#endif
#if ENABLE_artsIdEdtCaptures
extern __thread artsArrayList *artsThreadLocalEdtCaptureList;
#endif
#if ENABLE_artsIdDbCaptures
extern __thread artsArrayList *artsThreadLocalDbCaptureList;
#endif

// Note: Saved counter data is stored directly in artsNodeInfo:
// - savedCounters[threadId][counterIndex]: final counter values
// - captureArrays[threadId][counterIndex]: capture history (PERIODIC)
// - arts_id tracking data stored in __thread variables during runtime,
//   then merged at output time

// We do not implement system-wide counters due to the overhead of
// synchronization and network communication
// Also, we exclude most of the calculation part to reduce the impact on
// performance

// Counter initialization is handled internally by runtime init/cleanup.

void artsCounterCaptureStart();
void artsCounterCaptureStop();
void artsCounterIncrementBy(artsCounter *counter, uint64_t num);
void artsCounterDecrementBy(artsCounter *counter, uint64_t num);
void artsCounterTimerStart(artsCounter *counter);
void artsCounterTimerEnd(artsCounter *counter);
void artsCounterWrite(const char *outputFolder, unsigned int nodeId,
                      unsigned int threadId);
void artsCounterSendToMaster(void);
void artsCounterAllocateClusterArrays(void);
void artsCounterCollectCluster(void);

// arts_id tracking wrapper functions (integrated with counter infrastructure)
void artsCounterRecordArtsIdEdt(uint64_t arts_id, uint64_t exec_ns,
                                uint64_t stall_ns);
void artsCounterRecordArtsIdDb(uint64_t arts_id, uint64_t bytes_local,
                               uint64_t bytes_remote, uint64_t cache_misses);
void artsCounterCaptureArtsIdEdt(uint64_t arts_id, uint64_t exec_ns,
                                 uint64_t stall_ns);
void artsCounterCaptureArtsIdDb(uint64_t arts_id, uint64_t bytes_accessed,
                                uint8_t access_type);

#ifdef __cplusplus
}
#endif
#endif /* ARTS_COUNTER_COUNTER_H */
