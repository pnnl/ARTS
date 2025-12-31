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

typedef enum artsCounterType {
  edtCounter = 0,
  sleepCounter,
  totalCounter,
  signalEventCounter,
  signalPersistentEventCounter,
  signalEdtCounter,
  edtCreateCounter,
  eventCreateCounter,
  persistentEventCreateCounter,
  dbCreateCounter,
  smartDbCreateCounter,
  mallocMemory,
  callocMemory,
  freeMemory,
  guidAllocCounter,
  guidLookupCounter,
  getDbCounter,
  putDbCounter,
  contextSwitch,
  yield,
  remoteMemoryMove,
  memoryFootprint,
  edtRunningTime,
  numEdtsCreated,
  numEdtsAcquired,
  numEdtsFinished,
  remoteBytesSent,
  remoteBytesReceived,
  numDbsCreated,
  // Twin-Diff counters
  twinDiffUsed,
  twinDiffSkipped,
  twinDiffBytesSaved,
  twinDiffComputeTime,
  // Acquire-Mode counters
  acquireReadMode,
  acquireWriteMode,
  ownerUpdatesSaved,
  ownerUpdatesPerformed,
  // arts_id tracking counters
  artsIdEdtMetrics,
  artsIdDbMetrics,
  artsIdEdtCaptures,
  artsIdDbCaptures,
  // Per-node timing counters
  initializationTime,
  endToEndTime,
  finalizationTime,
  NUM_COUNTER_TYPES,
} artsCounterType;

typedef enum artsCounterReduceType {
  artsCounterSum = 0,
  artsCounterMax,
  artsCounterMin,
} artsCounterReduceType;

// Capture mode: determines when/how often counters are captured
typedef enum artsCounterCaptureMode {
  artsCaptureModeOff = 0,       // Counter disabled
  artsCaptureModeOnce = 1,      // Single value at the end (no periodic capture)
  artsCaptureModesPeriodic = 2, // Periodic capture during execution
} artsCounterCaptureMode;

// Capture level: determines the aggregation level for output
typedef enum artsCounterCaptureLevel {
  artsCaptureLevelThread = 0,  // Per-thread output (no reduction)
  artsCaptureLevelNode = 1,    // Per-node output (reduce across threads)
  artsCaptureLevelCluster = 2, // Cluster output (reduce across all nodes)
} artsCounterCaptureLevel;

typedef struct {
  uint64_t count;
  uint64_t start;
} artsCounter;

typedef struct {
  artsCounter counters[NUM_COUNTER_TYPES];
  artsArrayList *captures[NUM_COUNTER_TYPES]; // Per-thread captures
// arts_id tracking (compile-time conditional)
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  artsIdHashTable
      artsIdMetricsTable; // Hash table for per-arts_id aggregate metrics
#endif
#if ENABLE_artsIdEdtCaptures || ENABLE_artsIdDbCaptures
  artsArrayList *artsIdEdtCaptureList; // Per-invocation EDT captures
  artsArrayList *artsIdDbCaptureList;  // Per-invocation DB captures
#endif
} artsCounterCaptures;

// Node-level captures
typedef struct {
  artsArrayList *counterCaptures[NUM_COUNTER_TYPES];
// arts_id tracking (compile-time conditional)
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  artsIdHashTable artsIdMetricsTable; // Reduced hash table for NODE mode
#endif
#if ENABLE_artsIdEdtCaptures || ENABLE_artsIdDbCaptures
  artsArrayList *artsIdEdtCaptureList; // Reduced EDT captures for NODE mode
  artsArrayList *artsIdDbCaptureList;  // Reduced DB captures for NODE mode
#endif
} artsCounterReduces;

// We do not implement system-wide counters due to the overhead of
// synchronization and network communication
// Also, we exclude most of the calculation part to reduce the impact on
// performance

void artsCounterStart(unsigned int startPoint);
void artsCounterStop();
void artsCounterReset(artsCounter *counter);
void artsCounterIncrementBy(artsCounter *counter, uint64_t num);
void artsCounterDecrementBy(artsCounter *counter, uint64_t num);
void artsCounterTimerStart(artsCounter *counter);
void artsCounterTimerEnd(artsCounter *counter);
void artsCounterWriteThread(const char *outputFolder, unsigned int nodeId,
                            unsigned int threadId);
void artsCounterWriteNode(const char *outputFolder, unsigned int nodeId);
void artsCounterWriteCluster(const char *outputFolder);

// arts_id tracking wrapper functions (integrated with counter infrastructure)
void artsCounterRecordArtsIdEdt(uint64_t arts_id, uint64_t exec_ns,
                                uint64_t stall_ns);
void artsCounterRecordArtsIdDb(uint64_t arts_id, uint64_t bytes_local,
                               uint64_t bytes_remote, uint64_t cache_misses);
void artsCounterCaptureArtsIdEdt(uint64_t arts_id, uint64_t exec_ns,
                                 uint64_t stall_ns);
void artsCounterCaptureArtsIdDb(uint64_t arts_id, uint64_t bytes_accessed,
                                uint8_t access_type);

static const char *const artsCounterNames[] = {"edtCounter",
                                               "sleepCounter",
                                               "totalCounter",
                                               "signalEventCounter",
                                               "signalPersistentEventCounter",
                                               "signalEdtCounter",
                                               "edtCreateCounter",
                                               "eventCreateCounter",
                                               "persistentEventCreateCounter",
                                               "dbCreateCounter",
                                               "smartDbCreateCounter",
                                               "mallocMemory",
                                               "callocMemory",
                                               "freeMemory",
                                               "guidAllocCounter",
                                               "guidLookupCounter",
                                               "getDbCounter",
                                               "putDbCounter",
                                               "contextSwitch",
                                               "yield",
                                               "remoteMemoryMove",
                                               "memoryFootprint",
                                               "edtRunningTime",
                                               "numEdtsCreated",
                                               "numEdtsAcquired",
                                               "numEdtsFinished",
                                               "remoteBytesSent",
                                               "remoteBytesReceived",
                                               "numDbsCreated",
                                               "twinDiffUsed",
                                               "twinDiffSkipped",
                                               "twinDiffBytesSaved",
                                               "twinDiffComputeTime",
                                               "acquireReadMode",
                                               "acquireWriteMode",
                                               "ownerUpdatesSaved",
                                               "ownerUpdatesPerformed",
                                               "artsIdEdtMetrics",
                                               "artsIdDbMetrics",
                                               "artsIdEdtCaptures",
                                               "artsIdDbCaptures",
                                               "initializationTime",
                                               "endToEndTime",
                                               "finalizationTime"};

static const unsigned int artsCounterReduceTypes[] = {
    artsCounterSum, // edtCounter
    artsCounterSum, // sleepCounter
    artsCounterSum, // totalCounter
    artsCounterSum, // signalEventCounter
    artsCounterSum, // signalPersistentEventCounter
    artsCounterSum, // signalEdtCounter
    artsCounterSum, // edtCreateCounter
    artsCounterSum, // eventCreateCounter
    artsCounterSum, // persistentEventCreateCounter
    artsCounterSum, // dbCreateCounter
    artsCounterSum, // smartDbCreateCounter
    artsCounterSum, // mallocMemory
    artsCounterSum, // callocMemory
    artsCounterSum, // freeMemory
    artsCounterSum, // guidAllocCounter
    artsCounterSum, // guidLookupCounter
    artsCounterSum, // getDbCounter
    artsCounterSum, // putDbCounter
    artsCounterSum, // contextSwitch
    artsCounterSum, // yield
    artsCounterSum, // remoteMemoryMove
    artsCounterSum, // memoryFootprint
    artsCounterSum, // edtRunningTime
    artsCounterSum, // numEdtsCreated
    artsCounterSum, // numEdtsAcquired
    artsCounterSum, // numEdtsFinished
    artsCounterSum, // remoteBytesSent
    artsCounterSum, // remoteBytesReceived
    artsCounterSum, // numDbsCreated
    artsCounterSum, // twinDiffUsed
    artsCounterSum, // twinDiffSkipped
    artsCounterSum, // twinDiffBytesSaved
    artsCounterSum, // twinDiffComputeTime
    artsCounterSum, // acquireReadMode
    artsCounterSum, // acquireWriteMode
    artsCounterSum, // ownerUpdatesSaved
    artsCounterSum, // ownerUpdatesPerformed
    artsCounterSum, // artsIdEdtMetrics
    artsCounterSum, // artsIdDbMetrics
    artsCounterSum, // artsIdEdtCaptures
    artsCounterSum, // artsIdDbCaptures
    artsCounterSum, // initializationTime
    artsCounterSum, // endToEndTime
    artsCounterSum  // finalizationTime
};

#ifdef __cplusplus
}
#endif
#endif /* ARTS_COUNTER_COUNTER_H */
