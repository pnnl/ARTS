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
#include "arts/introspection/Counter.h"

#include <pthread.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "arts/arts.h"
#include "arts/introspection/ArtsIdCounter.h"
#include "arts/introspection/JsonWriter.h"
#include "arts/network/Remote.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/network/RemoteFunctions.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"
#include "arts/utils/Atomics.h"

// Access to network ports for setting up inbound queues during counter collection
extern unsigned int ports;

// Arrays are defined as static const in Preamble.h (included via arts.h)

// Thread-local counter storage - simple array of counters.
// Each thread updates these directly during execution.
// No captures here - capture thread handles periodic snapshots separately.
__thread artsCounter artsThreadLocalCounters[NUM_COUNTER_TYPES];

// arts_id tracking stored separately per-thread (compile-time conditional)
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
__thread artsIdHashTable artsThreadLocalArtsIdMetrics;
#endif
#if ENABLE_artsIdEdtCaptures
__thread artsArrayList *artsThreadLocalEdtCaptureList = NULL;
#endif
#if ENABLE_artsIdDbCaptures
__thread artsArrayList *artsThreadLocalDbCaptureList = NULL;
#endif

// Capture thread state - only used for periodic counter capture
static pthread_t captureThread;
static volatile bool captureThreadRunning = false;

// First capture epoch - used to normalize all epochs to start from 0
static uint64_t firstCaptureEpoch = 0;

// Time synchronization variables (exported for RemoteFunctions.c)
// timeOffset = workerTime - masterTime (positive if worker is ahead)
// To get synchronized time: localTime - timeOffset = masterTime
volatile int64_t artsCounterTimeOffset = 0;
volatile bool artsCounterTimeSyncReceived = false;

// Cluster reduction state (exported for RemoteFunctions.c)
// Master allocates these arrays before workers send their data
volatile unsigned int artsCounterReduceReceived = 0;
uint64_t **artsClusterCounterValues = NULL; // [counterIndex][nodeId]
uint64_t ***artsClusterCaptureEpochs =
    NULL; // [counterIndex][nodeId][captureIdx]
uint64_t ***artsClusterCaptureValues =
    NULL; // [counterIndex][nodeId][captureIdx]
uint64_t **artsClusterCaptureCounts = NULL; // [counterIndex][nodeId]

// Get synchronized timestamp (adjusted to master node's clock)
// Call this function only after time synchronization is complete
// artsCounterTimeSyncReceived == true
static inline uint64_t artsGetSyncedTimeStamp(void) {
  return (uint64_t)((int64_t)artsGetTimeStamp() - artsCounterTimeOffset);
}

static uint64_t artsCounterCaptureCounter(artsCounter *counter) {
  uint64_t expected = counter->start;
  while (expected) {
    uint64_t start = artsGetTimeStamp();
    if (artsAtomicCswapU64(&counter->start, expected, start) != expected) {
      expected = counter->start;
    } else {
      artsAtomicFetchAddU64(&counter->count, start - expected);
      expected = 0;
    }
  }
  return counter->count;
}

static void *artsCounterCaptureThread(void *args) {
  (void)args; // Unused - capture thread doesn't need its own counters

  // Validate counterCaptureInterval to prevent division by zero
  if (artsNodeInfo.counterCaptureInterval == 0) {
    ARTS_INFO("counterCaptureInterval is 0, capture thread exiting");
    return NULL;
  }

  // Validate interval to prevent overflow (max ~18 billion ms before overflow)
  if (artsNodeInfo.counterCaptureInterval > UINT64_MAX / 1000000) {
    ARTS_INFO("counterCaptureInterval too large, capture thread exiting");
    return NULL;
  }

  // milli to nano
  uint64_t intervalNs = artsNodeInfo.counterCaptureInterval * 1000000;

  // Time synchronization: align captures to synchronized time.
  // All nodes use master node's time (via artsGetSyncedTimeStamp) so they
  // all capture at the same logical time, preventing drift between nodes.
  // We compute the next capture time as the next multiple of intervalNs
  // after the current synchronized time.
  uint64_t syncedTime = artsGetSyncedTimeStamp();
  // Calculate the capture epoch (floor of syncedTime / intervalNs)
  uint64_t captureEpoch = syncedTime / intervalNs;
  // Next capture will be at the next interval boundary
  // Check for overflow before multiplication
  if (captureEpoch > UINT64_MAX / intervalNs - 1) {
    ARTS_INFO("Capture epoch overflow, capture thread exiting");
    return NULL;
  }
  uint64_t nextCaptureTime = (captureEpoch + 1) * intervalNs;

  // Record the first capture epoch for normalization (epochs will start from 0)
  firstCaptureEpoch = captureEpoch + 1;

  while (captureThreadRunning) {
    syncedTime = artsGetSyncedTimeStamp();
    // Calculate sleep time until next aligned capture (in synced time)
    int64_t sleepNs = (int64_t)(nextCaptureTime - syncedTime);

    // Track current epoch for this capture
    uint64_t currentEpoch = captureEpoch + 1;

    if (sleepNs > 0) {
      nanosleep((const struct timespec[]){{sleepNs / 1000000000,
                                           sleepNs % 1000000000}},
                NULL);
      // Advance to the next expected capture time
      captureEpoch++;
      nextCaptureTime += intervalNs;
    } else {
      // We are late: nextCaptureTime is already in the past.
      // Compute how many intervals we are behind and jump forward
      // to the next future aligned capture time to avoid drift.
      uint64_t intervalsBehind =
          (uint64_t)(((-sleepNs) / (int64_t)intervalNs) + 1);
      ARTS_INFO(
          "Counter capture lagging: syncedTime=%lf ms, "
          "nextCaptureTime=%lf ms, lateBy=%lf ms, skipping %lu interval(s)",
          (double)syncedTime / 1000000.0, (double)nextCaptureTime / 1000000.0,
          (double)(-sleepNs) / 1000000.0, intervalsBehind);
      captureEpoch += intervalsBehind;
      currentEpoch = captureEpoch;
      nextCaptureTime += intervalsBehind * intervalNs;
    }

    // Capture counters from all threads - store in nodeInfo.captureArrays
    // Skip threads with NULL liveCounters (not registered or already closed)
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsCounterModeArray[i] == artsCounterModePeriodic) {
        for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
          // Read counter value from thread's __thread storage via liveCounters
          // pointer NULL means thread hasn't registered yet or has already
          // closed
          artsCounter *threadCounters = artsNodeInfo.liveCounters[t];
          if (!threadCounters) {
            continue;
          }
          artsCounterCapture capture;
          capture.epoch = currentEpoch;
          capture.value = artsCounterCaptureCounter(&threadCounters[i]);
          if (artsNodeInfo.captureArrays[t][i]) {
            artsPushToArrayList(artsNodeInfo.captureArrays[t][i], &capture);
          }
        }
      }
    }
  }
  return NULL;
}

void artsCounterCaptureStart() {
  if (captureThreadRunning) {
    ARTS_DEBUG("Trying to start capture thread which is already running");
    artsDebugGenerateSegFault();
  }

  bool needCaptureThread = false;

  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    // Need capture thread for PERIODIC mode counters
    if (artsCounterModeArray[i] == artsCounterModePeriodic) {
      needCaptureThread = true;
      break;
    }
  }

  if (needCaptureThread) {
    // RTT-based time synchronization: workers send request to master,
    // master responds with its timestamp, workers calculate offset using RTT.
    // This provides better accuracy than one-way broadcast.

    if (artsGlobalRankId == 0) {
      // Master node: no offset needed, just set ready
      artsCounterTimeOffset = 0;
      artsCounterTimeSyncReceived = true;
      ARTS_INFO("Time sync: Master node (rank 0), offset=0");
    } else {
      // Worker nodes: send sync request and wait for response
      artsRemoteTimeSyncRequest();
      uint64_t timeout = artsGetTimeStamp() + 5000000000ULL; // 5 seconds
      while (!artsCounterTimeSyncReceived && artsGetTimeStamp() < timeout) {
        usleep(1000); // Wait 1ms
      }
      if (!artsCounterTimeSyncReceived) {
        ARTS_INFO("Time sync: Timeout waiting for master response, "
                  "using local time (offset=0)");
        artsCounterTimeOffset = 0;
        artsCounterTimeSyncReceived = true;
      }
    }

    captureThreadRunning = true;

    int ret =
        pthread_create(&captureThread, NULL, artsCounterCaptureThread, NULL);
    if (ret) {
      ARTS_DEBUG("Failed to create capture thread: %d", ret);
      captureThreadRunning = false;
    } else {
      ARTS_INFO("Counter capture thread started");
    }
  }
}

void artsCounterCaptureStop() {
  if (!captureThreadRunning) {
    // No capture thread to stop - this is fine, counters still work
    return;
  }
  captureThreadRunning = false;
  int ret = pthread_join(captureThread, NULL);
  if (ret) {
    ARTS_DEBUG("Failed to join capture thread: %d", ret);
  }
}

void artsCounterIncrementBy(artsCounter *counter, uint64_t num) {
  artsAtomicFetchAddU64(&counter->count, num);
}

void artsCounterDecrementBy(artsCounter *counter, uint64_t num) {
  artsAtomicFetchSubU64(&counter->count, num);
}

void artsCounterTimerStart(artsCounter *counter) {
  if (artsAtomicCswapU64(&counter->start, 0, artsGetTimeStamp())) {
    ARTS_DEBUG("Trying to start a timer that is already started");
    artsDebugGenerateSegFault();
  }
}

void artsCounterTimerEnd(artsCounter *counter) {
  uint64_t end = artsGetTimeStamp();
  uint64_t start = artsAtomicSwapU64(&counter->start, 0);
  if (!start) {
    ARTS_DEBUG("Trying to end a timer that is not started");
    artsDebugGenerateSegFault();
  }
  artsAtomicFetchAddU64(&counter->count, end - start);
}

// Helper: apply one reduction step
static inline uint64_t artsApplyReduction(uint64_t accumulator, uint64_t value,
                                          artsCounterReduceMethod reduceMethod,
                                          unsigned int sourceIndex) {
  switch (reduceMethod) {
  case artsCounterReduceSum:
    return accumulator + value;
  case artsCounterReduceMax:
    return (accumulator < value) ? value : accumulator;
  case artsCounterReduceMin:
    return (accumulator > value) ? value : accumulator;
  case artsCounterReduceMaster:
    return (sourceIndex == 0) ? value : accumulator;
  }
  return accumulator;
}

// Helper: convert reduce method to string for JSON output
static inline const char *
artsReduceMethodToString(artsCounterReduceMethod reduceMethod) {
  switch (reduceMethod) {
  case artsCounterReduceSum:
    return "SUM";
  case artsCounterReduceMax:
    return "MAX";
  case artsCounterReduceMin:
    return "MIN";
  case artsCounterReduceMaster:
    return "MASTER";
  }
  return "SUM";
}

// Helper: convert capture mode to string for JSON output
static inline const char *artsCounterModeToString(unsigned int mode) {
  switch (mode) {
  case artsCounterModeOnce:
    return "ONCE";
  case artsCounterModePeriodic:
    return "PERIODIC";
  default:
    return "OFF";
  }
}

// Helper: check if counter name contains "Time" (case-insensitive)
static bool artsCounterIsTimeCounter(const char *name) {
  if (!name)
    return false;
  const char *p = name;
  while (*p) {
    if ((*p == 'T' || *p == 't') && (*(p + 1) == 'I' || *(p + 1) == 'i') &&
        (*(p + 2) == 'M' || *(p + 2) == 'm') &&
        (*(p + 3) == 'E' || *(p + 3) == 'e')) {
      return true;
    }
    p++;
  }
  return false;
}

// Helper: write capture history array to JSON as compact single line
// Format: [[epoch, value], [epoch, value], ...]
// Epochs are normalized to start from 0 by subtracting firstCaptureEpoch
static void artsWriteCaptureHistory(artsJsonWriter *writer, uint64_t *epochs,
                                    uint64_t *values, uint64_t count) {
  if (count == 0)
    return;

  // Build compact JSON string: [[e,v],[e,v],...]
  // Estimate size: each entry is at most ~40 chars, plus brackets
  size_t bufSize = count * 45 + 10;
  char *buf = (char *)artsMalloc(bufSize);
  char *p = buf;
  *p++ = '[';

  for (uint64_t c = 0; c < count; c++) {
    if (c > 0)
      *p++ = ',';
    uint64_t normalizedEpoch = epochs[c] - firstCaptureEpoch;
    p += sprintf(p, "[%llu,%llu]", (unsigned long long)normalizedEpoch,
                 (unsigned long long)values[c]);
  }
  *p++ = ']';
  *p = '\0';

  artsJsonWriterWriteRawArray(writer, "captureHistory", buf);
  artsFree(buf);
}

// Helper: write common metadata fields to JSON
static void artsWriteCommonMetadata(artsJsonWriter *writer) {
  artsJsonWriterWriteUInt64(writer, "timestamp", (uint64_t)time(NULL));
  artsJsonWriterWriteString(writer, "version", "1.7.0");
  if (artsNodeInfo.counterFolder) {
    artsJsonWriterWriteString(writer, "counterFolder",
                              artsNodeInfo.counterFolder);
  }
}

// Helper: check if any counters exist at given level, return count
static unsigned int artsCountersAtLevel(unsigned int level) {
  unsigned int count = 0;
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] != artsCounterModeOff &&
        artsCounterLevelArray[i] == level) {
      count++;
    }
  }
  return count;
}

// Helper: open output file, creating directory if needed
static FILE *artsOpenCounterFile(const char *outputFolder,
                                 const char *filename) {
  struct stat st = {0};
  if (stat(outputFolder, &st) == -1) {
    mkdir(outputFolder, 0755);
  }
  char filepath[1024];
  snprintf(filepath, sizeof(filepath), "%s/%s", outputFolder, filename);
  return fopen(filepath, "w");
}

// Helper: finalize and close JSON file
static void artsCloseCounterFile(artsJsonWriter *writer, FILE *fp) {
  artsJsonWriterEndObject(writer);
  artsJsonWriterFinish(writer);
  fputc('\n', fp);
  fclose(fp);
}

// Helper function to compute node-level reduced value across all threads
// Safe to call after threads have closed - uses savedCounters data
static uint64_t artsComputeNodeReducedValue(unsigned int index) {
  artsCounterReduceMethod reduceMethod = artsCounterReduceMethodArray[index];
  uint64_t nodeValue = (reduceMethod == artsCounterReduceMin) ? UINT64_MAX : 0;
  for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
    artsCounter *saved = artsNodeInfo.savedCounters[t];
    if (!saved) {
      continue; // Thread never registered or data not available
    }
    nodeValue =
        artsApplyReduction(nodeValue, saved[index].count, reduceMethod, t);
  }
  return nodeValue;
}

// Helper function to compute node-level reduced captures for PERIODIC mode
// Returns arrays of epochs and values, sets count. Caller must free both.
static void artsComputeNodeReducedCaptures(unsigned int counterIndex,
                                           uint64_t **outEpochs,
                                           uint64_t **outValues,
                                           uint64_t *outCount) {
  *outEpochs = NULL;
  *outValues = NULL;
  *outCount = 0;

  // Find the max number of captures across all threads
  uint64_t maxCaptures = 0;
  for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
    artsArrayList *threadList = artsNodeInfo.captureArrays[t][counterIndex];
    if (threadList && threadList->index > maxCaptures) {
      maxCaptures = threadList->index;
    }
  }

  if (maxCaptures == 0) {
    return;
  }

  // Allocate output arrays (upper bound size)
  *outEpochs = (uint64_t *)artsMalloc(maxCaptures * sizeof(uint64_t));
  *outValues = (uint64_t *)artsMalloc(maxCaptures * sizeof(uint64_t));

  // Create iterators for all threads
  artsArrayListIterator **iters = (artsArrayListIterator **)artsCalloc(
      artsNodeInfo.totalThreadCount, sizeof(artsArrayListIterator *));
  artsCounterCapture **currentCaptures = (artsCounterCapture **)artsCalloc(
      artsNodeInfo.totalThreadCount, sizeof(artsCounterCapture *));

  for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
    artsArrayList *threadList = artsNodeInfo.captureArrays[t][counterIndex];
    if (threadList && threadList->index > 0) {
      iters[t] = artsNewArrayListIterator(threadList);
      if (artsArrayListHasNext(iters[t])) {
        currentCaptures[t] = (artsCounterCapture *)artsArrayListNext(iters[t]);
      }
    }
  }

  // Reduce by epoch - process captures in epoch order
  uint64_t capturesWritten = 0;
  while (capturesWritten < maxCaptures) {
    // Find minimum epoch among all current captures
    uint64_t minEpoch = UINT64_MAX;
    for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
      if (currentCaptures[t] && currentCaptures[t]->epoch < minEpoch) {
        minEpoch = currentCaptures[t]->epoch;
      }
    }
    if (minEpoch == UINT64_MAX)
      break;

    // Reduce all captures at this epoch
    artsCounterReduceMethod reduceMethod =
        artsCounterReduceMethodArray[counterIndex];
    uint64_t reducedValue =
        (reduceMethod == artsCounterReduceMin) ? UINT64_MAX : 0;
    bool hasValue = false;
    for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
      if (currentCaptures[t] && currentCaptures[t]->epoch == minEpoch) {
        hasValue = true;
        reducedValue = artsApplyReduction(
            reducedValue, currentCaptures[t]->value, reduceMethod, t);
        // Advance this thread's iterator
        if (artsArrayListHasNext(iters[t])) {
          currentCaptures[t] =
              (artsCounterCapture *)artsArrayListNext(iters[t]);
        } else {
          currentCaptures[t] = NULL;
        }
      }
    }

    if (hasValue) {
      (*outEpochs)[capturesWritten] = minEpoch;
      (*outValues)[capturesWritten] = reducedValue;
      capturesWritten++;
    }
  }

  *outCount = capturesWritten;

  // Cleanup iterators
  for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
    if (iters[t]) {
      artsDeleteArrayListIterator(iters[t]);
    }
  }
  artsFree(iters);
  artsFree(currentCaptures);
}

// Helper: write arts_id metrics to JSON
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
static void artsWriteArtsIdMetrics(artsJsonWriter *writer,
                                   artsIdHashTable *table, bool writeEdt,
                                   bool writeDb) {
  artsJsonWriterBeginObject(writer, "artsIdMetrics");

  if (writeEdt) {
    artsJsonWriterBeginArray(writer, "edts");
    for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
      if (table->edt_metrics[i].valid) {
        artsJsonWriterBeginObject(writer, NULL);
        artsJsonWriterWriteUInt64(writer, "arts_id",
                                  table->edt_metrics[i].arts_id);
        artsJsonWriterWriteUInt64(writer, "invocations",
                                  table->edt_metrics[i].invocations);
        artsJsonWriterWriteUInt64(writer, "total_exec_ns",
                                  table->edt_metrics[i].total_exec_ns);
        artsJsonWriterWriteUInt64(writer, "total_stall_ns",
                                  table->edt_metrics[i].total_stall_ns);
        artsJsonWriterEndObject(writer);
      }
    }
    artsJsonWriterEndArray(writer);
    artsJsonWriterWriteUInt64(writer, "edt_collisions", table->edt_collisions);
  }

  if (writeDb) {
    artsJsonWriterBeginArray(writer, "dbs");
    for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
      if (table->db_metrics[i].valid) {
        artsJsonWriterBeginObject(writer, NULL);
        artsJsonWriterWriteUInt64(writer, "arts_id",
                                  table->db_metrics[i].arts_id);
        artsJsonWriterWriteUInt64(writer, "invocations",
                                  table->db_metrics[i].invocations);
        artsJsonWriterWriteUInt64(writer, "bytes_local",
                                  table->db_metrics[i].bytes_local);
        artsJsonWriterWriteUInt64(writer, "bytes_remote",
                                  table->db_metrics[i].bytes_remote);
        artsJsonWriterWriteUInt64(writer, "cache_misses",
                                  table->db_metrics[i].cache_misses);
        artsJsonWriterEndObject(writer);
      }
    }
    artsJsonWriterEndArray(writer);
    artsJsonWriterWriteUInt64(writer, "db_collisions", table->db_collisions);
  }

  artsJsonWriterEndObject(writer);
}
#endif

static void artsCounterWriteThread(const char *outputFolder,
                                   unsigned int nodeId, unsigned int threadId) {
  if (!artsCountersAtLevel(artsCounterLevelThread))
    return;

  char filename[64];
  snprintf(filename, sizeof(filename), "n%u_t%u.json", nodeId, threadId);
  FILE *fp = artsOpenCounterFile(outputFolder, filename);
  if (!fp)
    return;

  artsJsonWriter writer;
  artsJsonWriterInit(&writer, fp, 2);
  artsJsonWriterBeginObject(&writer, NULL);

  // Metadata
  artsJsonWriterBeginObject(&writer, "metadata");
  artsJsonWriterWriteUInt64(&writer, "nodeId", nodeId);
  artsJsonWriterWriteUInt64(&writer, "threadId", threadId);
  artsWriteCommonMetadata(&writer);
  artsJsonWriterWriteUInt64(&writer, "counterCaptureInterval",
                            artsNodeInfo.counterCaptureInterval);
  artsJsonWriterEndObject(&writer);

  // Counters
  artsJsonWriterBeginObject(&writer, "counters");
  artsCounter *saved = artsNodeInfo.savedCounters[threadId];
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] == artsCounterModeOff ||
        artsCounterLevelArray[i] != artsCounterLevelThread)
      continue;

    artsJsonWriterBeginObject(&writer, artsCounterNames[i]);
    artsJsonWriterWriteString(&writer, "captureMode",
                              artsCounterModeToString(artsCounterModeArray[i]));
    artsJsonWriterWriteString(&writer, "captureLevel", "THREAD");

    // Always write final value
    uint64_t finalValue = saved[i].count;
    artsJsonWriterWriteUInt64(&writer, "value", finalValue);
    if (artsCounterIsTimeCounter(artsCounterNames[i])) {
      artsJsonWriterWriteDouble(&writer, "value_ms",
                                (double)finalValue / 1000000.0);
    }

    if (artsCounterModeArray[i] == artsCounterModePeriodic) {
      artsArrayList *captureList = artsNodeInfo.captureArrays[threadId][i];
      if (captureList && captureList->index > 0) {
        uint64_t count = captureList->index;
        uint64_t *epochs = (uint64_t *)artsMalloc(count * sizeof(uint64_t));
        uint64_t *values = (uint64_t *)artsMalloc(count * sizeof(uint64_t));
        artsArrayListIterator *iter = artsNewArrayListIterator(captureList);
        for (uint64_t idx = 0; artsArrayListHasNext(iter) && idx < count;
             idx++) {
          artsCounterCapture *cap =
              (artsCounterCapture *)artsArrayListNext(iter);
          if (cap) {
            epochs[idx] = cap->epoch;
            values[idx] = cap->value;
          }
        }
        artsDeleteArrayListIterator(iter);
        artsWriteCaptureHistory(&writer, epochs, values, count);
        artsFree(epochs);
        artsFree(values);
      }
    }
    artsJsonWriterEndObject(&writer);
  }
  artsJsonWriterEndObject(&writer);

#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  bool edtThread =
      artsCounterModeArray[artsIdEdtMetrics] != artsCounterModeOff &&
      artsCounterLevelArray[artsIdEdtMetrics] == artsCounterLevelThread;
  bool dbThread =
      artsCounterModeArray[artsIdDbMetrics] != artsCounterModeOff &&
      artsCounterLevelArray[artsIdDbMetrics] == artsCounterLevelThread;
  if (edtThread || dbThread) {
    artsWriteArtsIdMetrics(
        &writer, &artsNodeInfo.savedCounters[threadId]->artsIdMetricsTable,
        edtThread, dbThread);
  }
#endif

  artsCloseCounterFile(&writer, fp);
}

static void artsCounterWriteNode(const char *outputFolder,
                                 unsigned int nodeId) {
  if (!artsCountersAtLevel(artsCounterLevelNode))
    return;

  char filename[64];
  snprintf(filename, sizeof(filename), "n%u.json", nodeId);
  FILE *fp = artsOpenCounterFile(outputFolder, filename);
  if (!fp)
    return;

  artsJsonWriter writer;
  artsJsonWriterInit(&writer, fp, 2);
  artsJsonWriterBeginObject(&writer, NULL);

  // Metadata
  artsJsonWriterBeginObject(&writer, "metadata");
  artsJsonWriterWriteUInt64(&writer, "nodeId", nodeId);
  artsWriteCommonMetadata(&writer);
  artsJsonWriterWriteUInt64(&writer, "captureInterval",
                            artsNodeInfo.counterCaptureInterval);
  artsJsonWriterWriteUInt64(&writer, "totalThreads",
                            artsNodeInfo.totalThreadCount);
  artsJsonWriterEndObject(&writer);

  // Counters
  artsJsonWriterBeginObject(&writer, "counters");
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] == artsCounterModeOff ||
        artsCounterLevelArray[i] != artsCounterLevelNode)
      continue;

    artsJsonWriterBeginObject(&writer, artsCounterNames[i]);
    artsJsonWriterWriteString(&writer, "captureMode",
                              artsCounterModeToString(artsCounterModeArray[i]));
    artsJsonWriterWriteString(&writer, "captureLevel", "NODE");
    artsJsonWriterWriteString(
        &writer, "reduceMethod",
        artsReduceMethodToString(artsCounterReduceMethodArray[i]));

    uint64_t finalValue = artsComputeNodeReducedValue(i);
    artsJsonWriterWriteUInt64(&writer, "value", finalValue);
    if (artsCounterIsTimeCounter(artsCounterNames[i])) {
      artsJsonWriterWriteDouble(&writer, "value_ms",
                                (double)finalValue / 1000000.0);
    }

    if (artsCounterModeArray[i] == artsCounterModePeriodic) {
      uint64_t *epochs, *values, count;
      artsComputeNodeReducedCaptures(i, &epochs, &values, &count);
      artsWriteCaptureHistory(&writer, epochs, values, count);
      if (epochs)
        artsFree(epochs);
      if (values)
        artsFree(values);
    }
    artsJsonWriterEndObject(&writer);
  }
  artsJsonWriterEndObject(&writer);

#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  bool edtNode =
      artsCounterModeArray[artsIdEdtMetrics] != artsCounterModeOff &&
      artsCounterLevelArray[artsIdEdtMetrics] == artsCounterLevelNode;
  bool dbNode = artsCounterModeArray[artsIdDbMetrics] != artsCounterModeOff &&
                artsCounterLevelArray[artsIdDbMetrics] == artsCounterLevelNode;
  if (edtNode || dbNode) {
    // Use thread 0's metrics as representative for node level
    // TODO: implement proper node-level reduction of arts_id metrics
    if (artsNodeInfo.savedCounters[0]) {
      artsWriteArtsIdMetrics(&writer,
                             &artsNodeInfo.savedCounters[0]->artsIdMetricsTable,
                             edtNode, dbNode);
    }
  }
#endif

  artsCloseCounterFile(&writer, fp);
}

// Flag to track if cluster counter collection is already done
static bool artsClusterCountersCollected = false;

// Worker nodes: send counter data to master (called on COUNTER_COLLECT_MSG)
void artsCounterSendToMaster(void) {
  unsigned int clusterCounterCount =
      artsCountersAtLevel(artsCounterLevelCluster);
  if (!clusterCounterCount || artsGlobalRankId == 0)
    return; // Only workers send

  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] == artsCounterModeOff ||
        artsCounterLevelArray[i] != artsCounterLevelCluster)
      continue;
    uint64_t nodeValue = artsComputeNodeReducedValue(i);
    if (artsCounterModeArray[i] == artsCounterModePeriodic) {
      uint64_t *epochs, *values, count;
      artsComputeNodeReducedCaptures(i, &epochs, &values, &count);
      artsRemoteCounterReduceSend(i, nodeValue, epochs, values, count);
      if (epochs)
        artsFree(epochs);
      if (values)
        artsFree(values);
    } else {
      artsRemoteCounterReduceSend(i, nodeValue, NULL, NULL, 0);
    }
  }
}

// Lock for thread-safe cluster array allocation
static volatile unsigned int clusterArrayAllocLock = 0;

// Pre-allocate cluster arrays before receiving worker data
// Thread-safe: can be called from multiple receiver threads
void artsCounterAllocateClusterArrays(void) {
  if (artsGlobalRankId != 0)
    return;

  unsigned int clusterCounterCount =
      artsCountersAtLevel(artsCounterLevelCluster);
  if (!clusterCounterCount)
    return;

  // Fast path: already allocated?
  if (artsClusterCounterValues)
    return;

  // Acquire lock for thread-safe allocation
  while (__sync_lock_test_and_set(&clusterArrayAllocLock, 1)) {
    // Spin wait
  }

  // Double-check after acquiring lock
  if (artsClusterCounterValues) {
    __sync_lock_release(&clusterArrayAllocLock);
    return;
  }

  unsigned int numNodes = artsGlobalRankCount;

  // Allocate storage arrays for cluster counter values
  artsClusterCounterValues =
      (uint64_t **)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t *));
  artsClusterCaptureEpochs =
      (uint64_t ***)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t **));
  artsClusterCaptureValues =
      (uint64_t ***)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t **));
  artsClusterCaptureCounts =
      (uint64_t **)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t *));

  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] == artsCounterModeOff ||
        artsCounterLevelArray[i] != artsCounterLevelCluster)
      continue;
    artsClusterCounterValues[i] =
        (uint64_t *)artsCalloc(numNodes, sizeof(uint64_t));
    artsClusterCaptureEpochs[i] =
        (uint64_t **)artsCalloc(numNodes, sizeof(uint64_t *));
    artsClusterCaptureValues[i] =
        (uint64_t **)artsCalloc(numNodes, sizeof(uint64_t *));
    artsClusterCaptureCounts[i] =
        (uint64_t *)artsCalloc(numNodes, sizeof(uint64_t));
  }

  // Release lock
  __sync_lock_release(&clusterArrayAllocLock);
}

// Master: prepare cluster counter collection before network shutdown
// Waits for worker data (arrays must be pre-allocated)
void artsCounterCollectCluster(void) {
  if (artsGlobalRankId != 0 || artsClusterCountersCollected)
    return;

  unsigned int clusterCounterCount =
      artsCountersAtLevel(artsCounterLevelCluster);
  if (!clusterCounterCount)
    return;

  unsigned int numNodes = artsGlobalRankCount;

  // Ensure arrays are allocated (in case this is called before pre-allocation)
  artsCounterAllocateClusterArrays();

  // Store master's own data
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] == artsCounterModeOff ||
        artsCounterLevelArray[i] != artsCounterLevelCluster)
      continue;
    artsClusterCounterValues[i][0] = artsComputeNodeReducedValue(i);
    if (artsCounterModeArray[i] == artsCounterModePeriodic) {
      artsComputeNodeReducedCaptures(i, &artsClusterCaptureEpochs[i][0],
                                     &artsClusterCaptureValues[i][0],
                                     &artsClusterCaptureCounts[i][0]);
    }
  }

  // Wait for worker data - actively receive since receiver threads may have stopped
  unsigned int expectedMessages = (numNodes - 1) * clusterCounterCount;
  if (expectedMessages > 0) {
    // Set up this thread to poll ALL inbound queues
    // The main thread doesn't normally have inbound queues assigned since it's a worker
    unsigned int totalInboundQueues = (artsGlobalRankCount - 1) * ports;
    artsRemoteSetThreadInboundQueues(0, totalInboundQueues);

    uint64_t timeout = artsGetTimeStamp() + 30000000000ULL; // 30 second timeout
    while (artsCounterReduceReceived < expectedMessages &&
           artsGetTimeStamp() < timeout) {
      // Actively receive instead of just sleeping - receiver threads may have
      // stopped after artsRuntimeStop() was called
      artsServerTryToReceive(&artsNodeInfo.buf, &artsNodeInfo.packetSize,
                             &artsNodeInfo.stealRequestLock);
    }
  }

  artsClusterCountersCollected = true;
}

static void artsCounterWriteCluster(const char *outputFolder) {
  unsigned int clusterCounterCount =
      artsCountersAtLevel(artsCounterLevelCluster);
  if (!clusterCounterCount)
    return;

  unsigned int numNodes = artsGlobalRankCount;

  if (artsGlobalRankId == 0) {
    // If not already collected (multi-node case does collection in shutdown)
    if (!artsClusterCountersCollected) {
      // Master: allocate, wait for workers, reduce and write
      artsClusterCounterValues =
          (uint64_t **)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t *));
      artsClusterCaptureEpochs =
          (uint64_t ***)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t **));
      artsClusterCaptureValues =
          (uint64_t ***)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t **));
      artsClusterCaptureCounts =
          (uint64_t **)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t *));

      for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
        if (artsCounterModeArray[i] == artsCounterModeOff ||
            artsCounterLevelArray[i] != artsCounterLevelCluster)
          continue;
        artsClusterCounterValues[i] =
            (uint64_t *)artsCalloc(numNodes, sizeof(uint64_t));
        artsClusterCaptureEpochs[i] =
            (uint64_t **)artsCalloc(numNodes, sizeof(uint64_t *));
        artsClusterCaptureValues[i] =
            (uint64_t **)artsCalloc(numNodes, sizeof(uint64_t *));
        artsClusterCaptureCounts[i] =
            (uint64_t *)artsCalloc(numNodes, sizeof(uint64_t));
        if (artsCounterReduceMethodArray[i] == artsCounterReduceMin) {
          for (unsigned int n = 0; n < numNodes; n++)
            artsClusterCounterValues[i][n] = UINT64_MAX;
        }
      }

      // Store master's own data
      for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
        if (artsCounterModeArray[i] == artsCounterModeOff ||
            artsCounterLevelArray[i] != artsCounterLevelCluster)
          continue;
        artsClusterCounterValues[i][0] = artsComputeNodeReducedValue(i);
        if (artsCounterModeArray[i] == artsCounterModePeriodic) {
          artsComputeNodeReducedCaptures(i, &artsClusterCaptureEpochs[i][0],
                                         &artsClusterCaptureValues[i][0],
                                         &artsClusterCaptureCounts[i][0]);
        }
      }

      // Wait for workers (only in single-node or if collection not done yet)
      unsigned int expectedMessages = (numNodes - 1) * clusterCounterCount;
      if (expectedMessages > 0) {
        // Set up this thread to poll ALL inbound queues
        unsigned int totalInboundQueues = (artsGlobalRankCount - 1) * ports;
        artsRemoteSetThreadInboundQueues(0, totalInboundQueues);

        uint64_t timeout = artsGetTimeStamp() + 30000000000ULL;
        while (artsCounterReduceReceived < expectedMessages &&
               artsGetTimeStamp() < timeout) {
          // Actively receive - receiver threads may have stopped
          artsServerTryToReceive(&artsNodeInfo.buf, &artsNodeInfo.packetSize,
                                 &artsNodeInfo.stealRequestLock);
        }
      }
    }

    // Write cluster file
    FILE *fp = artsOpenCounterFile(outputFolder, "cluster.json");
    if (!fp)
      goto cleanup;

    artsJsonWriter writer;
    artsJsonWriterInit(&writer, fp, 2);
    artsJsonWriterBeginObject(&writer, NULL);

    artsJsonWriterBeginObject(&writer, "metadata");
    artsJsonWriterWriteUInt64(&writer, "numNodes", numNodes);
    artsWriteCommonMetadata(&writer);
    artsJsonWriterWriteString(&writer, "level", "CLUSTER");
    artsJsonWriterEndObject(&writer);

    artsJsonWriterBeginObject(&writer, "counters");
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsCounterModeArray[i] == artsCounterModeOff ||
          artsCounterLevelArray[i] != artsCounterLevelCluster)
        continue;

      artsCounterReduceMethod reduceMethod = artsCounterReduceMethodArray[i];
      artsJsonWriterBeginObject(&writer, artsCounterNames[i]);
      artsJsonWriterWriteString(
          &writer, "captureMode",
          artsCounterModeToString(artsCounterModeArray[i]));
      artsJsonWriterWriteString(&writer, "captureLevel", "CLUSTER");
      artsJsonWriterWriteString(&writer, "reduceMethod",
                                artsReduceMethodToString(reduceMethod));

      if (artsCounterModeArray[i] == artsCounterModeOnce) {
        uint64_t clusterValue =
            (reduceMethod == artsCounterReduceMin) ? UINT64_MAX : 0;
        for (unsigned int n = 0; n < numNodes; n++)
          clusterValue = artsApplyReduction(
              clusterValue, artsClusterCounterValues[i][n], reduceMethod, n);
        artsJsonWriterWriteUInt64(&writer, "value", clusterValue);
        if (artsCounterIsTimeCounter(artsCounterNames[i])) {
          artsJsonWriterWriteDouble(&writer, "value_ms",
                                    (double)clusterValue / 1000000.0);
        }
      } else {
        // PERIODIC: first write final value, then capture history
        uint64_t clusterValue =
            (reduceMethod == artsCounterReduceMin) ? UINT64_MAX : 0;
        for (unsigned int n = 0; n < numNodes; n++)
          clusterValue = artsApplyReduction(
              clusterValue, artsClusterCounterValues[i][n], reduceMethod, n);
        artsJsonWriterWriteUInt64(&writer, "value", clusterValue);
        if (artsCounterIsTimeCounter(artsCounterNames[i])) {
          artsJsonWriterWriteDouble(&writer, "value_ms",
                                    (double)clusterValue / 1000000.0);
        }

        // Then write capture history reduced by epoch as compact single line
        uint64_t *nodePositions =
            (uint64_t *)artsCalloc(numNodes, sizeof(uint64_t));

        // First pass: count how many entries we'll have
        uint64_t entryCount = 0;
        uint64_t *tempPositions =
            (uint64_t *)artsCalloc(numNodes, sizeof(uint64_t));
        while (1) {
          uint64_t minEpoch = UINT64_MAX;
          for (unsigned int n = 0; n < numNodes; n++) {
            if (artsClusterCaptureEpochs[i][n] &&
                tempPositions[n] < artsClusterCaptureCounts[i][n]) {
              uint64_t epoch = artsClusterCaptureEpochs[i][n][tempPositions[n]];
              if (epoch < minEpoch)
                minEpoch = epoch;
            }
          }
          if (minEpoch == UINT64_MAX)
            break;
          unsigned int nodesWithEpoch = 0;
          for (unsigned int n = 0; n < numNodes; n++) {
            if (artsClusterCaptureEpochs[i][n] &&
                tempPositions[n] < artsClusterCaptureCounts[i][n] &&
                artsClusterCaptureEpochs[i][n][tempPositions[n]] == minEpoch) {
              nodesWithEpoch++;
              tempPositions[n]++;
            }
          }
          if (nodesWithEpoch == numNodes)
            entryCount++;
        }
        artsFree(tempPositions);

        // Build compact JSON string
        size_t bufSize = entryCount * 45 + 10;
        char *buf = (char *)artsMalloc(bufSize);
        char *p = buf;
        *p++ = '[';
        bool first = true;

        while (1) {
          uint64_t minEpoch = UINT64_MAX;
          for (unsigned int n = 0; n < numNodes; n++) {
            if (artsClusterCaptureEpochs[i][n] &&
                nodePositions[n] < artsClusterCaptureCounts[i][n]) {
              uint64_t epoch = artsClusterCaptureEpochs[i][n][nodePositions[n]];
              if (epoch < minEpoch)
                minEpoch = epoch;
            }
          }
          if (minEpoch == UINT64_MAX)
            break;

          uint64_t reducedValue =
              (reduceMethod == artsCounterReduceMin) ? UINT64_MAX : 0;
          unsigned int nodesWithEpoch = 0;
          for (unsigned int n = 0; n < numNodes; n++) {
            if (artsClusterCaptureEpochs[i][n] &&
                nodePositions[n] < artsClusterCaptureCounts[i][n] &&
                artsClusterCaptureEpochs[i][n][nodePositions[n]] == minEpoch) {
              nodesWithEpoch++;
              reducedValue = artsApplyReduction(
                  reducedValue,
                  artsClusterCaptureValues[i][n][nodePositions[n]],
                  reduceMethod, n);
              nodePositions[n]++;
            }
          }
          if (nodesWithEpoch == numNodes) {
            if (!first)
              *p++ = ',';
            first = false;
            uint64_t normalizedEpoch = minEpoch - firstCaptureEpoch;
            p += sprintf(p, "[%llu,%llu]", (unsigned long long)normalizedEpoch,
                         (unsigned long long)reducedValue);
          }
        }
        *p++ = ']';
        *p = '\0';

        artsJsonWriterWriteRawArray(&writer, "captureHistory", buf);
        artsFree(buf);
        artsFree(nodePositions);
      }
      artsJsonWriterEndObject(&writer);
    }
    artsJsonWriterEndObject(&writer);
    artsCloseCounterFile(&writer, fp);

  cleanup:
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsClusterCounterValues && artsClusterCounterValues[i])
        artsFree(artsClusterCounterValues[i]);
      if (artsClusterCaptureEpochs && artsClusterCaptureEpochs[i]) {
        for (unsigned int n = 0; n < numNodes; n++)
          if (artsClusterCaptureEpochs[i][n])
            artsFree(artsClusterCaptureEpochs[i][n]);
        artsFree(artsClusterCaptureEpochs[i]);
      }
      if (artsClusterCaptureValues && artsClusterCaptureValues[i]) {
        for (unsigned int n = 0; n < numNodes; n++)
          if (artsClusterCaptureValues[i][n])
            artsFree(artsClusterCaptureValues[i][n]);
        artsFree(artsClusterCaptureValues[i]);
      }
      if (artsClusterCaptureCounts && artsClusterCaptureCounts[i])
        artsFree(artsClusterCaptureCounts[i]);
    }
    if (artsClusterCounterValues)
      artsFree(artsClusterCounterValues);
    if (artsClusterCaptureEpochs)
      artsFree(artsClusterCaptureEpochs);
    if (artsClusterCaptureValues)
      artsFree(artsClusterCaptureValues);
    if (artsClusterCaptureCounts)
      artsFree(artsClusterCaptureCounts);
    artsClusterCounterValues = NULL;
    artsClusterCaptureEpochs = NULL;
    artsClusterCaptureValues = NULL;
    artsClusterCaptureCounts = NULL;

  } else {
    // Worker: compute and send to master
    artsCounterSendToMaster();
  }
}

// Unified counter write function - single entry point
void artsCounterWrite(const char *outputFolder, unsigned int nodeId,
                      unsigned int threadId) {
  if (!outputFolder)
    return;

  // Write thread-level counters for this thread
  artsCounterWriteThread(outputFolder, nodeId, threadId);

  // Thread 0 handles node and cluster level output
  if (threadId == 0) {
    artsCounterWriteNode(outputFolder, nodeId);
    artsCounterWriteCluster(outputFolder);
  }
}

// ============================================================================
// arts_id tracking wrapper functions (integrated with counter infrastructure)
// ============================================================================

void artsCounterRecordArtsIdEdt(uint64_t arts_id, uint64_t exec_ns,
                                uint64_t stall_ns) {
#if ENABLE_artsIdEdtMetrics
  if (artsCounterMode[artsIdEdtMetrics] != artsCounterModeOff) {
    artsIdRecordEdtMetrics(arts_id, exec_ns, stall_ns,
                           &artsThreadLocalArtsIdMetrics);
  }
#else
  (void)arts_id;
  (void)exec_ns;
  (void)stall_ns;
#endif
}

void artsCounterRecordArtsIdDb(uint64_t arts_id, uint64_t bytes_local,
                               uint64_t bytes_remote, uint64_t cache_misses) {
#if ENABLE_artsIdDbMetrics
  if (artsCounterMode[artsIdDbMetrics] != artsCounterModeOff) {
    artsIdRecordDbMetrics(arts_id, bytes_local, bytes_remote, cache_misses,
                          &artsThreadLocalArtsIdMetrics);
  }
#else
  (void)arts_id;
  (void)bytes_local;
  (void)bytes_remote;
  (void)cache_misses;
#endif
}

void artsCounterCaptureArtsIdEdt(uint64_t arts_id, uint64_t exec_ns,
                                 uint64_t stall_ns) {
#if ENABLE_artsIdEdtCaptures
  if (artsCounterMode[artsIdEdtCaptures] != artsCounterModeOff) {
    artsIdCaptureEdtExecution(arts_id, exec_ns, stall_ns,
                              artsThreadLocalEdtCaptureList);
  }
#else
  (void)arts_id;
  (void)exec_ns;
  (void)stall_ns;
#endif
}

void artsCounterCaptureArtsIdDb(uint64_t arts_id, uint64_t bytes_accessed,
                                uint8_t access_type) {
#if ENABLE_artsIdDbCaptures
  if (artsCounterMode[artsIdDbCaptures] != artsCounterModeOff) {
    artsIdCaptureDbAccess(arts_id, bytes_accessed, access_type,
                          artsThreadLocalDbCaptureList);
  }
#else
  (void)arts_id;
  (void)bytes_accessed;
  (void)access_type;
#endif
}
