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
#include <stdlib.h>
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

// Time synchronization variables (exported for RemoteFunctions.c)
// timeOffset = workerTime - masterTime (positive if worker is ahead)
// To get synchronized time: localTime - timeOffset = masterTime
volatile int64_t artsCounterTimeOffset = 0;
volatile bool artsCounterTimeSyncReceived = false;

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
  //
  // Epochs are RELATIVE to capture thread start, not absolute timestamps.
  // This makes epoch numbers small (0, 1, 2...) and meaningful for comparing
  // captures across different runs. All nodes use the same baseline alignment
  // so their epoch numbers match.
  uint64_t syncedTime = artsGetSyncedTimeStamp();

  // Compute baseline: align to the current interval boundary
  // This ensures all nodes that start within the same interval get the same
  // baseline, producing identical epoch numbers across the cluster.
  uint64_t baselineTime = (syncedTime / intervalNs) * intervalNs;

  // Calculate the capture epoch relative to baseline (starts at 0)
  uint64_t captureEpoch = 0;
  // Next capture will be at the first interval boundary after baseline
  // (i.e., baseline + intervalNs for epoch 0)
  uint64_t nextCaptureTime = baselineTime + intervalNs;

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

    if (artsGlobalRankId == artsGlobalMasterRankId) {
      // Master node: no offset needed, just set ready
      artsCounterTimeOffset = 0;
      artsCounterTimeSyncReceived = true;
      ARTS_INFO("Time sync: Master node (rank %u), offset=0",
                artsGlobalMasterRankId);
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
// Epochs are absolute (synced across nodes) for proper offline merging
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
    p += sprintf(p, "[%llu,%llu]", (unsigned long long)epochs[c],
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
  // Write if we have NODE-level counters OR CLUSTER-level counters
  if (!artsCountersAtLevel(artsCounterLevelNode) &&
      !artsCountersAtLevel(artsCounterLevelCluster))
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

  // Write CLUSTER-level counters (node-reduced values for later cluster reduction)
  // These will be read by master node for file-based cluster aggregation
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] == artsCounterModeOff ||
        artsCounterLevelArray[i] != artsCounterLevelCluster)
      continue;

    // MASTER reduce method counters: only emit on master node
    // Workers don't need to emit these since cluster reduction uses master's value only
    if (artsCounterReduceMethodArray[i] == artsCounterReduceMaster &&
        artsGlobalRankId != artsGlobalMasterRankId)
      continue;

    artsJsonWriterBeginObject(&writer, artsCounterNames[i]);
    artsJsonWriterWriteString(&writer, "captureMode",
                              artsCounterModeToString(artsCounterModeArray[i]));
    artsJsonWriterWriteString(&writer, "captureLevel", "CLUSTER");
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

// Unified counter write function - single entry point
void artsCounterWrite(const char *outputFolder, unsigned int nodeId,
                      unsigned int threadId) {
  if (!outputFolder)
    return;

  // Write thread-level counters for this thread
  artsCounterWriteThread(outputFolder, nodeId, threadId);

  // Thread 0 handles node level output
  if (threadId == 0) {
    artsCounterWriteNode(outputFolder, nodeId);
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

// ============================================================================
// Cluster-level counter aggregation (master node reads all n{id}.json files)
// ============================================================================

// Maximum capture history entries per counter
#define MAX_CAPTURE_HISTORY 10000

// Parsed counter data from a single node's JSON file
typedef struct {
  uint64_t value;
  uint64_t captureCount;
  uint64_t *captureEpochs;
  uint64_t *captureValues;
} artsClusterCounterData;

// Skip whitespace in JSON
static const char *artsJsonSkipWhitespace(const char *p) {
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
    p++;
  return p;
}

// Find a key in JSON object, return pointer to value start
static const char *artsJsonFindKey(const char *json, const char *key) {
  char searchKey[256];
  snprintf(searchKey, sizeof(searchKey), "\"%s\"", key);
  const char *found = strstr(json, searchKey);
  if (!found)
    return NULL;
  found += strlen(searchKey);
  found = artsJsonSkipWhitespace(found);
  if (*found != ':')
    return NULL;
  return artsJsonSkipWhitespace(found + 1);
}

// Parse uint64 from JSON
static uint64_t artsJsonParseUInt64(const char *p) {
  return strtoull(p, NULL, 10);
}

// Parse capture history array: [[epoch,value],[epoch,value],...]
// Returns number of entries parsed
static uint64_t artsJsonParseCaptureHistory(const char *p, uint64_t *epochs,
                                            uint64_t *values,
                                            uint64_t maxEntries) {
  if (!p || *p != '[')
    return 0;
  p++; // Skip opening [

  uint64_t count = 0;
  while (*p && *p != ']' && count < maxEntries) {
    p = artsJsonSkipWhitespace(p);
    if (*p == ',')
      p++;
    p = artsJsonSkipWhitespace(p);
    if (*p != '[')
      break;
    p++; // Skip inner [

    // Parse epoch
    epochs[count] = strtoull(p, (char **)&p, 10);
    p = artsJsonSkipWhitespace(p);
    if (*p == ',')
      p++;
    p = artsJsonSkipWhitespace(p);

    // Parse value
    values[count] = strtoull(p, (char **)&p, 10);
    p = artsJsonSkipWhitespace(p);
    if (*p == ']')
      p++; // Skip inner ]
    count++;
  }
  return count;
}

// Find the end of a JSON object (matching braces)
static const char *artsJsonFindObjectEnd(const char *p) {
  if (*p != '{')
    return NULL;
  int depth = 1;
  p++;
  while (*p && depth > 0) {
    if (*p == '{')
      depth++;
    else if (*p == '}')
      depth--;
    else if (*p == '"') {
      p++;
      while (*p && *p != '"') {
        if (*p == '\\')
          p++;
        p++;
      }
    }
    if (*p)
      p++;
  }
  return p;
}

// Read and parse a node's counter JSON file
static bool artsReadNodeCounterFile(const char *filepath,
                                    artsClusterCounterData *counterData) {
  FILE *fp = fopen(filepath, "r");
  if (!fp)
    return false;

  // Get file size
  fseek(fp, 0, SEEK_END);
  long fileSize = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  if (fileSize <= 0 || fileSize > 10 * 1024 * 1024) { // Max 10MB
    fclose(fp);
    return false;
  }

  char *json = (char *)artsMalloc(fileSize + 1);
  size_t bytesRead = fread(json, 1, fileSize, fp);
  fclose(fp);
  json[bytesRead] = '\0';

  // Find "counters" object
  const char *counters = artsJsonFindKey(json, "counters");
  if (!counters) {
    artsFree(json);
    return false;
  }

  // Parse each CLUSTER-level counter
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] == artsCounterModeOff ||
        artsCounterLevelArray[i] != artsCounterLevelCluster) {
      continue;
    }

    // Find this counter's object
    const char *counterObj = artsJsonFindKey(counters, artsCounterNames[i]);
    if (!counterObj)
      continue;

    // Find the end of this counter object for scoped searching
    const char *counterEnd = artsJsonFindObjectEnd(counterObj);
    size_t counterLen = counterEnd - counterObj;
    char *counterJson = (char *)artsMalloc(counterLen + 1);
    memcpy(counterJson, counterObj, counterLen);
    counterJson[counterLen] = '\0';

    // Parse "value"
    const char *valuePtr = artsJsonFindKey(counterJson, "value");
    if (valuePtr) {
      counterData[i].value = artsJsonParseUInt64(valuePtr);
    }

    // Parse "captureHistory" if present
    const char *historyPtr = artsJsonFindKey(counterJson, "captureHistory");
    if (historyPtr) {
      counterData[i].captureEpochs =
          (uint64_t *)artsMalloc(MAX_CAPTURE_HISTORY * sizeof(uint64_t));
      counterData[i].captureValues =
          (uint64_t *)artsMalloc(MAX_CAPTURE_HISTORY * sizeof(uint64_t));
      counterData[i].captureCount = artsJsonParseCaptureHistory(
          historyPtr, counterData[i].captureEpochs,
          counterData[i].captureValues, MAX_CAPTURE_HISTORY);
    }

    artsFree(counterJson);
  }

  artsFree(json);
  return true;
}

// Merge capture histories from all nodes for a single counter
static void artsMergeCaptureHistories(artsClusterCounterData *nodeData,
                                      unsigned int nodeCount,
                                      unsigned int counterIndex,
                                      uint64_t **outEpochs, uint64_t **outValues,
                                      uint64_t *outCount) {
  *outEpochs = NULL;
  *outValues = NULL;
  *outCount = 0;

  // Find max capture count across all nodes
  uint64_t maxCaptures = 0;
  for (unsigned int n = 0; n < nodeCount; n++) {
    if (nodeData[n * NUM_COUNTER_TYPES + counterIndex].captureCount >
        maxCaptures) {
      maxCaptures = nodeData[n * NUM_COUNTER_TYPES + counterIndex].captureCount;
    }
  }
  if (maxCaptures == 0)
    return;

  // Allocate output arrays
  *outEpochs = (uint64_t *)artsMalloc(maxCaptures * sizeof(uint64_t));
  *outValues = (uint64_t *)artsMalloc(maxCaptures * sizeof(uint64_t));

  // Track current index for each node
  uint64_t *nodeIndices =
      (uint64_t *)artsCalloc(nodeCount, sizeof(uint64_t));

  uint64_t count = 0;
  artsCounterReduceMethod reduceMethod =
      artsCounterReduceMethodArray[counterIndex];

  while (count < maxCaptures) {
    // Find minimum epoch among current positions
    uint64_t minEpoch = UINT64_MAX;
    for (unsigned int n = 0; n < nodeCount; n++) {
      artsClusterCounterData *data =
          &nodeData[n * NUM_COUNTER_TYPES + counterIndex];
      if (nodeIndices[n] < data->captureCount &&
          data->captureEpochs[nodeIndices[n]] < minEpoch) {
        minEpoch = data->captureEpochs[nodeIndices[n]];
      }
    }
    if (minEpoch == UINT64_MAX)
      break;

    // Reduce all values at this epoch
    uint64_t reducedValue =
        (reduceMethod == artsCounterReduceMin) ? UINT64_MAX : 0;
    bool hasValue = false;

    for (unsigned int n = 0; n < nodeCount; n++) {
      artsClusterCounterData *data =
          &nodeData[n * NUM_COUNTER_TYPES + counterIndex];
      if (nodeIndices[n] < data->captureCount &&
          data->captureEpochs[nodeIndices[n]] == minEpoch) {
        hasValue = true;
        reducedValue = artsApplyReduction(
            reducedValue, data->captureValues[nodeIndices[n]], reduceMethod, n);
        nodeIndices[n]++;
      }
    }

    if (hasValue) {
      (*outEpochs)[count] = minEpoch;
      (*outValues)[count] = reducedValue;
      count++;
    }
  }

  *outCount = count;
  artsFree(nodeIndices);
}

// Write cluster-aggregated counter file
void artsCounterWriteCluster(const char *outputFolder, unsigned int nodeCount) {
  if (!outputFolder || nodeCount == 0)
    return;

  // Check if we have any CLUSTER-level counters
  if (!artsCountersAtLevel(artsCounterLevelCluster))
    return;

  ARTS_INFO("Aggregating cluster counters from %u nodes", nodeCount);

  // Allocate storage for all nodes' counter data
  // Layout: nodeData[nodeId * NUM_COUNTER_TYPES + counterIndex]
  artsClusterCounterData *nodeData = (artsClusterCounterData *)artsCalloc(
      nodeCount * NUM_COUNTER_TYPES, sizeof(artsClusterCounterData));

  // Read each node's JSON file
  unsigned int nodesRead = 0;
  for (unsigned int n = 0; n < nodeCount; n++) {
    char filepath[1024];
    snprintf(filepath, sizeof(filepath), "%s/n%u.json", outputFolder, n);
    if (artsReadNodeCounterFile(filepath,
                                &nodeData[n * NUM_COUNTER_TYPES])) {
      nodesRead++;
    } else {
      ARTS_INFO("Warning: Could not read counter file for node %u", n);
    }
  }

  if (nodesRead == 0) {
    ARTS_INFO("No node counter files found, skipping cluster aggregation");
    artsFree(nodeData);
    return;
  }

  // Open output file
  FILE *fp = artsOpenCounterFile(outputFolder, "cluster.json");
  if (!fp) {
    artsFree(nodeData);
    return;
  }

  artsJsonWriter writer;
  artsJsonWriterInit(&writer, fp, 2);
  artsJsonWriterBeginObject(&writer, NULL);

  // Metadata
  artsJsonWriterBeginObject(&writer, "metadata");
  artsJsonWriterWriteString(&writer, "type", "cluster");
  artsJsonWriterWriteUInt64(&writer, "nodeCount", nodeCount);
  artsJsonWriterWriteUInt64(&writer, "nodesRead", nodesRead);
  artsWriteCommonMetadata(&writer);
  artsJsonWriterWriteUInt64(&writer, "captureInterval",
                            artsNodeInfo.counterCaptureInterval);
  artsJsonWriterEndObject(&writer);

  // Counters
  artsJsonWriterBeginObject(&writer, "counters");
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterModeArray[i] == artsCounterModeOff ||
        artsCounterLevelArray[i] != artsCounterLevelCluster) {
      continue;
    }

    artsCounterReduceMethod reduceMethod = artsCounterReduceMethodArray[i];

    artsJsonWriterBeginObject(&writer, artsCounterNames[i]);
    artsJsonWriterWriteString(&writer, "captureMode",
                              artsCounterModeToString(artsCounterModeArray[i]));
    artsJsonWriterWriteString(&writer, "captureLevel", "CLUSTER");
    artsJsonWriterWriteString(&writer, "reduceMethod",
                              artsReduceMethodToString(reduceMethod));

    // Reduce final values across all nodes
    uint64_t clusterValue =
        (reduceMethod == artsCounterReduceMin) ? UINT64_MAX : 0;
    for (unsigned int n = 0; n < nodeCount; n++) {
      clusterValue = artsApplyReduction(
          clusterValue, nodeData[n * NUM_COUNTER_TYPES + i].value,
          reduceMethod, n);
    }
    artsJsonWriterWriteUInt64(&writer, "value", clusterValue);
    if (artsCounterIsTimeCounter(artsCounterNames[i])) {
      artsJsonWriterWriteDouble(&writer, "value_ms",
                                (double)clusterValue / 1000000.0);
    }

    // Merge and write capture histories if PERIODIC
    if (artsCounterModeArray[i] == artsCounterModePeriodic) {
      uint64_t *epochs, *values, count;
      artsMergeCaptureHistories(nodeData, nodeCount, i, &epochs, &values,
                                &count);
      if (count > 0) {
        artsWriteCaptureHistory(&writer, epochs, values, count);
        artsFree(epochs);
        artsFree(values);
      }
    }

    artsJsonWriterEndObject(&writer);
  }
  artsJsonWriterEndObject(&writer);

  artsCloseCounterFile(&writer, fp);

  // Cleanup allocated capture history arrays
  for (unsigned int n = 0; n < nodeCount; n++) {
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      artsClusterCounterData *data = &nodeData[n * NUM_COUNTER_TYPES + i];
      if (data->captureEpochs)
        artsFree(data->captureEpochs);
      if (data->captureValues)
        artsFree(data->captureValues);
    }
  }
  artsFree(nodeData);

  ARTS_INFO("Cluster counter aggregation complete: %s/cluster.json",
            outputFolder);
}
