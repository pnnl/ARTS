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
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "arts/arts.h"
#include "arts/introspection/ArtsIdCounter.h"
#include "arts/introspection/JsonWriter.h"
#include "arts/runtime/network/RemoteFunctions.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"
#include "arts/utils/Atomics.h"

// Arrays are defined as static const in Preamble.h (included via arts.h)

static uint64_t countersOn = 0;
static pthread_t captureThread;
static volatile bool captureThreadRunning = false;

// Time synchronization variables (exported for RemoteFunctions.c)
// timeOffset = workerTime - masterTime (positive if worker is ahead)
// To get synchronized time: localTime - timeOffset = masterTime
volatile int64_t artsCounterTimeOffset = 0;
volatile bool artsCounterTimeSyncReceived = false;

// Cluster counter reduction storage (exported for RemoteFunctions.c)
// Only used on master node (rank 0) to aggregate values from all nodes
uint64_t *artsClusterCounters = NULL;
volatile unsigned int artsClusterNodesReceived = 0;

// Get synchronized timestamp (adjusted to master node's clock)
static inline uint64_t artsGetSyncedTimeStamp(void) {
  // Use acquire semantics to ensure we see the offset written by sync response
  int64_t offset = __atomic_load_n(&artsCounterTimeOffset, __ATOMIC_ACQUIRE);
  return (uint64_t)((int64_t)artsGetTimeStamp() - offset);
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
  // We must set artsCounters to prevent invalid memory access
  artsThreadInfo.artsCounters = (artsCounter *)args;

  // Validate counterCaptureInterval to prevent division by zero
  if (artsNodeInfo.counterCaptureInterval == 0) {
    ARTS_INFO("counterCaptureInterval is 0, capture thread exiting");
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
  uint64_t nextCaptureTime = (captureEpoch + 1) * intervalNs;

  while (captureThreadRunning) {
    if (countersOn) {
      syncedTime = artsGetSyncedTimeStamp();
      // Calculate sleep time until next aligned capture (in synced time)
      int64_t sleepNs = (int64_t)(nextCaptureTime - syncedTime);

      if (sleepNs > 0) {
        ARTS_INFO("Debug: nextCaptureTime=%lf ms, syncedTime=%lf ms, "
                  "sleepTime=%lf ms, offset=%ld ns",
                  (double)nextCaptureTime / 1000000.0,
                  (double)syncedTime / 1000000.0, (double)sleepNs / 1000000.0,
                  __atomic_load_n(&artsCounterTimeOffset, __ATOMIC_RELAXED));
        nanosleep((const struct timespec[]){{sleepNs / 1000000000,
                                             sleepNs % 1000000000}},
                  NULL);
        // Advance to the next expected capture time
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
        nextCaptureTime += intervalsBehind * intervalNs;
      }
    } else {
      // Counters not yet started, wait for interval and re-check
      nanosleep((const struct timespec[]){{intervalNs / 1000000000,
                                           intervalNs % 1000000000}},
                NULL);
      // Re-align to synced clock when counters start
      syncedTime = artsGetSyncedTimeStamp();
      captureEpoch = syncedTime / intervalNs;
      nextCaptureTime = (captureEpoch + 1) * intervalNs;
      continue;
    }
    // Capture counters - only for PERIODIC mode counters
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsCaptureModeArray[i] == artsCaptureModesPeriodic) {
        // Only this thread accesses node captures so no atomic needed
        uint64_t *capture = NULL;
        // CLUSTER level behaves like NODE level during capture
        if (artsCaptureLevelArray[i] == artsCaptureLevelNode ||
            artsCaptureLevelArray[i] == artsCaptureLevelCluster) {
          capture = (uint64_t *)artsNextFreeFromArrayList(
              artsNodeInfo.counterReduces.counterCaptures[i]);
          if (artsCounterReduceTypes[i] == artsCounterSum) {
            *capture = 0;
          } else if (artsCounterReduceTypes[i] == artsCounterMax) {
            *capture = 0;
          } else if (artsCounterReduceTypes[i] == artsCounterMin) {
            *capture = UINT64_MAX;
          }
        }
        // Capture and reduce from all threads
        for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
          artsCounterCaptures *threadCapture = &artsNodeInfo.counterCaptures[t];
          uint64_t captured =
              artsCounterCaptureCounter(&threadCapture->counters[i]);
          artsPushToArrayList(threadCapture->captures[i], &captured);
          if (capture) {
            if (artsCounterReduceTypes[i] == artsCounterSum) {
              *capture += captured;
            } else if (artsCounterReduceTypes[i] == artsCounterMax) {
              if (*capture < captured) {
                *capture = captured;
              }
            } else if (artsCounterReduceTypes[i] == artsCounterMin) {
              if (*capture > captured) {
                *capture = captured;
              }
            }
          }
        }
      }
    }

// Reduce arts_id hash tables for NODE level with PERIODIC mode
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
    if ((artsCaptureModeArray[artsIdEdtMetrics] == artsCaptureModesPeriodic &&
         (artsCaptureLevelArray[artsIdEdtMetrics] == artsCaptureLevelNode ||
          artsCaptureLevelArray[artsIdEdtMetrics] ==
              artsCaptureLevelCluster)) ||
        (artsCaptureModeArray[artsIdDbMetrics] == artsCaptureModesPeriodic &&
         (artsCaptureLevelArray[artsIdDbMetrics] == artsCaptureLevelNode ||
          artsCaptureLevelArray[artsIdDbMetrics] == artsCaptureLevelCluster))) {
      // Reduce all thread hash tables into node-level hash table
      for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
        artsIdReduceHashTables(
            &artsNodeInfo.counterReduces.artsIdMetricsTable,
            &artsNodeInfo.counterCaptures[t].artsIdMetricsTable);
      }
    }
#endif
  }
  return NULL;
}

void artsCounterStart(unsigned int startPoint) {
  if (artsNodeInfo.counterStartPoint == startPoint) {
    if (countersOn) {
      ARTS_DEBUG("Trying to start counters which are already started at %lu",
                 countersOn);
      artsDebugGenerateSegFault();
    }
    countersOn = artsGetTimeStamp();

    bool needCaptureThread = false;
    bool hasNodeLevel = false;

    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      // Need capture thread for PERIODIC mode counters
      if (artsCaptureModeArray[i] == artsCaptureModesPeriodic) {
        needCaptureThread = true;
        if (artsCaptureLevelArray[i] == artsCaptureLevelNode) {
          hasNodeLevel = true;
        }
      }
    }

    if (needCaptureThread) {
      // Time synchronization across nodes using RTT-based algorithm:
      // Worker nodes initiate sync requests to master (rank 0).
      // Master responds with its timestamp.
      // Worker calculates offset accounting for network latency.
      //
      // Algorithm (NTP-like):
      //   T1 = worker send time (worker clock)
      //   T2 = master receive time (master clock)
      //   T3 = worker receive time (worker clock)
      //   RTT = T3 - T1 (clock offsets cancel)
      //   offset = ((T1 - T2) + (T3 - T2)) / 2
      //          = (T1 + T3) / 2 - T2
      // This assumes symmetric network latency.

      if (artsGlobalRankId == 0) {
        // Master node: no offset needed, just mark as synced
        artsCounterTimeOffset = 0;
        artsCounterTimeSyncReceived = true;
        ARTS_INFO("Time sync: Master node (rank 0), offset=0");
      } else {
        // Worker nodes: send sync request to master and wait for response
        artsRemoteTimeSyncReqSend(0); // Send to master (rank 0)

        // Wait for sync response with timeout
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

      int ret = pthread_create(&captureThread, NULL, artsCounterCaptureThread,
                               artsThreadInfo.artsCounters);
      if (ret) {
        ARTS_DEBUG("Failed to create capture thread: %d", ret);
        captureThreadRunning = false;
      } else {
        ARTS_INFO("Counter capture thread started (THREAD=%s, NODE=%s)",
                  needCaptureThread ? "yes" : "no",
                  hasNodeLevel ? "yes" : "no");
      }
    }
  }
}

void artsCounterStop() {
  uint64_t temp = countersOn;
  countersOn = 0;
  if (temp == 0) {
    ARTS_DEBUG("Trying to stop counters which are not started");
    artsDebugGenerateSegFault();
  }

  if (captureThreadRunning) {
    captureThreadRunning = false;
    int ret = pthread_join(captureThread, NULL);
    if (ret) {
      ARTS_DEBUG("Failed to join capture thread: %d", ret);
    } else {
      ARTS_INFO("Counter capture thread stopped");
    }
  }

  ARTS_INFO("Counter on time: %lu", artsGetTimeStamp() - temp);
}

void artsCounterReset(artsCounter *counter) {
  counter->count = 0;
  counter->start = 0;
}

void artsCounterIncrementBy(artsCounter *counter, uint64_t num) {
  if (countersOn) {
    artsAtomicFetchAddU64(&counter->count, num);
  }
}

void artsCounterDecrementBy(artsCounter *counter, uint64_t num) {
  if (countersOn) {
    artsAtomicFetchSubU64(&counter->count, num);
  }
}

void artsCounterTimerStart(artsCounter *counter) {
  if (countersOn) {
    if (artsAtomicCswapU64(&counter->start, 0, artsGetTimeStamp())) {
      ARTS_DEBUG("Trying to start a timer that is already started");
      artsDebugGenerateSegFault();
    }
  }
}

void artsCounterTimerEnd(artsCounter *counter) {
  if (countersOn) {
    uint64_t end = artsGetTimeStamp();
    uint64_t start = artsAtomicSwapU64(&counter->start, 0);
    if (!start) {
      ARTS_DEBUG("Trying to end a timer that is not started");
      artsDebugGenerateSegFault();
    }
    artsAtomicFetchAddU64(&counter->count, end - start);
  }
}

void artsCounterWriteThread(const char *outputFolder, unsigned int nodeId,
                            unsigned int threadId) {
  if (!outputFolder) {
    return;
  }

  // Check if any counters have THREAD level output
  bool hasThreadLevel = false;
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCaptureModeArray[i] != artsCaptureModeOff &&
        artsCaptureLevelArray[i] == artsCaptureLevelThread) {
      hasThreadLevel = true;
      break;
    }
  }
  if (!hasThreadLevel) {
    return;
  }

  struct stat st = {0};
  if (stat(outputFolder, &st) == -1)
    mkdir(outputFolder, 0755);

  char filename[1024];
  snprintf(filename, sizeof(filename), "%s/n%u_t%u.json", outputFolder, nodeId,
           threadId);

  FILE *fp = fopen(filename, "w");
  if (!fp)
    return;

  artsJsonWriter writer;
  artsJsonWriterInit(&writer, fp, 2);

  artsJsonWriterBeginObject(&writer, NULL);

  artsJsonWriterBeginObject(&writer, "metadata");
  artsJsonWriterWriteUInt64(&writer, "nodeId", nodeId);
  artsJsonWriterWriteUInt64(&writer, "threadId", threadId);
  artsJsonWriterWriteUInt64(&writer, "timestamp", (uint64_t)time(NULL));
  artsJsonWriterWriteString(&writer, "version", "1.0");
  artsJsonWriterWriteUInt64(&writer, "startPoint",
                            artsNodeInfo.counterStartPoint);
  if (artsNodeInfo.counterFolder) {
    artsJsonWriterWriteString(&writer, "counterFolder",
                              artsNodeInfo.counterFolder);
  }
  artsJsonWriterWriteUInt64(&writer, "counterCaptureInterval",
                            artsNodeInfo.counterCaptureInterval);
  artsJsonWriterEndObject(&writer);

  artsJsonWriterBeginObject(&writer, "counters");
  artsCounterCaptures *threadCounters = &artsNodeInfo.counterCaptures[threadId];
  for (uint64_t i = 0; i < NUM_COUNTER_TYPES; i++) {
    // Write counters with THREAD level (both ONCE and PERIODIC modes)
    if (artsCaptureModeArray[i] != artsCaptureModeOff &&
        artsCaptureLevelArray[i] == artsCaptureLevelThread) {
      artsCounter *counter = &threadCounters->counters[i];
      artsJsonWriterBeginObject(&writer, artsCounterNames[i]);
      artsJsonWriterWriteUInt64(&writer, "count", counter->count);

      // Write capture mode and level information
      const char *captureModeStr = "OFF";
      switch (artsCaptureModeArray[i]) {
      case artsCaptureModeOnce:
        captureModeStr = "ONCE";
        break;
      case artsCaptureModesPeriodic:
        captureModeStr = "PERIODIC";
        break;
      default:
        break;
      }
      artsJsonWriterWriteString(&writer, "captureMode", captureModeStr);
      artsJsonWriterWriteString(&writer, "captureLevel", "THREAD");

      // Write capture history only for PERIODIC mode
      if (artsCaptureModeArray[i] == artsCaptureModesPeriodic) {
        artsArrayList *captureList = threadCounters->captures[i];
        if (captureList && captureList->index > 0) {
          artsJsonWriterWriteUInt64(&writer, "captureCount",
                                    captureList->index);
          artsJsonWriterBeginArray(&writer, "captureHistory");
          artsArrayListIterator *iter = artsNewArrayListIterator(captureList);
          while (artsArrayListHasNext(iter)) {
            uint64_t *value = (uint64_t *)artsArrayListNext(iter);
            if (value) {
              artsJsonWriterWriteUInt64(&writer, NULL, *value);
            }
          }
          artsDeleteArrayListIterator(iter);
          artsJsonWriterEndArray(&writer);
        }
      }

      artsJsonWriterEndObject(&writer);
    }
  }
  artsJsonWriterEndObject(&writer);

// Write arts_id metrics (THREAD level)
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  if ((artsCaptureModeArray[artsIdEdtMetrics] != artsCaptureModeOff &&
       artsCaptureLevelArray[artsIdEdtMetrics] == artsCaptureLevelThread) ||
      (artsCaptureModeArray[artsIdDbMetrics] != artsCaptureModeOff &&
       artsCaptureLevelArray[artsIdDbMetrics] == artsCaptureLevelThread)) {
    artsJsonWriterBeginObject(&writer, "artsIdMetrics");
    artsIdHashTable *table =
        &artsNodeInfo.counterCaptures[threadId].artsIdMetricsTable;

    // Export EDT metrics
    if (artsCounterMode[artsIdEdtMetrics] == artsCounterModeThread) {
      artsJsonWriterBeginArray(&writer, "edts");
      for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
        if (table->edt_metrics[i].valid) {
          artsJsonWriterBeginObject(&writer, NULL);
          artsJsonWriterWriteUInt64(&writer, "arts_id",
                                    table->edt_metrics[i].arts_id);
          artsJsonWriterWriteUInt64(&writer, "invocations",
                                    table->edt_metrics[i].invocations);
          artsJsonWriterWriteUInt64(&writer, "total_exec_ns",
                                    table->edt_metrics[i].total_exec_ns);
          artsJsonWriterWriteUInt64(&writer, "total_stall_ns",
                                    table->edt_metrics[i].total_stall_ns);
          artsJsonWriterEndObject(&writer);
        }
      }
      artsJsonWriterEndArray(&writer);
      artsJsonWriterWriteUInt64(&writer, "edt_collisions",
                                table->edt_collisions);
    }

    // Export DB metrics
    if (artsCounterMode[artsIdDbMetrics] == artsCounterModeThread) {
      artsJsonWriterBeginArray(&writer, "dbs");
      for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
        if (table->db_metrics[i].valid) {
          artsJsonWriterBeginObject(&writer, NULL);
          artsJsonWriterWriteUInt64(&writer, "arts_id",
                                    table->db_metrics[i].arts_id);
          artsJsonWriterWriteUInt64(&writer, "invocations",
                                    table->db_metrics[i].invocations);
          artsJsonWriterWriteUInt64(&writer, "bytes_local",
                                    table->db_metrics[i].bytes_local);
          artsJsonWriterWriteUInt64(&writer, "bytes_remote",
                                    table->db_metrics[i].bytes_remote);
          artsJsonWriterWriteUInt64(&writer, "cache_misses",
                                    table->db_metrics[i].cache_misses);
          artsJsonWriterEndObject(&writer);
        }
      }
      artsJsonWriterEndArray(&writer);
      artsJsonWriterWriteUInt64(&writer, "db_collisions", table->db_collisions);
    }

    artsJsonWriterEndObject(&writer);
  }
#endif

  artsJsonWriterEndObject(&writer);
  artsJsonWriterFinish(&writer);
  fputc('\n', fp);
  fclose(fp);
}

void artsCounterWriteNode(const char *outputFolder, unsigned int nodeId) {
  if (!outputFolder) {
    return;
  }

  // Check if any counters have NODE level output
  bool hasNodeLevel = false;
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCaptureModeArray[i] != artsCaptureModeOff &&
        artsCaptureLevelArray[i] == artsCaptureLevelNode) {
      hasNodeLevel = true;
      break;
    }
  }
  if (!hasNodeLevel) {
    return;
  }

  struct stat st = {0};
  if (stat(outputFolder, &st) == -1)
    mkdir(outputFolder, 0755);

  char filename[1024];
  snprintf(filename, sizeof(filename), "%s/n%u.json", outputFolder, nodeId);

  FILE *fp = fopen(filename, "w");
  if (!fp)
    return;

  artsJsonWriter writer;
  artsJsonWriterInit(&writer, fp, 2);

  artsJsonWriterBeginObject(&writer, NULL);

  artsJsonWriterBeginObject(&writer, "metadata");
  artsJsonWriterWriteUInt64(&writer, "nodeId", nodeId);
  artsJsonWriterWriteUInt64(&writer, "timestamp", (uint64_t)time(NULL));
  artsJsonWriterWriteString(&writer, "version", "1.0");
  artsJsonWriterWriteUInt64(&writer, "startPoint",
                            artsNodeInfo.counterStartPoint);
  if (artsNodeInfo.counterFolder) {
    artsJsonWriterWriteString(&writer, "counterFolder",
                              artsNodeInfo.counterFolder);
  }
  artsJsonWriterWriteUInt64(&writer, "captureInterval",
                            artsNodeInfo.counterCaptureInterval);
  artsJsonWriterWriteUInt64(&writer, "totalThreads",
                            artsNodeInfo.totalThreadCount);
  artsJsonWriterEndObject(&writer);

  artsJsonWriterBeginObject(&writer, "counters");
  for (uint64_t i = 0; i < NUM_COUNTER_TYPES; i++) {
    // Write NODE level counters (both ONCE and PERIODIC modes)
    if (artsCaptureModeArray[i] != artsCaptureModeOff &&
        artsCaptureLevelArray[i] == artsCaptureLevelNode) {

      artsJsonWriterBeginObject(&writer, artsCounterNames[i]);

      // Write capture mode and level
      const char *captureModeStr = "ONCE";
      if (artsCaptureModeArray[i] == artsCaptureModesPeriodic) {
        captureModeStr = "PERIODIC";
      }
      artsJsonWriterWriteString(&writer, "captureMode", captureModeStr);
      artsJsonWriterWriteString(&writer, "captureLevel", "NODE");

      // Write reduction type
      const char *reduceTypeStr = "SUM";
      switch (artsCounterReduceTypes[i]) {
      case artsCounterSum:
        reduceTypeStr = "SUM";
        break;
      case artsCounterMax:
        reduceTypeStr = "MAX";
        break;
      case artsCounterMin:
        reduceTypeStr = "MIN";
        break;
      }
      artsJsonWriterWriteString(&writer, "reduceType", reduceTypeStr);

      // Write the current (final) reduced value
      uint64_t finalValue = 0;
      for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
        artsCounterCaptures *threadCounters = &artsNodeInfo.counterCaptures[t];
        switch (artsCounterReduceTypes[i]) {
        case artsCounterSum:
          finalValue += threadCounters->counters[i].count;
          break;
        case artsCounterMax:
          if (finalValue < threadCounters->counters[i].count) {
            finalValue = threadCounters->counters[i].count;
          }
          break;
        case artsCounterMin:
          if (t == 0 || finalValue > threadCounters->counters[i].count) {
            finalValue = threadCounters->counters[i].count;
          }
          break;
        }
      }
      artsJsonWriterWriteUInt64(&writer, "value", finalValue);
      // Also write value in milliseconds for timing counters
      artsJsonWriterWriteDouble(&writer, "value_ms",
                                (double)finalValue / 1000000.0);

      // Write capture history only for PERIODIC mode
      if (artsCaptureModeArray[i] == artsCaptureModesPeriodic) {
        artsArrayList *reduceList =
            artsNodeInfo.counterReduces.counterCaptures[i];
        if (reduceList && reduceList->index > 0) {
          artsJsonWriterWriteUInt64(&writer, "captureCount", reduceList->index);
          artsJsonWriterBeginArray(&writer, "captureHistory");
          artsArrayListIterator *iter = artsNewArrayListIterator(reduceList);
          while (artsArrayListHasNext(iter)) {
            uint64_t *value = (uint64_t *)artsArrayListNext(iter);
            if (value) {
              artsJsonWriterWriteUInt64(&writer, NULL, *value);
            }
          }
          artsDeleteArrayListIterator(iter);
          artsJsonWriterEndArray(&writer);
        }
      }

      artsJsonWriterEndObject(&writer);
    }
  }
  artsJsonWriterEndObject(&writer);

// Write arts_id metrics (NODE level - reduced across all threads)
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  if ((artsCaptureModeArray[artsIdEdtMetrics] != artsCaptureModeOff &&
       artsCaptureLevelArray[artsIdEdtMetrics] == artsCaptureLevelNode) ||
      (artsCaptureModeArray[artsIdDbMetrics] != artsCaptureModeOff &&
       artsCaptureLevelArray[artsIdDbMetrics] == artsCaptureLevelNode)) {
    artsJsonWriterBeginObject(&writer, "artsIdMetrics");
    artsIdHashTable *table = &artsNodeInfo.counterReduces.artsIdMetricsTable;

    // Export reduced EDT metrics
    if (artsCounterMode[artsIdEdtMetrics] == artsCounterModeNode) {
      artsJsonWriterBeginArray(&writer, "edts");
      for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
        if (table->edt_metrics[i].valid) {
          artsJsonWriterBeginObject(&writer, NULL);
          artsJsonWriterWriteUInt64(&writer, "arts_id",
                                    table->edt_metrics[i].arts_id);
          artsJsonWriterWriteUInt64(&writer, "invocations",
                                    table->edt_metrics[i].invocations);
          artsJsonWriterWriteUInt64(&writer, "total_exec_ns",
                                    table->edt_metrics[i].total_exec_ns);
          artsJsonWriterWriteUInt64(&writer, "total_stall_ns",
                                    table->edt_metrics[i].total_stall_ns);
          artsJsonWriterEndObject(&writer);
        }
      }
      artsJsonWriterEndArray(&writer);
      artsJsonWriterWriteUInt64(&writer, "edt_collisions",
                                table->edt_collisions);
    }

    // Export reduced DB metrics
    if (artsCounterMode[artsIdDbMetrics] == artsCounterModeNode) {
      artsJsonWriterBeginArray(&writer, "dbs");
      for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
        if (table->db_metrics[i].valid) {
          artsJsonWriterBeginObject(&writer, NULL);
          artsJsonWriterWriteUInt64(&writer, "arts_id",
                                    table->db_metrics[i].arts_id);
          artsJsonWriterWriteUInt64(&writer, "invocations",
                                    table->db_metrics[i].invocations);
          artsJsonWriterWriteUInt64(&writer, "bytes_local",
                                    table->db_metrics[i].bytes_local);
          artsJsonWriterWriteUInt64(&writer, "bytes_remote",
                                    table->db_metrics[i].bytes_remote);
          artsJsonWriterWriteUInt64(&writer, "cache_misses",
                                    table->db_metrics[i].cache_misses);
          artsJsonWriterEndObject(&writer);
        }
      }
      artsJsonWriterEndArray(&writer);
      artsJsonWriterWriteUInt64(&writer, "db_collisions", table->db_collisions);
    }

    artsJsonWriterEndObject(&writer);
  }
#endif

  artsJsonWriterEndObject(&writer);
  artsJsonWriterFinish(&writer);
  fputc('\n', fp);
  fclose(fp);
}

void artsCounterWriteCluster(const char *outputFolder) {
  if (!outputFolder) {
    return;
  }

  // Check if any counters have CLUSTER level output
  bool hasClusterLevel = false;
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCaptureModeArray[i] != artsCaptureModeOff &&
        artsCaptureLevelArray[i] == artsCaptureLevelCluster) {
      hasClusterLevel = true;
      break;
    }
  }
  if (!hasClusterLevel) {
    return;
  }

  unsigned int numNodes = artsGlobalRankCount;

  // Only master (rank 0) writes cluster output
  if (artsGlobalRankId != 0) {
    // Non-master nodes send their counter values to master
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsCaptureModeArray[i] != artsCaptureModeOff &&
          artsCaptureLevelArray[i] == artsCaptureLevelCluster) {
        // Compute this node's reduced value across all threads
        // Initialize based on reduce type for correct MIN handling
        uint64_t nodeValue;
        if (artsCounterReduceTypes[i] == artsCounterMin) {
          nodeValue = UINT64_MAX;
        } else {
          nodeValue = 0;
        }
        for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
          artsCounterCaptures *threadCounters =
              &artsNodeInfo.counterCaptures[t];
          switch (artsCounterReduceTypes[i]) {
          case artsCounterSum:
            nodeValue += threadCounters->counters[i].count;
            break;
          case artsCounterMax:
            if (nodeValue < threadCounters->counters[i].count) {
              nodeValue = threadCounters->counters[i].count;
            }
            break;
          case artsCounterMin:
            if (nodeValue > threadCounters->counters[i].count) {
              nodeValue = threadCounters->counters[i].count;
            }
            break;
          }
        }
        // Send to master
        artsRemoteCounterReduceSend(0, i, nodeValue);
      }
    }
    // Signal master that we're done sending
    artsRemoteCounterReduceDoneSend(0);
    return;
  }

  // Master node code below

  // Allocate cluster counter storage if not already done
  if (!artsClusterCounters) {
    artsClusterCounters =
        (uint64_t *)artsCalloc(NUM_COUNTER_TYPES, sizeof(uint64_t));
    // Initialize all counters based on reduce type for defensive safety
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsCounterReduceTypes[i] == artsCounterMin) {
        artsClusterCounters[i] = UINT64_MAX;
      } else {
        artsClusterCounters[i] = 0;
      }
    }
  }

  // First, add this node's (master's) own values to the cluster counters
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCaptureModeArray[i] != artsCaptureModeOff &&
        artsCaptureLevelArray[i] == artsCaptureLevelCluster) {
      // Compute this node's reduced value across all threads
      // Initialize based on reduce type for correct MIN handling
      uint64_t nodeValue;
      if (artsCounterReduceTypes[i] == artsCounterMin) {
        nodeValue = UINT64_MAX;
      } else {
        nodeValue = 0;
      }
      for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
        artsCounterCaptures *threadCounters = &artsNodeInfo.counterCaptures[t];
        switch (artsCounterReduceTypes[i]) {
        case artsCounterSum:
          nodeValue += threadCounters->counters[i].count;
          break;
        case artsCounterMax:
          if (nodeValue < threadCounters->counters[i].count) {
            nodeValue = threadCounters->counters[i].count;
          }
          break;
        case artsCounterMin:
          if (nodeValue > threadCounters->counters[i].count) {
            nodeValue = threadCounters->counters[i].count;
          }
          break;
        }
      }
      // Aggregate into cluster counters
      switch (artsCounterReduceTypes[i]) {
      case artsCounterSum:
        artsClusterCounters[i] += nodeValue;
        break;
      case artsCounterMax:
        if (artsClusterCounters[i] < nodeValue) {
          artsClusterCounters[i] = nodeValue;
        }
        break;
      case artsCounterMin:
        if (artsClusterCounters[i] > nodeValue) {
          artsClusterCounters[i] = nodeValue;
        }
        break;
      }
    }
  }

  // Wait for all other nodes to send their counter values
  // numNodes - 1 because master doesn't send to itself
  // Timeout after 30 seconds to prevent indefinite blocking
  if (numNodes > 1) {
    unsigned int timeoutMs = 30000; // 30 second timeout
    uint64_t startTime = artsGetTimeStamp();
    while (artsClusterNodesReceived < numNodes - 1) {
      uint64_t elapsedMs =
          (artsGetTimeStamp() - startTime) / 1000000ULL; // convert ns to ms
      if (elapsedMs >= timeoutMs) {
        break;
      }
      usleep(1000); // 1ms sleep to avoid busy waiting
    }
    if (artsClusterNodesReceived < numNodes - 1) {
      ARTS_INFO("Cluster counter aggregation timed out: received %u/%u nodes",
                artsClusterNodesReceived, numNodes - 1);
    }
  }

  // Now write the cluster output file
  struct stat st = {0};
  if (stat(outputFolder, &st) == -1)
    mkdir(outputFolder, 0755);

  char filename[1024];
  snprintf(filename, sizeof(filename), "%s/c.json", outputFolder);

  FILE *fp = fopen(filename, "w");
  if (!fp)
    return;

  artsJsonWriter writer;
  artsJsonWriterInit(&writer, fp, 2);

  artsJsonWriterBeginObject(&writer, NULL);

  artsJsonWriterBeginObject(&writer, "metadata");
  artsJsonWriterWriteUInt64(&writer, "numNodes", numNodes);
  artsJsonWriterWriteUInt64(&writer, "timestamp", (uint64_t)time(NULL));
  artsJsonWriterWriteString(&writer, "version", "1.0");
  artsJsonWriterWriteString(&writer, "level", "CLUSTER");
  artsJsonWriterWriteUInt64(&writer, "startPoint",
                            artsNodeInfo.counterStartPoint);
  if (artsNodeInfo.counterFolder) {
    artsJsonWriterWriteString(&writer, "counterFolder",
                              artsNodeInfo.counterFolder);
  }
  artsJsonWriterEndObject(&writer);

  artsJsonWriterBeginObject(&writer, "counters");
  for (uint64_t i = 0; i < NUM_COUNTER_TYPES; i++) {
    // Write CLUSTER level counters (both ONCE and PERIODIC modes)
    if (artsCaptureModeArray[i] != artsCaptureModeOff &&
        artsCaptureLevelArray[i] == artsCaptureLevelCluster) {

      artsJsonWriterBeginObject(&writer, artsCounterNames[i]);

      // Write capture mode and level
      const char *captureModeStr = "ONCE";
      if (artsCaptureModeArray[i] == artsCaptureModesPeriodic) {
        captureModeStr = "PERIODIC";
      }
      artsJsonWriterWriteString(&writer, "captureMode", captureModeStr);
      artsJsonWriterWriteString(&writer, "captureLevel", "CLUSTER");

      // Write reduction type
      const char *reduceTypeStr = "SUM";
      switch (artsCounterReduceTypes[i]) {
      case artsCounterSum:
        reduceTypeStr = "SUM";
        break;
      case artsCounterMax:
        reduceTypeStr = "MAX";
        break;
      case artsCounterMin:
        reduceTypeStr = "MIN";
        break;
      }
      artsJsonWriterWriteString(&writer, "reduceType", reduceTypeStr);

      // Write the cluster-reduced value
      artsJsonWriterWriteUInt64(&writer, "value", artsClusterCounters[i]);
      // Also write value in milliseconds for timing counters
      artsJsonWriterWriteDouble(&writer, "value_ms",
                                (double)artsClusterCounters[i] / 1000000.0);

      artsJsonWriterEndObject(&writer);
    }
  }
  artsJsonWriterEndObject(&writer);

  artsJsonWriterEndObject(&writer);
  artsJsonWriterFinish(&writer);
  fputc('\n', fp);
  fclose(fp);

  // Free the cluster counters storage
  if (artsClusterCounters) {
    artsFree(artsClusterCounters);
    artsClusterCounters = NULL;
  }
}

// ============================================================================
// arts_id tracking wrapper functions (integrated with counter infrastructure)
// ============================================================================

void artsCounterRecordArtsIdEdt(uint64_t arts_id, uint64_t exec_ns,
                                uint64_t stall_ns) {
#if ENABLE_artsIdEdtMetrics
  if (countersOn && artsCounterMode[artsIdEdtMetrics] != artsCounterModeOff) {
    unsigned int threadId = artsThreadInfo.threadId;
    artsIdHashTable *hash_table =
        &artsNodeInfo.counterCaptures[threadId].artsIdMetricsTable;
    artsIdRecordEdtMetrics(arts_id, exec_ns, stall_ns, hash_table);
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
  if (countersOn && artsCounterMode[artsIdDbMetrics] != artsCounterModeOff) {
    unsigned int threadId = artsThreadInfo.threadId;
    artsIdHashTable *hash_table =
        &artsNodeInfo.counterCaptures[threadId].artsIdMetricsTable;
    artsIdRecordDbMetrics(arts_id, bytes_local, bytes_remote, cache_misses,
                          hash_table);
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
  if (countersOn && artsCounterMode[artsIdEdtCaptures] != artsCounterModeOff) {
    unsigned int threadId = artsThreadInfo.threadId;
    artsArrayList *captures =
        artsNodeInfo.counterCaptures[threadId].artsIdEdtCaptureList;
    artsIdCaptureEdtExecution(arts_id, exec_ns, stall_ns, captures);
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
  if (countersOn && artsCounterMode[artsIdDbCaptures] != artsCounterModeOff) {
    unsigned int threadId = artsThreadInfo.threadId;
    artsArrayList *captures =
        artsNodeInfo.counterCaptures[threadId].artsIdDbCaptureList;
    artsIdCaptureDbAccess(arts_id, bytes_accessed, access_type, captures);
  }
#else
  (void)arts_id;
  (void)bytes_accessed;
  (void)access_type;
#endif
}
