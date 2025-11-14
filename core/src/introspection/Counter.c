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
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"
#include "arts/utils/Atomics.h"

extern const unsigned int artsCounterMode[];

static uint64_t countersOn = 0;
static pthread_t captureThread;
static volatile bool captureThreadRunning = false;

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
  unsigned int cnt = 0;
  // milli to nano
  uint64_t sleepTime = artsNodeInfo.counterCaptureInterval * 1000000;
  while (captureThreadRunning) {
    cnt++;
    if (countersOn) {
      uint64_t adjustedSleepTime =
          countersOn + cnt * sleepTime - artsGetTimeStamp();
      ARTS_INFO(
          "Debug: countersOn=%lf ms, cnt=%u, counterCaptureInterval=%lf ms, "
          "adjustedSleepTime=%lf ms\n",
          (double)countersOn / 1000000.0, cnt, (double)sleepTime / 1000000.0,
          (double)adjustedSleepTime / 1000000.0);
      nanosleep((const struct timespec[]){{adjustedSleepTime / 1000000000,
                                           adjustedSleepTime % 1000000000}},
                NULL);
    } else {
      nanosleep((const struct timespec[]){{sleepTime / 1000000000,
                                           sleepTime % 1000000000}},
                NULL);
      continue;
    }
    // Capture counters
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsCounterMode[i] == artsCounterModeThread ||
          artsCounterMode[i] == artsCounterModeNode) {
        // Only this thread accesses node captures so no atomic needed
        uint64_t *capture = NULL;
        if (artsCounterMode[i] == artsCounterModeNode) {
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

    // Reduce arts_id hash tables for NODE mode
    #if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
    if (artsCounterMode[artsIdEdtMetrics] == artsCounterModeNode ||
        artsCounterMode[artsIdDbMetrics] == artsCounterModeNode) {
      // Reduce all thread hash tables into node-level hash table
      for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
        artsIdReduceHashTables(&artsNodeInfo.counterReduces.artsIdMetricsTable,
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
    bool hasNodeMode = false;

    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsCounterMode[i] == artsCounterModeThread ||
          artsCounterMode[i] == artsCounterModeNode) {
        needCaptureThread = true;
        if (artsCounterMode[i] == artsCounterModeNode) {
          hasNodeMode = true;
        }
      }
    }

    if (needCaptureThread) {
      captureThreadRunning = true;

      int ret = pthread_create(&captureThread, NULL, artsCounterCaptureThread,
                               artsThreadInfo.artsCounters);
      if (ret) {
        ARTS_DEBUG("Failed to create capture thread: %d", ret);
        captureThreadRunning = false;
      } else {
        ARTS_INFO("Counter capture thread started (THREAD=%s, NODE=%s)",
                  needCaptureThread ? "yes" : "no", hasNodeMode ? "yes" : "no");
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

  bool hasThreadMode = false;
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterMode[i] == artsCounterModeThread) {
      hasThreadMode = true;
      break;
    }
  }
  if (!hasThreadMode) {
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
    if (artsCounterMode[i] == artsCounterModeThread) {
      artsCounter *counter = &threadCounters->counters[i];
      artsJsonWriterBeginObject(&writer, artsCounterNames[i]);
      artsJsonWriterWriteUInt64(&writer, "count", counter->count);

      // Write mode information
      const char *modeStr = "OFF";
      switch (artsCounterMode[i]) {
      case artsCounterModeOnce:
        modeStr = "ONCE";
        break;
      case artsCounterModeThread:
        modeStr = "THREAD";
        break;
      case artsCounterModeNode:
        modeStr = "NODE";
        break;
      default:
        break;
      }
      artsJsonWriterWriteString(&writer, "mode", modeStr);

      artsArrayList *captureList = threadCounters->captures[i];
      if (captureList && captureList->index > 0) {
        artsJsonWriterWriteUInt64(&writer, "captureCount", captureList->index);
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

      artsJsonWriterEndObject(&writer);
    }
  }
  artsJsonWriterEndObject(&writer);

  // Write arts_id metrics (THREAD mode)
  #if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  if (artsCounterMode[artsIdEdtMetrics] == artsCounterModeThread ||
      artsCounterMode[artsIdDbMetrics] == artsCounterModeThread) {
    artsJsonWriterBeginObject(&writer, "artsIdMetrics");
    artsIdHashTable *table = &artsNodeInfo.counterCaptures[threadId].artsIdMetricsTable;

    // Export EDT metrics
    if (artsCounterMode[artsIdEdtMetrics] == artsCounterModeThread) {
      artsJsonWriterBeginArray(&writer, "edts");
      for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
        if (table->edt_metrics[i].valid) {
          artsJsonWriterBeginObject(&writer, NULL);
          artsJsonWriterWriteUInt64(&writer, "arts_id", table->edt_metrics[i].arts_id);
          artsJsonWriterWriteUInt64(&writer, "invocations", table->edt_metrics[i].invocations);
          artsJsonWriterWriteUInt64(&writer, "total_exec_ns", table->edt_metrics[i].total_exec_ns);
          artsJsonWriterWriteUInt64(&writer, "total_stall_ns", table->edt_metrics[i].total_stall_ns);
          artsJsonWriterEndObject(&writer);
        }
      }
      artsJsonWriterEndArray(&writer);
      artsJsonWriterWriteUInt64(&writer, "edt_collisions", table->edt_collisions);
    }

    // Export DB metrics
    if (artsCounterMode[artsIdDbMetrics] == artsCounterModeThread) {
      artsJsonWriterBeginArray(&writer, "dbs");
      for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
        if (table->db_metrics[i].valid) {
          artsJsonWriterBeginObject(&writer, NULL);
          artsJsonWriterWriteUInt64(&writer, "arts_id", table->db_metrics[i].arts_id);
          artsJsonWriterWriteUInt64(&writer, "invocations", table->db_metrics[i].invocations);
          artsJsonWriterWriteUInt64(&writer, "bytes_local", table->db_metrics[i].bytes_local);
          artsJsonWriterWriteUInt64(&writer, "bytes_remote", table->db_metrics[i].bytes_remote);
          artsJsonWriterWriteUInt64(&writer, "cache_misses", table->db_metrics[i].cache_misses);
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

  bool hasNodeMode = false;
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (artsCounterMode[i] == artsCounterModeNode) {
      hasNodeMode = true;
      break;
    }
  }
  if (!hasNodeMode) {
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
    // Only write NODE mode counters in the node output
    if (artsCounterMode[i] == artsCounterModeNode) {
      artsJsonWriterBeginObject(&writer, artsCounterNames[i]);
      artsJsonWriterWriteString(&writer, "mode", "NODE");

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
      artsJsonWriterWriteUInt64(&writer, "finalValue", finalValue);

      // Write reduced capture history
      artsArrayList *reduceList = artsNodeInfo.counterReduces.counterCaptures[i];
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

      artsJsonWriterEndObject(&writer);
    }
  }
  artsJsonWriterEndObject(&writer);

  // Write arts_id metrics (NODE mode - reduced across all threads)
  #if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  if (artsCounterMode[artsIdEdtMetrics] == artsCounterModeNode ||
      artsCounterMode[artsIdDbMetrics] == artsCounterModeNode) {
    artsJsonWriterBeginObject(&writer, "artsIdMetrics");
    artsIdHashTable *table = &artsNodeInfo.counterReduces.artsIdMetricsTable;

    // Export reduced EDT metrics
    if (artsCounterMode[artsIdEdtMetrics] == artsCounterModeNode) {
      artsJsonWriterBeginArray(&writer, "edts");
      for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
        if (table->edt_metrics[i].valid) {
          artsJsonWriterBeginObject(&writer, NULL);
          artsJsonWriterWriteUInt64(&writer, "arts_id", table->edt_metrics[i].arts_id);
          artsJsonWriterWriteUInt64(&writer, "invocations", table->edt_metrics[i].invocations);
          artsJsonWriterWriteUInt64(&writer, "total_exec_ns", table->edt_metrics[i].total_exec_ns);
          artsJsonWriterWriteUInt64(&writer, "total_stall_ns", table->edt_metrics[i].total_stall_ns);
          artsJsonWriterEndObject(&writer);
        }
      }
      artsJsonWriterEndArray(&writer);
      artsJsonWriterWriteUInt64(&writer, "edt_collisions", table->edt_collisions);
    }

    // Export reduced DB metrics
    if (artsCounterMode[artsIdDbMetrics] == artsCounterModeNode) {
      artsJsonWriterBeginArray(&writer, "dbs");
      for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
        if (table->db_metrics[i].valid) {
          artsJsonWriterBeginObject(&writer, NULL);
          artsJsonWriterWriteUInt64(&writer, "arts_id", table->db_metrics[i].arts_id);
          artsJsonWriterWriteUInt64(&writer, "invocations", table->db_metrics[i].invocations);
          artsJsonWriterWriteUInt64(&writer, "bytes_local", table->db_metrics[i].bytes_local);
          artsJsonWriterWriteUInt64(&writer, "bytes_remote", table->db_metrics[i].bytes_remote);
          artsJsonWriterWriteUInt64(&writer, "cache_misses", table->db_metrics[i].cache_misses);
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

// ============================================================================
// arts_id tracking wrapper functions (integrated with counter infrastructure)
// ============================================================================

void artsCounterRecordArtsIdEdt(uint64_t arts_id, uint64_t exec_ns, uint64_t stall_ns) {
  #if ENABLE_artsIdEdtMetrics
  if (countersOn && artsCounterMode[artsIdEdtMetrics] != artsCounterModeOff) {
    unsigned int threadId = artsThreadInfo.threadId;
    artsIdHashTable *hash_table = &artsNodeInfo.counterCaptures[threadId].artsIdMetricsTable;
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
    artsIdHashTable *hash_table = &artsNodeInfo.counterCaptures[threadId].artsIdMetricsTable;
    artsIdRecordDbMetrics(arts_id, bytes_local, bytes_remote, cache_misses, hash_table);
  }
  #else
  (void)arts_id;
  (void)bytes_local;
  (void)bytes_remote;
  (void)cache_misses;
  #endif
}

void artsCounterCaptureArtsIdEdt(uint64_t arts_id, uint64_t exec_ns, uint64_t stall_ns) {
  #if ENABLE_artsIdEdtCaptures
  if (countersOn && artsCounterMode[artsIdEdtCaptures] != artsCounterModeOff) {
    unsigned int threadId = artsThreadInfo.threadId;
    artsArrayList *captures = artsNodeInfo.counterCaptures[threadId].artsIdEdtCaptureList;
    artsIdCaptureEdtExecution(arts_id, exec_ns, stall_ns, captures);
  }
  #else
  (void)arts_id;
  (void)exec_ns;
  (void)stall_ns;
  #endif
}

void artsCounterCaptureArtsIdDb(uint64_t arts_id, uint64_t bytes_accessed, uint8_t access_type) {
  #if ENABLE_artsIdDbCaptures
  if (countersOn && artsCounterMode[artsIdDbCaptures] != artsCounterModeOff) {
    unsigned int threadId = artsThreadInfo.threadId;
    artsArrayList *captures = artsNodeInfo.counterCaptures[threadId].artsIdDbCaptureList;
    artsIdCaptureDbAccess(arts_id, bytes_accessed, access_type, captures);
  }
  #else
  (void)arts_id;
  (void)bytes_accessed;
  (void)access_type;
  #endif
}
