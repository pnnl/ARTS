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

#include "arts/introspection/Metrics.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// #include "arts/arts.h"
#include "arts/runtime/Globals.h"
// #include "arts/runtime/network/RemoteFunctions.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"
#include "arts/utils/Atomics.h"

#define NANOSECS 1000000000
#define localTimeStamp artsGetTimeStamp
#define globalTimeStamp artsGetTimeStamp

const char *const artsMetricName[] = {"artsEdtThroughput",
                                      "artsEdtQueue",
                                      "artsEdtStealAttempt",
                                      "artsEdtSteal",
                                      "artsEdtLastLocalHit",
                                      "artsEdtSignalThroughput",
                                      "artsEventSignalThroughput",
                                      "artsPersistentEventSignalThroughput",
                                      "artsGetBW",
                                      "artsPutBW",
                                      "artsNetworkSendBW",
                                      "artsNetworkRecieveBW",
                                      "artsNetworkQueuePush",
                                      "artsNetworkQueuePop",
                                      "artsYield",
                                      "artsGpuEdt",
                                      "artsGpuGC",
                                      "artsGpuGCBW",
                                      "artsGpuBWPush",
                                      "artsGpuBWPull",
                                      "artsGpuBufferFlush",
                                      "artsGpuSync",
                                      "artsGpuSyncDelete",
                                      "artsMallocBW",
                                      "artsFreeBW",
                                      "artsRemoteShutdownMsg",
                                      "artsRemoteEdtSignalMsg",
                                      "artsRemoteSignalEdtWithPtrMsg",
                                      "artsRemoteEventSatisfySlotMsg",
                                      "artsRemoteAddDependenceMsg",
                                      "artsRemoteDbRequestMsg",
                                      "artsRemoteDbSendMsg",
                                      "artsRemoteInvalidateDbMsg",
                                      "artsRemoteDbUpdateGuidMsg",
                                      "artsRemoteDbUpdateMsg",
                                      "artsRemoteDbDestroyMsg",
                                      "artsRemoteDbDestroyForwardMsg",
                                      "artsRemoteDbCleanForwardMsg",
                                      "artsRemoteDbMoveReqMsg",
                                      "artsRemoteEdtMoveMsg",
                                      "artsRemoteEventMoveMsg",
                                      "artsRemoteDbMoveMsg",
                                      "artsRemotePingpong_testMsg",
                                      "artsRemoteMetricUpdateMsg",
                                      "artsRemoteDbFullRequestMsg",
                                      "artsRemoteDbFullSendMsg",
                                      "artsRemoteDbFullSendAlready_localMsg",
                                      "artsRemoteGetFromDbMsg",
                                      "artsRemotePutInDbMsg",
                                      "artsRemoteSendMsg",
                                      "artsEpochInitMsg",
                                      "artsEpochInitPoolMsg",
                                      "artsEpochReqMsg",
                                      "artsEpochSendMsg",
                                      "artsEpochDeleteMsg",
                                      "artsAtomicAddArraydbMsg",
                                      "artsAtomicCasArraydbMsg",
                                      "artsRemoteBufferSendMsg",
                                      "artsRemoteContextSigMsg",
                                      "artsRemoteDbRenameMsg",
                                      "artsDefaultMemorySize",
                                      "artsEdtMemorySize",
                                      "artsEventMemorySize",
                                      "artsPersistentEventMemorySize",
                                      "artsDbMemorySize",
                                      "artsBufferMemorySize",
                                      "artsDbCount"};

uint64_t **countWindow;
uint64_t **timeWindow;
uint64_t **maxTotal;

char *printTotalsToFile = NULL;
volatile unsigned int inspectorOn = 0;
artsInspector *inspector = NULL;
artsInspectorStats *stats = NULL;
artsInspectorShots *inspectorShots = NULL;
artsPacketInspector *packetInspector = NULL;

__thread bool inspectorIgnore = 0;

static int metricDefaultEnabled = 1;
static int metricEnabledOverride[artsLastMetricType];
static bool metricOverrideInitialized = false;

static void ensureMetricOverrides() {
  if (!metricOverrideInitialized) {
    for (int i = 0; i < artsLastMetricType; i++)
      metricEnabledOverride[i] = -1;
    metricOverrideInitialized = true;
  }
}

static int metricIndexFromName(const char *name) {
  if (!name)
    return -1;
  for (int i = 0; i < artsLastMetricType; i++) {
    const char *candidate = artsMetricName[i];
    if (candidate && !strcasecmp(candidate, name))
      return i;
  }
  return -1;
}

static inline bool metricIsEnabled(artsMetricType type) {
  if (!metricOverrideInitialized)
    return metricDefaultEnabled;
  int override = metricEnabledOverride[type];
  if (override != -1)
    return override;
  return metricDefaultEnabled;
}

static artsPerformanceUnit *getMetric(artsMetricType type,
                                      artsMetricLevel level);
static artsMetricLevel updatePerformanceCoreMetric(unsigned int core,
                                                   artsMetricType type,
                                                   artsMetricLevel level,
                                                   uint64_t toAdd, bool sub);

// void artsMetricsConfigSetDefaultEnabled(bool enabled) {
//   ensureMetricOverrides();
//   metricDefaultEnabled = enabled ? 1 : 0;
// }

// void artsMetricsConfigSetEnabled(const char *name, bool enabled) {
//   ensureMetricOverrides();
//   int index = metricIndexFromName(name);
//   if (index >= 0)
//     metricEnabledOverride[index] = enabled ? 1 : 0;
// }

// void artsMetricsTriggerEvent(artsMetricType metricType, artsMetricLevel
// level,
//                              uint64_t value) {
//   if (!inspectorOn || inspectorIgnore || !metricIsEnabled(metricType))
//     return;

//   updatePerformanceCoreMetric(artsThreadInfo.threadId, metricType, level,
//   value,
//                               false);
// }

// void artsMetricsTriggerTimerEvent(artsMetricType metricType,
//                                   artsMetricLevel level, bool start) {
//   if (!inspectorOn || inspectorIgnore || !metricIsEnabled(metricType))
//     return;

//   artsPerformanceUnit *metric = getMetric(metricType, level);
//   if (metric) {
//     if (start)
//       metric->firstTimeStamp = metric->timeMethod();
//   }
// }

// void artsMetricsToggleThread() {
//   inspectorIgnore = !inspectorIgnore;
//   ARTS_DEBUG("II: %u\n", inspectorIgnore);
// }

// uint64_t artsMetricsGetInspectorTime() {
//   return inspector ? inspector->startTimeStamp : 0;
// }

// bool artsMetricsIsActive() { return (inspectorOn); }

// void artsMetricsStart(unsigned int startPoint) {
//   if (inspector && inspector->startPoint == startPoint) {
//     inspectorOn = 1;
//     inspector->startTimeStamp = globalTimeStamp();
//   }
// }

// void artsMetricsStop() {
//   if (inspector) {
//     inspectorOn = 0;
//     inspector->endTimeStamp = globalTimeStamp();
//   }
// }

static void printMetrics() {
  for (unsigned int i = 0; i < artsLastMetricType; i++) {
    ARTS_INFO("%35s %ld %ld %ld %ld %ld %ld %ld %ld %ld\n", artsMetricName[i],
              countWindow[i][0], countWindow[i][1], countWindow[i][2],
              timeWindow[i][0], timeWindow[i][1], timeWindow[i][2],
              maxTotal[i][0], maxTotal[i][1], maxTotal[i][2]);
  }
}

// void artsMetricsInitIntrospector(unsigned int startPoint) {
//   ARTS_DEBUG("countWindow %u\n", sizeof(uint64_t *) * artsLastMetricType);
//   countWindow = artsMalloc(sizeof(uint64_t *) * artsLastMetricType);
//   ARTS_DEBUG("timeWindow %u\n", sizeof(uint64_t *) * artsLastMetricType);
//   timeWindow = artsMalloc(sizeof(uint64_t *) * artsLastMetricType);
//   ARTS_DEBUG("maxTotal %u\n", sizeof(uint64_t *) * artsLastMetricType);
//   maxTotal = artsMalloc(sizeof(uint64_t *) * artsLastMetricType);

//   for (unsigned int i = 0; i < artsLastMetricType; i++) {
//     ARTS_DEBUG("countWindow[%u] %u\n", i, sizeof(uint64_t) *
//     artsMETRICLEVELS); countWindow[i] = artsMalloc(sizeof(uint64_t) *
//     artsMETRICLEVELS); ARTS_DEBUG("timeWindow[%u] %u\n", i, sizeof(uint64_t)
//     * artsMETRICLEVELS); timeWindow[i] = artsMalloc(sizeof(uint64_t) *
//     artsMETRICLEVELS); ARTS_DEBUG("maxTotal[%u] %u\n", i, sizeof(uint64_t) *
//     artsMETRICLEVELS); maxTotal[i] = artsMalloc(sizeof(uint64_t) *
//     artsMETRICLEVELS); for (unsigned int j = 0; j < artsMETRICLEVELS; j++) {
//       countWindow[i][j] = -1;
//       timeWindow[i][j] = -1;
//       maxTotal[i][j] = -1;
//     }
//   }

//   if (!artsGlobalRankId)
//     printMetrics();
//   ARTS_DEBUG("inspector %u\n", sizeof(artsInspector));
//   inspector = artsCalloc(1, sizeof(artsInspector));
//   inspector->startPoint = startPoint;
//   ARTS_DEBUG("inspector->coreMetric %u\n", sizeof(artsPerformanceUnit) *
//                                                artsLastMetricType *
//                                                artsNodeInfo.totalThreadCount);
//   inspector->coreMetric =
//       artsCalloc(artsLastMetricType * artsNodeInfo.totalThreadCount,
//                  sizeof(artsPerformanceUnit));
//   for (unsigned int i = 0; i < artsNodeInfo.totalThreadCount; i++) {
//     for (unsigned int j = 0; j < artsLastMetricType; j++) {
//       inspector->coreMetric[i * artsLastMetricType + j].maxTotal =
//           maxTotal[j][0];
//       inspector->coreMetric[i * artsLastMetricType + j].timeMethod =
//           localTimeStamp;
//     }
//   }

//   inspector->nodeMetric =
//       artsCalloc(artsLastMetricType, sizeof(artsPerformanceUnit));
//   for (unsigned int j = 0; j < artsLastMetricType; j++) {
//     inspector->nodeMetric[j].maxTotal = maxTotal[j][1];
//     inspector->nodeMetric[j].timeMethod = globalTimeStamp;
//   }

//   inspector->systemMetric =
//       artsCalloc(artsLastMetricType, sizeof(artsPerformanceUnit));
//   for (unsigned int j = 0; j < artsLastMetricType; j++) {
//     inspector->systemMetric[j].maxTotal = maxTotal[j][2];
//     inspector->systemMetric[j].timeMethod = globalTimeStamp;
//   }

//   ARTS_DEBUG("stats %u\n", sizeof(artsInspectorStats));
//   stats = artsCalloc(1, sizeof(artsInspectorStats));
//   ARTS_DEBUG("packetInspector %u\n", sizeof(artsPacketInspector));
//   packetInspector = artsCalloc(1, sizeof(artsPacketInspector));
//   packetInspector->minPacket = (uint64_t)-1;
//   packetInspector->maxPacket = 0;
//   packetInspector->intervalMin = (uint64_t)-1;
//   packetInspector->intervalMax = 0;
// }

static bool metricTryLock(artsMetricLevel level, artsPerformanceUnit *metric) {
  if (level == artsThread)
    return true;

  unsigned int local;
  while (1) {
    local = artsAtomicCswap(&metric->lock, 0U, 1U);
    if (local != 2U)
      break;
  }
  return (local == 0U);
}

static void metricLock(artsMetricLevel level, artsPerformanceUnit *metric) {
  if (level == artsThread)
    return;
  while (!artsAtomicCswap(&metric->lock, 0U, 1U))
    ;
}

static void metricUnlock(artsPerformanceUnit *metric) { metric->lock = 0U; }

static artsPerformanceUnit *getMetric(artsMetricType type,
                                      artsMetricLevel level) {
  artsPerformanceUnit *metric = NULL;
  if (inspector) {
    switch (level) {
    case artsThread:
      metric =
          &inspector->coreMetric[artsThreadInfo.threadId * artsLastMetricType +
                                 type];
      break;
    case artsNode:
      metric = &inspector->nodeMetric[type];
      break;
    case artsSystem:
      metric = &inspector->systemMetric[type];
      break;
    default:
      metric = NULL;
      break;
    }
  }
  return metric;
}

// uint64_t artsMetricsGetTotal(artsMetricType type, artsMetricLevel level) {
//   artsPerformanceUnit *metric = getMetric(type, level);
//   return (metric) ? metric->totalCount : 0;
// }

// double artsMetricsGetRate(artsMetricType type, artsMetricLevel level,
//                           bool last) {
//   artsPerformanceUnit *metric = getMetric(type, level);
//   if (metric) {
//     uint64_t localWindowTimeStamp;
//     uint64_t localWindowCountStamp;
//     uint64_t localCurrentCountStamp;
//     uint64_t localCurrentTimeStamp;

//     metricLock(level, metric);
//     if (last) {
//       localWindowTimeStamp = metric->lastWindowTimeStamp;
//       localWindowCountStamp = metric->lastWindowCountStamp;
//       localCurrentCountStamp = metric->windowCountStamp;
//       localCurrentTimeStamp = metric->windowTimeStamp;
//       metricUnlock(metric);
//     } else {
//       localWindowTimeStamp = metric->windowTimeStamp;
//       localWindowCountStamp = metric->windowCountStamp;
//       metricUnlock(metric);
//       localCurrentCountStamp = metric->totalCount;
//       localCurrentTimeStamp = metric->timeMethod();
//     }

//     if (localCurrentCountStamp && localCurrentTimeStamp) {
//       double num = (double)(localCurrentCountStamp - localWindowCountStamp);
//       double den = (double)(localCurrentTimeStamp - localWindowTimeStamp);
//       ARTS_INFO("%u %s %lf / %lf\n", level, artsMetricName[type], num, den);
//       return num / den / 1E9;
//     }
//   }
//   return 0;
// }

// double artsMetricsGetTotalRate(artsMetricType type, artsMetricLevel level) {
//   artsPerformanceUnit *metric = getMetric(type, level);
//   if (metric) {
//     double num = (double)metric->totalCount;
//     double den = (double)metric->timeMethod() - inspector->startTimeStamp;
//     ARTS_INFO("%u %s %lf / %lf\n", level, artsMetricName[type], num, den);
//     return num / den;
//   }
//   return 0;
// }

// double artsMetricsTest(artsMetricType type, artsMetricLevel level,
//                        uint64_t num) {
//   artsPerformanceUnit *metric = getMetric(type, level);
//   if (metric && num) {
//     double tot = (double)metric->totalCount;
//     if (tot) {
//       double dif = (double)metric->timeMethod() - inspector->startTimeStamp;
//       double temp = ((double)num * dif) / tot;
//       return temp;
//     }
//     return 100000;
//   }
//   return 0;
// }

// uint64_t artsMetricsGetRateU64(artsMetricType type, artsMetricLevel level,
//                                bool last) {
//   artsPerformanceUnit *metric = getMetric(type, level);
//   if (metric) {
//     uint64_t localWindowTimeStamp;
//     uint64_t localWindowCountStamp;
//     uint64_t localCurrentCountStamp;
//     uint64_t localCurrentTimeStamp;

//     metricLock(level, metric);
//     if (last) {
//       localWindowTimeStamp = metric->lastWindowTimeStamp;
//       localWindowCountStamp = metric->lastWindowCountStamp;
//       localCurrentCountStamp = metric->windowCountStamp;
//       localCurrentTimeStamp = metric->windowTimeStamp;
//       metricUnlock(metric);
//     } else {
//       localWindowTimeStamp = metric->windowTimeStamp;
//       localWindowCountStamp = metric->windowCountStamp;
//       metricUnlock(metric);
//       localCurrentCountStamp = metric->totalCount;
//       localCurrentTimeStamp = metric->timeMethod();
//     }

//     if (localCurrentCountStamp && localCurrentTimeStamp &&
//         localCurrentCountStamp > localWindowCountStamp) {
//       return (localCurrentTimeStamp - localWindowTimeStamp) /
//              (localCurrentCountStamp - localWindowCountStamp);
//     }
//   }
//   return 0;
// }

// uint64_t artsMetricsGetRateU64Diff(artsMetricType type, artsMetricLevel
// level,
//                                    uint64_t *total) {
//   artsPerformanceUnit *metric = getMetric(type, level);
//   if (metric) {
//     metricLock(level, metric);
//     uint64_t localWindowTimeStamp = metric->windowTimeStamp;
//     uint64_t localWindowCountStamp = metric->windowCountStamp;
//     uint64_t lastWindowTimeStamp = metric->lastWindowTimeStamp;
//     uint64_t lastWindowCountStamp = metric->lastWindowCountStamp;
//     metricUnlock(metric);

//     uint64_t localCurrentCountStamp = metric->totalCount;
//     uint64_t localCurrentTimeStamp = metric->timeMethod();
//     *total = localCurrentCountStamp;
//     if (localCurrentCountStamp) {
//       uint64_t diff = localCurrentCountStamp - localWindowCountStamp;
//       if (diff && localWindowTimeStamp) {
//         return (localCurrentTimeStamp - localWindowTimeStamp) / diff;
//       }
//       diff = localWindowCountStamp - lastWindowCountStamp;
//       if (diff && localWindowCountStamp && lastWindowTimeStamp) {
//         return (localWindowCountStamp - lastWindowTimeStamp) / diff;
//       }
//     }
//   }
//   return 0;
// }

// uint64_t artsMetricsGetTotalRateU64(artsMetricType type, artsMetricLevel
// level,
//                                     uint64_t *total, uint64_t *timeStamp) {
//   artsPerformanceUnit *metric = getMetric(type, level);
//   if (metric) {
//     uint64_t localCurrentCountStamp = *total = metric->totalCount;
//     uint64_t localCurrentTimeStamp = metric->timeMethod();
//     *timeStamp = localCurrentTimeStamp;
//     uint64_t startTime = metric->firstTimeStamp;
//     if (startTime && localCurrentCountStamp) {
//       return (localCurrentTimeStamp - startTime) / localCurrentCountStamp;
//     }
//   }
//   return 0;
// }

// void artsMetricsHandleRemoteUpdate(artsMetricType type, artsMetricLevel
// level,
//                                    uint64_t toAdd, bool sub) {
//   artsPerformanceUnit *metric = getMetric(type, level);
//   if (metric) {
//     metricLock(level, metric);
//     if (sub) {
//       metric->windowCountStamp -= toAdd;
//       metric->totalCount -= toAdd;
//     } else {
//       metric->windowCountStamp += toAdd;
//       metric->totalCount += toAdd;
//     }
//     metricUnlock(metric);
//     artsAtomicAddU64(&stats->remoteUpdates, 1);
//   }
// }

static void internalUpdateMax(artsMetricLevel level,
                              artsPerformanceUnit *metric, uint64_t total) {
  uint64_t entry = metric->maxTotal;
  uint64_t localMax = metric->maxTotal;
  if (localMax > total)
    return;
  if (level == artsThread)
    metric->maxTotal = total;
  else {
    while (localMax < total) {
      localMax = artsAtomicCswapU64(&metric->maxTotal, localMax, total);
    }
  }
}

static uint64_t internalObserveMax(artsMetricLevel level,
                                   artsPerformanceUnit *metric) {
  uint64_t max = -1;
  if (metric->maxTotal != -1) {
    if (level == artsThread) {
      max = metric->maxTotal;
      metric->maxTotal = metric->totalCount;
    } else {
      max = artsAtomicSwapU64(&metric->maxTotal, metric->totalCount);
    }
  }
  return max;
}

static bool singleMetricUpdate(artsMetricType type, artsMetricLevel level,
                               uint64_t *toAdd, bool *sub,
                               artsPerformanceUnit *metric) {
  if (!countWindow[type][level] || !timeWindow[type][level])
    return true;

  if (countWindow[type][level] == -1 && timeWindow[type][level] == -1)
    return false;

  uint64_t totalStamp;
  if (*toAdd) {
    if (*sub) {
      if (metric->totalCount < *toAdd) {
        ARTS_INFO(
            "Potential Inspection Underflow Detected! Level: %s Type: %s\n",
            level, artsMetricName[type]);
        artsDebugPrintStack();
      }
      totalStamp = (level == artsThread)
                       ? metric->totalCount -= *toAdd
                       : artsAtomicSubU64(&metric->totalCount, *toAdd);
    } else {
      totalStamp = (level == artsThread)
                       ? metric->totalCount += *toAdd
                       : artsAtomicAddU64(&metric->totalCount, *toAdd);
    }
    internalUpdateMax(level, metric, totalStamp);
  }

  uint64_t localWindowTimeStamp = metric->windowTimeStamp;
  uint64_t localWindowCountStamp = metric->windowCountStamp;

  uint64_t timeStamp = metric->timeMethod();
  if (!localWindowTimeStamp) {
    if (!artsAtomicCswapU64(&metric->windowTimeStamp, 0, timeStamp))
      metric->firstTimeStamp = metric->windowTimeStamp;
    return false;
  }

  uint64_t elapsed =
      (timeStamp > localWindowTimeStamp) ? timeStamp - localWindowTimeStamp : 0;
  uint64_t last = (totalStamp > localWindowCountStamp)
                      ? totalStamp - localWindowCountStamp
                      : localWindowCountStamp - totalStamp;

  if (last >= countWindow[type][level] || elapsed >= timeWindow[type][level]) {
    if (!metricTryLock(level, metric))
      return false;
    if (localWindowTimeStamp != metric->windowTimeStamp) {
      metricUnlock(metric);
      return false;
    }
    ARTS_DEBUG("Check metric %d %d %" PRIu64 " %" PRIu64 " vs %" PRIu64
               " %" PRIu64 "\n",
               level, type, last, elapsed, countWindow[type][level],
               timeWindow[type][level]);
    ARTS_DEBUG("Updating metric %d %d %" PRIu64 " %" PRIu64 "\n", level, type,
               metric->windowCountStamp, metric->windowTimeStamp);
    metric->lastWindowTimeStamp = metric->windowTimeStamp;
    metric->lastWindowCountStamp = metric->windowCountStamp;
    metric->lastWindowMaxTotal = metric->windowMaxTotal;
    metric->windowCountStamp = metric->totalCount;
    metric->windowMaxTotal = internalObserveMax(level, metric);
    metric->windowTimeStamp = metric->timeMethod();
    if (metric->windowCountStamp > metric->lastWindowCountStamp) {
      *toAdd = metric->windowCountStamp - metric->lastWindowCountStamp;
      *sub = false;
    } else {
      *toAdd = metric->lastWindowCountStamp - metric->windowCountStamp;
      *sub = true;
    }
    metricUnlock(metric);
    return true;
  }
  return false;
}

static void takeRateShot(artsMetricType type, artsMetricLevel level,
                         bool last) {
  if (inspectorShots && level >= inspectorShots->traceLevel) {
    if (!countWindow[type][level] || !timeWindow[type][level])
      return;
    ARTS_DEBUG("TRACING LEVEL %d\n", level);

    // int traceOn = artsThreadInfo.mallocTrace;
    // artsThreadInfo.mallocTrace = 0;
    artsPerformanceUnit *metric = getMetric(type, level);
    if (metric) {
      artsArrayList *list = NULL;
      unsigned int *lock = NULL;
      switch (level) {
      case artsThread:
        list = inspectorShots
                   ->coreMetric[artsThreadInfo.threadId * artsLastMetricType +
                                type];
        lock = NULL;
        break;

      case artsNode:
        list = inspectorShots->nodeMetric[type];
        lock = &inspectorShots->nodeLock[type];
        break;

      case artsSystem:
        list = inspectorShots->systemMetric[type];
        lock = &inspectorShots->systemLock[type];
        break;

      default:
        list = NULL;
        lock = NULL;
        break;
      }

      if (list) {
        if (lock) {
          unsigned int local;
          while (1) {
            local = artsAtomicCswap(lock, 0U, 2U);
            if (local == 2U) {
              // artsThreadInfo.mallocTrace = 1;
              return;
            }
            if (!local)
              break;
          }
        }
        artsMetricShot shot;
        if (last) {
          metricLock(level, metric);
          shot.maxTotal = metric->windowMaxTotal;
          shot.windowTimeStamp = metric->lastWindowTimeStamp;
          shot.windowCountStamp = metric->lastWindowCountStamp;
          shot.currentTimeStamp = metric->windowTimeStamp;
          shot.currentCountStamp = metric->windowCountStamp;
          metricUnlock(metric);
        } else {
          metricLock(level, metric);
          shot.windowTimeStamp = metric->windowTimeStamp;
          shot.windowCountStamp = metric->windowCountStamp;
          metricUnlock(metric);

          shot.maxTotal = metric->maxTotal;
          shot.currentCountStamp = metric->totalCount;
          shot.currentTimeStamp = metric->timeMethod();
        }

        // artsThreadInfo.mallocTrace = 0;
        artsPushToArrayList(list, &shot);
        // artsThreadInfo.mallocTrace = 1;

        if (lock)
          *lock = 0U;
      }
    }
    // artsThreadInfo.mallocTrace = traceOn;
  }
}

static artsMetricLevel updatePerformanceCoreMetric(unsigned int core,
                                                   artsMetricType type,
                                                   artsMetricLevel level,
                                                   uint64_t toAdd, bool sub) {
  if (type <= artsFirstMetricType || type >= artsLastMetricType) {
    ARTS_INFO("Wrong Introspection Type %d\n", type);
    artsDebugGenerateSegFault();
  }

  artsMetricLevel updatedLevel = artsNoLevel;
  if (inspectorOn) {
    switch (level) {
    case artsThread:
      ARTS_DEBUG("Thread updated up to %d %" PRIu64 " %u %s\n", updatedLevel,
                 toAdd, sub, artsMetricName[type]);
      if (!singleMetricUpdate(
              type, artsThread, &toAdd, &sub,
              &inspector->coreMetric[core * artsLastMetricType + type]))
        break;
      takeRateShot(type, artsThread, true);
      updatedLevel = artsThread;

    case artsNode:
      ARTS_DEBUG("Node   updated up to %d %" PRIu64 " %u %s\n", updatedLevel,
                 toAdd, sub, artsMetricName[type]);
      if (!singleMetricUpdate(type, artsNode, &toAdd, &sub,
                              &inspector->nodeMetric[type]))
        break;
      artsAtomicAddU64(&stats->nodeUpdates, 1);
      takeRateShot(type, artsNode, true);
      updatedLevel = artsNode;

    case artsSystem:
      ARTS_DEBUG("System updated up to %d %" PRIu64 " %u %s\n", updatedLevel,
                 toAdd, sub, artsMetricName[type]);
      if (singleMetricUpdate(type, artsSystem, &toAdd, &sub,
                             &inspector->systemMetric[type])) {
        uint64_t timeToSend = inspector->systemMetric[type].timeMethod();
        // int traceOn = artsThreadInfo.mallocTrace;
        // artsThreadInfo.mallocTrace = 0;
        // for (unsigned int i = 0; i < artsGlobalRankCount; i++)
        //   if (i != artsGlobalRankId)
        //     artsRemoteMetricUpdate(i, type, level, timeToSend, toAdd, sub);
        // artsThreadInfo.mallocTrace = traceOn;
        artsAtomicAddU64(&stats->systemUpdates, 1);
        if (artsGlobalRankCount > 1)
          artsAtomicAddU64(&stats->systemMessages, artsGlobalRankCount - 1);
        takeRateShot(type, artsSystem, true);
        updatedLevel = artsSystem;
      }
    default:
      break;
    }
  }
  return updatedLevel;
}

static void setThreadMetric(artsMetricType type, uint64_t value) {
  if (countWindow[type][artsThread] == -1 && timeWindow[type][artsThread] == -1)
    return;

  artsPerformanceUnit *metric = getMetric(type, artsThread);
  if (metric) {
    bool shot = true;

    metric->lastWindowCountStamp = metric->windowCountStamp;
    metric->lastWindowTimeStamp = metric->windowTimeStamp;
    metric->lastWindowMaxTotal = metric->maxTotal;

    uint64_t localTime = metric->timeMethod();
    if (!metric->firstTimeStamp) {
      shot = false;
      metric->firstTimeStamp = localTime;
    }

    metric->totalCount = value;
    metric->windowCountStamp = value;
    metric->windowTimeStamp = localTime;
    if (metric->maxTotal < value)
      metric->maxTotal = value;
    if (shot) {
      takeRateShot(type, artsThread, true);
    }
  }
}

void artsMetricsReadConfigFile(char *filename) {
  char *line = NULL;
  size_t length = 0;
  FILE *fp = fopen(filename, "r");
  if (!fp)
    return;

  char temp[artsMAXMETRICNAME];

  while (getline(&line, &length, fp) != -1) {
    ARTS_DEBUG("%s", line);
    if (line[0] != '#') {
      int paramRead = 0;
      sscanf(line, "%s", temp);
      size_t offset = strlen(temp);
      unsigned int metricIndex = -1;
      for (unsigned int i = 0; i < artsLastMetricType; i++) {
        if (!strcmp(temp, artsMetricName[i])) {
          metricIndex = i;
          break;
        }
      }

      if (metricIndex < artsLastMetricType) {
        for (unsigned int i = 0; i < artsMETRICLEVELS; i++) {
          while (line[offset] == ' ')
            offset++;
          paramRead += sscanf(&line[offset], "%" SCNu64 "",
                              &countWindow[metricIndex][i]);
          sscanf(&line[offset], "%s", temp);
          offset += strlen(temp);
          ARTS_DEBUG("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i,
                     offset, countWindow[metricIndex][i]);
        }

        for (unsigned int i = 0; i < artsMETRICLEVELS; i++) {
          while (line[offset] == ' ')
            offset++;
          paramRead +=
              sscanf(&line[offset], "%" SCNu64 "", &timeWindow[metricIndex][i]);
          sscanf(&line[offset], "%s", temp);
          offset += strlen(temp);
          ARTS_DEBUG("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i,
                     offset, timeWindow[metricIndex][i]);
        }

        for (unsigned int i = 0; i < artsMETRICLEVELS; i++) {
          while (line[offset] == ' ')
            offset++;
          paramRead +=
              sscanf(&line[offset], "%" SCNu64 "", &maxTotal[metricIndex][i]);
          sscanf(&line[offset], "%s", temp);
          offset += strlen(temp);
          ARTS_DEBUG("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i,
                     offset, maxTotal[metricIndex][i]);
        }
      }

      if (metricIndex >= artsLastMetricType ||
          paramRead < artsMETRICLEVELS * 2) {
        ARTS_INFO("FAILED to init metric %s\n", temp);
      }
    }
  }
  fclose(fp);

  if (line)
    free(line);
}

// void artsMetricsPrintInspectorTime() {
//   printf("Stat 0 Node %u Start %" PRIu64 " End %" PRIu64 "\n",
//   artsGlobalRankId,
//          inspector->startTimeStamp, inspector->endTimeStamp);
// }

// void artsMetricsPrintInspectorStats() {
//   printf("Stat 3 Node %u Node_Updates %" PRIu64 " System_Updates %" PRIu64
//          " Remote_Updates  %" PRIu64 " System_Messages %" PRIu64 "\n",
//          artsGlobalRankId, stats->nodeUpdates, stats->systemUpdates,
//          stats->remoteUpdates, stats->systemMessages);
// }

// void artsMetricsPrintModelTotalMetrics(artsMetricLevel level) {
//   if (level == artsNode)
//     printf("Stat 1 Node %u edt %" PRIu64 " edt_signal %" PRIu64
//            " event_signal %" PRIu64 " network_sent %" PRIu64
//            " network_recv %" PRIu64 " malloc %" PRIu64 " free %" PRIu64 "\n",
//            artsGlobalRankId, artsMetricsGetTotal(artsEdtThroughput, level),
//            artsMetricsGetTotal(artsEdtSignalThroughput, level),
//            artsMetricsGetTotal(artsEventSignalThroughput, level),
//            artsMetricsGetTotal(artsNetworkSendBW, level),
//            artsMetricsGetTotal(artsNetworkRecieveBW, level),
//            artsMetricsGetTotal(artsMallocBW, level),
//            artsMetricsGetTotal(artsFreeBW, level));
//   else if (level == artsThread) {
//     ARTS_INFO("Stat 1 Thread %u edt %" PRIu64 " edt_signal %" PRIu64
//               " event_signal %" PRIu64 " network_sent %" PRIu64
//               " network_recv %" PRIu64 " malloc %" PRIu64 " free %" PRIu64
//               "\n", artsThreadInfo.threadId,
//               artsMetricsGetTotal(artsEdtThroughput, level),
//               artsMetricsGetTotal(artsEdtSignalThroughput, level),
//               artsMetricsGetTotal(artsEventSignalThroughput, level),
//               artsMetricsGetTotal(artsNetworkSendBW, level),
//               artsMetricsGetTotal(artsNetworkRecieveBW, level),
//               artsMetricsGetTotal(artsMallocBW, level),
//               artsMetricsGetTotal(artsFreeBW, level));
//   }
// }

static inline void updatePacketExtreme(uint64_t val, volatile uint64_t *old,
                                       bool min) {
  uint64_t local = *old;
  uint64_t res;
  if (min) {
    while (val < local) {
      res = artsAtomicCswapU64(old, local, val);
      if (res == local)
        break;
      local = res;
    }
  } else {
    while (val > local) {
      res = artsAtomicCswapU64(old, local, val);
      if (res == local)
        break;
      local = res;
    }
  }
}

// void artsMetricsUpdatePacketInfo(uint64_t bytes) {
//   if (packetInspector) {
//     artsReaderLock(&packetInspector->reader, &packetInspector->writer);
//     artsAtomicAddU64(&packetInspector->totalBytes, bytes);
//     artsAtomicAddU64(&packetInspector->totalPackets, 1U);
//     updatePacketExtreme(bytes, &packetInspector->maxPacket, false);
//     updatePacketExtreme(bytes, &packetInspector->minPacket, true);
//     artsReaderUnlock(&packetInspector->reader);

//     artsReaderLock(&packetInspector->intervalReader,
//                    &packetInspector->intervalWriter);
//     artsAtomicAddU64(&packetInspector->intervalBytes, bytes);
//     artsAtomicAddU64(&packetInspector->intervalPackets, 1U);
//     updatePacketExtreme(bytes, &packetInspector->intervalMax, false);
//     updatePacketExtreme(bytes, &packetInspector->intervalMin, true);
//     artsReaderUnlock(&packetInspector->intervalReader);
//   }
// }

// void artsMetricsPacketStats(uint64_t *totalBytes, uint64_t *totalPackets,
//                             uint64_t *minPacket, uint64_t *maxPacket) {
//   if (packetInspector) {
//     artsWriterLock(&packetInspector->reader, &packetInspector->writer);
//     (*totalBytes) = packetInspector->totalBytes;
//     (*totalPackets) = packetInspector->totalPackets;
//     (*minPacket) = packetInspector->minPacket;
//     (*maxPacket) = packetInspector->maxPacket;
//     artsWriterUnlock(&packetInspector->writer);
//   }
// }

// void artsMetricsIntervalPacketStats(uint64_t *totalBytes,
//                                     uint64_t *totalPackets, uint64_t
//                                     *minPacket, uint64_t *maxPacket) {
//   if (packetInspector) {
//     artsWriterLock(&packetInspector->intervalReader,
//                    &packetInspector->intervalWriter);
//     (*totalBytes) = artsAtomicSwapU64(&packetInspector->totalBytes, 0);
//     (*totalPackets) = artsAtomicSwapU64(&packetInspector->totalPackets, 0);
//     (*minPacket) = artsAtomicSwapU64(&packetInspector->minPacket, 0);
//     (*maxPacket) = artsAtomicSwapU64(&packetInspector->maxPacket, 0);
//     artsWriterUnlock(&packetInspector->intervalWriter);
//   }
// }
