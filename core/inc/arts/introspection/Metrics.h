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
#ifndef ARTSMETRICS_H
#define ARTSMETRICS_H

#include "arts/utils/ArrayList.h"

#ifdef __cplusplus
extern "C" {
#endif

#define artsMETRICLEVELS 3
#define artsMAXMETRICNAME 64

/* #define artsMallocWithType(size, type) \
  ({ \
    if (artsMetricsIsActive()) \
      artsThreadInfo.mallocType = type; \
    void *__ptr = artsMalloc(size); \
    if (artsMetricsIsActive()) \
      artsThreadInfo.mallocType = artsDefaultMemorySize; \
    __ptr; \
  }) */

/* #define artsMallocAlignWithType(size, align, type) \
  ({ \
    if (artsMetricsIsActive()) \
      artsThreadInfo.mallocType = type; \
    void *__ptr = artsMallocAlign(size, align); \
    if (artsMetricsIsActive()) \
      artsThreadInfo.mallocType = artsDefaultMemorySize; \
    __ptr; \
  }) */

/* #define artsCallocWithType(nmemb, size, type) \
  ({ \
    if (artsMetricsIsActive()) \
      artsThreadInfo.mallocType = type; \
    void *__ptr = artsCalloc(nmemb, size); \
    if (artsMetricsIsActive()) \
      artsThreadInfo.mallocType = artsDefaultMemorySize; \
    __ptr; \
  }) */

/* #define artsCallocAlignWithType(nmemb, size, align, type) \
  ({ \
    if (artsMetricsIsActive()) \
      artsThreadInfo.mallocType = type; \
    void *__ptr = artsCallocAlign(nmemb, size, align); \
    if (artsMetricsIsActive()) \
      artsThreadInfo.mallocType = artsDefaultMemorySize; \
    __ptr; \
  }) */

#define artsMallocWithType(size, type) artsMalloc(size)
#define artsMallocAlignWithType(size, align, type) artsMallocAlign(size, align)
#define artsCallocWithType(nmemb, size, type) artsCalloc(nmemb, size)
#define artsCallocAlignWithType(nmemb, size, align, type)                      \
  artsCallocAlign(nmemb, size, align)

extern const char *const artsMetricName[];

typedef enum artsMetricType {
  artsFirstMetricType = -1,
  artsEdtThroughput,
  artsEdtQueue,
  artsEdtStealAttempt,
  artsEdtSteal,
  artsEdtLastLocalHit,
  artsEdtSignalThroughput,
  artsEventSignalThroughput,
  artsPersistentEventSignalThroughput,
  artsGetBW,
  artsPutBW,
  artsNetworkSendBW,
  artsNetworkRecieveBW,
  artsNetworkQueuePush,
  artsNetworkQueuePop,
  artsYieldBW,
  artsGpuEdt,
  artsGpuGC,
  artsGpuGCBW,
  artsGpuBWPush,
  artsGpuBWPull,
  artsGpuBufferFlush,
  artsGpuSync,
  artsGpuSyncDelete,
  artsMallocBW,
  artsFreeBW,
  artsRemoteShutdownMsg,
  artsRemoteEdtSignalMsg,
  artsRemoteSignalEdtWithPtrMsg,
  artsRemoteEventSatisfySlotMsg,
  artsRemoteAddDependenceMsg,
  artsRemoteDbRequestMsg,
  artsRemoteDbSendMsg,
  artsRemoteInvalidateDbMsg,
  artsRemoteDbUpdateGuidMsg,
  artsRemoteDbUpdateMsg,
  artsRemoteDbDestroyMsg,
  artsRemoteDbDestroyForwardMsg,
  artsRemoteDbCleanForwardMsg,
  artsRemoteDbMoveReqMsg,
  artsRemoteEdtMoveMsg,
  artsRemoteEventMoveMsg,
  artsRemoteDbMoveMsg,
  artsRemotePingpong_testMsg,
  artsRemoteMetricUpdateMsg,
  artsRemoteDbFullRequestMsg,
  artsRemoteDbFullSendMsg,
  artsRemoteDbFullSendAlready_localMsg,
  artsRemoteGetFromDbMsg,
  artsRemotePutInDbMsg,
  artsRemoteSendMsg,
  artsEpochInitMsg,
  artsEpochInitPoolMsg,
  artsEpochReqMsg,
  artsEpochSendMsg,
  artsEpochDeleteMsg,
  artsAtomicAddArraydbMsg,
  artsAtomicCasArraydbMsg,
  artsRemoteBufferSendMsg,
  artsRemoteContextSigMsg,
  artsRemoteDbRenameMsg,
  artsDefaultMemorySize,
  artsEdtMemorySize,
  artsEventMemorySize,
  artsPersistentEventMemorySize,
  artsDbMemorySize,
  artsBufferMemorySize,
  artsDbCount,
  artsLastMetricType
} artsMetricType;

typedef enum artsMetricLevel {
  artsNoLevel = -1,
  artsThread,
  artsNode,
  artsSystem
} artsMetricLevel;

typedef struct {
  volatile unsigned int reader;
  char pad1[60];
  volatile unsigned int writer;
  char pad2[60];
  volatile unsigned int intervalReader;
  char pad3[60];
  volatile unsigned int intervalWriter;
  char pad4[60];
  volatile uint64_t totalBytes;
  volatile uint64_t totalPackets;
  volatile uint64_t minPacket;
  volatile uint64_t maxPacket;
  volatile uint64_t intervalBytes;
  volatile uint64_t intervalPackets;
  volatile uint64_t intervalMin;
  volatile uint64_t intervalMax;
} artsPacketInspector;

struct artsPerformanceUnit {
  volatile uint64_t totalCount;
  char pad1[56];
  volatile uint64_t maxTotal;
  char pad2[56];
  uint64_t firstTimeStamp;
  char pad3[56];
  volatile unsigned int lock;
  char pad4[60];
  volatile uint64_t windowCountStamp;
  volatile uint64_t windowTimeStamp;
  volatile uint64_t windowMaxTotal;
  volatile uint64_t lastWindowCountStamp;
  volatile uint64_t lastWindowTimeStamp;
  volatile uint64_t lastWindowMaxTotal;
  uint64_t (*timeMethod)(void);
} __attribute__((aligned(64)));

typedef struct artsPerformanceUnit artsPerformanceUnit;

typedef struct {
  unsigned int startPoint;
  uint64_t startTimeStamp;
  uint64_t endTimeStamp;
  artsPerformanceUnit *coreMetric;
  artsPerformanceUnit *nodeMetric;
  artsPerformanceUnit *systemMetric;
} artsInspector;

typedef struct {
  uint64_t nodeUpdates;
  uint64_t systemUpdates;
  uint64_t systemMessages;
  uint64_t remoteUpdates;
} artsInspectorStats;

typedef struct {
  uint64_t windowCountStamp;
  uint64_t windowTimeStamp;
  uint64_t currentCountStamp;
  uint64_t currentTimeStamp;
  uint64_t maxTotal;
} artsMetricShot;

typedef struct {
  artsMetricLevel traceLevel;
  uint64_t initialStart;
  artsArrayList **coreMetric;
  artsArrayList **nodeMetric;
  artsArrayList **systemMetric;
  unsigned int *nodeLock;
  unsigned int *systemLock;
  char *prefix;
} artsInspectorShots;

// void artsMetricsConfigSetDefaultEnabled(bool enabled);
// void artsMetricsConfigSetEnabled(const char *name, bool enabled);
// void artsMetricsTriggerEvent(artsMetricType metricType, artsMetricLevel
// level,
//                              uint64_t value);
// void artsMetricsTriggerTimerEvent(artsMetricType metricType,
//                                   artsMetricLevel level, bool start);
// void artsMetricsToggleThread();
// uint64_t artsMetricsGetInspectorTime();
// bool artsMetricsIsActive();
// void artsMetricsStart(unsigned int startPoint);
// void artsMetricsStop();
// void artsMetricsInitIntrospector(unsigned int startPoint);
// uint64_t artsMetricsGetTotal(artsMetricType type, artsMetricLevel level);
// double artsMetricsGetRate(artsMetricType type, artsMetricLevel level,
//                           bool last);
// double artsMetricsGetTotalRate(artsMetricType type, artsMetricLevel level);
// double artsMetricsTest(artsMetricType type, artsMetricLevel level,
//                        uint64_t num);
// uint64_t artsMetricsGetRateU64(artsMetricType type, artsMetricLevel level,
//                                bool last);
// uint64_t artsMetricsGetRateU64Diff(artsMetricType type, artsMetricLevel
// level,
//                                    uint64_t *diff);
// uint64_t artsMetricsGetTotalRateU64(artsMetricType type, artsMetricLevel
// level,
//                                     uint64_t *total, uint64_t *timeStamp);
// void artsMetricsHandleRemoteUpdate(artsMetricType type, artsMetricLevel
// level,
//                                    uint64_t toAdd, bool sub);
// void artsMetricsPrintInspectorTime();
// void artsMetricsPrintInspectorStats();
// void artsMetricsPrintModelTotalMetrics(artsMetricLevel level);
// void artsMetricsUpdatePacketInfo(uint64_t bytes);
// void artsMetricsPacketStats(uint64_t *totalBytes, uint64_t *totalPackets,
//                             uint64_t *minPacket, uint64_t *maxPacket);
// void artsMetricsIntervalPacketStats(uint64_t *totalBytes,
//                                     uint64_t *totalPackets, uint64_t
//                                     *minPacket, uint64_t *maxPacket);

#define artsMetricsConfigSetDefaultEnabled(enabled) ((void)0)
#define artsMetricsConfigSetEnabled(name, enabled) ((void)0)
#define artsMetricsTriggerEvent(metricType, level, value) ((void)0)
#define artsMetricsTriggerTimerEvent(metricType, level, start) ((void)0)
#define artsMetricsToggleThread()
#define artsMetricsGetInspectorTime() 0
#define artsMetricsIsActive() 0
#define artsMetricsStart(startPoint)
#define artsMetricsStop()
#define artsMetricsInitIntrospector(startPoint)
#define artsMetricsGetTotal(type, level) 0
#define artsMetricsGetRate(type, level, last) 0
#define artsMetricsGetTotalRate(type, level) 0
#define artsMetricsTest(type, level, num) 0
#define artsMetricsGetRateU64(type, level, last) 0
#define artsMetricsGetRateU64Diff(type, level, diff) 0
#define artsMetricsGetTotalRateU64(type, level, total, timeStamp) 0
#define artsMetricsHandleRemoteUpdate(type, level, toAdd, sub) 0
#define artsMetricsPrintInspectorTime()
#define artsMetricsPrintInspectorStats()
#define artsMetricsPrintModelTotalMetrics(level)
#define artsMetricsUpdatePacketInfo(bytes)
#define artsMetricsPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsMetricsIntervalPacketStats(totalBytes, totalPackets, minPacket,    \
                                       maxPacket)

#ifdef __cplusplus
}
#endif

#endif /* ARTSMETRICS_H */
