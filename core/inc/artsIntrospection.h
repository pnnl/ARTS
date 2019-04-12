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
#ifndef ARTSINTROSPECTION_H
#define ARTSINTROSPECTION_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"
#include "artsConfig.h"
#include "artsArrayList.h"

#define artsMETRICLEVELS 3
#define artsMAXMETRICNAME 64
#define ARTSSETMEMSHOTTYPE(type) if(artsInternalInspecting()) artsThreadInfo.mallocType =  type
#define artsMEMTRACEON artsThreadInfo.mallocTrace = 1;
#define artsMEMTRACEOFF artsThreadInfo.mallocTrace = 0;

#define artsMETRICNAME const char * const artsMetricName[] = { \
"artsEdtThroughput", \
"artsEdtQueue", \
"artsEdtStealAttempt", \
"artsEdtSteal", \
"artsEdtLastLocalHit", \
"artsEdtSignalThroughput", \
"artsEventSignalThroughput", \
"artsGetBW", \
"artsPutBW", \
"artsNetworkSendBW", \
"artsNetworkRecieveBW", \
"artsNetworkQueuePush", \
"artsNetworkQueuePop", \
"artsYield", \
"artsMallocBW", \
"artsFreeBW", \
"artsRemoteShutdownMsg", \
"artsRemoteEdtStealMsg", \
"artsRemoteEdtRecvMsg", \
"artsRemoteEdtFailMsg", \
"artsRemoteEdtSignalMsg", \
"artsRemoteEventSatisfyMsg", \
"artsRemoteEventSatisfySlotMsg", \
"artsRemoteDbRequestMsg", \
"artsRemoteDbsendMsg", \
"artsRemoteEdtMoveMsg", \
"artsRemoteGuidRouteMsg", \
"artsRemoteEventMoveMsg", \
"artsRemoteAddDependenceMsg", \
"artsRemoteInvalidateDbMsg", \
"artsRemoteDbMoveMsg", \
"artsRemoteDbUpdateGuidMsg", \
"artsRemoteDbUpdateMsg", \
"artsRemoteDbDestroyMsg", \
"artsRemoteDbDestroyForwardMsg", \
"artsRemoteDbCleanForwardMsg", \
"artsRemotePingPongTestMsg", \
"artsDbLockMsg", \
"artsDbUnlockMsg", \
"artsDbLockAllDbsMsg", \
"artsRemoteMetricUpdateMsg", \
"artsActiveMessageMsg", \
"artsRemoteDbFullRequestMsg", \
"artsRemoteDbFullSendMsg", \
"artsRemoteDbFullSendAlreadyLocalMsg", \
"artsRemoteGetFromDbMsg", \
"artsRemotePutInDbMsg", \
"artsRemoteSignalEdtWithPtrMsg", \
"artsRemoteSendMsg", \
"artsEpochInitMsg", \
"artsEpochInitPoolMsg", \
"artsEpochReqMsg", \
"artsEpochSendMsg", \
"artsEpochDeleteMsg", \
"artsAtomicAddArrayDbMsg", \
"artsAtomicCasArrayDbMsg", \
"artsRemoteBufferSendMsg", \
"artsRemoteDbMoveReqMsg", \
"artsDefaultMemorySize", \
"artsEdtMemorySize", \
"artsEventMemorySize", \
"artsDbMemorySize", \
"artsBufferMemorySize", \
"artsDbCount" };

typedef enum artsMetricType 
{
    artsFirstMetricType = -1,
    artsEdtThroughput,
    artsEdtQueue,
    artsEdtStealAttempt,
    artsEdtSteal,
    artsEdtLastLocalHit,
    artsEdtSignalThroughput,
    artsEventSignalThroughput,
    artsGetBW,
    artsPutBW,
    artsNetworkSendBW,
    artsNetworkRecieveBW,
    artsNetworkQueuePush,
    artsNetworkQueuePop,
    artsYieldBW,
    artsMallocBW,
    artsFreeBW,
    artsRemoteShutdownMsg,
    artsRemoteEdtStealMsg,
    artsRemoteEdtRecvMsg,
    artsRemoteEdtFailMsg,
    artsRemoteEdtSignalMsg,
    artsRemoteEventSatisfyMsg,
    artsRemoteEventSatisfySlotMsg,
    artsRemoteDbRequestMsg,
    artsRemoteDbsendMsg,
    artsRemoteEdtMoveMsg,
    artsRemoteGuidRouteMsg,
    artsRemoteEventMoveMsg,
    artsRemoteAddDependenceMsg,
    artsRemoteInvalidateDbMsg,
    artsRemoteDbMoveMsg,
    artsRemoteDbUpdateGuidMsg,
    artsRemoteDbUpdateMsg,
    artsRemoteDbDestroyMsg,
    artsRemoteDbDestroyForwardMsg,
    artsRemoteDbCleanForwardMsg,
    artsRemotePingPongTestMsg,
    artsDbLockMsg,
    artsDbUnlockMsg,
    artsDbLockAllDbsMsg,
    artsRemoteMetricUpdateMsg,
    artsActiveMessageMsg,
    artsRemoteDbFullRequestMsg,
    artsRemoteDbFullSendMsg,
    artsRemoteDbFullSendAlreadyLocalMsg,
    artsRemoteGetFromDbMsg,
    artsRemotePutInDbMsg,
    artsRemoteSignalEdtWithPtrMsg,
    artsRemoteSendMsg, 
    artsEpochInitMsg,
    artsEpochInitPoolMsg,
    artsEpochReqMsg, 
    artsEpochSendMsg,
    artsEpochDeleteMsg,
    artsAtomicAddArrayDbMsg,
    artsAtomicCasArrayDbMsg,
    artsRemoteBufferSendMsg,
    artsRemoteDbMoveReqMsg,
    artsDefaultMemorySize,
    artsEdtMemorySize,
    artsEventMemorySize,
    artsDbMemorySize,
    artsBufferMemorySize,
    artsDbCount,
    artsLastMetricType
} artsMetricType;

typedef enum artsMetricLevel 
{
    artsNoLevel = -1,
    artsThread,
    artsNode,
    artsSystem
} artsMetricLevel;

typedef struct
{
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

struct artsPerformanceUnit
{
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
}  __attribute__ ((aligned(64)));

typedef struct artsPerformanceUnit artsPerformanceUnit;

typedef struct
{
    unsigned int startPoint;
    uint64_t startTimeStamp;
    uint64_t endTimeStamp;
    artsPerformanceUnit * coreMetric;
    artsPerformanceUnit * nodeMetric;
    artsPerformanceUnit * systemMetric;
} artsInspector;

typedef struct
{
    uint64_t nodeUpdates;
    uint64_t systemUpdates;
    uint64_t systemMessages;
    uint64_t remoteUpdates;
} artsInspectorStats;

typedef struct 
{
    uint64_t windowCountStamp;
    uint64_t windowTimeStamp;
    uint64_t currentCountStamp;
    uint64_t currentTimeStamp;
    uint64_t maxTotal;
} artsMetricShot;

typedef struct
{
    artsMetricLevel traceLevel;
    uint64_t initialStart;
    artsArrayList ** coreMetric;
    artsArrayList ** nodeMetric;
    artsArrayList ** systemMetric;
    unsigned int * nodeLock;
    unsigned int * systemLock;
    char * prefix;
} artsInspectorShots;

void artsInternalReadInspectorConfigFile(char * filename);
void artsInternalStartInspector(unsigned int startPoint);
void artsInternalStopInspector(void);
bool artsInternalInspecting(void);
void artsInternalInitIntrospector(struct artsConfig * config);
uint64_t artsInternalGetPerformanceMetricTotal(artsMetricType type, artsMetricLevel level);
uint64_t artsInternalGetPerformanceMetricRateU64(artsMetricType type, artsMetricLevel level, bool last);
uint64_t artsInternalGetPerformanceMetricRateU64Diff(artsMetricType type, artsMetricLevel level, uint64_t * diff);
uint64_t artsInternalGetTotalMetricRateU64(artsMetricType type, artsMetricLevel level, uint64_t * total, uint64_t * timeStamp);
double artsInternalGetPerformanceMetricRate(artsMetricType type, artsMetricLevel level, bool last);
bool artsInternalSingleMetricUpdate(artsMetricType type, artsMetricLevel level, uint64_t *toAdd, bool *sub, artsPerformanceUnit * metric);
void artsInternalHandleRemoteMetricUpdate(artsMetricType type, artsMetricLevel level, uint64_t toAdd, bool sub);
artsMetricLevel artsInternalUpdatePerformanceMetric(artsMetricType type, artsMetricLevel level, uint64_t toAdd, bool sub);
artsMetricLevel artsInternalUpdatePerformanceCoreMetric(unsigned int core, artsMetricType type, artsMetricLevel level, uint64_t toAdd, bool sub);
void artsInternalWriteMetricShotFile(unsigned int threadId, unsigned int nodeId);
void internalPrintTotals(unsigned int nodeId);
void printModelTotalMetrics(artsMetricLevel level);
void artsInternalUpdatePacketInfo(uint64_t bytes);
void artsInternalPacketStats(uint64_t * totalBytes, uint64_t * totalPackets, uint64_t * minPacket, uint64_t * maxPacket);
void artsInternalIntervalPacketStats(uint64_t * totalBytes, uint64_t * totalPackets, uint64_t * minPacket, uint64_t * maxPacket);
void printInspectorStats(void);
void printInspectorTime(void);
void artsInternalSetThreadPerformanceMetric(artsMetricType type, uint64_t value);
uint64_t artsGetInspectorTime(void);
double artsInternalGetPerformanceMetricTotalRate(artsMetricType type, artsMetricLevel level);
double artsMetricTest(artsMetricType type, artsMetricLevel level, uint64_t num);

#ifdef INSPECTOR

#define artsReadInspectorConfigFile(filename) artsInternalReadInspectorConfigFile(filename)
#define artsInitIntrospector(config) artsInternalInitIntrospector(config)
#define artsStartInspector(startPoint) artsInternalStartInspector(startPoint)
#define artsStopInspector() artsInternalStopInspector()
#define artsInspecting() artsInternalInspecting()
#define artsGetPerformanceMetricTotal(type, level) artsInternalGetPerformanceMetricTotal(type, level)
#define artsGetPerformanceMetricRate(type, level, last) artsInternalGetPerformanceMetricRate(type, level, last)
#define artsGetPerformanceMetricTotalRate(type, level) artsInternalGetPerformanceMetricTotalRate(type, level)
#define artsGetPerformanceMetricRateU64(type, level, last) artsInternalGetPerformanceMetricRateU64(type, level, last)
#define artsGetPerformanceMetricRateU64Diff(type, level, diff) artsInternalGetPerformanceMetricRateU64Diff(type, level, diff)
#define artsGetTotalMetricRateU64(type, level, total, timeStamp) artsInternalGetTotalMetricRateU64(type, level, total, timeStamp)
#define artsSingleMetricUpdate(type, level, toAdd, sub, metric) artsInternalSingleMetricUpdate(type, level, toAdd, sub, metric)
#define artsUpdatePerformanceMetric(type, level, toAdd, sub) artsInternalUpdatePerformanceMetric(type, level, toAdd, sub)
#define artsUpdatePerformanceCoreMetric(core, type, level, toAdd, sub) artsInternalUpdatePerformanceCoreMetric(core, type, level, toAdd, sub)
#define artsWriteMetricShotFile(threadId, nodeId) artsInternalWriteMetricShotFile(threadId, nodeId)
#define artsHandleRemoteMetricUpdate(type, level, toAdd, sub) artsInternalHandleRemoteMetricUpdate(type, level, toAdd, sub)
#define artsIntrospectivePrintTotals(nodeId) internalPrintTotals(nodeId)
#define artsUpdatePacketInfo(bytes) artsInternalUpdatePacketInfo(bytes)
#define artsPacketStats(totalBytes, totalPackets, minPacket, maxPacket) artsInternalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket) artsInternalIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsSetThreadPerformanceMetric(type, value) artsInternalSetThreadPerformanceMetric(type, value)

#else

#define artsReadInspectorConfigFile(filename)
#define artsInitIntrospector(config)
#define artsStartInspector(startPoint)
#define artsStopInspector()
#define artsInspecting() 0
#define artsGetPerformanceMetricTotal(type, level) 0
#define artsGetPerformanceMetricRate(type, level, last) 0
#define artsGetPerformanceMetricTotalRate(type, level) 0
#define artsGetPerformanceMetricRateU64(type, level, last) 0
#define artsGetPerformanceMetricRateU64Diff(type, level, diff) 0
#define artsGetTotalMetricRateU64(type, level, total, timeStamp) 0
#define artsSingleMetricUpdate(type, level, toAdd, sub, metric, timeStamp) 0
#define artsUpdatePerformanceMetric(type, level, toAdd, sub) -1
#define artsUpdatePerformanceCoreMetric(core, type, level, toAdd, sub) -1
#define artsWriteMetricShotFile(threadId, nodeId)
#define artsHandleRemoteMetricUpdate(type, level, toAdd, sub)
#define artsIntrospectivePrintTotals(nodeId)
#define artsUpdatePacketInfo(bytes)
#define artsPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsSetThreadPerformanceMetric(type, value)

#endif
#ifdef __cplusplus
}
#endif

#endif /* artsINTROSPECTION_H */

