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
#include "arts/runtime/Runtime.h"

#include "arts/gas/Guid.h"
#include "arts/gas/RouteTable.h"
#include "arts/introspection/ArtsIdCounter.h"
#include "arts/introspection/Counter.h"
#include "arts/introspection/Metrics.h"
#include "arts/introspection/Preamble.h"
#include "arts/network/Remote.h"
#include "arts/network/RemoteProtocol.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/compute/EdtFunctions.h"
#include "arts/runtime/memory/DbFunctions.h"
#include "arts/runtime/sync/TerminationDetection.h"
#include "arts/system/AbstractMachineModel.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/TMT.h"
#include "arts/system/TMTLite.h"
#include "arts/system/Threads.h"
#include "arts/utils/ArrayList.h"
#include "arts/utils/Atomics.h"
#include "arts/utils/Deque.h"

#ifdef USE_GPU
#include "arts/gpu/GpuRuntime.cuh"
#include "arts/gpu/GpuStream.h"
#endif

#define PACKET_SIZE 4096
#define NETWORK_BACKOFF_INCREMENT 0

extern unsigned int numNumaDomains;
extern int mainArgc;
extern char **mainArgv;
#if defined(__APPLE__)
extern void initPerNode(unsigned int nodeId, int argc, char **argv)
    __attribute__((weak_import));
extern void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc,
                          char **argv) __attribute__((weak_import));
extern void artsMain(int argc, char **argv) __attribute__((weak_import));
#else
extern void initPerNode(unsigned int nodeId, int argc, char **argv)
    __attribute__((weak));
extern void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc,
                          char **argv) __attribute__((weak));
extern void artsMain(int argc, char **argv) __attribute__((weak));
#endif

// Weak implementations of optional user functions
__attribute__((weak)) void initPerNode(unsigned int nodeId, int argc,
                                       char **argv) {}
__attribute__((weak)) void initPerWorker(unsigned int nodeId,
                                         unsigned int workerId, int argc,
                                         char **argv) {}
__attribute__((weak)) void artsMain(int argc, char **argv) {}

struct artsRuntimeShared artsNodeInfo;
__thread struct artsRuntimePrivate artsThreadInfo;

typedef bool (*scheduler_t)(void);
#ifdef USE_GPU
scheduler_t schedulerLoop[] = {(scheduler_t)artsDefaultSchedulerLoop,
                               (scheduler_t)artsNetworkBeforeStealSchedulerLoop,
                               (scheduler_t)artsNetworkFirstSchedulerLoop,
                               (scheduler_t)artsGpuSchedulerLoop,
                               (scheduler_t)artsGpuSchedulerBackoffLoop,
                               (scheduler_t)artsGpuSchedulerDemandLoop};
#else
scheduler_t schedulerLoop[] = {(scheduler_t)artsDefaultSchedulerLoop,
                               (scheduler_t)artsNetworkBeforeStealSchedulerLoop,
                               (scheduler_t)artsNetworkFirstSchedulerLoop};
#endif

void artsMainEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                 artsEdtDep_t depv[]) {
  ARTS_DEBUG("Runtime Main EDT called");
  if (artsMain)
    artsMain(mainArgc, mainArgv);
  ARTS_DEBUG("Runtime Main EDT finished");
}

void artsRuntimeNodeInit(unsigned int workerThreads,
                         unsigned int receivingThreads,
                         unsigned int senderThreads,
                         unsigned int receiverThreads,
                         unsigned int totalThreads, bool remoteStealingOn,
                         struct artsConfig *config) {
  artsThreadSetOsThreadCount(config->osThreadCount);
  artsNodeInfo.scheduler = schedulerLoop[config->scheduler];
  artsNodeInfo.deque = (struct artsDeque **)artsMalloc(
      sizeof(struct artsDeque *) * totalThreads);
  artsNodeInfo.receiverDeque =
      receiverThreads ? (struct artsDeque **)artsMalloc(
                            sizeof(struct artsDeque *) * receiverThreads)
                      : NULL;
  artsNodeInfo.gpuDeque = (struct artsDeque **)artsMalloc(
      sizeof(struct artsDeque *) * totalThreads);
  artsNodeInfo.routeTable =
      (artsRouteTable_t **)artsCalloc(totalThreads, sizeof(artsRouteTable_t *));
  artsNodeInfo.gpuRouteTable =
      config->gpu ? (artsRouteTable_t **)artsCalloc(config->gpu,
                                                    sizeof(artsRouteTable_t *))
                  : NULL;
  artsNodeInfo.remoteRouteTable =
      artsNewRouteTable(config->routeTableEntries, config->routeTableSize);
  artsNodeInfo.localSpin =
      (volatile bool **)artsCalloc(totalThreads, sizeof(bool *));
  artsNodeInfo.memoryMoves =
      (unsigned int **)artsCalloc(totalThreads, sizeof(unsigned int *));
  artsNodeInfo.atomicWaits = (struct atomicCreateBarrierInfo **)artsCalloc(
      totalThreads, sizeof(struct atomicCreateBarrierInfo *));
  artsNodeInfo.workerThreadCount = workerThreads;
  artsNodeInfo.senderThreadCount = senderThreads;
  artsNodeInfo.receiverThreadCount = receiverThreads;
  artsNodeInfo.totalThreadCount = totalThreads;
  artsNodeInfo.readyToPush = totalThreads;
  artsNodeInfo.readyToParallelStart = totalThreads;
  artsNodeInfo.readyToInspect = totalThreads;
  artsNodeInfo.readyToExecute = totalThreads;
  artsNodeInfo.readyToClean = totalThreads;
  artsNodeInfo.sendLock = 0U;
  artsNodeInfo.recvLock = 0U;
  artsNodeInfo.shutdownCount = artsGlobalRankCount - 1;
  artsNodeInfo.shutdownStarted = 0;
  artsNodeInfo.readyToShutdown = artsGlobalRankCount - 1;
  artsNodeInfo.stealRequestLock = !remoteStealingOn;
  artsNodeInfo.buf = (char *)artsMalloc(PACKET_SIZE);
  artsNodeInfo.packetSize = PACKET_SIZE;
  artsNodeInfo.printNodeStats = config->printNodeStats;
  artsNodeInfo.shutdownEpoch = (config->shutdownEpoch) ? 1 : NULL_GUID;
  artsNodeInfo.shadLoopStride = config->shadLoopStride;
  artsNodeInfo.tMT = config->tMT;
  artsNodeInfo.gpu = config->gpu;
  artsNodeInfo.gpuRouteTableSize = config->gpuRouteTableSize;
  artsNodeInfo.gpuRouteTableEntries = config->gpuRouteTableEntries;
  artsNodeInfo.gpuLocality = config->gpuLocality;
  artsNodeInfo.gpuFit = config->gpuFit;
  artsNodeInfo.gpuLCSync = config->gpuLCSync;
  artsNodeInfo.gpuMaxEdts = config->gpuMaxEdts;
  artsNodeInfo.gpuMaxMemory = config->gpuMaxMemory;
  artsNodeInfo.gpuP2P = config->gpuP2P;
  artsNodeInfo.gpuBuffOn = config->gpuBuffOn;
  artsNodeInfo.freeDbAfterGpuRun = config->freeDbAfterGpuRun;
  artsNodeInfo.runGpuGcIdle = config->runGpuGcIdle;
  artsNodeInfo.runGpuGcPreEdt = config->runGpuGcPreEdt;
  artsNodeInfo.deleteZerosGpuGc = config->deleteZerosGpuGc;
  artsNodeInfo.pinThreads = config->pinThreads;
  artsNodeInfo.keys = (uint64_t **)artsCalloc(totalThreads, sizeof(uint64_t *));
  artsNodeInfo.globalGuidThreadId =
      (uint64_t *)artsCalloc(totalThreads, sizeof(uint64_t));
  artsNodeInfo.counterFolder = config->counterFolder;

  // Allocate liveCounters array - pointers to each thread's __thread counters
  // These will be registered by each thread during artsRuntimePrivateInit
  artsNodeInfo.liveCounters =
      (artsCounter **)artsCalloc(totalThreads, sizeof(artsCounter *));

  // Allocate savedCounters - nodeInfo's own counter space for final values
  // Pre-allocate all counters for all threads upfront
  artsNodeInfo.savedCounters =
      (artsCounter **)artsCalloc(totalThreads, sizeof(artsCounter *));
  for (unsigned int t = 0; t < totalThreads; t++) {
    artsNodeInfo.savedCounters[t] =
        (artsCounter *)artsCalloc(NUM_COUNTER_TYPES, sizeof(artsCounter));
  }

  // Allocate captureArrays for periodic capture (capture thread writes here)
  // Pre-allocate all ArrayLists for PERIODIC mode counters
  artsNodeInfo.captureArrays =
      (artsArrayList ***)artsCalloc(totalThreads, sizeof(artsArrayList **));
  for (unsigned int t = 0; t < totalThreads; t++) {
    artsNodeInfo.captureArrays[t] = (artsArrayList **)artsCalloc(
        NUM_COUNTER_TYPES, sizeof(artsArrayList *));
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (artsCounterModeArray[i] == artsCounterModePeriodic) {
        artsNodeInfo.captureArrays[t][i] =
            artsNewArrayList(sizeof(artsCounterCapture), 16);
      }
    }
  }

  artsNodeInfo.counterCaptureInterval = config->counterCaptureInterval;

  // Note: Node-level counter reduction is done at output time using
  // savedCounters. arts_id node-level reduction also computed at output time.

  artsTMTNodeInit(workerThreads);
  artsInitTMTLitePerNode(workerThreads);
#ifdef USE_GPU
  if (artsNodeInfo.gpu) // TODO: Multi-Node init
    artsNodeInitGpus();
#endif
}

void artsRuntimeGlobalCleanup() {
  artsCounterCaptureStop();
  // Write all counter outputs (thread, node, and cluster levels)
  for (unsigned int t = 0; t < artsNodeInfo.totalThreadCount; t++) {
    artsCounterWrite(artsNodeInfo.counterFolder, artsGlobalRankId, t);
  }
  artsCleanUpDbs();
  artsFree(artsNodeInfo.deque);
  artsFree(artsNodeInfo.receiverDeque);
  artsFree(artsNodeInfo.gpuDeque);
  artsFree(artsNodeInfo.gpuRouteTable);
  artsFree((void *)artsNodeInfo.localSpin);
  artsFree(artsNodeInfo.memoryMoves);
  artsFree(artsNodeInfo.atomicWaits);
  artsFree(artsNodeInfo.buf);
  artsFree(artsNodeInfo.keys);
  artsFree(artsNodeInfo.globalGuidThreadId);
#ifdef USE_GPU
  if (artsNodeInfo.gpu)
    artsCleanupGpus();
#endif
}

void artsThreadZeroNodeStart() {

  setGlobalGuidOn();
  createShutdownEpoch();

  artsCounterCaptureStart();
  INITIALIZATION_TIME_STOP();
  END_TO_END_TIME_START();

  if (initPerNode)
    initPerNode(artsGlobalRankId, mainArgc, mainArgv);

#ifdef USE_GPU
  artsInitPerGpuWrapper(mainArgc, mainArgv);
#endif
  setGuidGeneratorAfterParallelStart();

  artsAtomicSub(&artsNodeInfo.readyToParallelStart, 1U);
  while (artsNodeInfo.readyToParallelStart) {
  }
  if (initPerWorker && artsThreadInfo.worker)
    initPerWorker(artsGlobalRankId, artsThreadInfo.groupId, mainArgc, mainArgv);

  if (artsMain && !artsGlobalRankId)
    artsEdtCreate(artsMainEdt, 0, 0, NULL, 0);

  artsIncrementFinishedEpochList();

  artsAtomicSub(&artsNodeInfo.readyToInspect, 1U);
  while (artsNodeInfo.readyToInspect) {
  }
  artsAtomicSub(&artsNodeInfo.readyToExecute, 1U);
  while (artsNodeInfo.readyToExecute) {
  }
}

void artsRuntimePrivateInit(struct threadMask *unit,
                            struct artsConfig *config) {
  artsNodeInfo.deque[unit->id] = artsThreadInfo.myDeque =
      artsDequeNew(config->dequeSize);
  artsNodeInfo.gpuDeque[unit->id] = artsThreadInfo.myGpuDeque =
      (config->gpu) ? artsDequeNew(config->dequeSize) : NULL;
  if (unit->worker) {
    artsNodeInfo.routeTable[unit->id] =
        artsNewRouteTable(config->routeTableEntries, config->routeTableSize);
#ifdef USE_GPU
    if (config->gpu) // TODO: Multi-Node init
      artsWorkerInitGpus();
#endif
  }

  if (unit->networkSend || unit->networkReceive) {
    if (unit->networkSend) {
      unsigned int size =
          artsGlobalRankCount * config->ports / artsNodeInfo.senderThreadCount;
      unsigned int rem =
          artsGlobalRankCount * config->ports % artsNodeInfo.senderThreadCount;
      unsigned int start;
      if (unit->groupPos < rem) {
        start = unit->groupPos * (size + 1);
        artsRemoteSetThreadOutboundQueues(start, start + size + 1);
      } else {
        start = rem * (size + 1) + (unit->groupPos - rem) * size;
        artsRemoteSetThreadOutboundQueues(start, start + size);
      }
    }
    if (unit->networkReceive) {
      artsNodeInfo.receiverDeque[unit->groupPos] = artsNodeInfo.deque[unit->id];
      unsigned int size = (artsGlobalRankCount - 1) * config->ports /
                          artsNodeInfo.receiverThreadCount;
      unsigned int rem = (artsGlobalRankCount - 1) * config->ports %
                         artsNodeInfo.receiverThreadCount;
      unsigned int start;
      if (unit->groupPos < rem) {
        start = unit->groupPos * (size + 1);
        // ARTS_INFO("%d %d %d %d", start, size, unit->groupPos, rem);
        artsRemoteSetThreadInboundQueues(start, start + size + 1);
      } else {
        start = rem * (size + 1) + (unit->groupPos - rem) * size;
        // ARTS_INFO("%d %d %d %d", start, size, unit->groupPos, rem);
        artsRemoteSetThreadInboundQueues(start, start + size);
      }
    }
  }
  artsNodeInfo.localSpin[unit->id] = &artsThreadInfo.alive;
  artsThreadInfo.alive = true;
  artsNodeInfo.memoryMoves[unit->id] =
      (unsigned int *)&artsThreadInfo.oustandingMemoryMoves;
  artsNodeInfo.atomicWaits[unit->id] = &artsThreadInfo.atomicWait;
  artsThreadInfo.atomicWait.wait = true;
  artsThreadInfo.oustandingMemoryMoves = 0;
  artsThreadInfo.coreId = unit->unitId;
  artsThreadInfo.threadId = unit->id;
  artsThreadInfo.groupId = unit->groupPos;
  artsThreadInfo.clusterId = unit->clusterId;
  artsThreadInfo.worker = unit->worker;
  artsThreadInfo.networkSend = unit->networkSend;
  artsThreadInfo.networkReceive = unit->networkReceive;
  artsThreadInfo.backOff = 1;
  artsThreadInfo.currentEdtGuid = 0;
  artsThreadInfo.mallocType = artsDefaultMemorySize;
  artsThreadInfo.mallocTrace = 1;
  artsThreadInfo.localCounting = 1;
  artsThreadInfo.shadLock = 0;

  // Register thread-local counter storage with nodeInfo
  artsNodeInfo.liveCounters[unit->id] = artsThreadLocalCounters;
#if ENABLE_artsIdEdtMetrics || ENABLE_artsIdDbMetrics
  artsIdInitHashTable(&artsThreadLocalArtsIdMetrics);
#endif
#if ENABLE_artsIdEdtCaptures
  artsThreadLocalEdtCaptureList =
      artsNewArrayList(sizeof(artsIdEdtCapture), 1024);
#endif
#if ENABLE_artsIdDbCaptures
  artsThreadLocalDbCaptureList =
      artsNewArrayList(sizeof(artsIdDbCapture), 1024);
#endif

  artsGuidKeyGeneratorInit();

  if (artsThreadInfo.worker) {
    if (artsNodeInfo.tMT && artsThreadInfo.worker) // @awmm
    {
      ARTS_DEBUG("tMT: PthreadLayer: preparing aliasing for master thread %d",
                 unit->id);
      artsTMTRuntimePrivateInit(unit, &artsThreadInfo);
    }
    artsInitTMTLitePerWorker(artsThreadInfo.groupId);
  }

  artsAtomicSub(&artsNodeInfo.readyToPush, 1U);
  while (artsNodeInfo.readyToPush) {
  };
  if (unit->id) {
    artsAtomicSub(&artsNodeInfo.readyToParallelStart, 1U);
    while (artsNodeInfo.readyToParallelStart) {
    };

    if (artsThreadInfo.worker) {
      if (initPerWorker)
        initPerWorker(artsGlobalRankId, artsThreadInfo.groupId, mainArgc,
                      mainArgv);
      artsIncrementFinishedEpochList();
    }

    artsAtomicSub(&artsNodeInfo.readyToInspect, 1U);
    while (artsNodeInfo.readyToInspect) {
    };
    artsAtomicSub(&artsNodeInfo.readyToExecute, 1U);
    while (artsNodeInfo.readyToExecute) {
    };
  }
  artsThreadInfo.drand_buf[0] = 1202107158 + unit->id * 1999;
  artsThreadInfo.drand_buf[1] = 0;
  artsThreadInfo.drand_buf[2] = 0;
}

void artsRuntimePrivateCleanup() {
  if (artsThreadInfo.worker) {
    artsTMTRuntimePrivateCleanup();
    artsTMTLitePrivateCleanUp(artsThreadInfo.groupId);
  }
  artsAtomicSub(&artsNodeInfo.readyToClean, 1U);
  while (artsNodeInfo.readyToClean) {
  };
  artsRemoteThreadOutboundQueuesCleanup();
  artsRemoteThreadInboundQueuesCleanup();
  if (artsThreadInfo.myDeque)
    artsDequeDelete(artsThreadInfo.myDeque);
  if (artsThreadInfo.myNodeDeque)
    artsDequeDelete(artsThreadInfo.myNodeDeque);
  if (artsThreadInfo.myGpuDeque)
    artsDequeDelete(artsThreadInfo.myGpuDeque);
}

void artsRuntimeStop() {
  unsigned int i;
  for (i = 0; i < artsNodeInfo.totalThreadCount; i++) {
    while (!artsNodeInfo.localSpin[i])
      ;
    (*artsNodeInfo.localSpin[i]) = false;
  }
  artsTMTRuntimeStop();
  artsTMTLiteShutdown();
}

void artsHandleRemoteStolenEdt(struct artsEdt *edt) {
  ARTS_DEBUG("Processing stolen EDT[Id:%lu, Guid:%lu] on core %d", edt->arts_id,
             edt->currentEdt, artsThreadInfo.coreId);
  incrementQueueEpoch(edt->epochGuid);
  globalShutdownGuidIncQueue();
#ifdef USE_GPU
  if (artsNodeInfo.gpu &&
      (!artsThreadInfo.myDeque || !artsThreadInfo.myGpuDeque))
    artsStoreNewEdts(edt);
  else
#endif
  {
    if (edt->header.type == ARTS_EDT)
      artsDequePushFront(artsThreadInfo.myDeque, edt, 0);
    else if (edt->header.type == ARTS_GPU_EDT)
      artsDequePushFront(artsThreadInfo.myGpuDeque, edt, 0);
  }
}

void artsHandleReadyEdt(struct artsEdt *edt) {
  ARTS_INFO("EDT[Id:%lu, Guid:%lu] is ready", edt->arts_id, edt->currentEdt);
  acquireDbs(edt);
  if (artsAtomicSub(&edt->depcNeeded, 1U) == 0) {
    INCREMENT_NUM_EDTS_ACQUIRED_BY(1);
    incrementQueueEpoch(edt->epochGuid);
    globalShutdownGuidIncQueue();
#ifdef USE_GPU
    if (artsNodeInfo.gpu &&
        (!artsThreadInfo.myDeque || !artsThreadInfo.myGpuDeque))
      artsStoreNewEdts(edt);
    else
#endif
    {
      if (edt->header.type == ARTS_EDT)
        artsDequePushFront(artsThreadInfo.myDeque, edt, 0);
      else if (edt->header.type == ARTS_GPU_EDT)
        artsDequePushFront(artsThreadInfo.myGpuDeque, edt, 0);
    }
    artsMetricsTriggerEvent(artsEdtQueue, artsThread, 1);
  }
}

void artsRunEdt(struct artsEdt *edt) {
  uint32_t depc = edt->depc;
  artsEdtDep_t *depv = (artsEdtDep_t *)(((uint64_t *)(edt + 1)) + edt->paramc);

  artsEdt_t func = edt->funcPtr;
  uint32_t paramc = edt->paramc;
  uint64_t *paramv = (uint64_t *)(edt + 1);

  ARTS_INFO("Running EDT[Id:%lu, Guid:%lu, Deps: %u, Params: %u, "
            "DepvPtr: %p]",
            edt->arts_id, edt->currentEdt, depc, paramc, depv);
  prepDbs(depc, depv, false);

  artsSetThreadLocalEdtInfo(edt);

  EDT_RUNNING_TIME_START();
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  func(paramc, paramv, depc, depv);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  EDT_RUNNING_TIME_STOP();

  // Record arts_id metrics via counter infrastructure
  uint64_t exec_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000ULL +
                     (end_time.tv_nsec - start_time.tv_nsec);
  artsCounterRecordArtsIdEdt(edt->arts_id, exec_ns, 0);

  INCREMENT_NUM_EDTS_FINISHED_BY(1);

  artsMetricsTriggerEvent(artsEdtThroughput, artsThread, 1);

  artsUnsetThreadLocalEdtInfo();

  // This is for a synchronous path
  if (edt->outputBuffer != NULL_GUID)
    artsSetBuffer(edt->outputBuffer, artsCalloc(1, sizeof(unsigned int)),
                  sizeof(unsigned int));

  ARTS_INFO("EDT[Id:%lu, Guid:%lu] Finished", edt->arts_id, edt->currentEdt);
  releaseDbs(depc, depv, false);
  artsEdtDelete(edt);
  // This is for debugging purposes
  decOustandingEdts(1);
}

inline struct artsEdt *artsRuntimeStealFromNetwork() {
  struct artsEdt *edt = NULL;
  if (artsGlobalRankCount > 1) {
    unsigned int index = artsThreadInfo.threadId;
    for (unsigned int i = 0; i < artsNodeInfo.receiverThreadCount; i++) {
      index = (index + 1) % artsNodeInfo.receiverThreadCount;
      if ((edt = (struct artsEdt *)artsDequePopBack(
               artsNodeInfo.receiverDeque[index])) != NULL)
        break;
    }
  }
  return edt;
}

inline struct artsEdt *artsRuntimeStealFromWorker() {
  struct artsEdt *edt = NULL;
  if (artsNodeInfo.totalThreadCount > 1) {
    long unsigned int stealLoc;
    do {
      stealLoc = jrand48(artsThreadInfo.drand_buf);
      stealLoc = stealLoc % artsNodeInfo.totalThreadCount;
    } while (stealLoc == artsThreadInfo.threadId);
    edt = (struct artsEdt *)artsDequePopBack(artsNodeInfo.deque[stealLoc]);
  }
  return edt;
}

bool artsNetworkFirstSchedulerLoop() {
  struct artsEdt *edtFound;
  if (!(edtFound = artsRuntimeStealFromNetwork())) {
    if (!(edtFound = (struct artsEdt *)artsDequePopFront(
              artsThreadInfo.myNodeDeque))) {
      if (!(edtFound =
                (struct artsEdt *)artsDequePopFront(artsThreadInfo.myDeque)))
        edtFound = artsRuntimeStealFromWorker();
    }
  }
  if (edtFound) {
    artsRunEdt(edtFound);
    return true;
  }
  return false;
}

bool artsNetworkBeforeStealSchedulerLoop() {
  struct artsEdt *edtFound;
  if (!(edtFound =
            (struct artsEdt *)artsDequePopFront(artsThreadInfo.myNodeDeque))) {
    if (!(edtFound =
              (struct artsEdt *)artsDequePopFront(artsThreadInfo.myDeque))) {
      if (!(edtFound = artsRuntimeStealFromNetwork()))
        edtFound = artsRuntimeStealFromWorker();
    }
  }

  if (edtFound) {
    artsRunEdt(edtFound);
    return true;
  }
  return false;
}

struct artsEdt *artsFindEdt() {
  struct artsEdt *edtFound = NULL;
  if (!(edtFound =
            (struct artsEdt *)artsDequePopFront(artsThreadInfo.myDeque))) {
    if (!edtFound)
      if (!(edtFound = artsRuntimeStealFromWorker()))
        edtFound = artsRuntimeStealFromNetwork();

    if (edtFound)
      artsMetricsTriggerEvent(artsEdtSteal, artsThread, 1);
  }
  return edtFound;
}

bool artsDefaultSchedulerLoop() {
  struct artsEdt *edtFound = NULL;
  if (!(edtFound =
            (struct artsEdt *)artsDequePopFront(artsThreadInfo.myDeque))) {
    if (!edtFound)
      if (!(edtFound = artsRuntimeStealFromWorker()))
        edtFound = artsRuntimeStealFromNetwork();

    if (edtFound)
      artsMetricsTriggerEvent(artsEdtSteal, artsThread, 1);
  }

  if (edtFound) {
    artsRunEdt(edtFound);
    // artsWakeUpContext();
    return true;
  }
  checkOutstandingEdts(10000000);
  artsNextContext();
  // artsTMTSchedulerYield();
  //        usleep(1);
  return false;
}

int artsRuntimeLoop() {
  if (artsThreadInfo.networkReceive) {
    while (artsThreadInfo.alive) {
      artsServerTryToReceive(&artsNodeInfo.buf, &artsNodeInfo.packetSize,
                             &artsNodeInfo.stealRequestLock);
    }
  } else if (artsThreadInfo.networkSend) {
    while (artsThreadInfo.alive) {
      if (artsNodeInfo.shutdownStarted &&
          artsNodeInfo.shutdownTimeout > artsGetTimeStamp())
        artsRuntimeStop();
      else
        artsRemoteAsyncSend();
    }
  } else if (artsThreadInfo.worker) {
    while (artsThreadInfo.alive) {
      artsNodeInfo.scheduler();
    }
  }
  return 0;
}
