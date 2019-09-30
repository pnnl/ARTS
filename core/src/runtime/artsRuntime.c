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
#include "artsRuntime.h"
#include "artsAtomics.h"
#include "artsGlobals.h"
#include "artsDeque.h"
#include "artsGuid.h"
#include "artsRemote.h"
#include "artsRemoteFunctions.h"
#include "artsOutOfOrder.h"
#include "artsAbstractMachineModel.h"
#include "artsRouteTable.h"
#include "artsEdtFunctions.h"
#include "artsEventFunctions.h"
#include "artsDbFunctions.h"
#include "artsTerminationDetection.h"
#include "artsThreads.h"
#include "artsArrayList.h"
#include "artsDebug.h"
#include "artsCounter.h"
#include "artsIntrospection.h"
#include "artsTMT.h"

#ifdef USE_GPU
#include "artsGpuRuntime.h"
#include "artsGpuStream.h"
#include "artsGpuRouteTable.h"
#endif

#define DPRINTF( ... )
#define PACKET_SIZE 4096
#define NETWORK_BACKOFF_INCREMENT 0

extern unsigned int numNumaDomains;
extern int mainArgc;
extern char **mainArgv;
extern void initPerNode(unsigned int nodeId, int argc, char** argv) __attribute__((weak));
extern void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char**argv) __attribute__((weak));
extern void artsMain(int argc, char** argv) __attribute__((weak));

struct artsRuntimeShared artsNodeInfo;
__thread struct artsRuntimePrivate artsThreadInfo;

typedef bool (*scheduler_t)();
#ifdef USE_GPU
scheduler_t schedulerLoop[] = 
    {
        artsDefaultSchedulerLoop, 
        artsNetworkBeforeStealSchedulerLoop, 
        artsNetworkFirstSchedulerLoop, 
        artsGpuSchedulerLoop
    };
#else
scheduler_t schedulerLoop[] = 
    {
        artsDefaultSchedulerLoop, 
        artsNetworkBeforeStealSchedulerLoop, 
        artsNetworkFirstSchedulerLoop
    };
#endif

void artsMainEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsMain(mainArgc, mainArgv);
}

void artsRuntimeNodeInit(unsigned int workerThreads, unsigned int receivingThreads, unsigned int senderThreads, unsigned int receiverThreads, unsigned int totalThreads, bool remoteStealingOn, struct artsConfig * config)
{
    artsThreadSetOsThreadCount(config->osThreadCount);
    artsNodeInfo.scheduler = schedulerLoop[config->scheduler];
    artsNodeInfo.deque = (struct artsDeque**) artsMalloc(sizeof(struct artsDeque*)*totalThreads);
    artsNodeInfo.receiverDeque = (struct artsDeque**) artsMalloc(sizeof(struct artsDeque*)*receiverThreads);
    artsNodeInfo.gpuDeque = (struct artsDeque**) artsMalloc(sizeof(struct artsDeque*)*totalThreads);
    artsNodeInfo.routeTable = (artsRouteTable_t**) artsCalloc(sizeof(artsRouteTable_t*)*totalThreads);
    artsNodeInfo.gpuRouteTable = (artsRouteTable_t**) artsCalloc(sizeof(artsRouteTable_t*)*config->gpu);
    artsNodeInfo.remoteRouteTable = artsNewRouteTable(config->routeTableEntries, config->routeTableSize);
    artsNodeInfo.localSpin = (volatile bool**) artsCalloc(sizeof(bool*)*totalThreads);
    artsNodeInfo.memoryMoves = (unsigned int**) artsCalloc(sizeof(unsigned int*)*totalThreads);
    artsNodeInfo.atomicWaits = (struct atomicCreateBarrierInfo **) artsCalloc(sizeof(struct atomicCreateBarrierInfo*)*totalThreads);
    artsNodeInfo.workerThreadCount      = workerThreads;
    artsNodeInfo.senderThreadCount      = senderThreads;
    artsNodeInfo.receiverThreadCount    = receiverThreads;
    artsNodeInfo.totalThreadCount       = totalThreads;
    artsNodeInfo.readyToPush            = totalThreads;
    artsNodeInfo.readyToParallelStart   = totalThreads;
    artsNodeInfo.readyToInspect         = totalThreads;
    artsNodeInfo.readyToExecute         = totalThreads;
    artsNodeInfo.readyToClean           = totalThreads;
    artsNodeInfo.sendLock = 0U;
    artsNodeInfo.recvLock = 0U;
    artsNodeInfo.shutdownCount = artsGlobalRankCount-1;
    artsNodeInfo.shutdownStarted=0;
    artsNodeInfo.readyToShutdown = artsGlobalRankCount-1;
    artsNodeInfo.stealRequestLock = !remoteStealingOn;
    artsNodeInfo.buf = artsMalloc( PACKET_SIZE );
    artsNodeInfo.packetSize = PACKET_SIZE;
    artsNodeInfo.printNodeStats = config->printNodeStats;
    artsNodeInfo.shutdownEpoch = (config->shutdownEpoch) ? 1 : NULL_GUID ;
    artsNodeInfo.shadLoopStride = config->shadLoopStride;
    artsNodeInfo.tMT = config->tMT;
    artsNodeInfo.gpu = config->gpu;
    artsNodeInfo.gpuRouteTableSize = config->gpuRouteTableSize;
    artsNodeInfo.gpuRouteTableEntries = config->gpuRouteTableEntries;
    artsNodeInfo.gpuLocality = config->gpuLocality;
    artsNodeInfo.gpuFit = config->gpuFit;
    artsNodeInfo.gpuMaxEdts = config->gpuMaxEdts;
    artsNodeInfo.gpuMaxMemory = config->gpuMaxMemory;
    artsNodeInfo.gpuP2P = config->gpuP2P;
    artsNodeInfo.gpuBuffOn = config->gpuBuffOn;
    artsNodeInfo.freeDbAfterGpuRun = config->freeDbAfterGpuRun;
    artsNodeInfo.runGpuGcIdle = config->runGpuGcIdle;
    artsNodeInfo.runGpuGcPreEdt = config->runGpuGcPreEdt;
    artsNodeInfo.deleteZerosGpuGc = config->deleteZerosGpuGc;
    artsNodeInfo.pinThreads = config->pinThreads;
    artsNodeInfo.keys = artsCalloc(sizeof(uint64_t*) * totalThreads);
    artsNodeInfo.globalGuidThreadId = artsCalloc(sizeof(uint64_t) * totalThreads);
    artsTMTNodeInit(workerThreads);
    artsInitIntrospector(config);
#ifdef USE_GPU
    if(artsNodeInfo.gpu) // TODO: Multi-Node init
        artsNodeInitGpus();
#endif
}

void artsRuntimeGlobalCleanup()
{
    artsIntrospectivePrintTotals(artsGlobalRankId);
    artsCleanUpDbs();
    artsFree(artsNodeInfo.deque);
    artsFree(artsNodeInfo.gpuDeque);
    artsFree((void *)artsNodeInfo.localSpin);
    artsFree(artsNodeInfo.memoryMoves);
    artsFree(artsNodeInfo.atomicWaits);
#ifdef USE_GPU
    if(artsNodeInfo.gpu)
        artsCleanupGpus();
#endif
}

void artsThreadZeroNodeStart()
{
    artsStartInspector(1);

    setGlobalGuidOn();
    createShutdownEpoch();

    if(initPerNode)
        initPerNode(artsGlobalRankId, mainArgc, mainArgv);
    setGuidGeneratorAfterParallelStart();

    artsStartInspector(2);
    ARTSSTARTCOUNTING(2);
    artsAtomicSub(&artsNodeInfo.readyToParallelStart, 1U);
    while(artsNodeInfo.readyToParallelStart){ }
    if(initPerWorker && artsThreadInfo.worker)
        initPerWorker(artsGlobalRankId, artsThreadInfo.groupId, mainArgc, mainArgv);

    if(artsMain && !artsGlobalRankId)
        artsEdtCreate(artsMainEdt, 0, 0, NULL, 0);

    artsIncrementFinishedEpochList();

    artsAtomicSub(&artsNodeInfo.readyToInspect, 1U);
    while(artsNodeInfo.readyToInspect){ }
    ARTSSTARTCOUNTING(3);
    artsStartInspector(3);
    artsAtomicSub(&artsNodeInfo.readyToExecute, 1U);
    while(artsNodeInfo.readyToExecute){ }
}

void artsRuntimePrivateInit(struct threadMask * unit, struct artsConfig  * config)
{
    artsNodeInfo.deque[unit->id] = artsThreadInfo.myDeque = artsDequeNew(config->dequeSize);
    artsNodeInfo.gpuDeque[unit->id] = artsThreadInfo.myGpuDeque = (config->gpu) ? artsDequeNew(config->dequeSize) : NULL;
    if(unit->worker)
    {
        artsNodeInfo.routeTable[unit->id] =  artsNewRouteTable(config->routeTableEntries, config->routeTableSize);
#ifdef USE_GPU
    if(config->gpu) // TODO: Multi-Node init
        artsWorkerInitGpus();
#endif
    }

    if(unit->networkSend || unit->networkReceive)
    {
        if(unit->networkSend)
        {
            unsigned int size = artsGlobalRankCount*config->ports / artsNodeInfo.senderThreadCount;
            unsigned int rem = artsGlobalRankCount*config->ports % artsNodeInfo.senderThreadCount;
            unsigned int start;
            if(unit->groupPos < rem)
            {
                start = unit->groupPos*(size+1);
                artsRemotSetThreadOutboundQueues(start, start+size+1);
            }
            else
            {
                start = rem*(size+1) + (unit->groupPos - rem) * size ;
                artsRemotSetThreadOutboundQueues(start, start+size);
            }
        }
        if(unit->networkReceive)
        {
            artsNodeInfo.receiverDeque[unit->groupPos] = artsNodeInfo.deque[unit->id];
            unsigned int size = (artsGlobalRankCount-1)*config->ports / artsNodeInfo.receiverThreadCount;
            unsigned int rem = (artsGlobalRankCount-1)*config->ports % artsNodeInfo.receiverThreadCount;
            unsigned int start;
            if(unit->groupPos < rem)
            {
                start = unit->groupPos*(size+1);
                //PRINTF("%d %d %d %d\n", start, size, unit->groupPos, rem);
                artsRemotSetThreadInboundQueues(start, start+size+1);
            }
            else
            {
                start = rem*(size+1) + (unit->groupPos - rem) * size ;
                //PRINTF("%d %d %d %d\n", start, size, unit->groupPos, rem);
                artsRemotSetThreadInboundQueues(start, start+size);
            }

        }
    }
    artsNodeInfo.localSpin[unit->id] = &artsThreadInfo.alive;
    artsThreadInfo.alive = true;
    artsNodeInfo.memoryMoves[unit->id] = (unsigned int *)&artsThreadInfo.oustandingMemoryMoves;
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
    artsGuidKeyGeneratorInit();
    INITCOUNTERLIST(unit->id, artsGlobalRankId, config->counterFolder, config->counterStartPoint);

    if (artsNodeInfo.tMT) // @awmm
    {
        DPRINTF("tMT: PthreadLayer: preparing aliasing for master thread %d\n", unit->id);
        artsTMTRuntimePrivateInit(unit, &artsThreadInfo);
    }

    artsAtomicSub(&artsNodeInfo.readyToPush, 1U);
    while(artsNodeInfo.readyToPush){  };
    if(unit->id)
    {
        artsAtomicSub(&artsNodeInfo.readyToParallelStart, 1U);
        while(artsNodeInfo.readyToParallelStart){ };

        if(artsThreadInfo.worker)
        {
            if(initPerWorker)
                initPerWorker(artsGlobalRankId, artsThreadInfo.groupId, mainArgc, mainArgv);
            artsIncrementFinishedEpochList();
        }

        artsAtomicSub(&artsNodeInfo.readyToInspect, 1U);
        while(artsNodeInfo.readyToInspect) { };
        artsAtomicSub(&artsNodeInfo.readyToExecute, 1U);
        while(artsNodeInfo.readyToExecute) { };
    }
    artsThreadInfo.drand_buf[0] = 1202107158 + unit->id * 1999;
    artsThreadInfo.drand_buf[1] = 0;
    artsThreadInfo.drand_buf[2] = 0;
}

void artsRuntimePrivateCleanup()
{
    artsTMTRuntimePrivateCleanup();
    artsAtomicSub(&artsNodeInfo.readyToClean, 1U);
    while(artsNodeInfo.readyToClean){ };
    if(artsThreadInfo.myDeque)
        artsDequeDelete(artsThreadInfo.myDeque);
    if(artsThreadInfo.myNodeDeque)
        artsDequeDelete(artsThreadInfo.myNodeDeque);
    if(artsThreadInfo.myGpuDeque)
        artsDequeDelete(artsThreadInfo.myGpuDeque);
#if defined(COUNT) || defined(MODELCOUNT)
    artsWriteCountersToFile(artsThreadInfo.threadId, artsGlobalRankId);
#endif
    artsWriteMetricShotFile(artsThreadInfo.threadId, artsGlobalRankId);
}

void artsRuntimeStop()
{
    unsigned int i;
    for(i=0; i<artsNodeInfo.totalThreadCount; i++)
    {
        while(!artsNodeInfo.localSpin[i]);
        (*artsNodeInfo.localSpin[i]) = false;
    }
    artsTMTRuntimeStop();
    artsStopInspector();
}

void artsHandleRemoteStolenEdt(struct artsEdt *edt)
{
    DPRINTF("push stolen %d\n",artsThreadInfo.coreId);
    incrementQueueEpoch(edt->epochGuid);
    globalShutdownGuidIncQueue();
#ifdef USE_GPU
    if(artsNodeInfo.gpu && (!artsThreadInfo.myDeque || !artsThreadInfo.myGpuDeque))
        artsStoreNewEdts(edt);
    else
#endif
    {
        if(edt->header.type == ARTS_EDT)
            artsDequePushFront(artsThreadInfo.myDeque, edt, 0);
        if(edt->header.type == ARTS_GPU_EDT)
            artsDequePushFront(artsThreadInfo.myGpuDeque, edt, 0);
    }
}

void artsHandleReadyEdt(struct artsEdt * edt)
{
    acquireDbs(edt);
    if(artsAtomicSub(&edt->depcNeeded,1U) == 0)
    {
        incrementQueueEpoch(edt->epochGuid);
        globalShutdownGuidIncQueue();
#ifdef USE_GPU
        if(artsNodeInfo.gpu && (!artsThreadInfo.myDeque || !artsThreadInfo.myGpuDeque))
            artsStoreNewEdts(edt);
        else
#endif
        {
            if(edt->header.type == ARTS_EDT)
            {
                artsDequePushFront(artsThreadInfo.myDeque, edt, 0);
            }
            if(edt->header.type == ARTS_GPU_EDT)
                artsDequePushFront(artsThreadInfo.myGpuDeque, edt, 0);
        }

        artsUpdatePerformanceMetric(artsEdtQueue, artsThread, 1, false);
    }
}

static inline void artsRunEdt(void *edtPacket)
{
    struct artsEdt *edt = edtPacket;
    uint32_t depc = edt->depc;
    artsEdtDep_t * depv = (artsEdtDep_t *)(((uint64_t *)(edt + 1)) + edt->paramc);

    artsEdt_t func = edt->funcPtr;
    uint32_t paramc = edt->paramc;
    uint64_t *paramv = (uint64_t *)(edt + 1);

    prepDbs(depc, depv);

    artsSetThreadLocalEdtInfo(edt);
    ARTSCOUNTERTIMERSTART(edtCounter);

    func(paramc, paramv, depc, depv);

    ARTSCOUNTERTIMERENDINCREMENT(edtCounter);
    artsUpdatePerformanceMetric(artsEdtThroughput, artsThread, 1, false);

    artsUnsetThreadLocalEdtInfo();

    if(edt->outputBuffer != NULL_GUID) //This is for a synchronous path
        artsSetBuffer(edt->outputBuffer, artsCalloc(sizeof(unsigned int)), sizeof(unsigned int));

    releaseDbs(depc, depv);
    artsEdtDelete(edtPacket);
    decOustandingEdts(1); //This is for debugging purposes
}

inline struct artsEdt * artsRuntimeStealFromNetwork()
{
    struct artsEdt *edt = NULL;
    if(artsGlobalRankCount > 1)
    {
        unsigned int index = artsThreadInfo.threadId;
        for (unsigned int i=0; i<artsNodeInfo.receiverThreadCount; i++)
        {
            index = (index + 1) % artsNodeInfo.receiverThreadCount;
            if(edt = artsDequePopBack(artsNodeInfo.receiverDeque[index]))
                break;
        }
    }
    return edt;
}

inline struct artsEdt * artsRuntimeStealFromWorker()
{
    struct artsEdt *edt = NULL;
    if(artsNodeInfo.totalThreadCount > 1)
    {
        long unsigned int stealLoc;
        do
        {
            stealLoc = jrand48(artsThreadInfo.drand_buf);
            stealLoc = stealLoc % artsNodeInfo.totalThreadCount;
        } while(stealLoc == artsThreadInfo.threadId);
        edt = artsDequePopBack(artsNodeInfo.deque[stealLoc]);
    }
    return edt;
}

bool artsNetworkFirstSchedulerLoop()
{
    struct artsEdt *edtFound;
    if(!(edtFound = artsRuntimeStealFromNetwork()))
    {
        if(!(edtFound = artsDequePopFront(artsThreadInfo.myNodeDeque)))
        {
            if(!(edtFound = artsDequePopFront(artsThreadInfo.myDeque)))
                edtFound = artsRuntimeStealFromWorker();
        }
    }
    if(edtFound)
    {
        artsRunEdt(edtFound);
        return true;
    }
    return false;
}

bool artsNetworkBeforeStealSchedulerLoop()
{
    struct artsEdt *edtFound;
    if(!(edtFound = artsDequePopFront(artsThreadInfo.myNodeDeque)))
    {
        if(!(edtFound = artsDequePopFront(artsThreadInfo.myDeque)))
        {
            if(!(edtFound = artsRuntimeStealFromNetwork()))
                edtFound = artsRuntimeStealFromWorker();
        }
    }

    if(edtFound)
    {
        artsRunEdt(edtFound);
        return true;
    }
    return false;
}

bool artsDefaultSchedulerLoop()
{
    struct artsEdt * edtFound = NULL;
    if(!(edtFound = artsDequePopFront(artsThreadInfo.myDeque)))
    {
        if(!edtFound)
            if(!(edtFound = artsRuntimeStealFromWorker()))
                edtFound = artsRuntimeStealFromNetwork();

        if(edtFound)
            artsUpdatePerformanceMetric(artsEdtSteal, artsThread, 1, false);
    }

    if(edtFound)
    {
        artsRunEdt(edtFound);
        artsWakeUpContext();
        return true;
    }
    else
    {
        checkOutstandingEdts(10000000);
        artsNextContext();
//        usleep(1);
    }
    return false;
}

int artsRuntimeLoop()
{
    ARTSCOUNTERTIMERSTART(totalCounter);
    if(artsThreadInfo.networkReceive)
    {
        while(artsThreadInfo.alive)
        {
            artsServerTryToRecieve(&artsNodeInfo.buf, &artsNodeInfo.packetSize, &artsNodeInfo.stealRequestLock);
        }
    }
    else if(artsThreadInfo.networkSend)
    {
        while(artsThreadInfo.alive)
        {
            if(artsNodeInfo.shutdownStarted && artsNodeInfo.shutdownTimeout > artsGetTimeStamp())
                artsRuntimeStop();
            else
                artsRemoteAsyncSend();
        }
    }
    else if(artsThreadInfo.worker)
    {
        while(artsThreadInfo.alive)
        {
            artsNodeInfo.scheduler();
        }
    }
    ARTSCOUNTERTIMERENDINCREMENT(totalCounter);
    return 0;
}
