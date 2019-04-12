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

#include "arts.h"
#include "artsGuid.h"
#include "artsRemote.h"
#include "artsRemoteFunctions.h"
#include "artsGlobals.h"
#include "artsAtomics.h"
#include "artsCounter.h"
#include "artsIntrospection.h"
#include "artsRuntime.h"
#include "artsEdtFunctions.h"
#include "artsOutOfOrder.h"
#include "artsRouteTable.h"
#include "artsDebug.h"
#include "artsTerminationDetection.h"
#include "artsArrayList.h"
#include "artsQueue.h"

artsGuid_t artsEdtCreateShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv)
{
    ARTSEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t) + sizeof(artsGuid_t);
    artsGuid_t guid = NULL_GUID;
    artsEdtCreateInternal(&guid, route, artsThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, 0, false, NULL_GUID, true);
    ARTSEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

artsGuid_t artsActiveMessageShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, void * data, unsigned int size, artsGuid_t epochGuid)
{
    unsigned int rank = route; //route / numNumaDomains;
    unsigned int cluster = 0; //route % numNumaDomains;
    artsGuid_t guid = NULL_GUID;
    bool useEpoch = (epochGuid != NULL_GUID);
    
//    if(route == artsGlobalRankId && artsInspecting())
//    {
//        uint64_t edts = artsGetPerformanceMetricTotal(artsEdtThroughput, artsNode);
//        uint64_t queued = artsGetPerformanceMetricTotal(artsEdtQueue, artsNode);
//        uint64_t totalQueued = (queued > edts) ? queued - edts : 0;
//        double queueRate = artsMetricTest(artsEdtThroughput, artsNode, totalQueued);
//        double doneRate = artsMetricTest(artsEdtThroughput, artsNode, 1);
//        
//        
////        PRINTF("doneRate: %lf queueRate: %lf edts: %lu queued: %lu\n", doneRate, queueRate, edts, queued);
////        if(queueRate != 0 && doneRate != 0)
//        {
//            if( artsNodeInfo.workerThreadCount==1 ||
//                ( !edts && artsNodeInfo.workerThreadCount * 2 > queued) ||
//                ( totalQueued && (queueRate > doneRate) )
//              )
//            {
////                PRINTF("%lf * 1.5 = %lf < %lf Queued: %lu\n", doneRate, doneRate*1.5, queueRate, totalQueued);
//                artsEdtDep_t dep;
//                dep.ptr = data;
//                dep.mode = DB_MODE_PTR;
//                dep.guid = NULL_GUID;
//                
//                ARTSCOUNTERTIMERSTART(edtCounter);
//                
//                artsGuid_t result = funcPtr(paramc, paramv, 1, &dep);
//                
//                ARTSCOUNTERTIMERENDINCREMENT(edtCounter);
//                artsUpdatePerformanceMetric(artsEdtThroughput, artsThread, 1, false);
//                
//                return NULL_GUID;
//            }
//        }
//    }
    
    if(size) {
        unsigned int depSpace = sizeof(artsEdtDep_t);
        unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t) + depSpace;
        artsEdtCreateInternal(&guid, rank, cluster, edtSpace, NULL_GUID, funcPtr, paramc, paramv, 1, useEpoch, epochGuid, true);
        
//        PRINTF("MEMCPY: %u\n", size);
        void * ptr = artsMalloc(size);
        memcpy(ptr, data, size);
        artsSignalEdtPtr(guid, 0, ptr, size);
    }
    else
    {
        unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t);
        artsEdtCreateInternal(&guid, rank, cluster, edtSpace, NULL_GUID, funcPtr, paramc, paramv, 0, useEpoch, epochGuid, false);
    }
    return guid;
}

void artsSynchronousActiveMessageShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, void * data, unsigned int size)
{
    unsigned int rank = route; //route / numNumaDomains;
    unsigned int cluster = 0; //route % numNumaDomains;
    unsigned int waitFlag = 1;
    void * waitPtr = &waitFlag;
    artsGuid_t waitGuid = artsAllocateLocalBuffer((void **)&waitPtr, sizeof(unsigned int), 1, NULL_GUID);
    
    artsGuid_t guid = NULL_GUID;
    if(size) {
        unsigned int depSpace = sizeof(artsEdtDep_t);
        unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t) + depSpace;
        artsEdtCreateInternal(&guid, rank, cluster, edtSpace, waitGuid, funcPtr, paramc, paramv, 1, false, NULL_GUID, true);

        void * ptr = artsMalloc(size);
        memcpy(ptr, data, size);
        artsSignalEdtPtr(guid, 0, ptr, size);
    }
    else
    {
        unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t);
        artsEdtCreateInternal(&guid, rank, cluster, edtSpace, waitGuid, funcPtr, paramc, paramv, 0, false, NULL_GUID, false);
    }
    
    while(waitFlag) {
        artsYield();
    }
}

void artsIncLockShad()
{
    artsThreadInfo.shadLock++;
}

void artsDecLockShad()
{
    artsThreadInfo.shadLock--;
}

void artsCheckLockShad()
{
    if(artsThreadInfo.shadLock)
    {
        PRINTF("ARTS: Cannot perform synchronous call under lock Worker: %u ShadLock: %u\n", artsThreadInfo.groupId, artsThreadInfo.shadLock);
        artsDebugGenerateSegFault();
    }
}

void artsStartIntroShad(unsigned int start)
{
    artsStartInspector(start);
    ARTSSTARTCOUNTING(start);
}

void artsStopIntroShad()
{
    artsStopInspector();
    ARTSCOUNTERSOFF();
}

unsigned int artsGetShadLoopStride()
{
    return artsNodeInfo.shadLoopStride;
}

artsGuid_t artsAllocateLocalBufferShad(void ** buffer, uint32_t * sizeToWrite, artsGuid_t epochGuid)
{
    if(epochGuid)
        incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    
    artsBuffer_t * stub = artsMalloc(sizeof(artsBuffer_t));
    stub->buffer = *buffer;
    stub->sizeToWrite = sizeToWrite;
    stub->size = 0;
    stub->uses = 1;
    stub->epochGuid = epochGuid;
    
    artsGuid_t guid = artsGuidCreateForRank(artsGlobalRankId, ARTS_BUFFER);
    artsRouteTableAddItem(stub, guid, artsGlobalRankId, false);
    return guid;
}

