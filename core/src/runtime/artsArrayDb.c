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

#include "artsArrayDb.h"
#include "artsAtomics.h"
#include "artsGlobals.h"
#include "artsGuid.h"
#include "artsDbFunctions.h"
#include "artsEdtFunctions.h"
#include "artsRemoteFunctions.h"
#include "artsDbFunctions.h"
#include "artsTerminationDetection.h"
#include "artsRouteTable.h"
#include "artsOutOfOrder.h"
#include "artsDebug.h"

unsigned int artsGetSizeArrayDb(artsArrayDb_t * array)
{
    return array->elementsPerBlock * array->numBlocks;
}

void * copyDb(void * ptr, unsigned int size, artsGuid_t guid)
{
    struct artsDb * db = ((struct artsDb *)ptr) - 1;
    struct artsDb * newDb = artsCalloc(size);
    memcpy(newDb, db, size);
    newDb->guid = guid;
    return (void*) (newDb+1);
}

artsArrayDb_t * artsNewArrayDbWithGuid(artsGuid_t guid, unsigned int elementSize, unsigned int numElements)
{
    unsigned int numBlocks = artsGlobalRankCount;
    unsigned int elementsPerBlock = numElements / numBlocks;
    if(!elementsPerBlock)
    {
        elementsPerBlock = 1;
        numBlocks = numElements;
    }
    else if(numElements % numBlocks)
    {
        elementsPerBlock++;
        numElements = elementsPerBlock * numBlocks;
    }
    
    PRINTF("Elements: %u Blocks: %u Element Size: %u\n", numElements, numBlocks, elementSize);
    
    unsigned int allocSize = sizeof(artsArrayDb_t) + elementSize * elementsPerBlock;
    artsArrayDb_t * block = NULL;
    if(numBlocks)
    {
        //We have to manually create the db so it isn't updated before we send it...
        unsigned int dbSize = sizeof(struct artsDb) + allocSize;
        struct artsDb * toSend = artsCalloc(dbSize);
        artsDbCreateInternal(guid, toSend, allocSize, dbSize, ARTS_DB_PIN);
        
        block = (artsArrayDb_t *)(toSend+1);
        block->elementSize = elementSize;
        block->elementsPerBlock = elementsPerBlock;
        block->numBlocks = numBlocks;

        for(unsigned int i=0; i<artsGlobalRankCount; i++)
        {
            if(i!=artsGlobalRankId)
            {
                artsRemoteMemoryMoveNoFree(i, guid, toSend, allocSize + sizeof(struct artsDb), ARTS_REMOTE_DB_MOVE_MSG);
            }
        }
        
        artsDbCreateWithGuidAndData(guid, block, allocSize);
    }
    return block;
}

artsGuid_t artsNewArrayDb(artsArrayDb_t **addr, unsigned int elementSize, unsigned int numElements)
{
    artsGuid_t guid = artsReserveGuidRoute(ARTS_DB_PIN, artsGlobalRankId);
    *addr = artsNewArrayDbWithGuid(guid, elementSize, numElements);
    return guid;
}

artsGuid_t getArrayDbGuid(artsArrayDb_t * array)
{
    struct artsDb * db = ((struct artsDb *)array) - 1;
    return db->guid;
}

unsigned int getOffsetFromIndex(artsArrayDb_t * array, unsigned int index)
{    
    unsigned int base = sizeof(artsArrayDb_t);
    unsigned int local = (index % array->elementsPerBlock) * array->elementSize;
//    PRINTF("array: %p base: %u index: %u elementsPerBlock: %u mod: %u elementSize: %u\n", array, base, index, array->elementsPerBlock, index%array->elementsPerBlock, array->elementSize);
    return base + local;
}

unsigned int getRankFromIndex(artsArrayDb_t * array, unsigned int index)
{
    return index/array->elementsPerBlock;
}

void artsGetFromArrayDb(artsGuid_t edtGuid, unsigned int slot, artsArrayDb_t * array, unsigned int index)
{
    if(index < array->elementsPerBlock*array->numBlocks)
    {
        artsGuid_t guid = getArrayDbGuid(array);
        unsigned int rank = getRankFromIndex(array, index);
        unsigned int offset = getOffsetFromIndex(array, index);
//        PRINTF("Get index: %u rank: %u offset: %u\n", index, rank, offset);
        artsGetFromDbAt(edtGuid, guid, slot, offset, array->elementSize, rank);
    }
    else
    {
        PRINTF("Index >= Array Size: %u >= %u * %u\n", index, array->elementsPerBlock, array->numBlocks);
        artsDebugGenerateSegFault();
    } 
}

void artsPutInArrayDb(void * ptr, artsGuid_t edtGuid, unsigned int slot, artsArrayDb_t * array, unsigned int index)
{
    artsGuid_t guid = getArrayDbGuid(array);
    unsigned int rank = getRankFromIndex(array, index);
    unsigned int offset = getOffsetFromIndex(array, index);
    artsPutInDbAt(ptr, edtGuid, guid, slot, offset, array->elementSize, rank);
}

void artsForEachInArrayDb(artsArrayDb_t * array, artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv)
{
    uint64_t * args = artsMalloc(sizeof(uint64_t) * (paramc+1));
    memcpy(&args[1], paramv, sizeof(uint64_t) * paramc);
    
    unsigned int size = artsGetSizeArrayDb(array);
    for(unsigned int i=0; i<size; i++)
    {
        args[0] = i;
        unsigned int route = getRankFromIndex(array, i);
        artsGuid_t guid = artsEdtCreate(funcPtr, route, paramc+1, args, 1);
        artsGetFromArrayDb(guid, 0, array, i);
    }
}

void artsGatherArrayDb(artsArrayDb_t * array, artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint64_t depc)
{
    unsigned int offset = getOffsetFromIndex(array, 0);
    unsigned int size = array->elementSize * array->elementsPerBlock;
    artsGuid_t arrayGuid = getArrayDbGuid(array);
    
    artsGuid_t guid = artsEdtCreate(funcPtr, route, paramc, paramv, array->numBlocks + depc);
    for(unsigned int i=0; i<array->numBlocks; i++)
    {
        artsGetFromDbAt(guid, arrayGuid, i, offset, size, i);
    }
}

void loopPolicy(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsEdt_t funcPtr = (artsEdt_t)paramv[0];
    unsigned int stride = paramv[1];
    unsigned int end = paramv[2];
    unsigned int start = paramv[3];

    artsArrayDb_t * array = depv[0].ptr;
    unsigned int offset = getOffsetFromIndex(array, start);
    char * raw = depv[0].ptr;
    
    for(unsigned int i=start; i<end; i+=stride)
    {
        paramv[3] = i;
        depv[0].ptr = (void*)(&raw[offset]);
        funcPtr(paramc-3, &paramv[3], 1, depv);
        offset+=array->elementSize;
    }
    depv[0].ptr = (void*)raw;
}

void artsForEachInArrayDbAtData(artsArrayDb_t * array, unsigned int stride, artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv)
{
    unsigned int blockSize = array->elementsPerBlock;
    unsigned int size = artsGetSizeArrayDb(array);
    if(size%stride)
    {
        PRINTF("WARNING: Size is not divisible by stride!");
    }
    artsGuid_t guid = getArrayDbGuid(array);
    uint64_t * args = artsMalloc(sizeof(uint64_t) * (paramc+4));
    if(paramc)
        memcpy(&args[4], paramv, sizeof(uint64_t) * paramc);
    args[0] = (uint64_t)funcPtr;
    args[1] = stride;
    for(unsigned int i=0; i<size; i+=blockSize)
    {
        args[2] = (i+blockSize < size) ? i+blockSize : size;
        args[3] = i;
        artsActiveMessageWithDbAt(loopPolicy, paramc+4, args, 0, guid, getRankFromIndex(array, i));
    }
}

void internalAtomicAddInArrayDb(artsGuid_t dbGuid, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid)
{
    struct artsDb * db = artsRouteTableLookupItem(dbGuid);
    if(db)
    {
        artsArrayDb_t * array = (artsArrayDb_t*)(db+1);
        //Do this so when we increment finished we can check the term status
        incrementQueueEpoch(epochGuid);
        globalShutdownGuidIncQueue();
        unsigned int offset = getOffsetFromIndex(array, index);
        unsigned int * data = (unsigned int*)(((char*) array) + offset);
        unsigned int result = artsAtomicAdd(data, toAdd);
//        PRINTF("index: %u result: %u\n", index, result);
        
        if(edtGuid)
        {
//            PRINTF("Signaling edtGuid: %lu\n", edtGuid);
            artsSignalEdtValue(edtGuid, slot, result);
        }

        incrementFinishedEpoch(epochGuid);
        globalShutdownGuidIncFinished();
    }
    else
    {
        artsOutOfOrderAtomicAddInArrayDb(dbGuid, index, toAdd, edtGuid, slot, epochGuid);
    }
}

void artsAtomicAddInArrayDb(artsArrayDb_t * array, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot)
{
    artsGuid_t dbGuid = getArrayDbGuid(array);
    artsGuid_t epochGuid = artsGetCurrentEpochGuid();
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    unsigned int rank = getRankFromIndex(array, index);
    if(rank==artsGlobalRankId)
        internalAtomicAddInArrayDb(dbGuid, index, toAdd, edtGuid, slot, epochGuid);
    else
        artsRemoteAtomicAddInArrayDb(rank, dbGuid, index, toAdd, edtGuid, slot, epochGuid);
}

void internalAtomicCompareAndSwapInArrayDb(artsGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid)
{
    struct artsDb * db = artsRouteTableLookupItem(dbGuid);
    if(db)
    {
        artsArrayDb_t * array = (artsArrayDb_t*)(db+1);
        //Do this so when we increment finished we can check the term status
        incrementQueueEpoch(epochGuid);
        globalShutdownGuidIncQueue();
        unsigned int offset = getOffsetFromIndex(array, index);
        unsigned int * data = (unsigned int*)(((char*) array) + offset);
        unsigned int result = artsAtomicCswap(data, oldValue, newValue);
//        PRINTF("index: %u result: %u\n", index, result);
        
        if(edtGuid)
        {
//            PRINTF("Signaling edtGuid: %lu\n", edtGuid);
            artsSignalEdtValue(edtGuid, slot, result);
        }

        incrementFinishedEpoch(epochGuid);
        globalShutdownGuidIncFinished();
    }
    else
    {
        artsOutOfOrderAtomicCompareAndSwapInArrayDb(dbGuid, index, oldValue, newValue, edtGuid, slot, epochGuid);
    }
}

void artsAtomicCompareAndSwapInArrayDb(artsArrayDb_t * array, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot)
{
    artsGuid_t dbGuid = getArrayDbGuid(array);
    artsGuid_t epochGuid = artsGetCurrentEpochGuid();
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    unsigned int rank = getRankFromIndex(array, index);
    if(rank==artsGlobalRankId)
        internalAtomicCompareAndSwapInArrayDb(dbGuid, index, oldValue, newValue, edtGuid, slot, epochGuid);
    else
        artsRemoteAtomicCompareAndSwapInArrayDb(rank, dbGuid, index, oldValue, newValue, edtGuid, slot, epochGuid);
}
