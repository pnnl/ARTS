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

//Some help https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
//and https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
//Once this *class* works we will put a stream(s) in create a thread local
//stream.  Then we will push stuff!
#include "artsGpuRuntime.h"
#include "artsGpuStream.h"
#include "artsGpuStreamBuffer.h"
#include "artsEdtFunctions.h"
#include "artsDbFunctions.h"
#include "artsRuntime.h"
#include "artsGlobals.h"
#include "artsDeque.h"
#include "artsOutOfOrder.h"
#include "artsDebug.h"

#define DPRINTF(...)
// #define DPRINTF(...) PRINTF(__VA_ARGS__)

#define CHECKCORRECT(x) {                                   \
  cudaError_t err;                                          \
  if( (err = (x)) != cudaSuccess )                          \
    PRINTF("FAILED %s: %s\n", #x, cudaGetErrorString(err)); \
}

void * artsCudaMallocHost(unsigned int size)
{
    void * ptr = NULL;
    CHECKCORRECT(cudaMallocHost(&ptr, size));
    // ptr = artsCalloc(size);
    if(!ptr)
    {
        PRINTF("artsCudaMallocHost failed\n");
        artsDebugPrintStack();
        exit(1);
    }
    return ptr;
}

void artsCudaFreeHost(void * ptr)
{
    if(ptr)
        CHECKCORRECT(cudaFreeHost(ptr));
    // artsFree(ptr);
}

void * artsCudaMalloc(unsigned int size)
{
    void * ptr = NULL;
    CHECKCORRECT(cudaMalloc(&ptr, size));
    if(!ptr)
    {
        PRINTF("artsCudaMalloc failed\n");
        artsDebugPrintStack();
        exit(1);
    }
    return ptr;
}

void artsCudaFree(void * ptr)
{
    if(ptr)
        CHECKCORRECT(cudaFree(ptr));
}

dim3 * artsGetGpuGrid()
{
    return artsLocalGrid;
}

dim3 * artsGetGpuBlock()
{
    return artsLocalBlock;
}

cudaStream_t * artsGetGpuStream()
{
    return artsLocalStream;
}

int artsGetGpuId()
{
    return artsLocalGpuId;
}

unsigned int artsGetNumGpus()
{
    return artsNodeInfo.gpu;
}

artsGuid_t internalEdtCreateGpu(artsEdt_t funcPtr, artsGuid_t * guid, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, 
    artsGuid_t endGuid, uint32_t slot, artsGuid_t dataGuid, bool hasDepv, bool passThrough, bool lib)
{
//    ARTSEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int depSpace = (hasDepv) ? depc * sizeof(artsEdtDep_t) : 0;
    unsigned int edtSpace = sizeof(artsGpuEdt_t) + paramc * sizeof(uint64_t) + depSpace;

    artsGpuEdt_t * edt = (artsGpuEdt_t *) artsMalloc(edtSpace);
    edt->wrapperEdt.invalidateCount = 1;
    edt->grid = grid;
    edt->block = block;
    edt->endGuid = endGuid;
    edt->slot = slot;
    edt->dataGuid = dataGuid;
    edt->passthrough = passThrough;
    edt->lib = lib;
    
    artsEdtCreateInternal((struct artsEdt *) edt, ARTS_GPU_EDT, guid, route, artsThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, true, NULL_GUID, hasDepv);
//    ARTSEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return *guid;
}

artsGuid_t artsEdtCreateGpuDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, artsGuid_t dataGuid, bool hasDepv)
{
    artsGuid_t guid = NULL_GUID;
    return internalEdtCreateGpu(funcPtr, &guid, route, paramc, paramv, depc, grid, block, endGuid, slot, dataGuid, hasDepv, false, false);
}

artsGuid_t artsEdtCreateGpuPTDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, unsigned int passSlot, bool hasDepv)
{
    artsGuid_t guid = NULL_GUID;
    return internalEdtCreateGpu(funcPtr, &guid, route, paramc, paramv, depc, grid, block, endGuid, slot, (artsGuid_t) passSlot, hasDepv, true, false);
}

artsGuid_t artsEdtCreateGpu(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, artsGuid_t dataGuid)
{
    return artsEdtCreateGpuDep(funcPtr, route, paramc, paramv, depc, grid, block, endGuid, slot, dataGuid, true);
}

artsGuid_t artsEdtCreateGpuPT(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, unsigned int passSlot)
{
    return artsEdtCreateGpuPTDep(funcPtr, route, paramc, paramv, depc, grid, block, endGuid, slot, passSlot, true);
}

artsGuid_t artsEdtCreateGpuPTWithGuid(artsEdt_t funcPtr, artsGuid_t guid, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, unsigned int passSlot)
{
    return internalEdtCreateGpu(funcPtr, &guid, artsGuidGetRank(guid), paramc, paramv, depc, grid, block, endGuid, slot, (artsGuid_t) passSlot, true, true, false);
}

artsGuid_t artsEdtCreateGpuLib(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block)
{
    artsGuid_t guid = NULL_GUID;
    return internalEdtCreateGpu(funcPtr, &guid, route, paramc, paramv, depc, grid, block, NULL_GUID, 0, NULL_GUID, true, false, true);
}

void artsRunGpu(void *edtPacket, artsGpu_t * artsGpu)
{
    artsGpuEdt_t * edt    = (artsGpuEdt_t *) edtPacket;
    artsEdt_t      func   = edt->wrapperEdt.funcPtr;
    uint32_t       paramc = edt->wrapperEdt.paramc;
    uint32_t       depc   = edt->wrapperEdt.depc;
    uint64_t     * paramv = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv   = (artsEdtDep_t *)(paramv + paramc);

    int savedDevice;
    cudaGetDevice(&savedDevice);
    CHECKCORRECT(cudaSetDevice(artsGpu->device));

    if(artsNodeInfo.runGpuGcPreEdt)
    {
        DPRINTF("Running Pre Edt GPU GC: %u\n", artsGpu->device);
        uint64_t freeMemSize = artsGpuCleanUpRouteTable((unsigned int) -1, artsNodeInfo.deleteZerosGpuGc, (unsigned int) artsGpu->device);
        artsAtomicAddU64(&artsGpu->availGlobalMem, freeMemSize);
        artsAtomicAddU64(&freeBytes, freeMemSize);
    }

    artsAtomicAdd(&artsGpu->runningEdts, 1U);

    prepDbs(depc, depv);
    artsScheduleToGpu(func, paramc, paramv, depc, depv, edtPacket, artsGpu);

    CHECKCORRECT(cudaSetDevice(savedDevice));
}

void artsGpuHostWrapUp(void *edtPacket, artsGuid_t toSignal, uint32_t slot, artsGuid_t dataGuid)
{
    artsGpuEdt_t * edt    = (artsGpuEdt_t *)edtPacket;
    uint32_t       paramc = edt->wrapperEdt.paramc;
    uint32_t       depc   = edt->wrapperEdt.depc;
    uint64_t     * paramv = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv   = (artsEdtDep_t *)(paramv + paramc);

    if(edt->lib)
    {
        edt->wrapperEdt.invalidateCount = 0;
        artsRouteTableFireOO(edt->wrapperEdt.currentEdt, artsOutOfOrderHandler);
    }

    DPRINTF("TO SIGNAL: %lu -> %lu slot: %u\n", toSignal, dataGuid, slot);
    //Signal next
    if(toSignal)
    {
        if (edt->passthrough)
            artsSignalEdt(toSignal, slot, depv[dataGuid].guid);
        else
        {
            artsType_t mode = artsGuidGetType(toSignal);
            if(mode == ARTS_EDT || mode == ARTS_GPU_EDT)
                artsSignalEdt(toSignal, slot, dataGuid);
            if(mode == ARTS_EVENT)
                artsEventSatisfySlot(toSignal, dataGuid, slot);
            if(mode == ARTS_BUFFER) //This is for us to be able to block in a host edt
                artsSetBuffer(toSignal, 0, 0);
        }
    }

    releaseDbs(depc, depv);
    artsEdtDelete((struct artsEdt *)edtPacket);

}

struct artsEdt * artsRuntimeStealGpuTask()
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
        edt = (struct artsEdt *) artsDequePopBack(artsNodeInfo.gpuDeque[stealLoc]);
    }
    return edt;
}

bool artsGpuSchedulerLoop()
{
    artsGpu_t * artsGpu = NULL;
    artsHandleNewEdts();

    struct artsEdt * edtFound = (struct artsEdt *) NULL;
    if(!(edtFound = (struct artsEdt *)artsDequePopFront(artsThreadInfo.myGpuDeque)))
    {
        if(!edtFound)
            edtFound = artsRuntimeStealGpuTask();
    }

    bool ranGpuEdt = false;
    if(edtFound)
    {
        artsGpu = artsFindGpu(edtFound);
        if (artsGpu)
        {
            artsRunGpu(edtFound, artsGpu);
            ranGpuEdt = true;
        }
        else
            artsDequePushFront(artsThreadInfo.myGpuDeque, edtFound, 0);
    }

    if(!ranGpuEdt)
        checkStreams(artsNodeInfo.gpuBuffOn);

    bool ranCpuEdt = artsDefaultSchedulerLoop();
    if(artsNodeInfo.runGpuGcIdle && !ranGpuEdt && !ranCpuEdt)
    {
        long unsigned int gpuId = jrand48(artsThreadInfo.drand_buf);
        gpuId = gpuId % artsNodeInfo.gpu;
        artsGpu = &artsGpus[gpuId];
        DPRINTF("Running Idle GPU GC: %u\n", gpuId);
        int savedDevice;
        cudaGetDevice(&savedDevice);
        CHECKCORRECT(cudaSetDevice(artsGpu->device));

        uint64_t freeMemSize = artsGpuCleanUpRouteTable((unsigned int) -1, artsNodeInfo.deleteZerosGpuGc, (unsigned int) artsGpu->device);
        artsAtomicAddU64(&artsGpu->availGlobalMem, freeMemSize);
        artsAtomicAddU64(&freeBytes, freeMemSize);

        CHECKCORRECT(cudaSetDevice(savedDevice));
    }

    return ranCpuEdt;
}

void artsPutInDbFromGpu(void * ptr, artsGuid_t dbGuid, unsigned int offset, unsigned int size, bool freeData)
{
    unsigned int rank = artsGuidGetRank(dbGuid);
    if(rank==artsGlobalRankId)
    {
        struct artsDb * db = (struct artsDb*) artsRouteTableLookupItem(dbGuid);
        if(db)
        {
            void * data = (void*)(((char*) (db+1)) + offset);
            // memcpy(data, ptr, size);
            CHECKCORRECT(cudaMemcpyAsync(data, ptr, size, cudaMemcpyDeviceToHost, *artsLocalStream));

        }
        else
        {
            void * cpyPtr = artsMalloc(size);
            // memcpy(cpyPtr, ptr, size);
            CHECKCORRECT(cudaMemcpyAsync(cpyPtr, ptr, size, cudaMemcpyDeviceToHost, *artsLocalStream));
            artsOutOfOrderPutInDb(cpyPtr, NULL_GUID, dbGuid, 0, offset, size, NULL_GUID);
        }
        if(freeData)
            artsGpuRouteTableAddItemToDeleteRace(ptr, 0, dbGuid, artsLocalGpuId);
    }
}
