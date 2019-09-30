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
#include "artsDbFunctions.h"
#include "artsGpuStream.h"
#include "artsGpuStreamBuffer.h"
#include "artsGpuRuntime.h"
#include "artsGlobals.h"
#include "artsDeque.h"
#include "artsGpuRouteTable.h"
#include "artsDebug.h"
#include "artsEdtFunctions.h"

#define DPRINTF( ... )
// #define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

int random(void * edtPacket);
int allOrNothing(void * edtPacket);
int atleastOne(void * edtPacket);
int firstFit(uint64_t mask, size_t size);
int bestFit(uint64_t mask, size_t size);
int worstFit(uint64_t mask, size_t size);
int roundRobinFit(uint64_t mask, size_t size);
bool tryReserve(int gpu, size_t size);

volatile unsigned int hits = 0;
volatile unsigned int misses = 0;
volatile unsigned int freeBytes = 0;

artsGpu_t * artsGpus;

typedef int (*locality_t) (void * edt);

locality_t localityScheme[] = {
    random,
    allOrNothing,
    atleastOne
};

locality_t locality; // Locality function ptr

typedef int (*fit_t) (uint64_t mask, size_t size);

fit_t fitScheme[] = {
    firstFit,
    bestFit,
    worstFit,
    roundRobinFit
};

fit_t fit; // Fit function ptr

bool P2P; // Enable DB access across devices

__thread volatile unsigned int * newEdtLock = 0;
__thread artsArrayList * newEdts = NULL;


//These are for the library version of GPU EDTs
//The user can query to get these values
//We still want to collect them for scheduling purposes
__thread dim3 * artsLocalGrid;
__thread dim3 * artsLocalBlock;
__thread cudaStream_t * artsLocalStream;
__thread int artsLocalGpuId;

#ifdef __cplusplus
extern "C" 
{
#endif
    extern void initPerGpu(int devId, cudaStream_t * stream) __attribute__((weak));
    extern void cleanPerGpu(int devId, cudaStream_t * stream) __attribute__((weak));
#ifdef __cplusplus
}
#endif

void artsNodeInitGpus()
{
    int numAvailGpus = 0;
    locality = localityScheme[artsNodeInfo.gpuLocality];
    fit = fitScheme[artsNodeInfo.gpuFit];
    P2P = artsNodeInfo.gpuP2P;
    CHECKCORRECT(cudaGetDeviceCount(&numAvailGpus));
    if(numAvailGpus < artsNodeInfo.gpu)
    {
        PRINTF("Requested %d gpus but only %d available\n", numAvailGpus, artsNodeInfo.gpu);
        artsNodeInfo.gpu = numAvailGpus;
    }

    DPRINTF("gpuRouteTableSize: %u gpuRouteTableEntries: %u freeDbAfterGpuRun: %u runGpuGcIdle: %u runGpuGcPreEdt: %u deleteZerosGpuGc: %u\n",
        artsNodeInfo.gpuRouteTableSize, artsNodeInfo.gpuRouteTableEntries, 
        artsNodeInfo.freeDbAfterGpuRun, artsNodeInfo.runGpuGcIdle, 
        artsNodeInfo.runGpuGcPreEdt, artsNodeInfo.deleteZerosGpuGc);

    DPRINTF("NUM DEV: %d\n", artsNodeInfo.gpu);
    artsGpus = (artsGpu_t *) artsCalloc(sizeof(artsGpu_t) * artsNodeInfo.gpu);

    int savedDevice;
    CHECKCORRECT(cudaGetDevice(&savedDevice));

    // Initialize artsGpu with 1 stream/GPU
    for (int i=0; i<artsNodeInfo.gpu; ++i)
    {
        artsGpus[i].device = i;
        DPRINTF("Setting %d\n", i);
        CHECKCORRECT(cudaSetDevice(i));
        CHECKCORRECT(cudaStreamCreate(&artsGpus[i].stream)); // Make it scalable
        artsNodeInfo.gpuRouteTable[i] = artsGpuNewRouteTable(artsNodeInfo.gpuRouteTableEntries, artsNodeInfo.gpuRouteTableSize);
        CHECKCORRECT(cudaMemGetInfo((size_t*)&artsGpus[i].availGlobalMem, (size_t*)&artsGpus[i].totalGlobalMem));
        CHECKCORRECT(cudaGetDeviceProperties(&artsGpus[i].prop, artsGpus[i].device));
        if (artsGpus[i].availGlobalMem > artsNodeInfo.gpuMaxMemory)
            artsGpus[i].availGlobalMem = artsNodeInfo.gpuMaxMemory;
        if(initPerGpu)
            initPerGpu(i, &artsGpus[i].stream);
    }
    CHECKCORRECT(cudaSetDevice(savedDevice));
}

void artsWorkerInitGpus()
{
    newEdtLock = (unsigned int*) artsCalloc(sizeof(unsigned int));
    newEdts = artsNewArrayList(sizeof(void*), 32);
}

void artsStoreNewEdts(void * edt)
{
    artsLock(newEdtLock);
    artsPushToArrayList(newEdts, &edt);
    artsUnlock(newEdtLock);
}

void artsHandleNewEdts()
{
    artsLock(newEdtLock);
    uint64_t size = artsLengthArrayList(newEdts);
    if(size)
    {
        for(uint64_t i=0; i<size; i++)
        {
            struct artsEdt ** edt = (struct artsEdt**) artsGetFromArrayList(newEdts, i);
            if((*edt)->header.type == ARTS_EDT)
                artsDequePushFront(artsThreadInfo.myDeque, (*edt), 0);
            if((*edt)->header.type == ARTS_GPU_EDT)
                artsDequePushFront(artsThreadInfo.myGpuDeque, (*edt), 0);
        }
        artsResetArrayList(newEdts);
    }    
    artsUnlock(newEdtLock);
}

void artsCleanupGpus()
{
    unsigned int freedSize = 0;
    int savedDevice;
    cudaGetDevice(&savedDevice);
    for (int i=0; i<artsNodeInfo.gpu; i++)
    {
        CHECKCORRECT(cudaSetDevice(artsGpus[i].device));
        if(cleanPerGpu)
            cleanPerGpu(i, &artsGpus[i].stream);
        freedSize += artsGpuFreeAll(artsGpus[i].device);
        CHECKCORRECT(cudaStreamSynchronize(artsGpus[i].stream));
        CHECKCORRECT(cudaStreamDestroy(artsGpus[i].stream));
    }
    CHECKCORRECT(cudaSetDevice(savedDevice));
    PRINTF("Occupancy :\n");
    for (int i=0; i<artsGetNumGpus(); ++i)
        PRINTF("\tGPU[%d] = %f\n", i, artsGpus[i].occupancy);
    PRINTF("HITS: %u MISSES: %u FREED BYTES: %u BYTES FREED ON EXIT %u\n", hits, misses, freeBytes, freedSize);
    PRINTF("HIT RATIO: %lf\n", (double)hits/(double)(hits+misses));
}

void CUDART_CB artsWrapUp(cudaStream_t stream, cudaError_t status, void * data)
{
    artsGpuCleanUp_t * gc = (artsGpuCleanUp_t*) data;
    
    artsGpu_t * artsGpu = &artsGpus[gc->gpuId];
    artsAtomicSub(&artsGpu->availableEdtSlots, 1U);
    artsAtomicSub(&artsGpu->runningEdts, 1U);

    //Shouldn't have to touch newly ready edts regardless of streams and devices
    artsGpuEdt_t * edt      = (artsGpuEdt_t *) gc->edt;
    uint32_t       paramc   = edt->wrapperEdt.paramc;
    uint32_t       depc     = edt->wrapperEdt.depc;
    uint64_t     * paramv   = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv     = (artsEdtDep_t *)(paramv + paramc);

    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr)
        {
            //True says to mark it for deletion... Change this to false to further delay delete!
            artsGpuRouteTableReturnDb(depv[i].guid, artsNodeInfo.freeDbAfterGpuRun, gc->gpuId);
            DPRINTF("Returning Db: %lu id: %d\n", depv[i].guid, gc->gpuId);
        }
    }

    //Definitely mark the dev closure to be deleted as there is no reuse!
    artsGpuRouteTableReturnDb(edt->wrapperEdt.currentEdt, true, gc->gpuId);
    newEdtLock = gc->newEdtLock;
    newEdts    = gc->newEdts;
    artsGpuHostWrapUp(gc->edt, edt->endGuid, edt->slot, edt->dataGuid);
    DPRINTF("FINISHED GPU CALLS %s\n", cudaGetErrorString(status));
}

void artsScheduleToGpuInternal(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, dim3 grid, dim3 block, void * edtPtr, artsGpu_t * artsGpu)
{
//    For now this should push the following into the stream:
//    1. Copy data from host to device
//    2. Push kernel
//    3. Copy data from device to host
//    4. Call host callback function artsGpuHostWrapUp

    static volatile unsigned int Gpulock;

    void * devClosure  = NULL;
    void * hostClosure = NULL;

    uint64_t         * devParamv  = NULL;
    artsEdtDep_t     * devDepv    = NULL;

    artsGpuCleanUp_t * hostGCPtr = NULL;
    uint64_t         * hostParamv = NULL;
    artsEdtDep_t     * hostDepv   = NULL;

    DPRINTF("Paramc: %u Depc: %u edt: %p\n", paramc, depc, edtPtr);

    //Get size of closure
    size_t devClosureSize = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc;
    size_t hostClosureSize = devClosureSize + sizeof(artsGpuCleanUp_t);
    DPRINTF("devClosureSize: %u hostClosureSize: %u\n", devClosureSize, hostClosureSize);

    //Allocate Closure for GPU
    if(devClosureSize)
    {
        devClosure = artsCudaMalloc(devClosureSize);
        devParamv = (uint64_t*) devClosure;
        devDepv = (artsEdtDep_t *)(devParamv + paramc);
        DPRINTF("Allocated dev closure\n");
    }

    if(hostClosureSize)
    {
        //Allocate closure for host
        hostClosure = artsCudaMallocHost(hostClosureSize);
        hostGCPtr = (artsGpuCleanUp_t *) hostClosure;
        hostParamv = (uint64_t*)(hostGCPtr + 1);
        hostDepv = (artsEdtDep_t *)(hostParamv + paramc);
        DPRINTF("Allocated host closure\n");

        //Fill Host closure
        hostGCPtr->gpuId = artsGpu->device;
        hostGCPtr->newEdtLock = newEdtLock;
        hostGCPtr->newEdts = newEdts;
        hostGCPtr->devClosure = devClosure;
        hostGCPtr->edt = (struct artsEdt*)edtPtr;
        for(unsigned int i=0; i<paramc; i++)
            hostParamv[i] = paramv[i];
        DPRINTF("Filled host closure\n");

        artsGuid_t edtGuid = hostGCPtr->edt->currentEdt;
        // artsGpuRouteTableAddItemRace(hostGCPtr, hostClosureSize, edtGuid, artsGpu->device);
        artsGpuRouteTableAddItemRace(hostGCPtr, 0, edtGuid, artsGpu->device);
        DPRINTF("Added edtGuid: %lu size: %u to gpu: %d routing table\n", edtGuid, hostClosureSize, artsGpu->device);        
    }

     artsGpuEdt_t * gpuEdt = (artsGpuEdt_t*) hostGCPtr->edt;

    //Allocate space for DB on GPU and Move Data
    for (unsigned int i=0; i<depc; ++i)
    {
        if(depv[i].ptr)
        {
            void * dataPtr = artsGpuRouteTableLookupDb(depv[i].guid, artsGpu->device);
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size_t size = db->header.size - sizeof(struct artsDb);
            if (!dataPtr)
            {
                bool successfulAdd = false;
                artsItemWrapper_t * wrapper = artsGpuRouteTableReserveItemRace(&successfulAdd, size, depv[i].guid, artsGpu->device);
                
                if(successfulAdd) //We won, so allocate and move data
                {
                    DPRINTF("Adding %lu %u id: %d\n", depv[i].guid, size, artsGpu->device);
                    dataPtr = artsCudaMalloc(size);
                    pushDataToStream(artsGpu->device, dataPtr, depv[i].ptr, size, artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
                    //Must have already launched the memcpy before setting realData or races will ensue
                    wrapper->realData = dataPtr;
                    artsAtomicAdd(&misses, 1U);
                }
                else //Someone beat us to creating the data... So we must free
                {
                    while(!artsAtomicFetchAddU64((uint64_t*)&wrapper->realData, 0)); //Spin till the data memcpy is launched
                    dataPtr = (void*) wrapper->realData;
                    artsAtomicAddSizet(&artsGpu->availGlobalMem, size);
                    artsAtomicAdd(&hits, 1U);
                }
            }
            else
            {
                artsAtomicAddSizet(&artsGpu->availGlobalMem, size);
                artsAtomicAdd(&hits, 1U);
            }
            
            hostDepv[i].ptr = dataPtr;
        }
        else
            hostDepv[i].ptr = NULL;

        hostDepv[i].guid = depv[i].guid;
        hostDepv[i].mode = depv[i].mode;    
    }
    DPRINTF("Allocated, added, and moved dbs\n");
    
    pushDataToStream(artsGpu->device, devClosure, (void*)hostParamv, devClosureSize, artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
    DPRINTF("Filled GPU Closure\n");

    if(gpuEdt->lib)
    {
        artsLocalGrid = &gpuEdt->grid;
        artsLocalBlock = &gpuEdt->block;
        artsLocalStream = &artsGpu->stream;
        artsLocalGpuId = artsGpu->device;
        artsSetThreadLocalEdtInfo(hostGCPtr->edt);
        artsRouteTableResetOO(hostGCPtr->edt->currentEdt);

        hostGCPtr->edt->funcPtr(paramc, hostParamv, depc, hostDepv);

        artsUnsetThreadLocalEdtInfo();
    }
    else
    {
        pushKernelToStream(artsGpu->device, paramc, devParamv, depc, devDepv, fnPtr, grid, block, artsNodeInfo.gpuBuffOn);
    }
    
    //Move data back
    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr && depv[i].mode == ARTS_DB_GPU_WRITE)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
            getDataFromStream(artsGpu->device, depv[i].ptr, hostDepv[i].ptr, size, artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
            // CHECKCORRECT(cudaStreamSynchronize(artsGpu->stream));
        }
    }

    pushWrapUpToStream(artsGpu->device, hostClosure, artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
}

void artsScheduleToGpu(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, void * edtPtr, artsGpu_t * artsGpu)
{
    artsGpuEdt_t * edt = (artsGpuEdt_t *)edtPtr;
    artsScheduleToGpuInternal(fnPtr, paramc, paramv, depc, depv, edt->grid, edt->block, edtPtr, artsGpu);
}

void artsGpuSynchronize(artsGpu_t * artsGpu)
{
    CHECKCORRECT(cudaStreamSynchronize(artsGpu->stream));
}

void artsGpuStreamBusy(artsGpu_t* artsGpu)
{
    CHECKCORRECT(cudaStreamQuery(artsGpu->stream));
}

void freeGpuItem(artsRouteItem_t * item)
{
    artsType_t type = artsGuidGetType(item->key);
    artsItemWrapper_t * wrapper = (artsItemWrapper_t*) item->data;
    if(type == ARTS_GPU_EDT)
    {
        artsGpuCleanUp_t * hostGCPtr = (artsGpuCleanUp_t *) wrapper->realData;
        DPRINTF("FREEING DEV PTR: %p\n", hostGCPtr->devClosure);
        artsCudaFree(hostGCPtr->devClosure);
        DPRINTF("FREEING HOST PTR: %p\n", hostGCPtr);
        artsCudaFreeHost(hostGCPtr);
    }
    else if(type > ARTS_BUFFER && type < ARTS_LAST_TYPE)  //DBs
        artsCudaFree((void*)wrapper->realData);

    wrapper->realData = NULL;
    wrapper->timestamp = 0;
    item->key = 0;
    item->lock = 0;
}

bool tryReserve(int gpu, size_t size)
{
    artsGpu_t * artsGpu = &artsGpus[gpu];
    DPRINTF("Trying to reserve %lu of available %lu on GPU[%d]\n", size, artsGpu->availGlobalMem, artsGpu->device);
    if (artsAtomicFetchAdd(&artsGpu->availableEdtSlots, 1U) < artsNodeInfo.gpuMaxEdts)
    {
        volatile size_t availSize = artsGpu->availGlobalMem;
        while (availSize >= size)
        {
            if (artsAtomicCswapSizet(&artsGpu->availGlobalMem, availSize, availSize-size))
                return true;
            availSize = artsGpu->availGlobalMem;
        }
    }
    artsAtomicSub(&artsGpu->availableEdtSlots, 1U);
    DPRINTF("Failed to reserve %lu of available %lu on GPU[%d]\n", size, artsGpu->availGlobalMem, artsGpu->device);
    return false;
}

int firstFit(uint64_t mask, size_t size)
{
    int random = jrand48(artsThreadInfo.drand_buf);
    for (int i = 0; i < artsNodeInfo.gpu; i++)
    {
        int index = (i+random) % artsNodeInfo.gpu;
        uint64_t checkMask = 1 << index;
        if (mask && checkMask)
            if (tryReserve(index, size))
            {
                DPRINTF("Reserved Successfully on %u\n", index);
                return index;
            }
    }
    return -1;
}

int roundRobinFit(uint64_t mask, size_t size)
{
    static volatile unsigned int next = 0;
    unsigned int start = artsAtomicFetchAdd(&next, 1U);
    for (int i = 0; i < artsNodeInfo.gpu; i++)
    {
        int index = (i+start) % artsNodeInfo.gpu;
        uint64_t checkMask = 1 << index;
        if (mask && checkMask)
            if (tryReserve(index, size))
            {
                DPRINTF("Reserved Successfully on %u\n", index);
                return index;
            }
    }
    return -1;
}

int bestFit(uint64_t mask, size_t size)
{
    int selectedGpu = -1;
    size_t selectedGpuAvailSize;
    int random = jrand48(artsThreadInfo.drand_buf);
    for (int i = 0; i < artsNodeInfo.gpu; i++)
    {
        int index = (i+random) % artsNodeInfo.gpu;
        uint64_t checkMask = 1 << index;
        if (mask && checkMask)
        {
            if (selectedGpu != -1)
            {
                if (artsGpus[index].availGlobalMem-size > selectedGpuAvailSize)
                    continue;
            }
            if (tryReserve(index, size))
            {
                // If successful relinquish previous allocation.
                artsAtomicAddSizet(&artsGpus[selectedGpu].availGlobalMem, size);
                selectedGpu = index;
                selectedGpuAvailSize = artsGpus[index].availGlobalMem;
            }
        }
    }
    return selectedGpu;
}

int worstFit(uint64_t mask, size_t size)
{
    int selectedGpu = -1;
    size_t selectedGpuAvailSize;
    int random = jrand48(artsThreadInfo.drand_buf);
    for (int i = 0; i < artsNodeInfo.gpu; i++)
    {
        int index = (i+random) % artsNodeInfo.gpu;
        uint64_t checkMask = 1 << index;
        if (mask && checkMask)
        {
            if (selectedGpu != -1)
            {
                if (artsGpus[index].availGlobalMem-size < selectedGpuAvailSize)
                    continue;
            }
            if (tryReserve(index, size))
            {
                // If successful relinquish previous allocation.
                artsAtomicAddSizet(&artsGpus[selectedGpu].availGlobalMem, size);
                selectedGpu = index;
                selectedGpuAvailSize = artsGpus[index].availGlobalMem;
            }
        }
    }
    return selectedGpu;
}

int random(void * edtPacket)
{
    artsGpuEdt_t * edt = (artsGpuEdt_t *) edtPacket;
    uint32_t       paramc = edt->wrapperEdt.paramc;
    uint32_t       depc   = edt->wrapperEdt.depc;
    uint64_t     * paramv = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv   = (artsEdtDep_t *)(paramv + paramc);

    // Size to be allocated on the GPU
    size_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc;
    for (unsigned int i = 0; i < depc; i++)
    {
        struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
        size += db->header.size - sizeof(struct artsDb);
    }
    // size = size + db size;
    uint64_t mask = ~0;
    return fit(mask, size);
}

int allOrNothing(void * edtPacket)
{
    artsGpuEdt_t * edt = (artsGpuEdt_t *) edtPacket;
    uint32_t       paramc = edt->wrapperEdt.paramc;
    uint32_t       depc   = edt->wrapperEdt.depc;
    uint64_t     * paramv = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv   = (artsEdtDep_t *)(paramv + paramc);

    // Size to be allocated on the GPU
    size_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc;
    for (unsigned int i = 0; i < depc; i++)
    {
        struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
        size += db->header.size - sizeof(struct artsDb);
    }

    uint64_t mask=0;
    for (unsigned int i=0; i<depc; ++i)
        mask &= artsGpuLookupDb(depv[i].guid);

    DPRINTF("Mask: %p\n", mask);

    if (mask) // All DBs in GPU
        return fit(mask, size); // No need to fit since all Dbs are in a GPU
    else
        return random(edtPacket);
}

int atleastOne(void * edtPacket)
{
    artsGpuEdt_t * edt = (artsGpuEdt_t *) edtPacket;
    uint32_t       paramc = edt->wrapperEdt.paramc;
    uint32_t       depc   = edt->wrapperEdt.depc;
    uint64_t     * paramv = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv   = (artsEdtDep_t *)(paramv + paramc);

    // Size to be allocated on the GPU
    size_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc;
    for (unsigned int i = 0; i < depc; i++)
    {
        struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
        size += db->header.size - sizeof(struct artsDb);
    }

    uint64_t mask=0;
    for (unsigned int i=0; i<depc; ++i)
        mask |= artsGpuLookupDb(depv[i].guid);

    DPRINTF("Mask: %p\n", mask);

    if (mask) // At least one DB in GPU
        return fit(mask, size);
    else
        return random(edtPacket);
}

artsGpu_t * artsFindGpu(void * data)
{
    artsGpu_t * ret = NULL;
    int gpu = -1;

    gpu = locality(data);
    DPRINTF("Choosing gpu: %d\n", gpu);
    if(gpu > -1 && gpu < artsNodeInfo.gpu)
        ret = &artsGpus[gpu];

    return ret;
}