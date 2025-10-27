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

// Some help https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
// and
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
// Once this *class* works we will put a stream(s) in create a thread local
// stream.  Then we will push stuff!
#include "arts/gpu/GpuStream.h"

#include "arts/arts.h"
#include "arts/gas/Guid.h"
#include "arts/gpu/GpuLCSyncFunctions.cuh"
#include "arts/gpu/GpuRouteTable.h"
#include "arts/gpu/GpuRuntime.cuh"
#include "arts/gpu/GpuStreamBuffer.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/compute/EdtFunctions.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"
#include "arts/utils/Atomics.h"
#include "arts/utils/Deque.h"

int random(void *edtPacket);
int allOrNothing(void *edtPacket);
int atleastOne(void *edtPacket);
int hashOnDBZero(void *edtPacket);
int hashLargest(void *edtPacket);
int firstFit(uint64_t mask, uint64_t size, unsigned int totalThreads);
int bestFit(uint64_t mask, uint64_t size, unsigned int totalThreads);
int worstFit(uint64_t mask, uint64_t size, unsigned int totalThreads);
int roundRobinFit(uint64_t mask, uint64_t size, unsigned int totalThreads);
bool tryReserve(int gpu, uint64_t size, unsigned int threads);

volatile unsigned int hits = 0;
volatile unsigned int misses = 0;
volatile uint64_t freeBytes = 0;

artsGpu_t *artsGpus;

typedef int (*locality_t)(void *edt);

locality_t localityScheme[] = {random, allOrNothing, atleastOne, hashOnDBZero,
                               hashLargest};

locality_t locality; // Locality function ptr

typedef int (*fit_t)(uint64_t mask, uint64_t size, unsigned int totalThreads);

fit_t fitScheme[] = {firstFit, bestFit, worstFit, roundRobinFit};

fit_t fit; // Fit function ptr

__thread volatile unsigned int *newEdtLock = 0;
__thread artsArrayList *newEdts = NULL;

// These are for the library version of GPU EDTs
// The user can query to get these values
// We still want to collect them for scheduling purposes
__thread dim3 *artsLocalGrid;
__thread dim3 *artsLocalBlock;
__thread cudaStream_t *artsLocalStream;
__thread int artsLocalGpuId;

#ifdef __cplusplus
extern "C" {
#endif
extern void initPerGpu(unsigned int nodeId, int devId, cudaStream_t *stream,
                       int argc, char **argv) __attribute__((weak));
extern void cleanPerGpu(unsigned int nodeId, int devId, cudaStream_t *stream)
    __attribute__((weak));
#ifdef __cplusplus
}
#endif

bool **gpuAdjList = NULL;
void artsFullyConnectGpus(bool p2p, bool disconnectP2P) {
  if (!gpuAdjList) {
    gpuAdjList = (bool **)artsCalloc(artsNodeInfo.gpu, sizeof(bool *));
    for (unsigned int i = 0; i < artsNodeInfo.gpu; i++)
      gpuAdjList[i] = (bool *)artsCalloc(artsNodeInfo.gpu, sizeof(bool));
  }
  if (p2p) {
    for (int src = 0; src < artsNodeInfo.gpu; src++) {
      artsCudaSetDevice(src, false);
      for (int dst = 0; dst < artsNodeInfo.gpu; dst++) {
        if (src != dst) {
          int hasAccess = 0;
          CHECKCORRECT(cudaDeviceCanAccessPeer(&hasAccess, src, dst));
          if (hasAccess) {
            if (disconnectP2P) {
              CHECKCORRECT(cudaDeviceDisablePeerAccess(dst));
            } else {
              gpuAdjList[src][dst] = 1;
              CHECKCORRECT(cudaDeviceEnablePeerAccess(dst, 0));
            }
          }
        }
      }
    }
  }
}

void artsNodeInitGpus() {
  int numAvailGpus = 0;
  locality = localityScheme[artsNodeInfo.gpuLocality];
  fit = fitScheme[artsNodeInfo.gpuFit];
  CHECKCORRECT(cudaGetDeviceCount(&numAvailGpus));
  if (numAvailGpus < artsNodeInfo.gpu) {
    ARTS_INFO("Requested %d gpus but only %d available\n", numAvailGpus,
              artsNodeInfo.gpu);
    artsNodeInfo.gpu = numAvailGpus;
  }

  ARTS_DEBUG(
      "gpuRouteTableSize: %u gpuRouteTableEntries: %u freeDbAfterGpuRun: "
      "%u runGpuGcIdle: %u runGpuGcPreEdt: %u deleteZerosGpuGc: %u\n",
      artsNodeInfo.gpuRouteTableSize, artsNodeInfo.gpuRouteTableEntries,
      artsNodeInfo.freeDbAfterGpuRun, artsNodeInfo.runGpuGcIdle,
      artsNodeInfo.runGpuGcPreEdt, artsNodeInfo.deleteZerosGpuGc);

  ARTS_DEBUG("NUM DEV: %d\n", artsNodeInfo.gpu);
  artsGpus = (artsGpu_t *)artsCalloc(artsNodeInfo.gpu, sizeof(artsGpu_t));

  artsCudaSetDevice(-1, true);

  // Initialize artsGpu with 1 stream/GPU
  for (int i = 0; i < artsNodeInfo.gpu; ++i) {
    artsGpus[i].device = i;
    ARTS_DEBUG("Setting %d\n", i);
    artsCudaSetDevice(i, false);
    CHECKCORRECT(cudaStreamCreate(&artsGpus[i].stream)); // Make it scalable
    artsNodeInfo.gpuRouteTable[i] = artsGpuNewRouteTable(
        artsNodeInfo.gpuRouteTableEntries, artsNodeInfo.gpuRouteTableSize);
    size_t tempFreeMem = 0;
    size_t tempMaxMem = 0;
    CHECKCORRECT(cudaMemGetInfo((size_t *)&tempFreeMem, (size_t *)&tempMaxMem));
    CHECKCORRECT(
        cudaGetDeviceProperties(&artsGpus[i].prop, artsGpus[i].device));
    artsGpus[i].availGlobalMem = (uint64_t)tempFreeMem;
    artsGpus[i].totalGlobalMem = (uint64_t)tempMaxMem;
    if (artsGpus[i].availGlobalMem > artsNodeInfo.gpuMaxMemory)
      artsGpus[i].availGlobalMem = artsNodeInfo.gpuMaxMemory;
    ARTS_DEBUG("to Start: %lu\n", artsGpus[i].availGlobalMem);
  }

  artsFullyConnectGpus(artsNodeInfo.gpuP2P, false);

  artsCudaRestoreDevice();
}

void artsInitPerGpuWrapper(int argc, char **argv) {
  if (initPerGpu) {
    artsCudaSetDevice(-1, true);
    for (int i = 0; i < artsNodeInfo.gpu; ++i) {
      ARTS_DEBUG("Set device: %u\n", i);
      artsCudaSetDevice(i, false);
      initPerGpu(artsGlobalRankId, i, &artsGpus[i].stream, argc, argv);
    }
    artsCudaRestoreDevice();
  }
}

void artsWorkerInitGpus() {
  newEdtLock = (unsigned int *)artsCalloc(1, sizeof(unsigned int));
  newEdts = artsNewArrayList(sizeof(void *), 32);
}

void artsStoreNewEdts(void *edt) {
  artsLock(newEdtLock);
  artsPushToArrayList(newEdts, &edt);
  artsUnlock(newEdtLock);
}

void artsHandleNewEdts() {
  artsLock(newEdtLock);
  uint64_t size = artsLengthArrayList(newEdts);
  if (size) {
    for (uint64_t i = 0; i < size; i++) {
      struct artsEdt **edt =
          (struct artsEdt **)artsGetFromArrayList(newEdts, i);
      if ((*edt)->header.type == ARTS_EDT)
        artsDequePushFront(artsThreadInfo.myDeque, (*edt), 0);
      if ((*edt)->header.type == ARTS_GPU_EDT)
        artsDequePushFront(artsThreadInfo.myGpuDeque, (*edt), 0);
    }
    artsResetArrayList(newEdts);
  }
  artsUnlock(newEdtLock);
}

void artsCleanupGpus() {
  uint64_t freedSize = 0;
  artsCudaSetDevice(-1, false);

  artsFullyConnectGpus(artsNodeInfo.gpuP2P, true);

  for (int i = 0; i < artsNodeInfo.gpu; i++) {
    artsCudaSetDevice(artsGpus[i].device, false);
    if (cleanPerGpu)
      cleanPerGpu(artsGlobalRankId, i, &artsGpus[i].stream);
    freedSize += artsGpuFreeAll(artsGpus[i].device);
    CHECKCORRECT(cudaStreamSynchronize(artsGpus[i].stream));
    CHECKCORRECT(cudaStreamDestroy(artsGpus[i].stream));
  }
  artsCudaRestoreDevice();
  ARTS_INFO("Occupancy :\n");
  for (int i = 0; i < artsGetNumGpus(); ++i)
    ARTS_INFO("\tGPU[%d] = %f\n", i, artsGpus[i].occupancy);
  ARTS_INFO("HITS: %u MISSES: %u FREED BYTES: %u BYTES FREED ON EXIT %lu\n",
            hits, misses, freeBytes, freedSize);
  ARTS_INFO("HIT RATIO: %lf\n", (double)hits / (double)(hits + misses));
}

void CUDART_CB artsWrapUp(cudaStream_t stream, cudaError_t status, void *data) {
  // artsToggleThreadInspection();

  artsGpuCleanUp_t *gc = (artsGpuCleanUp_t *)data;

  artsGpu_t *artsGpu = &artsGpus[gc->gpuId];
  artsAtomicSub(&artsGpu->availableEdtSlots, 1U);
  artsAtomicSub(&artsGpu->runningEdts, 1U);

  // Shouldn't have to touch newly ready edts regardless of streams and devices
  artsGpuEdt_t *edt = (artsGpuEdt_t *)gc->edt;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);

  unsigned int totalThreads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;
  artsAtomicSub(&artsGpu->availableThreads, totalThreads);

  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      if (artsGuidGetType(depv[i].guid) == ARTS_DB_GPU_WRITE) {
        artsGpuInvalidateRouteTables(depv[i].guid, gc->gpuId);
      }
      // True says to mark it for deletion... Change this to false to further
      // delay delete!
      //  bool markDelete = (artsGuidGetType(depv[i].guid) != ARTS_DB_GPU_WRITE)
      //  && artsNodeInfo.freeDbAfterGpuRun;
      bool markDelete = artsNodeInfo.freeDbAfterGpuRun;
      bool res = artsGpuRouteTableReturnDb(depv[i].guid, markDelete, gc->gpuId);
      // artsGpuRouteTableReturnDb(depv[i].guid, artsNodeInfo.freeDbAfterGpuRun,
      // gc->gpuId);
      ARTS_DEBUG("Returning Db: %lu id: %d res: %u\n", depv[i].guid, gc->gpuId,
                 res);
    }
  }

  // Definitely mark the dev closure to be deleted as there is no reuse!
  artsGpuRouteTableReturnDb(edt->wrapperEdt.currentEdt, true, gc->gpuId);
  newEdtLock = gc->newEdtLock;
  newEdts = gc->newEdts;
  artsGpuHostWrapUp(gc->edt, edt->endGuid, edt->slot, edt->dataGuid);
  ARTS_DEBUG("FINISHED GPU CALLS %s\n", cudaGetErrorString(status));
  // artsToggleThreadInspection();
}

void CUDART_CB artsWrapUpHostFunc(void *data) {
  artsWrapUp(NULL, cudaSuccess, data);
}

void artsScheduleToGpuInternal(artsEdt_t fnPtr, uint32_t paramc,
                               uint64_t *paramv, uint32_t depc,
                               artsEdtDep_t *depv, dim3 grid, dim3 block,
                               void *edtPtr, artsGpu_t *artsGpu) {
  //    For now this should push the following into the stream:
  //    1. Copy data from host to device
  //    2. Push kernel
  //    3. Copy data from device to host
  //    4. Call host callback function artsGpuHostWrapUp

  static volatile unsigned int Gpulock;

  void *devClosure = NULL;
  void *hostClosure = NULL;

  uint64_t *devGpuId = NULL;
  uint64_t *devParamv = NULL;
  artsEdtDep_t *devDepv = NULL;

  artsGpuCleanUp_t *hostGCPtr = NULL;
  uint64_t *hostGpuId = NULL;
  uint64_t *hostParamv = NULL;
  artsEdtDep_t *hostDepv = NULL;

  ARTS_DEBUG("Paramc: %u Depc: %u edt: %p\n", paramc, depc, edtPtr);

  // Get size of closure
  uint64_t devClosureSize =
      sizeof(uint64_t) * (paramc + 1) + sizeof(artsEdtDep_t) * depc;
  uint64_t hostClosureSize = devClosureSize + sizeof(artsGpuCleanUp_t);
  ARTS_DEBUG("devClosureSize: %u hostClosureSize: %u\n", devClosureSize,
             hostClosureSize);

  // Allocate Closure for GPU
  if (devClosureSize) {
    devClosure = artsCudaMalloc(devClosureSize);
    devGpuId = (uint64_t *)devClosure;
    devParamv = devGpuId + 1;
    devDepv = (artsEdtDep_t *)(devParamv + paramc);
    ARTS_DEBUG("Allocated dev closure\n");
  }

  if (hostClosureSize) {
    // Allocate closure for host
    hostClosure = artsCudaMallocHost(hostClosureSize);
    hostGCPtr = (artsGpuCleanUp_t *)hostClosure;
    hostGpuId = (uint64_t *)(hostGCPtr + 1);
    hostParamv = hostGpuId + 1;
    hostDepv = (artsEdtDep_t *)(hostParamv + paramc);
    ARTS_DEBUG("Allocated host closure\n");

    // Fill Host closure
    hostGCPtr->gpuId = artsGpu->device;
    hostGCPtr->newEdtLock = newEdtLock;
    hostGCPtr->newEdts = newEdts;
    hostGCPtr->devClosure = devClosure;
    hostGCPtr->edt = (struct artsEdt *)edtPtr;
    *hostGpuId = (uint64_t)artsGpu->device;
    for (unsigned int i = 0; i < paramc; i++)
      hostParamv[i] = paramv[i];
    ARTS_DEBUG("Filled host closure\n");

    artsGuid_t edtGuid = hostGCPtr->edt->currentEdt;
    // artsGpuRouteTableAddItemRace(hostGCPtr, hostClosureSize, edtGuid,
    // artsGpu->device);
    artsGpuRouteTableAddItemRace(hostGCPtr, devClosureSize, edtGuid,
                                 artsGpu->device);
    ARTS_DEBUG("Added edtGuid: %lu size: %u to gpu: %d routing table\n",
               edtGuid, hostClosureSize, artsGpu->device);
  }

  artsGpuEdt_t *gpuEdt = (artsGpuEdt_t *)hostGCPtr->edt;

  // Allocate space for DB on GPU and Move Data
  for (unsigned int i = 0; i < depc; ++i) {
    if (depv[i].ptr) {
      artsType_t mode =
          artsGuidGetType(depv[i].guid); // use this mode since it is the type
                                         // of DB depv[i].mode is access type
      unsigned int gpuVersion, timeStamp;
      void *dataPtr = artsGpuRouteTableLookupDb(depv[i].guid, artsGpu->device,
                                                &gpuVersion, &timeStamp);
      struct artsDb *db = (struct artsDb *)depv[i].ptr - 1;
      uint64_t size = db->header.size;
      uint64_t allocSize = (mode == ARTS_DB_LC) ? size * 2 : size;
      if (!dataPtr) {
        bool successfulAdd = false;
        ARTS_DEBUG("WRAPPER SIZE: %lu\n", allocSize);
        artsItemWrapper_t *wrapper = artsGpuRouteTableReserveItemRace(
            &successfulAdd, allocSize, depv[i].guid, artsGpu->device,
            false); //(mode == ARTS_DB_LC));

        if (successfulAdd) // We won, so allocate and move data
        {
          ARTS_DEBUG("Adding %lu %u id: %d mode: %s\n", depv[i].guid, allocSize,
                     artsGpu->device, _artsTypeName[depv[i].mode]);
          dataPtr = artsCudaMalloc(allocSize);
          void *src = (void *)db;
          if (mode == ARTS_DB_LC)
            src = makeLCShadowCopy(db);
          if (depv[i].mode == ARTS_DB_LC_NO_COPY ||
              depv[i].mode == ARTS_DB_GPU_MEMSET)
            src = NULL;
          pushDataToStream(artsGpu->device, dataPtr, src, size,
                           artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
          // Must have already launched the memcpy before setting realData or
          // races will ensue
          wrapper->realData = dataPtr;
          ARTS_DEBUG("Malloc[%d]: %p %p\n", artsGpu->device, wrapper, dataPtr);
          artsAtomicAdd(&misses, 1U);
        } else // Someone beat us to creating the data... So we must free
        {
          while (!artsAtomicFetchAddU64((uint64_t *)&wrapper->realData, 0))
            ; // Spin till the data memcpy is launched
          dataPtr = (void *)wrapper->realData;
          if (mode == ARTS_DB_GPU_WRITE && depv[i].mode == ARTS_DB_GPU_MEMSET) {
            pushDataToStream(artsGpu->device, dataPtr, NULL, size,
                             artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
          }
          artsAtomicAddU64(&artsGpu->availGlobalMem, allocSize);
          artsAtomicAdd(&hits, 1U);
        }
      } else {
        artsAtomicAddU64(&artsGpu->availGlobalMem, allocSize);
        artsAtomicAdd(&hits, 1U);
      }
      struct artsDb *newDb = (struct artsDb *)dataPtr;
      hostDepv[i].ptr = (void *)(newDb + 1);
    } else {
      ARTS_DEBUG("Depv: %u is null edt: %lu\n", i,
                 gpuEdt->wrapperEdt.currentEdt);
      hostDepv[i].ptr = NULL;
    }

    hostDepv[i].guid = depv[i].guid;
    hostDepv[i].mode = depv[i].mode;
  }
  ARTS_DEBUG("Allocated, added, and moved dbs\n");

  pushDataToStream(artsGpu->device, devClosure, (void *)hostGpuId,
                   devClosureSize, artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
  ARTS_DEBUG("Filled GPU Closure\n");

  if (gpuEdt->lib) {
    artsLocalGrid = &gpuEdt->grid;
    artsLocalBlock = &gpuEdt->block;
    artsLocalStream = &artsGpu->stream;
    artsLocalGpuId = artsGpu->device;
    artsSetThreadLocalEdtInfo(hostGCPtr->edt);
    artsRouteTableResetOO(hostGCPtr->edt->currentEdt);

    hostGCPtr->edt->funcPtr(paramc, hostParamv, depc, hostDepv);
    artsMetricsTriggerEvent(artsGpuEdt, artsThread, 1);

    artsUnsetThreadLocalEdtInfo();
  } else {
    pushKernelToStream(artsGpu->device, paramc, devParamv, depc, devDepv, fnPtr,
                       grid, block, artsNodeInfo.gpuBuffOn);
  }

  // Move data back
  for (unsigned int i = 0; i < depc; i++) {
    artsType_t mode = artsGuidGetType(depv[i].guid);
    if (depv[i].ptr && mode == ARTS_DB_GPU_WRITE) {
      struct artsDb *db = (struct artsDb *)depv[i].ptr - 1;
      size_t size = (size_t)(db->header.size - sizeof(struct artsDb));
      getDataFromStream(artsGpu->device, depv[i].ptr, hostDepv[i].ptr, size,
                        artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
      // CHECKCORRECT(cudaStreamSynchronize(artsGpu->stream));
    }
  }

  pushWrapUpToStream(artsGpu->device, hostClosure,
                     artsNodeInfo.gpuBuffOn && !gpuEdt->lib);
}

void artsScheduleToGpu(artsEdt_t fnPtr, uint32_t paramc, uint64_t *paramv,
                       uint32_t depc, artsEdtDep_t *depv, void *edtPtr,
                       artsGpu_t *artsGpu) {
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPtr;
  artsScheduleToGpuInternal(fnPtr, paramc, paramv, depc, depv, edt->grid,
                            edt->block, edtPtr, artsGpu);
}

void artsGpuSynchronize(artsGpu_t *artsGpu) {
  CHECKCORRECT(cudaStreamSynchronize(artsGpu->stream));
}

void artsGpuStreamBusy(artsGpu_t *artsGpu) {
  CHECKCORRECT(cudaStreamQuery(artsGpu->stream));
}

void freeGpuItem(artsRouteItem_t *item) {
  artsType_t type = artsGuidGetType(item->key);
  artsItemWrapper_t *wrapper = (artsItemWrapper_t *)item->data;
  if (type == ARTS_GPU_EDT) {
    artsGpuCleanUp_t *hostGCPtr = (artsGpuCleanUp_t *)wrapper->realData;
    ARTS_DEBUG("FREEING DEV PTR: %p\n", hostGCPtr->devClosure);
    artsCudaFree(hostGCPtr->devClosure);
    ARTS_DEBUG("FREEING HOST PTR: %p\n", hostGCPtr);
    artsCudaFreeHost(hostGCPtr);
  } else if (type == ARTS_DB_LC) {
    int validRank = -1;
    struct artsDb *db =
        (struct artsDb *)artsRouteTableLookupDb(item->key, &validRank, false);
    if (db) {
      unsigned int size = db->header.size;
      struct artsDb *tempSpace = (struct artsDb *)artsMallocAlign(size, 16);

      artsLCMeta_t host;
      host.guid = item->key;
      host.data = (void *)(db + 1);
      host.dataSize = db->header.size - sizeof(struct artsDb);
      host.hostVersion = &db->version;
      host.hostTimeStamp = &db->timeStamp;
      host.gpuVersion = 0;
      host.gpuTimeStamp = 0;
      host.gpu = -1;
      host.readLock = &db->reader;
      host.writeLock = &db->writer;

      // artsCudaMemCpyFromDev(tempSpace, (void*) wrapper->realData, size);
      getDataFromStreamNow(artsGetCurrentGpu(), tempSpace,
                           (void *)wrapper->realData, size, false);

      artsLCMeta_t dev;
      dev.guid = item->key;
      dev.data = (void *)(tempSpace + 1);
      dev.dataSize = tempSpace->header.size - sizeof(struct artsDb);
      dev.hostVersion = &tempSpace->version;
      dev.hostTimeStamp = &tempSpace->timeStamp;
      dev.gpuVersion = item->touched;
      dev.gpuTimeStamp = wrapper->timeStamp;
      dev.gpu = -1;
      dev.readLock = NULL;
      dev.writeLock = NULL;

      lcSyncFunction[artsNodeInfo.gpuLCSync](&host, &dev);

      artsRouteTableReturnDb(item->key, false);
      artsFree(tempSpace);
      artsCudaFree((void *)wrapper->realData);

      artsMetricsTriggerEvent(artsGpuSyncDelete, artsThread, 1);
    } else
      ARTS_INFO("Trying to delete an LC but there is no DB to back up to\n");
  } else if (type > ARTS_BUFFER && type < ARTS_LAST_TYPE) // DBs
    artsCudaFree((void *)wrapper->realData);

  wrapper->realData = NULL;
  wrapper->timeStamp = 0;
  item->key = 0;
  item->lock = 0;
  item->touched = 0;
}

__thread unsigned int runGCFlag = 0;

bool tryReserve(int gpu, uint64_t size, unsigned int threads) {
  artsGpu_t *artsGpu = &artsGpus[gpu];
  ARTS_DEBUG("Trying to reserve %lu of available %lu on GPU[%d]\n", size,
             artsGpu->availGlobalMem, artsGpu->device);
  // if(artsAtomicFetchAdd(&artsGpu->availableThreads, threads) < 1024)
  {
    if (artsAtomicFetchAdd(&artsGpu->availableEdtSlots, 1U) <
        artsNodeInfo.gpuMaxEdts) {
      volatile uint64_t availSize = artsGpu->availGlobalMem;
      while (availSize >= size) {
        if (artsAtomicCswapU64(&artsGpu->availGlobalMem, availSize,
                               availSize - size)) {
          runGCFlag = 0;
          return true;
        }
        availSize = artsGpu->availGlobalMem;
      }
      runGCFlag = gpu + 1;
    }
    artsAtomicSub(&artsGpu->availableEdtSlots, 1U);
  }
  // artsAtomicSub(&artsGpu->availableThreads, threads);
  ARTS_DEBUG("Failed Avail threads: %u + %u\n", artsGpu->availableThreads,
             threads);
  ARTS_DEBUG("Failed to reserve %lu of available %lu on GPU[%d]\n", size,
             artsGpu->availGlobalMem, artsGpu->device);
  return false;
}

int firstFit(uint64_t mask, uint64_t size, unsigned int totalThreads) {
  int random = jrand48(artsThreadInfo.drand_buf);
  for (int i = 0; i < artsNodeInfo.gpu; i++) {
    int index = (i + random) % artsNodeInfo.gpu;
    uint64_t checkMask = 1 << index;
    if (mask && checkMask)
      if (tryReserve(index, size, totalThreads)) {
        ARTS_DEBUG("Reserved Successfully on %u\n", index);
        return index;
      }
  }
  return -1;
}

int roundRobinFit(uint64_t mask, uint64_t size, unsigned int totalThreads) {
  static volatile unsigned int next = 0;
  unsigned int start = artsAtomicFetchAdd(&next, 1U);
  for (int i = 0; i < artsNodeInfo.gpu; i++) {
    int index = (i + start) % artsNodeInfo.gpu;
    uint64_t checkMask = 1 << index;
    if (mask && checkMask)
      if (tryReserve(index, size, totalThreads)) {
        ARTS_DEBUG("Reserved Successfully on %u\n", index);
        return index;
      }
  }
  return -1;
}

int bestFit(uint64_t mask, uint64_t size, unsigned int totalThreads) {
  int selectedGpu = -1;
  uint64_t selectedGpuAvailSize;
  int random = jrand48(artsThreadInfo.drand_buf);
  for (int i = 0; i < artsNodeInfo.gpu; i++) {
    int index = (i + random) % artsNodeInfo.gpu;
    uint64_t checkMask = 1 << index;
    if (mask && checkMask) {
      if (selectedGpu != -1) {
        if (artsGpus[index].availGlobalMem - size > selectedGpuAvailSize)
          continue;
      }
      if (tryReserve(index, size, totalThreads)) {
        // If successful relinquish previous allocation.
        artsAtomicAddU64(&artsGpus[selectedGpu].availGlobalMem, size);
        selectedGpu = index;
        selectedGpuAvailSize = artsGpus[index].availGlobalMem;
      }
    }
  }
  return selectedGpu;
}

int worstFit(uint64_t mask, uint64_t size, unsigned int totalThreads) {
  int selectedGpu = -1;
  uint64_t selectedGpuAvailSize;
  int random = jrand48(artsThreadInfo.drand_buf);
  for (int i = 0; i < artsNodeInfo.gpu; i++) {
    int index = (i + random) % artsNodeInfo.gpu;
    uint64_t checkMask = 1 << index;
    if (mask && checkMask) {
      if (selectedGpu != -1) {
        if (artsGpus[index].availGlobalMem - size < selectedGpuAvailSize)
          continue;
      }
      if (tryReserve(index, size, totalThreads)) {
        // If successful relinquish previous allocation.
        artsAtomicAddU64(&artsGpus[selectedGpu].availGlobalMem, size);
        selectedGpu = index;
        selectedGpuAvailSize = artsGpus[index].availGlobalMem;
      }
    }
  }
  return selectedGpu;
}

uint64_t getDbSizeNeeded(uint32_t depc, artsEdtDep_t *depv) {
  uint64_t size = 0;
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      struct artsDb *db = (struct artsDb *)depv[i].ptr - 1;
      size += db->header.size;
      if (artsGuidGetType(depv[i].guid) == ARTS_DB_LC)
        size += db->header.size;
    }
  }
  return size;
}

int random(void *edtPacket) {
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPacket;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);
  unsigned int totalThreads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  uint64_t mask = ~0;
  return fit(mask, size, totalThreads);
}

int allOrNothing(void *edtPacket) {
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPacket;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);
  unsigned int totalThreads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  uint64_t mask = 0;
  for (unsigned int i = 0; i < depc; ++i)
    mask &= artsGpuLookupDb(depv[i].guid);

  ARTS_DEBUG("Mask: %p\n", mask);

  if (mask) { // All DBs in GPU
    return fit(mask, size,
               totalThreads); // No need to fit since all Dbs are in a GPU
  }
  return random(edtPacket);
}

int atleastOne(void *edtPacket) {
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPacket;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);
  unsigned int totalThreads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  uint64_t mask = 0;
  for (unsigned int i = 0; i < depc; ++i)
    mask |= artsGpuLookupDb(depv[i].guid);

  ARTS_DEBUG("Mask: %p\n", mask);

  if (mask) { // At least one DB in GPU
    return fit(mask, size, totalThreads);
  }
  return random(edtPacket);
}

int hashOnDBZero(void *edtPacket) {
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPacket;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);
  unsigned int totalThreads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  uint64_t key = (depv[0].guid) ? artsGetGuidKey(depv[0].guid) : 0;
  uint64_t index = key % (uint64_t)artsNodeInfo.gpu;
  if (index > artsNodeInfo.gpu) {
    ARTS_INFO("WHATS WRONG WITH THE HASH %u\n", index);
    artsDebugGenerateSegFault();
  }
  ARTS_DEBUG("HASH: %lu %u\n", depv[0].guid, index);
  if (tryReserve(index, size, totalThreads))
    return index;
  return -1;
}

int hashLargest(void *edtPacket) {
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPacket;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);
  unsigned int totalThreads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  // uint64_t mask = 0;
  uint64_t largest = 0;
  for (unsigned int i = 0; i < depc; ++i) {
    uint64_t key = (depv[i].guid) ? artsGetGuidKey(depv[i].guid) : 0;
    largest = (key > largest) ? key : largest;
  }

  uint64_t index = largest % (uint64_t)artsNodeInfo.gpu;
  if (tryReserve(index, size, totalThreads)) {
    ARTS_DEBUG("Index: %u\n", index);
    return index;
  }
  return -1;
}

int artsReserveEdtRequiredGpu(int *gpu, void *edtPacket) {
  bool ret = false;
  *gpu = -1;
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPacket;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);
  unsigned int totalThreads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  if (edt->gpuToRunOn > -1) {
    // Size to be allocated on the GPU
    uint64_t size = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc +
                    getDbSizeNeeded(depc, depv);
    if (tryReserve(edt->gpuToRunOn, size, totalThreads)) {
      *gpu = edt->gpuToRunOn;
      ret = true;
    }
  }
  return ret;
}

artsGpu_t *artsFindGpu(void *data) {
  artsGpu_t *ret = NULL;
  int gpu;
  if (!artsReserveEdtRequiredGpu(&gpu, data))
    gpu = locality(data);
  ARTS_DEBUG("Choosing gpu: %d\n", gpu);
  if (gpu > -1 && gpu < artsNodeInfo.gpu)
    ret = &artsGpus[gpu];

  return ret;
}