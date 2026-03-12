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
#include "arts/gpu/GpuRuntime.cuh"

#include "arts/gas/OutOfOrder.h"
#include "arts/gpu/GpuLCSyncFunctions.cuh"
#include "arts/gpu/GpuRouteTable.h"
#include "arts/gpu/GpuStream.h"
#include "arts/gpu/GpuStreamBuffer.h"
#include "arts/introspection/Metrics.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/Runtime.h"
#include "arts/runtime/compute/EdtFunctions.h"
#include "arts/runtime/memory/DbFunctions.h"
#include "arts/runtime/sync/TerminationDetection.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"
#include "arts/utils/Atomics.h"
#include "arts/utils/Deque.h"

__thread int artsSavedDeviceId = -1;
__thread int artsCurrentDeviceId = -1;

int artsGetCurrentGpu() {
  if (artsCurrentDeviceId == -1)
    CHECKCORRECT(cudaGetDevice(&artsCurrentDeviceId));

  return artsCurrentDeviceId;
}

bool artsCudaSetDevice(int id, bool save) {
  if (artsCurrentDeviceId == -1)
    CHECKCORRECT(cudaGetDevice(&artsCurrentDeviceId));

  if (save)
    artsSavedDeviceId = artsCurrentDeviceId;

  if (id > -1 && id < artsNodeInfo.gpu && id != artsCurrentDeviceId) {
    CHECKCORRECT(cudaSetDevice(id));
    artsCurrentDeviceId = id;
    return true;
  }

  return false;
}

bool artsCudaRestoreDevice() {
  return artsCudaSetDevice(artsSavedDeviceId, false);
}

void *artsCudaMallocHost(unsigned int size) {
  void *ptr = NULL;
  CHECKCORRECT(cudaMallocHost(&ptr, size));
  // ptr = artsCalloc(1, size);
  if (!ptr) {
    artsDebugPrintStack();
    exit(1);
  }
  return ptr;
}

void artsCudaFreeHost(void *ptr) {
  if (ptr)
    CHECKCORRECT(cudaFreeHost(ptr));
  // artsFree(ptr);
}

void *artsCudaMalloc(unsigned int size) {
  void *ptr = NULL;
  CHECKCORRECT(cudaMalloc(&ptr, size));
  if (!ptr) {
    ARTS_INFO("artsCudaMalloc failed %lu\n",
              artsGpus[artsCurrentDeviceId].availGlobalMem);
    artsDebugPrintStack();
    exit(1);
  }
  return ptr;
}

void artsCudaFree(void *ptr) {
  if (ptr)
    CHECKCORRECT(cudaFree(ptr));
}

void artsCudaMemCpyFromDev(void *dst, void *src, size_t count) {
  CHECKCORRECT(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void artsCudaMemCpyToDev(void *dst, void *src, size_t count) {
  CHECKCORRECT(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

dim3 *artsGetGpuGrid() { return artsLocalGrid; }

dim3 *artsGetGpuBlock() { return artsLocalBlock; }

cudaStream_t *artsGetGpuStream() { return artsLocalStream; }

int artsGetGpuId() { return artsLocalGpuId; }

unsigned int artsGetNumGpus() { return artsNodeInfo.gpu; }

artsGuid_t internalEdtCreateGpu(artsEdt_t funcPtr, artsGuid_t *guid,
                                unsigned int route, uint32_t paramc,
                                uint64_t *paramv, uint32_t depc, dim3 grid,
                                dim3 block, artsGuid_t endGuid, uint32_t slot,
                                artsGuid_t dataGuid, bool hasDepv,
                                bool passThrough, bool lib, int gpuToRunOn) {
  //    ARTSEDTCOUNTERTIMERSTART(edtCreateCounter);
  unsigned int depSpace = (hasDepv) ? depc * sizeof(artsEdtDep_t) : 0;
  unsigned int edtSpace =
      sizeof(artsGpuEdt_t) + paramc * sizeof(uint64_t) + depSpace;

  artsGpuEdt_t *edt = (artsGpuEdt_t *)artsCalloc(1, edtSpace);
  edt->wrapperEdt.invalidateCount = 1;
  edt->grid = grid;
  edt->block = block;
  edt->gpuToRunOn = gpuToRunOn;
  edt->endGuid = endGuid;
  edt->slot = slot;
  edt->dataGuid = dataGuid;
  edt->passthrough = passThrough;
  edt->lib = lib;

  // artsIntrospectionEdtCreateBegin();
  bool created = artsEdtCreateInternal(
      (struct artsEdt *)edt, ARTS_GPU_EDT, guid, route,
      artsThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv,
      depc, true, NULL_GUID, hasDepv, 0);
  // artsIntrospectionEdtCreateFinish(created);
  //    ARTSEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
  return *guid;
}

artsGuid_t artsEdtCreateGpuDep(artsEdt_t funcPtr, unsigned int route,
                               uint32_t paramc, uint64_t *paramv, uint32_t depc,
                               dim3 grid, dim3 block, artsGuid_t endGuid,
                               uint32_t slot, artsGuid_t dataGuid,
                               bool hasDepv) {
  artsGuid_t guid = NULL_GUID;
  return internalEdtCreateGpu(funcPtr, &guid, route, paramc, paramv, depc, grid,
                              block, endGuid, slot, dataGuid, hasDepv, false,
                              false, -1);
}

artsGuid_t artsEdtCreateGpuPTDep(artsEdt_t funcPtr, unsigned int route,
                                 uint32_t paramc, uint64_t *paramv,
                                 uint32_t depc, dim3 grid, dim3 block,
                                 artsGuid_t endGuid, uint32_t slot,
                                 unsigned int passSlot, bool hasDepv) {
  artsGuid_t guid = NULL_GUID;
  return internalEdtCreateGpu(funcPtr, &guid, route, paramc, paramv, depc, grid,
                              block, endGuid, slot, (artsGuid_t)passSlot,
                              hasDepv, true, false, -1);
}

artsGuid_t artsEdtCreateGpu(artsEdt_t funcPtr, unsigned int route,
                            uint32_t paramc, uint64_t *paramv, uint32_t depc,
                            dim3 grid, dim3 block, artsGuid_t endGuid,
                            uint32_t slot, artsGuid_t dataGuid) {
  return artsEdtCreateGpuDep(funcPtr, route, paramc, paramv, depc, grid, block,
                             endGuid, slot, dataGuid, true);
}

artsGuid_t artsEdtCreateGpuWithGuid(artsEdt_t funcPtr, artsGuid_t guid,
                                    uint32_t paramc, uint64_t *paramv,
                                    uint32_t depc, dim3 grid, dim3 block,
                                    artsGuid_t endGuid, uint32_t slot,
                                    artsGuid_t dataGuid) {
  return internalEdtCreateGpu(funcPtr, &guid, artsGuidGetRank(guid), paramc,
                              paramv, depc, grid, block, endGuid, slot,
                              dataGuid, true, false, false, -1);
}

artsGuid_t artsEdtCreateGpuPT(artsEdt_t funcPtr, unsigned int route,
                              uint32_t paramc, uint64_t *paramv, uint32_t depc,
                              dim3 grid, dim3 block, artsGuid_t endGuid,
                              uint32_t slot, unsigned int passSlot) {
  return artsEdtCreateGpuPTDep(funcPtr, route, paramc, paramv, depc, grid,
                               block, endGuid, slot, passSlot, true);
}

artsGuid_t artsEdtCreateGpuPTWithGuid(artsEdt_t funcPtr, artsGuid_t guid,
                                      uint32_t paramc, uint64_t *paramv,
                                      uint32_t depc, dim3 grid, dim3 block,
                                      artsGuid_t endGuid, uint32_t slot,
                                      unsigned int passSlot) {
  return internalEdtCreateGpu(funcPtr, &guid, artsGuidGetRank(guid), paramc,
                              paramv, depc, grid, block, endGuid, slot,
                              (artsGuid_t)passSlot, true, true, false, -1);
}

artsGuid_t artsEdtCreateGpuLib(artsEdt_t funcPtr, unsigned int route,
                               uint32_t paramc, uint64_t *paramv, uint32_t depc,
                               dim3 grid, dim3 block) {
  artsGuid_t guid = NULL_GUID;
  return internalEdtCreateGpu(funcPtr, &guid, route, paramc, paramv, depc, grid,
                              block, NULL_GUID, 0, NULL_GUID, true, false, true,
                              -1);
}

artsGuid_t artsEdtCreateGpuLibWithGuid(artsEdt_t funcPtr, artsGuid_t guid,
                                       uint32_t paramc, uint64_t *paramv,
                                       uint32_t depc, dim3 grid, dim3 block) {
  return internalEdtCreateGpu(funcPtr, &guid, artsGuidGetRank(guid), paramc,
                              paramv, depc, grid, block, NULL_GUID, 0,
                              NULL_GUID, true, false, true, -1);
}

artsGuid_t artsEdtCreateGpuDirect(artsEdt_t funcPtr, unsigned int route,
                                  unsigned int gpu, uint32_t paramc,
                                  uint64_t *paramv, uint32_t depc, dim3 grid,
                                  dim3 block, artsGuid_t endGuid, uint32_t slot,
                                  artsGuid_t dataGuid, bool hasDepv) {
  artsGuid_t guid = NULL_GUID;
  return internalEdtCreateGpu(funcPtr, &guid, route, paramc, paramv, depc, grid,
                              block, endGuid, slot, dataGuid, hasDepv, false,
                              false, gpu);
}

artsGuid_t artsEdtCreateGpuLibDirect(artsEdt_t funcPtr, unsigned int route,
                                     unsigned int gpu, uint32_t paramc,
                                     uint64_t *paramv, uint32_t depc, dim3 grid,
                                     dim3 block) {
  artsGuid_t guid = NULL_GUID;
  return internalEdtCreateGpu(funcPtr, &guid, route, paramc, paramv, depc, grid,
                              block, NULL_GUID, 0, NULL_GUID, true, false, true,
                              gpu);
}

void artsRunGpu(void *edtPacket, artsGpu_t *artsGpu) {
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPacket;
  artsEdt_t func = edt->wrapperEdt.funcPtr;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);

  artsCudaSetDevice(artsGpu->device, true);

  if (artsNodeInfo.runGpuGcPreEdt) {
    // ARTS_INFO("Running Pre Edt GPU GC: %u\n", artsGpu->device);
    uint64_t freeMemSize = artsGpuCleanUpRouteTable(
        (unsigned int)-1, artsNodeInfo.deleteZerosGpuGc,
        (unsigned int)artsGpu->device);
    artsAtomicAddU64(&artsGpu->availGlobalMem, freeMemSize);
    artsAtomicAddU64(&freeBytes, freeMemSize);
  }

  artsAtomicAdd(&artsGpu->runningEdts, 1U);

  prepDbs(depc, depv, true);
  artsScheduleToGpu(func, paramc, paramv, depc, depv, edtPacket, artsGpu);

  artsCudaRestoreDevice();
}

void artsGpuHostWrapUp(void *edtPacket, artsGuid_t toSignal, uint32_t slot,
                       artsGuid_t dataGuid) {
  artsGpuEdt_t *edt = (artsGpuEdt_t *)edtPacket;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  uint64_t *paramv = (uint64_t *)(edt + 1);
  artsEdtDep_t *depv = (artsEdtDep_t *)(paramv + paramc);

  releaseDbs(depc, depv, true);

  if (edt->lib) {
    edt->wrapperEdt.invalidateCount = 0;
    artsRouteTableFireOO(edt->wrapperEdt.currentEdt, artsOutOfOrderHandler);
  } else if (edt->wrapperEdt.epochGuid) {
    incrementFinishedEpoch(edt->wrapperEdt.epochGuid);
  }

  ARTS_DEBUG("TO SIGNAL: %lu -> %lu slot: %u\n", toSignal, dataGuid, slot);
  // Signal next
  if (toSignal) {
    if (edt->passthrough)
      artsSignalEdt(toSignal, slot, depv[dataGuid].guid);
    else {
      artsType_t mode = artsGuidGetType(toSignal);
      if (mode == ARTS_EDT || mode == ARTS_GPU_EDT)
        artsSignalEdt(toSignal, slot, dataGuid);
      if (mode == ARTS_EVENT)
        artsEventSatisfySlot(toSignal, dataGuid, slot);
      if (mode ==
          ARTS_BUFFER) // This is for us to be able to block in a host edt
        artsSetBuffer(toSignal, 0, 0);
    }
  }
  artsEdtDelete((struct artsEdt *)edtPacket);
}

struct artsEdt *artsRuntimeStealGpuTask() {
  struct artsEdt *edt = NULL;
  if (artsNodeInfo.totalThreadCount > 1) {
    long unsigned int stealLoc;
    do {
      stealLoc = jrand48(artsThreadInfo.drand_buf);
      stealLoc = stealLoc % artsNodeInfo.totalThreadCount;
    } while (stealLoc == artsThreadInfo.threadId);
    edt = (struct artsEdt *)artsDequePopBack(artsNodeInfo.gpuDeque[stealLoc]);
  }
  return edt;
}

bool artsGpuSchedulerLoop() {
  artsGpu_t *artsGpu = NULL;
  artsHandleNewEdts();

  struct artsEdt *edtFound = (struct artsEdt *)NULL;
  if (!(edtFound =
            (struct artsEdt *)artsDequePopFront(artsThreadInfo.myGpuDeque))) {
    if (!edtFound)
      edtFound = artsRuntimeStealGpuTask();
  }

  bool ranGpuEdt = false;
  if (edtFound) {
    artsGpu = artsFindGpu(edtFound);
    if (artsGpu) {
      artsRunGpu(edtFound, artsGpu);
      ranGpuEdt = true;
    } else
      artsDequePushFront(artsThreadInfo.myGpuDeque, edtFound, 0);
  }

  if (!ranGpuEdt)
    checkStreams(artsNodeInfo.gpuBuffOn);

  bool ranCpuEdt = artsDefaultSchedulerLoop();
  if (artsNodeInfo.runGpuGcIdle && !ranGpuEdt && !ranCpuEdt) {
    long unsigned int gpuId = jrand48(artsThreadInfo.drand_buf);
    gpuId = gpuId % artsNodeInfo.gpu;
    artsGpu = &artsGpus[gpuId];
    ARTS_DEBUG("Running Idle GPU GC: %u\n", gpuId);
    artsCudaSetDevice(artsGpu->device, true);

    uint64_t freeMemSize = artsGpuCleanUpRouteTable(
        (unsigned int)-1, artsNodeInfo.deleteZerosGpuGc,
        (unsigned int)artsGpu->device);
    artsAtomicAddU64(&artsGpu->availGlobalMem, freeMemSize);
    artsAtomicAddU64(&freeBytes, freeMemSize);

    artsCudaRestoreDevice();
  }

  return ranCpuEdt;
}

#define GCHARDLIMIT 2000000000000
__thread uint64_t backoff = 1;
__thread uint64_t gcCounter = 0;

bool artsGpuSchedulerBackoffLoop() {
  artsGpu_t *artsGpu = NULL;
  artsHandleNewEdts();

  struct artsEdt *edtFound = (struct artsEdt *)NULL;
  if (!(edtFound =
            (struct artsEdt *)artsDequePopFront(artsThreadInfo.myGpuDeque))) {
    if (!edtFound)
      edtFound = artsRuntimeStealGpuTask();
  }

  bool ranGpuEdt = false;
  if (edtFound) {
    artsGpu = artsFindGpu(edtFound);
    if (artsGpu) {
      artsRunGpu(edtFound, artsGpu);
      ranGpuEdt = true;
    } else
      artsDequePushFront(artsThreadInfo.myGpuDeque, edtFound, 0);
  }

  if (!ranGpuEdt)
    checkStreams(artsNodeInfo.gpuBuffOn);

  bool ranCpuEdt = artsDefaultSchedulerLoop();

  if (ranCpuEdt || ranGpuEdt)
    backoff = 1;

  if (!ranGpuEdt && !ranCpuEdt) {
    if (artsNodeInfo.runGpuGcIdle && gcCounter % backoff == 0) {
      long unsigned int gpuId = jrand48(artsThreadInfo.drand_buf);
      gpuId = gpuId % artsNodeInfo.gpu;
      artsGpu = &artsGpus[gpuId];
      ARTS_DEBUG("Running Idle GPU GC: %u\n", gpuId);
      artsCudaSetDevice(artsGpu->device, true);

      uint64_t freeMemSize = artsGpuCleanUpRouteTable(
          (unsigned int)-1, artsNodeInfo.deleteZerosGpuGc,
          (unsigned int)artsGpu->device);
      artsAtomicAddU64(&artsGpu->availGlobalMem, freeMemSize);
      artsAtomicAddU64(&freeBytes, freeMemSize);

      artsCudaRestoreDevice();

      if (backoff < GCHARDLIMIT)
        backoff *= 32;
      if (!backoff)
        backoff = 1;
      ARTS_DEBUG("Backoff: %u\n", backoff);
    }
    gcCounter++;
  }

  return ranCpuEdt;
}

extern __thread unsigned int runGCFlag;

bool artsGpuSchedulerDemandLoop() {
  artsGpu_t *artsGpu = NULL;
  artsHandleNewEdts();

  struct artsEdt *edtFound = (struct artsEdt *)NULL;
  if (!(edtFound =
            (struct artsEdt *)artsDequePopFront(artsThreadInfo.myGpuDeque))) {
    if (!edtFound)
      edtFound = artsRuntimeStealGpuTask();
  }

  bool ranGpuEdt = false;
  if (edtFound) {
    artsGpu = artsFindGpu(edtFound);
    if (artsGpu) {
      artsRunGpu(edtFound, artsGpu);
      ranGpuEdt = true;
    } else
      artsDequePushFront(artsThreadInfo.myGpuDeque, edtFound, 0);
  }

  if (!ranGpuEdt)
    checkStreams(artsNodeInfo.gpuBuffOn);

  bool ranCpuEdt = artsDefaultSchedulerLoop();

  if (!ranGpuEdt && !ranCpuEdt) {
    if (artsNodeInfo.runGpuGcIdle && runGCFlag) {
      long unsigned int gpuId = runGCFlag - 1;
      runGCFlag = 0;

      artsGpu = &artsGpus[gpuId];
      ARTS_DEBUG("Running Idle GPU GC: %u\n", gpuId);
      artsCudaSetDevice(artsGpu->device, true);

      uint64_t freeMemSize = artsGpuCleanUpRouteTable(
          (unsigned int)-1, artsNodeInfo.deleteZerosGpuGc,
          (unsigned int)artsGpu->device);
      artsAtomicAddU64(&artsGpu->availGlobalMem, freeMemSize);
      artsAtomicAddU64(&freeBytes, freeMemSize);

      artsCudaRestoreDevice();
    }
  }

  return ranCpuEdt;
}

void artsPutInDbFromGpu(void *ptr, artsGuid_t dbGuid, unsigned int offset,
                        unsigned int size, bool freeData) {
  unsigned int rank = artsGuidGetRank(dbGuid);
  if (rank == artsGlobalRankId) {
    struct artsDb *db = (struct artsDb *)artsRouteTableLookupItem(dbGuid);
    if (db) {
      void *data = (void *)(((char *)(db + 1)) + offset);
      // memcpy(data, ptr, size);
      CHECKCORRECT(cudaMemcpyAsync(data, ptr, size, cudaMemcpyDeviceToHost,
                                   *artsLocalStream));

    } else {
      void *cpyPtr = artsMalloc(size);
      // memcpy(cpyPtr, ptr, size);
      CHECKCORRECT(cudaMemcpyAsync(cpyPtr, ptr, size, cudaMemcpyDeviceToHost,
                                   *artsLocalStream));
      artsOutOfOrderPutInDb(cpyPtr, NULL_GUID, dbGuid, 0, offset, size,
                            NULL_GUID);
    }
    if (freeData)
      artsGpuRouteTableAddItemToDeleteRace(ptr, 0, dbGuid, artsLocalGpuId);
  }
}

artsLCSyncFunction_t lcSyncFunction[] = {
    artsMemcpyGpuDb,         artsGetLatestGpuDb,
    artsGetRandomGpuDb,      artsGetNonZerosUnsignedInt,
    artsGetMinDbUnsignedInt, artsAddDbUnsignedInt,
    artsXorDbUint64};

artsLCSyncFunctionGpu_t lcSyncFunctionGpu[] = {
    artsCopyGpuDb,           artsCopyGpuDb,
    artsCopyGpuDb,           artsNonZeroGpuDbUnsignedInt,
    artsMinGpuDbUnsignedInt, artsAddGpuDbUnsignedInt,
    artsXorGpuDbUint64};

unsigned int lcSyncElementSize[] = {sizeof(unsigned int), sizeof(unsigned int),
                                    sizeof(unsigned int), sizeof(unsigned int),
                                    sizeof(unsigned int), sizeof(unsigned int),
                                    sizeof(uint64_t)};

void internalLCSyncGPU(artsGuid_t acqGuid, struct artsDb *db) {
  if (db) {
    artsLCMeta_t host;
    artsLCMeta_t dev;
    host.guid = acqGuid;
    host.data = (void *)(db + 1);
    host.dataSize = db->header.size - sizeof(struct artsDb);
    host.hostVersion = &db->version;
    host.hostTimeStamp = &db->timeStamp;
    host.gpuVersion = 0;
    host.gpuTimeStamp = 0;
    host.gpu = -1;
    host.readLock = &db->reader;
    host.writeLock = &db->writer;

    artsCudaSetDevice(-1, true);

    bool copyOnly = false;
    unsigned int size = db->header.size;
    struct artsDb *tempSpace = (struct artsDb *)artsMallocAlign(size, 16);

    gpuGCWriteLock(); // Don't let the gc take our copies...
    ARTS_DEBUG("FUNCTION: %u\n", artsNodeInfo.gpuLCSync);
    unsigned int remMask = gpuLCReduce(
        acqGuid, db, lcSyncFunctionGpu[artsNodeInfo.gpuLCSync], &copyOnly);
    ARTS_DEBUG("RemMask: %u\n", remMask);
    for (int i = 0; i < artsNodeInfo.gpu; i++) {
      if (remMask & (1 << i)) {
        ARTS_DEBUG("Merging: %u\n", i);
        unsigned int gpuVersion;
        unsigned int timeStamp;
        void *dataPtr = artsGpuRouteTableLookupDbRes(acqGuid, i, &gpuVersion,
                                                     &timeStamp, false);
        if (dataPtr) {
          if (!copyOnly)
            artsGpuInvalidateOnRouteTable(acqGuid, i);

          artsCudaSetDevice(i, false);
          getDataFromStreamNow(i, tempSpace, dataPtr, size, false);
          artsGpuRouteTableReturnDb(acqGuid, !copyOnly, i);

          dev.guid = acqGuid;
          dev.data = (void *)(tempSpace + 1);
          dev.dataSize = tempSpace->header.size - sizeof(struct artsDb);
          dev.hostVersion = &tempSpace->version;
          dev.hostTimeStamp = &tempSpace->timeStamp;
          dev.gpuVersion = gpuVersion;
          dev.gpuTimeStamp = timeStamp;
          dev.gpu = i;
          dev.readLock = NULL;
          dev.writeLock = NULL;
          if (copyOnly)
            lcSyncFunction[0](&host, &dev);
          else
            lcSyncFunction[artsNodeInfo.gpuLCSync](&host, &dev);

          artsMetricsTriggerEvent(artsGpuSync, artsThread, 1);
        }
      } else {
        ARTS_DEBUG("NO DB COPY ON GPU %d\n", i);
      }
    }
    gpuGCWriteUnlock();
    artsFree(tempSpace);
    artsCudaRestoreDevice();
  }
}

void internalLCSyncCPU(artsGuid_t acqGuid, struct artsDb *db) {

  // struct artsDb * db = (struct artsDb*) artsRouteTableLookupItem(acqGuid);
  if (db) {
    artsLCMeta_t host;
    artsLCMeta_t dev;
    host.guid = acqGuid;
    host.data = (void *)(db + 1);
    host.dataSize = db->header.size - sizeof(struct artsDb);
    host.hostVersion = &db->version;
    host.hostTimeStamp = &db->timeStamp;
    host.gpuVersion = 0;
    host.gpuTimeStamp = 0;
    host.gpu = -1;
    host.readLock = &db->reader;
    host.writeLock = &db->writer;

    // artsCudaSetDevice(-1, true);

    unsigned int size = db->header.size;
    struct artsDb *tempSpace = (struct artsDb *)artsMallocAlign(size, 16);
    gpuGCWriteLock(); // Don't let the gc take our copies...
    for (int i = 0; i < artsNodeInfo.gpu; i++) {
      unsigned int gpuVersion;
      unsigned int timeStamp;
      ARTS_DEBUG("acqGuid: %lu type: %u i: %u\n", acqGuid,
                 artsGuidGetType(acqGuid), i);
      void *dataPtr =
          artsGpuRouteTableLookupDb(acqGuid, i, &gpuVersion, &timeStamp);
      if (dataPtr) {
        ARTS_DEBUG("i: %u %lu\n", i, acqGuid);
        artsGpuInvalidateOnRouteTable(acqGuid, i);
        // artsCudaSetDevice(i, false);

        // artsCudaMemCpyFromDev(tempSpace, dataPtr, size);
        getDataFromStreamNow(i, tempSpace, dataPtr, size, false);
        artsGpuRouteTableReturnDb(acqGuid, true, i);

        dev.guid = acqGuid;
        dev.data = (void *)(tempSpace + 1);
        dev.dataSize = tempSpace->header.size - sizeof(struct artsDb);
        dev.hostVersion = &tempSpace->version;
        dev.hostTimeStamp = &tempSpace->timeStamp;
        dev.gpuVersion = gpuVersion;
        dev.gpuTimeStamp = timeStamp;
        dev.gpu = i;
        dev.readLock = NULL;
        dev.writeLock = NULL;
        lcSyncFunction[artsNodeInfo.gpuLCSync](&host, &dev);
      } else {
        ARTS_DEBUG("NO DB COPY ON GPU %d\n", i);
      }
    }
    gpuGCWriteUnlock();
    artsFree(tempSpace);
    // artsCudaRestoreDevice();
  }
}
