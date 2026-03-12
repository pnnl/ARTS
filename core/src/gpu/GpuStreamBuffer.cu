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
#include "arts/gpu/GpuStreamBuffer.h"

#include "arts/gpu/GpuRuntime.cuh"
#include "arts/introspection/Metrics.h"
#include "arts/runtime/Globals.h"
#include "arts/system/ArtsPrint.h"
#include "arts/utils/Atomics.h"

#define CHECKSTREAM 4096
#define MAXSTREAM 32
#define MAXBUFFER 128

volatile unsigned int streamCheckCount[MAXSTREAM] = {0};

volatile unsigned int buffLock[MAXSTREAM] = {0};
unsigned int hostToDevCount[MAXSTREAM] = {0};
unsigned int kernelToDevCount[MAXSTREAM] = {0};
unsigned int devToHostCount[MAXSTREAM] = {0};
unsigned int wrapUpCount[MAXSTREAM] = {0};

artsBufferMemMove_t hostToDevBuff[MAXSTREAM][MAXBUFFER];
artsBufferKernel_t kernelToDevBuff[MAXSTREAM][MAXBUFFER];
artsBufferMemMove_t devToHostBuff[MAXSTREAM][MAXBUFFER];
void *wrapUpBuff[MAXSTREAM][MAXBUFFER];

void checkOccupancy(artsEdt_t fnPtr, unsigned int gpuId, dim3 block) {
  int maxActiveBlocks;
  int blockSize = (int)block.x * block.y * block.z;
  struct cudaDeviceProp prop = artsGpus[gpuId].prop;

  CHECKCORRECT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, (const void *)fnPtr, (int)blockSize, 0));
  float occupancy = (maxActiveBlocks * blockSize / prop.warpSize) /
                    (float)(prop.maxThreadsPerMultiProcessor / prop.warpSize);

  // Moving average of occupancy
  artsLock(&artsGpus[gpuId].deviceLock);
  artsGpus[gpuId].occupancy = (occupancy + (artsGpus[gpuId].totalEdts - 1) *
                                               artsGpus[gpuId].occupancy) /
                              (++artsGpus[gpuId].totalEdts);
  artsUnlock(&artsGpus[gpuId].deviceLock);
}

bool pushDataToStream(unsigned int gpuId, void *dst, void *src, size_t count,
                      bool buff) {
  if (buff) {
    artsLock(&buffLock[gpuId]);
    hostToDevBuff[gpuId][hostToDevCount[gpuId]].dst = dst;
    hostToDevBuff[gpuId][hostToDevCount[gpuId]].src = src;
    hostToDevBuff[gpuId][hostToDevCount[gpuId]].count = count;
    hostToDevCount[gpuId]++;

    bool ret = false;
    if (hostToDevCount[gpuId] == MAXBUFFER)
      ret = flushStream(gpuId);
    artsUnlock(&buffLock[gpuId]);
    return ret;
  }

  if (src) {
    CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice,
                                 artsGpus[gpuId].stream));
    artsMetricsTriggerEvent(artsGpuBWPush, artsThread, count);
  } else
    CHECKCORRECT(cudaMemsetAsync(dst, 0, count, artsGpus[gpuId].stream));
  return true;
}

bool getDataFromStream(unsigned int gpuId, void *dst, void *src, size_t count,
                       bool buff) {
  if (buff) {
    artsLock(&buffLock[gpuId]);
    devToHostBuff[gpuId][devToHostCount[gpuId]].dst = dst;
    devToHostBuff[gpuId][devToHostCount[gpuId]].src = src;
    devToHostBuff[gpuId][devToHostCount[gpuId]].count = count;
    devToHostCount[gpuId]++;

    bool ret = false;
    if (devToHostCount[gpuId] == MAXBUFFER)
      ret = flushStream(gpuId);
    artsUnlock(&buffLock[gpuId]);
    return ret;
  }
  CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                               artsGpus[gpuId].stream));
  artsMetricsTriggerEvent(artsGpuBWPull, artsThread, count);
  return true;
}

bool pushKernelToStream(unsigned int gpuId, uint32_t paramc, uint64_t *paramv,
                        uint32_t depc, artsEdtDep_t *depv, artsEdt_t fnPtr,
                        dim3 grid, dim3 block, bool buff) {
  if (buff) {
    artsLock(&buffLock[gpuId]);
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].paramc = paramc;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].paramv = paramv;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].depc = depc;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].depv = depv;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].fnPtr = fnPtr;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].grid[0] = grid.x;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].grid[1] = grid.y;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].grid[2] = grid.z;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].block[0] = block.x;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].block[1] = block.y;
    kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].block[2] = block.z;
    kernelToDevCount[gpuId]++;

    bool ret = false;
    if (kernelToDevCount[gpuId] == MAXBUFFER)
      ret = flushStream(gpuId);
    artsUnlock(&buffLock[gpuId]);
    return ret;
  }

  void *kernelArgs[] = {&paramc, &paramv, &depc, &depv};
  CHECKCORRECT(cudaLaunchKernel((const void *)fnPtr, grid, block,
                                (void **)kernelArgs, (size_t)0,
                                artsGpus[gpuId].stream));
  checkOccupancy(fnPtr, gpuId, block);
  artsMetricsTriggerEvent(artsGpuEdt, artsThread, 1);
  return true;
}

bool pushWrapUpToStream(unsigned int gpuId, void *hostClosure, bool buff) {
  if (buff) {
    artsLock(&buffLock[gpuId]);
    wrapUpBuff[gpuId][wrapUpCount[gpuId]] = hostClosure;
    wrapUpCount[gpuId]++;

    bool ret = false;
    if (wrapUpCount[gpuId] == MAXBUFFER)
      ret = flushStream(gpuId);
    artsUnlock(&buffLock[gpuId]);
    return ret;
  }

#if CUDART_VERSION >= 10000
  CHECKCORRECT(cudaLaunchHostFunc(artsGpus[gpuId].stream, artsWrapUpHostFunc,
                                  hostClosure));
#else
  CHECKCORRECT(cudaStreamAddCallback(artsGpus[gpuId].stream, artsWrapUp,
                                     hostClosure, 0));
#endif
  return true;
}

bool flushMemStream(unsigned int gpuId, unsigned int *count,
                    artsBufferMemMove_t *buff, enum cudaMemcpyKind kind) {
  unsigned int max = *count;
  if (max > 0) {
    uint64_t dataSize = 0;
    for (unsigned int i = 0; i < max; i++) {
      if (buff[i].src) {
        // ARTS_INFO("i: %u %p %p %u %p\n", i, buff[i].dst, buff[i].src,
        // buff[i].count,  &artsGpus[gpuId].stream);
        CHECKCORRECT(cudaMemcpyAsync(buff[i].dst, buff[i].src, buff[i].count,
                                     kind, artsGpus[gpuId].stream));
        dataSize += buff[i].count;
      } else
        CHECKCORRECT(cudaMemsetAsync(buff[i].dst, 0, buff[i].count,
                                     artsGpus[gpuId].stream));
    }
    *count = 0;
    if (kind == cudaMemcpyHostToDevice) {
      artsMetricsTriggerEvent(artsGpuBWPush, artsThread, dataSize);
    } else {
      artsMetricsTriggerEvent(artsGpuBWPull, artsThread, dataSize);
    }
    return true;
  }
  return false;
}

bool flushKernelStream(unsigned int gpuId) {
  bool ret = (kernelToDevCount[gpuId] > 0);
  if (ret) {
    for (unsigned int i = 0; i < kernelToDevCount[gpuId]; i++) {
      void *kernelArgs[] = {
          &kernelToDevBuff[gpuId][i].paramc, &kernelToDevBuff[gpuId][i].paramv,
          &kernelToDevBuff[gpuId][i].depc, &kernelToDevBuff[gpuId][i].depv};
      dim3 grid(kernelToDevBuff[gpuId][i].grid[0],
                kernelToDevBuff[gpuId][i].grid[1],
                kernelToDevBuff[gpuId][i].grid[2]);
      dim3 block(kernelToDevBuff[gpuId][i].block[0],
                 kernelToDevBuff[gpuId][i].block[1],
                 kernelToDevBuff[gpuId][i].block[2]);
      CHECKCORRECT(cudaLaunchKernel(
          (const void *)kernelToDevBuff[gpuId][i].fnPtr, grid, block,
          (void **)kernelArgs, (size_t)0, artsGpus[gpuId].stream));
      checkOccupancy(kernelToDevBuff[gpuId][i].fnPtr, gpuId, block);
    }
    artsMetricsTriggerEvent(artsGpuEdt, artsThread, kernelToDevCount[gpuId]);
    kernelToDevCount[gpuId] = 0;
  }
  return ret;
}

bool flushWrapUpStream(unsigned int gpuId) {
  bool ret = (wrapUpCount[gpuId] > 0);
  for (unsigned int i = 0; i < wrapUpCount[gpuId]; i++) {
#if CUDART_VERSION >= 10000
    CHECKCORRECT(cudaLaunchHostFunc(artsGpus[gpuId].stream, artsWrapUpHostFunc,
                                    wrapUpBuff[gpuId][i]));
#else
    CHECKCORRECT(cudaStreamAddCallback(artsGpus[gpuId].stream, artsWrapUp,
                                       wrapUpBuff[gpuId][i], 0));
#endif
  }
  wrapUpCount[gpuId] = 0;
  return ret;
}

bool flushStream(unsigned int gpuId) {
  ARTS_DEBUG("%u %u %u %u\n", hostToDevCount[gpuId], kernelToDevCount[gpuId],
             devToHostCount[gpuId], wrapUpCount[gpuId]);
  if (hostToDevCount[gpuId] || kernelToDevCount[gpuId] ||
      devToHostCount[gpuId] || wrapUpCount[gpuId]) {
    artsCudaSetDevice(gpuId, true);

    flushMemStream(gpuId, &hostToDevCount[gpuId], hostToDevBuff[gpuId],
                   cudaMemcpyHostToDevice);
    flushKernelStream(gpuId);
    flushMemStream(gpuId, &devToHostCount[gpuId], devToHostBuff[gpuId],
                   cudaMemcpyDeviceToHost);
    flushWrapUpStream(gpuId);

    artsCudaRestoreDevice();
    artsMetricsTriggerEvent(artsGpuBufferFlush, artsThread, 1);
    return true;
  }
  return false;
}

void copyGputoGpu(void *dst, unsigned int dstGpuId, void *src,
                  unsigned int srcGpuId, unsigned int size) {
  // We need to lock in a fixed order, so smallest first
  unsigned int first = (dstGpuId < srcGpuId) ? dstGpuId : srcGpuId;
  unsigned int second = (dstGpuId == first) ? srcGpuId : dstGpuId;
  artsLock(&buffLock[first]);
  artsLock(&buffLock[second]);

  // Flush the streams to make sure everything is done
  flushStream(dstGpuId);
  flushStream(srcGpuId);
  CHECKCORRECT(cudaStreamSynchronize(artsGpus[srcGpuId].stream));

  // Next lets move the data
  CHECKCORRECT(cudaMemcpyPeerAsync(dst, dstGpuId, src, srcGpuId, size,
                                   artsGpus[dstGpuId].stream));

  artsCudaRestoreDevice();

  // Unlock in the correct order
  artsUnlock(&buffLock[second]);
  artsUnlock(&buffLock[first]);
}

void doReductionNow(unsigned int gpuId, void *sink, void *src,
                    artsLCSyncFunctionGpu_t fnPtr, unsigned int elementSize,
                    unsigned int size) {
  artsLock(&buffLock[gpuId]);
  flushStream(gpuId);

  artsCudaSetDevice(gpuId, true);
  size -= sizeof(struct artsDb);

  // Next lets run the reduce function on the dbData and the shadow copy (dst)
  unsigned int tileSize = size / elementSize;
  ARTS_INFO("TileSize: %u\n", tileSize);
  if (tileSize < 32) {
    dim3 block(tileSize, 1, 1); // For volta...
    dim3 grid(1, 1, 1);
    void *kernelArgs[] = {&sink, &src};
    ARTS_INFO("SRC: %p DST: %p\n", sink, src);
    CHECKCORRECT(cudaLaunchKernel((const void *)fnPtr, grid, block,
                                  (void **)kernelArgs, (size_t)0,
                                  artsGpus[gpuId].stream));
  } else {
    dim3 block(32, 1, 1); // For volta...
    dim3 grid((tileSize + 32 - 1) / 32, 1, 1);
    void *kernelArgs[] = {&sink, &src};
    CHECKCORRECT(cudaLaunchKernel((const void *)fnPtr, grid, block,
                                  (void **)kernelArgs, (size_t)0,
                                  artsGpus[gpuId].stream))
  }

  artsCudaRestoreDevice();
  artsUnlock(&buffLock[gpuId]);
}

void reduceDatafromGpus(void *dst, unsigned int dstGpuId, void *src,
                        unsigned int srcGpuId, unsigned int size,
                        artsLCSyncFunctionGpu_t fnPtr, unsigned int elementSize,
                        void *dbData) {
  ARTS_DEBUG("ELEMENT SIZE: %lu\n", elementSize);
  // We need to lock in a fixed order, so smallest first
  unsigned int first = (dstGpuId < srcGpuId) ? dstGpuId : srcGpuId;
  unsigned int second = (dstGpuId == first) ? srcGpuId : dstGpuId;
  artsLock(&buffLock[first]);
  artsLock(&buffLock[second]);

  // Flush the streams to make sure everything is done
  flushStream(dstGpuId);
  flushStream(srcGpuId);
  CHECKCORRECT(cudaStreamSynchronize(artsGpus[srcGpuId].stream));
  // I think we don't need to synchronize the destination stream since we are
  // just adding to it...
  //  CHECKCORRECT(cudaStreamSynchronize(artsGpus[dstGpuId].stream));

  // Next lets move the data
  CHECKCORRECT(cudaMemcpyPeerAsync(dst, dstGpuId, src, srcGpuId, size,
                                   artsGpus[dstGpuId].stream));

  artsCudaSetDevice(dstGpuId, true);

  // Lets remove the db header part
  size -= sizeof(struct artsDb);

  // Next lets run the reduce function on the dbData and the shadow copy (dst)
  unsigned int tileSize = size / elementSize;
  ARTS_DEBUG("TileSize: %u\n", tileSize);
  if (tileSize < 32) {
    dim3 block(tileSize, 1, 1); // For volta...
    dim3 grid(1, 1, 1);
    void *kernelArgs[] = {&dbData, &dst};
    ARTS_DEBUG("SRC: %p DST: %p\n", dbData, dst);
    CHECKCORRECT(cudaLaunchKernel((const void *)fnPtr, grid, block,
                                  (void **)kernelArgs, (size_t)0,
                                  artsGpus[dstGpuId].stream));
  } else {
    dim3 block(32, 1, 1); // For volta...
    dim3 grid((tileSize + 32 - 1) / 32, 1, 1);
    void *kernelArgs[] = {&dbData, &dst};
    ARTS_DEBUG("SRC: %p DST: %p\n", dbData, dst);
    CHECKCORRECT(cudaLaunchKernel((const void *)fnPtr, grid, block,
                                  (void **)kernelArgs, (size_t)0,
                                  artsGpus[dstGpuId].stream))
  }

  // CHECKCORRECT(cudaStreamSynchronize(artsGpus[srcGpuId].stream));
  // CHECKCORRECT(cudaStreamSynchronize(artsGpus[dstGpuId].stream));
  artsCudaRestoreDevice();

  // Unlock in the correct order
  artsUnlock(&buffLock[second]);
  artsUnlock(&buffLock[first]);
}

void getDataFromStreamNow(unsigned int gpuId, void *dst, void *src,
                          size_t count, bool buff) {
  if (buff) {
    artsLock(&buffLock[gpuId]);
    flushStream(gpuId);
    artsUnlock(&buffLock[gpuId]);
  }
  ARTS_DEBUG("GETTING[%u]: %p %p size: %u\n", gpuId, dst, src, count);
  CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                               artsGpus[gpuId].stream));
  CHECKCORRECT(cudaStreamSynchronize(artsGpus[gpuId].stream));
}

bool checkStreams(bool buffOn) {
  if (buffOn) {
    bool ret = false;
    for (unsigned int i = 0; i < artsNodeInfo.gpu; i++) {
      if (hostToDevCount[i] || kernelToDevCount[i] || devToHostCount[i] ||
          wrapUpCount[i]) {
        artsAtomicFetchAdd(&streamCheckCount[i], 1U);
        if (streamCheckCount[i] % CHECKSTREAM == 0) {
          artsLock(&buffLock[i]);
          ret |= flushStream(i);
          artsUnlock(&buffLock[i]);
        }
      }
    }
    return ret;
  }
  return false;
}