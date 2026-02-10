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
#include "arts/gpu/gpu_stream_buffer.h"

#include "arts/gpu/gpu_runtime.cuh"
#include "arts/introspection/metrics.h"
#include "arts/runtime/globals.h"
#include "arts/system/arts_print.h"
#include "arts/utils/atomics.h"

#define CHECKSTREAM 4096
#define MAXSTREAM 32
#define MAXBUFFER 128

volatile unsigned int streamCheckCount[MAXSTREAM] = {0};

volatile unsigned int buffLock[MAXSTREAM] = {0};
unsigned int hostToDevCount[MAXSTREAM] = {0};
unsigned int kernelToDevCount[MAXSTREAM] = {0};
unsigned int devToHostCount[MAXSTREAM] = {0};
unsigned int wrapUpCount[MAXSTREAM] = {0};

arts_buffer_mem_move_t hostToDevBuff[MAXSTREAM][MAXBUFFER];
arts_buffer_kernel_t kernelToDevBuff[MAXSTREAM][MAXBUFFER];
arts_buffer_mem_move_t devToHostBuff[MAXSTREAM][MAXBUFFER];
void *wrapUpBuff[MAXSTREAM][MAXBUFFER];

void checkOccupancy(arts_edt_t fn_ptr, unsigned int gpu_id, dim3 block) {
  int maxActiveBlocks;
  int block_size = (int)block.x * block.y * block.z;
  struct cudaDeviceProp prop = arts_gpus[gpu_id].prop;

  CHECKCORRECT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, (const void *)fn_ptr, (int)block_size, 0));
  float occupancy = (maxActiveBlocks * block_size / prop.warpSize) /
                    (float)(prop.maxThreadsPerMultiProcessor / prop.warpSize);

  // Moving average of occupancy
  arts_lock(&arts_gpus[gpu_id].deviceLock);
  arts_gpus[gpu_id].occupancy = (occupancy + (arts_gpus[gpu_id].totalEdts - 1) *
                                               arts_gpus[gpu_id].occupancy) /
                              (++arts_gpus[gpu_id].totalEdts);
  arts_unlock(&arts_gpus[gpu_id].deviceLock);
}

bool push_data_to_stream(unsigned int gpu_id, void *dst, void *src, size_t count,
                      bool buff) {
  if (buff) {
    arts_lock(&buffLock[gpu_id]);
    hostToDevBuff[gpu_id][hostToDevCount[gpu_id]].dst = dst;
    hostToDevBuff[gpu_id][hostToDevCount[gpu_id]].src = src;
    hostToDevBuff[gpu_id][hostToDevCount[gpu_id]].count = count;
    hostToDevCount[gpu_id]++;

    bool ret = false;
    if (hostToDevCount[gpu_id] == MAXBUFFER)
      ret = flush_stream(gpu_id);
    arts_unlock(&buffLock[gpu_id]);
    return ret;
  }

  if (src) {
    CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice,
                                 arts_gpus[gpu_id].stream));
    ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_BW_PUSH, ARTS_METRIC_THREAD, count);
  } else
    CHECKCORRECT(cudaMemsetAsync(dst, 0, count, arts_gpus[gpu_id].stream));
  return true;
}

bool get_data_from_stream(unsigned int gpu_id, void *dst, void *src, size_t count,
                       bool buff) {
  if (buff) {
    arts_lock(&buffLock[gpu_id]);
    devToHostBuff[gpu_id][devToHostCount[gpu_id]].dst = dst;
    devToHostBuff[gpu_id][devToHostCount[gpu_id]].src = src;
    devToHostBuff[gpu_id][devToHostCount[gpu_id]].count = count;
    devToHostCount[gpu_id]++;

    bool ret = false;
    if (devToHostCount[gpu_id] == MAXBUFFER)
      ret = flush_stream(gpu_id);
    arts_unlock(&buffLock[gpu_id]);
    return ret;
  }
  CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                               arts_gpus[gpu_id].stream));
  ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_BW_PULL, ARTS_METRIC_THREAD, count);
  return true;
}

bool push_kernel_to_stream(unsigned int gpu_id, uint32_t paramc, const uint64_t *paramv,
                        uint32_t depc, arts_edt_dep_t *depv, arts_edt_t fn_ptr,
                        dim3 grid, dim3 block, bool buff) {
  if (buff) {
    arts_lock(&buffLock[gpu_id]);
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].paramc = paramc;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].paramv = paramv;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].depc = depc;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].depv = depv;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].fn_ptr = fn_ptr;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].grid[0] = grid.x;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].grid[1] = grid.y;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].grid[2] = grid.z;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].block[0] = block.x;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].block[1] = block.y;
    kernelToDevBuff[gpu_id][kernelToDevCount[gpu_id]].block[2] = block.z;
    kernelToDevCount[gpu_id]++;

    bool ret = false;
    if (kernelToDevCount[gpu_id] == MAXBUFFER)
      ret = flush_stream(gpu_id);
    arts_unlock(&buffLock[gpu_id]);
    return ret;
  }

  void *kernelArgs[] = {&paramc, &paramv, &depc, &depv};
  CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                (void **)kernelArgs, (size_t)0,
                                arts_gpus[gpu_id].stream));
  checkOccupancy(fn_ptr, gpu_id, block);
  ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_EDT, ARTS_METRIC_THREAD, 1);
  return true;
}

bool push_wrap_up_to_stream(unsigned int gpu_id, void *host_closure, bool buff) {
  if (buff) {
    arts_lock(&buffLock[gpu_id]);
    wrapUpBuff[gpu_id][wrapUpCount[gpu_id]] = host_closure;
    wrapUpCount[gpu_id]++;

    bool ret = false;
    if (wrapUpCount[gpu_id] == MAXBUFFER)
      ret = flush_stream(gpu_id);
    arts_unlock(&buffLock[gpu_id]);
    return ret;
  }

#if CUDART_VERSION >= 10000
  CHECKCORRECT(cudaLaunchHostFunc(arts_gpus[gpu_id].stream, artsWrapUpHostFunc,
                                  host_closure));
#else
  CHECKCORRECT(cudaStreamAddCallback(arts_gpus[gpu_id].stream, artsWrapUp,
                                     host_closure, 0));
#endif
  return true;
}

bool flush_mem_stream(unsigned int gpu_id, unsigned int *count,
                    arts_buffer_mem_move_t *buff, enum cudaMemcpyKind kind) {
  unsigned int max = *count;
  if (max > 0) {
    uint64_t dataSize = 0;
    for (unsigned int i = 0; i < max; i++) {
      if (buff[i].src) {
        // ARTS_INFO("i: %u %p %p %u %p\n", i, buff[i].dst, buff[i].src,
        // buff[i].count,  &arts_gpus[gpu_id].stream);
        CHECKCORRECT(cudaMemcpyAsync(buff[i].dst, buff[i].src, buff[i].count,
                                     kind, arts_gpus[gpu_id].stream));
        dataSize += buff[i].count;
      } else
        CHECKCORRECT(cudaMemsetAsync(buff[i].dst, 0, buff[i].count,
                                     arts_gpus[gpu_id].stream));
    }
    *count = 0;
    if (kind == cudaMemcpyHostToDevice) {
      ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_BW_PUSH, ARTS_METRIC_THREAD, dataSize);
    } else {
      ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_BW_PULL, ARTS_METRIC_THREAD, dataSize);
    }
    return true;
  }
  return false;
}

bool flush_kernel_stream(unsigned int gpu_id) {
  bool ret = (kernelToDevCount[gpu_id] > 0);
  if (ret) {
    for (unsigned int i = 0; i < kernelToDevCount[gpu_id]; i++) {
      void *kernelArgs[] = {
          &kernelToDevBuff[gpu_id][i].paramc, &kernelToDevBuff[gpu_id][i].paramv,
          &kernelToDevBuff[gpu_id][i].depc, &kernelToDevBuff[gpu_id][i].depv};
      dim3 grid(kernelToDevBuff[gpu_id][i].grid[0],
                kernelToDevBuff[gpu_id][i].grid[1],
                kernelToDevBuff[gpu_id][i].grid[2]);
      dim3 block(kernelToDevBuff[gpu_id][i].block[0],
                 kernelToDevBuff[gpu_id][i].block[1],
                 kernelToDevBuff[gpu_id][i].block[2]);
      CHECKCORRECT(cudaLaunchKernel(
          (const void *)kernelToDevBuff[gpu_id][i].fn_ptr, grid, block,
          (void **)kernelArgs, (size_t)0, arts_gpus[gpu_id].stream));
      checkOccupancy(kernelToDevBuff[gpu_id][i].fn_ptr, gpu_id, block);
    }
    ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_EDT, ARTS_METRIC_THREAD, kernelToDevCount[gpu_id]);
    kernelToDevCount[gpu_id] = 0;
  }
  return ret;
}

bool flush_wrap_up_stream(unsigned int gpu_id) {
  bool ret = (wrapUpCount[gpu_id] > 0);
  for (unsigned int i = 0; i < wrapUpCount[gpu_id]; i++) {
#if CUDART_VERSION >= 10000
    CHECKCORRECT(cudaLaunchHostFunc(arts_gpus[gpu_id].stream, artsWrapUpHostFunc,
                                    wrapUpBuff[gpu_id][i]));
#else
    CHECKCORRECT(cudaStreamAddCallback(arts_gpus[gpu_id].stream, artsWrapUp,
                                       wrapUpBuff[gpu_id][i], 0));
#endif
  }
  wrapUpCount[gpu_id] = 0;
  return ret;
}

bool flush_stream(unsigned int gpu_id) {
  ARTS_DEBUG("%u %u %u %u\n", hostToDevCount[gpu_id], kernelToDevCount[gpu_id],
             devToHostCount[gpu_id], wrapUpCount[gpu_id]);
  if (hostToDevCount[gpu_id] || kernelToDevCount[gpu_id] ||
      devToHostCount[gpu_id] || wrapUpCount[gpu_id]) {
    arts_cuda_set_device(gpu_id, true);

    flush_mem_stream(gpu_id, &hostToDevCount[gpu_id], hostToDevBuff[gpu_id],
                   cudaMemcpyHostToDevice);
    flush_kernel_stream(gpu_id);
    flush_mem_stream(gpu_id, &devToHostCount[gpu_id], devToHostBuff[gpu_id],
                   cudaMemcpyDeviceToHost);
    flush_wrap_up_stream(gpu_id);

    arts_cuda_restore_device();
    ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_BUFFER_FLUSH, ARTS_METRIC_THREAD, 1);
    return true;
  }
  return false;
}

void copy_gputo_gpu(void *dst, unsigned int dst_gpu_id, void *src,
                  unsigned int src_gpu_id, unsigned int size) {
  // We need to lock in a fixed order, so smallest first
  unsigned int first = (dst_gpu_id < src_gpu_id) ? dst_gpu_id : src_gpu_id;
  unsigned int second = (dst_gpu_id == first) ? src_gpu_id : dst_gpu_id;
  arts_lock(&buffLock[first]);
  arts_lock(&buffLock[second]);

  // Flush the streams to make sure everything is done
  flush_stream(dst_gpu_id);
  flush_stream(src_gpu_id);
  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[src_gpu_id].stream));

  // Next lets move the data
  CHECKCORRECT(cudaMemcpyPeerAsync(dst, dst_gpu_id, src, src_gpu_id, size,
                                   arts_gpus[dst_gpu_id].stream));

  arts_cuda_restore_device();

  // Unlock in the correct order
  arts_unlock(&buffLock[second]);
  arts_unlock(&buffLock[first]);
}

void do_reduction_now(unsigned int gpu_id, void *sink, void *src,
                    artsLCSyncFunctionGpu_t fn_ptr, unsigned int element_size,
                    unsigned int size) {
  arts_lock(&buffLock[gpu_id]);
  flush_stream(gpu_id);

  arts_cuda_set_device(gpu_id, true);
  size -= sizeof(struct arts_db);

  // Next lets run the reduce function on the db_data and the shadow copy (dst)
  unsigned int tile_size = size / element_size;
  ARTS_INFO("TileSize: %u\n", tile_size);
  if (tile_size < 32) {
    dim3 block(tile_size, 1, 1); // For volta...
    dim3 grid(1, 1, 1);
    void *kernelArgs[] = {&sink, &src};
    ARTS_INFO("SRC: %p DST: %p\n", sink, src);
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernelArgs, (size_t)0,
                                  arts_gpus[gpu_id].stream));
  } else {
    dim3 block(32, 1, 1); // For volta...
    dim3 grid((tile_size + 32 - 1) / 32, 1, 1);
    void *kernelArgs[] = {&sink, &src};
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernelArgs, (size_t)0,
                                  arts_gpus[gpu_id].stream))
  }

  arts_cuda_restore_device();
  arts_unlock(&buffLock[gpu_id]);
}

void reduce_datafrom_gpus(void *dst, unsigned int dst_gpu_id, void *src,
                        unsigned int src_gpu_id, unsigned int size,
                        artsLCSyncFunctionGpu_t fn_ptr, unsigned int element_size,
                        void *db_data) {
  ARTS_DEBUG("ELEMENT SIZE: %lu\n", element_size);
  // We need to lock in a fixed order, so smallest first
  unsigned int first = (dst_gpu_id < src_gpu_id) ? dst_gpu_id : src_gpu_id;
  unsigned int second = (dst_gpu_id == first) ? src_gpu_id : dst_gpu_id;
  arts_lock(&buffLock[first]);
  arts_lock(&buffLock[second]);

  // Flush the streams to make sure everything is done
  flush_stream(dst_gpu_id);
  flush_stream(src_gpu_id);
  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[src_gpu_id].stream));
  // I think we don't need to synchronize the destination stream since we are
  // just adding to it...
  //  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[dst_gpu_id].stream));

  // Next lets move the data
  CHECKCORRECT(cudaMemcpyPeerAsync(dst, dst_gpu_id, src, src_gpu_id, size,
                                   arts_gpus[dst_gpu_id].stream));

  arts_cuda_set_device(dst_gpu_id, true);

  // Lets remove the db header part
  size -= sizeof(struct arts_db);

  // Next lets run the reduce function on the db_data and the shadow copy (dst)
  unsigned int tile_size = size / element_size;
  ARTS_DEBUG("TileSize: %u\n", tile_size);
  if (tile_size < 32) {
    dim3 block(tile_size, 1, 1); // For volta...
    dim3 grid(1, 1, 1);
    void *kernelArgs[] = {&db_data, &dst};
    ARTS_DEBUG("SRC: %p DST: %p\n", db_data, dst);
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernelArgs, (size_t)0,
                                  arts_gpus[dst_gpu_id].stream));
  } else {
    dim3 block(32, 1, 1); // For volta...
    dim3 grid((tile_size + 32 - 1) / 32, 1, 1);
    void *kernelArgs[] = {&db_data, &dst};
    ARTS_DEBUG("SRC: %p DST: %p\n", db_data, dst);
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernelArgs, (size_t)0,
                                  arts_gpus[dst_gpu_id].stream))
  }

  // CHECKCORRECT(cudaStreamSynchronize(arts_gpus[src_gpu_id].stream));
  // CHECKCORRECT(cudaStreamSynchronize(arts_gpus[dst_gpu_id].stream));
  arts_cuda_restore_device();

  // Unlock in the correct order
  arts_unlock(&buffLock[second]);
  arts_unlock(&buffLock[first]);
}

void get_data_from_stream_now(unsigned int gpu_id, void *dst, void *src,
                          size_t count, bool buff) {
  if (buff) {
    arts_lock(&buffLock[gpu_id]);
    flush_stream(gpu_id);
    arts_unlock(&buffLock[gpu_id]);
  }
  ARTS_DEBUG("GETTING[%u]: %p %p size: %u\n", gpu_id, dst, src, count);
  CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                               arts_gpus[gpu_id].stream));
  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[gpu_id].stream));
}

bool check_streams(bool buff_on) {
  if (buff_on) {
    bool ret = false;
    for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
      if (hostToDevCount[i] || kernelToDevCount[i] || devToHostCount[i] ||
          wrapUpCount[i]) {
        arts_atomic_fetch_add(&streamCheckCount[i], 1U);
        if (streamCheckCount[i] % CHECKSTREAM == 0) {
          arts_lock(&buffLock[i]);
          ret |= flush_stream(i);
          arts_unlock(&buffLock[i]);
        }
      }
    }
    return ret;
  }
  return false;
}