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
#include "arts/runtime/globals.h"
#include "arts/system/arts_print.h"
#include "arts/utils/atomics.h"

#define CHECKSTREAM 4096
#define MAXSTREAM   32
#define MAXBUFFER   128

volatile unsigned int stream_check_count[MAXSTREAM] = {0};

volatile unsigned int buff_lock[MAXSTREAM] = {0};
unsigned int host_to_dev_count[MAXSTREAM] = {0};
unsigned int kernel_to_dev_count[MAXSTREAM] = {0};
unsigned int dev_to_host_count[MAXSTREAM] = {0};
unsigned int wrap_up_count[MAXSTREAM] = {0};

arts_buffer_mem_move_t host_to_dev_buff[MAXSTREAM][MAXBUFFER];
arts_buffer_kernel_t kernel_to_dev_buff[MAXSTREAM][MAXBUFFER];
arts_buffer_mem_move_t dev_to_host_buff[MAXSTREAM][MAXBUFFER];
void *wrap_up_buff[MAXSTREAM][MAXBUFFER];

void check_occupancy(arts_edt_t fn_ptr, unsigned int gpu_id, dim3 block) {
  int max_active_blocks;
  int block_size = (int)(block.x * block.y * block.z);
  struct cudaDeviceProp prop = arts_gpus[gpu_id].prop;

  CHECKCORRECT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, (const void *)fn_ptr, block_size, 0));
  float occupancy =
      ((float)(max_active_blocks * block_size) / (float)prop.warpSize) /
      ((float)prop.maxThreadsPerMultiProcessor / (float)prop.warpSize);

  // Cumulative (running) average of occupancy
  arts_lock(&arts_gpus[gpu_id].deviceLock);
  arts_gpus[gpu_id].occupancy =
      (occupancy + ((float)(arts_gpus[gpu_id].totalEdts - 1) *
                    arts_gpus[gpu_id].occupancy)) /
      (float)(++arts_gpus[gpu_id].totalEdts);
  arts_unlock(&arts_gpus[gpu_id].deviceLock);
}

bool push_data_to_stream(unsigned int gpu_id, void *dst, void *src,
                         size_t count, bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    host_to_dev_buff[gpu_id][host_to_dev_count[gpu_id]].dst = dst;
    host_to_dev_buff[gpu_id][host_to_dev_count[gpu_id]].src = src;
    host_to_dev_buff[gpu_id][host_to_dev_count[gpu_id]].count = count;
    host_to_dev_count[gpu_id]++;

    bool ret = false;
    if (host_to_dev_count[gpu_id] == MAXBUFFER) {
      ret = flush_stream(gpu_id);
    }
    arts_unlock(&buff_lock[gpu_id]);
    return ret;
  }

  if (src) {
    CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice,
                                 arts_gpus[gpu_id].stream));
  } else {
    CHECKCORRECT(cudaMemsetAsync(dst, 0, count, arts_gpus[gpu_id].stream));
  }
  return true;
}

bool get_data_from_stream(unsigned int gpu_id, void *dst, void *src,
                          size_t count, bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    dev_to_host_buff[gpu_id][dev_to_host_count[gpu_id]].dst = dst;
    dev_to_host_buff[gpu_id][dev_to_host_count[gpu_id]].src = src;
    dev_to_host_buff[gpu_id][dev_to_host_count[gpu_id]].count = count;
    dev_to_host_count[gpu_id]++;

    bool ret = false;
    if (dev_to_host_count[gpu_id] == MAXBUFFER) {
      ret = flush_stream(gpu_id);
    }
    arts_unlock(&buff_lock[gpu_id]);
    return ret;
  }
  CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                               arts_gpus[gpu_id].stream));
  return true;
}

bool push_kernel_to_stream(unsigned int gpu_id, uint32_t paramc,
                           const uint64_t *paramv, uint32_t depc,
                           arts_edt_dep_t *depv, arts_edt_t fn_ptr, dim3 grid,
                           dim3 block, bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].paramc = paramc;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].paramv = paramv;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].depc = depc;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].depv = depv;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].fn_ptr = fn_ptr;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].grid[0] = grid.x;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].grid[1] = grid.y;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].grid[2] = grid.z;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].block[0] = block.x;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].block[1] = block.y;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].block[2] = block.z;
    kernel_to_dev_count[gpu_id]++;

    bool ret = false;
    if (kernel_to_dev_count[gpu_id] == MAXBUFFER) {
      ret = flush_stream(gpu_id);
    }
    arts_unlock(&buff_lock[gpu_id]);
    return ret;
  }

  void *kernel_args[] = {&paramc, &paramv, &depc, &depv};
  CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                (void **)kernel_args, (size_t)0,
                                arts_gpus[gpu_id].stream));
  check_occupancy(fn_ptr, gpu_id, block);
  return true;
}

bool push_wrap_up_to_stream(unsigned int gpu_id, void *host_closure,
                            bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    wrap_up_buff[gpu_id][wrap_up_count[gpu_id]] = host_closure;
    wrap_up_count[gpu_id]++;

    bool ret = false;
    if (wrap_up_count[gpu_id] == MAXBUFFER) {
      ret = flush_stream(gpu_id);
    }
    arts_unlock(&buff_lock[gpu_id]);
    return ret;
  }

#if CUDART_VERSION >= 10000
  CHECKCORRECT(cudaLaunchHostFunc(arts_gpus[gpu_id].stream,
                                  arts_wrap_up_host_func, host_closure));
#else
  CHECKCORRECT(cudaStreamAddCallback(arts_gpus[gpu_id].stream, arts_wrap_up,
                                     host_closure, 0));
#endif
  return true;
}

bool flush_mem_stream(unsigned int gpu_id, unsigned int *count,
                      arts_buffer_mem_move_t *buff, enum cudaMemcpyKind kind) {
  unsigned int max = *count;
  if (max > 0) {
    uint64_t data_size = 0;
    for (unsigned int i = 0; i < max; i++) {
      if (buff[i].src) {
        // ARTS_INFO("i: %u %p %p %u %p\n", i, buff[i].dst, buff[i].src,
        // buff[i].count,  &arts_gpus[gpu_id].stream);
        CHECKCORRECT(cudaMemcpyAsync(buff[i].dst, buff[i].src, buff[i].count,
                                     kind, arts_gpus[gpu_id].stream));
        data_size += buff[i].count;
      } else {
        CHECKCORRECT(cudaMemsetAsync(buff[i].dst, 0, buff[i].count,
                                     arts_gpus[gpu_id].stream));
      }
    }
    *count = 0;
    return true;
  }
  return false;
}

bool flush_kernel_stream(unsigned int gpu_id) {
  bool ret = (kernel_to_dev_count[gpu_id] > 0);
  if (ret) {
    for (unsigned int i = 0; i < kernel_to_dev_count[gpu_id]; i++) {
      void *kernel_args[] = {&kernel_to_dev_buff[gpu_id][i].paramc,
                             &kernel_to_dev_buff[gpu_id][i].paramv,
                             &kernel_to_dev_buff[gpu_id][i].depc,
                             &kernel_to_dev_buff[gpu_id][i].depv};
      dim3 grid(kernel_to_dev_buff[gpu_id][i].grid[0],
                kernel_to_dev_buff[gpu_id][i].grid[1],
                kernel_to_dev_buff[gpu_id][i].grid[2]);
      dim3 block(kernel_to_dev_buff[gpu_id][i].block[0],
                 kernel_to_dev_buff[gpu_id][i].block[1],
                 kernel_to_dev_buff[gpu_id][i].block[2]);
      CHECKCORRECT(cudaLaunchKernel(
          (const void *)kernel_to_dev_buff[gpu_id][i].fn_ptr, grid, block,
          (void **)kernel_args, (size_t)0, arts_gpus[gpu_id].stream));
      check_occupancy(kernel_to_dev_buff[gpu_id][i].fn_ptr, gpu_id, block);
    }
    kernel_to_dev_count[gpu_id] = 0;
  }
  return ret;
}

bool flush_wrap_up_stream(unsigned int gpu_id) {
  bool ret = (wrap_up_count[gpu_id] > 0);
  for (unsigned int i = 0; i < wrap_up_count[gpu_id]; i++) {
#if CUDART_VERSION >= 10000
    CHECKCORRECT(cudaLaunchHostFunc(arts_gpus[gpu_id].stream,
                                    arts_wrap_up_host_func,
                                    wrap_up_buff[gpu_id][i]));
#else
    CHECKCORRECT(cudaStreamAddCallback(arts_gpus[gpu_id].stream, arts_wrap_up,
                                       wrap_up_buff[gpu_id][i], 0));
#endif
  }
  wrap_up_count[gpu_id] = 0;
  return ret;
}

bool flush_stream(unsigned int gpu_id) {
  ARTS_DEBUG("%u %u %u %u\n", host_to_dev_count[gpu_id],
             kernel_to_dev_count[gpu_id], dev_to_host_count[gpu_id],
             wrap_up_count[gpu_id]);
  if (host_to_dev_count[gpu_id] || kernel_to_dev_count[gpu_id] ||
      dev_to_host_count[gpu_id] || wrap_up_count[gpu_id]) {
    arts_cuda_set_device((int)gpu_id, true);

    flush_mem_stream(gpu_id, &host_to_dev_count[gpu_id],
                     host_to_dev_buff[gpu_id], cudaMemcpyHostToDevice);
    flush_kernel_stream(gpu_id);
    flush_mem_stream(gpu_id, &dev_to_host_count[gpu_id],
                     dev_to_host_buff[gpu_id], cudaMemcpyDeviceToHost);
    flush_wrap_up_stream(gpu_id);

    arts_cuda_restore_device();
    return true;
  }
  return false;
}

void copy_gputo_gpu(void *dst, unsigned int dst_gpu_id, void *src,
                    unsigned int src_gpu_id, unsigned int size) {
  // We need to lock in a fixed order, so smallest first
  unsigned int first = (dst_gpu_id < src_gpu_id) ? dst_gpu_id : src_gpu_id;
  unsigned int second = (dst_gpu_id == first) ? src_gpu_id : dst_gpu_id;
  arts_lock(&buff_lock[first]);
  arts_lock(&buff_lock[second]);

  // Flush the streams to make sure everything is done
  flush_stream(dst_gpu_id);
  flush_stream(src_gpu_id);
  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[src_gpu_id].stream));

  // Next lets move the data
  CHECKCORRECT(cudaMemcpyPeerAsync(dst, dst_gpu_id, src, src_gpu_id, size,
                                   arts_gpus[dst_gpu_id].stream));

  arts_cuda_restore_device();

  // Unlock in the correct order
  arts_unlock(&buff_lock[second]);
  arts_unlock(&buff_lock[first]);
}

void do_reduction_now(unsigned int gpu_id, void *sink, void *src,
                      arts_lc_sync_function_gpu_t fn_ptr,
                      unsigned int element_size, unsigned int size) {
  arts_lock(&buff_lock[gpu_id]);
  flush_stream(gpu_id);

  arts_cuda_set_device((int)gpu_id, true);
  size -= sizeof(struct arts_db_s);

  // Next lets run the reduce function on the db_data and the shadow copy (dst)
  unsigned int tile_size = size / element_size;
  ARTS_INFO("TileSize: %u\n", tile_size);
  if (tile_size < 32) {
    dim3 block(tile_size, 1, 1);  // For volta...
    dim3 grid(1, 1, 1);
    void *kernel_args[] = {&sink, &src};
    ARTS_INFO("SRC: %p DST: %p\n", sink, src);
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernel_args, (size_t)0,
                                  arts_gpus[gpu_id].stream));
  } else {
    dim3 block(32, 1, 1);  // For volta...
    dim3 grid((tile_size + 32 - 1) / 32, 1, 1);
    void *kernel_args[] = {&sink, &src};
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernel_args, (size_t)0,
                                  arts_gpus[gpu_id].stream));
  }

  arts_cuda_restore_device();
  arts_unlock(&buff_lock[gpu_id]);
}

void reduce_datafrom_gpus(void *dst, unsigned int dst_gpu_id, void *src,
                          unsigned int src_gpu_id, unsigned int size,
                          arts_lc_sync_function_gpu_t fn_ptr,
                          unsigned int element_size, void *db_data) {
  ARTS_DEBUG("ELEMENT SIZE: %lu\n", element_size);
  // We need to lock in a fixed order, so smallest first
  unsigned int first = (dst_gpu_id < src_gpu_id) ? dst_gpu_id : src_gpu_id;
  unsigned int second = (dst_gpu_id == first) ? src_gpu_id : dst_gpu_id;
  arts_lock(&buff_lock[first]);
  arts_lock(&buff_lock[second]);

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

  arts_cuda_set_device((int)dst_gpu_id, true);

  // Lets remove the db header part
  size -= sizeof(struct arts_db_s);

  // Next lets run the reduce function on the db_data and the shadow copy (dst)
  unsigned int tile_size = size / element_size;
  ARTS_DEBUG("TileSize: %u\n", tile_size);
  if (tile_size < 32) {
    dim3 block(tile_size, 1, 1);  // For volta...
    dim3 grid(1, 1, 1);
    void *kernel_args[] = {&db_data, &dst};
    ARTS_DEBUG("SRC: %p DST: %p\n", db_data, dst);
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernel_args, (size_t)0,
                                  arts_gpus[dst_gpu_id].stream));
  } else {
    dim3 block(32, 1, 1);  // For volta...
    dim3 grid((tile_size + 32 - 1) / 32, 1, 1);
    void *kernel_args[] = {&db_data, &dst};
    ARTS_DEBUG("SRC: %p DST: %p\n", db_data, dst);
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernel_args, (size_t)0,
                                  arts_gpus[dst_gpu_id].stream));
  }

  // CHECKCORRECT(cudaStreamSynchronize(arts_gpus[src_gpu_id].stream));
  // CHECKCORRECT(cudaStreamSynchronize(arts_gpus[dst_gpu_id].stream));
  arts_cuda_restore_device();

  // Unlock in the correct order
  arts_unlock(&buff_lock[second]);
  arts_unlock(&buff_lock[first]);
}

void get_data_from_stream_now(unsigned int gpu_id, void *dst, void *src,
                              size_t count, bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    flush_stream(gpu_id);
    arts_unlock(&buff_lock[gpu_id]);
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
      if (host_to_dev_count[i] || kernel_to_dev_count[i] ||
          dev_to_host_count[i] || wrap_up_count[i]) {
        arts_atomic_fetch_add(&stream_check_count[i], 1U);
        if (stream_check_count[i] % CHECKSTREAM == 0) {
          arts_lock(&buff_lock[i]);
          ret |= flush_stream(i);
          arts_unlock(&buff_lock[i]);
        }
      }
    }
    return ret;
  }
  return false;
}