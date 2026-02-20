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
#ifndef ARTS_GPU_GPUSTREAMBUFFER_H
#define ARTS_GPU_GPUSTREAMBUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime_api.h>

#include "arts/gpu/gpu_lc_sync_functions.cuh"
#include "arts/runtime/rt.h"

typedef struct {
  void *dst;
  void *src;
  size_t count;
} arts_buffer_mem_move_t;

typedef struct {
  uint32_t paramc;
  const uint64_t *paramv;
  uint32_t depc;
  arts_edt_dep_t *depv;
  arts_edt_t fn_ptr;
  unsigned int grid[3];
  unsigned int block[3];
} arts_buffer_kernel_t;

// CHECKCORRECT(cudaMemcpyAsync(data_ptr, depv[i].ptr, size,
// cudaMemcpyHostToDevice, arts_gpu->stream));
bool push_data_to_stream(unsigned int gpu_id, void *dst, void *src,
                         size_t count, bool buff);
bool get_data_from_stream(unsigned int gpu_id, void *dst, void *src,
                          size_t count, bool buff);

//  void * kernelArgs[] = { &paramc, &devParamv, &depc, &devDepv };
// CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
// (void**)kernelArgs, (size_t)0, arts_gpu->stream));
bool push_kernel_to_stream(unsigned int gpu_id, uint32_t paramc,
                           const uint64_t *paramv, uint32_t depc,
                           arts_edt_dep_t *depv, arts_edt_t fn_ptr, dim3 grid,
                           dim3 block, bool buff);

// #if CUDART_VERSION >= 10000
//     CHECKCORRECT(cudaLaunchHostFunc(arts_gpu->stream, artsWrapUp,
//     host_closure));
// #else
//     CHECKCORRECT(cudaStreamAddCallback(arts_gpu->stream, artsWrapUp,
//     host_closure, 0));
// #endif
bool push_wrap_up_to_stream(unsigned int gpu_id, void *host_closure, bool buff);

bool flush_mem_stream(unsigned int gpu_id, unsigned int *count,
                      arts_buffer_mem_move_t *buff, enum cudaMemcpyKind kind);
bool flush_kernel_stream(unsigned int gpu_id);
bool flush_wrap_up_stream(unsigned int gpu_id);

bool flush_stream(unsigned int gpu_id);
bool check_streams(bool buff_on);

void reduce_datafrom_gpus(void *dst, unsigned int dst_gpu_id, void *src,
                          unsigned int src_gpu_id, unsigned int size,
                          arts_lc_sync_function_gpu_t fn_ptr,
                          unsigned int element_size, void *db_data);
void get_data_from_stream_now(unsigned int gpu_id, void *dst, void *src,
                              size_t count, bool buff);
void copy_gputo_gpu(void *dst, unsigned int dst_gpu_id, void *src,
                    unsigned int src_gpu_id, unsigned int size);
void do_reduction_now(unsigned int gpu_id, void *sink, void *src,
                      arts_lc_sync_function_gpu_t fn_ptr,
                      unsigned int element_size, unsigned int size);

#ifdef __cplusplus
}
#endif

#endif /* ARTSGPUSTREAMBUFFER_H */
