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
#ifndef ARTS_GPU_GPUSTREAM_H
#define ARTS_GPU_GPUSTREAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime_api.h>

#include "arts/defs.h"
#include "arts/gas/route_table.h"
#include "arts/gpu.h"
#include "arts/gpu/gpu_lc.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/utils/array_list.h"

#define CHECKCORRECT(x)                                                        \
  do {                                                                         \
    cudaError_t err;                                                           \
    if ((err = (x)) != cudaSuccess) {                                          \
      ARTS_ERROR("CUDA operation failed: %s: %s", #x,                          \
                 cudaGetErrorString(err));                                     \
    }                                                                          \
  } while (0)

typedef struct {
  unsigned int gpu_id;
  volatile unsigned int *new_edt_lock;
  arts_array_list_t *new_edts;
  void *dev_closure;
  struct arts_edt_s *edt;
} arts_gpu_clean_up_t;

typedef struct {
  int device;
  volatile uint64_t avail_global_mem;
  volatile uint64_t total_global_mem;
  struct cudaDeviceProp prop;
  volatile float occupancy;
  volatile unsigned int device_lock;
  volatile unsigned int total_edts;
  volatile unsigned int available_edt_slots;
  volatile unsigned int running_edts;
  volatile unsigned int available_threads;
  cudaStream_t stream;
} arts_gpu_t;

extern arts_gpu_t *arts_gpus;

void arts_node_init_gpus();
arts_gpu_t *arts_find_gpu(void *data);

void arts_init_per_gpu_wrapper(int argc, char **argv);
void arts_worker_init_gpus();
void arts_cleanup_gpus();
void arts_schedule_to_gpu_internal(arts_edt_t fn_ptr, uint32_t paramc,
                                   const uint64_t *paramv, uint32_t depc,
                                   arts_edt_dep_t *depv, dim3 grid, dim3 block,
                                   void *edt_ptr, arts_gpu_t *arts_gpu);
void arts_schedule_to_gpu(arts_edt_t fn_ptr, uint32_t paramc,
                          const uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t *depv, void *edt_ptr,
                          arts_gpu_t *arts_gpu);
void arts_wrap_up(cudaStream_t stream, cudaError_t status, void *data);
void arts_wrap_up_host_func(void *data);

void arts_store_new_edts(void *edt);
void arts_handle_new_edts();
void free_gpu_item(arts_route_item_t *item);

/* Multi-GPU peer copy + reduce: launches a cudaMemcpyPeer followed by the
 * given reduction kernel.  Defined in gpu_stream.cu; called cross-TU from the
 * LC sync path. */
void reduce_datafrom_gpus(void *dst, unsigned int dst_gpu_id, void *src,
                          unsigned int src_gpu_id, unsigned int size,
                          arts_lc_sync_function_gpu_t fn_ptr,
                          unsigned int element_size, void *db_data);

extern ARTS_THREAD_LOCAL arts_dim3_t *arts_local_grid;
extern ARTS_THREAD_LOCAL arts_dim3_t *arts_local_block;
extern ARTS_THREAD_LOCAL cudaStream_t *arts_local_stream;
extern ARTS_THREAD_LOCAL int arts_local_gpu_id;

extern volatile unsigned int hits;
extern volatile unsigned int misses;
extern volatile uint64_t free_bytes;

#ifdef __cplusplus
}
#endif

#endif /* ARTSGPUSTREAM_H */
