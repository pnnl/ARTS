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
#ifndef ARTS_GPU_GPURUNTIME_H
#define ARTS_GPU_GPURUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arts/gpu/gpu_stream.h"
#include "arts/runtime/rt.h"

typedef struct {
  struct arts_edt_s wrapperEdt;
  dim3 grid;
  dim3 block;
  int gpuToRunOn;
  arts_guid_t end_guid;
  arts_guid_t data_guid;
  uint32_t slot;
  bool passthrough;
  bool lib;
} arts_gpu_edt_t;

int arts_get_current_gpu();
bool arts_cuda_set_device(int id, bool save);
bool arts_cuda_restore_device();

void *arts_cuda_malloc_host(unsigned int size);
void arts_cuda_free_host(void *ptr);
void *arts_cuda_malloc(unsigned int size);
void arts_cuda_free(void *ptr);
void arts_cuda_mem_cpy_from_dev(void *dst, void *src, size_t count);
void arts_cuda_mem_cpy_to_dev(void *dst, void *src, size_t count);
arts_guid_t arts_edt_create_gpu_dep(arts_edt_t func_ptr, unsigned int route,
                                    uint32_t paramc, const uint64_t *paramv,
                                    uint32_t depc, dim3 grid, dim3 block,
                                    arts_guid_t end_guid, uint32_t slot,
                                    arts_guid_t data_guid, bool has_depv);
arts_guid_t arts_edt_create_gpu(arts_edt_t func_ptr, unsigned int route,
                                uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, dim3 grid, dim3 block,
                                arts_guid_t end_guid, uint32_t slot,
                                arts_guid_t data_guid);
arts_guid_t arts_edt_create_gpu_pt(arts_edt_t func_ptr, unsigned int route,
                                   uint32_t paramc, const uint64_t *paramv,
                                   uint32_t depc, dim3 grid, dim3 block,
                                   arts_guid_t end_guid, uint32_t slot,
                                   unsigned int pass_slot);
arts_guid_t arts_edt_create_gpu_pt_dep(arts_edt_t func_ptr, unsigned int route,
                                       uint32_t paramc, const uint64_t *paramv,
                                       uint32_t depc, dim3 grid, dim3 block,
                                       arts_guid_t end_guid, uint32_t slot,
                                       unsigned int pass_slot, bool has_depv);
arts_guid_t arts_edt_create_gpu_lib(arts_edt_t func_ptr, unsigned int route,
                                    uint32_t paramc, const uint64_t *paramv,
                                    uint32_t depc, dim3 grid, dim3 block);
arts_guid_t arts_edt_create_gpu_pt_with_guid(
    arts_edt_t func_ptr, arts_guid_t guid, uint32_t paramc,
    const uint64_t *paramv, uint32_t depc, dim3 grid, dim3 block,
    arts_guid_t end_guid, uint32_t slot, unsigned int pass_slot);
arts_guid_t arts_edt_create_gpu_direct(arts_edt_t func_ptr, unsigned int route,
                                       unsigned int gpu, uint32_t paramc,
                                       const uint64_t *paramv, uint32_t depc,
                                       dim3 grid, dim3 block,
                                       arts_guid_t end_guid, uint32_t slot,
                                       arts_guid_t data_guid, bool has_depv);
arts_guid_t arts_edt_create_gpu_lib_direct(
    arts_edt_t func_ptr, unsigned int route, unsigned int gpu, uint32_t paramc,
    const uint64_t *paramv, uint32_t depc, dim3 grid, dim3 block);
arts_guid_t arts_edt_create_gpu_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                          uint32_t paramc,
                                          const uint64_t *paramv, uint32_t depc,
                                          dim3 grid, dim3 block,
                                          arts_guid_t end_guid, uint32_t slot,
                                          arts_guid_t data_guid);
arts_guid_t arts_edt_create_gpu_lib_with_guid(arts_edt_t func_ptr,
                                              arts_guid_t guid, uint32_t paramc,
                                              const uint64_t *paramv,
                                              uint32_t depc, dim3 grid,
                                              dim3 block);

dim3 *arts_get_gpu_grid();
dim3 *arts_get_gpu_block();
cudaStream_t *arts_get_gpu_stream();
int arts_get_gpu_id();
unsigned int arts_get_num_gpus();
void arts_put_in_db_from_gpu(void *ptr, arts_guid_t db_guid,
                             unsigned int offset, unsigned int size,
                             bool free_data);

void arts_gpu_host_wrap_up(void *edt_packet, arts_guid_t to_signal,
                           uint32_t slot, arts_guid_t data_guid);
void arts_run_gpu(void *edt_packet, arts_gpu_t *arts_gpu);
bool arts_gpu_scheduler_loop();

void arts_lc_sync(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid);
void arts_gpu_signal_edt_memset(arts_guid_t edt_guid, uint32_t slot,
                                arts_guid_t data_guid);
void internal_lc_sync_cpu(arts_guid_t acq_guid, struct arts_db_s *db);
void internal_lc_sync_gpu(arts_guid_t acq_guid, struct arts_db_s *db);

#define GET_GPU_INDEX() internal_get_gpu_index(paramv)
__device__ static uint64_t internal_get_gpu_index(const uint64_t *paramv) {
  return *(paramv - 1);
}

#ifdef __cplusplus
}
#endif

#endif /* ARTSGPURUNTIME_H */
