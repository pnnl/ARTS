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

/*
 * gpu_edt_basic.cu
 *
 * Tests basic GPU EDT creation and execution:
 *   - arts_edt_create_gpu: create a GPU EDT that runs a __global__ kernel
 *   - arts_edt_create_gpu_direct: create a GPU EDT targeting a specific GPU
 *   - Kernel writes results to a DB, host EDT verifies correctness
 *   - GET_GPU_INDEX macro
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"

#define N_ELEMENTS 64

/* ---------- Test 1: arts_edt_create_gpu ---------- */

/* Kernel: each thread writes its thread index + 1 into the DB */
__global__ void write_kernel(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if (idx < N_ELEMENTS) {
    data[idx] = idx + 1;
  }
}

/* Host EDT: verify kernel wrote correct values */
void verify_write(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int pass = 1;
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    if (data[i] != i + 1) {
      arts_printf("FAIL test1: data[%u] = %u, expected %u\n", i, data[i],
                  i + 1);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS test1: arts_edt_create_gpu basic kernel execution\n");
  }
  arts_shutdown();
}

/* ---------- Test 2: arts_edt_create_gpu_direct + GET_GPU_INDEX ---------- */

/* Kernel: writes the GPU index from GET_GPU_INDEX() into each element */
__global__ void gpu_index_kernel(uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t gpu_id = GET_GPU_INDEX();
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if (idx < N_ELEMENTS) {
    data[idx] = (unsigned int)gpu_id;
  }
}

/* Host EDT: verify GPU index values (all elements should equal gpu 0) */
void verify_gpu_index(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int pass = 1;
  /* All elements were written by GPU 0, so all should be 0 */
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    if (data[i] != 0) {
      arts_printf("FAIL test2: data[%u] = %u, expected 0\n", i, data[i]);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS test2: arts_edt_create_gpu_direct + GET_GPU_INDEX\n");
  }

  /* --- Now run test 1 --- */
  unsigned int node_id = arts_get_current_node();
  unsigned int *addr = NULL;
  arts_guid_t db_guid1 = arts_guid_reserve(ARTS_DB_GPU, 0);
  addr = (unsigned int *)arts_db_create_with_guid(
      db_guid1, sizeof(unsigned int) * N_ELEMENTS, NULL);
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    addr[i] = 0;
  }

  arts_hint_t hint_0 = {0, 0};
  arts_guid_t verify_guid = arts_edt_create(verify_write, 0, NULL, 1, &hint_0);

  dim3 threads(N_ELEMENTS, 1, 1);
  dim3 grid(1, 1, 1);
  arts_guid_t gpu_edt =
      arts_edt_create_gpu(write_kernel, node_id, 0, NULL, 1, grid, threads,
                          verify_guid, 0, db_guid1);
  arts_signal_edt(gpu_edt, 0, db_guid1, DB_MODE_EW);
}

extern "C" void arts_init_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream, int argc, char **argv) {
  (void)node_id;
  (void)dev_id;
  (void)stream;
  (void)argc;
  (void)argv;
}

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int node_id = arts_get_current_node();

  /* Test 2: arts_edt_create_gpu_direct targeting gpu 0 */
  unsigned int *addr = NULL;
  arts_guid_t db_guid = arts_guid_reserve(ARTS_DB_GPU, 0);
  addr = (unsigned int *)arts_db_create_with_guid(
      db_guid, sizeof(unsigned int) * N_ELEMENTS, NULL);
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    addr[i] = (unsigned int)-1;
  }

  arts_hint_t hint_0 = {0, 0};
  arts_guid_t verify_guid =
      arts_edt_create(verify_gpu_index, 0, NULL, 1, &hint_0);

  dim3 threads(N_ELEMENTS, 1, 1);
  dim3 grid(1, 1, 1);

  /* Create GPU EDT targeting GPU 0 directly */
  arts_guid_t gpu_edt =
      arts_edt_create_gpu_direct(gpu_index_kernel, node_id, 0, 0, NULL, 1, grid,
                                 threads, verify_guid, 0, db_guid, true);
  arts_signal_edt(gpu_edt, 0, db_guid, DB_MODE_EW);
}

extern "C" void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream) {
  (void)node_id;
  (void)dev_id;
  (void)stream;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
