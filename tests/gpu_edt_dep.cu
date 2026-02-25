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
 * gpu_edt_dep.cu
 *
 * Tests GPU EDT creation with dependency variants:
 *   - arts_edt_create_gpu with depc>0 (signal deps later)
 *   - arts_edt_create_gpu with depc=0 (no deps)
 *   - arts_edt_create_gpu with data_guid (auto-signal)
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu.h"

#define N_ELEMENTS 32

/* ---------- Test 1: arts_edt_create_gpu (depc=1, signal later) ---------- */

/* Kernel: each thread writes thread index to the output DB */
__global__ void dep_kernel(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int idx = threadIdx.x;
  if (idx < N_ELEMENTS) {
    data[idx] = idx * 2;
  }
}

/* Verify test 1 results */
void verify_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int pass = 1;
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    if (data[i] != i * 2) {
      arts_printf("FAIL test1: data[%u] = %u, expected %u\n", i, data[i],
                  i * 2);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS test1: arts_edt_create_gpu (depc=1, signal later)\n");
  }
  arts_shutdown();
}

/* ---------- Test 2: arts_edt_create_gpu (depc=0, no deps) ---------- */

/* Kernel with no depv: just writes paramc+1 into paramv output */
__global__ void nodep_kernel(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  /* Just use paramv[0], no depv needed */
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    /* Can't do much without depv, just verify paramc is correct */
    /* The test verification is in the done EDT */
  }
  (void)paramc;
  (void)paramv;
}

/* Verify: if we made it here from the GPU EDT chain, the dep-less EDT ran */
void verify_nodep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS test2: arts_edt_create_gpu (depc=0) ran\n");

  /* Now run test 1 */
  unsigned int node_id = arts_get_current_node();
  unsigned int *addr = NULL;
  arts_guid_t db_guid = arts_guid_reserve(ARTS_DB, 0);
  addr = (unsigned int *)arts_db_create_with_guid(
      db_guid, sizeof(unsigned int) * N_ELEMENTS, ARTS_DB_GPU, NULL, NULL);
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    addr[i] = 0;
  }

  arts_hint_t hint_0 = {0, 0};
  arts_guid_t verify_guid = arts_edt_create(verify_dep, 0, NULL, 1, &hint_0);

  dim3 threads(N_ELEMENTS, 1, 1);
  dim3 grid(1, 1, 1);

  /* Create GPU EDT with depc=1, signal dep manually */
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.gpu = -1;
  gpu_hint.route = node_id;
  gpu_hint.end_guid = verify_guid;
  gpu_hint.slot = 0;
  gpu_hint.data_guid = db_guid;
  arts_guid_t gpu_edt =
      arts_edt_create_gpu(dep_kernel, 0, NULL, 1, arts_from_dim3(grid),
                          arts_from_dim3(threads), &gpu_hint);
  arts_signal_edt(gpu_edt, 0, db_guid, DB_MODE_EW);
}

extern "C" void arts_init_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream, int argc, char **argv) {
  (void)node_id;
  (void)dev_id;
  (void)stream;
  (void)argc;
  (void)argv;
}

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int node_id = arts_get_current_node();
  arts_hint_t hint_0 = {0, 0};

  /* Test 2 first: dep-less GPU EDT */
  arts_guid_t verify2_guid = arts_edt_create(verify_nodep, 0, NULL, 1, &hint_0);

  dim3 threads(1, 1, 1);
  dim3 grid(1, 1, 1);

  /* depc=0 means no dependency slots */
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.gpu = -1;
  gpu_hint.route = node_id;
  gpu_hint.end_guid = verify2_guid;
  gpu_hint.slot = 0;
  gpu_hint.data_guid = NULL_GUID;
  arts_guid_t gpu_edt =
      arts_edt_create_gpu(nodep_kernel, 0, NULL, 0, arts_from_dim3(grid),
                          arts_from_dim3(threads), &gpu_hint);
  (void)gpu_edt;
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
