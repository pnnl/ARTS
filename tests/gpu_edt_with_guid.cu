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
 * gpu_edt_with_guid.cu
 *
 * Tests GPU EDT creation with pre-reserved GUIDs:
 *   - arts_edt_create_gpu_with_guid: GPU kernel EDT with specific GUID
 *   - arts_edt_create_gpu_with_guid (lib=true): GPU lib (host) EDT with
 *     specific GUID
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu.h"

#define N_ELEMENTS 16

/* ---------- Test 1: arts_edt_create_gpu_with_guid ---------- */

/* Kernel: write constant 42 into every element of dep[0] */
__global__ void fill_kernel(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int idx = threadIdx.x;
  if (idx < N_ELEMENTS) {
    data[idx] = 42;
  }
}

void verify_fill(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int pass = 1;
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    if (data[i] != 42) {
      arts_printf("FAIL test1: data[%u] = %u, expected 42\n", i, data[i]);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS test1: arts_edt_create_gpu_with_guid\n");
  }
  arts_shutdown();
}

/* ---------- Test 2: arts_edt_create_gpu_with_guid (lib=true) ---------- */

/* Host function scheduled as GPU lib EDT */
void lib_work(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t done_guid = (arts_guid_t)paramv[0];
  arts_printf("PASS test2: arts_edt_create_gpu_with_guid (lib=true) ran\n");

  /* Proceed to test 1 */
  unsigned int *addr = NULL;
  arts_guid_t db_guid = arts_guid_reserve(ARTS_DB, 0);
  addr = (unsigned int *)arts_db_create_with_guid(
      db_guid, sizeof(unsigned int) * N_ELEMENTS, ARTS_DB_GPU_PIN, NULL, NULL);
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    addr[i] = 0;
  }

  arts_edt_hint_t hint_0 = {0, 0};
  arts_guid_t verify_guid = arts_edt_create(verify_fill, 0, NULL, 1, &hint_0);

  dim3 threads(N_ELEMENTS, 1, 1);
  dim3 grid(1, 1, 1);

  /* Use pre-reserved GUID for GPU EDT */
  arts_guid_t edt_guid = arts_guid_reserve(ARTS_EDT, 0);
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.gpu = -1;
  gpu_hint.end_guid = verify_guid;
  gpu_hint.slot = 0;
  gpu_hint.data_guid = db_guid;
  arts_edt_create_gpu_with_guid(fill_kernel, edt_guid, 0, NULL, 1,
                                arts_from_dim3(grid), arts_from_dim3(threads),
                                &gpu_hint);
  arts_add_dependence(db_guid, edt_guid, 0, DB_MODE_RW);

  (void)done_guid;
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

  /* Test 2: create a GPU lib EDT with a pre-reserved GUID */
  arts_guid_t lib_guid = arts_guid_reserve(ARTS_EDT, 0);
  arts_edt_hint_t hint_0 = {0, 0};
  arts_guid_t placeholder_guid =
      arts_edt_create(verify_fill, 0, NULL, 1, &hint_0);
  uint64_t args[] = {(uint64_t)placeholder_guid};

  dim3 threads(1, 1, 1);
  dim3 grid(1, 1, 1);
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.gpu = -1;
  gpu_hint.lib = true;
  arts_edt_create_gpu_with_guid(lib_work, lib_guid, 1, args, 0,
                                arts_from_dim3(grid), arts_from_dim3(threads),
                                &gpu_hint);
  (void)lib_guid;
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
