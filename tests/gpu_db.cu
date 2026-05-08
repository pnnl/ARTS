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
 * gpu_db.cu
 *
 * Tests GPU datablock operations:
 *   - ARTS_DB_GPU_PIN type creation with arts_guid_reserve +
 * arts_db_create_with_guid
 *   - arts_put_in_db_from_gpu: copy GPU device memory into a host DB
 *   - GPU kernel writes to device memory, then arts_put_in_db_from_gpu
 *     transfers to a host DB for verification
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu.h"

#define N_ELEMENTS 32

/* Kernel: fill device memory with patterned values */
__global__ void fill_device_mem(uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int *dev_data = (unsigned int *)paramv[0];
  unsigned int idx = threadIdx.x;
  if (idx < N_ELEMENTS) {
    dev_data[idx] = idx * 3 + 7;
  }
}

/* Host lib EDT: after kernel completes, use arts_put_in_db_from_gpu */
void transfer_to_db(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int *dev_data = (unsigned int *)paramv[0];
  arts_guid_t done_guid = (arts_guid_t)paramv[1];

  /* Create host DB to receive the data */
  unsigned int *host = NULL;
  arts_guid_t db_guid = arts_db_create(
      (void **)&host, sizeof(unsigned int) * N_ELEMENTS, ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);

  /* Copy from GPU device memory into the host DB */
  arts_put_in_db_from_gpu(dev_data, db_guid, 0,
                          sizeof(unsigned int) * N_ELEMENTS, false);

  /* Signal done EDT with the host DB */
  arts_add_dependence(db_guid, done_guid, 0, DB_MODE_RW);
}

/* Verify the data transferred correctly */
void verify_transfer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int pass = 1;
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    if (data[i] != i * 3 + 7) {
      arts_printf("FAIL: data[%u] = %u, expected %u\n", i, data[i], i * 3 + 7);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS: arts_put_in_db_from_gpu transfer verified\n");
  }
  arts_shutdown();
}

unsigned int *dev_buffer = NULL;

extern "C" void arts_init_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream, int argc, char **argv) {
  (void)node_id;
  (void)stream;
  (void)argc;
  (void)argv;
  if (dev_id == 0) {
    dev_buffer =
        (unsigned int *)arts_cuda_malloc(sizeof(unsigned int) * N_ELEMENTS);
  }
}

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int node_id = arts_get_current_rank();

  /* Test 1: ARTS_DB_GPU_PIN creation */
  unsigned int *addr = NULL;
  arts_guid_t gpu_db = arts_guid_reserve(ARTS_DB, 0);
  addr = (unsigned int *)arts_db_create_with_guid(
      gpu_db, sizeof(unsigned int) * N_ELEMENTS, ARTS_DB_GPU_PIN, ARTS_DB_PROP_NONE, NULL);
  if (addr != NULL) {
    arts_printf("PASS test1: ARTS_DB_GPU_PIN created with non-NULL addr\n");
  } else {
    arts_printf("FAIL test1: ARTS_DB_GPU_PIN addr is NULL\n");
  }

  /* Test 2: GPU kernel writes to device buffer, then transfer to host DB */
  arts_edt_hint_t hint_0 = {0, 0};
  arts_guid_t done_guid = arts_edt_create(verify_transfer, 0, NULL, 1, &hint_0);

  uint64_t args[] = {(uint64_t)dev_buffer, (uint64_t)done_guid};
  dim3 threads(N_ELEMENTS, 1, 1);
  dim3 grid(1, 1, 1);

  /* Chain: kernel -> lib_transfer -> done */
  arts_gpu_hint_t lib_hint = {};
  lib_hint.rank = node_id;
  lib_hint.gpu = 0;
  lib_hint.lib = true;
  arts_guid_t lib_edt =
      arts_edt_create_gpu(transfer_to_db, 2, args, 0, arts_from_dim3(grid),
                          arts_from_dim3(threads), &lib_hint);
  arts_gpu_hint_t kern_hint = {};
  kern_hint.rank = node_id;
  kern_hint.gpu = 0;
  kern_hint.end_guid = lib_edt;
  kern_hint.slot = 0;
  kern_hint.data_guid = NULL_GUID;
  arts_guid_t gpu_edt = arts_edt_create_gpu(
      fill_device_mem, 1, (uint64_t *)&dev_buffer, 0, arts_from_dim3(grid),
      arts_from_dim3(threads), &kern_hint);
  (void)gpu_edt;
}

extern "C" void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream) {
  (void)node_id;
  (void)stream;
  if (dev_id == 0 && dev_buffer != NULL) {
    arts_cuda_free(dev_buffer);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
