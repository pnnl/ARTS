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
 * gpu_edt_passthrough.cu
 *
 * Tests GPU EDT passthrough mode:
 *   - arts_edt_create_gpu with hint.passthrough=true: passthrough EDT that
 *     forwards a dep slot
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu.h"

#define N_ELEMENTS 16

/* Kernel: increment each element in dep[0] by 1 */
__global__ void increment_kernel(uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int idx = threadIdx.x;
  if (idx < N_ELEMENTS) {
    data[idx] += 1;
  }
}

/* Verify: data should all be 1 (initialized to 0, incremented by kernel) */
void verify_passthrough(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int pass = 1;
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    if (data[i] != 1) {
      arts_printf("FAIL: data[%u] = %u, expected 1\n", i, data[i]);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS: arts_edt_create_gpu passthrough mode\n");
  }
  arts_shutdown();
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

  unsigned int node_id = arts_get_current_rank();

  /* Create a DB with initial zeros */
  unsigned int *addr = NULL;
  arts_guid_t db_guid = arts_guid_reserve(ARTS_GUID_DB, 0);
  addr = (unsigned int *)arts_db_create_with_guid(
      db_guid, sizeof(unsigned int) * N_ELEMENTS, ARTS_DB_GPU_PIN, NULL, NULL);
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    addr[i] = 0;
  }

  /* Create final verify EDT with 1 dep */
  arts_edt_hint_t hint_0 = {0, 0};
  arts_guid_t verify_guid =
      arts_edt_create(verify_passthrough, 0, NULL, 1, &hint_0);

  dim3 threads(N_ELEMENTS, 1, 1);
  dim3 grid(1, 1, 1);

  /*
   * Passthrough mode: the GPU kernel runs, and upon completion the dep at
   * the slot stored in hint.data_guid is forwarded to end_guid at slot.
   * data_guid = 0 means depv[0] (our DB) is passed through.
   */
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.gpu = -1;
  gpu_hint.rank = node_id;
  gpu_hint.end_guid = verify_guid;
  gpu_hint.slot = 0;
  gpu_hint.data_guid = (arts_guid_t)0;
  gpu_hint.passthrough = true;
  arts_guid_t gpu_edt =
      arts_edt_create_gpu(increment_kernel, 0, NULL, 1, arts_from_dim3(grid),
                          arts_from_dim3(threads), &gpu_hint);
  arts_add_dependence(db_guid, gpu_edt, 0, DB_MODE_RW);
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
