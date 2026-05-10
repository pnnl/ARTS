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
 * gpu_lc_sync_basic.cu
 *
 * Tests ARTS_DB_GPU (Locality Class) datablock with arts_lc_sync:
 *   - Create an ARTS_DB_GPU datablock (CPU-GPU coherence)
 *   - GPU kernel modifies the LC DB
 *   - arts_lc_sync synchronizes the data back from GPU to CPU
 *   - Host EDT verifies the LC DB has correct data
 *
 * Pattern matches the existing lc_sync.cu test structure.
 */

#include <stdio.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/gpu.h"

#define N_ELEMENTS 8

/* Kernel: each thread writes its index value into the LC DB */
__global__ void lc_write_kernel(uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int idx = threadIdx.x;
  if (idx < N_ELEMENTS) {
    data[idx] = (idx + 1) * 10;
  }
}

/* Host EDT: verify LC DB synced values from GPU */
void verify_lc(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int pass = 1;
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    if (data[i] != (i + 1) * 10) {
      arts_printf("FAIL: data[%u] = %u, expected %u\n", i, data[i],
                  (i + 1) * 10);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS: ARTS_DB_GPU + arts_lc_sync basic test\n");
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

  /* Create an LC datablock */
  arts_guid_t lc_guid = arts_guid_reserve(ARTS_GUID_DB, 0);
  unsigned int *addr = (unsigned int *)arts_db_create_with_guid(
      lc_guid, sizeof(unsigned int) * N_ELEMENTS, ARTS_DB_GPU, NULL, NULL);

  /* Initialize to sentinel values */
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    addr[i] = (unsigned int)-1;
  }

  arts_edt_hint_t hint_0 = {0, 0};

  /*
   * done_guid has 2 deps:
   *   slot 0: LC sync (will deliver the synced DB)
   *   slot 1: signal from GPU EDT completion
   */
  arts_guid_t done_guid = arts_edt_create(verify_lc, 0, NULL, 2, &hint_0);

  /* Wire LC sync: when done_guid's slot 0 fires, LC DB is synced from GPU */
  arts_lc_sync(done_guid, 0, lc_guid);

  dim3 threads(N_ELEMENTS, 1, 1);
  dim3 grid(1, 1, 1);

  /* Create GPU EDT targeting GPU 0, signals done_guid slot 1 on completion */
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.rank = node_id;
  gpu_hint.gpu = 0;
  gpu_hint.end_guid = done_guid;
  gpu_hint.slot = 1;
  gpu_hint.data_guid = NULL_GUID;
  arts_guid_t gpu_edt =
      arts_edt_create_gpu(lc_write_kernel, 0, NULL, 1, arts_from_dim3(grid),
                          arts_from_dim3(threads), &gpu_hint);
  arts_add_dependence(lc_guid, gpu_edt, 0, DB_MODE_RW);
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
