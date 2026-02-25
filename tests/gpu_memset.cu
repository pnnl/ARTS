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
 * gpu_memset.cu
 *
 * Tests DB_MODE_MEMSET for GPU zero-initialization:
 *   - arts_gpu_signal_edt_memset: signal an EDT dep slot with GPU memset
 *   - The signaled slot receives a zero-initialized GPU DB
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu.h"

#define N_ELEMENTS 64

/* Kernel: verify all values are zero and write pass/fail into first element */
__global__ void check_zeroed(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int idx = threadIdx.x;
  /* Only thread 0 does the check to avoid races */
  if (idx == 0) {
    unsigned int all_zero = 1;
    for (unsigned int i = 0; i < N_ELEMENTS; i++) {
      if (data[i] != 0) {
        all_zero = 0;
      }
    }
    /* Write result: 1 = all zeroes (pass), 0 = not all zeroes (fail) */
    data[0] = all_zero;
  }
}

/* Host EDT: verify kernel's zero-check result */
void verify_memset(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data != NULL && data[0] == 1) {
    arts_printf("  PASS: arts_gpu_signal_edt_memset zero-init verified\n");
  } else {
    arts_printf("  FAIL: arts_gpu_signal_edt_memset data not zeroed\n");
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

  unsigned int node_id = arts_get_current_node();

  /* Create a GPU DB with non-zero initial data */
  arts_guid_t db_guid = arts_guid_reserve(ARTS_DB, 0);
  unsigned int *addr = (unsigned int *)arts_db_create_with_guid(
      db_guid, sizeof(unsigned int) * N_ELEMENTS, ARTS_DB_GPU, NULL, NULL);
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    addr[i] = 0xDEADBEEF;
  }

  arts_hint_t hint_0 = {0, 0};

  /* Create done EDT */
  arts_guid_t done_guid = arts_edt_create(verify_memset, 0, NULL, 1, &hint_0);

  dim3 threads(N_ELEMENTS, 1, 1);
  dim3 grid(1, 1, 1);

  /* Create GPU EDT to check if memset zeroed the data.
   * data_guid = db_guid so the result is delivered to done EDT. */
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.route = node_id;
  gpu_hint.gpu = 0;
  gpu_hint.end_guid = done_guid;
  gpu_hint.slot = 0;
  gpu_hint.data_guid = db_guid;
  arts_guid_t gpu_edt =
      arts_edt_create_gpu(check_zeroed, 0, NULL, 1, arts_from_dim3(grid),
                          arts_from_dim3(threads), &gpu_hint);

  /* Signal the GPU EDT with MEMSET mode — this should zero-init the DB
   * before the kernel accesses it */
  arts_gpu_signal_edt_memset(gpu_edt, 0, db_guid);
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
