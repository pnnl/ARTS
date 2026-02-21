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
 * gpu_multi_kernel.cu
 *
 * Tests launching multiple GPU kernels across available GPUs:
 *   - One kernel per GPU, each writes its GPU index into a shared LC DB
 *   - Uses arts_edt_create_gpu_direct to target specific GPUs
 *   - Fan-in pattern: all GPU EDTs signal a single done EDT
 *   - Similar to the existing lc_sync.cu pattern but exercises multi-GPU
 */

#include <stdio.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"

/* Maximum supported GPUs for this test */
#define MAX_GPUS 8

/* Kernel: write GPU index into the shared DB at position [gpu_id] */
__global__ void tag_kernel(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t gpu_id = GET_GPU_INDEX();
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    data[gpu_id] = (unsigned int)(gpu_id + 1);
  }
}

/* Verify that each GPU wrote its expected value */
void verify_multi(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int total = (unsigned int)paramv[0];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  unsigned int pass = 1;

  for (unsigned int i = 0; i < total; i++) {
    if (data[i] != i + 1) {
      arts_printf("FAIL: data[%u] = %u, expected %u\n", i, data[i], i + 1);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS: Multi-GPU kernel fan-in (%u GPUs)\n", total);
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

extern "C" void arts_main_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int total_gpus = arts_get_total_gpus();
  unsigned int node_id = arts_get_current_node();

  if (total_gpus == 0) {
    arts_printf("SKIP: no GPUs available\n");
    arts_shutdown();
    return;
  }
  if (total_gpus > MAX_GPUS) {
    total_gpus = MAX_GPUS;
  }

  /* Create an LC DB large enough for all GPUs */
  arts_guid_t lc_guid = arts_guid_reserve(ARTS_DB_LC, 0);
  unsigned int *addr = (unsigned int *)arts_db_create_with_guid(
      lc_guid, sizeof(unsigned int) * total_gpus, NULL);
  for (unsigned int i = 0; i < total_gpus; i++) {
    addr[i] = 0;
  }

  arts_hint_t hint_0 = {0, 0};

  /*
   * done EDT: slot 0 = LC sync result, slots 1..total_gpus = GPU EDT signals
   */
  uint64_t args[] = {(uint64_t)total_gpus};
  arts_guid_t done_guid =
      arts_edt_create(verify_multi, 1, args, total_gpus + 1, &hint_0);

  /* Wire LC sync on slot 0 */
  arts_lc_sync(done_guid, 0, lc_guid);

  dim3 threads(1, 1, 1);
  dim3 grid(1, 1, 1);

  /* Launch one kernel per GPU */
  for (unsigned int i = 0; i < total_gpus; i++) {
    arts_guid_t gpu_edt =
        arts_edt_create_gpu_direct(tag_kernel, node_id, i, 0, NULL, 1, grid,
                                   threads, done_guid, i + 1, NULL_GUID, true);
    arts_signal_edt(gpu_edt, 0, lc_guid, ARTS_MODE_EW);
  }
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
