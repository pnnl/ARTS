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
 * gpu_memory.cu
 *
 * Tests GPU memory allocation and copy functions:
 *   - arts_cuda_malloc / arts_cuda_free (device memory)
 *   - arts_cuda_malloc_host / arts_cuda_free_host (pinned host memory)
 *   - arts_cuda_mem_cpy_to_dev / arts_cuda_mem_cpy_from_dev (host<->device)
 *   - arts_get_num_gpus
 *   - arts_get_gpus_per_rank
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu.h"

#define N_ELEMENTS 128

unsigned int *dev_buffer = NULL;

extern "C" void arts_init_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream, int argc, char **argv) {
  (void)node_id;
  (void)stream;
  (void)argc;
  (void)argv;

  /* Test: arts_cuda_malloc allocates device memory */
  if (dev_id == 0) {
    dev_buffer =
        (unsigned int *)arts_cuda_malloc(sizeof(unsigned int) * N_ELEMENTS);
    if (dev_buffer != NULL) {
      arts_printf("PASS test1: arts_cuda_malloc returned non-NULL\n");
    } else {
      arts_printf("FAIL test1: arts_cuda_malloc returned NULL\n");
    }
  }
}

void run_mem_tests(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int pass;

  /* Test 2: arts_cuda_malloc_host (pinned host memory) */
  unsigned int *host_pinned =
      (unsigned int *)arts_cuda_malloc_host(sizeof(unsigned int) * N_ELEMENTS);
  if (host_pinned != NULL) {
    arts_printf("PASS test2: arts_cuda_malloc_host returned non-NULL\n");
  } else {
    arts_printf("FAIL test2: arts_cuda_malloc_host returned NULL\n");
    arts_shutdown();
    return;
  }

  /* Fill host buffer with known values */
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    host_pinned[i] = i + 100;
  }

  /* Test 3: arts_cuda_mem_cpy_to_dev (host -> device) */
  arts_cuda_mem_cpy_to_dev(dev_buffer, host_pinned,
                           sizeof(unsigned int) * N_ELEMENTS);
  arts_printf("PASS test3: arts_cuda_mem_cpy_to_dev completed\n");

  /* Clear host buffer */
  memset(host_pinned, 0, sizeof(unsigned int) * N_ELEMENTS);

  /* Test 4: arts_cuda_mem_cpy_from_dev (device -> host) */
  arts_cuda_mem_cpy_from_dev(host_pinned, dev_buffer,
                             sizeof(unsigned int) * N_ELEMENTS);
  pass = 1;
  for (unsigned int i = 0; i < N_ELEMENTS; i++) {
    if (host_pinned[i] != i + 100) {
      arts_printf("FAIL test4: host_pinned[%u] = %u, expected %u\n", i,
                  host_pinned[i], i + 100);
      pass = 0;
    }
  }
  if (pass) {
    arts_printf("PASS test4: arts_cuda_mem_cpy_from_dev round-trip\n");
  }

  /* Test 5: arts_get_num_gpus, arts_get_gpus_per_rank */
  unsigned int num_gpus = arts_get_num_gpus();
  unsigned int total_gpus = arts_get_gpus_per_rank();
  arts_printf("INFO: arts_get_num_gpus=%u, arts_get_gpus_per_rank=%u\n", num_gpus,
              total_gpus);
  if (num_gpus > 0 && total_gpus >= num_gpus) {
    arts_printf("PASS test5: GPU count queries valid\n");
  } else {
    arts_printf("FAIL test5: unexpected GPU counts\n");
  }

  /* Cleanup */
  arts_cuda_free_host(host_pinned);
  arts_printf("PASS test6: arts_cuda_free_host completed\n");

  arts_shutdown();
}

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* Run the memory tests from a GPU lib EDT so we have CUDA context */
  dim3 threads(1, 1, 1);
  dim3 grid(1, 1, 1);
  unsigned int node_id = arts_get_current_rank();
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.rank = node_id;
  gpu_hint.gpu = 0;
  gpu_hint.lib = true;
  arts_edt_create_gpu(run_mem_tests, 0, NULL, 0, arts_from_dim3(grid),
                      arts_from_dim3(threads), &gpu_hint);
}

extern "C" void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream) {
  (void)node_id;
  (void)stream;
  if (dev_id == 0 && dev_buffer != NULL) {
    arts_cuda_free(dev_buffer);
    arts_printf("PASS test7: arts_cuda_free completed\n");
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
