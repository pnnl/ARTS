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
#include <stdio.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"

__global__ void temp(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t gpu_id = GET_GPU_INDEX();
  // printf("Hello from %lu\n", gpu_id);
  unsigned int *addr = (unsigned int *)depv[0].ptr;
  unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);
  addr[index] = (unsigned int)(gpu_id + 1);
}

void done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *tile = (unsigned int *)depv[0].ptr;
  for (unsigned int j = 0; j < arts_get_total_gpus(); j++) {
    printf("%u, ", tile[j]);
  }
  printf("\n");
  arts_shutdown();
}

extern "C" void init_per_node(unsigned int node_id, int argc, char **argv) {
  (void)node_id;
  (void)argc;
  (void)argv;
}

extern "C" void init_per_gpu(unsigned int node_id, int dev_id,
                             cudaStream_t *stream, int argc, char **argv) {
  (void)node_id;
  (void)dev_id;
  (void)stream;
  (void)argc;
  (void)argv;
}

extern "C" void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  (void)argc;
  (void)argv;
  if (!worker_id) {
    unsigned int *addr = NULL;
    arts_printf("creating size: %u\n", sizeof(unsigned int) * arts_get_total_gpus());
    arts_guid_t db_guid = arts_db_create(
        (void **)&addr, sizeof(unsigned int) * arts_get_total_gpus(), ARTS_DB_LC);
    for (uint64_t i = 0; i < arts_get_total_gpus(); i++) {
      addr[i] = (unsigned int)-1;
    }

    arts_guid_t done_guid =
        arts_edt_create(done, 0, 0, NULL, arts_get_total_gpus() + 1);
    arts_lc_sync(done_guid, 0, db_guid);
    // arts_signal_edt(done_guid, 0, db_guid);

    dim3 threads(arts_get_total_gpus(), 1, 1);
    dim3 grid(1, 1, 1);
    for (uint64_t i = 0; i < arts_get_total_gpus(); i++) {
      if (i == 3 || i == 4 || i == 7) {
        arts_printf("CREATING EDT for GPU: %lu\n", i);
        arts_guid_t edt_guid =
            arts_edt_create_gpu_direct(temp, node_id, i, 0, NULL, 1, grid, threads,
                                   done_guid, i + 1, NULL_GUID, true);
        arts_signal_edt(edt_guid, 0, db_guid);
      } else {
        arts_signal_edt(done_guid, i + 1, NULL_GUID);
      }
    }
  }
}

extern "C" void clean_per_gpu(unsigned int node_id, int dev_id,
                              cudaStream_t *stream) {
  (void)node_id;
  (void)dev_id;
  (void)stream;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
