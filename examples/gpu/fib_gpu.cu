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

uint64_t start = 0;

// This is the GPU kernel
__global__ void fib_join(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  unsigned int *x = (unsigned int *)depv[0].ptr;
  unsigned int *y = (unsigned int *)depv[1].ptr;
  unsigned int *res = (unsigned int *)depv[2].ptr;
  (*res) = (*x) + (*y);
}

void fib_fork(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  unsigned int next = 0; //(arts_get_current_node() + 1) % arts_get_total_nodes();
  //    ARTS_PRINTF("NODE: %u WORKER: %u NEXT: %u\n", arts_get_current_node(),
  //    arts_get_current_worker(), next);

  arts_guid_t done_guid = paramv[0];
  unsigned int slot = (unsigned int)paramv[1];

  arts_guid_t resGuid = depv[0].guid;
  unsigned int *resPtr = (unsigned int *)depv[0].ptr;

  if ((*resPtr) < 2)
    arts_signal_edt(done_guid, slot, resGuid);
  else {
    // Create two DB of type ARTS_DB_GPU
    unsigned int *x = NULL;
    arts_guid_t x_guid =
        arts_db_create((void **)&x, sizeof(unsigned int), ARTS_DB_GPU_WRITE);
    (*x) = (*resPtr) - 1;

    unsigned int *y = NULL;
    arts_guid_t y_guid =
        arts_db_create((void **)&y, sizeof(unsigned int), ARTS_DB_GPU_WRITE);
    (*y) = (*resPtr) - 2;

    // Create a continuation edt to run on the GPU
    dim3 grid(1);
    dim3 block(1);
    arts_guid_t join_guid = arts_edt_create_gpu(fib_join, next, 0, NULL, 3, grid,
                                           block, done_guid, slot, resGuid);
    arts_signal_edt(join_guid, 2, resGuid);

    // Create the forks which will run on the CPU
    uint64_t args[2] = {(uint64_t)join_guid, 0};
    arts_guid_t forkGuidX = arts_edt_create(fib_fork, next, 2, args, 1);
    arts_signal_edt(forkGuidX, 0, x_guid);

    args[1] = 1;
    arts_guid_t forkGuidY = arts_edt_create(fib_fork, next, 2, args, 1);
    arts_signal_edt(forkGuidY, 0, y_guid);
  }
}

void fib_done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  uint64_t time = arts_get_time_stamp() - start;
  unsigned int *resPtr = (unsigned int *)depv[0].ptr;
  ARTS_PRINTF("Fib %u: %u time: %lu nodes: %u workers: %u\n", paramv[0], *resPtr,
         time, arts_get_total_nodes(), arts_get_total_workers());
  arts_shutdown();
}

extern "C" void init_per_node(unsigned int node_id, int argc, char **argv) {}

extern "C" void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  if (!node_id && !worker_id) {
    unsigned int *resPtr = NULL;
    arts_guid_t resGuid =
        arts_db_create((void **)&resPtr, sizeof(unsigned int), ARTS_DB_GPU_WRITE);
    if (argc < 2) {
      ARTS_PRINTF("Format: ./fibGpu NUMBER\n");
      arts_shutdown();
      return;
    }
    *resPtr = atoi(argv[1]);

    arts_guid_t done_guid = arts_edt_create(fib_done, 0, 1, (uint64_t *)resPtr, 1);

    uint64_t args[] = {(uint64_t)done_guid, 0};
    arts_guid_t fibGuid = arts_edt_create(fib_fork, 0, 2, args, 1);
    arts_signal_edt(fibGuid, 0, resGuid);
    start = arts_get_time_stamp();
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
