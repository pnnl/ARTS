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
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *x = (unsigned int *)depv[0].ptr;
  unsigned int *y = (unsigned int *)depv[1].ptr;
  unsigned int *res = (unsigned int *)depv[2].ptr;
  (*res) = (*x) + (*y);
}

void fib_fork(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int next =
      0; //(arts_get_current_node() + 1) % arts_get_total_nodes();
  //    arts_printf("NODE: %u WORKER: %u NEXT: %u\n", arts_get_current_node(),
  //    arts_get_current_worker(), next);

  arts_guid_t done_guid = (arts_guid_t)paramv[0];
  unsigned int slot = (unsigned int)paramv[1];

  arts_guid_t res_guid = depv[0].guid;
  unsigned int *res_ptr = (unsigned int *)depv[0].ptr;

  if ((*res_ptr) < 2) {
    arts_signal_edt(done_guid, slot, res_guid, ARTS_DB_WRITE);
  } else {
    // Create two DB of type ARTS_DB_GPU
    unsigned int *x = NULL;
    arts_guid_t x_guid = arts_guid_reserve(ARTS_DB_GPU_WRITE, 0);
    x = (unsigned int *)arts_db_create_with_guid(x_guid, sizeof(unsigned int),
                                                 NULL);
    (*x) = (*res_ptr) - 1;

    unsigned int *y = NULL;
    arts_guid_t y_guid = arts_guid_reserve(ARTS_DB_GPU_WRITE, 0);
    y = (unsigned int *)arts_db_create_with_guid(y_guid, sizeof(unsigned int),
                                                 NULL);
    (*y) = (*res_ptr) - 2;

    // Create a continuation edt to run on the GPU
    dim3 grid(1);
    dim3 block(1);
    arts_guid_t join_guid = arts_edt_create_gpu(
        fib_join, next, 0, NULL, 3, grid, block, done_guid, slot, res_guid);
    arts_signal_edt(join_guid, 2, res_guid, ARTS_DB_WRITE);

    // Create the forks which will run on the CPU
    uint64_t args[2] = {(uint64_t)join_guid, 0};
    arts_hint_t hint_0 = {next, 0};
    arts_guid_t fork_guid_x =
        arts_edt_create(fib_fork, 2, args, 1, &hint_0);
    arts_signal_edt(fork_guid_x, 0, x_guid, ARTS_DB_WRITE);

    args[1] = 1;
    arts_hint_t hint_1 = {next, 0};
    arts_guid_t fork_guid_y =
        arts_edt_create(fib_fork, 2, args, 1, &hint_1);
    arts_signal_edt(fork_guid_y, 0, y_guid, ARTS_DB_WRITE);
  }
}

void fib_done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t time = arts_get_time_stamp() - start;
  unsigned int *res_ptr = (unsigned int *)depv[0].ptr;
  arts_printf("Fib %u: %u time: %lu nodes: %u workers: %u\n", paramv[0],
              *res_ptr, time, arts_get_total_nodes(), arts_get_total_workers());
  arts_shutdown();
}

extern "C" void arts_main_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];

  unsigned int *res_ptr = NULL;
  arts_guid_t res_guid = arts_guid_reserve(ARTS_DB_GPU_WRITE, 0);
  res_ptr = (unsigned int *)arts_db_create_with_guid(
      res_guid, sizeof(unsigned int), NULL);
  if (argc < 2) {
    arts_printf("Format: ./fibGpu NUMBER\n");
    arts_shutdown();
    return;
  }
  *res_ptr = (unsigned int)strtol(argv[1], NULL, 10);

  uint64_t done_args[] = {(uint64_t)*res_ptr};
  arts_hint_t hint_2 = {0, 0};
  arts_guid_t done_guid =
      arts_edt_create(fib_done, 1, done_args, 1, &hint_2);

  uint64_t args[] = {(uint64_t)done_guid, 0};
  arts_hint_t hint_3 = {0, 0};
  arts_guid_t fib_guid =
      arts_edt_create(fib_fork, 2, args, 1, &hint_3);
  arts_signal_edt(fib_guid, 0, res_guid, ARTS_DB_WRITE);
  start = arts_get_time_stamp();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
