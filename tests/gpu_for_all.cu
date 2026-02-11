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

#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"

#define GPULISTLEN 32

unsigned int **dev_ptr_raw; // This is a list of the search frontier in global
                            // memory on each gpu

// This will probably be where you want to do the actual traversal
__global__ void temp(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  // unsigned int gpu_id = (unsigned int) paramv[0]; //The current gpu we are on
  uint64_t gpu_id = GET_GPU_INDEX();
  unsigned int **addr =
      (unsigned int **)depv[0].ptr;  // This is the dev_ptr_raw -> tells us where
                                     // current frontier is on device
  unsigned int *local = addr[gpu_id]; // We need the one corresponding to our gpu

  unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);
  local[(GPULISTLEN - 1) - index] =
      (unsigned int)gpu_id; // index; //Just writing some blah blah value to sort
}

// This should be where we do the sorting and should launch the next iteration
void thrust_sort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t done_guid =
      (arts_guid_t)paramv[0]; // This can be the end if the frontier is empty
  unsigned int gpu_index = (unsigned int)paramv[1]; // gpu_index
  unsigned int *raw_ptr = dev_ptr_raw[gpu_index];   // The corresponding dev pointer
                                                    // (frontier) to our gpu

  unsigned int *tile = NULL; // This will hold a tile of the new frontier
  arts_guid_t tile_guid = arts_guid_reserve(ARTS_DB_GPU_READ, 0);
  tile = (unsigned int *)arts_db_create_with_guid(tile_guid, sizeof(unsigned int) * GPULISTLEN, NULL);

  thrust::device_ptr<unsigned int> dev_thrust_ptr(raw_ptr);
  thrust::sort(dev_thrust_ptr, dev_thrust_ptr + GPULISTLEN); // Do the sorting

  // Copy the data from the gpu to the host
  arts_put_in_db_from_gpu(thrust::raw_pointer_cast(dev_thrust_ptr), tile_guid, 0,
                     sizeof(unsigned int) * GPULISTLEN, false);

  // Probably should make some new edts and signal them with the data!
  // Or signal the end if we are done
  arts_signal_edt(
      done_guid, gpu_index,
      tile_guid); // don't really need tile_guid just doing it for testing
}

void done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  // This is just for testing...
  // We should see it is sorted
  for (unsigned int i = 0; i < depc; i++) {
    unsigned int *tile = (unsigned int *)depv[i].ptr;
    printf("GPU %u: ", i);
    for (unsigned int j = 0; j < GPULISTLEN; j++) {
      printf("%u, ", tile[j]);
    }
    printf("\n");
  }
  arts_shutdown();
}

extern "C" void arts_init_per_gpu(unsigned int node_id, int dev_id,
                             cudaStream_t *stream, int argc, char **argv) {
  (void)node_id;
  (void)stream;
  (void)argc;
  (void)argv;
  if (!dev_id) {
    dev_ptr_raw =
        (unsigned int **)calloc(arts_get_total_gpus(), sizeof(unsigned int *));
  }
  dev_ptr_raw[dev_id] =
      (unsigned int *)arts_cuda_malloc(sizeof(unsigned int) * GPULISTLEN);
}

extern "C" void arts_main_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int node_id = arts_get_current_node();
  unsigned int **addr;
  arts_guid_t db_guid = arts_guid_reserve(ARTS_DB_GPU_READ, 0);
  addr = (unsigned int **)arts_db_create_with_guid(db_guid, sizeof(unsigned int *) * arts_get_total_gpus(), NULL);
  for (uint64_t i = 0; i < arts_get_total_gpus(); i++) {
    addr[i] = dev_ptr_raw[i];
  }

  arts_guid_t done_guid = arts_edt_create(done, 0, NULL, arts_get_total_gpus(), &(arts_hint_t){.route = 0});

  dim3 threads(GPULISTLEN, 1, 1);
  dim3 grid(1, 1, 1);
  for (uint64_t i = 0; i < arts_get_total_gpus(); i++) {
    uint64_t args[] = {(uint64_t)done_guid, i};
    arts_guid_t edt_guid = arts_edt_create_gpu_lib_direct(thrust_sort, node_id, i, 2,
                                                   args, 1, grid, threads);
    arts_guid_t edt_guid2 = arts_edt_create_gpu_direct(
        temp, node_id, i, 1, &i, 1, grid, threads, edt_guid, 0, db_guid, true);
    arts_signal_edt(edt_guid2, 0, db_guid);
  }
}

extern "C" void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                              cudaStream_t *stream) {
  (void)node_id;
  (void)stream;
  arts_cuda_free(dev_ptr_raw[dev_id]);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
