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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"

#define ARRAYSIZE (1024 * 1024)
#define TILESIZE 128
// #define VERIFY 1
// #define VERIFYONGPU 0
#define SMTILE 32  // Hardcoded for Volta
#define NUMBINS 10 // Make it a variable

#define ARTS_PRINTF(...)
//  #define ARTS_PRINTF(...) ARTS_PRINTF(__VA_ARGS__)

uint64_t start = 0;

unsigned int input_array_size;
unsigned int tile_size;
unsigned int num_blocks = 1;

arts_guid_t input_array_guid = NULL_GUID;
arts_guid_t histo_guid = NULL_GUID;
arts_guid_t done_guid = NULL_GUID;
arts_guid_t final_sum_guid = NULL_GUID;

unsigned int *input_array = NULL;
unsigned int *final_histogram = NULL;

arts_guid_t *input_tile_guids = NULL;
arts_guid_t *partial_histo_guids = NULL;

__global__ void private_histogram(uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const unsigned int num_elements = (unsigned int)paramv[0];
  unsigned int *tile = (unsigned int *)depv[0].ptr;
  unsigned int *local_histo = (unsigned int *)depv[1].ptr;

  // Compute histograms in every GPU
  unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int step = blockDim.x; // 32

  step = blockDim.x * gridDim.x; // 0-8192 /32 => 0-31, 32-63...
  for (unsigned int i = index; i < num_elements; i += step) {
    if (i < num_elements) {
      atomicAdd(&local_histo[tile[i]], 1);
    }
  }
#if VERIFYONGPU
  __syncthreads();
  if (index == 0) {
    for (unsigned int i = 0; i < num_elements; i++) {
      ARTS_PRINTF("input[%u] = %u\n", i, tile[i]);
    }

    for (unsigned int i = 0; i < NUMBINS; i++) {
      ARTS_PRINTF("\thisto[%u] = %u\n", i, local_histo[i]);
    }
  }
  __syncthreads();
#endif
}

__global__ void reduce_histogram(uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  // Reduce histograms from all GPUs.
  const unsigned int num_local_histograms = depc - 1;
  unsigned int *final_histo = (unsigned int *)depv[0].ptr;
  unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  // TODO: This would work if local_histo is made private to blockIdx.x
  // if (blockIdx.x < num_local_histograms)
  // {
  //     int * local_histo = (int *) depv[1+blockIdx.x].ptr;
  //     if (threadIdx.x < NUMBINS)
  //         atomicAdd(&final_histo[threadIdx.x], local_histo[threadIdx.x]);
  // }

  if (blockIdx.x == 0) {
    for (unsigned int i = 0; i < num_local_histograms; i++) {
      unsigned int *local_histo = (unsigned int *)depv[1 + i].ptr;
      if (index < NUMBINS) {
        atomicAdd(&final_histo[index], local_histo[index]);
      }
    }
  }

#if VERIFYONGPU
  __syncthreads();
  if (index == 0) {
    for (unsigned int i = 0; i < num_local_histograms; i++) {
      unsigned int *local_histo = (unsigned int *)depv[1 + i].ptr;
      ARTS_PRINTF("local_histo[%d]\n", i);
      for (unsigned int j = 0; j < NUMBINS; j++) {
        ARTS_PRINTF("\thisto[%d] = %d\n", j, local_histo[j]);
      }
    }
  }
#endif
}

void finish_histogram(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  uint64_t time = arts_get_time_stamp() - start;
#if VERIFY
  unsigned int *histo_obtained = (unsigned int *)depv[0].ptr;
  unsigned int *histo_expected =
      (unsigned int *)calloc(NUMBINS, sizeof(unsigned int));

  for (unsigned int i = 0; i < input_array_size; i++) {
    histo_expected[input_array[i]]++;
  }

  for (unsigned int i = 0; i < NUMBINS; i++) {
    ARTS_PRINTF("histo[%u] = %u | finalHisto[%u] = %u\n", i, histo_expected[i], i,
           histo_obtained[i]);
  }

  for (unsigned int i = 0; i < NUMBINS; i++) {
    if (histo_expected[i] != histo_obtained[i]) {
      ARTS_PRINTF("Failed at histo[%u]\n", i);
      ARTS_PRINTF("Expected: %u | Obtained: %u\n", histo_expected[i],
             histo_obtained[i]);
      free(histo_expected);
      arts_shutdown();
      return;
    }
  }
  free(histo_expected);
  ARTS_PRINTF("Success %lu\n", time);
#else
  ARTS_PRINTF("Done %lu\n", time);
#endif
  arts_shutdown();
}

extern "C" void arts_main_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  unsigned int node_id = arts_get_current_node();

  if (argc == 1) {
    input_array_size = ARRAYSIZE;
    tile_size = TILESIZE;
  } else if (argc == 2) {
    input_array_size = (unsigned int)strtol(argv[1], NULL, 10);
    tile_size = TILESIZE;
  } else {
    input_array_size = (unsigned int)strtol(argv[1], NULL, 10);
    tile_size = (unsigned int)strtol(argv[2], NULL, 10);
  }

  num_blocks = (input_array_size + tile_size - 1) / tile_size;

  ARTS_PRINTF("ArraySize = %u | tile_size = %u | num_blocks: %u | num_gpus: %u\n",
         input_array_size, tile_size, num_blocks, arts_get_total_gpus());

  done_guid = arts_guid_reserve(ARTS_EDT, 0);
  final_sum_guid = arts_guid_reserve(ARTS_GPU_EDT, 0);
  histo_guid = arts_guid_reserve(ARTS_DB_GPU_WRITE, 0);

  input_tile_guids = arts_guid_reserve_round_robin(num_blocks, ARTS_DB_GPU_READ);
  partial_histo_guids = arts_guid_reserve_round_robin(num_blocks, ARTS_DB_GPU_WRITE);

  final_histogram = (unsigned int *)arts_db_create_with_guid(
      histo_guid, NUMBINS * sizeof(unsigned int), NULL);
  memset(final_histogram, 0, NUMBINS * sizeof(unsigned int));

  input_array = (unsigned int *)calloc(input_array_size, sizeof(unsigned int));

  ARTS_PRINTF("Loading input array with seed 7\n");

  srand(7); // NOLINT(cert-msc32-c,cert-msc51-cpp)
  for (unsigned int elem = 0; elem < input_array_size; elem++) {
    input_array[elem] = (unsigned int)(rand() % NUMBINS); // NOLINT(cert-msc30-c,cert-msc50-cpp)
  }

  dim3 threads(SMTILE);
  dim3 grid((tile_size + SMTILE - 1) / SMTILE);

  arts_edt_create_with_guid(finish_histogram, done_guid, 0, NULL, 2);
  arts_signal_edt(done_guid, 0, histo_guid, ARTS_DB_WRITE);

  arts_edt_create_gpu_with_guid(reduce_histogram, final_sum_guid, 0, NULL,
                             num_blocks + 1, grid, threads, done_guid, 0,
                             histo_guid);
  arts_signal_edt(final_sum_guid, 0, histo_guid, ARTS_DB_WRITE);

  for (unsigned int tile = 0; tile < num_blocks; tile++) {
    arts_guid_t input_tile_guid = input_tile_guids[tile];
    arts_guid_t partial_histo_guid = partial_histo_guids[tile];
    assert(arts_guid_get_rank(input_tile_guid) ==
           arts_guid_get_rank(partial_histo_guid));

    if (arts_guid_get_rank(input_tile_guid) == node_id) {
      unsigned int *input_tile = (unsigned int *)arts_db_create_with_guid(
          input_tile_guid, sizeof(unsigned int) * tile_size, NULL);
      memcpy(input_tile, &input_array[(size_t)tile * tile_size],
             tile_size * sizeof(unsigned int));

      unsigned int *partial_histo = (unsigned int *)arts_db_create_with_guid(
          partial_histo_guid, sizeof(unsigned int) * NUMBINS, NULL);
      memset(partial_histo, 0, NUMBINS * sizeof(unsigned int));

      uint64_t args[] = {tile_size};
      arts_guid_t priv_histo_guid =
          arts_edt_create_gpu(private_histogram, node_id, 2, args, 2, grid,
                           threads, final_sum_guid, 1 + tile, partial_histo_guid);
      arts_signal_edt(priv_histo_guid, 0, input_tile_guid, ARTS_DB_WRITE);
      arts_signal_edt(priv_histo_guid, 1, partial_histo_guid, ARTS_DB_WRITE);
    }
  }

  ARTS_PRINTF("Starting...\n");
  start = arts_get_time_stamp();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
