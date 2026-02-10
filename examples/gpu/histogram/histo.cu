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

#define ARRAYSIZE 1024 * 1024
#define TILESIZE 128
// #define VERIFY 1
// #define VERIFYONGPU 0
#define SMTILE 32  // Hardcoded for Volta
#define NUMBINS 10 // Make it a variable

#define ARTS_PRINTF(...)
//  #define ARTS_PRINTF(...) ARTS_PRINTF(__VA_ARGS__)

uint64_t start = 0;

unsigned int inputArraySize;
unsigned int tile_size;
unsigned int num_blocks = 1;

arts_guid_t inputArrayGuid = NULL_GUID;
arts_guid_t histoGuid = NULL_GUID;
arts_guid_t done_guid = NULL_GUID;
arts_guid_t finalSumGuid = NULL_GUID;

unsigned int *inputArray = NULL;
unsigned int *finalHistogram = NULL;

arts_guid_t *inputTileGuids = NULL;
arts_guid_t *partialHistoGuids = NULL;

__global__ void privateHistogram(uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc, arts_edt_dep_t depv[]) {
  const unsigned int num_elements = (unsigned int)paramv[0];
  unsigned int *tile = (unsigned int *)depv[0].ptr;
  unsigned int *localHisto = (unsigned int *)depv[1].ptr;

  // Compute histograms in every GPU
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x; // 32

  step = blockDim.x * gridDim.x; // 0-8192 /32 => 0-31, 32-63...
  for (unsigned int i = index; i < num_elements; i += step)
    if (i < num_elements)
      atomicAdd(&localHisto[tile[i]], 1);
#if VERIFYONGPU
  __syncthreads();
  if (index == 0) {
    for (unsigned int i = 0; i < num_elements; i++)
      ARTS_PRINTF("input[%u] = %u\n", i, tile[i]);

    for (unsigned int i = 0; i < NUMBINS; i++)
      ARTS_PRINTF("\thisto[%u] = %u\n", i, localHisto[i]);
  }
  __syncthreads();
#endif
}

__global__ void reduceHistogram(uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, arts_edt_dep_t depv[]) {
  // Reduce histograms from all GPUs.
  const unsigned int numLocalHistograms = depc - 1;
  unsigned int *finalHisto = (unsigned int *)depv[0].ptr;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: This would work if localHisto is made private to blockIdx.x
  // if (blockIdx.x < numLocalHistograms)
  // {
  //     int * localHisto = (int *) depv[1+blockIdx.x].ptr;
  //     if (threadIdx.x < NUMBINS)
  //         atomicAdd(&finalHisto[threadIdx.x], localHisto[threadIdx.x]);
  // }

  if (blockIdx.x == 0) {
    for (unsigned int i = 0; i < numLocalHistograms; i++) {
      unsigned int *localHisto = (unsigned int *)depv[1 + i].ptr;
      if (index < NUMBINS)
        atomicAdd(&finalHisto[index], localHisto[index]);
    }
  }

#if VERIFYONGPU
  __syncthreads();
  if (index == 0) {
    for (unsigned int i = 0; i < numLocalHistograms; i++) {
      unsigned int *localHisto = (unsigned int *)depv[1 + i].ptr;
      ARTS_PRINTF("localHisto[%d]\n", i);
      for (unsigned int j = 0; j < NUMBINS; j++)
        ARTS_PRINTF("\thisto[%d] = %d\n", j, localHisto[j]);
    }
  }
#endif
}

void finishHistogram(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  uint64_t time = arts_get_time_stamp() - start;
#if VERIFY
  unsigned int *histoObtained = (unsigned int *)depv[0].ptr;
  unsigned int *histoExpected =
      (unsigned int *)arts_calloc(NUMBINS, sizeof(unsigned int));

  for (unsigned int i = 0; i < inputArraySize; i++)
    histoExpected[inputArray[i]]++;

  for (unsigned int i = 0; i < NUMBINS; i++)
    ARTS_PRINTF("histo[%u] = %u | finalHisto[%u] = %u\n", i, histoExpected[i], i,
           histoObtained[i]);

  for (unsigned int i = 0; i < NUMBINS; i++) {
    if (histoExpected[i] != histoObtained[i]) {
      ARTS_PRINTF("Failed at histo[%u]\n", i);
      ARTS_PRINTF("Expected: %u | Obtained: %u\n", histoExpected[i],
             histoObtained[i]);
      arts_free(histoExpected);
      arts_shutdown();
      return;
    }
  }
  arts_free(histoExpected);
  ARTS_PRINTF("Success %lu\n", time);
#else
  ARTS_PRINTF("Done %lu\n", time);
#endif
  arts_shutdown();
  return;
}

extern "C" void init_per_node(unsigned int node_id, int argc, char **argv) {
  if (argc == 1) {
    inputArraySize = ARRAYSIZE;
    tile_size = TILESIZE;
  } else if (argc == 2) {
    inputArraySize = atoi(argv[1]);
    tile_size = TILESIZE;
  } else {
    inputArraySize = atoi(argv[1]);
    tile_size = atoi(argv[2]);
  }

  num_blocks = (inputArraySize + tile_size - 1) /
              tile_size; // TODO: Fix if inputArraySize is < tile_size

  if (!node_id)
    ARTS_PRINTF("ArraySize = %u | tile_size = %u | num_blocks: %u | num_gpus: %u\n",
           inputArraySize, tile_size, num_blocks, arts_get_total_gpus());

  done_guid = arts_reserve_guid_route(ARTS_EDT, 0);
  finalSumGuid = arts_reserve_guid_route(ARTS_GPU_EDT, 0);
  histoGuid = arts_reserve_guid_route(ARTS_DB_GPU_WRITE, 0);

  inputTileGuids = arts_reserve_guids_round_robin(num_blocks, ARTS_DB_GPU_READ);
  partialHistoGuids = arts_reserve_guids_round_robin(num_blocks, ARTS_DB_GPU_WRITE);

  if (!node_id) {
    finalHistogram = (unsigned int *)arts_db_create_with_guid(
        histoGuid, NUMBINS * sizeof(unsigned int));
    memset(finalHistogram, 0, NUMBINS * sizeof(unsigned int));
  }

  inputArray = (unsigned int *)arts_calloc(inputArraySize, sizeof(unsigned int));

  if (!node_id)
    ARTS_PRINTF("Loading input array with seed 7\n");

  srand(7);
  for (unsigned int elem = 0; elem < inputArraySize; elem++)
    inputArray[elem] = (unsigned int)(rand() % NUMBINS);
}

extern "C" void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  dim3 threads(SMTILE);
  dim3 grid((tile_size + SMTILE - 1) / SMTILE);

  if (!worker_id) {
    if (!node_id) {
      arts_edt_create_with_guid(finishHistogram, done_guid, 0, NULL, 2);
      arts_signal_edt(done_guid, 0, histoGuid);

      arts_edt_create_gpu_with_guid(reduceHistogram, finalSumGuid, 0, NULL,
                               num_blocks + 1, grid, threads, done_guid, 0,
                               histoGuid);
      arts_signal_edt(finalSumGuid, 0, histoGuid);
    }

    for (unsigned int tile = 0; tile < num_blocks; tile++) {
      arts_guid_t inputTileGuid = inputTileGuids[tile];
      arts_guid_t partialHistoGuid = partialHistoGuids[tile];
      assert(arts_guid_get_rank(inputTileGuid) ==
             arts_guid_get_rank(partialHistoGuid));

      if (arts_guid_get_rank(inputTileGuid) == node_id) {
        // Initialize the tile
        unsigned int *inputTile = (unsigned int *)arts_db_create_with_guid(
            inputTileGuid, sizeof(unsigned int) * tile_size);
        memcpy(inputTile, &inputArray[tile * tile_size],
               tile_size * sizeof(unsigned int));

        unsigned int *partialHisto = (unsigned int *)arts_db_create_with_guid(
            partialHistoGuid, sizeof(unsigned int) * NUMBINS);
        memset(partialHisto, 0, NUMBINS * sizeof(unsigned int));

        uint64_t args[] = {tile_size};
        arts_guid_t privHistoGuid =
            arts_edt_create_gpu(privateHistogram, node_id, 2, args, 2, grid,
                             threads, finalSumGuid, 1 + tile, partialHistoGuid);
        arts_signal_edt(privHistoGuid, 0, inputTileGuid);
        arts_signal_edt(privHistoGuid, 1, partialHistoGuid);
      }
    }
  }

  if (!node_id && !worker_id) {
    ARTS_PRINTF("Starting...\n");
    start = arts_get_time_stamp();
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}