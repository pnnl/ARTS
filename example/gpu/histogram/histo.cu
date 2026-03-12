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

#include "arts/arts.h"
#include "arts/gpu/GpuRuntime.cuh"

#define ARRAYSIZE 1024 * 1024
#define TILESIZE 128
// #define VERIFY 1
// #define VERIFYONGPU 0
#define SMTILE 32  // Hardcoded for Volta
#define NUMBINS 10 // Make it a variable

#define PRINTF(...)
//  #define PRINTF(...) PRINTF(__VA_ARGS__)

uint64_t start = 0;

unsigned int inputArraySize;
unsigned int tileSize;
unsigned int numBlocks = 1;

artsGuid_t inputArrayGuid = NULL_GUID;
artsGuid_t histoGuid = NULL_GUID;
artsGuid_t doneGuid = NULL_GUID;
artsGuid_t finalSumGuid = NULL_GUID;

unsigned int *inputArray = NULL;
unsigned int *finalHistogram = NULL;

artsGuid_t *inputTileGuids = NULL;
artsGuid_t *partialHistoGuids = NULL;

__global__ void privateHistogram(uint32_t paramc, uint64_t *paramv,
                                 uint32_t depc, artsEdtDep_t depv[]) {
  const unsigned int numElements = (unsigned int)paramv[0];
  unsigned int *tile = (unsigned int *)depv[0].ptr;
  unsigned int *localHisto = (unsigned int *)depv[1].ptr;

  // Compute histograms in every GPU
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x; // 32

  step = blockDim.x * gridDim.x; // 0-8192 /32 => 0-31, 32-63...
  for (unsigned int i = index; i < numElements; i += step)
    if (i < numElements)
      atomicAdd(&localHisto[tile[i]], 1);
#if VERIFYONGPU
  __syncthreads();
  if (index == 0) {
    for (unsigned int i = 0; i < numElements; i++)
      PRINTF("input[%u] = %u\n", i, tile[i]);

    for (unsigned int i = 0; i < NUMBINS; i++)
      PRINTF("\thisto[%u] = %u\n", i, localHisto[i]);
  }
  __syncthreads();
#endif
}

__global__ void reduceHistogram(uint32_t paramc, uint64_t *paramv,
                                uint32_t depc, artsEdtDep_t depv[]) {
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
      PRINTF("localHisto[%d]\n", i);
      for (unsigned int j = 0; j < NUMBINS; j++)
        PRINTF("\thisto[%d] = %d\n", j, localHisto[j]);
    }
  }
#endif
}

void finishHistogram(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                     artsEdtDep_t depv[]) {
  uint64_t time = artsGetTimeStamp() - start;
#if VERIFY
  unsigned int *histoObtained = (unsigned int *)depv[0].ptr;
  unsigned int *histoExpected =
      (unsigned int *)artsCalloc(NUMBINS, sizeof(unsigned int));

  for (unsigned int i = 0; i < inputArraySize; i++)
    histoExpected[inputArray[i]]++;

  for (unsigned int i = 0; i < NUMBINS; i++)
    PRINTF("histo[%u] = %u | finalHisto[%u] = %u\n", i, histoExpected[i], i,
           histoObtained[i]);

  for (unsigned int i = 0; i < NUMBINS; i++) {
    if (histoExpected[i] != histoObtained[i]) {
      PRINTF("Failed at histo[%u]\n", i);
      PRINTF("Expected: %u | Obtained: %u\n", histoExpected[i],
             histoObtained[i]);
      artsFree(histoExpected);
      artsShutdown();
      return;
    }
  }
  artsFree(histoExpected);
  PRINTF("Success %lu\n", time);
#else
  PRINTF("Done %lu\n", time);
#endif
  artsShutdown();
  return;
}

extern "C" void initPerNode(unsigned int nodeId, int argc, char **argv) {
  if (argc == 1) {
    inputArraySize = ARRAYSIZE;
    tileSize = TILESIZE;
  } else if (argc == 2) {
    inputArraySize = atoi(argv[1]);
    tileSize = TILESIZE;
  } else {
    inputArraySize = atoi(argv[1]);
    tileSize = atoi(argv[2]);
  }

  numBlocks = (inputArraySize + tileSize - 1) /
              tileSize; // TODO: Fix if inputArraySize is < tileSize

  if (!nodeId)
    PRINTF("ArraySize = %u | tileSize = %u | numBlocks: %u | numGpus: %u\n",
           inputArraySize, tileSize, numBlocks, artsGetTotalGpus());

  doneGuid = artsReserveGuidRoute(ARTS_EDT, 0);
  finalSumGuid = artsReserveGuidRoute(ARTS_GPU_EDT, 0);
  histoGuid = artsReserveGuidRoute(ARTS_DB_GPU_WRITE, 0);

  inputTileGuids = artsReserveGuidsRoundRobin(numBlocks, ARTS_DB_GPU_READ);
  partialHistoGuids = artsReserveGuidsRoundRobin(numBlocks, ARTS_DB_GPU_WRITE);

  if (!nodeId) {
    finalHistogram = (unsigned int *)artsDbCreateWithGuid(
        histoGuid, NUMBINS * sizeof(unsigned int));
    memset(finalHistogram, 0, NUMBINS * sizeof(unsigned int));
  }

  inputArray = (unsigned int *)artsCalloc(inputArraySize, sizeof(unsigned int));

  if (!nodeId)
    PRINTF("Loading input array with seed 7\n");

  srand(7);
  for (unsigned int elem = 0; elem < inputArraySize; elem++)
    inputArray[elem] = (unsigned int)(rand() % NUMBINS);
}

extern "C" void initPerWorker(unsigned int nodeId, unsigned int workerId,
                              int argc, char **argv) {
  dim3 threads(SMTILE);
  dim3 grid((tileSize + SMTILE - 1) / SMTILE);

  if (!workerId) {
    if (!nodeId) {
      artsEdtCreateWithGuid(finishHistogram, doneGuid, 0, NULL, 2);
      artsSignalEdt(doneGuid, 0, histoGuid);

      artsEdtCreateGpuWithGuid(reduceHistogram, finalSumGuid, 0, NULL,
                               numBlocks + 1, grid, threads, doneGuid, 0,
                               histoGuid);
      artsSignalEdt(finalSumGuid, 0, histoGuid);
    }

    for (unsigned int tile = 0; tile < numBlocks; tile++) {
      artsGuid_t inputTileGuid = inputTileGuids[tile];
      artsGuid_t partialHistoGuid = partialHistoGuids[tile];
      assert(artsGuidGetRank(inputTileGuid) ==
             artsGuidGetRank(partialHistoGuid));

      if (artsGuidGetRank(inputTileGuid) == nodeId) {
        // Initialize the tile
        unsigned int *inputTile = (unsigned int *)artsDbCreateWithGuid(
            inputTileGuid, sizeof(unsigned int) * tileSize);
        memcpy(inputTile, &inputArray[tile * tileSize],
               tileSize * sizeof(unsigned int));

        unsigned int *partialHisto = (unsigned int *)artsDbCreateWithGuid(
            partialHistoGuid, sizeof(unsigned int) * NUMBINS);
        memset(partialHisto, 0, NUMBINS * sizeof(unsigned int));

        uint64_t args[] = {tileSize};
        artsGuid_t privHistoGuid =
            artsEdtCreateGpu(privateHistogram, nodeId, 2, args, 2, grid,
                             threads, finalSumGuid, 1 + tile, partialHistoGuid);
        artsSignalEdt(privHistoGuid, 0, inputTileGuid);
        artsSignalEdt(privHistoGuid, 1, partialHistoGuid);
      }
    }
  }

  if (!nodeId && !workerId) {
    PRINTF("Starting...\n");
    start = artsGetTimeStamp();
  }
}

int main(int argc, char **argv) {
  artsRT(argc, argv);
  return 0;
}