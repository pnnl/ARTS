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

#include "arts.h"
#include "arts/utils/array_list.h"

#define CHECKCORRECT(x)                                                        \
  {                                                                            \
    cudaError_t err;                                                           \
    if ((err = (x)) != cudaSuccess)                                            \
      ARTS_PRINTF("FAILED %s: %s\n", #x, cudaGetErrorString(err));                  \
  }

#define MAX(a, b) ((a > b) ? a : b)

unsigned int *maxCycle = NULL;
arts_array_list_t *gpuList = NULL;
bool **adjList = NULL;
unsigned int order = 0;

bool setMaxCycle(unsigned int *cycle, unsigned int cycleSize) {
  uint64_t length = arts_length_array_list(gpuList);
  if (cycle[0] == cycle[cycleSize - 1]) {
    for (unsigned int i = 0; i < cycleSize - 1; i++) {
      unsigned int found = 0;
      for (uint64_t j = 0; j < length; j++) {
        int *temp = (int *)arts_get_from_array_list(gpuList, i);
        if (cycle[j] == *temp)
          found++;
      }
      if (found != 1)
        return false;
    }

    maxCycle = (unsigned int *)arts_calloc(cycleSize, sizeof(unsigned int));
    for (unsigned int i = 0; i < cycleSize; i++)
      maxCycle[i] = cycle[i];
    return true;
  }
  return false;
}

bool depthFirstRec(unsigned int vertex, unsigned int current,
                   unsigned int cycleSize, unsigned int *cycle) {
  cycle[current] = vertex;

  if (current + 1 == cycleSize) {
    return setMaxCycle(cycle, cycleSize);
  }
  for (unsigned int i = 0; i < order; i++) {
    if (adjList[vertex][i])
      if (depthFirstRec(i, current + 1, cycleSize, cycle))
        return true;
  }
  return false;
}

void depthFirst(unsigned int cycleSize) {
  unsigned int *cycle =
      (unsigned int *)arts_calloc(order + 1, sizeof(unsigned int));
  for (unsigned int i = 0; i < order; i++) {
    if (depthFirstRec(i, 0, cycleSize, cycle)) {
      for (unsigned int i = 0; i < cycleSize; i++)
        printf("%u ", maxCycle[i]);
      printf("\n");
      return;
    }
  }
}

bool **fullyConnect() {
  bool **adjList = (bool **)arts_calloc(order, sizeof(bool *));
  for (unsigned int i = 0; i < order; i++)
    adjList[i] = (bool *)arts_calloc(order, sizeof(bool));

  uint64_t length = arts_length_array_list(gpuList);
  for (uint64_t i = 0; i < length; i++) {
    int *src = (int *)arts_get_from_array_list(gpuList, i);
    CHECKCORRECT(cudaSetDevice(*src));
    for (uint64_t j = 0; j < length; j++) {
      if (i != j) {
        int hasAccess = 0;
        int *dst = (int *)arts_get_from_array_list(gpuList, j);
        CHECKCORRECT(cudaDeviceCanAccessPeer(&hasAccess, *src, *dst));
        if (hasAccess) {
          adjList[*src][*dst] = 1;
          CHECKCORRECT(cudaDeviceEnablePeerAccess(*dst, 0));
        }
      }
    }
  }
  return adjList;
}

void printAdjList() {
  for (unsigned int i = 0; i < order; i++) {
    ARTS_PRINTF("%u: ", i);
    for (unsigned int j = 0; j < order; j++) {
      printf("%u ", adjList[i][j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    ARTS_PRINTF("usage: bw gpu1 gpu2 ...\n");
    return 0;
  }

  unsigned int bw = atoi(argv[1]);
  gpuList = arts_new_array_list(sizeof(int), 8);

  for (unsigned int i = 0; i < argc - 2; i++) {
    unsigned int gpu = atoi(argv[2 + i]);
    order = MAX(order, gpu);
    arts_push_to_array_list(gpuList, &gpu);
  }
  unsigned int cycleSize = (unsigned int)arts_length_array_list(gpuList) + 1;
  order++;

  adjList = fullyConnect();
  printAdjList();
  depthFirst(cycleSize);
  return 0;
}