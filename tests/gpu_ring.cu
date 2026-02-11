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
    if ((err = (x)) != cudaSuccess) {                                          \
      arts_printf("FAILED %s: %s\n", #x, cudaGetErrorString(err));            \
    }                                                                          \
  }

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

unsigned int *max_cycle = NULL;
arts_array_list_t *gpu_list = NULL;
bool **adj_list = NULL;
unsigned int order = 0;

bool set_max_cycle(const unsigned int *cycle, unsigned int cycle_size) {
  uint64_t length = arts_length_array_list(gpu_list);
  if (cycle[0] == cycle[cycle_size - 1]) {
    for (unsigned int i = 0; i < cycle_size - 1; i++) {
      unsigned int found = 0;
      for (uint64_t j = 0; j < length; j++) {
        int *temp = (int *)arts_get_from_array_list(gpu_list, i);
        if (cycle[j] == (unsigned int)*temp) {
          found++;
        }
      }
      if (found != 1) {
        return false;
      }
    }

    max_cycle = (unsigned int *)calloc(cycle_size, sizeof(unsigned int));
    for (unsigned int i = 0; i < cycle_size; i++) {
      max_cycle[i] = cycle[i];
    }
    return true;
  }
  return false;
}

bool depth_first_rec(unsigned int vertex, unsigned int current,
                     unsigned int cycle_size, unsigned int *cycle) {
  cycle[current] = vertex;

  if (current + 1 == cycle_size) {
    return set_max_cycle(cycle, cycle_size);
  }
  for (unsigned int i = 0; i < order; i++) {
    if (adj_list[vertex][i]) {
      if (depth_first_rec(i, current + 1, cycle_size, cycle)) {
        return true;
      }
    }
  }
  return false;
}

void depth_first(unsigned int cycle_size) {
  unsigned int *cycle =
      (unsigned int *)calloc(order + 1, sizeof(unsigned int));
  for (unsigned int i = 0; i < order; i++) {
    if (depth_first_rec(i, 0, cycle_size, cycle)) {
      for (unsigned int j = 0; j < cycle_size; j++) {
        printf("%u ", max_cycle[j]);
      }
      printf("\n");
      return;
    }
  }
}

bool **fully_connect() {
  bool **local_adj_list = (bool **)calloc(order, sizeof(bool *));
  for (unsigned int i = 0; i < order; i++) {
    local_adj_list[i] = (bool *)calloc(order, sizeof(bool));
  }

  uint64_t length = arts_length_array_list(gpu_list);
  for (uint64_t i = 0; i < length; i++) {
    int *src = (int *)arts_get_from_array_list(gpu_list, i);
    CHECKCORRECT(cudaSetDevice(*src));
    for (uint64_t j = 0; j < length; j++) {
      if (i != j) {
        int has_access = 0;
        int *dst = (int *)arts_get_from_array_list(gpu_list, j);
        CHECKCORRECT(cudaDeviceCanAccessPeer(&has_access, *src, *dst));
        if (has_access) {
          local_adj_list[*src][*dst] = 1;
          CHECKCORRECT(cudaDeviceEnablePeerAccess(*dst, 0));
        }
      }
    }
  }
  return local_adj_list;
}

void print_adj_list() {
  for (unsigned int i = 0; i < order; i++) {
    arts_printf("%u: ", i);
    for (unsigned int j = 0; j < order; j++) {
      printf("%u ", adj_list[i][j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    arts_printf("usage: bw gpu1 gpu2 ...\n");
    return 0;
  }

  unsigned int bw = (unsigned int)strtol(argv[1], NULL, 10);
  (void)bw;
  gpu_list = arts_new_array_list(sizeof(int), 8);

  for (unsigned int i = 0; i < (unsigned int)(argc - 2); i++) {
    unsigned int gpu = (unsigned int)strtol(argv[2 + i], NULL, 10);
    order = MAX(order, gpu);
    arts_push_to_array_list(gpu_list, &gpu);
  }
  unsigned int cycle_size = (unsigned int)arts_length_array_list(gpu_list) + 1;
  order++;

  adj_list = fully_connect();
  print_adj_list();
  depth_first(cycle_size);
  return 0;
}
