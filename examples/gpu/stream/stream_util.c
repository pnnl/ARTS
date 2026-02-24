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
#include "stream_util.h"

#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"

void launch2_kernel_edt(arts_edt_t fun_ptr, unsigned int tile_size,
                        unsigned int total_size, double scalar,
                        arts_guid_t a_guid, arts_guid_t b_guid) {
  unsigned int tiles = total_size / tile_size;
  if (total_size % tile_size) {
    tiles++;
  }

  arts_guid_t to_signal =
      arts_allocate_local_buffer(NULL, 0, tiles + 1, NULL_GUID);

  unsigned int num_threads =
      (THREADSPERBLOCK < tile_size) ? THREADSPERBLOCK : tile_size;
  dim3 threads = {num_threads, 1, 1};
  dim3 grid = {tile_size / num_threads, 1, 1};

  uint64_t args[] = {0, (uint64_t)scalar};
  if (scalar != 0) {
    for (unsigned int i = 0; i < tiles; ++i) {
      args[0] = (i + 1 < tiles) ? tile_size : total_size - (i * tile_size);
      arts_guid_t edt_guid =
          arts_edt_create_gpu(fun_ptr, arts_get_current_node(), 2, args, 2,
                              grid, threads, to_signal, 0, NULL_GUID);
      arts_signal_edt(edt_guid, 0, arts_guid_from_index(a_guid, i), DB_MODE_EW);
      arts_signal_edt(edt_guid, 1, arts_guid_from_index(b_guid, i), DB_MODE_EW);
    }
  } else {
    for (unsigned int i = 0; i < tiles; ++i) {
      args[0] = (i + 1 < tiles) ? tile_size : total_size - (i * tile_size);
      arts_guid_t edt_guid =
          arts_edt_create_gpu(fun_ptr, arts_get_current_node(), 1, args, 2,
                              grid, threads, to_signal, 0, NULL_GUID);
      arts_signal_edt(edt_guid, 0, arts_guid_from_index(a_guid, i), DB_MODE_EW);
      arts_signal_edt(edt_guid, 1, arts_guid_from_index(b_guid, i), DB_MODE_EW);
    }
  }
  arts_block_for_buffer(to_signal);
}

void launch3_kernel_edt(arts_edt_t fun_ptr, unsigned int tile_size,
                        unsigned int total_size, double scalar,
                        arts_guid_t a_guid, arts_guid_t b_guid,
                        arts_guid_t c_guid) {
  unsigned int tiles = total_size / tile_size;
  if (total_size % tile_size) {
    tiles++;
  }

  arts_guid_t to_signal =
      arts_allocate_local_buffer(NULL, 0, tiles + 1, NULL_GUID);

  unsigned int num_threads =
      (THREADSPERBLOCK < tile_size) ? THREADSPERBLOCK : tile_size;
  unsigned int rem_threads = tile_size;
  dim3 threads = {num_threads, 1, 1};
  dim3 grid = {tile_size / num_threads, 1, 1};

  uint64_t args[] = {0, (uint64_t)scalar};
  if (scalar != 0) {
    for (unsigned int i = 0; i < tiles; ++i) {
      args[0] = (i + 1 < tiles) ? tile_size : total_size - (i * tile_size);
      arts_guid_t edt_guid =
          arts_edt_create_gpu(fun_ptr, arts_get_current_node(), 2, args, 3,
                              grid, threads, to_signal, 0, NULL_GUID);
      arts_signal_edt(edt_guid, 0, arts_guid_from_index(a_guid, i), DB_MODE_EW);
      arts_signal_edt(edt_guid, 1, arts_guid_from_index(b_guid, i), DB_MODE_EW);
      arts_signal_edt(edt_guid, 2, arts_guid_from_index(c_guid, i), DB_MODE_EW);
    }
  } else {
    for (unsigned int i = 0; i < tiles; ++i) {
      args[0] = (i + 1 < tiles) ? tile_size : total_size - (i * tile_size);
      arts_guid_t edt_guid =
          arts_edt_create_gpu(fun_ptr, arts_get_current_node(), 1, args, 3,
                              grid, threads, to_signal, 0, NULL_GUID);
      arts_signal_edt(edt_guid, 0, arts_guid_from_index(a_guid, i), DB_MODE_EW);
      arts_signal_edt(edt_guid, 1, arts_guid_from_index(b_guid, i), DB_MODE_EW);
      arts_signal_edt(edt_guid, 2, arts_guid_from_index(c_guid, i), DB_MODE_EW);
    }
  }
  arts_block_for_buffer(to_signal);
}

int checktick() {
  int i;
  int min_delta;
  int delta;
  double time_start;
  double time_end;
  double timesfound[M];

  /*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    time_start = mysecond();
    while (((time_end = mysecond()) - time_start) < 1.0E-6) {
    }
    timesfound[i] = time_start = time_end;
  }

  /*
   * Determine the minimum difference between these M values.
   * This result will be our estimate (in microseconds) for the
   * clock granularity.
   */

  min_delta = 1000000;
  for (i = 1; i < M; i++) {
    delta = (int)(1.0E6 * (timesfound[i] - timesfound[i - 1]));
    min_delta = MIN(min_delta, MAX(delta, 0));
  }

  return (min_delta);
}

double mysecond() {
  struct timeval tp;
  int i;

  i = gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + ((double)tp.tv_usec * 1.e-6));
}
void check_strea_mresults(unsigned int tile_size, unsigned int total_size,
                          double **a_tile, double **b_tile, double **c_tile) {
  double aj;
  double bj;
  double cj;
  double scalar;
  double asum;
  double bsum;
  double csum;
  double epsilon;
  int j;
  int k;

  /* reproduce initialization */
  aj = 1.0;
  bj = 2.0;
  cj = 0.0;
  /* a[] is modified during timing check */
  aj = 2.0E0 * aj;
  /* now execute timing loop */
  scalar = 3.0;
  for (k = 0; k < NTIMES; k++) {
    cj = aj;
    bj = scalar * cj;
    cj = aj + bj;
    aj = bj + (scalar * cj);
  }
  aj = aj * (double)(N);
  bj = bj * (double)(N);
  cj = cj * (double)(N);

  asum = 0.0;
  bsum = 0.0;
  csum = 0.0;

  unsigned int num_tiles = total_size / tile_size;
  if (total_size % tile_size) {
    num_tiles++;
  }

  unsigned int temp = 0;
  for (unsigned int i = 0; i < num_tiles; i++) {
    unsigned int end =
        (i + 1 < num_tiles) ? tile_size : total_size - (i * tile_size);
    for (unsigned int j = 0; j < end; j++) {
      asum += a_tile[i][j];
      bsum += b_tile[i][j];
      csum += c_tile[i][j];
      temp++;
    }
  }

#ifdef VERBOSE
  arts_printf("Results Comparison: \n");
  arts_printf("        Expected  : %f %f %f \n", aj, bj, cj);
  arts_printf("        Observed  : %f %f %f \n", asum, bsum, csum);
#endif

#define abs(a) ((a) >= 0 ? (a) : -(a)) // NOLINT(readability-identifier-naming)
  epsilon = 1.e-8;

  if (abs(aj - asum) / asum > epsilon) {
    arts_printf("Failed Validation on array a[]\n");
    arts_printf("        Expected  : %f \n", aj);
    arts_printf("        Observed  : %f \n", asum);
  } else if (abs(bj - bsum) / bsum > epsilon) {
    arts_printf("Failed Validation on array b[]\n");
    arts_printf("        Expected  : %f \n", bj);
    arts_printf("        Observed  : %f \n", bsum);
  } else if (abs(cj - csum) / csum > epsilon) {
    arts_printf("Failed Validation on array c[]\n");
    arts_printf("        Expected  : %f \n", cj);
    arts_printf("        Observed  : %f \n", csum);
  } else {
    arts_printf("Solution Validates\n");
  }
}