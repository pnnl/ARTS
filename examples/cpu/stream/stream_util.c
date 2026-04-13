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
#include <stdio.h>

#include "arts.h"

/*
void launch_2_kernel_edt(arts_edt_t fun_ptr, unsigned int tile_size,
                      unsigned int total_size, double scalar,
                      arts_guid_range *a_guid, arts_guid_range *b_guid) {
*/
void launch_2_kernel_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  arts_edt_t fun_ptr = (arts_edt_t) paramv[0];
  unsigned int tile_size = (unsigned int) paramv[1];
  unsigned int total_size = (unsigned int) paramv[2];
  double scalar = (double) paramv[3];
  arts_guid_t* a_guid = (arts_guid_t*) paramv[4];
  arts_guid_t* b_guid = (arts_guid_t*) paramv[5];
  arts_guid_t next_guid = (arts_guid_t) paramv[6];
  
  unsigned int tiles = total_size / tile_size;
  if (total_size % tile_size)
    tiles++;
  
  arts_guid_t timer_event;
  timer_event = arts_event_create(0, ARTS_EVENT_LATCH, tiles, NULL_GUID);
  arts_add_local_event_callback(timer_event, end_timer);
  arts_add_dependence(timer_event, next_guid, 0, DB_MODE_NULL);

  uint64_t args[] = {timer_event, next_guid, tile_size, 0, (uint64_t)scalar};
  uint64_t num_args = (scalar == 0) ? 4 : 5;
  unsigned int next = 0;
  arts_guid_t * edt_guids = (arts_guid_t*) malloc(sizeof(arts_guid_t)*tiles);
  for (unsigned int i = 0; i < tiles; ++i) {
      args[2] = (i + 1 < tiles) ? tile_size : total_size - i * tile_size;
      args[3] = i;
      edt_guids[i] = arts_edt_create(fun_ptr, num_args, args, 2,
                                     &(arts_hint_t){.route = next});
      next = (next + 1) % arts_get_total_nodes();
      arts_signal_edt(edt_guids[i], 0, a_guid[i], DB_MODE_RO);
  }
  start_timer();
  for (unsigned int i = 0; i < tiles; ++i) {
    arts_signal_edt(edt_guids[i], 1, b_guid[i], DB_MODE_RO);
  }
  free(edt_guids);
}

/*
void launch_3_kernel_edt(arts_edt_t fun_ptr, unsigned int tile_size,
                      unsigned int total_size, double scalar,
                      arts_guid_range *a_guid, arts_guid_range *b_guid,
                      arts_guid_range *c_guid) {
*/
void launch_3_kernel_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  arts_edt_t fun_ptr = (arts_edt_t) paramv[0];
  unsigned int tile_size = (unsigned int) paramv[1];
  unsigned int total_size = (unsigned int) paramv[2];
  double scalar = (double) paramv[3];
  arts_guid_t* a_guid = (arts_guid_t*) paramv[4];
  arts_guid_t* b_guid = (arts_guid_t*) paramv[5];
  arts_guid_t* c_guid = (arts_guid_t*) paramv[6];
  arts_guid_t next_guid = (arts_guid_t) paramv[7];
  arts_guid_t done_guid = get_done_guid();
                        
  unsigned int tiles = total_size / tile_size;
  if (total_size % tile_size)
    tiles++;

  arts_guid_t timer_event;
  timer_event = arts_event_create(0, ARTS_EVENT_LATCH, tiles, NULL_GUID);
  arts_add_local_event_callback(timer_event, end_timer);
  // Only chain timer_event -> next_guid when next_guid is not done_guid.
  // For the last triad, the triad kernels signal done_guid directly with data.
  if (next_guid != done_guid)
    arts_add_dependence(timer_event, next_guid, 0, DB_MODE_NULL);

  uint64_t args[] = {timer_event, next_guid, tile_size, 0, (uint64_t)scalar};
  uint64_t num_args = (scalar == 0) ? 4 : 5;
  unsigned int next = 0;

  arts_guid_t * edt_guids = (arts_guid_t*) malloc(sizeof(arts_guid_t)*tiles);
  for (unsigned int i = 0; i < tiles; ++i) {
      args[2] = (i + 1 < tiles) ? tile_size : total_size - i * tile_size;
      args[3] = i;
      edt_guids[i] = arts_edt_create(fun_ptr, num_args, args, 3,
                                     &(arts_hint_t){.route = next});
      next = (next + 1) % arts_get_total_nodes();
      arts_signal_edt(edt_guids[i], 0, a_guid[i], DB_MODE_RO);
      arts_signal_edt(edt_guids[i], 1, b_guid[i], DB_MODE_RO);
  }
  start_timer();
  for (unsigned int i = 0; i < tiles; ++i) {
    arts_signal_edt(edt_guids[i], 2, c_guid[i], DB_MODE_RO);
  }
  free(edt_guids);
}

void do_nothing(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            arts_edt_dep_t depv[]) {
}

void force_acquire_cxl_guids(unsigned int num_tiles, 
                      arts_guid_t *a_guid, arts_guid_t *b_guid,
                      arts_guid_t *c_guid) {
  for (unsigned int i = 0; i < num_tiles; i++) {
    arts_guid_t edt_guid = arts_edt_create(do_nothing, 0, NULL, 3, NULL);
    arts_signal_edt(edt_guid, 0, a_guid[i], DB_MODE_RO);
    arts_signal_edt(edt_guid, 1, b_guid[i], DB_MODE_RO);
    arts_signal_edt(edt_guid, 2, c_guid[i], DB_MODE_RO);
  }
} 

int check_tick() {
  int i, min_delta, delta;
  double t1, t2, times_found[M];

  /*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    t1 = my_second();
    while (((t2 = my_second()) - t1) < 1.0E-6)
      ;
    times_found[i] = t1 = t2;
  }

  /*
   * Determine the minimum difference between these M values.
   * This result will be our estimate (in microseconds) for the
   * clock granularity.
   */

  min_delta = 1000000;
  for (i = 1; i < M; i++) {
    delta = (int)(1.0E6 * (times_found[i] - times_found[i - 1]));
    min_delta = MIN(min_delta, MAX(delta, 0));
  }

  return (min_delta);
}

double my_second() {
  struct timeval tp;
  int i;

  i = gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void check_stream_results(unsigned int tile_size, unsigned int total_size,
                         double **a_tile, double **b_tile, double **c_tile) {
  double aj, bj, cj, scalar;
  double asum, bsum, csum;
  double epsilon;
  int j, k;

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
    aj = bj + scalar * cj;
  }
  aj = aj * (double)(N);
  bj = bj * (double)(N);
  cj = cj * (double)(N);

  asum = 0.0;
  bsum = 0.0;
  csum = 0.0;

  unsigned int num_tiles = total_size / tile_size;
  if (total_size % tile_size)
    num_tiles++;

  unsigned int temp = 0;
  for (unsigned int i = 0; i < num_tiles; i++) {
    unsigned int end = (i + 1 < num_tiles) ? tile_size : total_size - i * tile_size;
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

#define abs(a) ((a) >= 0 ? (a) : -(a))
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