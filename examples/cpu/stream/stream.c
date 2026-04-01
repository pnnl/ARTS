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

/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream_omp.c,v 5.4 2009/02/19 13:57:12 mccalpin Exp mccalpin $
 */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2003: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/cxl/wrapper.h"

#include "stream_util.h"

// #define SAFE 1

static double avg_time[4] = {0};
static double max_time[4] = {0};
static double min_time[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

static const char *label[4] = {
    "Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

static double bytes[4] = {2 * sizeof(double) * N, 2 * sizeof(double) * N,
                          3 * sizeof(double) * N, 3 * sizeof(double) * N};

int quantum;

unsigned int tile_size = TILESIZE;
unsigned int num_tiles;

arts_guid_t *a_tile_guids = NULL;
arts_guid_t *b_tile_guids = NULL;
arts_guid_t *c_tile_guids = NULL;

arts_guid_t done_guid = NULL_GUID;

arts_guid_t first_kernel = NULL_GUID;

double **a_tile;
double **b_tile;
double **c_tile;

uint64_t curr_node_tile_start_idx = -1;
uint64_t curr_node_tile_end_idx = -1;
uint64_t curr_num_tiles = 0;
unsigned int* tile_owners;

double times[4][NTIMES];

arts_guid_t get_total_owned_tiles() {
  return curr_num_tiles;
}

arts_guid_t get_done_guid() {
  return done_guid;
}

void populate_tile_range() {
  tile_owners = malloc(num_tiles*sizeof(unsigned int));
  uint32_t total_nodes = arts_get_total_nodes();
  uint64_t tiles_per_node = num_tiles/total_nodes;
  uint32_t curr_node = 0;
  uint64_t curr_count = 0;
  uint32_t node_id = arts_get_current_node();
  for (unsigned int i = 0; i < num_tiles; i++) {
    tile_owners[i] = curr_node;
    if (curr_node == node_id) {
      if (curr_node_tile_start_idx == -1)
        curr_node_tile_start_idx = i;
      curr_num_tiles += 1;
    } 
    curr_count += 1;
    if (curr_node != (arts_get_total_nodes()-1)) {
      if (curr_count == tiles_per_node) {
        if (curr_node == node_id) {
          curr_node_tile_end_idx = i+1;
          // curr_num_tiles += 1;
        }
        curr_node += 1;
        curr_count = 0;
      }
    }
  }
  if (node_id == (arts_get_total_nodes()-1))
    curr_node_tile_end_idx = num_tiles-1;
}

bool tile_in_range(uint64_t i) {
  if ((i >= curr_node_tile_start_idx) && (i <= curr_node_tile_end_idx)) 
    return true;
  else 
    return false;
}

unsigned int get_tile_owner(uint64_t i) {
  return tile_owners[i];
}

void start_timer() {
  static int iteration = 0;
  static int kernel = 0;
  double * time = &times[kernel][iteration];
  kernel++;
  if(kernel == 5) {
    iteration++;
    kernel = 0;
  }
  *time = my_second();
}

void end_timer(arts_edt_dep_t to_signal) {
  static int iteration = 0;
  static int kernel = 0;
  times[kernel][iteration] = my_second() - times[kernel][iteration];
  kernel++;
  if(kernel == 5) {
    iteration++;
    kernel = 0;
  }
}

void copy_kernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                           arts_edt_dep_t depv[]) {
  unsigned int len = (unsigned int)paramv[2];
  arts_guid_t next_edt = paramv[1];
  uint32_t slot = paramv[3];

  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  
  for (int idx=0; idx<len; idx++) {
    b[idx] = a[idx];
  }
  
  #if ARTS_USE_CXL
  arts_cxl_producer_flush(depv[0].guid);
  arts_cxl_producer_flush(depv[1].guid);
  #endif
  arts_signal_edt_null(next_edt, slot); 
}

void scale_kernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            arts_edt_dep_t depv[]) {
  unsigned int len = (unsigned int)paramv[2];
  arts_guid_t next_edt = paramv[1];
  uint32_t slot = paramv[3];
  double scale = (double)paramv[4];

  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  
  for (int idx=0; idx<len; idx++) {
    b[idx] = scale * a[idx];
  }

  #if ARTS_USE_CXL
  arts_cxl_producer_flush(depv[0].guid);
  arts_cxl_producer_flush(depv[1].guid);
  #endif
  arts_signal_edt_null(next_edt, slot);
}

void add_kernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t depv[]) {
  unsigned int len = (unsigned int)paramv[2];
  arts_guid_t next_edt = paramv[1];
  uint32_t slot = paramv[3];

  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;

  for (int idx=0; idx<len; idx++) {
    c[idx] = a[idx] + b[idx];
  }
   
  #if ARTS_USE_CXL
  arts_cxl_producer_flush(depv[0].guid);
  arts_cxl_producer_flush(depv[1].guid);
  arts_cxl_producer_flush(depv[2].guid);
  #endif
  arts_signal_edt_null(next_edt, slot);
}

void triad_kernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            arts_edt_dep_t depv[]) {
  unsigned int len = (unsigned int)paramv[2];
  arts_guid_t next_edt = paramv[1];
  uint32_t slot = paramv[3];
  double scale = (double)paramv[4];

  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;
  
  for (int idx=0; idx<len; idx++) {
    c[idx] = a[idx] + scale * b[idx];
  }
  
  #if ARTS_USE_CXL
  arts_cxl_producer_flush(depv[0].guid);
  arts_cxl_producer_flush(depv[1].guid);
  arts_cxl_producer_flush(depv[2].guid);
  arts_signal_edt(next_edt, slot, depv[2].guid, DB_MODE_RO);
  arts_signal_edt(next_edt, num_tiles + slot, depv[0].guid, DB_MODE_RO);
  arts_signal_edt(next_edt, (2*num_tiles) + slot, depv[1].guid, DB_MODE_RO);
  arts_signal_edt_null(next_edt, slot); 
  #else
  if (next_edt != done_guid)
    arts_signal_edt_null(next_edt, slot); 
  else {
    double* a_copy;
    double* b_copy;
    double* c_copy;
    arts_guid_t a_signal = arts_db_create((void**)&a_copy, tile_size*sizeof(double), ARTS_DB_DEFAULT, NULL);
    arts_guid_t b_signal = arts_db_create((void**)&b_copy, tile_size*sizeof(double), ARTS_DB_DEFAULT, NULL);
    arts_guid_t c_signal = arts_db_create((void**)&c_copy, tile_size*sizeof(double), ARTS_DB_DEFAULT, NULL);
    memcpy(a_copy, depv[2].ptr, sizeof(double)*tile_size);
    memcpy(b_copy, depv[0].ptr, sizeof(double)*tile_size);
    memcpy(c_copy, depv[1].ptr, sizeof(double)*tile_size);
    
    arts_signal_edt(next_edt, slot, a_signal, DB_MODE_RO);
    arts_signal_edt(next_edt, num_tiles + slot, b_signal, DB_MODE_RO);
    arts_signal_edt(next_edt, (2*num_tiles) + slot, c_signal, DB_MODE_RO);
  }
  #endif
}

void done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            arts_edt_dep_t depv[]) {
  int k, j;
  /*	--- SUMMARY --- */
  /*
  for (k = 1; k < NTIMES; k++) // note -- skip first iteration
  {
    for (j = 0; j < 4; j++) {
      avg_time[j] = avg_time[j] + times[j][k];
      min_time[j] = MIN(min_time[j], times[j][k]);
      max_time[j] = MAX(max_time[j], times[j][k]);
    }
  }

  arts_printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
  for (j = 0; j < 4; j++) {
    avg_time[j] = avg_time[j] / (double)(NTIMES - 1);

    arts_printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
           1.0E-06 * bytes[j] / min_time[j], avg_time[j], min_time[j], max_time[j]);
  }
  arts_printf(HLINE);
  */
  
  #if !ARTS_USE_CXL
  double** a_tile_all = malloc(sizeof(double*)*num_tiles);
  double** b_tile_all = malloc(sizeof(double*)*num_tiles);
  double** c_tile_all = malloc(sizeof(double*)*num_tiles);
  
  for (unsigned int i=0; i<num_tiles; i++) {
    a_tile_all[i] = malloc(sizeof(double)*tile_size);
    b_tile_all[i] = malloc(sizeof(double)*tile_size);
    c_tile_all[i] = malloc(sizeof(double)*tile_size);
    memcpy(a_tile_all[i], depv[i].ptr, sizeof(double)*tile_size);
    memcpy(b_tile_all[i], depv[num_tiles + i].ptr, sizeof(double)*tile_size);
    memcpy(c_tile_all[i], depv[(2*num_tiles) + i].ptr, sizeof(double)*tile_size);
  }
  #endif
 
  // Validate results on node 0
  if (!arts_get_current_node()) {
    #if ARTS_USE_CXL
    for (unsigned int i = 0; i < num_tiles; i++) {
      arts_cxl_consumer_flush(a_tile_guids[i]);
      arts_cxl_consumer_flush(b_tile_guids[i]);
      arts_cxl_consumer_flush(c_tile_guids[i]);
    }
    check_stream_results(tile_size, N, a_tile, b_tile, c_tile);
    #else
    check_stream_results(tile_size, N, a_tile_all, b_tile_all, c_tile_all);
    #endif
    arts_printf(HLINE);
  }
  
  free(a_tile_guids);
  free(b_tile_guids);
  free(c_tile_guids);
  free(a_tile);
  free(b_tile);
  free(c_tile);
  #if !ARTS_USE_CXL
  for (unsigned int i=0; i<num_tiles; i++) {
    free(a_tile_all[i]);
    free(b_tile_all[i]);
    free(c_tile_all[i]);
  }
  free(a_tile_all);
  free(b_tile_all);
  free(c_tile_all);
  #endif
  arts_shutdown();
}

void stream_driver(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  int j, k;

  double scalar = 3.0;
  arts_guid_t epoch_guid;
  unsigned int tiles = N / tile_size;
  if (N % tile_size) tiles++;
  unsigned int current_node = arts_get_current_node();
  uint32_t num_deps;
  
  // #if ARTS_USE_CXL
  num_deps = tiles;
  // #else
  // num_deps = curr_num_tiles;
  // #endif
  
  arts_guid_t prev_edt = done_guid;
  for (k = NTIMES-1; k >= 0; k--) {
    uint64_t args_triad[8] = {(uint64_t)triad_kernel, tile_size, N, scalar, (uint64_t)b_tile_guids,
                           (uint64_t)c_tile_guids, (uint64_t)a_tile_guids, prev_edt};
 
    prev_edt = arts_edt_create(launch_3_kernel_edt, 8, args_triad, num_deps,
                               &(arts_hint_t){.route = current_node});
    
    uint64_t args_add[8] = {(uint64_t)add_kernel, tile_size, N, 0,
                          (uint64_t)a_tile_guids, (uint64_t)b_tile_guids,
                          (uint64_t)c_tile_guids, prev_edt};
    prev_edt = arts_edt_create(launch_3_kernel_edt, 8, args_add, num_deps,
                               &(arts_hint_t){.route = current_node});
  
    uint64_t args_scale[7] = {(uint64_t)scale_kernel, tile_size, N, scalar,
                             (uint64_t)c_tile_guids, (uint64_t)b_tile_guids, prev_edt};
    prev_edt = arts_edt_create(launch_2_kernel_edt, 7, args_scale, num_deps,
                               &(arts_hint_t){.route = current_node});
    
    uint64_t args_copy[7] = {(uint64_t)copy_kernel, tile_size, N, 0,
                       (uint64_t)a_tile_guids, (uint64_t)c_tile_guids, prev_edt};
    if (k == 0) {
      // arts_edt_create(launch_2_kernel_edt, 0, 7, args_copy, 1);
      arts_edt_create_with_guid(launch_2_kernel_edt, first_kernel, 7,
                            args_copy, arts_get_total_workers());
    }
    else
      prev_edt = arts_edt_create(launch_2_kernel_edt, 7, args_copy, num_deps,
                                 &(arts_hint_t){.route = current_node});
  }
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  // if (!node_id) {
  if (argc > 1)
    tile_size = (unsigned int)atoi(argv[1]);
  num_tiles = N / tile_size;
  if (N % tile_size)
    num_tiles++;
  
  done_guid = arts_guid_reserve(ARTS_EDT, 0);
  
  // populate_tile_range();

  if (!node_id)
    arts_printf("N: %u tile_size: %u num_tiles: %u\n", N, tile_size, num_tiles);
  // }
  
  a_tile_guids = malloc(sizeof(arts_guid_t)*num_tiles);
  b_tile_guids = malloc(sizeof(arts_guid_t)*num_tiles);
  c_tile_guids = malloc(sizeof(arts_guid_t)*num_tiles);
  
  #if !ARTS_USE_CXL
  unsigned int owner = 0;
  for (unsigned int i = 0; i < num_tiles; i++) {
    // unsigned int owner = get_tile_owner(i);
    a_tile_guids[i] = arts_guid_reserve(ARTS_DB, owner);
    b_tile_guids[i] = arts_guid_reserve(ARTS_DB, owner);
    c_tile_guids[i] = arts_guid_reserve(ARTS_DB, owner);
    owner = (owner+1)%arts_get_total_nodes();
  }
  #endif
 #if ARTS_USE_CXL
  if (!node_id) {
  #endif
    a_tile = (double **)calloc(num_tiles, sizeof(double *));
    b_tile = (double **)calloc(num_tiles, sizeof(double *));
    c_tile = (double **)calloc(num_tiles, sizeof(double *));
    
    #if !ARTS_USE_CXL
    owner = 0;
    #endif
    for (unsigned int i = 0; i < num_tiles; i++) {
      #if ARTS_USE_CXL
      a_tile_guids[i] = arts_db_create((void **)&(a_tile[i]), tile_size * sizeof(double),
                                  ARTS_DB_CXL, NULL);
      b_tile_guids[i] = arts_db_create((void **)&(b_tile[i]), tile_size * sizeof(double),
                                  ARTS_DB_CXL, NULL);
      c_tile_guids[i] = arts_db_create((void **)&(c_tile[i]), tile_size * sizeof(double),
                                  ARTS_DB_CXL, NULL);
      #else
      // if (node_id == get_tile_owner(i)) {
      if (node_id == owner) {
        a_tile[i] = arts_db_create_with_guid(a_tile_guids[i], tile_size * sizeof(double),
                                             ARTS_DB_DEFAULT, NULL, NULL);
        b_tile[i] = arts_db_create_with_guid(b_tile_guids[i], tile_size * sizeof(double),
                                             ARTS_DB_DEFAULT, NULL, NULL);
        c_tile[i] = arts_db_create_with_guid(c_tile_guids[i], tile_size * sizeof(double),
                                             ARTS_DB_DEFAULT, NULL, NULL);
      #endif
      
        for (unsigned int j = 0; j < tile_size; j++) {
          a_tile[i][j] = 1.0;
          b_tile[i][j] = 2.0;
          c_tile[i][j] = 0.0;
        }
      #if !ARTS_USE_CXL
      }
    #endif
      #if ARTS_USE_CXL
      arts_cxl_producer_flush(a_tile_guids[i]);
      arts_cxl_producer_flush(b_tile_guids[i]);
      arts_cxl_producer_flush(c_tile_guids[i]);
      #else
      owner = (owner+1)%arts_get_total_nodes();
      #endif
    }

    arts_printf(HLINE);
    int bytes_per_word = sizeof(double);
    arts_printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
           bytes_per_word);
    arts_printf(HLINE);

    arts_printf("Array size = %d, Offset = %d\n", N, OFFSET);
    arts_printf("Total memory required = %.1f MB.\n",
           (3.0 * bytes_per_word) * ((double)N / 1048576.0));
    arts_printf("Each test is run %d times, but only\n", NTIMES);
    arts_printf("the *best* time for each is used.\n");
    arts_printf(HLINE);

    if ((quantum = check_tick()) >= 1)
      arts_printf(
          "Your clock granularity/precision appears to be %d microseconds.\n",
          quantum);
    else
      arts_printf(
          "Your clock granularity appears to be less than one microsecond.\n");
    first_kernel = arts_guid_reserve(ARTS_EDT, node_id);
  #if ARTS_USE_CXL
  }
  #endif
}

void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  #if ARTS_USE_CXL
  wbinv();
  if (!node_id) {
  #endif
    double t = my_second();
    #if !ARTS_USE_CXL
    unsigned int owner = 0;
    #endif
    for (unsigned int i = 0; i < num_tiles; i++) {
      #if !ARTS_USE_CXL
      // if (node_id == get_tile_owner(i)) {
      if (node_id == owner) {
      #endif
      if (i % arts_get_total_workers() == worker_id) {
        for (unsigned int j = 0; j < tile_size; j++)
          a_tile[i][j] = 2.0E0 * a_tile[i][j];
        #if ARTS_USE_CXL
        arts_cxl_producer_flush(a_tile_guids[i]);
        #endif
      }
      #if !ARTS_USE_CXL
      }
      owner = (owner+1)%arts_get_total_nodes();
      #endif
    }
    arts_signal_edt(first_kernel, arts_get_current_worker(), NULL_GUID, DB_MODE_NULL);
    t = 1.0E6 * (my_second() - t);

    if (!worker_id) {
      #if !ARTS_USE_CXL
      if (!node_id) {
      #endif
        arts_printf("Each test below will take on the order of %d microseconds.\n",
               (int)t);
          arts_printf("   (= %d clock ticks)\n", (int)(t / quantum));
        arts_printf("Increase the size of the arrays if this shows that\n");
        arts_printf("you are not getting at least 20 clock ticks per test.\n");

        arts_printf(HLINE);

        arts_printf("WARNING -- The above is only a rough guideline.\n");
        arts_printf("For best results, please be sure you know the\n");
        arts_printf("precision of your system timer.\n");
        arts_printf(HLINE);
      #if !ARTS_USE_CXL
      }
      #endif
      
      unsigned int tiles = N / tile_size;
      if (N % tile_size) tiles++;

      // done_guid = arts_edt_create(done, 0, 0, NULL, tiles);
      if (!node_id) {
        arts_edt_create_with_guid(done, done_guid, 0, NULL, tiles*3); // A tiles, B tiles, C tiles
      }
      if (!node_id)
        arts_edt_create(stream_driver, 0, NULL, 0, &(arts_hint_t){.route = 0});
    }
  #if ARTS_USE_CXL
  }
  #endif
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}