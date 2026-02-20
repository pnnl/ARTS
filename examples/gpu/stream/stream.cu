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

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gas/guid.h"

#include "stream_util.h"

// #define SAFE 1

static double avgtime[4] = {0};
static double maxtime[4] = {0};
static double mintime[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

static const char *label[4] = {
    "Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

static double bytes[4] = {2 * sizeof(double) * N, 2 * sizeof(double) * N,
                          3 * sizeof(double) * N, 3 * sizeof(double) * N};

int quantum;

unsigned int tile_size = 1024 * 1024;
unsigned int num_tiles;

arts_guid_range_t *a_tile_guids = NULL;
arts_guid_range_t *b_tile_guids = NULL;
arts_guid_range_t *c_tile_guids = NULL;

double **a_tile;
double **b_tile;
double **c_tile;

__global__ void copy_kernel(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int len = (unsigned int)paramv[0];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  int idx = (int)(threadIdx.x + (blockIdx.x * blockDim.x));
  if (idx < len) {
    b[idx] = a[idx];
  }
}

__global__ void scale_kernel(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int len = (unsigned int)paramv[0];
  double scale = (double)paramv[1];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  int idx = (int)(threadIdx.x + (blockIdx.x * blockDim.x));
  if (idx < len) {
    b[idx] = scale * a[idx];
  }
}

__global__ void add_kernel(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int len = (unsigned int)paramv[0];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;
  int idx = (int)(threadIdx.x + (blockIdx.x * blockDim.x));
  if (idx < len) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void triad_kernel(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int len = (unsigned int)paramv[0];
  double scale = (double)paramv[1];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;
  int idx = (int)(threadIdx.x + (blockIdx.x * blockDim.x));
  if (idx < len) {
    c[idx] = a[idx] + (scale * b[idx]);
  }
}

void stream_driver(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  int j;
  int k;

  double times[4][NTIMES];

  double scalar = 3.0;
  for (k = 0; k < NTIMES; k++) {
    times[0][k] = mysecond();
    launch2_kernel_edt(copy_kernel, tile_size, N, 0, a_tile_guids,
                       c_tile_guids);
    times[0][k] = mysecond() - times[0][k];

    times[1][k] = mysecond();
    launch2_kernel_edt(scale_kernel, tile_size, N, scalar, c_tile_guids,
                       b_tile_guids);
    times[1][k] = mysecond() - times[1][k];

    times[2][k] = mysecond();
    launch3_kernel_edt(add_kernel, tile_size, N, 0, a_tile_guids, b_tile_guids,
                       c_tile_guids);
    times[2][k] = mysecond() - times[2][k];

    times[3][k] = mysecond();
    launch3_kernel_edt(triad_kernel, tile_size, N, scalar, b_tile_guids,
                       c_tile_guids, a_tile_guids);
    times[3][k] = mysecond() - times[3][k];
  }

  /*	--- SUMMARY --- */
  for (k = 1; k < NTIMES; k++) /* note -- skip first iteration */
  {
    for (j = 0; j < 4; j++) {
      avgtime[j] = avgtime[j] + times[j][k];
      mintime[j] = MIN(mintime[j], times[j][k]);
      maxtime[j] = MAX(maxtime[j], times[j][k]);
    }
  }

  arts_printf(
      "Function      Rate (MB/s)   Avg time     Min time     Max time\n");
  for (j = 0; j < 4; j++) {
    avgtime[j] = avgtime[j] / (double)(NTIMES - 1);

    arts_printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
                1.0E-06 * bytes[j] / mintime[j], avgtime[j], mintime[j],
                maxtime[j]);
  }
  arts_printf(HLINE);

  /* --- Check Results --- */
  check_strea_mresults(tile_size, N, a_tile, b_tile, c_tile);
  arts_printf(HLINE);
  arts_shutdown();
}

extern "C" void arts_main_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];

  if (argc > 1) {
    tile_size = (unsigned int)strtol(argv[1], NULL, 10);
  }

  num_tiles = N / tile_size;
  if (N % tile_size) {
    num_tiles++;
  }

  arts_printf("N: %u tile_size: %u num_tiles: %u Gpus: %u\n", N, tile_size,
              num_tiles, arts_get_total_gpus());

  a_tile_guids = arts_guid_range_create_hash(ARTS_DB_GPU_WRITE, num_tiles, 0,
                                             arts_get_total_gpus());
  b_tile_guids = arts_guid_range_create_hash(ARTS_DB_GPU_WRITE, num_tiles, 0,
                                             arts_get_total_gpus());
  c_tile_guids = arts_guid_range_create_hash(ARTS_DB_GPU_WRITE, num_tiles, 0,
                                             arts_get_total_gpus());

  uint64_t a_hash = arts_guid_hash_key(arts_guid_range_get(a_tile_guids, 0));
  uint64_t b_hash = arts_guid_hash_key(arts_guid_range_get(b_tile_guids, 0));
  uint64_t c_hash = arts_guid_hash_key(arts_guid_range_get(c_tile_guids, 0));

#ifdef SAFE
  if (arts_get_num_gpus() > 1) {
    if (!(ARTS_LOOK_UP_CONFIG(free_db_after_gpu_run) &&
          ARTS_LOOK_UP_CONFIG(run_gpu_gc_pre_edt))) {
      if (ARTS_LOOK_UP_CONFIG(gpu_locality) != 3 || a_hash != b_hash ||
          a_hash != c_hash) {
        arts_printf(
            "For more than 1 GPU Stream requires gpu_locality to be set to "
            "3.\n");
        arts_printf("aHash: %lu bHash: %lu cHash: %lu\n", a_hash, b_hash,
                    c_hash);
        arts_shutdown();
        return;
      }
    }
  }
#endif

  a_tile = (double **)calloc(num_tiles, sizeof(double *));
  b_tile = (double **)calloc(num_tiles, sizeof(double *));
  c_tile = (double **)calloc(num_tiles, sizeof(double *));

  for (unsigned int i = 0; i < num_tiles; i++) {
    a_tile[i] = (double *)arts_db_create_with_guid(
        arts_guid_range_get(a_tile_guids, i), tile_size * sizeof(double), NULL);
    b_tile[i] = (double *)arts_db_create_with_guid(
        arts_guid_range_get(b_tile_guids, i), tile_size * sizeof(double), NULL);
    c_tile[i] = (double *)arts_db_create_with_guid(
        arts_guid_range_get(c_tile_guids, i), tile_size * sizeof(double), NULL);
    for (unsigned int j = 0; j < tile_size; j++) {
      a_tile[i][j] = 1.0;
      b_tile[i][j] = 2.0;
      c_tile[i][j] = 0.0;
    }
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

  if ((quantum = checktick()) >= 1) {
    arts_printf(
        "Your clock granularity/precision appears to be %d microseconds.\n",
        quantum);
  } else {
    arts_printf(
        "Your clock granularity appears to be less than one microsecond.\n");
  }

  double t = mysecond();
  for (unsigned int i = 0; i < num_tiles; i++) {
    for (unsigned int j = 0; j < tile_size; j++) {
      a_tile[i][j] = 2.0E0 * a_tile[i][j];
    }
  }
  t = 1.0E6 * (mysecond() - t);

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

  arts_hint_t hint_0 = {0, 0};
  arts_edt_create(stream_driver, 0, NULL, 0, &hint_0);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
