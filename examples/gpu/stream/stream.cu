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
arts_guid_range_t *cTileGuids = NULL;

double **a_tile;
double **b_tile;
double **c_tile;

__global__ void copyKernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                           arts_edt_dep_t depv[]) {
  unsigned int len = (unsigned int)paramv[0];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    b[idx] = a[idx];
}

__global__ void scaleKernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            arts_edt_dep_t depv[]) {
  unsigned int len = (unsigned int)paramv[0];
  double scale = (double)paramv[1];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    b[idx] = scale * a[idx];
}

__global__ void addKernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t depv[]) {
  unsigned int len = (unsigned int)paramv[0];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    c[idx] = a[idx] + b[idx];
}

__global__ void triadKernal(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            arts_edt_dep_t depv[]) {
  unsigned int len = (unsigned int)paramv[0];
  double scale = (double)paramv[1];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    c[idx] = a[idx] + scale * b[idx];
}

void streamDriver(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  int j, k;

  double times[4][NTIMES];

  double scalar = 3.0;
  for (k = 0; k < NTIMES; k++) {
    times[0][k] = mysecond();
    launch2_kernel_edt(copyKernel, tile_size, N, 0, a_tile_guids, cTileGuids);
    times[0][k] = mysecond() - times[0][k];

    times[1][k] = mysecond();
    launch2_kernel_edt(scaleKernel, tile_size, N, scalar, cTileGuids, b_tile_guids);
    times[1][k] = mysecond() - times[1][k];

    times[2][k] = mysecond();
    launch3_kernel_edt(addKernel, tile_size, N, 0, a_tile_guids, b_tile_guids,
                     cTileGuids);
    times[2][k] = mysecond() - times[2][k];

    times[3][k] = mysecond();
    launch3_kernel_edt(triadKernal, tile_size, N, scalar, b_tile_guids, cTileGuids,
                     a_tile_guids);
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

  ARTS_PRINTF("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
  for (j = 0; j < 4; j++) {
    avgtime[j] = avgtime[j] / (double)(NTIMES - 1);

    ARTS_PRINTF("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
           1.0E-06 * bytes[j] / mintime[j], avgtime[j], mintime[j], maxtime[j]);
  }
  ARTS_PRINTF(HLINE);

  /* --- Check Results --- */
  check_strea_mresults(tile_size, N, a_tile, b_tile, c_tile);
  ARTS_PRINTF(HLINE);
  arts_shutdown();
}

extern "C" void init_per_node(unsigned int node_id, int argc, char **argv) {
  if (argc > 1)
    tile_size = (unsigned int)atoi(argv[1]);

  num_tiles = N / tile_size;
  if (N % tile_size)
    num_tiles++;

  ARTS_PRINTF("N: %u tile_size: %u num_tiles: %u Gpus: %u\n", N, tile_size, num_tiles,
         arts_get_total_gpus());

  a_tile_guids = arts_new_guid_range_node_hash(ARTS_DB_GPU_WRITE, num_tiles, 0,
                                        arts_get_total_gpus());
  b_tile_guids = arts_new_guid_range_node_hash(ARTS_DB_GPU_WRITE, num_tiles, 0,
                                        arts_get_total_gpus());
  cTileGuids = arts_new_guid_range_node_hash(ARTS_DB_GPU_WRITE, num_tiles, 0,
                                        arts_get_total_gpus());

  uint64_t aHash = arts_hash_guid_key(arts_get_guid(a_tile_guids, 0));
  uint64_t bHash = arts_hash_guid_key(arts_get_guid(b_tile_guids, 0));
  uint64_t cHash = arts_hash_guid_key(arts_get_guid(cTileGuids, 0));

#ifdef SAFE
  if (arts_get_num_gpus() > 1) {
    if (!(ARTS_LOOK_UP_CONFIG(free_db_after_gpu_run) &&
          ARTS_LOOK_UP_CONFIG(run_gpu_gc_pre_edt))) {
      if (ARTS_LOOK_UP_CONFIG(gpu_locality) != 3 || aHash != bHash ||
          aHash != cHash) {
        ARTS_PRINTF("For more than 1 GPU Stream requires gpu_locality to be set to "
               "3.\n");
        ARTS_PRINTF("aHash: %lu bHash: %lu cHash: %lu\n", aHash, bHash, cHash);
        arts_shutdown();
      }
    }
  }
#endif

  if (!node_id) {
    a_tile = (double **)arts_calloc(num_tiles, sizeof(double *));
    b_tile = (double **)arts_calloc(num_tiles, sizeof(double *));
    c_tile = (double **)arts_calloc(num_tiles, sizeof(double *));

    for (unsigned int i = 0; i < num_tiles; i++) {
      a_tile[i] = (double *)arts_db_create_with_guid(arts_get_guid(a_tile_guids, i),
                                                tile_size * sizeof(double));
      b_tile[i] = (double *)arts_db_create_with_guid(arts_get_guid(b_tile_guids, i),
                                                tile_size * sizeof(double));
      c_tile[i] = (double *)arts_db_create_with_guid(arts_get_guid(cTileGuids, i),
                                                tile_size * sizeof(double));
      for (unsigned int j = 0; j < tile_size; j++) {
        a_tile[i][j] = 1.0;
        b_tile[i][j] = 2.0;
        c_tile[i][j] = 0.0;
      }
    }

    ARTS_PRINTF(HLINE);
    int BytesPerWord = sizeof(double);
    ARTS_PRINTF("This system uses %d bytes per DOUBLE PRECISION word.\n",
           BytesPerWord);
    ARTS_PRINTF(HLINE);

    ARTS_PRINTF("Array size = %d, Offset = %d\n", N, OFFSET);
    ARTS_PRINTF("Total memory required = %.1f MB.\n",
           (3.0 * BytesPerWord) * ((double)N / 1048576.0));
    ARTS_PRINTF("Each test is run %d times, but only\n", NTIMES);
    ARTS_PRINTF("the *best* time for each is used.\n");
    ARTS_PRINTF(HLINE);

    if ((quantum = checktick()) >= 1)
      ARTS_PRINTF(
          "Your clock granularity/precision appears to be %d microseconds.\n",
          quantum);
    else
      ARTS_PRINTF(
          "Your clock granularity appears to be less than one microsecond.\n");
  }
}

extern "C" void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  if (!node_id) {
    double t = mysecond();
    for (unsigned int i = 0; i < num_tiles; i++) {
      if (i % arts_get_total_workers() == worker_id) {
        for (unsigned int j = 0; j < tile_size; j++)
          a_tile[i][j] = 2.0E0 * a_tile[i][j];
      }
    }
    t = 1.0E6 * (mysecond() - t);

    if (!worker_id) {
      ARTS_PRINTF("Each test below will take on the order of %d microseconds.\n",
             (int)t);
      ARTS_PRINTF("   (= %d clock ticks)\n", (int)(t / quantum));
      ARTS_PRINTF("Increase the size of the arrays if this shows that\n");
      ARTS_PRINTF("you are not getting at least 20 clock ticks per test.\n");

      ARTS_PRINTF(HLINE);

      ARTS_PRINTF("WARNING -- The above is only a rough guideline.\n");
      ARTS_PRINTF("For best results, please be sure you know the\n");
      ARTS_PRINTF("precision of your system timer.\n");
      ARTS_PRINTF(HLINE);

      arts_edt_create(streamDriver, 0, 0, NULL, 0);
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
