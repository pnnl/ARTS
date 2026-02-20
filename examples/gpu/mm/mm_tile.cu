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

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"

#include "mm_util.h"

#define GPUMM 1
#define MATSIZE 1024
#define TILESIZE 32
// #define VERIFY 1
#define SMTILE 32 // Hardcoded for Volta

uint64_t start = 0;

int mat_size;
int tile_size;
unsigned int num_blocks = 1;

arts_guid_t a_mat_guid = NULL_GUID;
arts_guid_t b_mat_guid = NULL_GUID;
arts_guid_t c_mat_guid = NULL_GUID;
arts_guid_t done_guid = NULL_GUID;

double *a_matrix = NULL;
double *b_matrix = NULL;
double *c_matrix = NULL;

arts_guid_range_t *a_tile_guids = NULL;
arts_guid_range_t *b_tile_guids = NULL;

__global__ void mm_kernel(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const int blk = (int)paramv[0];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;

  int col = (int)(blockDim.x * blockIdx.x) + (int)threadIdx.x;
  int row = (int)(blockDim.y * blockIdx.y) + (int)threadIdx.y;

  double sum = 0;

  for (unsigned int k = 0; k < (unsigned int)blk; k++) {
    sum += a[(row * blk) + k] * b[(k * blk) + col];
  }
  c[(row * blk) + col] = sum;
}

void mm_kernel_cpu(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t to_signal = (arts_guid_t)paramv[1];
  unsigned int k = (unsigned int)paramv[2];
  arts_guid_t c_tile_guid = (arts_guid_t)paramv[3];
  const int blk = (int)paramv[0];
  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;

  for (unsigned int i = 0; i < (unsigned int)blk; i++) {
    // rows of B
    for (unsigned int j = 0; j < (unsigned int)blk; j++) {
      // rows of A and columns of B
      for (unsigned int kk = 0; kk < (unsigned int)blk; kk++) {
        c[(i * blk) + j] += a[(i * blk) + kk] * b[(kk * blk) + j];
      }
    }
  }
  arts_signal_edt(to_signal, k, c_tile_guid, ARTS_DB_WRITE);
}

void multiply_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t to_signal = (arts_guid_t)paramv[0];

  unsigned int row_size = tile_size;

  // unsigned int i = paramv[1];
  // unsigned int j = paramv[2];
  unsigned int k = paramv[3];

  // double * a_tile = (double*) depv[0].ptr;
  // double * b_tile = (double*) depv[1].ptr;
  double *c_tile = NULL;

  arts_guid_t a_tile_guid = depv[0].guid;
  arts_guid_t b_tile_guid = depv[1].guid;
  arts_guid_t c_tile_guid = arts_guid_reserve(ARTS_DB_GPU_WRITE, 0);
  c_tile = (double *)arts_db_create_with_guid(
      c_tile_guid, sizeof(double) * tile_size * tile_size, NULL);

  init_matrix(row_size, c_tile, false, true);

#if GPUMM
  dim3 threads(SMTILE, SMTILE);
  dim3 grid((tile_size + SMTILE - 1) / SMTILE,
            (tile_size + SMTILE - 1) / SMTILE);

  uint64_t args[] = {(uint64_t)tile_size};
  arts_guid_t mul_gpu_guid =
      arts_edt_create_gpu(mm_kernel, arts_get_current_node(), 1, args, 3, grid,
                          threads, to_signal, k, c_tile_guid);
  arts_signal_edt(mul_gpu_guid, 0, a_tile_guid, ARTS_DB_WRITE);
  arts_signal_edt(mul_gpu_guid, 1, b_tile_guid, ARTS_DB_WRITE);
  arts_signal_edt(mul_gpu_guid, 2, c_tile_guid, ARTS_DB_WRITE);
#else
  uint64_t args[] = {tile_size, to_signal, k, c_tile_guid};
  arts_hint_t hint_0 = {arts_get_current_node(), 0};
  arts_guid_t mul_gpu_guid =
      arts_edt_create(mm_kernel_cpu, 4, args, 3,
                      &hint_0);
  arts_signal_edt(mul_gpu_guid, 0, a_tile_guid, ARTS_DB_WRITE);
  arts_signal_edt(mul_gpu_guid, 1, b_tile_guid, ARTS_DB_WRITE);
  arts_signal_edt(mul_gpu_guid, 2, c_tile_guid, ARTS_DB_WRITE);
#endif
}

__global__ void sum_mm_kernel(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  const unsigned int column_size = (unsigned int)paramv[0];

  double *c_tile = (double *)depv[0].ptr;

  int row = (int)(blockDim.x * blockIdx.x) + (int)threadIdx.x;
  int col = (int)(blockDim.y * blockIdx.y) + (int)threadIdx.y;

  for (unsigned int k = 1; k < depc; ++k) {
    double *to_add = (double *)depv[k].ptr;
    c_tile[(row * column_size) + col] += to_add[(row * column_size) + col];
  }
}

void sum_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)paramc;
  arts_guid_t done_guid = (arts_guid_t)paramv[0];

  unsigned int column_size = tile_size;

  unsigned int row = paramv[1];
  unsigned int col = paramv[2];

  double *c_tile;
  unsigned int row_size = tile_size;
  arts_guid_t c_tile_guid = arts_guid_reserve(ARTS_DB_GPU_WRITE, 0);
  c_tile = (double *)arts_db_create_with_guid(
      c_tile_guid, sizeof(double) * tile_size * tile_size, NULL);
  init_matrix(row_size, c_tile, false, true);

  for (unsigned int i = 0; i < depc; i++) {
    double *to_add = (double *)depv[i].ptr;
    for (unsigned int j = 0; j < column_size; j++) {
      for (unsigned int k = 0; k < row_size; k++) {
        c_tile[(j * row_size) + k] += to_add[(j * row_size) + k];
      }
    }
  }
  arts_signal_edt(done_guid, 3 + (row * num_blocks + col), c_tile_guid,
                  ARTS_DB_WRITE);
}

void finish_block_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  double *c_mat = (double *)depv[0].ptr;

  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      double *c_tile = (double *)depv[3 + ((i * num_blocks) + j)].ptr;
      copy_block(i, j, tile_size, c_tile, mat_size, c_mat, false);
    }
  }

  uint64_t time = arts_get_time_stamp() - start;

#if VERIFY
  double *a_mat = (double *)depv[1].ptr;
  double *b_mat = (double *)depv[2].ptr;
  arts_printf("Verifying results...\n");
  double *temp = (double *)calloc(mat_size * mat_size, sizeof(double));
  for (unsigned int i = 0; i < (unsigned int)mat_size; ++i) {
    for (unsigned int j = 0; j < (unsigned int)mat_size; ++j) {
      for (unsigned int k = 0; k < (unsigned int)mat_size; ++k) {
        temp[(i * mat_size) + j] +=
            a_mat[(i * mat_size) + k] * b_mat[(k * mat_size) + j];
      }
    }
  }

  for (unsigned int i = 0; i < (unsigned int)mat_size; ++i) {
    for (unsigned int j = 0; j < (unsigned int)mat_size; ++j) {
      if (temp[(i * mat_size) + j] != c_mat[(i * mat_size) + j]) {
        arts_printf("Failed at c_mat[%u][%u]\n", i, j);
        arts_printf("Expected: %lf | Obtained: %lf\n", temp[(i * mat_size) + j],
                    c_mat[(i * mat_size) + j]);
        free(temp);
        arts_shutdown();
        return;
      }
    }
  }

  free(temp);
  arts_printf("Success %lu\n", time);
#else
  arts_printf("Done %lu\n", time);
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
    mat_size = MATSIZE;
    tile_size = TILESIZE;
  } else if (argc == 2) {
    mat_size = (int)strtol(argv[1], NULL, 10);
    tile_size = TILESIZE;
  } else {
    mat_size = (int)strtol(argv[1], NULL, 10);
    tile_size = (int)strtol(argv[2], NULL, 10);
  }
  num_blocks = (unsigned int)(mat_size / tile_size);
  done_guid = arts_guid_reserve(ARTS_EDT, 0);
  a_mat_guid = arts_guid_reserve(ARTS_DB, 0);
  b_mat_guid = arts_guid_reserve(ARTS_DB, 0);
  c_mat_guid = arts_guid_reserve(ARTS_DB, 0);

  a_tile_guids =
      arts_guid_range_create(ARTS_DB_GPU_READ, num_blocks * num_blocks, 0);
  b_tile_guids =
      arts_guid_range_create(ARTS_DB_GPU_READ, num_blocks * num_blocks, 0);

  a_matrix = (double *)arts_db_create_with_guid(
      a_mat_guid, (size_t)mat_size * mat_size * sizeof(double), NULL);
  b_matrix = (double *)arts_db_create_with_guid(
      b_mat_guid, (size_t)mat_size * mat_size * sizeof(double), NULL);
  c_matrix = (double *)arts_db_create_with_guid(
      c_mat_guid, (size_t)mat_size * mat_size * sizeof(double), NULL);

  init_matrix(mat_size, a_matrix, true, false);
  init_matrix(mat_size, b_matrix, false, false);
  init_matrix(mat_size, c_matrix, false, true);

  arts_printf("Starting\n");

  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      arts_guid_t a_tile_guid =
          arts_guid_range_get(a_tile_guids, (i * num_blocks) + j);
      double *a_tile = (double *)arts_db_create_with_guid(
          a_tile_guid, sizeof(double) * tile_size * tile_size, NULL);
      copy_block(i, j, tile_size, a_tile, mat_size, a_matrix, true);

      arts_guid_t b_tile_guid =
          arts_guid_range_get(b_tile_guids, (i * num_blocks) + j);
      double *b_tile = (double *)arts_db_create_with_guid(
          b_tile_guid, sizeof(double) * tile_size * tile_size, NULL);
      copy_block(i, j, tile_size, b_tile, mat_size, b_matrix, true);
    }
  }

  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
#if GPUMM
      uint64_t sum_args[] = {(uint64_t)tile_size};
      dim3 threads(SMTILE, SMTILE);
      dim3 grid((tile_size + SMTILE - 1) / SMTILE,
                (tile_size + SMTILE - 1) / SMTILE);

      arts_guid_t sum_guid = arts_edt_create_gpu_pt(
          sum_mm_kernel, node_id, 1, sum_args, num_blocks, grid, threads,
          done_guid, 3 + ((i * num_blocks) + j), 0);
#else
      uint64_t sum_args[] = {done_guid, i, j};
      arts_hint_t hint_1 = {node_id, 0};
      arts_guid_t sum_guid = arts_edt_create(sum_mm, 3, sum_args, num_blocks,
                                             &hint_1);
#endif
      for (unsigned int k = 0; k < num_blocks; k++) {
        uint64_t args[] = {(uint64_t)sum_guid, i, j, k};
        arts_hint_t hint_2 = {node_id, 0};
        arts_guid_t mul_guid = arts_edt_create(
            multiply_mm, 4, args, 2, &hint_2);
        arts_signal_edt(mul_guid, 0,
                        arts_guid_range_get(a_tile_guids, (i * num_blocks) + k),
                        ARTS_DB_WRITE);
        arts_signal_edt(mul_guid, 1,
                        arts_guid_range_get(b_tile_guids, (k * num_blocks) + j),
                        ARTS_DB_WRITE);
      }
    }
  }

  arts_edt_create_with_guid(finish_block_mm, done_guid, 0, NULL,
                            3 + (num_blocks * num_blocks));
  arts_signal_edt(done_guid, 0, c_mat_guid, ARTS_DB_WRITE);
  arts_signal_edt(done_guid, 1, a_mat_guid, ARTS_DB_WRITE);
  arts_signal_edt(done_guid, 2, b_mat_guid, ARTS_DB_WRITE);
  start = arts_get_time_stamp();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
