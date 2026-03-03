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

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "arts.h"
#include "arts/gpu.h"

#include "mm_util.h"

#define MATSIZE 1024
#define TILESIZE 32
// #define VERIFY 1

uint64_t start = 0;
cublasHandle_t *handle;

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

arts_guid_t a_tile_guids = NULL_GUID;
arts_guid_t b_tile_guids = NULL_GUID;

void multiply_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t to_signal = (arts_guid_t)paramv[0];
  unsigned int size =
      sizeof(double) * (unsigned int)tile_size * (unsigned int)tile_size;
  // unsigned int i = paramv[1];
  // unsigned int j = paramv[2];
  unsigned int k = paramv[3];

  double *a_tile_dev = (double *)depv[0].ptr;
  double *b_tile_dev = (double *)depv[1].ptr;
  // arts_guid_t a_tile_guid = depv[0].guid;
  // arts_guid_t b_tile_guid = depv[1].guid;
  arts_guid_t c_tile_guid = arts_guid_reserve(ARTS_DB, 0);
  arts_db_create_with_guid(c_tile_guid, size, ARTS_DB_GPU, NULL, NULL);

  double *c_tile_dev = (double *)arts_cuda_malloc(size);

  double alpha = 1.0;
  double beta = 0.0;

  cublasDgemm(handle[arts_get_gpu_id()], CUBLAS_OP_N, CUBLAS_OP_N, tile_size,
              tile_size, tile_size, &alpha, a_tile_dev, tile_size, b_tile_dev,
              tile_size, &beta, c_tile_dev, tile_size);

  arts_put_in_db_from_gpu(c_tile_dev, c_tile_guid, 0, size, true);
  arts_signal_edt(to_signal, k, c_tile_guid, DB_MODE_EW);
}

void sum_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)paramc;
  arts_guid_t done_guid = (arts_guid_t)paramv[0];
  unsigned int i = paramv[1];
  unsigned int j = paramv[2];

  double *c_tile = NULL;
  arts_guid_t c_tile_guid = arts_guid_reserve(ARTS_DB, 0);
  c_tile = (double *)arts_db_create_with_guid(
      c_tile_guid, sizeof(double) * tile_size * tile_size, ARTS_DB_GPU, NULL,
      NULL);
  init_matrix(tile_size, c_tile, false, true);

  thrust::device_ptr<double> c_ptr_dev((double *)depv[0].ptr);
  thrust::device_vector<double> c_tile_dev(c_ptr_dev,
                                           c_ptr_dev + (tile_size * tile_size));
  for (unsigned int k = 1; k < depc; ++k) {
    thrust::device_ptr<double> to_add_ptr_dev((double *)depv[k].ptr);
    thrust::device_vector<double> to_add_tile_dev(
        to_add_ptr_dev, to_add_ptr_dev + (tile_size * tile_size));
    thrust::transform(c_tile_dev.begin(), c_tile_dev.end(),
                      to_add_tile_dev.begin(), c_tile_dev.begin(),
                      thrust::plus<double>());
  }

  arts_put_in_db_from_gpu(thrust::raw_pointer_cast(c_tile_dev.data()),
                          c_tile_guid, 0,
                          sizeof(double) * tile_size * tile_size, false);
  arts_signal_edt(done_guid, 3 + ((i * num_blocks) + j), c_tile_guid,
                  DB_MODE_EW);
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
  for (unsigned int i = 0; i < mat_size; ++i) {
    for (unsigned int j = 0; j < mat_size; ++j) {
      for (unsigned int k = 0; k < mat_size; ++k) {
        temp[((i * mat_size)) + j] +=
            a_mat[((i * mat_size)) + k] * b_mat[((k * mat_size)) + j];
      }
    }
  }

  for (unsigned int i = 0; i < mat_size; ++i) {
    for (unsigned int j = 0; j < mat_size; ++j) {
      if (temp[((i * mat_size)) + j] != c_mat[((i * mat_size)) + j]) {
        arts_printf("Failed at c_mat[%u][%u]\n", i, j);
        arts_printf("Expected: %lf | Obtained: %lf\n",
                    temp[((i * mat_size)) + j], c_mat[((i * mat_size)) + j]);
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

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
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

  a_tile_guids = arts_guid_reserve_range(ARTS_DB, num_blocks * num_blocks, 0);
  b_tile_guids = arts_guid_reserve_range(ARTS_DB, num_blocks * num_blocks, 0);

  a_matrix = (double *)arts_db_create_with_guid(
      a_mat_guid, (size_t)mat_size * mat_size * sizeof(double), ARTS_DB_GPU,
      NULL, NULL);
  b_matrix = (double *)arts_db_create_with_guid(
      b_mat_guid, (size_t)mat_size * mat_size * sizeof(double), ARTS_DB_GPU,
      NULL, NULL);
  c_matrix = (double *)arts_db_create_with_guid(
      c_mat_guid, (size_t)mat_size * mat_size * sizeof(double), ARTS_DB_GPU,
      NULL, NULL);

  init_matrix(mat_size, a_matrix, true, false);
  init_matrix(mat_size, b_matrix, false, false);
  init_matrix(mat_size, c_matrix, false, true);

  arts_printf("Starting\n");

  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      arts_guid_t a_tile_guid =
          arts_guid_from_index(a_tile_guids, (i * num_blocks) + j);
      double *a_tile = (double *)arts_db_create_with_guid(
          a_tile_guid, sizeof(double) * tile_size * tile_size, ARTS_DB_GPU,
          NULL, NULL);
      copy_block(i, j, tile_size, a_tile, mat_size, a_matrix, true);

      arts_guid_t b_tile_guid =
          arts_guid_from_index(b_tile_guids, (i * num_blocks) + j);
      double *b_tile = (double *)arts_db_create_with_guid(
          b_tile_guid, sizeof(double) * tile_size * tile_size, ARTS_DB_GPU,
          NULL, NULL);
      copy_block(i, j, tile_size, b_tile, mat_size, b_matrix, true);
    }
  }

  dim3 threads(tile_size, tile_size);
  dim3 grid(1, 1);

  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      uint64_t sum_args[] = {(uint64_t)done_guid, i, j};
      arts_gpu_hint_t gpu_hint_sum = {};
      gpu_hint_sum.gpu = -1;
      gpu_hint_sum.route = node_id;
      gpu_hint_sum.lib = true;
      arts_guid_t sum_guid = arts_edt_create_gpu(
          sum_mm, 3, sum_args, num_blocks, arts_from_dim3(grid),
          arts_from_dim3(threads), &gpu_hint_sum);
      for (unsigned int k = 0; k < num_blocks; k++) {
        uint64_t args[] = {(uint64_t)sum_guid, i, j, k};
        arts_gpu_hint_t gpu_hint_mul = {};
        gpu_hint_mul.gpu = -1;
        gpu_hint_mul.route = node_id;
        gpu_hint_mul.lib = true;
        arts_guid_t mul_guid =
            arts_edt_create_gpu(multiply_mm, 4, args, 2, arts_from_dim3(grid),
                                arts_from_dim3(threads), &gpu_hint_mul);
        arts_signal_edt(
            mul_guid, 0,
            arts_guid_from_index(a_tile_guids, (i * num_blocks) + k),
            DB_MODE_EW);
        arts_signal_edt(
            mul_guid, 1,
            arts_guid_from_index(b_tile_guids, (k * num_blocks) + j),
            DB_MODE_EW);
      }
    }
  }

  arts_edt_create_with_guid(finish_block_mm, done_guid, 0, NULL,
                            3 + (num_blocks * num_blocks));
  arts_signal_edt(done_guid, 0, c_mat_guid, DB_MODE_EW);
  arts_signal_edt(done_guid, 1, a_mat_guid, DB_MODE_EW);
  arts_signal_edt(done_guid, 2, b_mat_guid, DB_MODE_EW);
  start = arts_get_time_stamp();
}

extern "C" void arts_init_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream, int argc,
                                  const char *argv) {
  (void)node_id;
  (void)stream;
  (void)argc;
  (void)argv;
  if (!dev_id) {
    handle =
        (cublasHandle_t *)calloc(arts_get_num_gpus(), sizeof(cublasHandle_t));
  }
  cublasStatus_t stat = cublasCreate(&handle[dev_id]);
  (void)stat;
}

extern "C" void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream) {
  (void)node_id;
  (void)stream;
  cublasStatus_t stat = cublasDestroy(handle[dev_id]);
  (void)stat;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
