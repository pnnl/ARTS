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
#include "arts/gpu/gpu_runtime.cuh"

#include "mm_util.h"

#define MATSIZE 1024
#define TILESIZE 32
// #define VERIFY 1

uint64_t start = 0;
cublasHandle_t *handle;

int matSize;
int tile_size;
unsigned int num_blocks = 1;

arts_guid_t a_mat_guid = NULL_GUID;
arts_guid_t b_mat_guid = NULL_GUID;
arts_guid_t c_mat_guid = NULL_GUID;
arts_guid_t done_guid = NULL_GUID;

double *aMatrix = NULL;
double *bMatrix = NULL;
double *cMatrix = NULL;

arts_guid_range_t *a_tile_guids = NULL;
arts_guid_range_t *b_tile_guids = NULL;

void multiply_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  arts_guid_t to_signal = paramv[0];
  unsigned int size = sizeof(double) * tile_size * tile_size;
  // unsigned int i = paramv[1];
  // unsigned int j = paramv[2];
  unsigned int k = paramv[3];

  double *aTileDev = (double *)depv[0].ptr;
  double *bTileDev = (double *)depv[1].ptr;
  double *cTileHost = NULL;

  // arts_guid_t aTileGuid = depv[0].guid;
  // arts_guid_t bTileGuid = depv[1].guid;
  arts_guid_t c_tile_guid =
      arts_db_create((void **)&cTileHost, size, ARTS_DB_GPU_WRITE);

  double *cTileDev = (double *)arts_cuda_malloc(size);

  double alpha = 1.0;
  double beta = 0.0;

  cublasDgemm(handle[arts_get_gpu_id()], CUBLAS_OP_N, CUBLAS_OP_N, tile_size,
              tile_size, tile_size, &alpha, aTileDev, tile_size, bTileDev,
              tile_size, &beta, cTileDev, tile_size);

  arts_put_in_db_from_gpu(cTileDev, c_tile_guid, 0, size, true);
  arts_signal_edt(to_signal, k, c_tile_guid);
}

void sum_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  arts_guid_t done_guid = paramv[0];
  unsigned int i = paramv[1];
  unsigned int j = paramv[2];

  double *c_tile = NULL;
  arts_guid_t c_tile_guid = arts_db_create(
      (void **)&c_tile, sizeof(double) * tile_size * tile_size, ARTS_DB_GPU_WRITE);
  init_matrix(tile_size, c_tile, false, true);

  thrust::device_ptr<double> cPtrDev((double *)depv[0].ptr);
  thrust::device_vector<double> cTileDev(cPtrDev,
                                         cPtrDev + (tile_size * tile_size));
  for (unsigned int k = 1; k < depc; ++k) {
    thrust::device_ptr<double> toAddPtrDev((double *)depv[k].ptr);
    thrust::device_vector<double> toAddTileDev(
        toAddPtrDev, toAddPtrDev + (tile_size * tile_size));
    thrust::transform(cTileDev.begin(), cTileDev.end(), toAddTileDev.begin(),
                      cTileDev.begin(), thrust::plus<double>());
  }

  arts_put_in_db_from_gpu(thrust::raw_pointer_cast(cTileDev.data()), c_tile_guid, 0,
                     sizeof(double) * tile_size * tile_size, false);
  arts_signal_edt(done_guid, 3 + (i * num_blocks + j), c_tile_guid);
}

void finish_block_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  double *c_mat = (double *)depv[0].ptr;

  for (unsigned int i = 0; i < num_blocks; i++)
    for (unsigned int j = 0; j < num_blocks; j++) {
      double *c_tile = (double *)depv[3 + (i * num_blocks + j)].ptr;
      copy_block(i, j, tile_size, c_tile, matSize, c_mat, false);
    }

  uint64_t time = arts_get_time_stamp() - start;

#if VERIFY
  double *a_mat = (double *)depv[1].ptr;
  double *b_mat = (double *)depv[2].ptr;
  ARTS_PRINTF("Verifying results...\n");
  double *temp = (double *)arts_calloc(matSize * matSize, sizeof(double));
  for (unsigned int i = 0; i < matSize; ++i)
    for (unsigned int j = 0; j < matSize; ++j)
      for (unsigned int k = 0; k < matSize; ++k)
        temp[i * matSize + j] += a_mat[i * matSize + k] * b_mat[k * matSize + j];

  for (unsigned int i = 0; i < matSize; ++i)
    for (unsigned int j = 0; j < matSize; ++j)
      if (temp[i * matSize + j] != c_mat[i * matSize + j]) {
        ARTS_PRINTF("Failed at c_mat[%u][%u]\n", i, j);
        ARTS_PRINTF("Expected: %lf | Obtained: %lf\n", temp[i * matSize + j],
               c_mat[i * matSize + j]);
        arts_free(temp);
        arts_shutdown();
        return;
      }

  arts_free(temp);
  ARTS_PRINTF("Success %lu\n", time);
#else
  ARTS_PRINTF("Done %lu\n", time);
#endif

  arts_shutdown();
}

extern "C" void init_per_node(unsigned int node_id, int argc, char **argv) {
  if (argc == 1) {
    matSize = MATSIZE;
    tile_size = TILESIZE;
  } else if (argc == 2) {
    matSize = atoi(argv[1]);
    tile_size = TILESIZE;
  } else {
    matSize = atoi(argv[1]);
    tile_size = atoi(argv[2]);
  }
  num_blocks = matSize / tile_size;
  done_guid = arts_reserve_guid_route(ARTS_EDT, 0);
  a_mat_guid = arts_reserve_guid_route(ARTS_DB_READ, 0);
  b_mat_guid = arts_reserve_guid_route(ARTS_DB_READ, 0);
  c_mat_guid = arts_reserve_guid_route(ARTS_DB_READ, 0);

  a_tile_guids = arts_new_guid_range_node(ARTS_DB_GPU_READ, num_blocks * num_blocks, 0);
  b_tile_guids = arts_new_guid_range_node(ARTS_DB_GPU_READ, num_blocks * num_blocks, 0);

  if (!node_id) {
    aMatrix = (double *)arts_db_create_with_guid(a_mat_guid, matSize * matSize *
                                                           sizeof(double));
    bMatrix = (double *)arts_db_create_with_guid(b_mat_guid, matSize * matSize *
                                                           sizeof(double));
    cMatrix = (double *)arts_db_create_with_guid(c_mat_guid, matSize * matSize *
                                                           sizeof(double));

    init_matrix(matSize, aMatrix, true, false);
    init_matrix(matSize, bMatrix, false, false);
    init_matrix(matSize, cMatrix, false, true);

    ARTS_PRINTF("Starting\n");
  }
}

extern "C" void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  unsigned int total_threads = arts_get_total_nodes() * arts_get_total_workers();
  unsigned int globalThreadId = node_id * arts_get_total_workers() + worker_id;

  if (!node_id && !worker_id) {
    for (unsigned int i = 0; i < num_blocks; i++) {
      for (unsigned int j = 0; j < num_blocks; j++) {
        arts_guid_t aTileGuid = arts_get_guid(a_tile_guids, i * num_blocks + j);
        double *a_tile = (double *)arts_db_create_with_guid(
            aTileGuid, sizeof(double) * tile_size * tile_size);
        copy_block(i, j, tile_size, a_tile, matSize, aMatrix, true);

        arts_guid_t bTileGuid = arts_get_guid(b_tile_guids, i * num_blocks + j);
        double *b_tile = (double *)arts_db_create_with_guid(
            bTileGuid, sizeof(double) * tile_size * tile_size);
        copy_block(i, j, tile_size, b_tile, matSize, bMatrix, true);
      }
    }
  }

  dim3 threads(tile_size, tile_size);
  dim3 grid(1, 1);

  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      if ((i * num_blocks + j) % total_threads == globalThreadId) {
        uint64_t sum_args[] = {(uint64_t)done_guid, i, j};
        arts_guid_t sum_guid = arts_edt_create_gpu_lib(sum_mm, node_id, 3, sum_args,
                                                 num_blocks, grid, threads);
        for (unsigned int k = 0; k < num_blocks; k++) {
          uint64_t args[] = {(uint64_t)sum_guid, i, j, k};
          arts_guid_t mul_guid = arts_edt_create_gpu_lib(multiply_mm, node_id, 4, args,
                                                   2, grid, threads);
          arts_signal_edt(mul_guid, 0, arts_get_guid(a_tile_guids, i * num_blocks + k));
          arts_signal_edt(mul_guid, 1, arts_get_guid(b_tile_guids, k * num_blocks + j));
        }
      }
    }
  }

  if (!node_id && !worker_id) {
    arts_edt_create_with_guid(finish_block_mm, done_guid, 0, NULL,
                          3 + num_blocks * num_blocks);
    arts_signal_edt(done_guid, 0, c_mat_guid);
    arts_signal_edt(done_guid, 1, a_mat_guid);
    arts_signal_edt(done_guid, 2, b_mat_guid);
    start = arts_get_time_stamp();
  }
}

extern "C" void initPerGpu(unsigned int node_id, int devId, cudaStream_t *stream,
                           int argc, char *argv) {
  if (!devId)
    handle =
        (cublasHandle_t *)arts_calloc(arts_get_num_gpus(), sizeof(cublasHandle_t));
  cublasStatus_t stat = cublasCreate(&handle[devId]);
}

extern "C" void cleanPerGpu(unsigned int node_id, int devId,
                            cudaStream_t *stream) {
  cublasStatus_t stat = cublasDestroy(handle[devId]);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}