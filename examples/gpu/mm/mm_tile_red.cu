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

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"

#include "mm_util.h"

#define MATSIZE 1024
#define TILESIZE 32
// #define VERIFY 1
#define SMTILE 32

#define ARTS_PRINTF(...)
// #define ARTS_PRINTF(...) ARTS_PRINTF(__VA_ARGS__)

uint64_t start = 0;

unsigned int matSize;
unsigned int tile_size;
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
arts_guid_t *cTileGuids = NULL;

cublasHandle_t *handle;

typedef struct {
  unsigned int numLeaves;
  unsigned int totalNodes;
  unsigned int interiorNodes;
  arts_guid_t *redDbGuids;
  arts_guid_t *redEdtGuids;
} binaryReductionTree_t;

binaryReductionTree_t **redTree = NULL;

unsigned int left(unsigned int i) { return 2 * i + 1; }
unsigned int right(unsigned int i) { return 2 * i + 2; }
unsigned int parent(unsigned int i) { return (i - 1) / 2; }

unsigned int reserveEdtGuids(arts_guid_t *allGuids, unsigned int index,
                             arts_type_t edtType) {
  if (allGuids[index])
    return arts_guid_get_rank(allGuids[index]);
  // left Rank
  unsigned int rank = reserveEdtGuids(allGuids, left(index), edtType);
  // always reserve left rank
  allGuids[index] = arts_reserve_guid_route(edtType, rank);
  ARTS_PRINTF("edt: %d -> %lu\n", index, allGuids[index]);
  // visit right rank
  reserveEdtGuids(allGuids, right(index), edtType);
  return rank;
}

binaryReductionTree_t *
initBinaryReductionTree(unsigned int numLeaves, arts_edt_t fun_ptr,
                        arts_type_t db_type, arts_type_t edtType, uint32_t paramc,
                        const uint64_t *paramv, dim3 grid, dim3 block,
                        arts_guid_t end_guid, uint32_t slot) {
  binaryReductionTree_t *tree =
      (binaryReductionTree_t *)arts_calloc(1, sizeof(binaryReductionTree_t));
  tree->numLeaves = numLeaves;
  tree->totalNodes = 2 * numLeaves - 1;
  tree->interiorNodes = tree->totalNodes - tree->numLeaves;

  // Create space for all the guids
  arts_guid_t *allGuids =
      (arts_guid_t *)arts_calloc(tree->totalNodes, sizeof(arts_guid_t));
  tree->redDbGuids = &allGuids[tree->interiorNodes];
  tree->redEdtGuids = allGuids;

  // Reserves the db guids
  for (unsigned int i = 0; i < tree->numLeaves; i++)
    allGuids[tree->interiorNodes + i] =
        arts_reserve_guid_route(db_type, i % arts_get_total_nodes());

  // Reserves the edt guids
  reserveEdtGuids(allGuids, 0, edtType);

  // Check all the guids
  for (unsigned int i = 0; i < tree->totalNodes; i++) {
    ARTS_PRINTF("i: %u guid: %lu rank: %u type: %u\n", i, allGuids[i],
           arts_guid_get_rank(allGuids[i]), arts_guid_get_type(allGuids[i]));
  }

  // Set up the signals
  for (unsigned int i = 0; i < tree->interiorNodes; i++) {
    if (arts_is_guid_local(tree->redEdtGuids[i])) {
      if (!i) {
        ARTS_PRINTF("Last: %lu -> %lu slot: %u\n", tree->redEdtGuids[i], end_guid,
               slot);
        arts_edt_create_gpu_pt_with_guid(fun_ptr, tree->redEdtGuids[i], paramc, paramv,
                                   2, grid, block, end_guid, slot, 0);
      } else {
        int parentIndex = parent(i);
        bool isRight = right(parentIndex) == i;
        arts_guid_t to_signal = tree->redEdtGuids[parentIndex];
        int toSignalSlot = (isRight) ? 1 : 0;
        ARTS_PRINTF("%lu -> %lu slot: %u parent: %d\n", tree->redEdtGuids[i],
               to_signal, toSignalSlot, parentIndex);
        arts_edt_create_gpu_pt_with_guid(fun_ptr, tree->redEdtGuids[i], paramc, paramv,
                                   2, grid, block, to_signal, toSignalSlot, 0);
      }
    }
  }

  return tree;
}

void fireBinaryReductionTree(binaryReductionTree_t *tree) {
  // Signal the top edts
  for (unsigned int i = 0; i < tree->numLeaves; i++) {
    int index = tree->interiorNodes + i;
    int parentIndex = parent(index);
    bool isRight = right(parentIndex) == index;
    arts_guid_t to_signal = tree->redEdtGuids[parentIndex];
    int toSignalSlot = (isRight) ? 1 : 0;
    ARTS_PRINTF("ToSignal: %lu slot: %u\n", to_signal, toSignalSlot);
    arts_signal_edt(to_signal, toSignalSlot, tree->redDbGuids[i]);
  }
}

void fireDbFromReductionTree(binaryReductionTree_t *tree,
                             unsigned int whichDb) {
  int index = tree->interiorNodes + whichDb;
  int parentIndex = parent(index);
  bool isRight = right(parentIndex) == index;
  arts_guid_t to_signal = tree->redEdtGuids[parentIndex];
  int toSignalSlot = (isRight) ? 1 : 0;
  ARTS_PRINTF("ToSignal: %lu slot: %u\n", to_signal, toSignalSlot);
  arts_signal_edt(to_signal, toSignalSlot, tree->redDbGuids[whichDb]);
}

void multiply_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  // arts_guid_t to_signal = paramv[0];
  unsigned int size = sizeof(double) * tile_size * tile_size;
  unsigned int i = paramv[1];
  unsigned int j = paramv[2];
  unsigned int k = paramv[3];

  // arts_guid_t aTileGuid = depv[0].guid;
  // arts_guid_t bTileGuid = depv[1].guid;
  arts_guid_t c_tile_guid = paramv[4];

  double *aTileDev = (double *)depv[0].ptr;
  double *bTileDev = (double *)depv[1].ptr;
  double *cTileDev = (double *)arts_cuda_malloc(size);

  double alpha = 1.0;
  double beta = 0.0;

  cublasDgemm(handle[arts_get_gpu_id()], CUBLAS_OP_N, CUBLAS_OP_N, tile_size,
              tile_size, tile_size, &alpha, aTileDev, tile_size, bTileDev,
              tile_size, &beta, cTileDev, tile_size);

  double *cTileHost = (double *)arts_db_create_with_guid(c_tile_guid, size);
  arts_put_in_db_from_gpu(cTileDev, c_tile_guid, 0, size, true);
  fireDbFromReductionTree(redTree[i * num_blocks + j], k);
  // arts_signal_edt(to_signal, k, c_tile_guid);
}

__global__ void sumMMKernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            arts_edt_dep_t depv[]) {
  const unsigned int column_size = (unsigned int)paramv[0];
  double *c_tile = (double *)depv[0].ptr;
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;
  for (unsigned int k = 1; k < depc; ++k) {
    double *to_add = (double *)depv[k].ptr;
    c_tile[row * column_size + col] += to_add[row * column_size + col];
  }
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

  uint64_t sum_args[] = {tile_size};
  dim3 threads(SMTILE, SMTILE);
  dim3 grid((tile_size + SMTILE - 1) / SMTILE, (tile_size + SMTILE - 1) / SMTILE);
  redTree = (binaryReductionTree_t **)arts_calloc(
      num_blocks * num_blocks, sizeof(binaryReductionTree_t *));
  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      redTree[i * num_blocks + j] = initBinaryReductionTree(
          num_blocks, sumMMKernel, ARTS_DB_GPU_WRITE, ARTS_GPU_EDT, 1, sum_args,
          grid, threads, done_guid, 3 + (i * num_blocks + j));
    }
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

  // uint64_t sum_args[] = {tile_size};
  dim3 threads(SMTILE, SMTILE);
  dim3 grid((tile_size + SMTILE - 1) / SMTILE, (tile_size + SMTILE - 1) / SMTILE);

  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      if ((i * num_blocks + j) % total_threads == globalThreadId) {
        // arts_guid_t sum_guid = arts_edt_create_gpu_pt (sumMMKernel, node_id, 1,
        // sum_args, num_blocks, grid, threads, done_guid, 3 + (i * num_blocks + j),
        // 0);
        arts_guid_t *c_guid = redTree[i * num_blocks + j]->redDbGuids;
        for (unsigned int k = 0; k < num_blocks; k++) {
          uint64_t args[] = {0, i, j, k, (uint64_t)c_guid[k]};
          arts_guid_t mul_guid = arts_edt_create_gpu_lib(multiply_mm, node_id, 5, args,
                                                   2, grid, threads);
          arts_signal_edt(mul_guid, 0, arts_get_guid(a_tile_guids, i * num_blocks + k));
          arts_signal_edt(mul_guid, 1, arts_get_guid(b_tile_guids, k * num_blocks + j));
        }
        // fireBinaryReductionTree(redTree[i*num_blocks + j]);
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
