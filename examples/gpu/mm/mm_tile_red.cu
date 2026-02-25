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
#include "arts/gpu.h"

#include "mm_util.h"

#define MATSIZE 1024
#define TILESIZE 32
// #define VERIFY 1
#define SMTILE 32

#define ARTS_PRINTF(...)
// #define ARTS_PRINTF(...) arts_printf(__VA_ARGS__)

uint64_t start = 0;

unsigned int mat_size;
unsigned int tile_size;
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
arts_guid_t *c_tile_guids = NULL;

cublasHandle_t *handle;

typedef struct {
  unsigned int num_leaves;
  unsigned int total_nodes;
  unsigned int interior_nodes;
  arts_guid_t *red_db_guids;
  arts_guid_t *red_edt_guids;
} binary_reduction_tree_t;

binary_reduction_tree_t **red_tree = NULL;

unsigned int left(unsigned int i) { return (2 * i) + 1; }
unsigned int right(unsigned int i) { return (2 * i) + 2; }
unsigned int parent(unsigned int i) { return (i - 1) / 2; }

unsigned int reserve_edt_guids(arts_guid_t *all_guids, unsigned int index,
                               arts_type_t edt_type) {
  if (all_guids[index]) {
    return arts_guid_get_rank(all_guids[index]);
  }
  // left Rank
  unsigned int rank = reserve_edt_guids(all_guids, left(index), edt_type);
  // always reserve left rank
  all_guids[index] = arts_guid_reserve(edt_type, rank);
  ARTS_PRINTF("edt: %d -> %lu\n", index, all_guids[index]);
  // visit right rank
  reserve_edt_guids(all_guids, right(index), edt_type);
  return rank;
}

binary_reduction_tree_t *
init_binary_reduction_tree(unsigned int num_leaves, arts_edt_t fun_ptr,
                           arts_type_t db_type, arts_type_t edt_type,
                           uint32_t paramc, const uint64_t *paramv, dim3 grid,
                           dim3 block, arts_guid_t end_guid, uint32_t slot) {
  binary_reduction_tree_t *tree =
      (binary_reduction_tree_t *)calloc(1, sizeof(binary_reduction_tree_t));
  tree->num_leaves = num_leaves;
  tree->total_nodes = (2 * num_leaves) - 1;
  tree->interior_nodes = tree->total_nodes - tree->num_leaves;

  // Create space for all the guids
  arts_guid_t *all_guids =
      (arts_guid_t *)calloc(tree->total_nodes, sizeof(arts_guid_t));
  tree->red_db_guids = &all_guids[tree->interior_nodes];
  tree->red_edt_guids = all_guids;

  // Reserves the db guids
  for (unsigned int i = 0; i < tree->num_leaves; i++) {
    all_guids[tree->interior_nodes + i] =
        arts_guid_reserve(db_type, i % arts_get_total_nodes());
  }

  // Reserves the edt guids
  reserve_edt_guids(all_guids, 0, edt_type);

  // Check all the guids
  for (unsigned int i = 0; i < tree->total_nodes; i++) {
    ARTS_PRINTF("i: %u guid: %lu rank: %u type: %u\n", i, all_guids[i],
                arts_guid_get_rank(all_guids[i]),
                arts_guid_get_type(all_guids[i]));
  }

  // Set up the signals
  for (unsigned int i = 0; i < tree->interior_nodes; i++) {
    if (arts_guid_is_local(tree->red_edt_guids[i])) {
      if (!i) {
        ARTS_PRINTF("Last: %lu -> %lu slot: %u\n", tree->red_edt_guids[i],
                    end_guid, slot);
        arts_gpu_hint_t gpu_hint_root = {};
        gpu_hint_root.gpu = -1;
        gpu_hint_root.end_guid = end_guid;
        gpu_hint_root.slot = slot;
        gpu_hint_root.data_guid = (arts_guid_t)0;
        gpu_hint_root.passthrough = true;
        arts_edt_create_gpu_with_guid(fun_ptr, tree->red_edt_guids[i], paramc,
                                      paramv, 2, arts_from_dim3(grid),
                                      arts_from_dim3(block), &gpu_hint_root);
      } else {
        unsigned int parent_index = parent(i);
        bool is_right = right(parent_index) == i;
        arts_guid_t to_signal = tree->red_edt_guids[parent_index];
        unsigned int to_signal_slot = (is_right) ? 1 : 0;
        ARTS_PRINTF("%lu -> %lu slot: %u parent: %d\n", tree->red_edt_guids[i],
                    to_signal, to_signal_slot, parent_index);
        arts_gpu_hint_t gpu_hint_inner = {};
        gpu_hint_inner.gpu = -1;
        gpu_hint_inner.end_guid = to_signal;
        gpu_hint_inner.slot = to_signal_slot;
        gpu_hint_inner.data_guid = (arts_guid_t)0;
        gpu_hint_inner.passthrough = true;
        arts_edt_create_gpu_with_guid(fun_ptr, tree->red_edt_guids[i], paramc,
                                      paramv, 2, arts_from_dim3(grid),
                                      arts_from_dim3(block), &gpu_hint_inner);
      }
    }
  }

  return tree;
}

void fire_binary_reduction_tree(binary_reduction_tree_t *tree) {
  // Signal the top edts
  for (unsigned int i = 0; i < tree->num_leaves; i++) {
    unsigned int index = tree->interior_nodes + i;
    unsigned int parent_index = parent(index);
    bool is_right = right(parent_index) == index;
    arts_guid_t to_signal = tree->red_edt_guids[parent_index];
    unsigned int to_signal_slot = (is_right) ? 1 : 0;
    ARTS_PRINTF("ToSignal: %lu slot: %u\n", to_signal, to_signal_slot);
    arts_signal_edt(to_signal, to_signal_slot, tree->red_db_guids[i],
                    DB_MODE_EW);
  }
}

void fire_db_from_reduction_tree(binary_reduction_tree_t *tree,
                                 unsigned int which_db) {
  unsigned int index = tree->interior_nodes + which_db;
  unsigned int parent_index = parent(index);
  bool is_right = right(parent_index) == index;
  arts_guid_t to_signal = tree->red_edt_guids[parent_index];
  unsigned int to_signal_slot = (is_right) ? 1 : 0;
  ARTS_PRINTF("ToSignal: %lu slot: %u\n", to_signal, to_signal_slot);
  arts_signal_edt(to_signal, to_signal_slot, tree->red_db_guids[which_db],
                  DB_MODE_EW);
}

void multiply_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  // arts_guid_t to_signal = paramv[0];
  unsigned int size = sizeof(double) * tile_size * tile_size;
  unsigned int i = paramv[1];
  unsigned int j = paramv[2];
  unsigned int k = paramv[3];

  // arts_guid_t a_tile_guid = depv[0].guid;
  // arts_guid_t b_tile_guid = depv[1].guid;
  arts_guid_t c_tile_guid = (arts_guid_t)paramv[4];

  double *a_tile_dev = (double *)depv[0].ptr;
  double *b_tile_dev = (double *)depv[1].ptr;
  double *c_tile_dev = (double *)arts_cuda_malloc(size);

  double alpha = 1.0;
  double beta = 0.0;

  cublasDgemm(handle[arts_get_gpu_id()], CUBLAS_OP_N, CUBLAS_OP_N,
              (int)tile_size, (int)tile_size, (int)tile_size, &alpha,
              a_tile_dev, (int)tile_size, b_tile_dev, (int)tile_size, &beta,
              c_tile_dev, (int)tile_size);

  double *c_tile_host = (double *)arts_db_create_with_guid(
      c_tile_guid, size, ARTS_DB_GPU, NULL, NULL);
  (void)c_tile_host;
  arts_put_in_db_from_gpu(c_tile_dev, c_tile_guid, 0, size, true);
  fire_db_from_reduction_tree(red_tree[(i * num_blocks) + j], k);
  // arts_signal_edt(to_signal, k, c_tile_guid, DB_MODE_EW);
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
  ARTS_PRINTF("Verifying results...\n");
  double *temp = (double *)calloc((size_t)mat_size * mat_size, sizeof(double));
  for (unsigned int i = 0; i < mat_size; ++i) {
    for (unsigned int j = 0; j < mat_size; ++j) {
      for (unsigned int k = 0; k < mat_size; ++k) {
        temp[(i * mat_size) + j] +=
            a_mat[(i * mat_size) + k] * b_mat[(k * mat_size) + j];
      }
    }
  }

  for (unsigned int i = 0; i < mat_size; ++i) {
    for (unsigned int j = 0; j < mat_size; ++j) {
      if (temp[(i * mat_size) + j] != c_mat[(i * mat_size) + j]) {
        ARTS_PRINTF("Failed at c_mat[%u][%u]\n", i, j);
        ARTS_PRINTF("Expected: %lf | Obtained: %lf\n", temp[(i * mat_size) + j],
                    c_mat[(i * mat_size) + j]);
        free(temp);
        arts_shutdown();
        return;
      }
    }
  }

  free(temp);
  ARTS_PRINTF("Success %lu\n", time);
#else
  ARTS_PRINTF("Done %lu\n", time);
#endif
  arts_shutdown();
}

void init_node(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int node_id = arts_get_current_node();

  num_blocks = mat_size / tile_size;
  done_guid = arts_guid_reserve(ARTS_EDT, 0);
  a_mat_guid = arts_guid_reserve(ARTS_DB, 0);
  b_mat_guid = arts_guid_reserve(ARTS_DB, 0);
  c_mat_guid = arts_guid_reserve(ARTS_DB, 0);

  a_tile_guids = arts_guid_reserve_range(ARTS_DB, num_blocks * num_blocks, 0);
  b_tile_guids = arts_guid_reserve_range(ARTS_DB, num_blocks * num_blocks, 0);

  if (!node_id) {
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

    ARTS_PRINTF("Starting\n");
  }

  uint64_t sum_args[] = {tile_size};
  dim3 threads(SMTILE, SMTILE);
  dim3 grid((tile_size + SMTILE - 1) / SMTILE,
            (tile_size + SMTILE - 1) / SMTILE);
  red_tree = (binary_reduction_tree_t **)calloc(
      (size_t)num_blocks * num_blocks, sizeof(binary_reduction_tree_t *));
  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      red_tree[(i * num_blocks) + j] = init_binary_reduction_tree(
          num_blocks, sum_mm_kernel, ARTS_DB, ARTS_EDT, 1, sum_args, grid,
          threads, done_guid, 3 + ((i * num_blocks) + j));
    }
  }
}

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  unsigned int node_id = arts_get_current_node();

  if (argc == 1) {
    mat_size = MATSIZE;
    tile_size = TILESIZE;
  } else if (argc == 2) {
    mat_size = (unsigned int)strtol(argv[1], NULL, 10);
    tile_size = TILESIZE;
  } else {
    mat_size = (unsigned int)strtol(argv[1], NULL, 10);
    tile_size = (unsigned int)strtol(argv[2], NULL, 10);
  }

  arts_guid_t init_epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    arts_hint_t hint_0 = {i, 0};
    arts_edt_create_with_epoch(init_node, paramc, paramv, 0, init_epoch_guid,
                               &hint_0);
  }
  arts_wait_on_handle(init_epoch_guid);

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

  dim3 threads(SMTILE, SMTILE);
  dim3 grid((tile_size + SMTILE - 1) / SMTILE,
            (tile_size + SMTILE - 1) / SMTILE);

  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      arts_guid_t *c_guid = red_tree[(i * num_blocks) + j]->red_db_guids;
      for (unsigned int k = 0; k < num_blocks; k++) {
        uint64_t args[] = {0, i, j, k, (uint64_t)c_guid[k]};
        arts_gpu_hint_t gpu_hint_mul = {};
        gpu_hint_mul.gpu = -1;
        gpu_hint_mul.route = node_id;
        gpu_hint_mul.lib = true;
        arts_guid_t mul_guid =
            arts_edt_create_gpu(multiply_mm, 5, args, 2, arts_from_dim3(grid),
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
