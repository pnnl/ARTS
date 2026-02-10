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
#include <string.h>

#include <sys/types.h>

#include "arts.h"

#define MATSIZE 3
#define TILE 1

uint64_t start = 0;

unsigned int num_blocks = 1;
arts_guid_t a_mat_guid = NULL_GUID;
arts_guid_t b_mat_guid = NULL_GUID;
arts_guid_t c_mat_guid = NULL_GUID;
arts_guid_range_t *a_tile_guids = NULL;
arts_guid_range_t *b_tile_guids = NULL;

void print_matrix(unsigned int row_size, float *mat) {
  unsigned int column_size = row_size;
  for (unsigned int i = 0; i < column_size; i++) {
    for (unsigned int j = 0; j < row_size; j++) {
      printf("%5.2f ", mat[(i * row_size) + j]);
    }
    printf("\n");
  }
}

void init_matrix(unsigned int row_size, float *mat, bool identity, bool zero) {
  unsigned int column_size = row_size;
  for (unsigned int i = 0; i < column_size; i++) {
    for (unsigned int j = 0; j < row_size; j++) {
      if (zero) {
        mat[(i * row_size) + j] = 0;
      } else if (identity) {
        if (i == j) {
          mat[(i * row_size) + j] = 1;
        } else {
          mat[(i * row_size) + j] = 0;
}
      } else {
        mat[(i * row_size) + j] = i * row_size + j;
}
    }
  }
}

void copy_block(unsigned int x, unsigned int y, unsigned int tile_row_size,
               float *tile, unsigned int row_size, float *mat, bool to_tile) {
  unsigned int tile_column_size = tile_row_size;
  unsigned int column_size = row_size;

  unsigned int x_offset = tile_row_size * x;
  unsigned int y_offset = tile_column_size * y;

  if (to_tile) {
    for (unsigned int i = 0; i < tile_column_size; i++) {
      memcpy(&tile[i * tile_row_size], &mat[((i + y_offset) * row_size) + x_offset],
             tile_row_size * sizeof(float));
}
  } else {
    for (unsigned int i = 0; i < tile_column_size; i++) {
      memcpy(&mat[((i + y_offset) * row_size) + x_offset], &tile[i * tile_row_size],
             tile_row_size * sizeof(float));
}
  }
}

void init_block_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  unsigned int i = paramv[0];
  unsigned int j = paramv[1];

  //    ARTS_PRINTF("%s %u %u\n", __func__, i, j);

  float *a_mat = (float *)depv[0].ptr;
  float *b_mat = (float *)depv[1].ptr;

  arts_guid_t a_guid = arts_get_guid(a_tile_guids, (i * num_blocks) + j);
  arts_guid_t b_guid = arts_get_guid(b_tile_guids, (i * num_blocks) + j);

  float *a_tile =
      (float *)arts_db_create_with_guid(a_guid, sizeof(float) * TILE * TILE);
  float *b_tile =
      (float *)arts_db_create_with_guid(b_guid, sizeof(float) * TILE * TILE);
  //    arts_db_create_with_guid_and_data(arts_guid_t guid, void * data, uint64_t size)

  copy_block(i, j, TILE, a_tile, MATSIZE, a_mat, true);
  copy_block(i, j, TILE, b_tile, MATSIZE, b_mat, true);
}

void mm_kernel_cpu(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  unsigned int I = paramv[4];
  unsigned int J = paramv[5];
  arts_guid_t to_signal = (arts_guid_t)paramv[1];
  unsigned int K = (unsigned int)paramv[2];
  //    ARTS_PRINTF("%s %u %u %u %u SIG: %u\n", __func__, I,K, K,J);
  arts_guid_t c_tile_guid = (arts_guid_t)paramv[3];
  const int blk = (int)paramv[0];
  float *A = (float *)depv[0].ptr;
  float *B = (float *)depv[1].ptr;
  float *C = (float *)depv[2].ptr;

  for (unsigned int i = 0; i < blk; i++) {
    // rows of B
    for (unsigned int j = 0; j < blk; j++) {
      // rows of A and columns of B
      for (unsigned int k = 0; k < blk; k++) {
        C[(i * blk) + j] += A[(i * blk) + k] * B[(k * blk) + j];
      }
    }
  }
  arts_signal_edt(to_signal, K, c_tile_guid);
}

void multiply_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  arts_guid_t to_signal = paramv[0];

  unsigned int row_size = TILE;
  unsigned int column_size = TILE;

  unsigned int i = paramv[1];
  unsigned int j = paramv[2];
  unsigned int k = paramv[3];

  ARTS_PRINTF("%s i: %u k: %u x % k: %u j: %u %lu %lu\n", __func__, i, k, k, j,
         depv[0].guid, depv[1].guid);

  float *a_tile = (float *)depv[0].ptr;
  float *b_tile = (float *)depv[1].ptr;
  float *c_tile = NULL;

  arts_guid_t c_tile_guid = arts_db_create(
      (void **)&c_tile, sizeof(float) * TILE * TILE, ARTS_DB_GPU_WRITE);
  init_matrix(row_size, c_tile, false, true);

  uint64_t args[] = {TILE, (uint64_t)to_signal, k, (uint64_t)c_tile_guid, i, j, k};
  arts_guid_t mul_gpu_guid = arts_edt_create(mm_kernel_cpu, 0, 7, args, 3);
  arts_signal_edt(mul_gpu_guid, 0, depv[0].guid);
  arts_signal_edt(mul_gpu_guid, 1, depv[1].guid);
  arts_signal_edt(mul_gpu_guid, 2, c_tile_guid);
}

void sum_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  arts_guid_t done_guid = paramv[0];

  unsigned int row_size = TILE;
  unsigned int column_size = TILE;

  unsigned int I = paramv[1];
  unsigned int J = paramv[2];

  //    ARTS_PRINTF("%s: i: %u j: %u %lu\n", __func__, I, J, done_guid);

  float *c_tile;
  arts_guid_t c_tile_guid = arts_db_create(
      (void **)&c_tile, sizeof(float) * TILE * TILE, ARTS_DB_GPU_WRITE);
  init_matrix(row_size, c_tile, false, true);

  for (unsigned int i = 0; i < depc; i++) {
    float *to_add = (float *)depv[i].ptr;
    for (unsigned int j = 0; j < column_size; j++) {
      for (unsigned int k = 0; k < row_size; k++) {
        c_tile[(j * row_size) + k] += to_add[(j * row_size) + k];
      }
    }
  }
  arts_signal_edt(done_guid, 1 + (I * num_blocks + J), c_tile_guid);
}

void finish_block_mm(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  ARTS_PRINTF("%s\n", __func__);
  arts_guid_t to_signal = paramv[0];
  float *c_mat = (float *)depv[0].ptr;
  for (unsigned int i = 0; i < num_blocks; i++) {
    for (unsigned int j = 0; j < num_blocks; j++) {
      float *c_tile = (float *)depv[1 + (i * num_blocks) + j].ptr;
      copy_block(i, j, TILE, c_tile, MATSIZE, c_mat, false);
    }
  }
  uint64_t time = arts_get_time_stamp() - start;
  //    print_matrix(MATSIZE, c_mat);
  ARTS_PRINTF("DONE %lu\n", time);
  arts_shutdown();
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  num_blocks = MATSIZE / TILE;

  a_mat_guid = arts_reserve_guid_route(ARTS_DB_READ, 0);
  b_mat_guid = arts_reserve_guid_route(ARTS_DB_READ, 0);
  c_mat_guid = arts_reserve_guid_route(ARTS_DB_READ, 0);

  a_tile_guids = arts_new_guid_range_node(ARTS_DB_PIN, num_blocks * num_blocks, 0);
  b_tile_guids = arts_new_guid_range_node(ARTS_DB_PIN, num_blocks * num_blocks, 0);
  if (!node_id) {
    float *a_mat = (float *)arts_db_create_with_guid(a_mat_guid, MATSIZE * MATSIZE *
                                                              sizeof(float));
    float *b_mat = (float *)arts_db_create_with_guid(b_mat_guid, MATSIZE * MATSIZE *
                                                              sizeof(float));
    float *c_mat = (float *)arts_db_create_with_guid(c_mat_guid, MATSIZE * MATSIZE *
                                                              sizeof(float));

    init_matrix(MATSIZE, a_mat, false, false);
    init_matrix(MATSIZE, b_mat, true, false);
    init_matrix(MATSIZE, c_mat, false, true);

    //        ARTS_PRINTF("A MATRIX\n");
    //        print_matrix(MATSIZE, a_mat);
    //        ARTS_PRINTF("B MATRIX\n");
    //        print_matrix(MATSIZE, b_mat);
    //        ARTS_PRINTF("C MATRIX\n");
    //        print_matrix(MATSIZE, c_mat);
    ARTS_PRINTF("Starting\n");
  }
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {
  if (!node_id && !worker_id) {
    arts_guid_t done_guid =
        arts_edt_create(finish_block_mm, 0, 0, NULL, 1 + (num_blocks * num_blocks));
    arts_signal_edt(done_guid, 0, c_mat_guid);

    for (unsigned int i = 0; i < num_blocks; i++) {
      for (unsigned int j = 0; j < num_blocks; j++) {
        uint64_t init_args[] = {i, j};
        arts_guid_t init_guid = arts_edt_create(init_block_mm, 0, 2, init_args, 2);
        arts_signal_edt(init_guid, 0, a_mat_guid);
        arts_signal_edt(init_guid, 1, b_mat_guid);

        uint64_t sum_args[] = {(uint64_t)done_guid, i, j};
        arts_guid_t sum_guid = arts_edt_create(sum_mm, 0, 3, sum_args, num_blocks);
        ARTS_PRINTF("SUMGUID: i: %u j: %u %lu\n", i, j, sum_guid);
        for (unsigned int k = 0; k < num_blocks; k++) {
          uint64_t args[] = {(uint64_t)sum_guid, i, j, k};
          arts_guid_t mul_guid = arts_edt_create(multiply_mm, 0, 4, args, 2);
          ARTS_PRINTF("%lu Signaling: i: %u k: %u %lu i: %u k: %u %lu\n", mul_guid, i,
                 k, arts_get_guid(a_tile_guids, (i * num_blocks) + k), k, j,
                 arts_get_guid(b_tile_guids, (k * num_blocks) + j));
          arts_signal_edt(mul_guid, 0, arts_get_guid(a_tile_guids, (i * num_blocks) + k));
          arts_signal_edt(mul_guid, 1, arts_get_guid(b_tile_guids, (k * num_blocks) + j));
        }
      }
    }
    start = arts_get_time_stamp();
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

// #include <stdio.h>
// #include <stdlib.h>
// #include "arts.h"
//
// #define NUMGUIDS 128
//
// arts_guid_t done_guid = NULL_GUID;
// arts_guid_range_t * guids = NULL;
//
// void create(uint32_t paramc, uint64_t * paramv, uint32_t depc, arts_edt_dep_t
// depv[])
//{
//     unsigned int i = paramv[0];
//     arts_guid_t guid = arts_get_guid(guids, i);
//     arts_db_create_with_guid(guid, sizeof(float));
// }
//
// void work(uint32_t paramc, uint64_t * paramv, uint32_t depc, arts_edt_dep_t
// depv[])
//{
//     arts_guid_t to_signal = (arts_guid_t) paramv[0];
//     arts_signal_edt(to_signal, 0, NULL_GUID);
// }
//
// void stage(uint32_t paramc, uint64_t * paramv, uint32_t depc, arts_edt_dep_t
// depv[])
//{
//     arts_guid_t to_signal = paramv[0];
//     unsigned int i      = paramv[1];
//
//     arts_guid_t guid    = arts_get_guid(guids, i);
//     arts_guid_t edt_guid = arts_edt_create(work, 0, 1, paramv, 1);
//     arts_signal_edt(edt_guid, 0, guid);
// }
//
// void finish(uint32_t paramc, uint64_t * paramv, uint32_t depc, arts_edt_dep_t
// depv[])
//{
//     ARTS_PRINTF("DONE\n");
//     arts_shutdown();
// }
//
// void init_per_node(unsigned int node_id, int argc, char** argv)
//{
//     done_guid = arts_reserve_guid_route(ARTS_EDT, 0);
//     guids = arts_new_guid_range_node(ARTS_DB_GPU, NUMGUIDS, 0);
// }
//
// void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
// char** argv)
//{
//     if(!node_id && !worker_id)
//     {
//         arts_edt_create_with_guid(finish, done_guid, 0, NULL, NUMGUIDS);
//
//         for(unsigned int i=0; i<NUMGUIDS; i++)
//         {
//             arts_edt_create(create, 0, 1, (uint64_t*)&i, 0);
//
//             uint64_t args[] = {done_guid, i};
//             arts_edt_create(stage, 0, 2, args, 0);
//         }
//     }
// }
//
// int main(int argc, char** argv)
//{
//     arts_rt(argc, argv);
//     return 0;
// }
