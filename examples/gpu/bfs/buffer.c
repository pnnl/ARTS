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
#include "buffer.h"

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"
#include <stdlib.h>

#define NUMBUFFERS 2

static volatile unsigned int current_buffer = 0;
static unsigned int ***gpu_buffer_ptr = NULL; // NUMBUFFERS per GPU (Many)
static unsigned int **cpu_buffer_ptr = NULL;  // NUMBUFFERS per Node (One)

static arts_guid_t *master_buffer_guids = NULL;
static arts_guid_t *buffer_guids = NULL;
static unsigned int ***buffer_ptr = NULL;

void create_buffers_on_cpu(unsigned int size) {
  unsigned int num_nodes = arts_get_total_nodes();
  unsigned int num_gpus = arts_get_total_gpus();
  unsigned int node_id = arts_get_current_node();

  cpu_buffer_ptr = (unsigned int **)calloc(NUMBUFFERS, sizeof(unsigned int *));
  for (unsigned int i = 0; i < NUMBUFFERS; i++) {
    cpu_buffer_ptr[i] = (unsigned int *)calloc(size, sizeof(unsigned int));
  }

  gpu_buffer_ptr =
      (unsigned int ***)calloc(NUMBUFFERS, sizeof(unsigned int **));
  for (unsigned int i = 0; i < NUMBUFFERS; i++) {
    gpu_buffer_ptr[i] =
        (unsigned int **)calloc(num_gpus, sizeof(unsigned int *));
  }

  master_buffer_guids = (arts_guid_t *)calloc(num_nodes, sizeof(arts_guid_t));
  for (unsigned int i = 0; i < num_nodes; i++) {
    master_buffer_guids[i] = arts_guid_reserve(ARTS_DB, i);
  }

  buffer_guids = (arts_guid_t *)arts_db_create_with_guid(
      master_buffer_guids[node_id],
      sizeof(arts_guid_t) * num_nodes * NUMBUFFERS, NULL);
  for (unsigned int i = 0; i < NUMBUFFERS; i++) {
    for (unsigned int j = 0; j < num_nodes; j++) {
      buffer_guids[(i * num_nodes) + j] =
          arts_guid_reserve(ARTS_DB_GPU, j);
    }
  }
}

void create_buffers_on_gpu(unsigned int gpu, unsigned int size) {
  if (!gpu_buffer_ptr) {
    arts_printf("Must run create_buffers_on_cpu first!\n");
  }

  for (unsigned int i = 0; i < NUMBUFFERS; i++) {
    gpu_buffer_ptr[i][gpu] =
        (unsigned int *)arts_cuda_malloc(sizeof(unsigned int) * size);
  }
}

void create_buffer_db() {
  unsigned int num_nodes = arts_get_total_nodes();
  unsigned int num_gpus = arts_get_total_gpus();
  unsigned int node_id = arts_get_current_node();

  buffer_ptr = (unsigned int ***)calloc(NUMBUFFERS, sizeof(unsigned int **));
  for (unsigned int j = 0; j < NUMBUFFERS; j++) {

    buffer_ptr[j] = (unsigned int **)arts_db_create_with_guid(
        buffer_guids[(j * num_nodes) + node_id],
        sizeof(unsigned int *) * (num_gpus + 1), NULL);
    for (uint64_t i = 0; i < num_gpus; i++) {
      buffer_ptr[j][i] = gpu_buffer_ptr[j][i];
    }
    buffer_ptr[j][num_gpus] = cpu_buffer_ptr[j];
  }
}

void free_buffers_on_gpu(unsigned int gpu) {
  if (!gpu_buffer_ptr) {
    arts_printf("Must run create_buffers_on_cpu first!\n");
  }

  for (unsigned int i = 0; i < NUMBUFFERS; i++) {
    arts_cuda_free(gpu_buffer_ptr[i][gpu]);
  }
}

void print_master_buffer_guids() {
  unsigned int num_nodes = arts_get_total_nodes();
  for (unsigned int i = 0; i < num_nodes; i++) {
    arts_printf("master_buffer_guids[%u]: %lu\n", i, master_buffer_guids[i]);
  }
}

void print_local_buffer_guids() {
  unsigned int num_nodes = arts_get_total_nodes();
  unsigned int node_id = arts_get_current_node();
  for (unsigned int i = 0; i < NUMBUFFERS; i++) {
    arts_printf("buffer_guids[%u][%u]: %lu\n", i, node_id,
                buffer_guids[(i * num_nodes) + node_id]);
  }
}

void print_buffer_ptr() {
  unsigned int num_gpus = arts_get_total_gpus();
  for (unsigned int i = 0; i < NUMBUFFERS; i++) {
    for (unsigned int j = 0; j < num_gpus + 1; j++) {
      arts_printf("buffer: %u buffer_ptr[%u]: %p\n", i, j, buffer_ptr[i][j]);
    }
  }
}

void print_raw_ptr() {
  unsigned int num_gpus = arts_get_total_gpus();
  for (unsigned int i = 0; i < NUMBUFFERS; i++) {
    for (unsigned int j = 0; j < num_gpus; j++) {
      arts_printf("buffer: %u gpu_buffer_ptr[%u]: %p\n", i, j,
                  gpu_buffer_ptr[i][j]);
    }
    arts_printf("buffer: %u cpu_buffer_ptr   : %p\n", i, cpu_buffer_ptr[i]);
  }
}

arts_guid_t get_buffer_guid(unsigned int node_id, uint64_t level) {
  unsigned int num_nodes = arts_get_total_nodes();
  uint64_t index = level % NUMBUFFERS;
  // arts_printf("Get index: %u node_id: %u %lu\n", index, node_id,
  // buffer_guids[index*num_nodes + node_id]);
  return buffer_guids[(index * num_nodes) + node_id];
}

unsigned int *get_local_buffer(unsigned int index, uint64_t level) {
  unsigned int num_gpus = arts_get_total_gpus();
  unsigned int buffer_index = level % NUMBUFFERS;
  if (index == num_gpus) {
    return cpu_buffer_ptr[buffer_index];
  }
  return gpu_buffer_ptr[buffer_index][index];
}

void reset_buffer(uint64_t level) {
  unsigned int num_nodes = arts_get_total_nodes();
  unsigned int node_id = arts_get_current_node();
  uint64_t index = level % NUMBUFFERS;

  for (unsigned int j = 0; j < num_nodes; j++) {
    buffer_guids[(index * num_nodes) + j] =
        arts_db_rename(buffer_guids[(index * num_nodes) + j]);
    // arts_printf("RENAME index: %u node: %u %lu\n", index, j,
    // buffer_guids[index*num_nodes + j]);
  }

  unsigned int offset = &buffer_guids[index * num_nodes] - buffer_guids;
  void *src = (void *)&buffer_guids[index * num_nodes];
  for (unsigned int i = 0; i < num_nodes; i++) {
    if (i != node_id) {
      arts_put_in_db_at(src, NULL_GUID, master_buffer_guids[i], -1, offset,
                        sizeof(arts_guid_t) * num_nodes, i);
    }
  }
}