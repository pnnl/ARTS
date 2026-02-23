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

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include <cuda_runtime_api.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"
#include "arts/utils/atomics.h"

#include "bfs_defs.h"
#include "bins.h"
#include "buffer.h"
#include "graph_util.cuh"

uint64_t start = 0;  // Timer

unsigned int bounds[PARTS];       // This is the boundaries that make up each
                                  // partition
arts_block_dist_t *distribution;  // The graph distribution
csr_graph_t *graph;               // Partitions of the graph
unsigned int **visited;  // This is the resulting parent list for each partition
arts_guid_t
    *visited_guid;  // This is the guid for each partition of the parent list
unsigned int *part_count;

void create_first_round(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]);
__global__ void gpu_bfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]);
void cpu_bfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]);
void launch_sort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]);
void cpu_sort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]);
void thrust_sort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]);
void launch_bfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]);

void print_device_list(thrust::device_ptr<unsigned int> dev_ptr,
                       unsigned int size) {
  arts_printf("FRONTIER SIZE: %u\n", size);
  for (unsigned int i = 0; i < size; i++) {
    unsigned int temp = *(dev_ptr + i);
    arts_printf("%u, ", temp);
  }
  arts_printf("\n");
}

void print_result() {
  for (unsigned int i = 0; i < PARTS; i++) {
    unsigned int size =
        sizeof(unsigned int) * get_block_size_for_partition(i, distribution);
    arts_printf("%u: %u\n", i, size);
    for (unsigned int j = 0; j < size; j++) {
      arts_printf("%u, ", visited[i][j]);
    }
    arts_printf("\n");
  }
}

void create_first_round(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_printf("%s\n", __func__);
  start = arts_get_time_stamp();

  // We are on the rank of the source
  uint64_t src = paramv[0];
  uint64_t next_level = 0;

  // Create the first search frontier!
  unsigned int *first_search_frontier = NULL;
  arts_guid_t first_search_frontier_guid = arts_guid_reserve(ARTS_DB_GPU, 0);
  first_search_frontier = (unsigned int *)arts_db_create_with_guid(
      first_search_frontier_guid, 2 * sizeof(unsigned int), NULL);
  first_search_frontier[0] = 1;                  // size of the frontier
  first_search_frontier[1] = (unsigned int)src;  // root
  arts_printf(
      "ROOT: %u GRAPH GUID: %lu VISITED GUID: %lu\n", first_search_frontier[1],
      get_guid_for_vertex_distr(first_search_frontier[1], distribution),
      visited_guid[get_owner_distr(first_search_frontier[1], distribution)]);

  // Create the first epoch
  arts_hint_t hint_0 = {arts_get_current_node(), 0};
  arts_guid_t launch_sort_guid =
      arts_edt_create(launch_sort, 1, &next_level, 1, &hint_0);
  arts_initialize_and_start_epoch(launch_sort_guid, 0);

  // Launching the first bfs
  arts_guid_t graph_guid =
      get_guid_for_vertex_distr(first_search_frontier[1], distribution);
  arts_guid_t visit_guid =
      visited_guid[get_owner_distr(first_search_frontier[1], distribution)];
  arts_guid_t bfs_guid = NULL_GUID;
  if (first_search_frontier[0] > GPU_THRESHOLD) {
    dim3 threads(1, 1, 1);
    dim3 grid(1, 1, 1);
    bfs_guid =
        arts_edt_create_gpu(gpu_bfs, arts_get_current_node(), 1, &next_level, 4,
                            grid, threads, NULL_GUID, 0, NULL_GUID);
    arts_printf("LAUNCHING GPU\n");
  } else {
    arts_hint_t hint_1 = {arts_get_current_node(), 0};
    bfs_guid = arts_edt_create(cpu_bfs, 1, &next_level, 4, &hint_1);
    arts_printf("LAUNCHING CPU\n");
  }
  arts_signal_edt(bfs_guid, 0, visit_guid, DB_MODE_EW);
  // arts_signal_edt(bfs_guid, 1,
  // next_search_frontier_addr_guid[arts_get_current_node()], DB_MODE_EW);
  arts_signal_edt(bfs_guid, 1,
                  get_buffer_guid(arts_get_current_node(), next_level),
                  DB_MODE_EW);
  arts_signal_edt(bfs_guid, 2, first_search_frontier_guid, DB_MODE_EW);
  arts_signal_edt(bfs_guid, 3, graph_guid, DB_MODE_EW);
}

__global__ void gpu_bfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t gpu_id = GET_GPU_INDEX();  // The current gpu we are on
  unsigned int local_level = (unsigned int)paramv[0];
  unsigned int *local_visited = (unsigned int *)depv[0].ptr;
  unsigned int **addr =
      (unsigned int **)depv[1].ptr;  // This is the dev_ptr_raw -> tells us
                                     // where next frontier is on device
  unsigned int *local =
      addr[gpu_id];  // We need the one corresponding to our gpu
  unsigned int *local_frontier_count = &local[GPULISTLEN];

  unsigned int current_frontier_size = *((unsigned int *)depv[2].ptr);
  unsigned int *current_frontier = ((unsigned int *)depv[2].ptr) + 1;
  csr_graph_t *local_graph = (csr_graph_t *)depv[3].ptr;

  int index = (int)(threadIdx.x + (blockIdx.x * blockDim.x));
  if (index < current_frontier_size) {
    vertex_t v = current_frontier[index];
    local_index_t vertex_index = get_local_index_gpu(v, local_graph);
    unsigned int old_level = local_visited[vertex_index];
    bool success = false;
    while (local_level < old_level) {
      success = (atomicCAS(&local_visited[vertex_index], old_level,
                           local_level) == old_level);
      old_level = local_visited[vertex_index];
    }

    if (success) {
      vertex_t *neighbors = NULL;
      uint64_t neighbor_count = 0;
      get_neighbors_gpu(local_graph, v, &neighbors, &neighbor_count);
      if (neighbor_count) {
        unsigned int frontier_index =
            atomicAdd(local_frontier_count, (unsigned int)neighbor_count);
        if (frontier_index < GPULISTLEN) {
          for (uint64_t i = 0; i < neighbor_count; ++i) {
            local[frontier_index + i] = neighbors[i];
          }
        }
      }
    }
  }
}

void cpu_bfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t index = arts_get_total_gpus();  // The current gpu we are on
  unsigned int local_level = (unsigned int)paramv[0];
  unsigned int *local_visited = (unsigned int *)depv[0].ptr;
  unsigned int **addr =
      (unsigned int **)depv[1].ptr;  // This is the dev_ptr_raw -> tells us
                                     // where next frontier is on device
  unsigned int *local =
      addr[index];  // We need the one corresponding to our gpu
  unsigned int *local_frontier_count = &local[GPULISTLEN];

  unsigned int current_frontier_size = *((unsigned int *)depv[2].ptr);
  unsigned int *current_frontier = ((unsigned int *)depv[2].ptr) + 1;
  csr_graph_t *local_graph = (csr_graph_t *)depv[3].ptr;

  for (unsigned int idx = 0; idx < current_frontier_size; idx++) {
    vertex_t v = current_frontier[idx];
    local_index_t vertex_index = get_local_index_csr(v, local_graph);
    unsigned int old_level = local_visited[vertex_index];
    bool success = false;
    while (local_level < old_level) {
      success = (arts_atomic_cswap(&local_visited[vertex_index], old_level,
                                   local_level) == old_level);
      old_level = local_visited[vertex_index];
    }

    if (success) {
      vertex_t *neighbors = NULL;
      uint64_t neighbor_count = 0;
      get_neighbors(local_graph, v, &neighbors, &neighbor_count);
      if (neighbor_count) {
        unsigned int frontier_index = arts_atomic_fetch_add(
            local_frontier_count, (unsigned int)neighbor_count);
        if (frontier_index < GPULISTLEN) {
          for (uint64_t i = 0; i < neighbor_count; ++i) {
            local[frontier_index + i] = neighbors[i];
          }
        }
      }
    }
  }
}

// LC will sync all the version coming into this edt and then we will start the
// next epoch
void do_partition_sync(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_printf("Just Synced Partitions! %lu\n", paramv[0]);
  arts_signal_edt((arts_guid_t)paramv[1], (uint32_t)-1, NULL_GUID,
                  DB_MODE_EW);
}

// There is only one of these per level.  It is signaled by the epoch containing
// the Bfs'es
void launch_sort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  uint64_t local_level = paramv[0];
  // uint64_t edtsRan = depv[0].guid;
  arts_printf("%s Level: %lu Edts Ran: %lu\n", __func__, local_level, edtsRan);

  // This is tricky.  We need to create the epoch for the next round since
  // thrust_sort will create the next rounds' Bfs'es.  In order to create the
  // epoch, we need the next round's launch_sort.
  uint64_t next_level = local_level + 1;
  arts_hint_t hint_2 = {arts_get_current_node(), 0};
  arts_guid_t next_launch_sort_guid =
      arts_edt_create(launch_sort, 1, &next_level, 1, &hint_2);
  arts_initialize_and_start_epoch(next_launch_sort_guid, 0);

  // While we are at it, lets create the next sync point, launch_bfs.
  arts_guid_t next_launch_bfs_guid =
      arts_guid_reserve(ARTS_EDT, arts_get_current_node());
  uint32_t next_launch_bfs_depc =
      arts_get_total_nodes() * (arts_get_total_gpus() + 1);

  // Lasly, we will launch a sort for every gpu in the system.
  // We need the nextBfsEpoch and the nextLaunchBfsGuids to kick off
  // launch_bfs...
  dim3 threads(1, 1, 1);
  dim3 grid(1, 1, 1);
  uint64_t args[] = {local_level, (uint64_t)next_launch_bfs_guid};
  for (unsigned int j = 0; j < arts_get_total_nodes(); j++) {
    for (uint64_t i = 0; i < arts_get_total_gpus(); i++) {
      arts_guid_t thrust_guid = arts_edt_create_gpu_lib_direct(
          thrust_sort, j, i, 2, args, 0, grid, threads);
      (void)thrust_guid;
    }
    // Launch CPU sort here!
    arts_hint_t hint_3 = {j, 0};
    arts_guid_t sort_guid = arts_edt_create(cpu_sort, 2, args, 0, &hint_3);
    (void)sort_guid;
  }

  // Double buffering!!!
  reset_buffer(local_level);

  // This uses the LC memory model if turned on
  if (DO_SYNC(local_level)) {
    for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
      uint64_t sync_args[] = {local_level, (uint64_t)next_launch_bfs_guid};
      arts_hint_t hint_4 = {i, 0};
      arts_guid_t edt_guid = arts_edt_create(do_partition_sync, 2, sync_args,
                                             part_count[i], &hint_4);
      unsigned int slot = 0;
      for (unsigned int j = 0; j < PARTS; j++) {
        if (i == arts_guid_get_rank(visited_guid[j])) {
          arts_lc_sync(edt_guid, slot++, visited_guid[j]);
        }
      }
    }
    next_launch_bfs_depc += arts_get_total_nodes();
  }
  arts_edt_create_with_guid(launch_bfs, next_launch_bfs_guid, 1, &next_level,
                            next_launch_bfs_depc);
}

void cpu_sort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  uint64_t local_level =
      paramv[0];  // This can be the end if the frontier is empty
  unsigned int *local = get_local_buffer(
      arts_get_total_gpus(),
      local_level);  // We need the one corresponding to our gpu
  unsigned int *local_frontier_count = &local[GPULISTLEN];

  // arts_printf("%s Level: %lu\n", __func__, local_level);
  arts_guid_t next_launch_bfs_guid =
      (arts_guid_t)paramv[1];  // This is the next sync point.
  arts_guid_t edt_guids_to_launch_bfs_guid =
      NULL_GUID;  // Where we will put a copy of all the new edts to start...

  // Get frontier count
  unsigned int new_frontier_count = *local_frontier_count;
  if (new_frontier_count <=
      GPULISTLEN)  // If it was bigger than the search frontier, we need to quit
  {
    // Sort the frontier
    std::sort(local, local + new_frontier_count);  // Do the sorting

    // Remove duplicates
    new_frontier_count =
        (unsigned int)(std::unique(local, local + new_frontier_count) - local);

    // Reset frontier
    *local_frontier_count = 0;

    // Get the boundery of each partition
    unsigned int upper_index_per_bound[PARTS];
    for (unsigned int i = 0; i < PARTS; i++) {
      upper_index_per_bound[i] =
          (unsigned int)(std::upper_bound(local, local + new_frontier_count,
                                          bounds[i]) -
                         local);
    }

    // Get the size of each partition
    unsigned int size_per_bound[PARTS];
    size_per_bound[0] = upper_index_per_bound[0];
    arts_printf("Upper: %u Size: %u\n", bounds[0], size_per_bound[0]);
    for (unsigned int i = 1; i < PARTS; i++) {
      size_per_bound[i] =
          upper_index_per_bound[i] - upper_index_per_bound[i - 1];
      arts_printf("Upper: %u Size: %u\n", bounds[i], size_per_bound[i]);
    }

    // TODO: Clear old dbs (previous frontiers)...
    arts_guid_t *edt_guids_to_launch_bfs =
        NULL;  // This will hold the new edt guids to launch
    edt_guids_to_launch_bfs_guid = arts_db_create(
        (void **)&edt_guids_to_launch_bfs, sizeof(arts_guid_t) * PARTS, NULL);

    uint64_t next_level = local_level + 1;
    unsigned int temp_index = 0;
    for (unsigned int i = 0; i < PARTS; i++) {
      if (size_per_bound[i]) {
        unsigned int *new_search_frontier =
            NULL;  // This will hold a tile of the new frontier
        arts_guid_t new_search_frontier_guid =
            arts_guid_reserve(ARTS_DB_GPU, 0);
        new_search_frontier = (unsigned int *)arts_db_create_with_guid(
            new_search_frontier_guid,
            sizeof(unsigned int) * (size_per_bound[i] + 1), NULL);
        *new_search_frontier = size_per_bound[i];

        // Copy the data from the gpu to the host
        memcpy((void *)(new_search_frontier + 1), (void *)(local + temp_index),
               sizeof(unsigned int) * size_per_bound[i]);
        temp_index += size_per_bound[i];

        // Create the new edt for each bfs
        unsigned int rank =
            arts_guid_get_rank(get_guid_for_partition_distr(distribution, i));
        if (size_per_bound[i] >= GPU_THRESHOLD)  // Create GPU EDT
        {
          dim3 threads(SMTILE, 1, 1);
          dim3 grid((size_per_bound[i] + SMTILE - 1) / SMTILE, 1,
                    1);  // Ceiling
          arts_printf("GPU PART: %u SMTILE: %u grid: %u\n", i, SMTILE,
                      (size_per_bound[i] + SMTILE - 1) / SMTILE);
          edt_guids_to_launch_bfs[i] =
              arts_edt_create_gpu(gpu_bfs, rank, 1, &next_level, 4, grid,
                                  threads, NULL_GUID, 0, NULL_GUID);

        } else  // Create CPU EDT
        {
          arts_printf("CPU PART: %u\n", i);
          arts_hint_t hint_5 = {rank, 0};
          edt_guids_to_launch_bfs[i] =
              arts_edt_create(cpu_bfs, 1, &next_level, 4, &hint_5);
        }

        arts_signal_edt(edt_guids_to_launch_bfs[i], 0, visited_guid[i],
                        DB_MODE_EW);
        if (!DO_SYNC(local_level)) {
          arts_signal_edt(edt_guids_to_launch_bfs[i], 1,
                          get_buffer_guid(rank, next_level), DB_MODE_EW);
        }
        arts_signal_edt(edt_guids_to_launch_bfs[i], 2, new_search_frontier_guid,
                        DB_MODE_EW);
        arts_signal_edt(edt_guids_to_launch_bfs[i], 3,
                        get_guid_for_partition_distr(distribution, i),
                        DB_MODE_EW);
        add_to_list(size_per_bound[i], arts_get_total_gpus());
      } else {
        edt_guids_to_launch_bfs[i] = NULL_GUID;
      }
    }
  }
  arts_signal_edt(next_launch_bfs_guid,
                  (arts_get_current_node() * (arts_get_total_gpus() + 1)) +
                      arts_get_total_gpus(),
                  edt_guids_to_launch_bfs_guid, DB_MODE_EW);
}

void thrust_sort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  uint64_t local_level =
      paramv[0];  // This can be the end if the frontier is empty
  unsigned int *raw_ptr = get_local_buffer(
      arts_get_gpu_id(),
      local_level);  // We need the one corresponding to our gpu

  // arts_printf("%s Level: %lu Gpu: %d\n", __func__, local_level,
  // arts_get_gpu_id());
  arts_guid_t next_launch_bfs_guid =
      (arts_guid_t)paramv[1];  // This is the next sync point.
  arts_guid_t edt_guids_to_launch_bfs_guid =
      NULL_GUID;  // Where we will put a copy of all the new edts to start...
  // unsigned int * raw_ptr = dev_ptr_raw[arts_get_gpu_id()]; //The
  // corresponding dev pointer (frontier) to our gpu

  // Get frontier count
  thrust::device_ptr<unsigned int> dev_counter_ptr(raw_ptr + GPULISTLEN);
  unsigned int new_frontier_count = *(dev_counter_ptr);
  if (new_frontier_count <=
      GPULISTLEN)  // If it was bigger than the search frontier, we need to quit
  {
    // Sort the frontier
    thrust::device_ptr<unsigned int> dev_ptr(raw_ptr);
    thrust::sort(dev_ptr, dev_ptr + new_frontier_count);  // Do the sorting
    TURNON(print_device_list(dev_ptr, new_frontier_count));

    // Remove duplicates
    new_frontier_count =
        (unsigned int)(thrust::unique(thrust::device, dev_ptr,
                                      dev_ptr + new_frontier_count) -
                       dev_ptr);
    TURNON(print_device_list(dev_ptr, new_frontier_count));

    // Reset frontier
    *(dev_counter_ptr) = 0;

    // Get the boundery of each partition
    unsigned int upper_index_per_bound[PARTS];
    for (unsigned int i = 0; i < PARTS; i++) {
      upper_index_per_bound[i] =
          (unsigned int)(thrust::upper_bound(thrust::device, dev_ptr,
                                             dev_ptr + new_frontier_count,
                                             bounds[i]) -
                         dev_ptr);
    }

    // Get the size of each partition
    unsigned int size_per_bound[PARTS];
    size_per_bound[0] = upper_index_per_bound[0];
    arts_printf("Upper: %u Size: %u\n", bounds[0], size_per_bound[0]);
    for (unsigned int i = 1; i < PARTS; i++) {
      size_per_bound[i] =
          upper_index_per_bound[i] - upper_index_per_bound[i - 1];
      arts_printf("Upper: %u Size: %u\n", bounds[i], size_per_bound[i]);
    }

    // TODO: Clear old dbs (previous frontiers)...
    arts_guid_t *edt_guids_to_launch_bfs =
        NULL;  // This will hold the new edt guids to launch
    edt_guids_to_launch_bfs_guid = arts_db_create(
        (void **)&edt_guids_to_launch_bfs, sizeof(arts_guid_t) * PARTS, NULL);

    uint64_t next_level = local_level + 1;
    unsigned int temp_index = 0;
    for (unsigned int i = 0; i < PARTS; i++) {
      if (size_per_bound[i]) {
        unsigned int *new_search_frontier =
            NULL;  // This will hold a tile of the new frontier
        arts_guid_t new_search_frontier_guid =
            arts_guid_reserve(ARTS_DB_GPU, 0);
        new_search_frontier = (unsigned int *)arts_db_create_with_guid(
            new_search_frontier_guid,
            sizeof(unsigned int) * (size_per_bound[i] + 1), NULL);
        *new_search_frontier = size_per_bound[i];

        // Copy the data from the gpu to the host
        arts_put_in_db_from_gpu(thrust::raw_pointer_cast(dev_ptr) + temp_index,
                                new_search_frontier_guid, sizeof(unsigned int),
                                sizeof(unsigned int) * size_per_bound[i],
                                false);
        temp_index += size_per_bound[i];

        // Create the new edt for each bfs
        unsigned int rank =
            arts_guid_get_rank(get_guid_for_partition_distr(distribution, i));
        if (size_per_bound[i] >= GPU_THRESHOLD)  // Create GPU EDT
        {
          dim3 threads(SMTILE, 1, 1);
          dim3 grid((size_per_bound[i] + SMTILE - 1) / SMTILE, 1,
                    1);  // Ceiling
          arts_printf("GPU PART: %u SMTILE: %u grid: %u\n", i, SMTILE,
                      (size_per_bound[i] + SMTILE - 1) / SMTILE);
          edt_guids_to_launch_bfs[i] =
              arts_edt_create_gpu(gpu_bfs, rank, 1, &next_level, 4, grid,
                                  threads, NULL_GUID, 0, NULL_GUID);

        } else  // Create CPU EDT
        {
          arts_printf("CPU PART: %u\n", i);
          arts_hint_t hint_6 = {rank, 0};
          edt_guids_to_launch_bfs[i] =
              arts_edt_create(cpu_bfs, 1, &next_level, 4, &hint_6);
        }

        arts_signal_edt(edt_guids_to_launch_bfs[i], 0, visited_guid[i],
                        DB_MODE_EW);
        if (!DO_SYNC(local_level)) {
          arts_signal_edt(edt_guids_to_launch_bfs[i], 1,
                          get_buffer_guid(rank, next_level), DB_MODE_EW);
        }
        arts_signal_edt(edt_guids_to_launch_bfs[i], 2, new_search_frontier_guid,
                        DB_MODE_EW);
        arts_signal_edt(edt_guids_to_launch_bfs[i], 3,
                        get_guid_for_partition_distr(distribution, i),
                        DB_MODE_EW);
        add_to_list(size_per_bound[i], arts_get_gpu_id());
      } else {
        edt_guids_to_launch_bfs[i] = NULL_GUID;
      }
    }
  }
  arts_signal_edt(next_launch_bfs_guid,
                  (arts_get_current_node() * (arts_get_total_gpus() + 1)) +
                      arts_get_gpu_id(),
                  edt_guids_to_launch_bfs_guid, DB_MODE_EW);
}

// This needs nodes * gpus signals.  Each db has PARTS guids to signal.
void launch_bfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t total_new_bfs = 0;
  uint64_t local_level = paramv[0];
  uint64_t last_level = local_level - 1;
  arts_printf("%s Level: %lu\n", __func__, local_level);

  if (local_level < MAXLEVEL) {
    // from each gpu, we get a bunch of bfs-es that need to be spawned
    unsigned int num_potential_bfs_dbs =
        arts_get_total_nodes() * (arts_get_total_gpus() + 1);
    for (unsigned int i = 0; i < num_potential_bfs_dbs; i++) {
      if (depv[i].guid) {
        arts_guid_t *guid_to_signal = (arts_guid_t *)depv[i].ptr;
        for (unsigned int j = 0; j < PARTS; j++) {
          if (guid_to_signal[j]) {
            if (DO_SYNC(last_level)) {
              unsigned int rank = arts_guid_get_rank(guid_to_signal[j]);
              arts_signal_edt(guid_to_signal[j], 1,
                              get_buffer_guid(rank, local_level), DB_MODE_EW);
            }
            total_new_bfs++;
          }
        }
      } else  // This means one of the frontiers was overflown
      {
        arts_printf("Next Search Frontier Overflow!\n");
        arts_printf("Failed Level: %lu\n", last_level);
        arts_printf("Shutting down...\n");
        arts_shutdown();
        return;
      }
    }
  }

  if (!total_new_bfs || local_level == MAXLEVEL) {
    uint64_t stop = arts_get_time_stamp();
    arts_printf("Time: %lu\n", stop - start);
    arts_printf("Level: %lu\n", local_level);
    arts_printf("Shutting down...\n");
    arts_shutdown();
  }
}

/********************************************************************************************/

void init_node(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  (void)argc;
  unsigned int node_id = arts_get_current_node();

  char *file_name = argv[1];
  unsigned int num_verts = 0;
  unsigned int num_edges = 0;
  get_properties(file_name, &num_verts, &num_edges);

  // Create graph partitions
  graph = (csr_graph_t *)calloc(PARTS, sizeof(csr_graph_t));
  distribution =
      init_block_distribution_block(num_verts, num_edges, PARTS, ARTS_DB_GPU);
  load_graph_no_weight_csr(file_name, distribution, true, false);

  // Find the boundaries for sorting
  for (unsigned int i = 0; i < PARTS; i++) {
    bounds[i] = partition_end_distr(i, distribution);
    arts_printf("Bounds[%u]: %lu guid: %lu\n", i, bounds[i],
                distribution->graphGuid[i]);
  }

  // Count the number of partitions per node for later...
  part_count =
      (unsigned int *)calloc(arts_get_total_nodes(), sizeof(unsigned int));
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    part_count[i] = 0;
  }

  // Create visited array per partition
  visited_guid = (arts_guid_t *)calloc(PARTS, sizeof(arts_guid_t));
  visited = (unsigned int **)calloc(PARTS, sizeof(unsigned int *));
  for (unsigned int i = 0; i < PARTS; i++) {
    unsigned int num_elements = get_block_size_for_partition(i, distribution);
    unsigned int size = sizeof(unsigned int) * num_elements;
    unsigned int rank =
        arts_guid_get_rank(get_guid_for_partition_distr(distribution, i));
    visited_guid[i] = arts_guid_reserve(DB_WRITE_TYPE, rank);
    part_count[rank]++;
    if (rank == node_id) {
      visited[i] =
          (unsigned int *)arts_db_create_with_guid(visited_guid[i], size, NULL);
      for (unsigned int j = 0; j < num_elements; j++) {
        visited[i][j] = UINT32_MAX;
      }
    }
  }

  create_buffers_on_cpu(sizeof(unsigned int) * (GPULISTLEN + 1));

  // Inits some data recording
  init_list_record();

  create_buffer_db();
}

extern "C" void arts_init_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream, int argc,
                                  const char *argv) {
  (void)node_id;
  (void)stream;
  (void)argc;
  (void)argv;
  create_buffers_on_gpu(dev_id, sizeof(unsigned int) * (GPULISTLEN + 1));
}

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;

  arts_guid_t init_epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    arts_hint_t hint_7 = {i, 0};
    arts_edt_create_with_epoch(init_node, paramc, paramv, 0, init_epoch_guid,
                               &hint_7);
  }
  arts_wait_on_handle(init_epoch_guid);

  // Spawn a task on the rank containing the source
  vertex_t source = ROOT;
  unsigned int owner_rank = get_owner_distr(source, distribution);
  uint64_t args_fr_rnd_one[] = {source};
  arts_hint_t hint_8 = {owner_rank, 0};
  arts_edt_create(create_first_round, 1, args_fr_rnd_one, 0, &hint_8);
}

extern "C" void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream) {
  (void)node_id;
  (void)stream;
  free_buffers_on_gpu(dev_id);
  write_bins_to_file(dev_id);
}

int main(int argc, char **argv) {
  DASHDASHFILE(argc, argv)
  arts_rt(argc, argv);
  return 0;
}
