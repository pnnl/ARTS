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

uint64_t start = 0; // Timer

unsigned int bounds[PARTS];      // This is the boundaries that make up each
                                 // partition
arts_block_dist_t *distribution; // The graph distribution
csr_graph_t *graph;              // Partitions of the graph
unsigned int **visited; // This is the resulting parent list for each partition
arts_guid_t
    *visitedGuid; // This is the guid for each partition of the parent list
unsigned int *partCount;

void createFirstRound(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]);
__global__ void gpuBfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]);
void cpuBfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]);
void launchSort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]);
void cpuSort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]);
void thrustSort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]);
void launchBfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]);

void printDeviceList(thrust::device_ptr<unsigned int> devPtr,
                     unsigned int size) {
  ARTS_PRINTF("FRONTIER SIZE: %u\n", size);
  for (unsigned int i = 0; i < size; i++) {
    unsigned int temp = *(devPtr + i);
    ARTS_PRINTF("%u, ", temp);
  }
  ARTS_PRINTF("\n");
}

void printResult() {
  for (unsigned int i = 0; i < PARTS; i++) {
    unsigned int size =
        sizeof(unsigned int) * get_block_size_for_partition(i, distribution);
    ARTS_PRINTF("%u: %u\n", i, size);
    for (unsigned int j = 0; j < size; j++)
      ARTS_PRINTF("%u, ", visited[i][j]);
    ARTS_PRINTF("\n");
  }
}

void createFirstRound(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  ARTS_PRINTF("%s\n", __func__);
  start = arts_get_time_stamp();

  // We are on the rank of the source
  uint64_t src = paramv[0];
  uint64_t nextLevel = 0;

  // Create the first search frontier!
  unsigned int *firstSearchFrontier = NULL;
  arts_guid_t firstSearchFrontierGuid =
      arts_db_create((void **)&firstSearchFrontier, 2 * sizeof(unsigned int),
                   ARTS_DB_GPU_READ);
  firstSearchFrontier[0] = 1;   // size of the frontier
  firstSearchFrontier[1] = src; // root
  ARTS_PRINTF("ROOT: %u GRAPH GUID: %lu VISITED GUID: %lu\n", firstSearchFrontier[1],
         get_guid_for_vertex_distr(firstSearchFrontier[1], distribution),
         visitedGuid[get_owner_distr(firstSearchFrontier[1], distribution)]);

  // Create the first epoch
  arts_guid_t launchSortGuid =
      arts_edt_create(launchSort, arts_get_current_node(), 1, &nextLevel, 1);
  arts_initialize_and_start_epoch(launchSortGuid, 0);

  // Launching the first bfs
  arts_guid_t graphGuid =
      get_guid_for_vertex_distr(firstSearchFrontier[1], distribution);
  arts_guid_t visitGuid =
      visitedGuid[get_owner_distr(firstSearchFrontier[1], distribution)];
  arts_guid_t bfsGuid = NULL_GUID;
  if (firstSearchFrontier[0] > GPU_THRESHOLD) {
    dim3 threads(1, 1, 1);
    dim3 grid(1, 1, 1);
    bfsGuid = arts_edt_create_gpu(gpuBfs, arts_get_current_node(), 1, &nextLevel, 4,
                               grid, threads, NULL_GUID, 0, NULL_GUID);
    ARTS_PRINTF("LAUNCHING GPU\n");
  } else {
    bfsGuid = arts_edt_create(cpuBfs, arts_get_current_node(), 1, &nextLevel, 4);
    ARTS_PRINTF("LAUNCHING CPU\n");
  }
  arts_signal_edt(bfsGuid, 0, visitGuid);
  // arts_signal_edt(bfsGuid, 1,
  // nextSearchFrontierAddrGuid[arts_get_current_node()]);
  arts_signal_edt(bfsGuid, 1, get_buffer_guid(arts_get_current_node(), nextLevel));
  arts_signal_edt(bfsGuid, 2, firstSearchFrontierGuid);
  arts_signal_edt(bfsGuid, 3, graphGuid);
}

__global__ void gpuBfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  uint64_t gpu_id = GET_GPU_INDEX(); // The current gpu we are on
  unsigned int localLevel = (unsigned int)paramv[0];
  unsigned int *localVisited = (unsigned int *)depv[0].ptr;
  unsigned int **addr =
      (unsigned int **)depv[1].ptr;  // This is the devPtrRaw -> tells us where
                                     // next frontier is on device
  unsigned int *local = addr[gpu_id]; // We need the one corresponding to our gpu
  unsigned int *localFrontierCount = &local[GPULISTLEN];

  unsigned int currentFrontierSize = *((unsigned int *)depv[2].ptr);
  unsigned int *currentFrontier = ((unsigned int *)depv[2].ptr) + 1;
  csr_graph_t *localGraph = (csr_graph_t *)depv[3].ptr;

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < currentFrontierSize) {
    vertex_t v = currentFrontier[index];
    local_index_t vertexIndex = getLocalIndexGpu(v, localGraph);
    unsigned int oldLevel = localVisited[vertexIndex];
    bool success = false;
    while (localLevel < oldLevel) {
      success = (atomicCAS(&localVisited[vertexIndex], oldLevel, localLevel) ==
                 oldLevel);
      oldLevel = localVisited[vertexIndex];
    }

    if (success) {
      vertex_t *neighbors = NULL;
      uint64_t neighbor_count = 0;
      getNeighborsGpu(localGraph, v, &neighbors, &neighbor_count);
      if (neighbor_count) {
        unsigned int frontierIndex =
            atomicAdd(localFrontierCount, (unsigned int)neighbor_count);
        if (frontierIndex < GPULISTLEN) {
          for (uint64_t i = 0; i < neighbor_count; ++i) {
            local[frontierIndex + i] = neighbors[i];
          }
        }
      }
    }
  }
}

void cpuBfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  uint64_t index = arts_get_total_gpus(); // The current gpu we are on
  unsigned int localLevel = (unsigned int)paramv[0];
  unsigned int *localVisited = (unsigned int *)depv[0].ptr;
  unsigned int **addr =
      (unsigned int **)depv[1].ptr;  // This is the devPtrRaw -> tells us where
                                     // next frontier is on device
  unsigned int *local = addr[index]; // We need the one corresponding to our gpu
  unsigned int *localFrontierCount = &local[GPULISTLEN];

  unsigned int currentFrontierSize = *((unsigned int *)depv[2].ptr);
  unsigned int *currentFrontier = ((unsigned int *)depv[2].ptr) + 1;
  csr_graph_t *localGraph = (csr_graph_t *)depv[3].ptr;

  for (unsigned int index = 0; index < currentFrontierSize; index++) {
    vertex_t v = currentFrontier[index];
    local_index_t vertexIndex = get_local_index_csr(v, localGraph);
    unsigned int oldLevel = localVisited[vertexIndex];
    bool success = false;
    while (localLevel < oldLevel) {
      success = (arts_atomic_cswap(&localVisited[vertexIndex], oldLevel,
                                 localLevel) == oldLevel);
      oldLevel = localVisited[vertexIndex];
    }

    if (success) {
      vertex_t *neighbors = NULL;
      uint64_t neighbor_count = 0;
      get_neighbors(localGraph, v, &neighbors, &neighbor_count);
      if (neighbor_count) {
        unsigned int frontierIndex =
            arts_atomic_fetch_add(localFrontierCount, (unsigned int)neighbor_count);
        if (frontierIndex < GPULISTLEN) {
          for (uint64_t i = 0; i < neighbor_count; ++i) {
            local[frontierIndex + i] = neighbors[i];
          }
        }
      }
    }
  }
}

// LC will sync all the version coming into this edt and then we will start the
// next epoch
void doPartionSync(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  ARTS_PRINTF("Just Synced Partitions! %lu\n", paramv[0]);
  arts_signal_edt(paramv[1], (uint32_t)-1, NULL_GUID);
}

// There is only one of these per level.  It is signaled by the epoch containing
// the Bfs'es
void launchSort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  uint64_t localLevel = paramv[0];
  // uint64_t edtsRan = depv[0].guid;
  ARTS_PRINTF("%s Level: %lu Edts Ran: %lu\n", __func__, localLevel, edtsRan);

  // This is tricky.  We need to create the epoch for the next round since
  // thrustSort will create the next rounds' Bfs'es.  In order to create the
  // epoch, we need the next round's launchSort.
  uint64_t nextLevel = localLevel + 1;
  arts_guid_t nextLaunchSortGuid =
      arts_edt_create(launchSort, arts_get_current_node(), 1, &nextLevel, 1);
  arts_initialize_and_start_epoch(nextLaunchSortGuid, 0);

  // While we are at it, lets create the next sync point, launchBfs.
  arts_guid_t nextLaunchBfsGuid =
      arts_reserve_guid_route(ARTS_EDT, arts_get_current_node());
  uint32_t nextLaunchBfsDepc = arts_get_total_nodes() * (arts_get_total_gpus() + 1);

  // Lasly, we will launch a sort for every gpu in the system.
  // We need the nextBfsEpoch and the nextLaunchBfsGuids to kick off
  // launchBfs...
  dim3 threads(1, 1, 1);
  dim3 grid(1, 1, 1);
  uint64_t args[] = {localLevel, (uint64_t)nextLaunchBfsGuid};
  for (unsigned int j = 0; j < arts_get_total_nodes(); j++) {
    for (uint64_t i = 0; i < arts_get_total_gpus(); i++) {
      arts_guid_t thrustGuid = arts_edt_create_gpu_lib_direct(thrustSort, j, i, 2,
                                                        args, 0, grid, threads);
    }
    // Launch CPU sort here!
    arts_guid_t sortGuid = arts_edt_create(cpuSort, j, 2, args, 0);
  }

  // Double buffering!!!
  reset_buffer(localLevel);

  // This uses the LC memory model if turned on
  if (DO_SYNC(localLevel)) {
    for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
      uint64_t syncArgs[] = {localLevel, (uint64_t)nextLaunchBfsGuid};
      arts_guid_t edt_guid =
          arts_edt_create(doPartionSync, i, 2, syncArgs, partCount[i]);
      unsigned int slot = 0;
      for (unsigned int j = 0; j < PARTS; j++) {
        if (i == arts_guid_get_rank(visitedGuid[j])) {
          arts_lc_sync(edt_guid, slot++, visitedGuid[j]);
        }
      }
    }
    nextLaunchBfsDepc += arts_get_total_nodes();
  }
  arts_edt_create_with_guid(launchBfs, nextLaunchBfsGuid, 1, &nextLevel,
                        nextLaunchBfsDepc);
}

void cpuSort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  uint64_t localLevel =
      paramv[0]; // This can be the end if the frontier is empty
  unsigned int *local =
      get_local_buffer(arts_get_total_gpus(),
                     localLevel); // We need the one corresponding to our gpu
  unsigned int *localFrontierCount = &local[GPULISTLEN];

  // ARTS_PRINTF("%s Level: %lu\n", __func__, localLevel);
  arts_guid_t nextLaunchBfsGuid = paramv[1]; // This is the next sync point.
  arts_guid_t edtGuidsToLaunchBfsGuid =
      NULL_GUID; // Where we will put a copy of all the new edts to start...

  // Get frontier count
  unsigned int newFrontierCount = *localFrontierCount;
  if (newFrontierCount <=
      GPULISTLEN) // If it was bigger than the search frontier, we need to quit
  {
    // Sort the frontier
    std::sort(local, local + newFrontierCount); // Do the sorting

    // Remove duplicates
    newFrontierCount = std::unique(local, local + newFrontierCount) - local;

    // Reset frontier
    *localFrontierCount = 0;

    // Get the boundery of each partition
    unsigned int upperIndexPerBound[PARTS];
    for (unsigned int i = 0; i < PARTS; i++)
      upperIndexPerBound[i] =
          std::upper_bound(local, local + newFrontierCount, bounds[i]) - local;

    // Get the size of each partition
    unsigned int sizePerBound[PARTS];
    sizePerBound[0] = upperIndexPerBound[0];
    ARTS_PRINTF("Upper: %u Size: %u\n", bounds[0], sizePerBound[0]);
    for (unsigned int i = 1; i < PARTS; i++) {
      sizePerBound[i] = upperIndexPerBound[i] - upperIndexPerBound[i - 1];
      ARTS_PRINTF("Upper: %u Size: %u\n", bounds[i], sizePerBound[i]);
    }

    // TODO: Clear old dbs (previous frontiers)...
    arts_guid_t *edtGuidsToLaunchBfs =
        NULL; // This will hold the new edt guids to launch
    edtGuidsToLaunchBfsGuid =
        arts_db_create((void **)&edtGuidsToLaunchBfs, sizeof(arts_guid_t) * PARTS,
                     ARTS_DB_READ);

    uint64_t nextLevel = localLevel + 1;
    unsigned tempIndex = 0;
    for (unsigned int i = 0; i < PARTS; i++) {
      if (sizePerBound[i]) {
        unsigned int *newSearchFrontier =
            NULL; // This will hold a tile of the new frontier
        arts_guid_t newSearchFrontierGuid = arts_db_create(
            (void **)&newSearchFrontier,
            sizeof(unsigned int) * (sizePerBound[i] + 1), ARTS_DB_GPU_READ);
        *newSearchFrontier = sizePerBound[i];

        // Copy the data from the gpu to the host
        memcpy((void *)(newSearchFrontier + 1), (void *)(local + tempIndex),
               sizeof(unsigned int) * sizePerBound[i]);
        tempIndex += sizePerBound[i];

        // Create the new edt for each bfs
        unsigned int rank =
            arts_guid_get_rank(get_guid_for_partition_distr(distribution, i));
        if (sizePerBound[i] >= GPU_THRESHOLD) // Create GPU EDT
        {
          dim3 threads(SMTILE, 1, 1);
          dim3 grid((sizePerBound[i] + SMTILE - 1) / SMTILE, 1, 1); // Ceiling
          ARTS_PRINTF("GPU PART: %u SMTILE: %u grid: %u\n", i, SMTILE,
                 (sizePerBound[i] + SMTILE - 1) / SMTILE);
          edtGuidsToLaunchBfs[i] =
              arts_edt_create_gpu(gpuBfs, rank, 1, &nextLevel, 4, grid, threads,
                               NULL_GUID, 0, NULL_GUID);

        } else // Create CPU EDT
        {
          ARTS_PRINTF("CPU PART: %u\n", i);
          edtGuidsToLaunchBfs[i] =
              arts_edt_create(cpuBfs, rank, 1, &nextLevel, 4);
        }

        arts_signal_edt(edtGuidsToLaunchBfs[i], 0, visitedGuid[i]);
        if (!DO_SYNC(localLevel)) {
          arts_signal_edt(edtGuidsToLaunchBfs[i], 1,
                        get_buffer_guid(rank, nextLevel));
        }
        arts_signal_edt(edtGuidsToLaunchBfs[i], 2, newSearchFrontierGuid);
        arts_signal_edt(edtGuidsToLaunchBfs[i], 3,
                      get_guid_for_partition_distr(distribution, i));
        add_to_list(sizePerBound[i], arts_get_total_gpus());
      } else
        edtGuidsToLaunchBfs[i] = NULL_GUID;
    }
  }
  arts_signal_edt(nextLaunchBfsGuid,
                arts_get_current_node() * (arts_get_total_gpus() + 1) +
                    arts_get_total_gpus(),
                edtGuidsToLaunchBfsGuid);
}

void thrustSort(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  uint64_t localLevel =
      paramv[0]; // This can be the end if the frontier is empty
  unsigned int *rawPtr = get_local_buffer(
      arts_get_gpu_id(), localLevel); // We need the one corresponding to our gpu

  // ARTS_PRINTF("%s Level: %lu Gpu: %d\n", __func__, localLevel, arts_get_gpu_id());
  arts_guid_t nextLaunchBfsGuid = paramv[1]; // This is the next sync point.
  arts_guid_t edtGuidsToLaunchBfsGuid =
      NULL_GUID; // Where we will put a copy of all the new edts to start...
  // unsigned int * rawPtr = devPtrRaw[arts_get_gpu_id()]; //The corresponding dev
  // pointer (frontier) to our gpu

  // Get frontier count
  thrust::device_ptr<unsigned int> devCounterPtr(rawPtr + GPULISTLEN);
  unsigned int newFrontierCount = *(devCounterPtr);
  if (newFrontierCount <=
      GPULISTLEN) // If it was bigger than the search frontier, we need to quit
  {
    // Sort the frontier
    thrust::device_ptr<unsigned int> devPtr(rawPtr);
    thrust::sort(devPtr, devPtr + newFrontierCount); // Do the sorting
    TURNON(printDeviceList(devPtr, newFrontierCount));

    // Remove duplicates
    newFrontierCount =
        thrust::unique(thrust::device, devPtr, devPtr + newFrontierCount) -
        devPtr;
    TURNON(printDeviceList(devPtr, newFrontierCount));

    // Reset frontier
    *(devCounterPtr) = 0;

    // Get the boundery of each partition
    unsigned int upperIndexPerBound[PARTS];
    for (unsigned int i = 0; i < PARTS; i++)
      upperIndexPerBound[i] =
          thrust::upper_bound(thrust::device, devPtr, devPtr + newFrontierCount,
                              bounds[i]) -
          devPtr;

    // Get the size of each partition
    unsigned int sizePerBound[PARTS];
    sizePerBound[0] = upperIndexPerBound[0];
    ARTS_PRINTF("Upper: %u Size: %u\n", bounds[0], sizePerBound[0]);
    for (unsigned int i = 1; i < PARTS; i++) {
      sizePerBound[i] = upperIndexPerBound[i] - upperIndexPerBound[i - 1];
      ARTS_PRINTF("Upper: %u Size: %u\n", bounds[i], sizePerBound[i]);
    }

    // TODO: Clear old dbs (previous frontiers)...
    arts_guid_t *edtGuidsToLaunchBfs =
        NULL; // This will hold the new edt guids to launch
    edtGuidsToLaunchBfsGuid =
        arts_db_create((void **)&edtGuidsToLaunchBfs, sizeof(arts_guid_t) * PARTS,
                     ARTS_DB_READ);

    uint64_t nextLevel = localLevel + 1;
    unsigned tempIndex = 0;
    for (unsigned int i = 0; i < PARTS; i++) {
      if (sizePerBound[i]) {
        unsigned int *newSearchFrontier =
            NULL; // This will hold a tile of the new frontier
        arts_guid_t newSearchFrontierGuid = arts_db_create(
            (void **)&newSearchFrontier,
            sizeof(unsigned int) * (sizePerBound[i] + 1), ARTS_DB_GPU_READ);
        *newSearchFrontier = sizePerBound[i];

        // Copy the data from the gpu to the host
        arts_put_in_db_from_gpu(thrust::raw_pointer_cast(devPtr) + tempIndex,
                           newSearchFrontierGuid, sizeof(unsigned int),
                           sizeof(unsigned int) * sizePerBound[i], false);
        tempIndex += sizePerBound[i];

        // Create the new edt for each bfs
        unsigned int rank =
            arts_guid_get_rank(get_guid_for_partition_distr(distribution, i));
        if (sizePerBound[i] >= GPU_THRESHOLD) // Create GPU EDT
        {
          dim3 threads(SMTILE, 1, 1);
          dim3 grid((sizePerBound[i] + SMTILE - 1) / SMTILE, 1, 1); // Ceiling
          ARTS_PRINTF("GPU PART: %u SMTILE: %u grid: %u\n", i, SMTILE,
                 (sizePerBound[i] + SMTILE - 1) / SMTILE);
          edtGuidsToLaunchBfs[i] =
              arts_edt_create_gpu(gpuBfs, rank, 1, &nextLevel, 4, grid, threads,
                               NULL_GUID, 0, NULL_GUID);

        } else // Create CPU EDT
        {
          ARTS_PRINTF("CPU PART: %u\n", i);
          edtGuidsToLaunchBfs[i] =
              arts_edt_create(cpuBfs, rank, 1, &nextLevel, 4);
        }

        arts_signal_edt(edtGuidsToLaunchBfs[i], 0, visitedGuid[i]);
        if (!DO_SYNC(localLevel)) {
          arts_signal_edt(edtGuidsToLaunchBfs[i], 1,
                        get_buffer_guid(rank, nextLevel));
        }
        arts_signal_edt(edtGuidsToLaunchBfs[i], 2, newSearchFrontierGuid);
        arts_signal_edt(edtGuidsToLaunchBfs[i], 3,
                      get_guid_for_partition_distr(distribution, i));
        add_to_list(sizePerBound[i], arts_get_gpu_id());
      } else
        edtGuidsToLaunchBfs[i] = NULL_GUID;
    }
  }
  arts_signal_edt(nextLaunchBfsGuid,
                arts_get_current_node() * (arts_get_total_gpus() + 1) +
                    arts_get_gpu_id(),
                edtGuidsToLaunchBfsGuid);
}

// This needs nodes * gpus signals.  Each db has PARTS guids to signal.
void launchBfs(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  uint64_t totalNewBfs = 0;
  uint64_t localLevel = paramv[0];
  uint64_t lastLevel = localLevel - 1;
  ARTS_PRINTF("%s Level: %lu\n", __func__, localLevel);

  if (localLevel < MAXLEVEL) {
    // from each gpu, we get a bunch of bfs-es that need to be spawned
    unsigned int numPotentialBfsDbs =
        arts_get_total_nodes() * (arts_get_total_gpus() + 1);
    for (unsigned int i = 0; i < numPotentialBfsDbs; i++) {
      if (depv[i].guid) {
        arts_guid_t *guidToSignal = (arts_guid_t *)depv[i].ptr;
        for (unsigned int j = 0; j < PARTS; j++) {
          if (guidToSignal[j]) {
            if (DO_SYNC(lastLevel)) {
              unsigned int rank = arts_guid_get_rank(guidToSignal[j]);
              arts_signal_edt(guidToSignal[j], 1,
                            get_buffer_guid(rank, localLevel));
            }
            totalNewBfs++;
          }
        }
      } else // This means one of the frontiers was overflown
      {
        ARTS_PRINTF("Next Search Frontier Overflow!\n");
        ARTS_PRINTF("Failed Level: %lu\n", lastLevel);
        ARTS_PRINTF("Shutting down...\n");
        arts_shutdown();
        return;
      }
    }
  }

  if (!totalNewBfs || localLevel == MAXLEVEL) {
    uint64_t stop = arts_get_time_stamp();
    ARTS_PRINTF("Time: %lu\n", stop - start);
    ARTS_PRINTF("Level: %lu\n", localLevel);
    ARTS_PRINTF("Shutting down...\n");
    arts_shutdown();
  }
}

/********************************************************************************************/

extern "C" void init_per_node(unsigned int node_id, int argc, char **argv) {
  char *fileName = argv[1];  //"/home/suet688/ca-HepTh.tsv";
                             ////"/home/firo017/datasets/ca-HepTh.tsv";
  unsigned int num_verts = 0; // 9877;
  unsigned int num_edges = 0; // 51946;
  getProperties(fileName, &num_verts, &num_edges);

  // Create graph partitions
  graph = (csr_graph_t *)arts_calloc(PARTS, sizeof(csr_graph_t));
  distribution =
      init_block_distribution_block(num_verts, num_edges, PARTS, ARTS_DB_GPU_READ);
  load_graph_no_weight_csr(fileName, distribution, true, false);

  // Find the boundaries for sorting
  for (unsigned int i = 0; i < PARTS; i++) {
    bounds[i] = partition_end_distr(i, distribution);
    ARTS_PRINTF("Bounds[%u]: %lu guid: %lu\n", i, bounds[i],
           distribution->graphGuid[i]);
  }

  // Count the number of partitions per node for later...
  partCount =
      (unsigned int *)arts_calloc(arts_get_total_nodes(), sizeof(unsigned int));
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++)
    partCount[i] = 0;

  // Create visited array per partition
  visitedGuid = (arts_guid_t *)arts_calloc(PARTS, sizeof(arts_guid_t));
  visited = (unsigned int **)arts_calloc(PARTS, sizeof(unsigned int *));
  for (unsigned int i = 0; i < PARTS; i++) {
    unsigned int num_elements = get_block_size_for_partition(i, distribution);
    unsigned int size = sizeof(unsigned int) * num_elements;
    // Put the visiter db on the same rank as the graph partition
    unsigned int rank =
        arts_guid_get_rank(get_guid_for_partition_distr(distribution, i));
    visitedGuid[i] = arts_reserve_guid_route(DB_WRITE_TYPE, rank);
    partCount[rank]++;
    // If the partition is on our node lets create the db and -1 it out
    if (rank == node_id) {
      visited[i] = (unsigned int *)arts_db_create_with_guid(visitedGuid[i], size);
      for (unsigned int j = 0; j < num_elements; j++)
        visited[i][j] = UINT32_MAX;
    }
  }

  create_buffers_on_cpu(sizeof(unsigned int) * (GPULISTLEN + 1));

  // Inits some data recording
  init_list_record();
}

extern "C" void initPerGpu(unsigned int node_id, int devId, cudaStream_t *stream,
                           int argc, char *argv) {
  create_buffers_on_gpu(devId, sizeof(unsigned int) * (GPULISTLEN + 1));
}

extern "C" void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  CHECK_CONSISTENCY(worker_id);
  if (!worker_id) {
    create_buffer_db();
    vertex_t source = ROOT;

    if (!node_id) {
      // Spawn a task on the rank containing the source
      unsigned int ownerRank = get_owner_distr(source, distribution);
      uint64_t argsFrRndOne[] = {source};
      arts_guid_t createFirstRoundGuid =
          arts_edt_create(createFirstRound, ownerRank, 1, argsFrRndOne, 0);
    }
    // print_master_buffer_guids();
    // print_local_buffer_guids();
    // print_buffer_ptr();
    // print_raw_ptr();
  }
}

extern "C" void cleanPerGpu(unsigned int node_id, int devId,
                            cudaStream_t *stream) {
  free_buffers_on_gpu(devId);
  write_bins_to_file(devId);
}

int main(int argc, char **argv) {
  DASHDASHFILE(argc, argv)
  arts_rt(argc, argv);
  return 0;
}
