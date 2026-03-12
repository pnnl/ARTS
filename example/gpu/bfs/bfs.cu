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

#include <cuda_runtime_api.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "arts/arts.h"
#include "arts/gpu/GpuRuntime.cuh"
#include "arts/runtime/RT.h"

#include "bfsDefs.h"
#include "bins.h"
#include "graphUtil.cuh"

uint64_t start = 0;       // Timer
unsigned int **devPtrRaw; // The pointers for our nextSearchFrontier on each gpu
artsGuid_t *nextSearchFrontierAddrGuid; // db that holds guids for the
                                        // nextSearchFrontiers (devPtrRaw)
unsigned int bounds[PARTS];      // This is the boundaries that make up each
                                 // partition
arts_block_dist_t *distribution; // The graph distribution
csr_graph_t *graph;              // Partitions of the graph
unsigned int **visited; // This is the resulting parent list for each partition
artsGuid_t
    *visitedGuid; // This is the guid for each partition of the parent list
unsigned int *partCount;

void createFirstRound(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                      artsEdtDep_t depv[]);
__global__ void bfs(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                    artsEdtDep_t depv[]);
void launchSort(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                artsEdtDep_t depv[]);
void thrustSort(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                artsEdtDep_t depv[]);
void launchBfs(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               artsEdtDep_t depv[]);

void printDeviceList(thrust::device_ptr<unsigned int> devPtr,
                     unsigned int size) {
  PRINTF("FRONTIER SIZE: %u\n", size);
  for (unsigned int i = 0; i < size; i++) {
    unsigned int temp = *(devPtr + i);
    PRINTF("%u, ", temp);
  }
  PRINTF("\n");
}

void printResult() {
  for (unsigned int i = 0; i < PARTS; i++) {
    unsigned int size =
        sizeof(unsigned int) * getBlockSizeForPartition(i, distribution);
    PRINTF("%u: %u\n", i, size);
    for (unsigned int j = 0; j < size; j++)
      PRINTF("%u, ", visited[i][j]);
    PRINTF("\n");
  }
}

void createFirstRound(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                      artsEdtDep_t depv[]) {
  PRINTF("%s\n", __func__);
  start = artsGetTimeStamp();

  // We are on the rank of the source
  uint64_t src = paramv[0];
  uint64_t nextLevel = 0;

  // Create the first search frontier!
  unsigned int *firstSearchFrontier = NULL;
  artsGuid_t firstSearchFrontierGuid =
      artsDbCreate((void **)&firstSearchFrontier, 2 * sizeof(unsigned int),
                   ARTS_DB_GPU_READ);
  firstSearchFrontier[0] = 1;   // size of the frontier
  firstSearchFrontier[1] = src; // root
  PRINTF("ROOT: %u GRAPH GUID: %lu VISITED GUID: %lu\n", firstSearchFrontier[1],
         getGuidForVertexDistr(firstSearchFrontier[1], distribution),
         visitedGuid[getOwnerDistr(firstSearchFrontier[1], distribution)]);

  // TODO: Do we need an epoch in the first round...
  // Create the first epoch
  artsGuid_t launchSortGuid =
      artsEdtCreate(launchSort, artsGetCurrentNode(), 1, &nextLevel, 1);
  artsInitializeAndStartEpoch(launchSortGuid, 0);

  dim3 threads(1, 1, 1);
  dim3 grid(1, 1, 1);
  // Launching the first bfs
  artsGuid_t graphGuid =
      getGuidForVertexDistr(firstSearchFrontier[1], distribution);
  artsGuid_t visitGuid =
      visitedGuid[getOwnerDistr(firstSearchFrontier[1], distribution)];
  artsGuid_t bfsGuid =
      artsEdtCreateGpu(bfs, artsGetCurrentNode(), 1, &nextLevel, 4, grid,
                       threads, NULL_GUID, 0, NULL_GUID);
  artsSignalEdt(bfsGuid, 0, visitGuid);
  artsSignalEdt(bfsGuid, 1, nextSearchFrontierAddrGuid[artsGetCurrentNode()]);
  artsSignalEdt(bfsGuid, 2, firstSearchFrontierGuid);
  artsSignalEdt(bfsGuid, 3, graphGuid);
  PRINTF("LAUNCHING\n");
}

__global__ void bfs(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                    artsEdtDep_t depv[]) {
  uint64_t gpuId = getGpuIndex(); // The current gpu we are on
  unsigned int localLevel = (unsigned int)paramv[0];
  unsigned int *localVisited = (unsigned int *)depv[0].ptr;
  unsigned int **addr =
      (unsigned int **)depv[1].ptr;  // This is the devPtrRaw -> tells us where
                                     // next frontier is on device
  unsigned int *local = addr[gpuId]; // We need the one corresponding to our gpu
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
      uint64_t neighborCount = 0;
      getNeighborsGpu(localGraph, v, &neighbors, &neighborCount);
      if (neighborCount) {
        unsigned int frontierIndex =
            atomicAdd(localFrontierCount, (unsigned int)neighborCount);
        if (frontierIndex < GPULISTLEN) {
          for (uint64_t i = 0; i < neighborCount; ++i) {
            local[frontierIndex + i] = neighbors[i];
          }
        }
      }
    }
  }
}

// LC will sync all the version coming into this edt and then we will start the
// next epoch
void doPartionSync(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                   artsEdtDep_t depv[]) {
  PRINTF("Just Synced Partitions! %lu\n", paramv[0]);
  artsSignalEdt(paramv[1], (uint32_t)-1, NULL_GUID);
}

// There is only one of these per level.  It is signaled by the epoch containing
// the Bfs'es
void launchSort(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                artsEdtDep_t depv[]) {
  uint64_t localLevel = paramv[0];
  // uint64_t edtsRan = depv[0].guid;
  PRINTF("%s Level: %lu Edts Ran: %lu\n", __func__, localLevel, edtsRan);

  // This is tricky.  We need to create the epoch for the next round since
  // thrustSort will create the next rounds' Bfs'es.  In order to create the
  // epoch, we need the next round's launchSort.
  uint64_t nextLevel = localLevel + 1;
  artsGuid_t nextLaunchSortGuid =
      artsEdtCreate(launchSort, artsGetCurrentNode(), 1, &nextLevel, 1);
  artsInitializeAndStartEpoch(nextLaunchSortGuid, 0);

  // While we are at it, lets create the next sync point, launchBfs.
  artsGuid_t nextLaunchBfsGuid =
      artsReserveGuidRoute(ARTS_EDT, artsGetCurrentNode());
  uint32_t nextLaunchBfsDepc = artsGetTotalNodes() * artsGetTotalGpus();

  // Lasly, we will launch a sort for every gpu in the system.
  // We need the nextBfsEpoch and the nextLaunchBfsGuids to kick off
  // launchBfs...
  dim3 threads(1, 1, 1);
  dim3 grid(1, 1, 1);
  uint64_t args[] = {localLevel, (uint64_t)nextLaunchBfsGuid};
  for (unsigned int j = 0; j < artsGetTotalNodes(); j++) {
    for (uint64_t i = 0; i < artsGetTotalGpus(); i++)
      artsGuid_t sortGuid = artsEdtCreateGpuLibDirect(thrustSort, j, i, 2, args,
                                                      0, grid, threads);
  }

  // This uses the LC memory model if turned on
  if (DO_SYNC(localLevel)) {
    for (unsigned int i = 0; i < artsGetTotalNodes(); i++) {
      uint64_t syncArgs[] = {localLevel, (uint64_t)nextLaunchBfsGuid};
      artsGuid_t edtGuid =
          artsEdtCreate(doPartionSync, i, 2, syncArgs, partCount[i]);
      PRINTF("edtGuid: %lu\n", edtGuid);
      unsigned int slot = 0;
      for (unsigned int j = 0; j < PARTS; j++) {
        if (i == artsGuidGetRank(visitedGuid[j])) {
          PRINTF("Signaling: %lu with part: %lu\n", edtGuid, visitedGuid[j]);
          artsLCSync(edtGuid, slot++, visitedGuid[j]);
        }
      }
    }
    nextLaunchBfsDepc += artsGetTotalNodes();
  }
  PRINTF("nextLaunchBfsDepc: %lu\n", nextLaunchBfsDepc);
  artsEdtCreateWithGuid(launchBfs, nextLaunchBfsGuid, 1, &nextLevel,
                        nextLaunchBfsDepc);
}

void thrustSort(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                artsEdtDep_t depv[]) {
  uint64_t localLevel =
      paramv[0]; // This can be the end if the frontier is empty
  PRINTF("%s Level: %lu Gpu: %d\n", __func__, localLevel, artsGetGpuId());
  artsGuid_t nextLaunchBfsGuid = paramv[1]; // This is the next sync point.
  artsGuid_t edtGuidsToLaunchBfsGuid =
      NULL_GUID; // Where we will put a copy of all the new edts to start...
  unsigned int *rawPtr =
      devPtrRaw[artsGetGpuId()]; // The corresponding dev pointer (frontier) to
                                 // our gpu

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
    PRINTF("Upper: %u Size: %u\n", bounds[0], sizePerBound[0]);
    for (unsigned int i = 1; i < PARTS; i++) {
      sizePerBound[i] = upperIndexPerBound[i] - upperIndexPerBound[i - 1];
      PRINTF("Upper: %u Size: %u\n", bounds[i], sizePerBound[i]);
    }

    // TODO: Clear old dbs (previous frontiers)...
    artsGuid_t *edtGuidsToLaunchBfs =
        NULL; // This will hold the new edt guids to launch
    edtGuidsToLaunchBfsGuid =
        artsDbCreate((void **)&edtGuidsToLaunchBfs, sizeof(artsGuid_t) * PARTS,
                     ARTS_DB_READ);

    uint64_t nextLevel = localLevel + 1;
    unsigned tempIndex = 0;
    for (unsigned int i = 0; i < PARTS; i++) {
      if (sizePerBound[i]) {
        unsigned int *newSearchFrontier =
            NULL; // This will hold a tile of the new frontier
        artsGuid_t newSearchFrontierGuid = artsDbCreate(
            (void **)&newSearchFrontier,
            sizeof(unsigned int) * (sizePerBound[i] + 1), ARTS_DB_GPU_READ);
        *newSearchFrontier = sizePerBound[i];

        // Copy the data from the gpu to the host
        artsPutInDbFromGpu(thrust::raw_pointer_cast(devPtr) + tempIndex,
                           newSearchFrontierGuid, sizeof(unsigned int),
                           sizeof(unsigned int) * sizePerBound[i], false);
        tempIndex += sizePerBound[i];

        dim3 threads(SMTILE, 1, 1);
        dim3 grid((sizePerBound[i] + SMTILE - 1) / SMTILE, 1, 1); // Ceiling
        PRINTF("SMTILE: %u grid: %u\n", SMTILE,
               (sizePerBound[i] + SMTILE - 1) / SMTILE);

        // Create the new edt for each bfs
        unsigned int rank =
            artsGuidGetRank(getGuidForPartitionDistr(distribution, i));
        edtGuidsToLaunchBfs[i] =
            artsEdtCreateGpu(bfs, rank, 1, &nextLevel, 4, grid, threads,
                             NULL_GUID, 0, NULL_GUID);
        artsSignalEdt(edtGuidsToLaunchBfs[i], 0, visitedGuid[i]);
        artsSignalEdt(edtGuidsToLaunchBfs[i], 2, newSearchFrontierGuid);
        artsSignalEdt(edtGuidsToLaunchBfs[i], 3,
                      getGuidForPartitionDistr(distribution, i));

        addToList(sizePerBound[i], artsGetGpuId());
      } else
        edtGuidsToLaunchBfs[i] = NULL_GUID;
    }
  }
  artsSignalEdt(nextLaunchBfsGuid,
                artsGetCurrentNode() * artsGetTotalGpus() + artsGetGpuId(),
                edtGuidsToLaunchBfsGuid);
}

// This needs nodes * gpus signals.  Each db has PARTS guids to signal.
void launchBfs(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               artsEdtDep_t depv[]) {
  uint64_t totalNewBfs = 0;
  uint64_t nextLevel = paramv[0];
  PRINTF("%s Level: %lu\n", __func__, nextLevel);

  if (nextLevel < MAXLEVEL) {
    // from each gpu, we get a bunch of bfs-es that need to be spawned
    unsigned int numPotentialBfsDbs = artsGetTotalNodes() * artsGetTotalGpus();
    for (unsigned int i = 0; i < numPotentialBfsDbs; i++) {
      if (depv[i].guid) {
        artsGuid_t *guidToSignal = (artsGuid_t *)depv[i].ptr;
        for (unsigned int j = 0; j < PARTS; j++) {
          if (guidToSignal[j]) {
            unsigned int rank = artsGuidGetRank(guidToSignal[j]);
            artsSignalEdt(guidToSignal[j], 1, nextSearchFrontierAddrGuid[rank]);
            totalNewBfs++;
          }
        }
      } else // This means one of the frontiers was overflown
      {
        PRINTF("Next Search Frontier Overflow!\n");
        PRINTF("Failed Level: %lu\n", nextLevel - 1);
        PRINTF("Shutting down...\n");
        artsShutdown();
        return;
      }
    }
  }

  if (!totalNewBfs || nextLevel == MAXLEVEL) {
    uint64_t stop = artsGetTimeStamp();
    PRINTF("Time: %lu\n", stop - start);
    PRINTF("Level: %lu\n", nextLevel);
    PRINTF("Shutting down...\n");
    artsShutdown();
  }
}

/********************************************************************************************/

extern "C" void initPerNode(unsigned int nodeId, int argc, char **argv) {
  char *fileName = argv[1];  //"/home/suet688/ca-HepTh.tsv";
                             ////"/home/firo017/datasets/ca-HepTh.tsv";
  unsigned int numVerts = 0; // 9877;
  unsigned int numEdges = 0; // 51946;
  getProperties(fileName, &numVerts, &numEdges);

  // Create graph partitions
  graph = (csr_graph_t *)artsCalloc(PARTS, sizeof(csr_graph_t));
  distribution =
      initBlockDistributionBlock(numVerts, numEdges, PARTS, ARTS_DB_GPU_READ);
  loadGraphNoWeightCsr(fileName, distribution, true, false);

  // Find the boundaries for sorting
  for (unsigned int i = 0; i < PARTS; i++) {
    bounds[i] = partitionEndDistr(i, distribution);
    PRINTF("Bounds[%u]: %lu guid: %lu\n", i, bounds[i],
           distribution->graphGuid[i]);
  }

  // Count the number of partitions per node for later...
  partCount =
      (unsigned int *)artsCalloc(artsGetTotalNodes(), sizeof(unsigned int));
  for (unsigned int i = 0; i < artsGetTotalNodes(); i++)
    partCount[i] = 0;

  // Create visited array per partition
  visitedGuid = (artsGuid_t *)artsCalloc(PARTS, sizeof(artsGuid_t));
  visited = (unsigned int **)artsCalloc(PARTS, sizeof(unsigned int *));
  for (unsigned int i = 0; i < PARTS; i++) {
    unsigned int numElements = getBlockSizeForPartition(i, distribution);
    unsigned int size = sizeof(unsigned int) * numElements;
    // Put the visiter db on the same rank as the graph partition
    unsigned int rank =
        artsGuidGetRank(getGuidForPartitionDistr(distribution, i));
    visitedGuid[i] = artsReserveGuidRoute(DB_WRITE_TYPE, rank);
    partCount[rank]++;
    // If the partition is on our node lets create the db and -1 it out
    if (rank == nodeId) {
      visited[i] = (unsigned int *)artsDbCreateWithGuid(visitedGuid[i], size);
      for (unsigned int j = 0; j < numElements; j++)
        visited[i][j] = UINT32_MAX;
    }
  }

  // Let's reserve the guids for the DBs that will hold the gpu array
  // (nextSearchFrontier)
  nextSearchFrontierAddrGuid =
      (artsGuid_t *)artsMalloc(sizeof(artsGuid_t) * artsGetTotalNodes());
  for (unsigned int i = 0; i < artsGetTotalNodes(); i++)
    nextSearchFrontierAddrGuid[i] = artsReserveGuidRoute(ARTS_DB_GPU_READ, i);

  // Create an array to hold the addresses of next search frontier for each gpu
  devPtrRaw =
      (unsigned int **)artsCalloc(artsGetTotalGpus(), sizeof(unsigned int *));

  // Inits some data recording
  initListRecord();
}

extern "C" void initPerGpu(unsigned int nodeId, int devId, cudaStream_t *stream,
                           int argc, char *argv) {
  // Create the next search frontier for each gpu
  devPtrRaw[devId] =
      (unsigned int *)artsCudaMalloc(sizeof(unsigned int) * (GPULISTLEN + 1));
}

extern "C" void initPerWorker(unsigned int nodeId, unsigned int workerId,
                              int argc, char **argv) {
  checkConsistency(workerId);
  if (!workerId) {
    vertex_t source = ROOT;
    // Lets create the db of to hold device address of the next search frontier
    // so gpu kernels can use them
    unsigned int **addr = (unsigned int **)artsDbCreateWithGuid(
        nextSearchFrontierAddrGuid[nodeId],
        sizeof(unsigned int *) * artsGetTotalGpus());
    for (uint64_t i = 0; i < artsGetTotalGpus(); i++)
      addr[i] = devPtrRaw[i];

    if (!nodeId) {
      // Spawn a task on the rank containing the source
      unsigned int ownerRank = getOwnerDistr(source, distribution);
      uint64_t argsFrRndOne[] = {source};
      artsGuid_t createFirstRoundGuid =
          artsEdtCreate(createFirstRound, ownerRank, 1, argsFrRndOne, 0);
    }
  }
}

extern "C" void cleanPerGpu(unsigned int nodeId, int devId,
                            cudaStream_t *stream) {
  artsCudaFree(devPtrRaw[devId]);
  writeBinsToFile(devId);
}

int main(int argc, char **argv) {
  DASHDASHFILE(argc, argv)
  artsRT(argc, argv);
  return 0;
}
