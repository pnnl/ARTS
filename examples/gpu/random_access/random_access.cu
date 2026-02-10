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

/*
 * This code has been contributed by the DARPA HPCS program.  Contact
 * David Koester <dkoester@mitre.org> or Bob Lucas <rflucas@isi.edu>
 * if you have questions.
 *
 * GUPS (Giga UPdates per Second) is a measurement that profiles the memory
 * architecture of a system and is a measure of performance similar to MFLOPS.
 * The HPCS HPCchallenge RandomAccess benchmark is intended to exercise the
 * GUPS capability of a system, much like the LINPACK benchmark is intended to
 * exercise the MFLOPS capability of a computer.  In each case, we would
 * expect these benchmarks to achieve close to the "peak" capability of the
 * memory system. The extent of the similarities between RandomAccess and
 * LINPACK are limited to both benchmarks attempting to calculate a peak system
 * capability.
 *
 * GUPS is calculated by identifying the number of memory locations that can be
 * randomly updated in one second, divided by 1 billion (1e9). The term
 * "randomly" means that there is little relationship between one address to be
 * updated and the next, except that they occur in the space of one half the
 * total system memory.  An update is a read-modify-write operation on a table
 * of 64-bit words. An address is generated, the value at that address read from
 * memory, modified by an integer operation (add, and, or, xor) with a literal
 * value, and that new value is written back to memory.
 *
 * We are interested in knowing the GUPS performance of both entire systems and
 * system subcomponents --- e.g., the GUPS rating of a distributed memory
 * multiprocessor the GUPS rating of an SMP node, and the GUPS rating of a
 * single processor.  While there is typically a scaling of FLOPS with processor
 * count, a similar phenomenon may not always occur for GUPS.
 *
 * For additional information on the GUPS metric, the HPCchallenge RandomAccess
 * Benchmark,and the rules to run RandomAccess or modify it to optimize
 * performance -- see http://icl.cs.utk.edu/hpcc/
 *
 */

/*
 * This file contains the computational core of the single cpu version
 * of GUPS.  The inner loop should easily be vectorized by compilers
 * with such support.
 *
 * This core is used by both the single_cpu and star_single_cpu tests.
 */

#include "random_access_defs.h"

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"
#include "arts/runtime/globals.h"

arts_guid_range_t *updateFrontierGuids = NULL;

unsigned int tile_size = TILESIZE;
unsigned int num_tiles = 0;
arts_guid_range_t *tileGuids = NULL;
uint64_t **tile = NULL;
#define getLocalIndex(v) ((v & (tableSize - 1)) % tile_size)
#define getOwnerIndex(v) ((v & (tableSize - 1)) / tile_size)
#define getTileGuid(v) arts_get_guid(tileGuids, getOwnerIndex(v))

uint64_t start = 0;
arts_guid_t done_guid = NULL_GUID;

/* Perform updates to main table.  The scalar equivalent is:
 *
 *     u64Int ran;
 *     ran = 1;
 *     for (i=0; i<NUPDATE; i++) {
 *       ran = (ran << 1) ^ (((s64Int) ran < 0) ? POLY : 0);
 *       table[ran & (TableSize-1)] ^= ran;
 *     }
 */

uint64_t hpccStartsCPU(int64_t N) {
  int64_t i, j;
  uint64_t m2[64];
  uint64_t temp;
  volatile uint64_t ran;
  volatile int64_t n = N;

  while (n < 0)
    n += PERIOD2;
  while (n > PERIOD2)
    n -= PERIOD2;
  if (n != 0) {
    temp = 0x1;
    for (i = 0; i < 64; i++) {
      m2[i] = temp;
      temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY2 : 0);
      temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY2 : 0);
    }

    for (i = 62; i >= 0; i--)
      if ((n >> i) & 1)
        break;

    ran = 0x2;
    while (i > 0) {
      temp = 0;
      for (j = 0; j < 64; j++)
        if ((ran >> j) & 1)
          temp ^= m2[j];
      ran = temp;
      i -= 1;
      if ((n >> i) & 1)
        ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY2 : 0);
    }
  } else
    ran = 0x1;

  ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY2 : 0);
  return ran;
}

/* Utility routine to start random number generator at Nth step */
__global__ void hpccStarts(int64_cu_t N, uint64_cu_t numUpdates,
                           uint64_cu_t num_tiles, uint64_cu_t tile_size,
                           uint64_cu_t tableSize, uint64_cu_t *rArray) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_cu_t local = 0; local < MAX_TOTAL_PENDING_UPDATES_CU; local++) {
    int64_cu_t localIndex = index * MAX_TOTAL_PENDING_UPDATES_CU + local;
    if (localIndex < numUpdates) {
      // Add the step offset
      volatile int64_cu_t n = N + localIndex;

      int i, j;
      uint64_cu_t m2[64];
      uint64_cu_t temp;

      uint64_cu_t ran = 0x1;

      while (n < 0)
        n += PERIOD;
      while (n > PERIOD)
        n -= PERIOD;

      if (n) {
        temp = 0x1;
        for (i = 0; i < 64; i++) {
          m2[i] = temp;
          temp = (temp << 1) ^ ((int64_cu_t)temp < 0 ? POLY : 0);
          temp = (temp << 1) ^ ((int64_cu_t)temp < 0 ? POLY : 0);
        }

        for (i = 62; i >= 0; i--)
          if ((n >> i) & 1)
            break;

        ran = 0x2;
        while (i > 0) {
          temp = 0;
          for (j = 0; j < 64; j++)
            if ((ran >> j) & 1)
              temp ^= m2[j];
          ran = temp;
          i -= 1;
          if ((n >> i) & 1)
            ran = (ran << 1) ^ ((int64_cu_t)ran < 0 ? POLY : 0);
        }
      } else
        ran = 0x1;

      ran = (ran << 1) ^ ((int64_cu_t)ran < 0 ? POLY : 0);

      rArray[localIndex + num_tiles] = ran;
      uint64_cu_t owner = getOwnerIndex(ran);
      atomicAdd(&rArray[owner], 1ULL);
    }
  }
}

__global__ void updateEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t depv[]) {
  // uint64_t gpu_id = GET_GPU_INDEX();
  // ARTS_PRINTF("Hello from %lu\n", gpu_id);
  uint64_cu_t tile_size = paramv[0];
  uint64_cu_t num_tiles = paramv[1];
  uint64_cu_t tableSize = paramv[2];
  uint64_cu_t numUpdates = paramv[3];
  uint64_cu_t part_index = paramv[4];

  uint64_cu_t *table = (uint64_cu_t *)depv[0].ptr;
  unsigned long long int *ran = (unsigned long long int *)depv[1].ptr;
  ran += num_tiles;

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  for (uint64_cu_t local = 0; local < MAX_TOTAL_PENDING_UPDATES_CU; local++) {
    uint64_cu_t localIndex = index * MAX_TOTAL_PENDING_UPDATES_CU + local;
    if (localIndex < numUpdates) {
      uint64_cu_t localRan = ran[localIndex];
      uint64_cu_t globalRanIndex = localRan & (tableSize - 1);
      if (globalRanIndex / tile_size == part_index) {
        uint64_cu_t localRanIndex = globalRanIndex % tile_size;
        atomicXor(&table[localRanIndex], localRan);
        // atomicAdd(&table[tile_size], 1);
      }
    }
  }
}

void randomEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {

  uint64_t numRemUpdates = paramv[0]; // Number of updates left for this GPU
  uint64_t numRandom = (numRemUpdates > MAX_UPDATES_PER_GPU_STEP)
                           ? MAX_UPDATES_PER_GPU_STEP
                           : numRemUpdates; // Number of updates in the step
  uint64_t step = paramv[1];
  uint64_t index = paramv[2];
  int64_t startIndex =
      (int64_t)(step * MAX_UPDATES_PER_GPU_STEP * arts_get_total_gpus() +
                index * numRandom);
  uint64_t *rArray = (uint64_t *)depv[0].ptr;
  uint64_t tableSize = TABLESIZE;

  if (numRemUpdates) {
    ARTS_PRINTF("Get Random: %lu step: %lu index: %lu startIndex: %lu\n", numRandom,
           step, index, startIndex);
    ARTS_PRINTF("TableSize: %lu rArray: %lu %p num_tiles: %lu\n", tableSize,
           depv[0].guid, depv[0].ptr, num_tiles);
    ARTS_PRINTF("rArray pointer: %p\n", rArray);

    // Call random function
    dim3 block(MAXTHREADS, 1, 1);
    dim3 grid(MAXTHREADBLOCKSPERSM * NUMBEROFSM, 1, 1);
    void *kernelArgs[] = {&startIndex, &numRandom, &num_tiles,
                          &tile_size,   &tableSize, &rArray};
    CHECKCORRECT(cudaLaunchKernel((const void *)hpccStarts, grid, block,
                                  (void **)kernelArgs));
    cudaDeviceSynchronize();

    // Get random counts
    unsigned int elemsToCopy = num_tiles; // + numRandom;
    uint64_t *count = (uint64_t *)arts_calloc(elemsToCopy, sizeof(uint64_t));
    arts_cuda_mem_cpy_from_dev(count, rArray, sizeof(uint64_t) * elemsToCopy);
    // for(uint64_t i=0; i<num_tiles; i++)
    //     ARTS_PRINTF("count[%lu]: %lu\n", i, count[i]);
    // for(uint64_t i=num_tiles; i<elemsToCopy; i++)
    //     ARTS_PRINTF("rand[%llu]: %llu vs %llu SAME: %u\n", i - num_tiles, count[i],
    //     HPCC_starts_CPU(startIndex + i - num_tiles),
    //     HPCC_starts_CPU(startIndex + i - num_tiles) == count[i]);

    // Reserve next randomEdt
    arts_guid_t nextRandomGuid =
        arts_reserve_guid_route(ARTS_EDT, arts_get_current_node());
    unsigned int nextRandomDeps = 1;

    // Create readOnly copy of DB
    arts_guid_t readOnly = arts_db_copy_to_new_type(depv[0].guid, ARTS_DB_GPU_READ);

    // Create update edts
    uint64_t updateArgs[] = {tile_size, num_tiles, tableSize, numRandom, 0};
    for (uint64_t i = 0; i < num_tiles; i++) {
      if (count[i]) {
        updateArgs[4] = i;
        dim3 block(MAXTHREADS, 1, 1);
        dim3 grid(MAXTHREADBLOCKSPERSM * NUMBEROFSM, 1, 1);
        ARTS_PRINTF("Launching for i: %lu count: %lu tileGuid: %lu\n", i, count[i],
               arts_get_guid(tileGuids, i));
        arts_guid_t updateGuid =
            arts_edt_create_gpu(updateEdt, arts_get_current_node(), 5, updateArgs, 2,
                             grid, block, nextRandomGuid, i + 1, NULL_GUID);
        arts_gpu_signal_edt_memset(updateGuid, 0, arts_get_guid(tileGuids, i));
        // arts_signal_edt(updateGuid, 0, arts_get_guid(tileGuids, i));
        arts_signal_edt(updateGuid, 1, readOnly);
        nextRandomDeps++;
      }
    }

    // Free the counts since we are done with them.
    arts_free(count);

    // Create next randomEdt
    uint64_t nextRandom = numRemUpdates - numRandom;
    uint64_t args[] = {nextRandom, step + 1, index};
    arts_edt_create_gpu_lib_with_guid(randomEdt, nextRandomGuid, 3, args,
                                nextRandomDeps, grid, block);
    arts_gpu_signal_edt_memset(nextRandomGuid, 0, depv[0].guid);
    // arts_signal_edt(nextRandomGuid, 0, depv[0].guid);
  } else
    arts_signal_edt(done_guid, (unsigned int)-1, NULL_GUID);
}

void syncEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  uint64_t time = arts_get_time_stamp() - start;
  ARTS_PRINTF("Time %lu\n", time);

  uint64_t *Table = (uint64_t *)arts_calloc(TABLESIZE, sizeof(uint64_t));
  for (uint64_t i = 0; i < TABLESIZE; i++)
    Table[i] = i;

#ifdef VALIDATE
  uint64_t temp = 0x1;
  uint64_t tableSize = TABLESIZE;
  for (uint64_t i = 0; i < NUPDATE; i++) {
    temp = (temp << 1) ^ (((int64_t)temp < 0) ? POLY2 : 0);
    Table[temp & (tableSize - 1)] ^= temp;
    // ARTS_PRINTF("i: %lu index: %lu rand: %lu Table: %lu\n", i, temp &
    // (tableSize-1), temp, Table[temp & (tableSize-1)]);
  }

  bool firstFailure = 1;
  uint64_t totalErrors = 0;
  uint64_t index = 0;
  for (unsigned int i = 0; i < num_tiles; i++) {
    uint64_t *tile = (uint64_t *)depv[i].ptr;
    for (unsigned int j = 0; j < tile_size; j++) {
      if (tile[j] != Table[index]) {
        if (firstFailure) {
          firstFailure = 0;
          ARTS_PRINTF(
              "FAILED on index:%lu Exp: %lu vs Rec: %lu updates: %lu -> %lu\n",
              index, tile[j], Table[index], tile[tile_size],
              Table[index] ^ tile[j]);
        }
        totalErrors++;
      }
      // else
      // ARTS_PRINTF("PASSED on index %lu %lu vs %lu updates: %lu -> %lu\n", index,
      // tile[j], Table[index], tile[tile_size], Table[index]^tile[j]);
      index++;
    }
  }
  if (totalErrors)
    ARTS_PRINTF("%lu errors of %lu!\n", totalErrors, index);
  else
    ARTS_PRINTF("Verified!\n");
#endif

  double GUPS = (double)NUPDATE / time;
  ARTS_PRINTF("GUPS: %lf MB: %lu\n", GUPS,
         (TABLESIZE * sizeof(uint64_t)) / (1024 * 1024));
  arts_shutdown();
}

extern "C" void init_per_node(unsigned int node_id, int argc, char **argv) {
  if (argc > 1)
    tile_size = (unsigned int)atoi(argv[1]);
  num_tiles = TABLESIZE / tile_size;
  ARTS_PRINTF("Random Access Table Size: %u Tile Size: %u Number of Tiles: %u\n",
         TABLESIZE, tile_size, num_tiles);

  // Create tiled table
  tileGuids = arts_new_guid_range_node(ARTS_DB_LC, num_tiles, node_id);
  tile = (uint64_t **)arts_calloc(num_tiles, sizeof(uint64_t *));
  uint64_t counter = 0;
  for (unsigned int i = 0; i < num_tiles; i++) {
    tile[i] = (uint64_t *)arts_db_create_with_guid(
        arts_get_guid(tileGuids, i), (tile_size + 1) * sizeof(uint64_t));
    ARTS_PRINTF("TileGuid[%u]: %lu -> %p\n", i, arts_get_guid(tileGuids, i), tile[i]);
    for (unsigned int j = 0; j < tile_size; j++)
      tile[i][j] = counter++;
    tile[i][tile_size] = 0;
  }

  // Create update frontiers.  The number of updates a frontier can hold is 1024
  // per thread
  unsigned int num_gpus = arts_get_total_gpus();
  unsigned int elemsPerFrontier = num_tiles + MAX_UPDATES_PER_GPU_STEP;
  updateFrontierGuids =
      arts_new_guid_range_node(ARTS_DB_GPU_WRITE, num_gpus, node_id);
  for (unsigned int i = 0; i < num_gpus; i++) {
    uint64_t *updateFrontier =
        (uint64_t *)arts_db_create_with_guid(arts_get_guid(updateFrontierGuids, i),
                                         elemsPerFrontier * sizeof(uint64_t));
    ARTS_PRINTF("updateFrontier[%u]: %lu %p\n", i,
           arts_get_guid(updateFrontierGuids, i), updateFrontier);
    for (unsigned int j = 0; j < elemsPerFrontier; j++)
      updateFrontier[j] = 0;
  }

  // Create a LC sync edt for all partitions
  done_guid = arts_reserve_guid_route(ARTS_EDT, 0);
}

extern "C" void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  if (ARTS_LOOK_UP_CONFIG(gpu_lc_sync) != 6) {
    ARTS_PRINTF(
        "For correct results set gpu_lc_sync=6 in arts.cfg\nShutting Down...\n");
    arts_shutdown();
  }

  if (!node_id && !worker_id) {
    unsigned int num_gpus = arts_get_total_gpus();
    uint64_t numUpdatesPerGpu = NUPDATE / num_gpus;
    ARTS_PRINTF("NumGpus: %u numUpdatesPerGpu: %lu\n", num_gpus, numUpdatesPerGpu);
    dim3 block(MAXTHREADS, 1, 1);
    dim3 grid(MAXTHREADBLOCKSPERSM * NUMBEROFSM, 1, 1);

    // Create NUPDATE / arts_get_total_gpus() getRandomEdts
    uint64_t args[] = {numUpdatesPerGpu, 0, 0};
    for (unsigned int i = 0; i < num_gpus; i++) {
      args[2] = i;
      arts_guid_t updateGuid =
          arts_edt_create_gpu_lib(randomEdt, 0, 3, args, 1, grid, block);
      arts_gpu_signal_edt_memset(updateGuid, 0,
                             arts_get_guid(updateFrontierGuids, i));
    }

    arts_edt_create_with_guid(syncEdt, done_guid, 0, NULL, num_gpus + num_tiles);
    for (unsigned int i = 0; i < num_tiles; i++)
      arts_lc_sync(done_guid, i, arts_get_guid(tileGuids, i));
  }
  start = arts_get_time_stamp();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}