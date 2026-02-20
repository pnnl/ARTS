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

#include <cuda_runtime_api.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/gpu/gpu_runtime.cuh"
#include "arts/gpu/gpu_stream.h"
#include "arts/runtime/globals.h"

#include "random_access_defs.h"

arts_guid_range_t *update_frontier_guids = NULL;

unsigned int tile_size = TILESIZE;
unsigned int num_tiles = 0;
arts_guid_range_t *tile_guids = NULL;
uint64_t **tile = NULL;
#define GET_LOCAL_INDEX(v) (((v) & (tableSize - 1)) % tile_size)
#define GET_OWNER_INDEX(v) (((v) & (tableSize - 1)) / tile_size)
#define GET_TILE_GUID(v) arts_guid_range_get(tile_guids, GET_OWNER_INDEX(v))

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

uint64_t hpcc_starts_cpu(int64_t num) {
  int64_t i;
  int64_t j;
  uint64_t m2_arr[64];
  uint64_t temp;
  volatile uint64_t ran;
  volatile int64_t n = num;

  while (n < 0) {
    n += PERIOD2;
  }
  while (n > PERIOD2) {
    n -= PERIOD2;
  }
  if (n != 0) {
    temp = 0x1;
    for (i = 0; i < 64; i++) {
      m2_arr[i] = temp;
      temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY2 : 0);
      temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY2 : 0);
    }

    for (i = 62; i >= 0; i--) {
      if ((n >> i) & 1) {
        break;
      }
    }

    ran = 0x2;
    while (i > 0) {
      temp = 0;
      for (j = 0; j < 64; j++) {
        if ((ran >> j) & 1) {
          temp ^= m2_arr[j];
        }
      }
      ran = temp;
      i -= 1;
      if ((n >> i) & 1) {
        ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY2 : 0);
      }
    }
  } else {
    ran = 0x1;
  }

  ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY2 : 0);
  return ran;
}

/* Utility routine to start random number generator at Nth step */
__global__ void hpcc_starts(int64_cu_t num, uint64_cu_t num_updates,
                            uint64_cu_t num_tiles, uint64_cu_t tile_size,
                            uint64_cu_t tableSize, uint64_cu_t *r_array) {
  int index = (int)(threadIdx.x + (blockIdx.x * blockDim.x));
  for (int64_cu_t local = 0; local < MAX_TOTAL_PENDING_UPDATES_CU; local++) {
    int64_cu_t local_index =
        (int64_cu_t)(index * MAX_TOTAL_PENDING_UPDATES_CU) + local;
    if (local_index < num_updates) {
      // Add the step offset
      volatile int64_cu_t n = num + local_index;

      int i;
      int j;
      uint64_cu_t m2_arr[64];
      uint64_cu_t temp;

      uint64_cu_t ran = 0x1;

      while (n < 0) {
        n += PERIOD;
      }
      while (n > PERIOD) {
        n -= PERIOD;
      }

      if (n) {
        temp = 0x1;
        for (i = 0; i < 64; i++) {
          m2_arr[i] = temp;
          temp = (temp << 1) ^ ((int64_cu_t)temp < 0 ? POLY : 0);
          temp = (temp << 1) ^ ((int64_cu_t)temp < 0 ? POLY : 0);
        }

        for (i = 62; i >= 0; i--) {
          if ((n >> i) & 1) {
            break;
          }
        }

        ran = 0x2;
        while (i > 0) {
          temp = 0;
          for (j = 0; j < 64; j++) {
            if ((ran >> j) & 1) {
              temp ^= m2_arr[j];
            }
          }
          ran = temp;
          i -= 1;
          if ((n >> i) & 1) {
            ran = (ran << 1) ^ ((int64_cu_t)ran < 0 ? POLY : 0);
          }
        }
      } else {
        ran = 0x1;
      }

      ran = (ran << 1) ^ ((int64_cu_t)ran < 0 ? POLY : 0);

      r_array[local_index + num_tiles] = ran;
      uint64_cu_t owner = GET_OWNER_INDEX(ran);
      atomicAdd(&r_array[owner], 1ULL);
    }
  }
}

__global__ void update_edt(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  // uint64_t gpu_id = GET_GPU_INDEX();
  // arts_printf("Hello from %lu\n", gpu_id);
  uint64_cu_t tile_size = paramv[0];
  uint64_cu_t num_tiles = paramv[1];
  uint64_cu_t table_size = paramv[2];
  uint64_cu_t num_updates = paramv[3];
  uint64_cu_t part_index = paramv[4];

  uint64_cu_t *table = (uint64_cu_t *)depv[0].ptr;
  unsigned long long int *ran = (unsigned long long int *)depv[1].ptr;
  ran += num_tiles;

  int index = (int)(threadIdx.x + (blockIdx.x * blockDim.x));
  for (uint64_cu_t local = 0; local < MAX_TOTAL_PENDING_UPDATES_CU; local++) {
    uint64_cu_t local_index =
        (uint64_cu_t)(index * MAX_TOTAL_PENDING_UPDATES_CU) + local;
    if (local_index < num_updates) {
      uint64_cu_t local_ran = ran[local_index];
      uint64_cu_t global_ran_index = local_ran & (table_size - 1);
      if (global_ran_index / tile_size == part_index) {
        uint64_cu_t local_ran_index = global_ran_index % tile_size;
        atomicXor(&table[local_ran_index], local_ran);
        // atomicAdd(&table[tile_size], 1);
      }
    }
  }
}

void random_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;

  uint64_t num_rem_updates = paramv[0]; // Number of updates left for this GPU
  uint64_t num_random = (num_rem_updates > (uint64_t)MAX_UPDATES_PER_GPU_STEP)
                            ? (uint64_t)MAX_UPDATES_PER_GPU_STEP
                            : num_rem_updates; // Number of updates in the step
  uint64_t step = paramv[1];
  uint64_t index = paramv[2];
  int64_t start_index = (int64_t)((step * (uint64_t)MAX_UPDATES_PER_GPU_STEP *
                                   arts_get_total_gpus()) +
                                  (index * num_random));
  uint64_t *r_array = (uint64_t *)depv[0].ptr;
  uint64_t table_size = TABLESIZE;

  if (num_rem_updates) {
    arts_printf("Get Random: %lu step: %lu index: %lu startIndex: %lu\n",
                num_random, step, index, start_index);
    arts_printf("TableSize: %lu rArray: %lu %p num_tiles: %lu\n", table_size,
                depv[0].guid, depv[0].ptr, num_tiles);
    arts_printf("rArray pointer: %p\n", r_array);

    // Call random function
    dim3 block(MAXTHREADS, 1, 1);
    dim3 grid(MAXTHREADBLOCKSPERSM * NUMBEROFSM, 1, 1);
    void *kernel_args[] = {&start_index, &num_random, &num_tiles,
                           &tile_size,   &table_size, &r_array};
    CHECKCORRECT(cudaLaunchKernel((const void *)hpcc_starts, grid, block,
                                  (void **)kernel_args));
    cudaDeviceSynchronize();

    // Get random counts
    unsigned int elems_to_copy = num_tiles; // + numRandom;
    uint64_t *count = (uint64_t *)calloc(elems_to_copy, sizeof(uint64_t));
    arts_cuda_mem_cpy_from_dev(count, r_array,
                               sizeof(uint64_t) * elems_to_copy);
    // for(uint64_t i=0; i<num_tiles; i++)
    //     arts_printf("count[%lu]: %lu\n", i, count[i]);
    // for(uint64_t i=num_tiles; i<elems_to_copy; i++)
    //     arts_printf("rand[%llu]: %llu vs %llu SAME: %u\n", i - num_tiles,
    //     count[i], hpcc_starts_cpu(start_index + i - num_tiles),
    //     hpcc_starts_cpu(start_index + i - num_tiles) == count[i]);

    // Reserve next random_edt
    arts_guid_t next_random_guid =
        arts_guid_reserve(ARTS_EDT, arts_get_current_node());
    unsigned int next_random_deps = 1;

    // Create readOnly copy of DB
    arts_guid_t read_only =
        arts_db_copy_to_new_type(depv[0].guid, ARTS_DB_GPU_READ);

    // Create update edts
    uint64_t update_args[] = {tile_size, num_tiles, table_size, num_random, 0};
    for (uint64_t i = 0; i < num_tiles; i++) {
      if (count[i]) {
        update_args[4] = i;
        dim3 block(MAXTHREADS, 1, 1);
        dim3 grid(MAXTHREADBLOCKSPERSM * NUMBEROFSM, 1, 1);
        arts_printf("Launching for i: %lu count: %lu tileGuid: %lu\n", i,
                    count[i], arts_guid_range_get(tile_guids, i));
        arts_guid_t update_guid = arts_edt_create_gpu(
            update_edt, arts_get_current_node(), 5, update_args, 2, grid, block,
            next_random_guid, i + 1, NULL_GUID);
        arts_gpu_signal_edt_memset(update_guid, 0,
                                   arts_guid_range_get(tile_guids, i));
        // arts_signal_edt(update_guid, 0, arts_guid_range_get(tile_guids, i),
        // ARTS_DB_WRITE);
        arts_signal_edt(update_guid, 1, read_only, ARTS_DB_WRITE);
        next_random_deps++;
      }
    }

    // Free the counts since we are done with them.
    free(count);

    // Create next random_edt
    uint64_t next_random = num_rem_updates - num_random;
    uint64_t args[] = {next_random, step + 1, index};
    arts_edt_create_gpu_lib_with_guid(random_edt, next_random_guid, 3, args,
                                      next_random_deps, grid, block);
    arts_gpu_signal_edt_memset(next_random_guid, 0, depv[0].guid);
    // arts_signal_edt(next_random_guid, 0, depv[0].guid, ARTS_DB_WRITE);
  } else {
    arts_signal_edt(done_guid, (unsigned int)-1, NULL_GUID, ARTS_DB_WRITE);
  }
}

void sync_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t time = arts_get_time_stamp() - start;
  arts_printf("Time %lu\n", time);

  uint64_t *table = (uint64_t *)calloc(TABLESIZE, sizeof(uint64_t));
  for (uint64_t i = 0; i < TABLESIZE; i++) {
    table[i] = i;
  }

#ifdef VALIDATE
  uint64_t temp = 0x1;
  uint64_t table_size = TABLESIZE;
  for (uint64_t i = 0; i < NUPDATE; i++) {
    temp = (temp << 1) ^ (((int64_t)temp < 0) ? POLY2 : 0);
    table[temp & (table_size - 1)] ^= temp;
    // arts_printf("i: %lu index: %lu rand: %lu Table: %lu\n", i, temp &
    // (table_size-1), temp, table[temp & (table_size-1)]);
  }

  bool first_failure = 1;
  uint64_t total_errors = 0;
  uint64_t index = 0;
  for (unsigned int i = 0; i < num_tiles; i++) {
    uint64_t *tile = (uint64_t *)depv[i].ptr;
    for (unsigned int j = 0; j < tile_size; j++) {
      if (tile[j] != table[index]) {
        if (first_failure) {
          first_failure = 0;
          arts_printf(
              "FAILED on index:%lu Exp: %lu vs Rec: %lu updates: %lu -> %lu\n",
              index, tile[j], table[index], tile[tile_size],
              table[index] ^ tile[j]);
        }
        total_errors++;
      }
      // else
      // arts_printf("PASSED on index %lu %lu vs %lu updates: %lu -> %lu\n",
      // index, tile[j], table[index], tile[tile_size],
      // table[index]^tile[j]);
      index++;
    }
  }
  if (total_errors) {
    arts_printf("%lu errors of %lu!\n", total_errors, index);
  } else {
    arts_printf("Verified!\n");
  }
#endif

  double gups = (double)NUPDATE / (double)time;
  arts_printf("GUPS: %lf MB: %lu\n", gups,
              (TABLESIZE * sizeof(uint64_t)) / (1024 * 1024));
  arts_shutdown();
}

extern "C" void arts_main_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  unsigned int node_id = arts_get_current_node();

  if (argc > 1) {
    tile_size = (unsigned int)strtol(argv[1], NULL, 10);
  }
  num_tiles = TABLESIZE / tile_size;
  arts_printf(
      "Random Access Table Size: %u Tile Size: %u Number of Tiles: %u\n",
      TABLESIZE, tile_size, num_tiles);

  // Create tiled table
  tile_guids = arts_guid_range_create(ARTS_DB_LC, num_tiles, node_id);
  tile = (uint64_t **)calloc(num_tiles, sizeof(uint64_t *));
  uint64_t counter = 0;
  for (unsigned int i = 0; i < num_tiles; i++) {
    tile[i] = (uint64_t *)arts_db_create_with_guid(
        arts_guid_range_get(tile_guids, i), (tile_size + 1) * sizeof(uint64_t),
        NULL);
    arts_printf("TileGuid[%u]: %lu -> %p\n", i,
                arts_guid_range_get(tile_guids, i), tile[i]);
    for (unsigned int j = 0; j < tile_size; j++) {
      tile[i][j] = counter++;
    }
    tile[i][tile_size] = 0;
  }

  // Create update frontiers
  unsigned int num_gpus = arts_get_total_gpus();
  unsigned int elems_per_frontier =
      num_tiles + (unsigned int)MAX_UPDATES_PER_GPU_STEP;
  update_frontier_guids =
      arts_guid_range_create(ARTS_DB_GPU_WRITE, num_gpus, node_id);
  for (unsigned int i = 0; i < num_gpus; i++) {
    uint64_t *update_frontier = (uint64_t *)arts_db_create_with_guid(
        arts_guid_range_get(update_frontier_guids, i),
        elems_per_frontier * sizeof(uint64_t), NULL);
    arts_printf("updateFrontier[%u]: %lu %p\n", i,
                arts_guid_range_get(update_frontier_guids, i), update_frontier);
    for (unsigned int j = 0; j < elems_per_frontier; j++) {
      update_frontier[j] = 0;
    }
  }

  done_guid = arts_guid_reserve(ARTS_EDT, 0);

  if (ARTS_LOOK_UP_CONFIG(gpu_lc_sync) != 6) {
    arts_printf("For correct results set gpu_lc_sync=6 in arts.cfg\nShutting "
                "Down...\n");
    arts_shutdown();
    return;
  }

  uint64_t num_updates_per_gpu = NUPDATE / num_gpus;
  arts_printf("NumGpus: %u numUpdatesPerGpu: %lu\n", num_gpus,
              num_updates_per_gpu);
  dim3 block(MAXTHREADS, 1, 1);
  dim3 grid(MAXTHREADBLOCKSPERSM * NUMBEROFSM, 1, 1);

  uint64_t args[] = {num_updates_per_gpu, 0, 0};
  for (unsigned int i = 0; i < num_gpus; i++) {
    args[2] = i;
    arts_guid_t update_guid =
        arts_edt_create_gpu_lib(random_edt, 0, 3, args, 1, grid, block);
    arts_gpu_signal_edt_memset(update_guid, 0,
                               arts_guid_range_get(update_frontier_guids, i));
  }

  arts_edt_create_with_guid(sync_edt, done_guid, 0, NULL, num_gpus + num_tiles);
  for (unsigned int i = 0; i < num_tiles; i++) {
    arts_lc_sync(done_guid, i, arts_guid_range_get(tile_guids, i));
  }
  start = arts_get_time_stamp();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
