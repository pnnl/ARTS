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

#include "arts.h"
#include "arts/memory/db.h"

arts_guid_t *update_frontier_guids = NULL;
arts_guid_t update_frontier_guid;

unsigned int tile_size = TILESIZE;
unsigned int num_tiles = 0;
// arts_guid_t *tile_guids = NULL;
arts_guid_t *tile_guids = NULL;
uint64_t **tile = NULL;
#define get_local_index(v) ((v & (table_size - 1)) % tile_size)
#define get_owner_index(v) ((v & (table_size - 1)) / tile_size)
#define get_tile_guid(v)   arts_guid_from_index(tile_guids, get_owner_index(v))

uint64_t start = 0;
arts_guid_t done_guid = NULL_GUID;
arts_guid_t update_guid = NULL_GUID;

/* Perform updates to main table.  The scalar equivalent is:
 *
 *     u64Int ran;
 *     ran = 1;
 *     for (i=0; i<NUPDATE; i++) {
 *       ran = (ran << 1) ^ (((s64Int) ran < 0) ? POLY : 0);
 *       table[ran & (TableSize-1)] ^= ran;
 *     }
 */

uint64_t hpcc_starts_cpu(int64_t N) {
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

// void hpcc_starts_tiled(int64_cu_t N, uint64_cu_t num_updates,
                          //  uint64_cu_t num_tiles, uint64_cu_t tile_size,
                          //  uint64_cu_t tableSize, uint64_cu_t *r_array) {
  
void hpcc_starts_tiled(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t depv[]) {
  uint64_cu_t N = paramv[0];
  uint64_cu_t index = paramv[1];
  uint64_cu_t num_updates = paramv[2];
  uint64_cu_t num_tiles = paramv[3];
  uint64_cu_t table_size = paramv[4];
  uint64_cu_t tile_size = paramv[5];
  uint64_cu_t* r_array = (uint64_cu_t*) paramv[6];
  arts_guid_t update_guid = paramv[7];
  // unsigned long long int* global_ran = paramv[8];
  arts_guid_t tile_guid = depv[0].guid;
  uint64_cu_t* table = depv[0].ptr;
  unsigned long long int* global_ran = depv[1].ptr;
  global_ran += num_tiles;
  
  for (int64_cu_t local = 0; local < MAX_TOTAL_PENDING_UPDATES_CU; local++) {
    int64_cu_t local_index = index * MAX_TOTAL_PENDING_UPDATES_CU + local;
    if (local_index < num_updates) {
      // Add the step offset
      volatile int64_cu_t n = N + local_index;

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

      r_array[local_index + num_tiles] = ran; //TODO: Check logic here
      uint64_cu_t owner = get_owner_index(ran);
      r_array[owner] += 1ULL;
    }
  }
  uint64_cu_t part_index = index;
  arts_cxl_producer_flush(tile_guid);
  arts_signal_edt(update_guid, (index+1), tile_guid, DB_MODE_EW);
}

/* Utility routine to start random number generator at Nth step */
void hpcc_starts(int64_cu_t N, uint64_cu_t num_updates,
                           uint64_cu_t num_tiles, uint64_cu_t tile_size,
                           uint64_cu_t table_size, uint64_cu_t *r_array,
                           arts_guid_t next_guid) {
  uint64_t next = arts_get_current_node();
  for (uint64_cu_t index = 0; index < num_tiles; index++) {
    uint64_t args[8] = {N, index, num_updates, num_tiles,
                       table_size, tile_size, (uint64_t) r_array, next_guid};
    arts_guid_t hpcc_tiled_guid = arts_edt_create(hpcc_starts_tiled, 8, args, 2,
                                                   &(arts_hint_t){.route = next});
    arts_signal_edt(hpcc_tiled_guid, 0, tile_guids[index], DB_MODE_EW);
    arts_signal_edt(hpcc_tiled_guid, 1, update_frontier_guid, DB_MODE_RO);
    next = (next+1)%arts_get_total_nodes();
  }
}

void update_edt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t depv[]) {
  uint64_cu_t tile_size = paramv[0];
  uint64_cu_t num_tiles = paramv[1];
  uint64_cu_t table_size = paramv[2];
  uint64_cu_t num_updates = paramv[3];
  uint64_cu_t part_index = paramv[4];
  arts_guid_t next_guid = paramv[5];
  uint32_t slot = paramv[6];
  
  arts_cxl_consumer_flush(depv[0].guid);  // Flush the tile before accessing
  arts_cxl_consumer_flush(depv[1].guid);  // Flush the update_frontier before accessing

  uint64_cu_t *table = (uint64_cu_t *)depv[0].ptr;
  unsigned long long int *ran = (unsigned long long int *)depv[1].ptr;
  ran += num_tiles;

  for (uint64_cu_t local = 0; local < num_updates; local++) {
    uint64_cu_t local_ran = ran[local];
    uint64_cu_t global_ran_index = local_ran & (table_size - 1);
    if (global_ran_index / tile_size == part_index) {  // Check if this update belongs to our partition
        uint64_cu_t local_ran_index = global_ran_index % tile_size;
        __atomic_fetch_xor(&table[local_ran_index], local_ran, __ATOMIC_SEQ_CST);
    }
  }
  arts_signal_edt_null(next_guid, slot);
  arts_cxl_producer_flush(depv[0].guid);
}

void update_driver(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  uint64_t tile_size = paramv[0];
  uint64_t num_tiles = paramv[1];
  uint64_t table_size = paramv[2];
  uint64_t num_random = paramv[3];
  uint64_t next_random_guid = paramv[4];

  arts_cxl_consumer_flush(depv[0].guid);
  for (unsigned int i = 0; i < num_tiles; i++) {
    arts_cxl_consumer_flush(depv[i+1].guid);
  }
  
  arts_guid_t read_only = depv[0].guid;
  uint64_t update_args[] = {tile_size, num_tiles, table_size, num_random, 0, next_random_guid, 0};
  uint64_t next = arts_get_current_node();
  for (uint64_t i = 0; i < num_tiles; i++) {
    update_args[4] = i;
    update_args[6] = i+1;
    arts_guid_t update_guid = arts_edt_create(update_edt,
                                7, update_args, 2,
                                &(arts_hint_t){.route = next});
    arts_signal_edt(update_guid, 0, depv[i+1].guid, DB_MODE_EW); // tile_guid
    arts_signal_edt(update_guid, 1, read_only, DB_MODE_RO);
    next = (next+1)%arts_get_total_nodes();
  }
}

void random_driver(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  uint64_t num_rem_updates = paramv[0];
  uint64_t num_random = (num_rem_updates > MAX_UPDATES_PER_CPU_STEP)
                          ? MAX_UPDATES_PER_CPU_STEP
                          : num_rem_updates; // Number of updates in the step
  uint64_t step = paramv[1];
  uint64_t index = paramv[2];

  int64_t start_index =
      (int64_t)(step * MAX_UPDATES_PER_CPU_STEP +
                index * num_random);
  arts_cxl_consumer_flush(depv[0].guid);
  uint64_t *r_array = (uint64_t *)depv[0].ptr;
  uint64_t table_size = TABLESIZE;

  //TODO: Probably need to separate hpcc and update edts. Need to wait for all hpcc
  // EDTs to finish before running update
  
  if (num_rem_updates) {
    // arts_guid_t next_guid = arts_guid_reserve(ARTS_EDT, 0);
    uint64_t next_random = num_rem_updates - num_random;
    uint64_t args[4] = {next_random, step+1, index, num_tiles};
    // arts_edt_create_with_guid(random_driver, next_guid, 4, args, num_tiles+1);
    arts_guid_t next_guid = arts_edt_create(random_driver, 4, args, num_tiles+1,
                                            &(arts_hint_t){.route = 0});
    arts_signal_edt(next_guid, 0, update_frontier_guid, DB_MODE_RO);

    uint64_t update_args[5] = {tile_size, num_tiles, table_size, num_random, next_guid};
    arts_guid_t update_guid = arts_edt_create(update_driver, 5, update_args, num_tiles+1,
                                              &(arts_hint_t){.route = 0});
    hpcc_starts(start_index, num_random, num_tiles,
                tile_size, table_size, (uint64_cu_t*) r_array, update_guid);
    arts_signal_edt(update_guid, 0, depv[0].guid, DB_MODE_RO);
  }
  else {
    arts_signal_edt_null(done_guid, num_tiles);
  }
}

void sync_edt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  uint64_t time = arts_get_time_stamp() - start;
  PRINTF("Time %lu\n", time);
  
  for (unsigned int i = 0; i < num_tiles; i++) {
    arts_cxl_consumer_flush(depv[i].guid);
  }

  uint64_t *Table = (uint64_t *)calloc(TABLESIZE, sizeof(uint64_t));
  for (uint64_t i = 0; i < TABLESIZE; i++)
    Table[i] = i;

#ifdef VALIDATE
  uint64_t temp = 0x1;
  uint64_t table_size = TABLESIZE;
  for (uint64_t i = 0; i < NUPDATE; i++) {
    temp = (temp << 1) ^ (((int64_t)temp < 0) ? POLY2 : 0);
    Table[temp & (table_size - 1)] ^= temp;
  }

  bool first_failure = 1;
  uint64_t total_errors = 0;
  uint64_t index = 0;
  for (unsigned int i = 0; i < num_tiles; i++) {
    uint64_t *tile = (uint64_t *)depv[i].ptr;
    for (unsigned int j = 0; j < tile_size; j++) {
      if (tile[j] != Table[index]) {
        if (first_failure) {
          first_failure = 0;
          PRINTF(
              "FAILED on index:%lu Exp: %lu vs Rec: %lu updates: %lu -> %lu\n",
              index, tile[j], Table[index], tile[tile_size],
              Table[index] ^ tile[j]);
        }
        total_errors++;
      }
      index++;
    }
  }
  if (total_errors)
    PRINTF("%lu errors of %lu!\n", total_errors, index);
  else
    PRINTF("Verified!\n");
#endif

  double GUPS = (double)NUPDATE / time;
  PRINTF("GUPS: %lf MB: %lu\n", GUPS,
         (TABLESIZE * sizeof(uint64_t)) / (1024 * 1024));
  arts_shutdown();
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  if (!node_id) {
    if (argc > 1)
      tile_size = (unsigned int)atoi(argv[1]);
    num_tiles = TABLESIZE / tile_size;
    PRINTF("Random Access Table Size: %u Tile Size: %u Number of Tiles: %u\n",
           TABLESIZE, tile_size, num_tiles);

    // Create tiled table
    tile_guids = (arts_guid_t*)calloc(num_tiles, sizeof(arts_guid_t)); // TODO: Make sure to free
    tile = (uint64_t **)calloc(num_tiles, sizeof(uint64_t *));
    uint64_t counter = 0;
    
    for (unsigned int i = 0; i < num_tiles; i++) {
      
      tile_guids[i] = arts_db_create((void**)&(tile[i]), (tile_size + 1)*sizeof(uint64_t),
                                 ARTS_DB_CXL, NULL);

      for (unsigned int j = 0; j < tile_size; j++)
        tile[i][j] = counter++;
      tile[i][tile_size] = 0;
      arts_cxl_producer_flush(tile_guids[i]);
    }
    
    // Create update frontiers.  The number of updates a frontier can hold is 1024
    // per thread
    unsigned int elems_per_frontier = num_tiles + MAX_TOTAL_PENDING_UPDATES;
    uint64_t *update_frontier;
    update_frontier_guid = arts_db_create((void**)&update_frontier, elems_per_frontier*sizeof(uint64_t), ARTS_DB_CXL, NULL);
    for (unsigned int j = 0; j < elems_per_frontier; j++)
      update_frontier[j] = 0;
    arts_cxl_producer_flush(update_frontier_guid);

    // Create a LC sync edt for all partitions
    done_guid = arts_guid_reserve(ARTS_EDT, 0);
    update_guid = arts_guid_reserve(ARTS_EDT, 0);
  }
}

void init_per_worker(unsigned int node_id, unsigned int worker_id,
                              int argc, char **argv) {
  if (!node_id && !worker_id) {
    PRINTF("Num updates: %lu\n", NUPDATE);
    uint64_t args[] = {NUPDATE, 0, 0, num_tiles};
    arts_edt_create_with_guid(random_driver, update_guid, 4, args, 1); 
    arts_signal_edt(update_guid, 0, update_frontier_guid, DB_MODE_RO);

    arts_edt_create_with_guid(sync_edt, done_guid, 0, NULL, 1 + num_tiles);
    for (unsigned int i = 0; i < num_tiles; i++) {
      arts_signal_edt(done_guid, i, tile_guids[i], DB_MODE_EW);
    }
  }
  start = arts_get_time_stamp();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
