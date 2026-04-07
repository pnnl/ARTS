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
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "arts.h"
#include "arts/memory/db.h"
#include "arts/cxl/wrapper.h"

// #define MEM_SIZE 2000000000ULL // ~2GB
#define MEM_SIZE 200000000ULL // ~200MB
#define ITERATIONS 10

/* Per-DB state: each DB covers (MEM_SIZE / num_dbs) bytes of logical data.
 * For the strided pattern the physical DB allocation is larger to accommodate
 * the gaps; db_alloc_size[] stores the actual allocated size for each DB. */
arts_guid_t *db_guids;   /* array[num_dbs] of DB GUIDs                  */
char       **db_mems;    /* array[num_dbs] of base pointers              */
uint64_t   *db_alloc_size; /* array[num_dbs] of physical allocation sizes */

uint32_t num_procs = 1;
uint64_t num_dbs   = 1;   /* default: 1 DB */
char access_pattern = '\0';
uint64_t access_size = -1;
uint64_t wait_time = -1;
uint64_t stride = 0; // Default value of 0
uint8_t scaling = 0; // 0: Strong, 1: Weak
struct timespec global_start, global_end;

// Print usage information
void print_usage(const char *program_name)
{
  fprintf(stderr, "Usage: %s -n <num_procs> -p <access_pattern> -s <access_size> -w <wait_time> [-d <num_dbs> -t <stride> -g]\n", program_name);
  fprintf(stderr, "  -n <num_procs>: Long integer specifying the number of inhibitor processes\n");
  fprintf(stderr, "  -p <access_pattern>: Character specifying the memory access pattern\n");
  fprintf(stderr, "  -s <access_size>: Long integer specifying the access size\n");
  fprintf(stderr, "  -w <wait_time>: Long integer specifying the wait time\n");
  fprintf(stderr, "  -d <num_dbs>: Optional long integer specifying the number of DBs to spread MEM_SIZE across (default: 1)\n");
  fprintf(stderr, "  -t <stride>: Optional long integer specifying the stride (default: 0)\n");
  fprintf(stderr, "  -g : Flag to weak scale\n");
}

void fill_array(char *array, long size)
{
  static unsigned char x = 1;
  for (long i = 0; i < size; i++) {
    x = (unsigned char)(x + 73);
    array[i] = x;
  }
}

void sleep_us(long us)
{
  struct timespec ts;
  ts.tv_sec = us / 1000000;
  ts.tv_nsec = (us % 1000000) * 1000;
  nanosleep(&ts, NULL);
}

/* Allocate num_dbs DBs, each covering (MEM_SIZE / num_dbs) logical bytes.
 * For the sequential and random patterns the physical size equals the logical
 * size.  For the strided pattern use init_cxl_memory_strided() instead. */
void init_cxl_memory()
{
  uint64_t db_logical_size = MEM_SIZE / num_dbs;

  db_guids      = malloc(sizeof(arts_guid_t) * num_dbs);
  db_mems       = malloc(sizeof(char *)      * num_dbs);
  db_alloc_size = malloc(sizeof(uint64_t)    * num_dbs);

  for (uint64_t d = 0; d < num_dbs; d++)
  {
    db_alloc_size[d] = db_logical_size;
    db_guids[d] = arts_db_create((void **)&db_mems[d], db_logical_size, ARTS_DB_CXL, NULL);
  }
}

/* Allocate num_dbs DBs for the strided ('l') pattern.
 * Each DB holds (MEM_SIZE / num_dbs) bytes of real data plus the gaps
 * introduced by the stride.  The stride is uniform across all DBs. */
void init_cxl_memory_strided()
{
  uint64_t db_logical_size = MEM_SIZE / num_dbs;

  uint64_t num_gaps;
  if (stride == 0)
    num_gaps = 0;
  else
    num_gaps = db_logical_size / stride;

  uint64_t db_physical_size = db_logical_size + (num_gaps * stride);

  db_guids      = malloc(sizeof(arts_guid_t) * num_dbs);
  db_mems       = malloc(sizeof(char *)      * num_dbs);
  db_alloc_size = malloc(sizeof(uint64_t)    * num_dbs);

  for (uint64_t d = 0; d < num_dbs; d++)
  {
    db_alloc_size[d] = db_physical_size;
    db_guids[d] = arts_db_create((void **)&db_mems[d], db_physical_size, ARTS_DB_CXL, NULL);
  }
}

void free_cxl_memory()
{
}

/* ---------------------------------------------------------------------------
 * EDT implementations
 *
 * Each EDT receives:
 *   paramv[0]  access_size
 *   paramv[1]  (uint64_t) base pointer of the DB assigned to this process
 *   paramv[2]  wait_time
 *   paramv[3]  done_guid
 *   paramv[4]  slot          – global process index (0 .. num_procs-1)
 *   paramv[5]  procs_per_db  – number of processes sharing this DB
 *   paramv[6]  local_slot    – index of this process within its DB
 *   paramv[7]  db_alloc_size – physical (allocated) size of the DB
 *              (for sequential/random this equals the logical size;
 *               for linear it includes stride gaps)
 *
 * The per-process area is computed entirely within [0, db_alloc_size) so no
 * process ever crosses a DB boundary.
 *
 * run_linear additionally receives:
 *   paramv[8]  stride
 * --------------------------------------------------------------------------- */

void run_sequential(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[])
{
  uint64_t size          = paramv[0];
  char    *db_mem        = (char *)paramv[1];
  uint64_t wait_time     = paramv[2];
  arts_guid_t done_guid  = paramv[3];
  uint32_t slot          = paramv[4];
  uint64_t procs_per_db  = paramv[5];
  uint64_t local_slot    = paramv[6];
  uint64_t db_alloc_size = paramv[7];

  int mode = 0;
  char *temp = malloc(sizeof(char) * size);

  /* Divide the DB evenly among the processes assigned to it */
  long area_size  = (long)(db_alloc_size / procs_per_db);
  long start_idx  = (long)(local_slot * area_size);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (uint32_t idx = 0; idx < ITERATIONS; idx++)
  {
    long curr_idx = start_idx;
    long end_idx  = start_idx + area_size;
    mode = (mode + 1) % 2;
    while ((curr_idx + size) < end_idx)
    {
      if (!mode)
      {
        fill_array(&(db_mem[curr_idx]), size);
        FLUSH_FENCE_PRODUCER((void *)&(db_mem[curr_idx]), ALIGN_UP(size, CACHELINE_SIZE));
        curr_idx += size;
        sleep_us(wait_time);
        mode = (mode + 1) % 2;
      }
      else
      {
        FLUSH_FENCE_CONSUMER((void *)&(db_mem[curr_idx]), ALIGN_UP(size, CACHELINE_SIZE));
        memcpy(temp, &(db_mem[curr_idx]), size);
        curr_idx += size;
        sleep_us(wait_time);
        mode = (mode + 1) % 2;
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t elapsed_microseconds = (end.tv_sec - start.tv_sec) * 1000000;
  elapsed_microseconds += (end.tv_nsec - start.tv_nsec) / 1000;
  free(temp);
  double *bandwidth;
  arts_guid_t bandwidth_guid = arts_db_create((void **)&bandwidth, sizeof(double), ARTS_DB_DEFAULT, NULL);
  *bandwidth = (((double)area_size * ITERATIONS)) / (elapsed_microseconds / 1e6);
  arts_signal_edt(done_guid, slot, bandwidth_guid, DB_MODE_RO);
}

void run_linear(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[])
{
  uint64_t size          = paramv[0];
  char    *db_mem        = (char *)paramv[1];
  uint64_t wait_time     = paramv[2];
  arts_guid_t done_guid  = paramv[3];
  uint32_t slot          = paramv[4];
  uint64_t procs_per_db  = paramv[5];
  uint64_t local_slot    = paramv[6];
  uint64_t db_phys_size  = paramv[7];
  uint64_t stride        = paramv[8];

  int mode = 0;
  char *temp = malloc(sizeof(char) * size);

  /* Divide the DB's physical allocation evenly among the processes in this DB */
  long area_size = (long)(db_phys_size / procs_per_db);
  long start_idx = (long)(local_slot * area_size);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (uint32_t idx = 0; idx < ITERATIONS; idx++)
  {
    long curr_idx = start_idx;
    long end_idx  = start_idx + area_size;
    mode = (mode + 1) % 2;
    while ((curr_idx + size + stride) < end_idx)
    {
      if (!mode)
      {
        fill_array(&(db_mem[curr_idx]), size);
        FLUSH_FENCE_PRODUCER((void *)&(db_mem[curr_idx]), ALIGN_UP(size, CACHELINE_SIZE));
        curr_idx += size + stride;
        sleep_us(wait_time);
        mode = (mode + 1) % 2;
      }
      else
      {
        FLUSH_FENCE_CONSUMER((void *)&(db_mem[curr_idx]), ALIGN_UP(size, CACHELINE_SIZE));
        memcpy(temp, &(db_mem[curr_idx]), size);
        curr_idx += size + stride;
        sleep_us(wait_time);
        mode = (mode + 1) % 2;
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t elapsed_microseconds = (end.tv_sec - start.tv_sec) * 1000000;
  elapsed_microseconds += (end.tv_nsec - start.tv_nsec) / 1000;
  free(temp);
  double *bandwidth;
  arts_guid_t bandwidth_guid = arts_db_create((void **)&bandwidth, sizeof(double), ARTS_DB_DEFAULT, NULL);
  *bandwidth = (((double)area_size * ITERATIONS)) / (elapsed_microseconds / 1e6);
  arts_signal_edt(done_guid, slot, bandwidth_guid, DB_MODE_RO);
}

void run_random(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[])
{
  uint64_t size          = paramv[0];
  char    *db_mem        = (char *)paramv[1];
  uint64_t wait_time     = paramv[2];
  arts_guid_t done_guid  = paramv[3];
  uint32_t slot          = paramv[4];
  uint64_t procs_per_db  = paramv[5];
  uint64_t local_slot    = paramv[6];
  uint64_t db_alloc_size = paramv[7];

  int mode = 0;
  char *temp = malloc(sizeof(char) * size);

  long area_size = (long)(db_alloc_size / procs_per_db);
  long start_idx = (long)(local_slot * area_size);

  /* Precompute random index table to eliminate overhead from the hot loop.
   * Each entry is a pre-aligned offset within [0, area_size - size].
   * We use a fixed table size that is large enough to avoid pattern repetition
   * but small enough to generate quickly. Using a power of two allows a
   * cheap bitmask instead of modulo in the hot loop. */
#define RAND_TABLE_BITS 14
#define RAND_TABLE_SIZE (1 << RAND_TABLE_BITS)   /* 16384 entries */
#define RAND_TABLE_MASK (RAND_TABLE_SIZE - 1)

  long num_blocks = area_size / size;   /* number of aligned blocks in area */
  long *rand_table = malloc(sizeof(long) * RAND_TABLE_SIZE);

  /* Use xorshift64 to fill table quickly with no stdlib overhead */
  uint64_t rng = 0xdeadbeefcafeULL ^ (uint64_t)slot; /* per-slot seed */
  for (int i = 0; i < RAND_TABLE_SIZE; i++)
  {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    /* Map to a block index, then scale to byte offset.
     * % num_blocks keeps us within the area; result is already
     * a multiple of size so no extra alignment is needed. */
    rand_table[i] = (long)(rng % (uint64_t)num_blocks) * size;
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  uint32_t tbl_idx = 0;   /* walks rand_table; stays in a register */

  for (uint32_t idx = 0; idx < ITERATIONS; idx++)
  {
    long ops = area_size / size;   /* access every block once per iteration */
    mode = (mode + 1) % 2;

    for (long op = 0; op < ops; op++)
    {
      /* Single bitmask replaces modulo — no branch, no division */
      long offset = rand_table[tbl_idx & RAND_TABLE_MASK];
      tbl_idx++;

      long curr_idx = start_idx + offset;

      if (!mode)
      {
      fill_array(&(db_mem[curr_idx]), size);
      FLUSH_FENCE_PRODUCER((void *)&(db_mem[curr_idx]),
      ALIGN_UP(size, CACHELINE_SIZE));
      sleep_us(wait_time);
      mode = (mode + 1) % 2;
      }
      else
      {
      FLUSH_FENCE_CONSUMER((void *)&(db_mem[curr_idx]),
      ALIGN_UP(size, CACHELINE_SIZE));
      memcpy(temp, &(db_mem[curr_idx]), size);
      sleep_us(wait_time);
      mode = (mode + 1) % 2;
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  uint64_t elapsed_microseconds = (end.tv_sec - start.tv_sec) * 1000000;
  elapsed_microseconds += (end.tv_nsec - start.tv_nsec) / 1000;

  free(temp);
  free(rand_table);

  double *bandwidth;
  arts_guid_t bandwidth_guid = arts_db_create((void **)&bandwidth,
  sizeof(double),
  ARTS_DB_DEFAULT, NULL);
  *bandwidth = (((double)area_size * ITERATIONS)) / (elapsed_microseconds / 1e6);
  arts_signal_edt(done_guid, slot, bandwidth_guid, DB_MODE_RO);
}

void done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[])
{
  clock_gettime(CLOCK_MONOTONIC, &global_end);
  printf("Finished running\n");
  double bw_sum = 0;
  for (uint32_t i = 0; i < depc; i++)
  {
    bw_sum += *((double *)depv[i].ptr);
    // printf("Received bandwidth: %f\n", *((double *)depv[i].ptr));
  }
  printf("Average per-process bandwidth: %f MB/s\n", ((bw_sum / 1e6) / depc));

  uint64_t elapsed_microseconds = (global_end.tv_sec - global_start.tv_sec) * 1000000;
  elapsed_microseconds += (global_end.tv_nsec - global_start.tv_nsec) / 1000;
  double global_bandwidth;
  if (!scaling)
    global_bandwidth = ((double)((uint64_t)MEM_SIZE * (uint64_t)ITERATIONS)) / (elapsed_microseconds / 1e6);
  else
    global_bandwidth = ((double)((uint64_t)MEM_SIZE * (uint64_t)ITERATIONS) * num_procs) / (elapsed_microseconds / 1e6);
  printf("Global concurrent bandwidth: %f MB/s\n", global_bandwidth / 1e6);
  arts_shutdown();
}

void init_per_node(unsigned int node_id, int argc, char **argv)
{
  if (!node_id)
  {
    int opt;
    num_procs = arts_get_total_workers();
    // Parse command line arguments
    while ((opt = getopt(argc, argv, "p:s:w:d:t:g")) != -1)
    {
      switch (opt)
      {
      // case 'n':
        // num_procs = strtol(optarg, NULL, 10);
        // break;
      case 'p':
        access_pattern = optarg[0];
        break;
      case 's':
        access_size = strtol(optarg, NULL, 10);
        break;
      case 'w':
        wait_time = strtol(optarg, NULL, 10);
        break;
      case 'd':
        num_dbs = strtol(optarg, NULL, 10);
        break;
      case 't':
        stride = ALIGN_UP(strtol(optarg, NULL, 10), CACHELINE_SIZE);
        break;
      case 'g':
        scaling = 1;
        break;
      default:
        print_usage(argv[0]);
        arts_shutdown();
      }
    }

    // Validate that all required arguments were provided
    if (access_pattern == '\0' || access_size == -1 || wait_time == -1)
    {
      fprintf(stderr, "Error: Missing required arguments\n");
      print_usage(argv[0]);
      arts_shutdown();
    }

    /* num_dbs must divide num_procs evenly so every DB gets the same number
     * of processes.  Clamp to num_procs if the user supplied a larger value. */
    if (num_dbs > num_procs)
    {
      fprintf(stderr, "Warning: num_dbs (%lu) > num_procs (%u); clamping to num_procs\n",
              num_dbs, num_procs);
      num_dbs = num_procs;
    }
    if (num_procs % num_dbs != 0)
    {
      fprintf(stderr, "Error: num_procs (%u) must be divisible by num_dbs (%lu)\n",
              num_procs, num_dbs);
      arts_shutdown();
    }

    // Print the parsed arguments (for verification)
    printf("Number of inhibitor processes: %d\n", num_procs);
    printf("Number of DBs: %lu\n", num_dbs);
    printf("Access Pattern: %c\n", access_pattern);
    printf("Access Size: %ld\n", access_size);
    printf("Wait Time: %ld\n", wait_time);
    printf("Stride: %ld\n", stride);
    if (scaling)
    {
      printf("Scaling: weak\n");
    }
    else
    {
      printf("Scaling: strong\n");
    }

    // Initialize CXL memory
    if (access_pattern == 'l')
    {
      printf("Initializing strided memory (%lu DB(s))\n", num_dbs);
      init_cxl_memory_strided();
    }
    else
    {
      printf("Initializing sequential memory (%lu DB(s))\n", num_dbs);
      init_cxl_memory();
    }

    for (uint32_t i = 0; i < num_dbs; i++)
    {
      arts_printf("DB %u hosted on FAM device: %lu\n", i, GET_CXL_DEV_ID(db_mems[i]));
    }

    /* Number of processes assigned to each DB */
    uint64_t procs_per_db = num_procs / num_dbs;

    switch (access_pattern)
    {
    case 's':
    {
      printf("Sequential access\n");
      unsigned int next = arts_get_current_node();
      arts_guid_t done_guid = arts_edt_create(done, 0, NULL, num_procs, NULL);
      clock_gettime(CLOCK_MONOTONIC, &global_start);
      for (uint32_t i = 0; i < num_procs; i++)
      {
        uint64_t db_idx     = i / procs_per_db;  /* which DB this process belongs to */
        uint64_t local_slot = i % procs_per_db;  /* index within that DB              */
        uint64_t args[] = {access_size, (uint64_t)db_mems[db_idx], wait_time,
                           done_guid, i, procs_per_db, local_slot,
                           db_alloc_size[db_idx]};
        arts_guid_t edt = arts_edt_create(run_sequential, 8, args, 0, &(arts_hint_t){.route = next});
        next = (next + 1) % arts_get_total_nodes();
      }
      break;
    }
    case 'l':
    {
      printf("Linear access\n");
      unsigned int next = arts_get_current_node();
      arts_guid_t done_guid = arts_edt_create(done, 0, NULL, num_procs, NULL);
      clock_gettime(CLOCK_MONOTONIC, &global_start);
      for (uint32_t i = 0; i < num_procs; i++)
      {
        uint64_t db_idx     = i / procs_per_db;
        uint64_t local_slot = i % procs_per_db;
        uint64_t args[] = {access_size, (uint64_t)db_mems[db_idx], wait_time,
                           done_guid, i, procs_per_db, local_slot,
                           db_alloc_size[db_idx], stride};
        arts_guid_t edt = arts_edt_create(run_linear, 9, args, 0, &(arts_hint_t){.route = next});
        next = (next + 1) % arts_get_total_nodes();
      }
      break;
    }
    case 'r':
    {
      printf("Random access\n");
      unsigned int next = arts_get_current_node();
      arts_guid_t done_guid = arts_edt_create(done, 0, NULL, num_procs, NULL);
      clock_gettime(CLOCK_MONOTONIC, &global_start);
      for (uint32_t i = 0; i < num_procs; i++)
      {
        uint64_t db_idx     = i / procs_per_db;
        uint64_t local_slot = i % procs_per_db;
        uint64_t args[] = {access_size, (uint64_t)db_mems[db_idx], wait_time,
                           done_guid, i, procs_per_db, local_slot,
                           db_alloc_size[db_idx]};
        arts_guid_t edt = arts_edt_create(run_random, 8, args, 0, &(arts_hint_t){.route = next});
        next = (next + 1) % arts_get_total_nodes();
      }
      break;
    }
    default:
      printf("Invalid access pattern. Defaulting to sequential\n");
      arts_shutdown();
      break;
    }
  }
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                     char **argv)
{
}

int main(int argc, char **argv)
{
  arts_rt(argc, argv);
  return 0;
}
