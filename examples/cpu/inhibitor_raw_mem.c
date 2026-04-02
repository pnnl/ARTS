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

// #define MEM_SIZE 2000000000 // ~2GB
// #define MEM_SIZE 200000000
#define MEM_SIZE 20000000
#define ITERATIONS 10

arts_guid_t mem_guid;
char *mem;
uint32_t num_procs = 1;
uint64_t num_dbs = 0;
char access_pattern = '\0';
uint64_t access_size = -1;
uint64_t wait_time = -1;
uint64_t stride = 0; // Default value of 0
uint8_t scaling = 0; // 0: Strong, 1: Weak
struct timespec global_start, global_end;

// Print usage information
void print_usage(const char *program_name)
{
  fprintf(stderr, "Usage: %s -n <num_procs> -p <access_pattern> -s <access_size> -w <wait_time> [-t <stride> -g]\n", program_name);
  fprintf(stderr, "  -n <num_procs>: Long integer specifying the number of inhibitor processes\n");
  fprintf(stderr, "  -p <access_pattern>: Character specifying the memory access pattern\n");
  fprintf(stderr, "  -s <access_size>: Long integer specifying the access size\n");
  fprintf(stderr, "  -w <wait_time>: Long integer specifying the wait time\n");
  fprintf(stderr, "  -t <stride>: Optional long integer specifying the stride (default: 0)\n");
  fprintf(stderr, "  -g : Flag to weak scale\n");
}

// Fill array with random bytes
void fill_array(char *array, long size)
{
  // static int seeded = 0;
  // if (!seeded)
  // {
  //   srand((unsigned int)time(NULL));
  //   seeded = 1;
  // }
  for (int i = 0; i < size; i++)
  {
    // array[i] = (char)(rand() % 256); // random byte 0–255
    array[i] = 1;
  }
}

void sleep_us(long us)
{
  struct timespec ts;
  ts.tv_sec = us / 1000000;
  ts.tv_nsec = (us % 1000000) * 1000;
  nanosleep(&ts, NULL);
}

void init_cxl_memory()
{
  mem_guid = arts_db_create((void **)&mem, MEM_SIZE, ARTS_DB_CXL, NULL);
}

// void init_cxl_memory_strided()
// {
//   mem_guids = (arts_guid_t *)malloc(sizeof(arts_guid_t) * num_dbs * 2); // Reserve guids for stride between each data guid
//   mem = (char **)malloc(sizeof(char *) * num_dbs * 2);
//   for (uint64_t i = 0; i < num_dbs * 2; i++)
//   {
//     // Even guids get accessed, odd guids are strides
//     if (i % 2)
//       mem_guids[i] = arts_db_create((void **)&(mem[i]), stride, ARTS_DB_CXL, NULL);
//     else
//       mem_guids[i] = arts_db_create((void **)&(mem[i]), access_size, ARTS_DB_CXL, NULL);
//     // memset(mem[i], 0, access_size);
//     // arts_cxl_producer_flush(mem_guids[i]);
//   }
// }

void free_cxl_memory()
{
}

void run_sequential(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[])
{
  uint64_t size = paramv[0];
  char *mem = (char *)paramv[1];
  uint64_t wait_time = paramv[2];
  arts_guid_t done_guid = paramv[3];
  uint32_t slot = paramv[4];
  uint64_t num_procs = paramv[5];
  printf("Num Procs: %lu\n", num_procs);
  int mode = 0;
  char *temp = malloc(sizeof(char) * size);
  long start_idx = slot * (MEM_SIZE / num_procs);
  long area_size = MEM_SIZE / num_procs;
  printf("Area Size: %ld \n", area_size);
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (uint32_t idx = 0; idx < ITERATIONS; idx++)
  {
    long curr_idx = start_idx;
    long end_idx = start_idx + area_size;
    mode = (mode + 1) % 2;
    while ((curr_idx + size) < end_idx)
    {
      if (!mode)
      {
        // printf("Rank %d: Writing...\n", rank);
        fill_array(&(mem[curr_idx]), size);
        FLUSH_FENCE_PRODUCER((void *)&(mem[curr_idx]), ALIGN_UP(size, CACHELINE_SIZE));
        curr_idx += size;
        // printf("Rank %d: Waiting...\n", rank);
        sleep_us(wait_time);
        mode = (mode + 1) % 2;
      }
      else
      {
        // printf("Rank %d: Reading...\n", rank);
        FLUSH_FENCE_CONSUMER((void *)&(mem[curr_idx]), ALIGN_UP(size, CACHELINE_SIZE));
        memcpy(temp, &(mem[curr_idx]), size);
        curr_idx += size;
        // printf("Rank %d: Waiting...\n", rank);
        sleep_us(wait_time);
        mode = (mode + 1) % 2;
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t elapsed_microseconds = (end.tv_sec - start.tv_sec) * 1000000; // Seconds to microseconds
  elapsed_microseconds += (end.tv_nsec - start.tv_nsec) / 1000;          // Nanoseconds to microseconds
  free(temp);
  double *bandwidth;
  arts_guid_t bandwidth_guid = arts_db_create((void **)&bandwidth, sizeof(double), ARTS_DB_DEFAULT, NULL);
  *bandwidth = (((double)area_size * ITERATIONS)) / (elapsed_microseconds / 1e6);
  arts_signal_edt(done_guid, slot, bandwidth_guid, DB_MODE_RO);
}

/*
void run_linear(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[])
{
  uint64_t access_size = paramv[0];
  uint64_t length = paramv[1];
  uint64_t wait_time = paramv[2];
  arts_guid_t done_guid = paramv[3];
  uint32_t slot = paramv[4];
  struct timespec start, end;
  int mode = 0;
  char *temp = malloc(sizeof(char) * access_size * length);
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (uint32_t idx = 0; idx < ITERATIONS; idx++)
  {
    for (uint64_t i = 0; i < length * 2; i += 2)
    {
      if (!mode)
      {
        fill_array(depv[i].ptr, access_size);
        arts_cxl_producer_flush(depv[i].guid);
        sleep_us(wait_time);
      }
      else
      {
        arts_cxl_consumer_flush(depv[i].guid);
        memcpy(temp + i, depv[i].ptr, access_size);
      }
    }
    mode = (mode + 1) % 2;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t elapsed_microseconds = (end.tv_sec - start.tv_sec) * 1000000; // Seconds to microseconds
  elapsed_microseconds += (end.tv_nsec - start.tv_nsec) / 1000;          // Nanoseconds to microseconds
  free(temp);
  double *bandwidth;
  arts_guid_t bandwidth_guid = arts_db_create((void **)&bandwidth, sizeof(double), ARTS_DB_DEFAULT, NULL);
  *bandwidth = ((double)(access_size * length * ITERATIONS)) / (elapsed_microseconds / 1e6);
  arts_signal_edt(done_guid, slot, bandwidth_guid, DB_MODE_RO);
}

void run_random(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[])
{
  uint64_t access_size = paramv[0];
  uint64_t length = paramv[1];
  uint64_t wait_time = paramv[2];
  arts_guid_t done_guid = paramv[3];
  uint32_t slot = paramv[4];
  struct timespec start, end;
  int mode = 0;
  char *temp = malloc(sizeof(char) * access_size * length);
  clock_gettime(CLOCK_MONOTONIC, &start);

  srand(time(NULL) + arts_get_current_worker());
  for (uint32_t i = 0; i < ITERATIONS; i++)
  {
    for (uint64_t j = 0; j < length; j++)
    {
      uint64_t idx = (uint64_t)rand() % length;
      if (!mode)
      {
        fill_array(depv[idx].ptr, access_size);
        arts_cxl_producer_flush(depv[idx].guid);
        sleep_us(wait_time);
      }
      else
      {
        arts_cxl_consumer_flush(depv[idx].guid);
        memcpy(temp + j, depv[idx].ptr, access_size);
      }
    }
    mode = (mode + 1) % 2;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t elapsed_microseconds = (end.tv_sec - start.tv_sec) * 1000000; // Seconds to microseconds
  elapsed_microseconds += (end.tv_nsec - start.tv_nsec) / 1000;          // Nanoseconds to microseconds
  free(temp);
  double *bandwidth;
  arts_guid_t bandwidth_guid = arts_db_create((void **)&bandwidth, sizeof(double), ARTS_DB_DEFAULT, NULL);
  *bandwidth = ((double)(access_size * length * ITERATIONS)) / (elapsed_microseconds / 1e6);
  arts_signal_edt(done_guid, slot, bandwidth_guid, DB_MODE_RO);
}
*/

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
  printf("Average bandwidth: %f MB/s\n", ((bw_sum / 1e6) / depc));

  uint64_t elapsed_microseconds = (global_end.tv_sec - global_start.tv_sec) * 1000000; // Seconds to microseconds
  elapsed_microseconds += (global_end.tv_nsec - global_start.tv_nsec) / 1000;          // Nanoseconds to microseconds
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
    // Parse command line arguments
    while ((opt = getopt(argc, argv, "n:p:s:w:t:g")) != -1)
    {
      switch (opt)
      {
      case 'n':
        num_procs = strtol(optarg, NULL, 10);
        break;
      case 'p':
        access_pattern = optarg[0];
        break;
      case 's':
        access_size = strtol(optarg, NULL, 10);
        break;
      case 'w':
        wait_time = strtol(optarg, NULL, 10);
        break;
      case 't':
        stride = strtol(optarg, NULL, 10);
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

    // Print the parsed arguments (for verification)
    printf("Number of inhibitor processes: %d\n", num_procs);
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
      printf("Initializing strided memory\n");
      // init_cxl_memory_strided();
    }
    else
    {
      printf("Initializing sequential memory\n");
      init_cxl_memory();
    }

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
        uint64_t args[] = {access_size, (uint64_t)mem, wait_time, done_guid, i, num_procs};
        arts_guid_t edt = arts_edt_create(run_sequential, 6, args, 0, &(arts_hint_t){.route = next});
        next = (next + 1) % arts_get_total_nodes();
      }
      break;
    }
    case 'l':
    {
      /*
      printf("Linear access\n");
      uint64_t addr_idx = 0;
      uint64_t dbs_per_proc = num_dbs / num_procs;
      unsigned int next = arts_get_current_node();
      arts_guid_t done_guid = arts_edt_create(done, 0, NULL, num_procs, NULL);
      for (uint32_t i = 0; i < num_procs; i++)
      {
        uint64_t args[] = {access_size, dbs_per_proc, wait_time, done_guid, i};
        arts_guid_t edt = arts_edt_create(run_linear, 5, args, dbs_per_proc * 2, &(arts_hint_t){.route = next});
        uint64_t db_count = 0;
        while (db_count < dbs_per_proc)
        {
          arts_signal_edt(edt, db_count * 2, mem_guids[addr_idx], DB_MODE_RO);         // data slot
          arts_signal_edt(edt, db_count * 2 + 1, mem_guids[addr_idx + 1], DB_MODE_RO); // stride slot
          addr_idx += 2;
          db_count += 1;
        }
        next = (next + 1) % arts_get_total_nodes();
        if (!i)
          clock_gettime(CLOCK_MONOTONIC, &global_start);
      }
      break;
      */
    }
    case 'r':
    {
      /*
      printf("Random access\n");
      uint64_t addr_idx = 0;
      uint64_t dbs_per_proc = num_dbs / num_procs;
      unsigned int next = arts_get_current_node();
      arts_guid_t done_guid = arts_edt_create(done, 0, NULL, num_procs, NULL);
      for (uint32_t i = 0; i < num_procs; i++)
      {
        uint64_t args[] = {access_size, dbs_per_proc, wait_time, done_guid, i};
        arts_guid_t edt = arts_edt_create(run_random, 5, args, dbs_per_proc, &(arts_hint_t){.route = next});
        uint64_t db_count = 0;
        while (db_count < dbs_per_proc)
        {
          arts_signal_edt(edt, db_count, mem_guids[addr_idx], DB_MODE_RO);
          addr_idx += 1;
          db_count += 1;
        }
        next = (next + 1) % arts_get_total_nodes();
        if (!i)
          clock_gettime(CLOCK_MONOTONIC, &global_start);
      }
      break;
      */
    }
    default:
      printf("Invalid access pattern. Defaulting to sequential\n");
      break;

      arts_shutdown();
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
