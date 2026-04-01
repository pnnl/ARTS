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
#ifndef STREAM_UTIL_H
#define STREAM_UTIL_H
#ifdef __cplusplus
extern "C" {
#endif

#include <float.h>
#include <limits.h>

#include <sys/time.h>

#include "arts.h"

#define CXL_DB 1
// #define N 20000000
#define N (1 << 20) 
// #define N 20
#define TILESIZE 131072
#define NTIMES 20
// #define NTIMES 2
// #define NTIMES 10
#define OFFSET 0

#define M 20

#define THREADSPERBLOCK 1

#define HLINE "-------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
    
    
arts_guid_t get_total_owned_tiles();
arts_guid_t get_done_guid();
    
// void launch_2_kernel_edt(arts_edt_t fun_ptr, unsigned int tile_size,
                    //   unsigned int total_size, double scalar,
                    //   arts_guid_t *a_guid, arts_guid_t *b_guid);

void launch_2_kernel_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]);

// void launch_3_kernel_edt(arts_edt_t fun_ptr, unsigned int tile_size,
                    //   unsigned int total_size, double scalar,
                    //   arts_guid_t *a_guid, arts_guid_t *b_guid,
                    //   arts_guid_t *c_guid);

void launch_3_kernel_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]);

void force_acquire_cxl_guids(unsigned int num_tiles, 
                      arts_guid_t *a_guid, arts_guid_t *b_guid,
                      arts_guid_t *c_guid); 

void update_kernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            arts_edt_dep_t depv[]);

void populate_tile_range();
bool tile_in_range(uint64_t i);
unsigned int get_tile_owner(uint64_t i);                   
void start_timer();
void end_timer(arts_edt_dep_t to_signal);

int check_tick();
double my_second();
void check_stream_results(unsigned int tile_size, unsigned int total_size,
                        double **a_tiles, double **b_tiles, double **c_tiles);

#ifdef __cplusplus
}
#endif

#endif