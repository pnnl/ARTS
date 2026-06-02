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
#ifndef ARTS_GPU_GPULC_H
#define ARTS_GPU_GPULC_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime_types.h"

typedef struct {
  uint64_t guid;
  void *data;
  uint64_t data_size;
  volatile unsigned int *host_version;
  unsigned int *host_time_stamp;
  unsigned int gpu_version;
  unsigned int gpu_time_stamp;
  int gpu;
  volatile unsigned int *read_lock;
  volatile unsigned int *write_lock;
} arts_lc_meta_t;

typedef void (*arts_lc_sync_function_t)(arts_lc_meta_t *host,
                                        arts_lc_meta_t *dev);
extern arts_lc_sync_function_t lc_sync_function[];

typedef void (*arts_lc_sync_function_gpu_t)(struct arts_db_s *src,
                                            struct arts_db_s *dst);
extern arts_lc_sync_function_gpu_t lc_sync_function_gpu[];

extern unsigned int lc_sync_element_size[];

void *make_lc_shadow_copy(struct arts_db_s *db);

void arts_memcpy_gpu_db(arts_lc_meta_t *host, arts_lc_meta_t *dev);
void arts_get_latest_gpu_db(arts_lc_meta_t *host, arts_lc_meta_t *dev);
void arts_get_random_gpu_db(arts_lc_meta_t *host, arts_lc_meta_t *dev);
void arts_get_non_zeros_unsigned_int(arts_lc_meta_t *host, arts_lc_meta_t *dev);
void arts_get_min_db_unsigned_int(arts_lc_meta_t *host, arts_lc_meta_t *dev);
void arts_add_db_unsigned_int(arts_lc_meta_t *host, arts_lc_meta_t *dev);
void arts_xor_db_uint64(arts_lc_meta_t *host, arts_lc_meta_t *dev);

unsigned int gpu_lc_reduce(arts_guid_t guid, struct arts_db_s *db,
                           arts_lc_sync_function_gpu_t db_fn, bool *copy_only);

__global__ void arts_copy_gpu_db(struct arts_db_s *src, struct arts_db_s *dst);
__global__ void arts_min_gpu_db_unsigned_int(struct arts_db_s *src,
                                             struct arts_db_s *dst);
__global__ void arts_non_zero_gpu_db_unsigned_int(struct arts_db_s *src,
                                                  struct arts_db_s *dst);
__global__ void arts_add_gpu_db_unsigned_int(struct arts_db_s *src,
                                             struct arts_db_s *dst);
__global__ void arts_xor_gpu_db_uint64(struct arts_db_s *sink,
                                       struct arts_db_s *src);

#ifdef __cplusplus
}
#endif

#endif
