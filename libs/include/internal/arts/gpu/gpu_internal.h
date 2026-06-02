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

/**
 * @file gpu_internal.h
 * @brief Internal GPU runtime structures and functions.
 *
 * This header is for runtime-internal use only — included from .cu files
 * that are compiled with NVCC.  User code should include arts/gpu.h instead.
 */
#ifndef ARTS_GPU_INTERNAL_H
#define ARTS_GPU_INTERNAL_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/gpu/gpu_stream.h"
#include "arts/runtime_types.h"

/**
 * @brief Internal GPU EDT descriptor.
 *
 * Wraps the base @c arts_edt_s with GPU-specific scheduling metadata.
 * Allocated as a single contiguous block: [arts_gpu_edt_t | paramv | depv |
 * modes].
 */
typedef struct {
  struct arts_edt_s wrapper_edt;
  arts_dim3_t grid;
  arts_dim3_t block;
  int gpu_to_run_on;
  arts_guid_t end_guid;
  arts_guid_t data_guid;
  uint32_t slot;
  bool passthrough;
  bool lib;
} arts_gpu_edt_t;

/* --- Internal GPU runtime functions --- */

void arts_gpu_host_wrap_up(void *edt_packet, arts_guid_t to_signal,
                           uint32_t slot, arts_guid_t data_guid);
void arts_run_gpu(void *edt_packet, arts_gpu_t *arts_gpu);
bool arts_gpu_scheduler_loop(void);

/* --- GPU-placement policy (gpu_placement.cu) --- */

typedef int (*locality_t)(void *edt);
typedef int (*fit_t)(uint64_t mask, uint64_t size, unsigned int total_threads);

extern locality_t locality_scheme[];
extern locality_t locality; /* selected locality scheme */
extern fit_t fit_scheme[];
extern fit_t fit; /* selected fit scheme */

/* Per-worker GC backpressure flag: set by a failed reservation in
 * gpu_placement.cu, consumed by the demand scheduler loop. */
extern ARTS_THREAD_LOCAL unsigned int run_gc_flag;

int arts_reserve_edt_required_gpu(int *gpu, void *edt_packet);

/* --- LC sync helpers (internal only — public decls in arts/gpu.h) --- */

void internal_lc_sync_gpu(arts_guid_t acq_guid, struct arts_db_s *db);

#ifdef __cplusplus
}
#endif
#endif /* ARTS_GPU_INTERNAL_H */
