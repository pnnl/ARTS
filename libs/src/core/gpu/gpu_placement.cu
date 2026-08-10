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

// GPU-placement policy: locality/fit selection schemes and per-EDT GPU
// reservation.  Decides which GPU an EDT runs on; the stream module performs
// the actual data movement and kernel launch.
#include "arts/gpu/gpu_stream.h"

#include "arts/db.h"
#include "arts/gas/guid.h"
#include "arts/gpu/gpu_internal.h"
#include "arts/gpu/gpu_route_table.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/random.h"

int random(void *edt_packet);
int all_or_nothing(void *edt_packet);
int atleast_one(void *edt_packet);
int hash_on_db_zero(void *edt_packet);
int hash_largest(void *edt_packet);
int first_fit(uint64_t mask, uint64_t size, unsigned int total_threads);
int best_fit(uint64_t mask, uint64_t size, unsigned int total_threads);
int worst_fit(uint64_t mask, uint64_t size, unsigned int total_threads);
int round_robin_fit(uint64_t mask, uint64_t size, unsigned int total_threads);
bool try_reserve(int gpu, uint64_t size, unsigned int threads);

/* locality_t / fit_t typedefs are declared in gpu_internal.h. */

locality_t locality_scheme[] = {random, all_or_nothing, atleast_one,
                                hash_on_db_zero, hash_largest};

locality_t locality; // Locality function ptr

fit_t fit_scheme[] = {first_fit, best_fit, worst_fit, round_robin_fit};

fit_t fit; // Fit function ptr

ARTS_THREAD_LOCAL unsigned int run_gc_flag = 0;

bool try_reserve(int gpu, uint64_t size, unsigned int threads) {
  (void)threads;
  arts_gpu_t *arts_gpu = &arts_gpus[gpu];
  ARTS_DEBUG("Trying to reserve %lu of available %lu on GPU[%d]\n", size,
             arts_gpu->avail_global_mem, arts_gpu->device);
  // if(arts_atomic_fetch_add(&arts_gpu->available_threads, threads) < 1024)
  {
    if (arts_atomic_fetch_add(&arts_gpu->available_edt_slots, 1U) <
        arts_node_info.gpu_max_edts) {
      volatile uint64_t avail_size = arts_gpu->avail_global_mem;
      while (avail_size >= size) {
        if (arts_atomic_cswap_u64(&arts_gpu->avail_global_mem, avail_size,
                                  avail_size - size)) {
          run_gc_flag = 0;
          return true;
        }
        avail_size = arts_gpu->avail_global_mem;
      }
      run_gc_flag = gpu + 1;
    }
    arts_atomic_sub(&arts_gpu->available_edt_slots, 1U);
  }
  // arts_atomic_sub(&arts_gpu->available_threads, threads);
  ARTS_DEBUG("Failed Avail threads: %u + %u\n", arts_gpu->available_threads,
             threads);
  ARTS_DEBUG("Failed to reserve %lu of available %lu on GPU[%d]\n", size,
             arts_gpu->avail_global_mem, arts_gpu->device);
  return false;
}

int first_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  /* Unsigned from the shared helper: a raw jrand48 draw is signed, and the
   * rotation below must start from a uniform offset. */
  unsigned int random = (unsigned int)arts_thread_safe_random();
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask & check_mask) {
      if (try_reserve(index, size, total_threads)) {
        ARTS_DEBUG("Reserved Successfully on %u\n", index);
        return index;
      }
    }
  }
  return -1;
}

int round_robin_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  static volatile unsigned int next = 0;
  unsigned int start = arts_atomic_fetch_add(&next, 1U);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + start) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask & check_mask) {
      if (try_reserve(index, size, total_threads)) {
        ARTS_DEBUG("Reserved Successfully on %u\n", index);
        return index;
      }
    }
  }
  return -1;
}

int best_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int selected_gpu = -1;
  uint64_t selected_gpu_avail_size = 0;
  /* Unsigned from the shared helper: a raw jrand48 draw is signed, and the
   * rotation below must start from a uniform offset. */
  unsigned int random = (unsigned int)arts_thread_safe_random();
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask & check_mask) {
      if (selected_gpu != -1) {
        if (arts_gpus[index].avail_global_mem - size >
            selected_gpu_avail_size) {
          continue;
        }
      }
      if (try_reserve(index, size, total_threads)) {
        // If successful relinquish previous allocation (if any).
        if (selected_gpu != -1) {
          arts_atomic_add_u64(&arts_gpus[selected_gpu].avail_global_mem, size);
        }
        selected_gpu = index;
        selected_gpu_avail_size = arts_gpus[index].avail_global_mem;
      }
    }
  }
  return selected_gpu;
}

int worst_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int selected_gpu = -1;
  uint64_t selected_gpu_avail_size = 0;
  /* Unsigned from the shared helper: a raw jrand48 draw is signed, and the
   * rotation below must start from a uniform offset. */
  unsigned int random = (unsigned int)arts_thread_safe_random();
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask & check_mask) {
      if (selected_gpu != -1) {
        if (arts_gpus[index].avail_global_mem - size <
            selected_gpu_avail_size) {
          continue;
        }
      }
      if (try_reserve(index, size, total_threads)) {
        // If successful relinquish previous allocation (if any).
        if (selected_gpu != -1) {
          arts_atomic_add_u64(&arts_gpus[selected_gpu].avail_global_mem, size);
        }
        selected_gpu = index;
        selected_gpu_avail_size = arts_gpus[index].avail_global_mem;
      }
    }
  }
  return selected_gpu;
}

uint64_t get_db_size_needed(uint32_t depc, arts_edt_dep_t *depv) {
  uint64_t size = 0;
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      struct arts_db_s *db = (struct arts_db_s *)depv[i].ptr - 1;
      size += arts_db_total_size(db);
      if (db->db_type == ARTS_DB_GPU) {
        size += arts_db_total_size(db);
      }
    }
  }
  return size;
}

int random(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  uint64_t mask = ~0;
  return fit(mask, size, total_threads);
}

int all_or_nothing(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  // Intersection of every dependency's GPU-presence set: a candidate GPU must
  // hold ALL dependencies.  The identity element for intersection is the full
  // set (all ones), so seed with ~0 and AND each lookup in; an empty depc keeps
  // the full set, and any dep resident nowhere zeroes the mask.
  uint64_t mask = ~(uint64_t)0;
  for (unsigned int i = 0; i < depc; ++i) {
    mask &= arts_gpu_lookup_db(depv[i].guid);
  }

  ARTS_DEBUG("Mask: %p\n", mask);

  if (mask) { // All DBs in GPU
    return fit(mask, size,
               total_threads); // No need to fit since all Dbs are in a GPU
  }
  return random(edt_packet);
}

int atleast_one(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  uint64_t mask = 0;
  for (unsigned int i = 0; i < depc; ++i) {
    mask |= arts_gpu_lookup_db(depv[i].guid);
  }

  ARTS_DEBUG("Mask: %p\n", mask);

  if (mask) { // At least one DB in GPU
    return fit(mask, size, total_threads);
  }
  return random(edt_packet);
}

int hash_on_db_zero(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  uint64_t key = (depv[0].guid) ? arts_guid_get_key(depv[0].guid) : 0;
  int index = (int)(key % (uint64_t)arts_node_info.gpu);
  if ((unsigned int)index >= arts_node_info.gpu) {
    ARTS_ERROR("GPU stream hash failed: index %d >= gpu count %u", index,
               arts_node_info.gpu);
  }
  ARTS_DEBUG("HASH: %lu %d\n", depv[0].guid, index);
  if (try_reserve(index, size, total_threads)) {
    return index;
  }
  return -1;
}

int hash_largest(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  // uint64_t mask = 0;
  uint64_t largest = 0;
  for (unsigned int i = 0; i < depc; ++i) {
    uint64_t key = (depv[i].guid) ? arts_guid_get_key(depv[i].guid) : 0;
    largest = (key > largest) ? key : largest;
  }

  int index = (int)(largest % (uint64_t)arts_node_info.gpu);
  if (try_reserve(index, size, total_threads)) {
    ARTS_DEBUG("Index: %d\n", index);
    return index;
  }
  return -1;
}

int arts_reserve_edt_required_gpu(int *gpu, void *edt_packet) {
  bool ret = false;
  *gpu = -1;
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  if (edt->gpu_to_run_on > -1) {
    // Size to be allocated on the GPU
    uint64_t size = (sizeof(uint64_t) * paramc) +
                    (sizeof(arts_edt_dep_t) * depc) +
                    get_db_size_needed(depc, depv);
    if (try_reserve(edt->gpu_to_run_on, size, total_threads)) {
      *gpu = edt->gpu_to_run_on;
      ret = true;
    }
  }
  return ret;
}

arts_gpu_t *arts_find_gpu(void *data) {
  arts_gpu_t *ret = NULL;
  int gpu;
  if (!arts_reserve_edt_required_gpu(&gpu, data)) {
    gpu = locality(data);
  }
  ARTS_DEBUG("Choosing gpu: %d\n", gpu);
  if (gpu > -1 && gpu < (int)arts_node_info.gpu) {
    ret = &arts_gpus[gpu];
  }

  return ret;
}
