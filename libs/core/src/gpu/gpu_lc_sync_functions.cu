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
#include "arts/gpu/gpu_lc_sync_functions.cuh"

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu/gpu_route_table.h"
#include "arts/gpu/gpu_stream_buffer.h"
#include "arts/runtime/globals.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

// To use this lock the unlock must be an even number
unsigned int version_lock(arts_lc_meta_t *meta) {
  arts_writer_lock(meta->read_lock, meta->write_lock);
  return *meta->host_version;
}

bool try_version_lock(arts_lc_meta_t *meta) {
  return arts_writer_try_lock(meta->read_lock, meta->write_lock);
}

void version_unlock(arts_lc_meta_t *meta) {
  arts_atomic_add(meta->host_version, 2U);
  arts_writer_unlock(meta->write_lock);
}

void *make_lc_shadow_copy(struct arts_db_s *db) {
  unsigned int size = db->header.size;
  void *dest = (void *)(((char *)db) + size);
  struct arts_db_s *shadow_copy = (struct arts_db_s *)dest;

  arts_writer_lock(&db->reader, &db->writer);
  unsigned int host_version = db->version;
  if (!shadow_copy->version || host_version != shadow_copy->version) {
    memcpy(dest, (void *)db, size);
  }
  arts_writer_unlock(&db->writer);
  return dest;
}

inline void arts_print_db_meta_data(arts_lc_meta_t *db) {
  (void)db;
  ARTS_DEBUG(
      "guid: %lu ptr: %p data_size: %lu host_version: %u gpu_version: %u "
      "gpu_time_stamp: %u gpu: %d",
      db->guid, db->data, db->data_size, *db->host_version,
      *db->host_time_stamp, db->gpu_version, db->gpu_time_stamp, db->gpu);
}

void arts_memcpy_gpu_db(arts_lc_meta_t *host, arts_lc_meta_t *dev) {
  unsigned int host_version = version_lock(host);
  memcpy(host->data, dev->data, host->data_size);
  *host->host_time_stamp = dev->gpu_time_stamp;
  version_unlock(host);
}

void arts_get_latest_gpu_db(arts_lc_meta_t *host, arts_lc_meta_t *dev) {
  unsigned int host_version = version_lock(host);
  if (*host->host_time_stamp < dev->gpu_time_stamp) {
    memcpy(host->data, dev->data, host->data_size);
    host->gpu_version = dev->gpu_version;
    host->gpu_time_stamp = dev->gpu_time_stamp;
    *host->host_time_stamp = dev->gpu_time_stamp;
    host->gpu = dev->gpu;
  }
  version_unlock(host);
}

void arts_get_random_gpu_db(arts_lc_meta_t *host, arts_lc_meta_t *dev) {
  bool first_flag = (host->gpu == -1);
  bool random_flag = ((arts_thread_safe_random() & 1) == 0);
  if (first_flag || random_flag) {
    if (try_version_lock(host)) {
      memcpy(host->data, dev->data, host->data_size);
      host->gpu_version = dev->gpu_version;
      host->gpu_time_stamp = dev->gpu_time_stamp;
      *host->host_time_stamp = dev->gpu_time_stamp;
      host->gpu = dev->gpu;
      // if(!first_flag && random_flag)
      // arts_gpu_invalidate_route_tables(host->guid, (unsigned int) -1);
      version_unlock(host);
    }
  }
}

void arts_get_non_zeros_unsigned_int(arts_lc_meta_t *host,
                                     arts_lc_meta_t *dev) {
  unsigned int num_elem = host->data_size / sizeof(unsigned int);
  unsigned int *dst = (unsigned int *)host->data;
  unsigned int *src = (unsigned int *)dev->data;
  unsigned int host_version = version_lock(host);
  for (unsigned int i = 0; i < num_elem; i++) {
    ARTS_DEBUG("src: %u dest: %u", src[i], dst[i]);
    if (src[i]) {
      dst[i] = src[i];
    }
  }
  version_unlock(host);
}

void arts_get_min_db_unsigned_int(arts_lc_meta_t *host, arts_lc_meta_t *dev) {
  unsigned int count = 0;
  unsigned int count2 = 0;
  unsigned int num_elem = host->data_size / sizeof(unsigned int);
  unsigned int *dst = (unsigned int *)host->data;
  unsigned int *src = (unsigned int *)dev->data;
  unsigned int host_version = version_lock(host);
  for (unsigned int i = 0; i < num_elem; i++) {
    if (src[i] < dst[i]) {
      ARTS_DEBUG("src: %u dst: %u", src[i], dst[i]);
      dst[i] = src[i];
      count++;
    }
    if (src[i] != (unsigned int)-1) {
      count2++;
    }
  }
  ARTS_DEBUG("%lu %u %u", host->guid, count, count2);
  version_unlock(host);
}

void arts_add_db_unsigned_int(arts_lc_meta_t *host, arts_lc_meta_t *dev) {
  unsigned int count = 0;
  unsigned int num_elem = host->data_size / sizeof(unsigned int);
  unsigned int *dst = (unsigned int *)host->data;
  unsigned int *src = (unsigned int *)dev->data;
  unsigned int host_version = version_lock(host);
  for (unsigned int i = 0; i < num_elem; i++) {
    dst[i] += src[i];
  }
  ARTS_DEBUG("%lu %u", host->guid, count);
  version_unlock(host);
}

void arts_xor_db_uint64(arts_lc_meta_t *host, arts_lc_meta_t *dev) {
  unsigned int count = 0;
  unsigned int num_elem = host->data_size / sizeof(uint64_t);
  uint64_t *dst = (uint64_t *)host->data;
  uint64_t *src = (uint64_t *)dev->data;
  uint64_t host_version = version_lock(host);
  for (unsigned int i = 0; i < num_elem; i++) {
    ARTS_DEBUG("xor[%u]: %lu -- %lu = %lu", i, dst[i], src[i], dst[i] ^ src[i]);
    dst[i] ^= src[i];
    count++;
  }
  ARTS_DEBUG("%lu %u", host->guid, count);
  version_unlock(host);
}

/***********************************************************************/

__global__ void arts_copy_gpu_db(struct arts_db_s *sink,
                                 struct arts_db_s *src) {
  unsigned int *src_data = (unsigned int *)(src + 1);
  unsigned int *sink_data = (unsigned int *)(sink + 1);

  int index = (int)((blockIdx.x * blockDim.x) + threadIdx.x);
  sink_data[index] = src_data[index];
}

__global__ void arts_min_gpu_db_unsigned_int(struct arts_db_s *sink,
                                             struct arts_db_s *src) {
  unsigned int *src_data = (unsigned int *)(src + 1);
  unsigned int *sink_data = (unsigned int *)(sink + 1);

  int index = (int)((blockIdx.x * blockDim.x) + threadIdx.x);
  if (src_data[index] < sink_data[index]) {
    sink_data[index] = src_data[index];
  }
}

__global__ void arts_non_zero_gpu_db_unsigned_int(struct arts_db_s *sink,
                                                  struct arts_db_s *src) {
  unsigned int *src_data = (unsigned int *)(src + 1);
  unsigned int *sink_data = (unsigned int *)(sink + 1);

  int index = (int)((blockIdx.x * blockDim.x) + threadIdx.x);
  if (sink_data[index] > 0) {
    sink_data[index] = src_data[index];
  }
}

__global__ void arts_add_gpu_db_unsigned_int(struct arts_db_s *sink,
                                             struct arts_db_s *src) {
  unsigned int *src_data = (unsigned int *)(src + 1);
  unsigned int *sink_data = (unsigned int *)(sink + 1);

  int index = (int)((blockIdx.x * blockDim.x) + threadIdx.x);
  sink_data[index] += src_data[index];
}

__global__ void arts_xor_gpu_db_uint64(struct arts_db_s *sink,
                                       struct arts_db_s *src) {
  unsigned long long *src_data = (unsigned long long *)(src + 1);
  unsigned long long *sink_data = (unsigned long long *)(sink + 1);

  int index = (int)((blockIdx.x * blockDim.x) + threadIdx.x);
  sink_data[index] ^= src_data[index];
}

/***********************************************************************/

#define GPUGROUPSIZE 4
#define GPUNUMGROUP 2

void gpu_reduction_launch(int root, int a, int b, unsigned int *rem_mask,
                          arts_guid_t guid, unsigned int size,
                          arts_lc_sync_function_gpu_t fn_ptr) {
  if (a < 0 || b < 0) {
    return;
  }

  if (root != a && root != b) {
    ARTS_ERROR("LC reduction tree invalid: root %d not in {%d, %d}", root, a,
               b);
  }

  ARTS_DEBUG("A: %d B: %d -> Root: %d guid: %lu", a, b, root, guid);
  unsigned int to_remove = (root == a) ? (unsigned int)b : (unsigned int)a;
  *rem_mask &= ~(1 << to_remove);

  void *db_data =
      arts_gpu_route_table_lookup_db_res(guid, root, NULL, NULL, false);
  void *dst = (void *)(((char *)db_data) + size);
  ARTS_DEBUG("%d %p %p", root, db_data, dst);

  void *src = arts_gpu_route_table_lookup_db_res(guid, (int)to_remove, NULL,
                                                 NULL, false);
  ARTS_DEBUG("%d %p", to_remove, src);

  ARTS_DEBUG("src: %p dst: %p size: %u", src, dst, size);
  reduce_datafrom_gpus(dst, root, src, (int)to_remove, size, fn_ptr,
                       lc_sync_element_size[arts_node_info.gpu_lc_sync],
                       db_data);
}

void gpu_shadow_reduction_launch(int root, arts_guid_t guid, unsigned int size,
                                 arts_lc_sync_function_gpu_t fn_ptr) {
  void *sink =
      arts_gpu_route_table_lookup_db_res(guid, root, NULL, NULL, false);
  void *src = (void *)(((char *)sink) + size);

  do_reduction_now(root, sink, src, fn_ptr, sizeof(unsigned int), size);
}

void gpu_copy_launch(int root, int a, int b, bool src_shadow, bool dst_shadow,
                     arts_guid_t guid, unsigned int size) {

  if (a < 0 || b < 0) {
    return;
  }

  if (root != a && root != b) {
    ARTS_ERROR("LC reduction tree invalid: root %d not in {%d, %d}", root, a,
               b);
  }

  ARTS_DEBUG("A: %d B: %d -> Root: %d", a, b, root);
  unsigned int to_remove = (root == a) ? (unsigned int)b : (unsigned int)a;

  void *dst = arts_gpu_route_table_lookup_db_res(guid, root, NULL, NULL, false);
  if (dst_shadow) {
    dst = (void *)(((char *)dst) + size);
  }
  ARTS_DEBUG("%d %p", root, dst);

  void *src = arts_gpu_route_table_lookup_db_res(guid, (int)to_remove, NULL,
                                                 NULL, false);
  if (src_shadow) {
    src = (void *)(((char *)src) + size);
  }
  ARTS_DEBUG("%d %p", to_remove, src);

  ARTS_DEBUG("src: %p dst: %p size: %u", src, dst, size);
  copy_gputo_gpu(dst, root, src, (int)to_remove, size);
}

void find_roots(unsigned int local, int *roots) {
  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    roots[i] = -1;
  }

  // Make a mask of 4 bits (GPUGROUPSIZE)
  unsigned int mask = 0;
  for (unsigned int j = 0; j < GPUGROUPSIZE; j++) {
    unsigned int bit = 1 << j;
    mask |= bit;
  }

  // Assumes grid... Add shifted local mask with mask and or results
  unsigned int local_roots = (unsigned int)-1;
  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    unsigned int temp_local = local >> (i * GPUGROUPSIZE);
    unsigned int temp = mask & temp_local;
    local_roots &= temp;
  }

  // Recover the roots
  for (int i = 0; i < GPUGROUPSIZE; i++) {
    if (local_roots & (1 << i)) {
      ARTS_DEBUG("FOUND MATCHING ROOTS");
      for (unsigned int j = 0; j < GPUNUMGROUP; j++) {
        roots[j] = (int)(i + (j * GPUGROUPSIZE));
      }
      return;
    }
  }

  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    // ARTS_INFO("i: %u", i);
    for (unsigned int j = 0; j < GPUGROUPSIZE; j++) {
      unsigned int bit = (i * GPUGROUPSIZE) + j;
      // ARTS_INFO("bit: %u", bit);
      if (local & (1 << bit)) {
        roots[i] = (int)bit;
        break;
      }
    }
  }
}

typedef struct {
  int a;
  int b;
  int root;
  int level;
} trav_t;

void add_to_trav(int root, int a, int b, unsigned int level, unsigned int *size,
                 trav_t *ds, unsigned int *max_level) {
  if (a < 0 || b < 0) {
    return;
  }

  unsigned int index = (*size);
  *size = *size + 1;
  ds[index].a = a;
  ds[index].b = b;
  ds[index].root = root;
  ds[index].level = (int)level;

  *max_level = (*max_level < level) ? level : *max_level;
}

int gpu_tree_reduction_rec(int root, unsigned int start, unsigned int stop,
                           unsigned int mask, unsigned int level,
                           unsigned int *list_size, trav_t *list,
                           unsigned int *max_level) {
  int local_root = -1;
  // ARTS_INFO("root: %u start: %u stop: %u", root, start, stop);
  int gpu_id[2] = {(int)start, (int)stop};

  if (stop - start > 1) // Recursive call
  {
    unsigned int middle = (1 + stop - start) / 2;
    gpu_id[0] = gpu_tree_reduction_rec(root, start, start + middle - 1, mask,
                                       level + 1, list_size, list, max_level);
    gpu_id[1] = gpu_tree_reduction_rec(root, start + middle, stop, mask,
                                       level + 1, list_size, list, max_level);
  }

  bool start_found = (gpu_id[0] >= 0) && ((mask & (1 << gpu_id[0])) != 0);
  bool stop_found = (gpu_id[1] >= 0) && ((mask & (1 << gpu_id[1])) != 0);

  if (start_found && stop_found) // Both are in the mask
  {
    if (root == gpu_id[0] || root == gpu_id[1]) {
      local_root = root;
    } else {
      local_root = gpu_id[0]; // This is the min
    }
  } else if (start_found && !stop_found) // Only start is in the mask
  {
    gpu_id[1] = -1;
    local_root = gpu_id[0];
  } else if (!start_found && stop_found) // Only stop is in the mask
  {
    gpu_id[0] = -1;
    local_root = gpu_id[1];
  } else // Neither start or stop is in the mask
  {
    gpu_id[1] = -1;
    gpu_id[0] = -1;
    // local_root = -1;
  }

  add_to_trav(local_root, gpu_id[0], gpu_id[1], level, list_size, list,
              max_level);
  return local_root;
}

void gpu_tree_reduction_start(unsigned int mask, unsigned int *list_size,
                              trav_t *list, unsigned int *max_level) {
  int root[GPUNUMGROUP];
  find_roots(mask, root);
  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    ARTS_DEBUG("Root[%d]: %d", i, root[i]);
    gpu_tree_reduction_rec(root[i], i * GPUGROUPSIZE,
                           ((i + 1) * GPUGROUPSIZE) - 1, mask, 2, list_size,
                           list, max_level);
  }
  add_to_trav(root[0], root[0], root[1], 1, list_size, list, max_level);
}

unsigned int gpu_tree_reduction(unsigned int mask, arts_guid_t guid,
                                unsigned int db_size,
                                arts_lc_sync_function_gpu_t db_fn) {
  ARTS_DEBUG("mask: %u", mask);
  unsigned int max_level = 0;
  unsigned int list_size = 0;
  trav_t list[GPUNUMGROUP * GPUGROUPSIZE];

  gpu_tree_reduction_start(mask, &list_size, list, &max_level);

  unsigned int rem_mask = mask;

  for (unsigned int i = max_level; i > 0; i--) {
    for (unsigned int j = 0; j < list_size; j++) {
      if (list[j].level == (int)i) {
        gpu_reduction_launch(list[j].root, list[j].a, list[j].b, &rem_mask,
                             guid, db_size, db_fn);
      }
    }
  }
  ARTS_DEBUG("rem_mask: %u", rem_mask);
  return rem_mask;
}

/***********************************************************/

bool check_max(unsigned int current_size, unsigned int *visited,
               unsigned int *max_size, unsigned int *max_visited,
               unsigned int cycle_size) {
  if (*max_size < current_size) {
    *max_size = current_size;
    memcpy(max_visited, visited, sizeof(unsigned int) * current_size);
    return (current_size == cycle_size) &&
           (max_visited[0] == max_visited[cycle_size - 1]);
  }
  return false;
}

extern bool **gpu_adj_list;
unsigned int gpu_depth_first_rec(unsigned int vertex, unsigned int cycle_size,
                                 unsigned int mask, unsigned int current,
                                 unsigned int *visited, unsigned int *max_size,
                                 unsigned int *max_visited) {
  unsigned int order = arts_get_total_gpus();
  visited[current++] = vertex; // Record order visited

  bool ret = check_max(current, visited, max_size, max_visited, cycle_size);

  unsigned int temp = ~(1 << vertex); // Mark off list
  mask &= temp;

  if (current + 1 == cycle_size) {
    // This means the next iteration is the final
    // one... Lets look for a cycle to make a ring
    mask |= 1 << visited[0];
  }

  if (current < cycle_size) {
    for (unsigned int i = 0; i < order; i++) {
      if ((mask & (1 << i)) && gpu_adj_list[vertex][i]) {
        ARTS_INFO("%u -> %u", vertex, i);
        if (gpu_depth_first_rec(i, cycle_size, mask, current, visited, max_size,
                                max_visited)) {
          return true;
        }
      }
    }
  }
  return ret;
}

unsigned int *gpu_depth_first(unsigned int mask, unsigned int *max_size) {
  unsigned int *ret = NULL;
  unsigned int cycle_size = 1; // Add one for the backedge
  for (unsigned int i = 0; i < sizeof(mask) * 8; i++) {
    if (mask & (1 << i)) {
      cycle_size++;
    }
  }

  unsigned int *visited =
      (unsigned int *)arts_calloc(cycle_size, sizeof(unsigned int));
  unsigned int *max_visited =
      (unsigned int *)arts_calloc(cycle_size, sizeof(unsigned int));
  for (unsigned int i = 0; i < arts_get_total_gpus(); i++) {
    if (mask & (1 << i)) {
      ARTS_INFO("i: %u", i);
      if (gpu_depth_first_rec(i, cycle_size, mask, 0, visited, max_size,
                              max_visited)) {
        ret = max_visited;
        break;
      }
    }
  }
  arts_free(visited);
  if (!ret) {
    arts_free(max_visited);
  }
  return ret;
}

bool gpu_ring_reduction(unsigned int mask, unsigned int guid,
                        unsigned int db_size,
                        arts_lc_sync_function_gpu_t fn_ptr) {
  // unsigned int rem_mask = mask;
  unsigned int cycle_size = 0;
  unsigned int *cycle = gpu_depth_first(mask, &cycle_size);
  if (cycle && cycle_size > 1) {
    unsigned int num_gpus = cycle_size - 1;
    ARTS_INFO("Cycle Size:%u", cycle_size);
    for (unsigned int i = 0; i < 1; i++) {
      for (unsigned int j = 1; j < cycle_size; j++) {
        gpu_copy_launch((int)cycle[j], (int)cycle[j - 1], (int)cycle[j],
                        (i != 0), true, guid, db_size);
      }

      for (unsigned int j = 0; j < num_gpus; j++) {
        gpu_shadow_reduction_launch((int)cycle[j], guid, db_size, fn_ptr);
      }
    }
    return true;
  }
  return false;
}

void gpu_lc_invalidate(unsigned int mask, arts_guid_t guid) {
  for (unsigned int i = 0; i < arts_get_total_gpus(); i++) {
    if (mask & (1 << i)) {
      arts_gpu_invalidate_on_route_table(guid, i);
      arts_gpu_route_table_return_db(guid, true, i);
    }
  }
}

unsigned int gpu_lc_return_db(unsigned int mask, arts_guid_t guid) {
  unsigned int rem_mask = 0;
  for (unsigned int i = 0; i < arts_get_total_gpus(); i++) {
    if (mask & (1 << i)) {
      if (!rem_mask && i == 2) {
        rem_mask = 1 << i;
      } else {
        arts_gpu_route_table_return_db(guid, false, i);
      }
    }
  }
  return rem_mask;
}

unsigned int gpu_lc_reduce(arts_guid_t guid, struct arts_db_s *db,
                           arts_lc_sync_function_gpu_t db_fn, bool *copy_only) {
  *copy_only = false;
  unsigned int rem_mask = 0;
  unsigned int size = db->header.size;
  // struct arts_db_s *shadow_copy = (struct arts_db_s *)(((char *)db) + size);

  arts_writer_lock(&db->reader, &db->writer);
  unsigned int mask = arts_gpu_lookup_db_fix(guid);
  if (mask) {
    // RING IS NOT WORKING...
    //  if(db->version == shadow_copy->version && gpu_ring_reduction(mask, guid,
    //  size, db_fn))
    //  {
    //      rem_mask = gpu_lc_return_db(mask, guid);
    //      *copy_only = true;
    //  }
    //  else
    {
      rem_mask = gpu_tree_reduction(mask, guid, size, db_fn);
      gpu_lc_invalidate(mask & ~rem_mask, guid);
    }
  }
  rem_mask = mask;
  arts_writer_unlock(&db->writer);
  return rem_mask;
}