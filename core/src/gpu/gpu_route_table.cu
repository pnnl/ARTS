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
#include "arts/gpu/gpu_route_table.h"

#include "arts.h"
#include "arts/gpu/gpu_stream.h"
#include "arts/introspection/metrics.h"
#include "arts/runtime/globals.h"
#include "arts/system/arts_print.h"
#include "arts/utils/atomics.h"

// Use to keep ordering for LC accesses
volatile unsigned int gpuNodeOrder = 0;

// Must be thread local
__thread uint64_t gpuItemSizeBypass = 0;

void setGpuItem(arts_route_item_t *item, void *data) {
  ARTS_DEBUG("gpuItemSizeBypass: %lu", gpuItemSizeBypass);
  arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)item->data;
  wrapper->realData = data;
  wrapper->size = gpuItemSizeBypass;
  gpuItemSizeBypass = 0;
}

unsigned int setGpuTimestamp(volatile unsigned int *time_stamp) {
  unsigned int newTimeStamp = arts_atomic_add(&gpuNodeOrder, 1);
  unsigned int oldTimeStamp = *time_stamp;
  while (oldTimeStamp < newTimeStamp) {
    if (arts_atomic_cswap(time_stamp, oldTimeStamp, newTimeStamp) == oldTimeStamp)
      return newTimeStamp;
    oldTimeStamp = *time_stamp;
  }
  return oldTimeStamp;
}

arts_route_table_t *arts_gpu_new_route_table(unsigned int route_table_size,
                                       unsigned int shift) {
  unsigned int totalElems = COLLISION_RESOLVES * route_table_size;
  arts_gpu_route_table_t *gpu_route_table =
      (arts_gpu_route_table_t *)arts_calloc(1, sizeof(arts_gpu_route_table_t));
  gpu_route_table->routingTable.data = (arts_route_item_t *)arts_calloc_align(
      totalElems, sizeof(arts_route_item_t), 16);
  gpu_route_table->routingTable.size = route_table_size;
  gpu_route_table->routingTable.shift = shift;
  gpu_route_table->routingTable.setFunc = setGpuItem;
  gpu_route_table->routingTable.freeFunc = free_gpu_item;
  gpu_route_table->routingTable.newFunc = arts_gpu_new_route_table;

  gpu_route_table->wrappers =
      (arts_item_wrapper_t *)arts_calloc(totalElems, sizeof(arts_item_wrapper_t));
  for (unsigned int i = 0; i < totalElems; i++)
    gpu_route_table->routingTable.data[i].data = &gpu_route_table->wrappers[i];

  return &gpu_route_table->routingTable;
}

uint64_t arts_gpu_lookup_db(arts_guid_t key) {
  uint64_t ret = 0;
  for (int i = 0; i < arts_node_info.gpu; ++i) {
    arts_route_table_t *gpu_route_table = arts_node_info.gpu_route_table[i];
    arts_route_item_t *location =
        arts_route_table_search_for_key(gpu_route_table, key, AVAILABLE_KEY);
    if (location)
      ret |= 1 << i;
  }
  return ret;
}

unsigned int arts_gpu_lookup_db_fix(arts_guid_t key) {
  unsigned int ret = 0;
  for (int i = 0; i < arts_node_info.gpu; ++i) {
    arts_route_table_t *gpu_route_table = arts_node_info.gpu_route_table[i];
    int dummy_rank;
    unsigned int *internalTouched;
    arts_route_item_t *location = NULL;
    location = (arts_route_item_t *)internal_route_table_lookup_db(
        gpu_route_table, key, &dummy_rank, &internalTouched);
    if (location) {
      // arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)location;
      ret |= (1 << i);
    }
  }
  return ret;
}

void *arts_gpu_route_table_add_item_race(void *item, uint64_t size, arts_guid_t key,
                                   unsigned int gpu_id) {
  // This is a bypass thread local variable to make the api nice...
  gpuItemSizeBypass = size;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  bool ret;
  arts_route_item_t *entry = internal_route_table_add_item_race(
      &ret, route_table, item, key, arts_global_rank_id, true, true, 0);
  arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)entry->data;
  setGpuTimestamp(&wrapper->time_stamp);
  return (void *)wrapper->realData;
}

arts_item_wrapper_t *arts_gpu_route_table_reserve_item_race(bool *added, uint64_t size,
                                                    arts_guid_t key,
                                                    unsigned int gpu_id,
                                                    bool add_to_use) {
  // This is a bypass thread local variable to make the api nice...
  gpuItemSizeBypass = size;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  arts_route_item_t *entry = internal_route_table_add_item_race(
      added, route_table, NULL, key, arts_global_rank_id, true, true,
      add_to_use ? 1 : 0);
  arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)entry->data;
  setGpuTimestamp(&wrapper->time_stamp);
  return wrapper;
}

void *arts_gpu_route_table_add_item_to_delete_race(void *item, uint64_t size,
                                           arts_guid_t key, unsigned int gpu_id) {
  // This is a bypass thread local variable to make the api nice...
  gpuItemSizeBypass = size;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  arts_route_item_t *entry = internal_route_table_add_deleted_item_race(
      route_table, item, key, arts_global_rank_id);
  arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)entry->data;
  setGpuTimestamp(&wrapper->time_stamp);
  return (void *)wrapper->realData;
}

void *arts_gpu_route_table_lookup_db_res(arts_guid_t key, int gpu_id,
                                   unsigned int *touched,
                                   unsigned int *time_stamp, bool res) {
  void *ret = NULL;
  int dummy_rank;
  unsigned int *internalTouched;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  arts_item_wrapper_t *wrapper = NULL;
  if (res)
    wrapper = (arts_item_wrapper_t *)internal_route_table_lookup_db(
        route_table, key, &dummy_rank, &internalTouched);
  else {
    arts_route_item_t *temp = (arts_route_item_t *)arts_route_table_search_for_key(
        route_table, key, AVAILABLE_KEY);
    wrapper = (temp) ? (arts_item_wrapper_t *)temp->data : NULL;
  }

  if (wrapper) {
    if (res) {
      if (time_stamp)
        *time_stamp = setGpuTimestamp(&wrapper->time_stamp);
      if (touched)
        *touched = internal_inc_db_version(internalTouched);
    }
    ret = (void *)wrapper->realData;
    ARTS_DEBUG("Wrapper: %p %p", wrapper, wrapper->realData);
  }
  return ret;
}

void *arts_gpu_route_table_lookup_db(arts_guid_t key, int gpu_id,
                                unsigned int *touched,
                                unsigned int *time_stamp) {
  return arts_gpu_route_table_lookup_db_res(key, gpu_id, touched, time_stamp, true);
}

bool arts_gpu_route_table_return_db(arts_guid_t key, bool mark_to_delete,
                               unsigned int gpu_id) {
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  return internal_route_table_return_db(route_table, key, mark_to_delete, false);
}

bool arts_gpu_invalidate_route_tables(arts_guid_t key, unsigned int keep_on_this_gpu) {
  bool ret = 0;
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    if (i != keep_on_this_gpu)
      ret |= internal_route_table_remove_item(arts_node_info.gpu_route_table[i], key);
  }
  return ret;
}

bool arts_gpu_invalidate_on_route_table(arts_guid_t key, unsigned int gpu_id) {
  return internal_route_table_remove_item(arts_node_info.gpu_route_table[gpu_id], key);
}

volatile unsigned int gpuReader = 0;
volatile unsigned int gpuWriter = 0;

void gpu_gc_read_lock() {
  while (1) {
    while (gpuWriter)
      ;
    arts_atomic_fetch_add(&gpuReader, 1U);
    if (gpuWriter == 0)
      break;
    arts_atomic_sub(&gpuReader, 1U);
  }
}

void gpu_gc_read_unlock() { arts_atomic_sub(&gpuReader, 1U); }

void gpu_gc_write_lock() {
  while (arts_atomic_cswap(&gpuWriter, 0U, 1U) != 0U)
    ;
  while (gpuReader)
    ;
}

void gpu_gc_write_unlock() { arts_atomic_swap(&gpuWriter, 0U); }

/*This takes three parameters to regulate what is deleted.  This will only clean
up DBs!
1.  size_to_clean - this is the desired space to clean up.  The gc will continue
untill it it reaches this size or it has made a full pass across the RT. Passing
-1 will make the gc clean up the entire RT.
2.  clean_zeros - this flag indicates if we should delete data that is not being
used by anyone. Will delete up to size_to_clean.
3.  gpu_id - the id of which GPU this RT belongs.  This is the contiguous id [0 -
num_gpus-1]. Pass -1 for a host RT. Returns the size of the memory freed!
*/
uint64_t arts_gpu_clean_up_route_table(unsigned int size_to_clean, bool clean_zeros,
                                  unsigned int gpu_id) {
  uint64_t freedSize = 0;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  arts_gpu_route_table_t *gpu_route_table = (arts_gpu_route_table_t *)route_table;
  // This is a lock to make sure LC sync works
  gpu_gc_read_lock();
  // Only one person can be running the gc at a time...
  if (arts_try_lock(&gpu_route_table->gcLock)) {
    arts_route_table_iterator_t iter;
    arts_reset_route_table_iterator(&iter, route_table);

    arts_route_item_t *item = arts_route_table_iterate(&iter);
    while (item && freedSize < size_to_clean) {
      // arts_print_item(item);
      arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)item->data;
      uint64_t size = wrapper->size;
      if (IS_DEL(item->lock)) {
        uint64_t comp_val = (AVAILABLE_ITEM | DELETE_ITEM);
        uint64_t new_val = (AVAILABLE_ITEM | DELETE_ITEM) + 1;
        uint64_t old_val = arts_atomic_cswap_u64(&item->lock, comp_val, new_val);
        if ((comp_val == old_val) && dec_item(route_table, item))
          freedSize += size;
      } else if (clean_zeros && !GET_COUNT(item->lock)) {
        uint64_t comp_val = AVAILABLE_ITEM;
        uint64_t new_val = (AVAILABLE_ITEM | DELETE_ITEM) + 1;
        uint64_t old_val = arts_atomic_cswap_u64(&item->lock, comp_val, new_val);
        if ((comp_val == old_val) && dec_item(route_table, item))
          freedSize += size;
      }
      item = arts_route_table_iterate(&iter);
    }
    ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_GC, ARTS_METRIC_THREAD, 1);
    ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_GCBW, ARTS_METRIC_THREAD, freedSize);
    arts_unlock(&gpu_route_table->gcLock);
  }
  gpu_gc_read_unlock();
  return freedSize;
}

uint64_t arts_gpu_free_all(unsigned int gpu_id) {
  uint64_t freedSize = 0;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];

  arts_route_table_iterator_t iter;
  arts_reset_route_table_iterator(&iter, route_table);

  arts_route_item_t *item = arts_route_table_iterate(&iter);
  while (item) {
    arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)item->data;
    freedSize += wrapper->size;
    free_gpu_item(item);
    item = arts_route_table_iterate(&iter);
  }
  return freedSize;
}