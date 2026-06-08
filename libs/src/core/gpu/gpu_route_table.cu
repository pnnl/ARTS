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
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

// Use to keep ordering for LC accesses
volatile unsigned int gpu_node_order = 0;

// Must be thread local
ARTS_THREAD_LOCAL uint64_t gpu_item_size_bypass = 0;

/* Peek the persistent wrapper published in a GPU-mirror slot's cb.  The GPU
 * table's wrappers live in the persistent wrappers[] array and are installed
 * with a NULL-deleter cb, so releasing the handle immediately is safe — the
 * wrapper is never freed by the cb. */
static arts_item_wrapper_t *gpu_slot_wrapper(arts_route_item_t *item) {
  return (arts_item_wrapper_t *)arts_route_item_peek_data(item);
}

/* Claim (or look up) the slot for `key` in `route_table` and install its
 * persistent wrapper into the slot cb (NULL deleter — the wrapper is owned by
 * the persistent wrappers[] array, never freed by the cb).  Returns the
 * wrapper; sets *installed to whether this caller performed the install. */
static arts_item_wrapper_t *gpu_install_wrapper(arts_route_table_t *route_table,
                                                arts_guid_t key,
                                                bool *installed) {
  arts_route_item_t *entry =
      arts_route_table_search_for_empty(route_table, key, false);
  size_t idx = (size_t)(entry - route_table->data);
  arts_gpu_route_table_t *gpu_route_table =
      (arts_gpu_route_table_t *)((char *)route_table -
                                 offsetof(arts_gpu_route_table_t,
                                          routingTable));
  arts_item_wrapper_t *wrapper = &gpu_route_table->wrappers[idx];
  /* Publish the persistent wrapper into THIS slot's cb (NULL deleter — the
   * wrapper lives in the persistent wrappers[] array, never freed by the cb).
   * Must target the located slot directly: the key→table map would resolve to
   * the global route table, not this per-device mirror table. */
  bool did_install = arts_route_item_install_data(entry, wrapper, NULL);
  if (installed) {
    *installed = did_install;
  }
  return wrapper;
}

unsigned int set_gpu_timestamp(volatile unsigned int *time_stamp) {
  unsigned int new_time_stamp = arts_atomic_add(&gpu_node_order, 1);
  unsigned int old_time_stamp = *time_stamp;
  while (old_time_stamp < new_time_stamp) {
    if (arts_atomic_cswap(time_stamp, old_time_stamp, new_time_stamp) ==
        old_time_stamp) {
      return new_time_stamp;
    }
    old_time_stamp = *time_stamp;
  }
  return old_time_stamp;
}

arts_route_table_t *arts_gpu_new_route_table(unsigned int route_table_size,
                                             unsigned int shift) {
  unsigned int total_elems = COLLISION_RESOLVES * route_table_size;
  arts_gpu_route_table_t *gpu_route_table =
      (arts_gpu_route_table_t *)arts_calloc(1, sizeof(arts_gpu_route_table_t));
  gpu_route_table->routingTable.data = (arts_route_item_t *)arts_calloc_align(
      total_elems, sizeof(arts_route_item_t), 16);
  gpu_route_table->routingTable.size = route_table_size;
  gpu_route_table->routingTable.shift = shift;
  /* setFunc/freeFunc fields removed in new route_table model -- task 1a.4
   * will revisit GPU route_table integration. */
  gpu_route_table->routingTable.newFunc = arts_gpu_new_route_table;

  /* Persistent per-slot wrapper array (parallel to routingTable.data[]).  The
   * slot's cb (value) starts NULL; the wrapper is installed into the slot cb on
   * first add via a NULL-deleter cb (the wrapper is never freed by the cb — it
   * lives in this persistent array). */
  gpu_route_table->wrappers = (arts_item_wrapper_t *)arts_calloc(
      total_elems, sizeof(arts_item_wrapper_t));

  return &gpu_route_table->routingTable;
}

uint64_t arts_gpu_lookup_db(arts_guid_t key) {
  uint64_t ret = 0;
  for (unsigned int i = 0; i < arts_node_info.gpu; ++i) {
    arts_route_table_t *gpu_route_table = arts_node_info.gpu_route_table[i];
    arts_route_item_t *location =
        arts_route_table_search_for_key(gpu_route_table, key);
    if (location) {
      ret |= (1 << i);
    }
  }
  return ret;
}

void *arts_gpu_route_table_add_item(void *item, uint64_t size, arts_guid_t key,
                                    unsigned int gpu_id) {
  // This is a bypass thread local variable to make the api nice...
  gpu_item_size_bypass = size;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  bool added;
  arts_item_wrapper_t *wrapper = gpu_install_wrapper(route_table, key, &added);
  if (added) {
    wrapper->real_data = item;
    wrapper->size = size;
  }
  gpu_item_size_bypass = 0;
  set_gpu_timestamp(&wrapper->time_stamp);
  return (void *)wrapper->real_data;
}

arts_item_wrapper_t *
arts_gpu_route_table_reserve_item(bool *added, uint64_t size, arts_guid_t key,
                                  unsigned int gpu_id, bool add_to_use) {
  // This is a bypass thread local variable to make the api nice...
  (void)add_to_use;
  gpu_item_size_bypass = size;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  bool installed = false;
  arts_item_wrapper_t *wrapper =
      gpu_install_wrapper(route_table, key, &installed);
  if (installed) {
    wrapper->size = size;
  }
  gpu_item_size_bypass = 0;
  if (added) {
    *added = installed;
  }
  set_gpu_timestamp(&wrapper->time_stamp);
  return wrapper;
}

void *arts_gpu_route_table_add_item_to_delete(void *item, uint64_t size,
                                              arts_guid_t key,
                                              unsigned int gpu_id) {
  // This is a bypass thread local variable to make the api nice...
  gpu_item_size_bypass = size;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  bool installed = false;
  arts_item_wrapper_t *wrapper =
      gpu_install_wrapper(route_table, key, &installed);
  if (installed) {
    wrapper->real_data = item;
    wrapper->size = size;
  }
  gpu_item_size_bypass = 0;
  set_gpu_timestamp(&wrapper->time_stamp);
  return (void *)wrapper->real_data;
}

void *arts_gpu_route_table_lookup_db_res(arts_guid_t key, int gpu_id,
                                         const unsigned int *touched,
                                         unsigned int *time_stamp, bool res) {
  void *ret = NULL;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  arts_item_wrapper_t *wrapper = NULL;
  /* New model: data ptr lookup (legacy state machine removed). */
  arts_route_item_t *temp = arts_route_table_search_for_key(route_table, key);
  wrapper =
      (temp) ? (arts_item_wrapper_t *)arts_route_item_peek_data(temp) : NULL;

  if (wrapper) {
    if (res) {
      if (time_stamp) {
        *time_stamp = set_gpu_timestamp(&wrapper->time_stamp);
      }
      /* touched (per-item version) removed in new route_item model --
       * task 1a.4 will revisit GPU LC versioning. */
      (void)touched;
    }
    ret = (void *)wrapper->real_data;
    ARTS_DEBUG("Wrapper: %p %p", wrapper, wrapper->real_data);
  }
  return ret;
}

void *arts_gpu_route_table_lookup_db(arts_guid_t key, int gpu_id,
                                     unsigned int *touched,
                                     unsigned int *time_stamp) {
  return arts_gpu_route_table_lookup_db_res(key, gpu_id, touched, time_stamp,
                                            true);
}

bool arts_gpu_route_table_return_db(arts_guid_t key, bool mark_to_delete,
                                    unsigned int gpu_id) {
  /* No ref count: route_table no longer takes refs.  task 1a.4 will
   * revisit GPU lifecycle. */
  (void)key;
  (void)mark_to_delete;
  (void)gpu_id;
  return false;
}

bool arts_gpu_invalidate_route_tables(arts_guid_t key,
                                      unsigned int keep_on_this_gpu) {
  /* internal_route_table_remove_item removed -- task 1a.4 will revisit. */
  (void)key;
  (void)keep_on_this_gpu;
  return false;
}

bool arts_gpu_invalidate_on_route_table(arts_guid_t key, unsigned int gpu_id) {
  /* internal_route_table_remove_item removed -- task 1a.4 will revisit. */
  (void)key;
  (void)gpu_id;
  return false;
}

volatile unsigned int gpu_reader = 0;
volatile unsigned int gpu_writer = 0;

void gpu_gc_read_lock() {
  while (1) {
    while (gpu_writer) {
    }
    arts_atomic_fetch_add(&gpu_reader, 1U);
    if (gpu_writer == 0) {
      break;
    }
    arts_atomic_sub(&gpu_reader, 1U);
  }
}

void gpu_gc_read_unlock() { arts_atomic_sub(&gpu_reader, 1U); }

void gpu_gc_write_lock() {
  while (arts_atomic_cswap(&gpu_writer, 0U, 1U) != 0U) {
  }
  while (gpu_reader) {
  }
}

void gpu_gc_write_unlock() { arts_atomic_swap(&gpu_writer, 0U); }

/*This takes three parameters to regulate what is deleted.  This will only clean
up DBs!
1.  size_to_clean - this is the desired space to clean up.  The gc will continue
untill it it reaches this size or it has made a full pass across the RT. Passing
-1 will make the gc clean up the entire RT.
2.  clean_zeros - this flag indicates if we should delete data that is not being
used by anyone. Will delete up to size_to_clean.
3.  gpu_id - the id of which GPU this RT belongs.  This is the contiguous id [0
- num_gpus-1]. Returns the size of the memory freed!
*/
uint64_t arts_gpu_clean_up_route_table(unsigned int size_to_clean,
                                       bool clean_zeros, unsigned int gpu_id) {
  uint64_t freed_size = 0;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];
  arts_gpu_route_table_t *gpu_route_table =
      (arts_gpu_route_table_t *)route_table;
  // This is a lock to make sure LC sync works
  gpu_gc_read_lock();
  // Only one person can be running the gc at a time...
  if (arts_try_lock(&gpu_route_table->gcLock)) {
    arts_route_table_iterator_t iter;
    arts_reset_route_table_iterator(&iter, route_table);

    arts_route_item_t *item = arts_route_table_iterate(&iter);
    while (item && freed_size < size_to_clean) {
      /* FIXME: GPU GC sweep needs new model -- task 1a.4
       * Legacy used item->lock bitfield (DELETE_ITEM/AVAILABLE_ITEM/count)
       * + dec_item(); both removed.  GC is a no-op until rewritten. */
      (void)clean_zeros;
      (void)route_table;
      arts_item_wrapper_t *wrapper = gpu_slot_wrapper(item);
      (void)wrapper;
      item = arts_route_table_iterate(&iter);
    }
    arts_unlock(&gpu_route_table->gcLock);
  }
  gpu_gc_read_unlock();
  return freed_size;
}

uint64_t arts_gpu_free_all(unsigned int gpu_id) {
  uint64_t freed_size = 0;
  arts_route_table_t *route_table = arts_node_info.gpu_route_table[gpu_id];

  arts_route_table_iterator_t iter;
  arts_reset_route_table_iterator(&iter, route_table);

  arts_route_item_t *item = arts_route_table_iterate(&iter);
  while (item) {
    arts_item_wrapper_t *wrapper = gpu_slot_wrapper(item);
    freed_size += wrapper->size;
    free_gpu_item(item);
    item = arts_route_table_iterate(&iter);
  }
  return freed_size;
}