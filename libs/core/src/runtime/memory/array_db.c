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

#include "arts/runtime/memory/array_db.h"

#include <string.h>

#include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/memory/db_functions.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/runtime/sync/termination_detection.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"

unsigned int arts_get_size_array_db(arts_array_db_t *array) {
  return array->elements_per_block * array->num_blocks;
}

void *copy_db(void *ptr, unsigned int size, arts_guid_t guid) {
  struct arts_db_s *db = ((struct arts_db_s *)ptr) - 1;
  struct arts_db_s *new_db = (struct arts_db_s *)arts_calloc_align(1, size, 16);
  memcpy(new_db, db, size);
  new_db->guid = guid;
  return (void *)(new_db + 1);
}

arts_array_db_t *arts_new_array_db_with_guid(arts_guid_t guid, unsigned int element_size,
                                      unsigned int num_elements) {
  unsigned int num_blocks = arts_global_rank_count;
  unsigned int elements_per_block = num_elements / num_blocks;
  if (!elements_per_block) {
    elements_per_block = 1;
    num_blocks = num_elements;
  } else if (num_elements % num_blocks) {
    elements_per_block++;
    num_elements = elements_per_block * num_blocks;
  }

  ARTS_INFO("Elements: %u Blocks: %u Element Size:%u", num_elements, num_blocks,
            element_size);

  unsigned int alloc_size =
      sizeof(arts_array_db_t) + ((unsigned long)element_size * elements_per_block);
  arts_array_db_t *block = NULL;
  if (num_blocks) {
    // We have to manually create the db so it isn't updated before we send
    // it...
    unsigned int db_size = sizeof(struct arts_db_s) + alloc_size;
    struct arts_db_s *to_send = (struct arts_db_s *)arts_calloc_align(1, db_size, 16);
    arts_db_create_internal(guid, to_send, alloc_size, db_size, ARTS_DB_PIN, 0);

    block = (arts_array_db_t *)(to_send + 1);
    block->element_size = element_size;
    block->elements_per_block = elements_per_block;
    block->num_blocks = num_blocks;

    for (unsigned int i = 0; i < arts_global_rank_count; i++) {
      if (i != arts_global_rank_id) {
        arts_remote_memory_move_no_free(i, guid, to_send,
                                   alloc_size + sizeof(struct arts_db_s),
                                   ARTS_REMOTE_DB_MOVE_MSG);
      }
    }

    arts_db_create_with_guid_and_data(guid, block, alloc_size);
  }
  return block;
}

arts_array_db_t *arts_new_local_array_db_with_guid(arts_guid_t guid,
                                           unsigned int element_size,
                                           unsigned int num_elements,
                                           void *data) {
  unsigned int num_blocks = 1;
  unsigned int elements_per_block = num_elements;

  ARTS_INFO("Elements: %u Blocks: %u Element Size:%u", num_elements, num_blocks,
            element_size);
  unsigned int alloc_size =
      sizeof(arts_array_db_t) + ((unsigned long)element_size * elements_per_block);
  arts_array_db_t *block = NULL;

  unsigned int db_size = sizeof(struct arts_db_s) + alloc_size;
  // struct arts_db_s *local = arts_calloc(1, db_size);
  struct arts_db_s *local = (struct arts_db_s *)arts_malloc_align(db_size, 16);
  arts_db_create_internal(guid, local, alloc_size, db_size, ARTS_DB_PIN, 0);

  block = (arts_array_db_t *)(local + 1);
  block->element_size = element_size;
  block->elements_per_block = elements_per_block;
  block->num_blocks = num_blocks;
  memcpy((char *)block + sizeof(arts_array_db_t), data,
         ((unsigned long)element_size * elements_per_block));

  arts_db_create_with_guid_and_data(guid, block, alloc_size);
  return block;
}

arts_guid_t arts_new_array_db(arts_array_db_t **addr, unsigned int element_size,
                          unsigned int num_elements) {
  arts_guid_t guid = arts_guid_reserve(ARTS_DB_PIN, arts_global_rank_id);
  *addr = arts_new_array_db_with_guid(guid, element_size, num_elements);
  return guid;
}

arts_guid_t get_array_db_guid(arts_array_db_t *array) {
  struct arts_db_s *db = ((struct arts_db_s *)array) - 1;
  return db->guid;
}

unsigned int get_offset_from_index(arts_array_db_t *array, unsigned int index) {
  unsigned int base = sizeof(arts_array_db_t);
  unsigned int local = (index % array->elements_per_block) * array->element_size;
  //    ARTS_INFO("array: %p base: %u index: %u elements_per_block: %u mod: %u
  //    element_size: %u", array, base, index, array->elements_per_block,
  //    index%array->elements_per_block, array->element_size);
  return base + local;
}

unsigned int get_rank_from_index(arts_array_db_t *array, unsigned int index) {
  return index / array->elements_per_block;
}

void arts_signal_array_db(arts_array_db_t *array, arts_guid_t edt_guid,
                       unsigned int slot) {
  arts_guid_t array_guid = get_array_db_guid(array);
  arts_signal_edt(edt_guid, slot, array_guid);
}

void arts_get_from_array_db(arts_guid_t edt_guid, unsigned int slot,
                        arts_array_db_t *array, unsigned int index) {
  if (index < array->elements_per_block * array->num_blocks) {
    arts_guid_t guid = get_array_db_guid(array);
    unsigned int rank = get_rank_from_index(array, index);
    unsigned int offset = get_offset_from_index(array, index);
    //        ARTS_INFO("Get index: %u rank: %u offset: %u", index, rank,
    //        offset);
    arts_get_from_db_at(edt_guid, guid, slot, offset, array->element_size, rank);
  } else {
    ARTS_INFO("Index >= Array Size:%u >= %u * %u", index,
              array->elements_per_block, array->num_blocks);
    arts_debug_generate_seg_fault();
  }
}

void arts_put_in_array_db(void *ptr, arts_guid_t edt_guid, unsigned int slot,
                      arts_array_db_t *array, unsigned int index) {
  arts_guid_t guid = get_array_db_guid(array);
  unsigned int rank = get_rank_from_index(array, index);
  unsigned int offset = get_offset_from_index(array, index);
  arts_put_in_db_at(ptr, edt_guid, guid, slot, offset, array->element_size, rank);
}

void arts_for_each_in_array_db(arts_array_db_t *array, arts_edt_t func_ptr,
                          uint32_t paramc, const uint64_t *paramv) {
  uint64_t *args = (uint64_t *)arts_malloc(sizeof(uint64_t) * (paramc + 1));
  memcpy(&args[1], paramv, sizeof(uint64_t) * paramc);

  unsigned int size = arts_get_size_array_db(array);
  for (unsigned int i = 0; i < size; i++) {
    args[0] = i;
    unsigned int route = get_rank_from_index(array, i);
    arts_guid_t guid = arts_edt_create(func_ptr, paramc + 1, args, 1,
                                       &(arts_hint_t){.route = route});
    arts_get_from_array_db(guid, 0, array, i);
  }
}

void arts_gather_array_db(arts_array_db_t *array, arts_edt_t func_ptr,
                       unsigned int route, uint32_t paramc, const uint64_t *paramv,
                       uint64_t depc) {
  if (route == -1) {
    route = arts_global_rank_id;
}
  unsigned int offset = get_offset_from_index(array, 0);
  unsigned int size = array->element_size * array->elements_per_block;
  arts_guid_t array_guid = get_array_db_guid(array);

  arts_guid_t guid =
      arts_edt_create(func_ptr, paramc, paramv, array->num_blocks + depc,
                      &(arts_hint_t){.route = route});
  for (unsigned int i = 0; i < array->num_blocks; i++) {
    arts_get_from_db_at(guid, array_guid, i, offset, size, i);
  }
}

void arts_gather_array_db_epoch(arts_array_db_t *array, arts_edt_t func_ptr,
                            unsigned int route, uint32_t paramc,
                            const uint64_t *paramv, uint64_t depc,
                            arts_guid_t epoch_guid) {
  if (route == -1) {
    route = arts_global_rank_id;
}
  unsigned int offset = get_offset_from_index(array, 0);
  unsigned int size = array->element_size * array->elements_per_block;
  arts_guid_t array_guid = get_array_db_guid(array);

  arts_guid_t guid = arts_edt_create_with_epoch(func_ptr, paramc, paramv,
                                           array->num_blocks + depc, epoch_guid,
                                           &(arts_hint_t){.route = route});
  for (unsigned int i = 0; i < array->num_blocks; i++) {
    arts_get_from_db_at(guid, array_guid, i, offset, size, i);
  }
}

void arts_gather_array_db_in_edt(arts_array_db_t *array, arts_guid_t to_edt_guid,
                            uint64_t slot_offset) {
  unsigned int offset = get_offset_from_index(array, 0);
  unsigned int size = array->element_size * array->elements_per_block;
  arts_guid_t array_guid = get_array_db_guid(array);

  for (unsigned int i = 0; i < array->num_blocks; i++) {
    arts_get_from_db_at(to_edt_guid, array_guid, slot_offset + i, offset, size, i);
  }
}

void loop_policy(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)depc;
  arts_edt_t func_ptr = (arts_edt_t)paramv[0];
  unsigned int stride = paramv[1];
  unsigned int end = paramv[2];
  unsigned int start = paramv[3];

  arts_array_db_t *array = (arts_array_db_t *)depv[0].ptr;
  unsigned int offset = get_offset_from_index(array, start);
  char *raw = (char *)depv[0].ptr;

  uint64_t loop_index = start;
  for (unsigned int i = start; i < end; i += stride) {
    loop_index = i;
    depv[0].ptr = (void *)(&raw[offset]);
    func_ptr(paramc - 3, &loop_index, 1, depv);
    offset += array->element_size;
  }
  depv[0].ptr = (void *)raw;
}

void arts_for_each_in_array_db_at_data(arts_array_db_t *array, unsigned int stride,
                                arts_edt_t func_ptr, uint32_t paramc,
                                const uint64_t *paramv) {
  unsigned int block_size = array->elements_per_block;
  unsigned int size = arts_get_size_array_db(array);
  if (size % stride) {
    ARTS_INFO("WARNING: Size is not divisible by stride!");
  }
  arts_guid_t guid = get_array_db_guid(array);
  uint64_t *args = (uint64_t *)arts_malloc(sizeof(uint64_t) * (paramc + 4));
  if (paramc) {
    memcpy(&args[4], paramv, sizeof(uint64_t) * paramc);
}
  args[0] = (uint64_t)func_ptr;
  args[1] = stride;
  for (unsigned int i = 0; i < size; i += block_size) {
    args[2] = (i + block_size < size) ? i + block_size : size;
    args[3] = i;
    unsigned int target_rank = get_rank_from_index(array, i);
    arts_guid_t am = arts_edt_create(loop_policy, paramc + 4, args, 1,
                                     &(arts_hint_t){.route = target_rank});
    arts_signal_edt(am, 0, guid);
  }
}

void internal_atomic_add_in_array_db(arts_guid_t db_guid, unsigned int index,
                                unsigned int to_add, arts_guid_t edt_guid,
                                unsigned int slot, arts_guid_t epoch_guid) {
  struct arts_db_s *db = (struct arts_db_s *)arts_route_table_lookup_item(db_guid);
  if (db) {
    arts_array_db_t *array = (arts_array_db_t *)(db + 1);
    // Do this so when we increment finished we can check the term status
    increment_queue_epoch(epoch_guid);
    global_shutdown_guid_inc_queue();
    unsigned int offset = get_offset_from_index(array, index);
    unsigned int *data = (unsigned int *)(((char *)array) + offset);
    unsigned int result = arts_atomic_add(data, to_add);
    //        ARTS_INFO("index: %u result: %u", index, result);

    if (edt_guid) {
      //            ARTS_INFO("Signaling edt_guid: %lu", edt_guid);
      arts_signal_edt_value(edt_guid, slot, result);
    }

    increment_finished_epoch(epoch_guid);
    global_shutdown_guid_inc_finished();
  } else {
    arts_out_of_order_atomic_add_in_array_db(db_guid, index, to_add, edt_guid, slot,
                                     epoch_guid);
  }
}

void arts_atomic_add_in_array_db(arts_array_db_t *array, unsigned int index,
                            unsigned int to_add, arts_guid_t edt_guid,
                            unsigned int slot) {
  arts_guid_t db_guid = get_array_db_guid(array);
  arts_guid_t epoch_guid = arts_get_current_epoch_guid();
  increment_active_epoch(epoch_guid);
  global_shutdown_guid_inc_active();
  unsigned int rank = get_rank_from_index(array, index);
  if (rank == arts_global_rank_id) {
    internal_atomic_add_in_array_db(db_guid, index, to_add, edt_guid, slot, epoch_guid);
  } else {
    arts_remote_atomic_add_in_array_db(rank, db_guid, index, to_add, edt_guid, slot,
                                 epoch_guid);
}
}

void internal_atomic_compare_and_swap_in_array_db(
    arts_guid_t db_guid, unsigned int index, unsigned int old_value,
    unsigned int new_value, arts_guid_t edt_guid, unsigned int slot,
    arts_guid_t epoch_guid) {
  struct arts_db_s *db = (struct arts_db_s *)arts_route_table_lookup_item(db_guid);
  if (db) {
    arts_array_db_t *array = (arts_array_db_t *)(db + 1);
    // Do this so when we increment finished we can check the term status
    increment_queue_epoch(epoch_guid);
    global_shutdown_guid_inc_queue();
    unsigned int offset = get_offset_from_index(array, index);
    unsigned int *data = (unsigned int *)(((char *)array) + offset);
    unsigned int result = arts_atomic_cswap(data, old_value, new_value);
    //        ARTS_INFO("index: %u result: %u", index, result);

    if (edt_guid) {
      //            ARTS_INFO("Signaling edt_guid: %lu", edt_guid);
      arts_signal_edt_value(edt_guid, slot, result);
    }

    increment_finished_epoch(epoch_guid);
    global_shutdown_guid_inc_finished();
  } else {
    arts_out_of_order_atomic_compare_and_swap_in_array_db(
        db_guid, index, old_value, new_value, edt_guid, slot, epoch_guid);
  }
}

void arts_atomic_compare_and_swap_in_array_db(arts_array_db_t *array, unsigned int index,
                                       unsigned int old_value,
                                       unsigned int new_value,
                                       arts_guid_t edt_guid, unsigned int slot) {
  arts_guid_t db_guid = get_array_db_guid(array);
  arts_guid_t epoch_guid = arts_get_current_epoch_guid();
  increment_active_epoch(epoch_guid);
  global_shutdown_guid_inc_active();
  unsigned int rank = get_rank_from_index(array, index);
  if (rank == arts_global_rank_id) {
    internal_atomic_compare_and_swap_in_array_db(db_guid, index, old_value, new_value,
                                          edt_guid, slot, epoch_guid);
  } else {
    arts_remote_atomic_compare_and_swap_in_array_db(rank, db_guid, index, old_value,
                                            new_value, edt_guid, slot, epoch_guid);
}
}
