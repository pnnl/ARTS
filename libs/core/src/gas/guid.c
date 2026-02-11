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
#include "arts/gas/guid.h"

#include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/runtime/globals.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"

uint64_t num_tables = 0;
uint64_t keys_per_thread = 0;
uint64_t global_guid_on = 0;
uint64_t min_global_guid_thread = 0;
uint64_t max_global_guid_thread = 0;

void set_global_guid_on() { global_guid_on = ((uint64_t)1) << 40; }

uint64_t *arts_guid_generator_get_key(unsigned int route, unsigned int type) {
  return &arts_node_info
              .keys[arts_thread_info.group_id][(route * ARTS_LAST_TYPE) + type];
}

arts_guid_t arts_guid_create_for_rank_internal(unsigned int route, unsigned int type,
                                         unsigned int guid_count) {
  arts_guid_bits_t guid;
  if (global_guid_on) {
    // Safeguard against wrap around
    if (global_guid_on > guid_count) {
      guid.fields.key = global_guid_on - guid_count;
      global_guid_on -= guid_count;
    } else {
      ARTS_INFO("Parallel Start out of guid keys");
      arts_debug_generate_seg_fault();
    }
  } else {
    uint64_t *key = arts_guid_generator_get_key(route, type);
    uint64_t value = *key;
    if (value + guid_count < keys_per_thread) {
      guid.fields.key =
          value + keys_per_thread *
                      arts_node_info.global_guid_thread_id[arts_thread_info.group_id];
      (*key) += guid_count;
    } else {
      ARTS_INFO("Out of guid keys");
      arts_debug_generate_seg_fault();
    }
  }
  guid.fields.type = type;
  guid.fields.rank = route;
  INCREMENT_GUID_ALLOC_COUNTER_BY(guid_count);
  return (arts_guid_t)guid.bits;
}

arts_guid_t arts_guid_create_for_rank(unsigned int route, unsigned int type) {
  return arts_guid_create_for_rank_internal(route, type, 1);
}

void set_guid_generator_after_parallel_start() {
  unsigned int num_of_tables = arts_node_info.worker_thread_count + 1;
  keys_per_thread = global_guid_on / ((uint64_t)num_of_tables * arts_global_rank_count);
  global_guid_on = 0;
}

void arts_guid_key_generator_init() {
  num_tables = (arts_global_rank_count == 1) ? arts_node_info.worker_thread_count
                                         : arts_node_info.worker_thread_count + 1;
  uint64_t local_id = (arts_thread_info.worker) ? arts_thread_info.thread_id
                                             : arts_node_info.worker_thread_count;
  min_global_guid_thread = num_tables * arts_global_rank_id;
  max_global_guid_thread = (arts_global_rank_count == 1)
                            ? min_global_guid_thread + num_tables
                            : min_global_guid_thread + num_tables - 1;
  //    global_guid_thread_id  = min_global_guid_thread + local_id;
  arts_node_info.global_guid_thread_id[arts_thread_info.group_id] =
      min_global_guid_thread + local_id;

  //    ARTS_INFO("num_tables: %lu local_id: %lu min_global_guid_thread: %lu
  //    max_global_guid_thread: %lu global_guid_thread_id: %lu", num_tables, local_id,
  //    min_global_guid_thread, max_global_guid_thread, global_guid_thread_id); keys =
  //    arts_malloc(sizeof(uint64_t) * ARTS_LAST_TYPE * arts_global_rank_count);
  arts_node_info.keys[arts_thread_info.group_id] = (uint64_t *)arts_malloc(
      sizeof(uint64_t) * ARTS_LAST_TYPE * arts_global_rank_count);
  for (unsigned int i = 0; i < ARTS_LAST_TYPE * arts_global_rank_count; i++) {
    arts_node_info.keys[arts_thread_info.group_id][i] = 1;
}
  //        keys[i] = 1;
}

arts_type_t arts_guid_get_type(arts_guid_t guid) {
  INCREMENT_GUID_LOOKUP_COUNTER_BY(1);
  arts_guid_bits_t address_info = (arts_guid_bits_t){.bits = guid};
  return (arts_type_t)address_info.fields.type;
}

unsigned int arts_guid_get_rank(arts_guid_t guid) {
  INCREMENT_GUID_LOOKUP_COUNTER_BY(1);
  arts_guid_bits_t address_info = (arts_guid_bits_t){.bits = guid};
  return address_info.fields.rank;
}

bool arts_guid_is_local(arts_guid_t guid) {
  return (arts_global_rank_id == arts_guid_get_rank(guid));
}

uint64_t arts_guid_get_key(arts_guid_t guid) {
  arts_guid_bits_t address_info = (arts_guid_bits_t){.bits = guid};
  return address_info.fields.key;
}

arts_guid_t arts_guid_reserve(arts_type_t type, unsigned int route) {
  arts_guid_t guid = NULL_GUID;
  if (route == -1) {
    route = arts_global_rank_id;
}
  route = route % arts_global_rank_count;
  if (type > ARTS_NULL && type < ARTS_LAST_TYPE) {
    guid = arts_guid_create_for_rank_internal(route, (unsigned int)type, 1);
    // ARTS_INFO("Allocation Guid %u", guid);
  } else {
    ARTS_INFO("Invalid type %u", type);
  }
  //    if(route == arts_global_rank_id)
  //        arts_route_table_add_item(NULL, guid, arts_global_rank_id, false);
  return guid;
}

arts_guid_t *arts_guid_reserve_round_robin(unsigned int size, arts_type_t type) {
  arts_guid_t *guids = NULL;
  if (type > ARTS_NULL && type < ARTS_LAST_TYPE) {
    guids = (arts_guid_t *)arts_malloc(size * sizeof(arts_guid_t));
    for (unsigned int i = 0; i < size; i++) {
      unsigned int route = i % arts_global_rank_count;
      guids[i] = arts_guid_create_for_rank(route, (unsigned int)type);
    }
  }
  return guids;
}

arts_guid_range_t *arts_guid_range_create(arts_type_t type, unsigned int size,
                                    unsigned int route) {
  if (route == -1) {
    route = arts_global_rank_id;
}
  arts_guid_range_t *range = NULL;
  if (size && type > ARTS_NULL && type < ARTS_LAST_TYPE) {
    range = (arts_guid_range_t *)arts_calloc(1, sizeof(arts_guid_range_t));
    range->size = size;
    range->start_guid =
        arts_guid_create_for_rank_internal(route, (unsigned int)type, size);
    //        if(artsIsGuidLocalExt(range->start_guid))
    //        {
    //            arts_guid_bits_t temp = (arts_guid_t) range->start_guid;
    //            for(unsigned int i=0; i<size; i++)
    //            {
    //                arts_route_table_add_item(NULL, temp.bits, route);
    //                temp.fields.key++;
    //            }
    //        }
  }
  return range;
}

arts_guid_range_t *arts_guid_range_create_hash(arts_type_t type, unsigned int size,
                                        unsigned int route,
                                        unsigned int hash_size) {
  arts_guid_range_t *range = NULL;
  if (size && type > ARTS_NULL && type < ARTS_LAST_TYPE) {
    range = (arts_guid_range_t *)arts_calloc(1, sizeof(arts_guid_range_t));
    range->size = size;
    range->start_guid = arts_guid_create_for_rank_internal(route, (unsigned int)type,
                                                     size + hash_size);
    arts_guid_bits_t temp = (arts_guid_bits_t){.bits = range->start_guid};
    for (unsigned int i = 0; i < hash_size; i++) {
      if (temp.fields.key % hash_size == 0) {
        range->start_guid = (arts_guid_t)temp.bits;
        break;
      }
      temp.fields.key++;
    }
  }
  return range;
}

arts_guid_t arts_guid_range_get(arts_guid_range_t *range, unsigned int index) {
  if (!range || index >= range->size) {
    return NULL_GUID;
  }
  arts_guid_bits_t ret = (arts_guid_bits_t){.bits = range->start_guid};
  ret.fields.key += index;
  return ret.bits;
}

arts_guid_t arts_guid_range_next(arts_guid_range_t *range) {
  arts_guid_t ret = NULL_GUID;
  if (range) {
    if (range->index < range->size) {
      ret = arts_guid_range_get(range, range->index++);
    }
  }
  return ret;
}

bool arts_guid_range_has_next(arts_guid_range_t *range) {
  if (range) {
    return (range->index < range->size);
}
  return false;
}

void arts_guid_range_reset_iter(arts_guid_range_t *range) {
  if (range) {
    range->index = 0;
}
}

bool arts_guid_range_contains(arts_guid_range_t *range, arts_guid_t guid) {
  arts_guid_bits_t start_guid = (arts_guid_bits_t){.bits = range->start_guid};
  arts_guid_bits_t to_check = (arts_guid_bits_t){.bits = guid};

  if (start_guid.fields.rank != to_check.fields.rank) {
    return false;
}

  if (start_guid.fields.type != to_check.fields.type) {
    return false;
}

  if (start_guid.fields.key <= to_check.fields.key &&
      to_check.fields.key < start_guid.fields.key + range->index) {
    return true;
}

  return false;
}

uint64_t arts_guid_hash_key(arts_guid_t guid) {
  uint64_t key = arts_guid_get_key(guid);
  return key % (uint64_t)arts_node_info.gpu;
}
