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

#include "arts/gas/route_table.h"

#include <stdlib.h>

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/memory/db_functions.h"
#include "arts/runtime/memory/db_list.h"
#include "arts/system/arts_print.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

#define INIT_INVALIDATE_SIZE 128
#define GUID_LOCK_SIZE 1024
volatile unsigned int guid_lock[GUID_LOCK_SIZE] = {0};

void set_item(arts_route_item_t *item, void *data) { item->data = data; }

void free_item(arts_route_item_t *item) {
  // arts_type_t type = arts_guid_get_type(item->key);
  // if (type > ARTS_BUFFER && type < ARTS_LAST_TYPE)
  //   arts_db_free(item->data);
  // else
  //   arts_free(item->data);
  arts_out_of_order_list_delete(&item->ooList);
  item->data = NULL;
  item->key = 0;
  item->lock = 0;
  item->touched = 0;
}

bool mark_reserve(arts_route_item_t *item, bool mark_use) {
  if (mark_use) {
    uint64_t mask = RESERVED_ITEM + 1;
    return !arts_atomic_cswap_u64(&item->lock, 0, mask);
  }
  return !arts_atomic_fetch_or_u64(&item->lock, RESERVED_ITEM);
}

bool mark_requested(arts_route_item_t *item) {
  uint64_t local;
  uint64_t temp;
  while (1) {
    local = item->lock;
    if ((local & RESERVED_ITEM) || (local & DELETE_ITEM)) {
      return false;
    }
    temp = local | RESERVED_ITEM;
    if (local == arts_atomic_cswap_u64(&item->lock, local, temp)) {
      return true;
    }
  }
}

bool mark_write(arts_route_item_t *item) {
  uint64_t local;
  uint64_t temp;
  while (1) {
    local = item->lock;
    if (local & RESERVED_ITEM) {
      temp = (local & ~RESERVED_ITEM) | AVAILABLE_ITEM;
      if (local == arts_atomic_cswap_u64(&item->lock, local, temp)) {
        return true;
      }
    } else {
      return false;
    }
  }
}

bool mark_delete(arts_route_item_t *item) {
  uint64_t res = arts_atomic_fetch_or_u64(&item->lock, DELETE_ITEM);
  return (res & DELETE_ITEM) != 0;
}

bool try_mark_delete(arts_route_item_t *item, uint64_t count_val) {
  uint64_t comp_val = AVAILABLE_ITEM + count_val;
  uint64_t new_val = (AVAILABLE_ITEM | DELETE_ITEM);
  uint64_t old_val = arts_atomic_cswap_u64(&item->lock, comp_val, new_val);
  return (comp_val == old_val);
}

void print_state(arts_route_item_t *item) {
  if (item) {
    uint64_t local = item->lock;
    if (IS_REQ(local)) {
      ARTS_INFO("%lu: reserved-available %p %s", item->key, local,
                GET_TYPE_NAME(arts_guid_get_type(item->key)));
    } else if (IS_RES(local)) {
      ARTS_INFO("%lu: reserved %p %s", item->key, local,
                GET_TYPE_NAME(arts_guid_get_type(item->key)));
    } else if (IS_AVAIL(local)) {
      ARTS_INFO("%lu: available %p %s", item->key, local,
                GET_TYPE_NAME(arts_guid_get_type(item->key)));
    } else if (IS_DEL(local)) {
      ARTS_INFO("%lu: deleted %p %s", item->key, local,
                GET_TYPE_NAME(arts_guid_get_type(item->key)));
    }
  } else {
    ARTS_INFO("NULL ITEM");
  }
}

// 11000 & 11100 = 11000, 10000 & 11100 = 10000, 11100 & 11100 = 11000
bool check_item_state(arts_route_item_t *item, item_state_t state) {
  if (item) {
    uint64_t local = item->lock;
    switch (state) {
    case RESERVED_KEY:
      return IS_RES(local);

    case REQUESTED_KEY:
      return IS_REQ(local);

    case AVAILABLE_KEY:
      return IS_AVAIL(local);

    case ALLOCATED_KEY:
      return IS_RES(local) || IS_AVAIL(local) || IS_REQ(local);

    case DELETED_KEY:
      return IS_DEL(local);

    case ANY_KEY:
      return local != 0;

    default:
      return false;
    }
  }
  return false;
}

inline bool check_min_item_state(arts_route_item_t *item, item_state_t state) {
  if (item) {
    uint64_t local = item->lock;
    item_state_t actual_state = NO_KEY;

    if (IS_DEL(local)) {
      actual_state = DELETED_KEY;

    } else if (IS_RES(local)) {
      actual_state = RESERVED_KEY;

    } else if (IS_REQ(local)) {
      actual_state = REQUESTED_KEY;

    } else if (IS_AVAIL(local)) {
      actual_state = AVAILABLE_KEY;
    }

    return (actual_state && actual_state >= state);
  }
  return false;
}

item_state_t get_item_state(arts_route_item_t *item) {
  if (item) {
    uint64_t local = item->lock;

    if (IS_RES(local)) {
      return RESERVED_KEY;
    }
    if (IS_AVAIL(local)) {
      return AVAILABLE_KEY;
    }

    if (IS_REQ(local)) {
      return REQUESTED_KEY;
    }

    if (IS_DEL(local)) {
      return DELETED_KEY;
    }
  }
  return NO_KEY;
}

bool inc_item(arts_route_item_t *item, unsigned int count, arts_guid_t key,
              arts_route_table_t *route_table) {
  while (1) {
    uint64_t local = item->lock;
    if (!(local & DELETE_ITEM) && CHECK_MAX_ITEM(local) && item->key == key) {
      if (local == arts_atomic_cswap_u64(&item->lock, local, local + count)) {
        if (item->key != key) // This is for an ABA problem
        {
          ARTS_DEBUG("The key changed on us from %lu -> %lu", key, item->key);
          dec_item(route_table, item);
          return false;
        }
        return true;
      }
    } else {
      break;
    }
  }
  return false;
}

bool dec_item(arts_route_table_t *route_table, arts_route_item_t *item) {
  uint64_t local = arts_atomic_sub_u64(&item->lock, 1);
  if (GET_COUNT(local) == 0) {
    if (SHOULD_DELETE(local)) {
      route_table->freeFunc(item);
      return true;
    }
  }
  return false;
}

uint64_t urand64() {
  uint64_t hi = lrand48();
  uint64_t md = lrand48();
  uint64_t lo = lrand48();
  uint64_t res = (hi << 42) + (md << 21) + lo;
  return res;
}

#define HASH64(x, y) ((uint64_t)(x) * (y))

static inline uint64_t get_route_table_key(uint64_t x, unsigned int shift) {
  uint64_t hash = 14695981039346656037U;
  switch (shift) {
  /*case 5:
      hash *= 31;
  case 6:
      hash *= 61;
  case 7:
      hash *= 127;
  case 8:
      hash *= 251;
  case 9:
      hash *= 509;*/
  case 10:
    hash *= 1021;
  case 11:
    hash *= 2039;
  case 12:
    hash *= 4093;
  case 13:
    hash *= 8191;
  case 14:
    hash *= 16381;
  case 15:
    hash *= 32749;
  case 16:
    hash *= 65521;
  case 17:
    hash *= 131071;
  case 18:
    hash *= 262139;
  case 19:
    hash *= 524287;
  case 20:
    hash *= 1048573;
  case 21:
    hash *= 2097143;
  case 22:
    hash *= 4194301;
  case 31:
    hash *= 2147483647;
  case 32:
    hash *= 4294967291;
  default:
    break;
  }

  return (HASH64(x, hash) >> (64 - shift)) * COLLISION_RESOLVES;
}
extern uint64_t num_tables;
extern uint64_t max_guid;
extern uint64_t keys_per_thread;
extern uint64_t min_global_guid_thread;
extern uint64_t max_global_guid_thread;

static inline arts_route_table_t *arts_get_route_table(arts_guid_t guid) {
  arts_guid_bits_t raw = (arts_guid_bits_t){.bits = guid};
  uint64_t key = raw.fields.key;
  if (keys_per_thread) {
    uint64_t global_thread = (key / keys_per_thread);
    if (min_global_guid_thread <= global_thread &&
        global_thread < max_global_guid_thread) {
      return arts_node_info.route_table[global_thread - min_global_guid_thread];
    }
  }
  return arts_node_info.remote_route_table;
}

arts_route_table_t *arts_new_route_table(unsigned int route_table_size,
                                         unsigned int shift) {
  arts_route_table_t *route_table =
      (arts_route_table_t *)arts_calloc(1, sizeof(arts_route_table_t));
  route_table->data = (arts_route_item_t *)arts_calloc_align(
      (size_t)COLLISION_RESOLVES * route_table_size, sizeof(arts_route_item_t),
      16);
  route_table->size = route_table_size;
  route_table->shift = shift;
  route_table->setFunc = set_item;
  route_table->freeFunc = free_item;
  route_table->newFunc = arts_new_route_table;
  return route_table;
}

arts_route_item_t *
arts_route_table_search_for_key(arts_route_table_t *route_table,
                                arts_guid_t key, item_state_t state) {
  arts_route_table_t *current = route_table;
  arts_route_table_t *next;
  uint64_t key_val;
  while (current) {
    key_val = get_route_table_key((uint64_t)key, current->shift);
    for (int i = 0; i < COLLISION_RESOLVES; i++) {
      if (check_item_state(&current->data[key_val], state)) {
        if (current->data[key_val].key == key) {
          return &current->data[key_val];
        }
      }
      key_val++;
    }
    arts_reader_lock(&current->readerLock, &current->writerLock);
    next = current->next;
    arts_reader_unlock(&current->readerLock);
    current = next;
  }
  return NULL;
}

arts_route_item_t *
arts_route_table_search_for_empty(arts_route_table_t *route_table,
                                  arts_guid_t key, bool mark_used) {
  arts_route_table_t *current = route_table;
  arts_route_table_t *next;
  uint64_t key_val;
  while (current != NULL) {
    key_val = get_route_table_key((uint64_t)key, current->shift);
    for (int i = 0; i < COLLISION_RESOLVES; i++) {
      if (!current->data[key_val].lock) {
        if (mark_reserve(&current->data[key_val], mark_used)) {
          current->data[key_val].key = key;
          return &current->data[key_val];
        }
      }
      key_val++;
    }

    arts_reader_lock(&current->readerLock, &current->writerLock);
    next = current->next;
    arts_reader_unlock(&current->readerLock);

    if (!next) {
      if (arts_writer_try_lock(&current->readerLock, &current->writerLock)) {
        next = current->next =
            current->newFunc(2 * current->size, current->shift + 1);
        arts_writer_unlock(&current->writerLock);
      } else {
        arts_reader_lock(&current->readerLock, &current->writerLock);
        next = current->next;
        arts_reader_unlock(&current->readerLock);
      }
    }
    current = next;
  }
  ARTS_ERROR("Route table search failed: impossible state (table=%p)",
             (void *)route_table);
}

void *internal_route_table_add_item(arts_route_table_t *route_table, void *item,
                                    arts_guid_t key, unsigned int rank,
                                    bool used) {
  arts_route_item_t *location =
      arts_route_table_search_for_empty(route_table, key, used);
  route_table->setFunc(location, item);
  location->rank = rank;
  mark_write(location);
  return location;
}

void *arts_route_table_add_item(void *item, arts_guid_t key, unsigned int rank,
                                bool used) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  return internal_route_table_add_item(route_table, item, key, rank, used);
}

bool internal_route_table_remove_item(arts_route_table_t *route_table,
                                      arts_guid_t key) {
  arts_route_item_t *item =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (item) {
    mark_delete(item);
    if (SHOULD_DELETE(item->lock)) {
      route_table->freeFunc(item);
    }
  }
  return 0;
}

bool arts_route_table_remove_item(arts_guid_t key) {
  // arts_route_table_t *route_table = arts_get_route_table(key);
  // return internal_route_table_remove_item(route_table, key);
  return arts_route_table_invalidate_item(key);
}

// This just doesn't delete the item itself... It is for DB rename
bool arts_route_table_hide_item(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (item) {
    item->data = NULL;
  }
  return 0;
}

// This locks the guid so it is useful when multiple people have the guid ahead
// of time The guid doesn't need to be locked if no one knows about it
arts_route_item_t *internal_route_table_add_item_race(
    bool *added_item, arts_route_table_t *route_table, void *item,
    arts_guid_t key, unsigned int rank, bool used_res, bool used_avail,
    unsigned int to_add_on_creation) {
  unsigned int pos = (unsigned int)(((uint64_t)key) % (uint64_t)GUID_LOCK_SIZE);
  *added_item = false;
  arts_route_item_t *found = NULL;
  while (!found) {
    if (guid_lock[pos] == 0) {
      if (!arts_atomic_cswap(&guid_lock[pos], 0U, 1U)) {
        found =
            arts_route_table_search_for_key(route_table, key, ALLOCATED_KEY);
        if (found) {
          if (check_item_state(found, RESERVED_KEY)) {
            route_table->setFunc(found, item);
            found->rank = rank;
            mark_write(found);
            if (used_res) {
              inc_item(found, 1, found->key, route_table);
            }
            *added_item = true;
          } else if (used_avail && check_item_state(found, AVAILABLE_KEY)) {
            inc_item(found, 1, found->key, route_table);
          }
        } else {
          found = (arts_route_item_t *)internal_route_table_add_item(
              route_table, item, key, rank, used_res);
          if (to_add_on_creation) {
            inc_item(found, to_add_on_creation, found->key, route_table);
          }
          *added_item = true;
        }
        guid_lock[pos] = 0U;
      }
    } else {
      found = arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
      if (found && used_avail) {
        inc_item(found, 1, found->key, route_table);
      }
    }
  }
  //    ARTS_INFO("found: %lu %p", key, found);
  return found;
}

arts_route_item_t *
internal_route_table_add_deleted_item_race(arts_route_table_t *route_table,
                                           void *item, arts_guid_t key,
                                           unsigned int rank) {
  unsigned int pos = (unsigned int)(((uint64_t)key) % (uint64_t)GUID_LOCK_SIZE);
  arts_route_item_t *found = NULL;
  while (!found) {
    if (guid_lock[pos] == 0) {
      if (!arts_atomic_cswap(&guid_lock[pos], 0U, 1U)) {
        found = arts_route_table_search_for_empty(route_table, key, false);
        route_table->setFunc(found, item);
        found->rank = rank;
        mark_delete(found);
        mark_write(found);
        guid_lock[pos] = 0U;
      }
    }
  }
  return found;
}

/*
 * arts_route_table_add_item_race — Insert or find an item under a global lock.
 *
 * If the GUID already has a RESERVED slot, transitions it to AVAILABLE
 * (the item was pre-reserved by arts_guid_reserve).  Otherwise, creates
 * a new entry.
 *
 * Returns true if this call actually added (or filled) the entry, false if
 * the entry already existed in AVAILABLE state.
 */
bool arts_route_table_add_item_race(void *item, arts_guid_t key,
                                    unsigned int rank, bool used) {
  bool ret;
  arts_route_table_t *route_table = arts_get_route_table(key);
  internal_route_table_add_item_race(&ret, route_table, item, key, rank, used,
                                     false, 0);
  ARTS_DEBUG("add_item_race: Key=%lu, added=%d", key, ret);
  return ret;
}

// This is used for the send aggregation
bool arts_route_table_reserve_item_race(arts_guid_t key,
                                        arts_route_item_t **item, bool used) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  unsigned int pos = (unsigned int)(((uint64_t)key) % (uint64_t)GUID_LOCK_SIZE);
  bool ret = false;
  *item = NULL;
  while (!(*item)) {
    if (guid_lock[pos] == 0) {
      if (!arts_atomic_cswap(&guid_lock[pos], 0U, 1U)) {
        *item =
            arts_route_table_search_for_key(route_table, key, ALLOCATED_KEY);
        if (!(*item)) {
          *item = arts_route_table_search_for_empty(route_table, key, used);
          ret = true;
        } else {
          if (used) {
            inc_item(*item, 1, (*item)->key, route_table);
          }
        }
        guid_lock[pos] = 0U;
      }
    } else {
      arts_route_item_t *temp =
          arts_route_table_search_for_key(route_table, key, ALLOCATED_KEY);
      if (temp && used) {
        inc_item(temp, 1, temp->key, route_table);
      }
      *item = temp;
    }
  }
  //    print_state(arts_route_table_search_for_key(route_table, key, ANY_KEY));
  return ret;
}

// This does the send aggregation
bool arts_route_table_add_sent(arts_guid_t key, void *edt, unsigned int slot,
                               bool aggregate) {
  arts_route_item_t *item = NULL;
  bool send_req;
  arts_route_table_t *route_table = arts_get_route_table(key);
  // I shouldn't be able to get to here if the db hasn't already been created
  // and I am the owner node thus item can't be null... or so it should be
  if (arts_guid_get_rank(key) == arts_global_rank_id) {
    item = arts_route_table_search_for_key(route_table, key, ALLOCATED_KEY);
    send_req = mark_requested(item);
  } else {
    send_req = arts_route_table_reserve_item_race(key, &item, true);
    if (!send_req && !inc_item(item, 1, item->key, route_table)) {
      ARTS_INFO("Item marked for deletion before it has arrived %u...",
                send_req);
    }
  }
  arts_out_of_order_handle_db_request_with_oo_list(
      &item->ooList, &item->data, (struct arts_edt_s *)edt, slot);
  return send_req || !aggregate;
}

void *arts_route_table_lookup_item(arts_guid_t key) {
  void *ret = NULL;
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *location =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (location) {
    ret = location->data;
  }
  return ret;
}

item_state_t arts_route_table_lookup_item_with_state(arts_guid_t key,
                                                     void ***data,
                                                     item_state_t min,
                                                     bool inc) {
  void *ret = NULL;
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *location =
      arts_route_table_search_for_key(route_table, key, min);
  if (location) {
    if (inc) {
      if (!inc_item(location, 1, location->key, route_table)) {
        *data = NULL;
        return NO_KEY;
      }
    }
    *data = &location->data;
    return get_item_state(location);
  }
  return NO_KEY;
}

void *internal_route_table_lookup_db(arts_route_table_t *route_table,
                                     arts_guid_t key, int *rank,
                                     unsigned int **touched) {
  *rank = -1;
  void *ret = NULL;
  *touched = NULL;
  arts_route_item_t *location =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (location) {
    *rank = (int)location->rank;
    if (inc_item(location, 1, location->key, route_table)) {
      ret = location->data;
      *touched = &location->touched;
    }
  }
  return ret;
}

unsigned int internal_inc_db_version(volatile unsigned int *touched) {
  return arts_atomic_add(touched, 1);
}

void *arts_route_table_lookup_db(arts_guid_t key, int *rank, bool touch) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  unsigned int *touched;
  void *data = internal_route_table_lookup_db(route_table, key, rank, &touched);
  if (data) {
    if (touch) {
      internal_inc_db_version(touched);
    }
  }
  return data;
}

bool internal_route_table_return_db(arts_route_table_t *route_table,
                                    arts_guid_t key, bool mark_to_delete,
                                    bool do_delete) {
  arts_route_item_t *location =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (location) {
    // Only mark it for deletion if it is the last one
    // Why make it unusable to other if there is still other
    // tasks that may benifit
    // True True
    if (mark_to_delete && do_delete) {
      // This should work if there is only one outstanding left... me.  The
      // dec_item needs to sub 1 to delete
      try_mark_delete(location, 1);
      return dec_item(route_table, location);
    }
    // True False
    if (mark_to_delete && !do_delete) {
      dec_item(route_table, location);
      try_mark_delete(location, 0);
      return false;
    }
    // False True || False False
    return dec_item(route_table, location);
  }
  return false;
}

bool arts_route_table_return_db(arts_guid_t key, bool mark_to_delete) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  bool is_remote = arts_guid_get_rank(key) != arts_global_rank_id;
  return internal_route_table_return_db(route_table, key, mark_to_delete,
                                        is_remote);
}

int arts_route_table_lookup_rank(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *location =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (location) {
    return (int)location->rank;
  }
  return -1;
}

int arts_route_table_set_rank(arts_guid_t key, int rank) {
  int ret = -1;
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *location =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (location) {
    ret = (int)location->rank;
    location->rank = rank;
  }
  return ret;
}

/*
 * arts_route_table_fire_oo — Replay all queued OO operations for a GUID.
 *
 * Called immediately after an item transitions to AVAILABLE state (e.g.,
 * after arts_route_table_add_item_race marks it writable).  Each OO entry
 * is dispatched via callback_t (typically arts_out_of_order_handler).
 */
void arts_route_table_fire_oo(arts_guid_t key,
                              void (*callback_t)(void *, void *)) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (item != NULL) {
    ARTS_DEBUG("fire_oo: Key=%lu, ooList=%p, data=%p", key,
               (void *)&item->ooList, item->data);
    arts_out_of_order_list_fire_callback(&item->ooList, item->data, callback_t);
  }
}

bool arts_route_table_add_oo(arts_guid_t key, void *data, bool inc) {
  arts_route_item_t *item = NULL;
  if (arts_route_table_reserve_item_race(key, &item, true) ||
      check_item_state(item, RESERVED_KEY)) {
    if (inc) {
      inc_item(item, 1, item->key, arts_get_route_table(key));
    }
    bool res = arts_out_of_order_list_add_item(&item->ooList, data);
    if (res) {
      INCREMENT_NUM_OO_ENQUEUE_BY(1);
    }
    return res;
  }
  if (inc) {
    inc_item(item, 1, item->key, arts_get_route_table(key));
  }
  return false;
}

bool arts_route_table_add_oo_existing(arts_guid_t key, void *data, bool inc) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (item) {
    if (inc) {
      inc_item(item, 1, item->key, route_table);
    }
    bool res = arts_out_of_order_list_add_item(&item->ooList, data);
    return res;
  }
  return false;
}

void arts_route_table_reset_oo(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item =
      arts_route_table_search_for_key(route_table, key, ANY_KEY);
  arts_out_of_order_list_reset(&item->ooList);
}

void **arts_route_table_get_oo_list(arts_guid_t key,
                                    struct arts_out_of_order_list_s **list) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (item != NULL) {
    *list = &item->ooList;
    return &item->data;
  }
  return NULL;
}

// This is just a wrapper for outside consumption...
void **arts_route_table_reserve(arts_guid_t key, bool *dec,
                                item_state_t *state) {
  bool res;
  *dec = false;
  arts_route_item_t *item = NULL;
  while (1) {
    res = arts_route_table_reserve_item_race(key, &item, true);
    if (!res) {
      // Check to make sure we can use it
      if (inc_item(item, 1, item->key, arts_get_route_table(key))) {
        *dec = true;
        break;
      }
      // If we were not keep trying...
    } else { // we were successful in reserving
      break;
    }
  }
  if (item) {
    *state = get_item_state(item);
  }
  return &item->data;
}

arts_route_item_t *get_item_from_data(arts_guid_t key, void *data) {
  if (data) {
    arts_route_item_t *item =
        (arts_route_item_t *)((char *)data - sizeof(arts_guid_t));
    if (key == item->key) {
      return item;
    }
  }
  return NULL;
}

void arts_route_table_dec_item(arts_guid_t key, void *data) {
  if (data) {
    arts_route_table_t *route_table = arts_get_route_table(key);
    dec_item(route_table, get_item_from_data(key, data));
  }
}

void arts_reset_route_table_iterator(arts_route_table_iterator_t *iter,
                                     arts_route_table_t *table) {
  iter->table = table;
  iter->index = 0;
}

arts_route_item_t *arts_route_table_iterate(arts_route_table_iterator_t *iter) {
  arts_route_table_t *current = iter->table;
  arts_route_table_t *next;
  while (current != NULL) {
    for (uint64_t i = iter->index;
         i < (uint64_t)current->size * COLLISION_RESOLVES; i++) {
      // arts_print_item(&current->data[i]);
      if (current->data[i].lock) {
        iter->index = i + 1;
        iter->table = current;
        return &current->data[i];
      }
    }
    iter->index = 0;
    arts_reader_lock(&current->readerLock, &current->writerLock);
    next = current->next;
    arts_reader_unlock(&current->readerLock);
    current = next;
  }
  return NULL;
}

void arts_print_item(arts_route_item_t *item) {
  if (item) {
    uint64_t local = item->lock;
    ARTS_INFO(
        "GUID: %lu DATA: %p RANK: %u LOCK: %p COUNTERS: %lu Res: %u Req: %u "
        "Avail: %u Del: %u",
        item->key, item->data, item->rank, local, GET_COUNT(local),
        IS_RES(local) != 0, IS_REQ(local) != 0, IS_AVAIL(local) != 0,
        IS_DEL(local) != 0);
  }
}

uint64_t arts_clean_up_route_table(arts_route_table_t *route_table) {
  uint64_t free_size = 0;
  arts_route_table_iterator_t iter;
  arts_reset_route_table_iterator(&iter, route_table);

  arts_route_item_t *item = arts_route_table_iterate(&iter);
  while (item) {
    // arts_print_item(item);
    arts_type_t type = arts_guid_get_type(item->key);
    // These are DB types
    if (type > ARTS_BUFFER && type < ARTS_LAST_TYPE) {
      struct arts_db_s *db = (struct arts_db_s *)item->data;
      if (db) {
        if (!arts_atomic_sub(&db->copy_count, 1)) {
          free_size += db->header.size;
          ARTS_DEBUG("Freeing DB[Guid:%lu] [Size:%lu]", item->key,
                     db->header.size);
          arts_db_free(db);
          free_item(item);
        }
      }
    }
    item = arts_route_table_iterate(&iter);
  }
  return free_size;
}

void arts_clean_up_dbs() {
  uint64_t free_size = 0;
  for (unsigned int i = 0; i < arts_node_info.total_thread_count; i++) {
    free_size += arts_clean_up_route_table(arts_node_info.route_table[i]);
  }
  // arts_clean_up_route_table(arts_node_info.remote_route_table);
  ARTS_INFO("Cleaned %lu bytes", free_size);
}

// To cleanup
// --------------------------------------------------------------------------->

bool arts_route_table_update_item(arts_guid_t key, void *data,
                                  unsigned int rank, item_state_t state) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  bool ret = false;
  arts_route_item_t *found = NULL;
  while (!found) {
    found = arts_route_table_search_for_key(route_table, key, state);
    if (found) {
      found->data = data;
      found->rank = rank;
      mark_write(found);
      ret = true;
    }
  }
  return ret;
}

bool arts_route_table_invalidate_item(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  return internal_route_table_remove_item(route_table, key);
}

void arts_route_table_add_rank_duplicate(arts_guid_t key, unsigned int rank) {}

bool arts_route_table_get_rank_duplicates(
    arts_guid_t key, unsigned int rank,
    struct arts_db_frontier_iterator_s *iter) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *location =
      arts_route_table_search_for_key(route_table, key, AVAILABLE_KEY);
  if (location) {
    if (rank != (unsigned int)-1) {
      // Blocks until the OO is done firing
      arts_out_of_order_list_reset(&location->ooList);
      location->rank = rank;
    }
    struct arts_db_s *db = (struct arts_db_s *)location->data;
    return arts_close_frontier((struct arts_db_list_s *)db->db_list, iter);
  }
  return false;
}
