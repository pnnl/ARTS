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

#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/out_of_order_list.h"
#include "arts/memory/db.h"
#include "arts/runtime_state.h"
#include "arts/sync/event.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

#define INIT_INVALIDATE_SIZE 128
#define GUID_LOCK_SIZE 1024
volatile unsigned int guid_lock[GUID_LOCK_SIZE] = {0};

static inline unsigned int arts_guid_lock_index(arts_guid_t key) {
  uint64_t hash = ((uint64_t)key) * 11400714819323198485ull;
  hash ^= hash >> 32;
  return (unsigned int)(hash % (uint64_t)GUID_LOCK_SIZE);
}

static inline void arts_guid_lock_release(volatile unsigned int *lock) {
  __atomic_store_n(lock, 0U, __ATOMIC_RELEASE);
}

static inline bool arts_guid_lock_is_free(volatile unsigned int *lock) {
  return __atomic_load_n(lock, __ATOMIC_RELAXED) == 0U;
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
  uint64_t key = ARTS_GUID_GET_KEY(guid);
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
  route_table->newFunc = arts_new_route_table;
  /* Vyukov MPSC OO list cannot be zero-initialized (head/tail must point
   * at the embedded stub).  Initialize every slot up front; slot reuse
   * across the table's lifetime is fine because the list is fully drained
   * by destroy/cleanup paths before any new push could land. */
  uint64_t total_slots = (uint64_t)COLLISION_RESOLVES * route_table_size;
  for (uint64_t i = 0; i < total_slots; i++) {
    arts_oo_list_init(&route_table->data[i].ooList);
  }
  return route_table;
}

/* Slot is empty when key == 0 (ARTS GUIDs never have key value 0).  This is
 * the new model: no lock bitfield, no reserved/available state -- only "key
 * claimed or not".  Once claimed, slot is permanent for that key. */
arts_route_item_t *
arts_route_table_search_for_key(arts_route_table_t *route_table,
                                arts_guid_t key) {
  arts_route_table_t *current = route_table;
  arts_route_table_t *next;
  uint64_t key_val;
  while (current) {
    key_val = get_route_table_key((uint64_t)key, current->shift);
    for (int i = 0; i < COLLISION_RESOLVES; i++) {
      arts_guid_t slot_key =
          __atomic_load_n(&current->data[key_val].key, __ATOMIC_ACQUIRE);
      if (slot_key == key) {
        return &current->data[key_val];
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

/* Linearly scan for an empty slot (key == 0) and atomically claim it for
 * `key` via CAS.  Caller (reserve_or_lookup) holds the per-GUID guid_lock,
 * but other threads may be claiming neighboring slots concurrently for
 * different GUIDs that hash into the same chunk -- hence the CAS. */
arts_route_item_t *
arts_route_table_search_for_empty(arts_route_table_t *route_table,
                                  arts_guid_t key, bool mark_used) {
  (void)mark_used; /* legacy: caller used to request "available + 1 ref" */
  arts_route_table_t *current = route_table;
  arts_route_table_t *next;
  uint64_t key_val;
  while (current != NULL) {
    key_val = get_route_table_key((uint64_t)key, current->shift);
    for (int i = 0; i < COLLISION_RESOLVES; i++) {
      arts_guid_t expected = (arts_guid_t)0;
      if (__atomic_compare_exchange_n(&current->data[key_val].key, &expected,
                                      key, false, __ATOMIC_ACQ_REL,
                                      __ATOMIC_ACQUIRE)) {
        return &current->data[key_val];
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

/* Reserve a slot for `key` (or look it up if already present).  Coordinates
 * concurrent reservers via the per-GUID guid_lock array so only one thread
 * actually creates the slot.  On return, *out points to the slot (key set,
 * data may be NULL). */
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  unsigned int pos = arts_guid_lock_index(key);
  arts_route_item_t *item = NULL;
  while (item == NULL) {
    if (arts_guid_lock_is_free(&guid_lock[pos])) {
      if (!arts_atomic_cswap(&guid_lock[pos], 0U, 1U)) {
        /* search by key first */
        item = arts_route_table_search_for_key(route_table, key);
        if (item == NULL) {
          /* allocate empty slot (search_for_empty installs key via CAS) */
          item = arts_route_table_search_for_empty(route_table, key,
                                                   /*mark_used*/ false);
          /* data is already NULL from calloc; ooList head is also NULL. */
        }
        arts_guid_lock_release(&guid_lock[pos]);
      }
    } else {
      /* spin briefly; lock holder will release soon and may have published */
      item = arts_route_table_search_for_key(route_table, key);
    }
  }
  *out = item;
}

void *arts_route_table_lookup_data(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return NULL;
  }
  return atomic_load_explicit(&item->data, memory_order_acquire);
}

void *arts_route_table_lookup_item(arts_guid_t key) {
  return arts_route_table_lookup_data(key);
}

void *arts_route_table_lookup_db(arts_guid_t key, int *rank, bool touch) {
  (void)touch; /* legacy touched field removed */
  if (rank) {
    *rank = (int)arts_guid_get_rank(key);
  }
  return arts_route_table_lookup_data(key);
}

int arts_route_table_lookup_rank(arts_guid_t key) {
  /* rank is now derivable directly from the GUID -- no need to consult the
   * route table. */
  return (int)arts_guid_get_rank(key);
}

int arts_route_table_set_rank(arts_guid_t key, int rank) {
  /* Rank is fixed by GUID encoding; legacy callers that "moved" entries are
   * being phased out in Task 1a.3.  No-op. */
  (void)key;
  (void)rank;
  return -1;
}

/* Install `data` into the slot for `key` and fire any pending OoO entries.
 * No-op if data is already non-NULL (idempotent). */
void *arts_route_table_add_item(void *data, arts_guid_t key, unsigned int rank,
                                bool used) {
  (void)rank; /* extractable from key */
  (void)used; /* ref_count removed */
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  atomic_store_explicit(&item->data, data, memory_order_release);
  /* Fire pending OoO entries -- installer fires after data store. */
  arts_route_table_fire_oo(key, arts_out_of_order_handler);
  return item;
}

/* CAS-install `data` (NULL -> data); returns true only if this caller won.
 * On win, fire pending OoO entries. */
bool arts_route_table_add_item_race(void *data, arts_guid_t key,
                                    unsigned int rank, bool used) {
  (void)rank;
  (void)used;
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  void *expected = NULL;
  bool installed = atomic_compare_exchange_strong_explicit(
      &item->data, &expected, data, memory_order_release, memory_order_acquire);
  if (installed) {
    arts_route_table_fire_oo(key, arts_out_of_order_handler);
  }
  return installed;
}

/* Legacy wrapper retained for the few internal call sites that pre-date the
 * `add_item_race` simplification.  used_res / used_avail / to_add_on_creation
 * referred to ref_count behavior that no longer exists. */
arts_route_item_t *internal_route_table_add_item_race(
    bool *added_item, arts_route_table_t *route_table, void *data,
    arts_guid_t key, unsigned int rank, bool used_res, bool used_avail,
    unsigned int to_add_on_creation) {
  (void)route_table;
  (void)rank;
  (void)used_res;
  (void)used_avail;
  (void)to_add_on_creation;
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  void *expected = NULL;
  bool installed = atomic_compare_exchange_strong_explicit(
      &item->data, &expected, data, memory_order_release, memory_order_acquire);
  if (added_item) {
    *added_item = installed;
  }
  if (installed) {
    arts_route_table_fire_oo(key, arts_out_of_order_handler);
  }
  return item;
}

arts_route_item_t *
internal_route_table_add_deleted_item_race(arts_route_table_t *route_table,
                                           void *data, arts_guid_t key,
                                           unsigned int rank) {
  /* Legacy path used to mark a slot DELETED on creation so subsequent
   * lookups would skip it.  In the new model there is no DELETE bit -- the
   * caller must consult the cache_s/v3 RC state machine instead.  Provide
   * a permissive install so existing callers still link. */
  (void)route_table;
  (void)rank;
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  atomic_store_explicit(&item->data, data, memory_order_release);
  return item;
}

/* OoO-integrated push.  Returns ENQUEUED if data was NULL (push deferred)
 * or AVAILABLE_NOW if data is/became non-NULL during the push. */
oo_add_result_t arts_route_table_add_oo_ex(arts_guid_t key, void *payload) {
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);

  /* Step 1: pre-push data check (fast path). */
  void *cur = atomic_load_explicit(&item->data, memory_order_acquire);
  if (cur != NULL) {
    return OO_RESULT_AVAILABLE_NOW;
  }

  /* Step 2: push to OO list.  MPSC push always succeeds (no
   * DRAIN_HAPPENED race window). */
  (void)arts_oo_list_push(&item->ooList, payload);

  /* Step 3: post-push data recheck (TOCTOU rescue).
   *
   * Installer's data store + fire_oo may have happened in the window
   * between our Step-1 lookup and the push.  In that case our entry is
   * stranded -- the installer's fire already ran and drained whatever was
   * in the list at that moment, but our payload landed afterwards and
   * will sit forever unless we trigger another drain.  Calling fire_oo
   * here drains our entry plus any other late arrivals. */
  cur = atomic_load_explicit(&item->data, memory_order_acquire);
  if (cur != NULL) {
    arts_route_table_fire_oo(key, arts_out_of_order_handler);
    return OO_RESULT_FIRED_BY_DRAIN;
  }
  return OO_RESULT_ENQUEUED;
}

/* Boolean wrapper for legacy 12 OoO callers in out_of_order.c.
 *
 * Returns true when the caller should do nothing (the deferred handler
 * has been or will be invoked elsewhere): ENQUEUED or FIRED_BY_DRAIN.
 *
 * Returns false only on AVAILABLE_NOW, where the payload was never
 * pushed and the caller must invoke the handler inline + free the
 * payload itself (legacy semantics). */
bool arts_route_table_add_oo(arts_guid_t key, void *payload, bool inc) {
  (void)inc; /* ref_count removed */
  return arts_route_table_add_oo_ex(key, payload) != OO_RESULT_AVAILABLE_NOW;
}

bool arts_route_table_add_oo_existing(arts_guid_t key, void *payload,
                                      bool inc) {
  (void)inc;
  return arts_route_table_add_oo_ex(key, payload) != OO_RESULT_AVAILABLE_NOW;
}

void arts_route_table_fire_oo(arts_guid_t key,
                              void (*callback)(void *data, void *ctx)) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return;
  }
  void *data = atomic_load_explicit(&item->data, memory_order_acquire);
  /* Installer always calls fire after storing data, so data is non-NULL. */
  arts_oo_list_drain(&item->ooList, callback, data);
}

void arts_route_table_drop_oo(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return;
  }
  /* No destroy callback -- only payload free. */
  arts_oo_list_drop_all(&item->ooList);
}

bool arts_route_table_mark_delete(arts_guid_t key) {
  /* In the new model, the cache_s / v3 RC layer is the source of truth for
   * DB destruction.  Clearing the route table's data ptr lets pending OoO
   * pushers see a NULL slot.  Slot itself remains permanent. */
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return false;
  }
  atomic_store_explicit(&item->data, (void *)NULL, memory_order_release);
  return true;
}

bool arts_route_table_hide_item(arts_guid_t key) {
  /* Hide is the same operation as mark_delete in the new model: clear the
   * data ptr without disturbing the slot. */
  return arts_route_table_mark_delete(key);
}

arts_route_item_t *get_item_from_data(arts_guid_t key, void *data) {
  if (data) {
    arts_route_item_t *item =
        (arts_route_item_t *)((char *)data - offsetof(arts_route_item_t, data));
    if (key == item->key) {
      return item;
    }
  }
  return NULL;
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
      arts_guid_t slot_key =
          __atomic_load_n(&current->data[i].key, __ATOMIC_ACQUIRE);
      if (slot_key != 0) {
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
    void *data = atomic_load_explicit(&item->data, memory_order_acquire);
    ARTS_INFO("GUID: %lu DATA: %p RANK: %u", item->key, data,
              arts_guid_get_rank(item->key));
  }
}

void arts_route_table_debug_guid(arts_guid_t key, const char *label) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item) {
    void *data = atomic_load_explicit(&item->data, memory_order_acquire);
    ARTS_INFO("[RT-DBG:%s] Guid:%lu data=%p rank=%u", label, key, data,
              arts_guid_get_rank(item->key));
  } else {
    ARTS_INFO("[RT-DBG:%s] Guid:%lu NOT FOUND", label, key);
  }
}

uint64_t arts_clean_up_route_table(arts_route_table_t *route_table) {
  uint64_t free_size = 0;
  arts_route_table_iterator_t iter;
  arts_reset_route_table_iterator(&iter, route_table);

  arts_route_item_t *item = arts_route_table_iterate(&iter);
  while (item) {
    arts_type_t type = arts_guid_get_type(item->key);
    /* Phase 2.2 (baseline regression repair): use atomic_exchange to claim
     * the data ptr.  EVENT / BUFFER lifecycle paths free their structs at
     * fire/destroy time WITHOUT NULLing the route_table slot (Task 2.1
     * removed arts_route_table_remove_item).  If we re-free those here
     * we get a tcache double-free.  Skip lifecycle-owned types and let
     * the per-type owner reclaim memory; cleanup only handles types whose
     * data ptr survives until shutdown (DBs in v2 dual-stack mode). */
    void *data = atomic_exchange_explicit(&item->data, (void *)NULL,
                                          memory_order_acq_rel);
    if (type == ARTS_DB) {
      struct arts_db_s *db = (struct arts_db_s *)data;
      if (db) {
        free_size += db->header.size;
        arts_db_free(db);
      }
    }
    /* ARTS_EVENT / ARTS_BUFFER / ARTS_EDT / ARTS_EPOCH: data is owned by
     * the execution / lifecycle path and freed there.  Cleanup just
     * drops the route_table reference (already done by atomic_exchange
     * above) plus the OoO list.  Re-freeing here would tcache-corrupt. */
    arts_oo_list_drop_all(&item->ooList);
    item = arts_route_table_iterate(&iter);
  }
  return free_size;
}

void arts_delete_route_table(arts_route_table_t *route_table) {
  if (!route_table) {
    return;
  }
  arts_delete_route_table(route_table->next);
  /* Safety sweep: drop any OO list memory that arts_clean_up_route_table
   * missed (e.g. entries claimed but never installed). */
  for (uint64_t i = 0; i < (uint64_t)route_table->size * COLLISION_RESOLVES;
       i++) {
    arts_oo_list_drop_all(&route_table->data[i].ooList);
  }
  arts_free(route_table->data);
  arts_free(route_table);
}

void arts_clean_up_dbs() {
  uint64_t free_size = 0;
  for (unsigned int i = 0; i < arts_node_info.total_thread_count; i++) {
    free_size += arts_clean_up_route_table(arts_node_info.route_table[i]);
  }
  free_size += arts_clean_up_route_table(arts_node_info.remote_route_table);
  ARTS_INFO("Cleaned %lu bytes", free_size);
}
