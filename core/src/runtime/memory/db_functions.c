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

#include "arts/runtime/memory/db_functions.h"

#include <assert.h>
#include <string.h>

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/introspection/arts_id_counter.h"
#include "arts/introspection/counter.h"
#include "arts/introspection/metrics.h"
#include "arts/introspection/Preamble.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/rt.h"
#include "arts/runtime/compute/edt_functions.h"
#include "arts/runtime/memory/db_list.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/runtime/sync/termination_detection.h"
#include "arts/system/arts_print.h"
#include "arts/utils/atomics.h"

#ifdef USE_GPU
#include "arts/gpu/gpu_runtime.cuh"
#endif

static inline arts_type_t arts_get_effective_mode(arts_edt_dep_t *dep) {
  return (dep->acquire_mode != ARTS_NULL) ? dep->acquire_mode : dep->mode;
}

ARTS_TYPE_NAME;

void *arts_db_malloc(arts_type_t mode, unsigned int size) {
  void *ptr = NULL;
#ifdef USE_GPU
  if (arts_node_info.gpu) {
    if (mode == ARTS_DB_LC)
      ptr = arts_cuda_malloc_host(size * 2);
    else if (mode == ARTS_DB_GPU_READ || mode == ARTS_DB_GPU_WRITE)
      ptr = arts_cuda_malloc_host(size);
  }
#endif
  if (!ptr) {
    ptr = arts_malloc_align(size, 16);
}
  return ptr;
}

void arts_db_free(void *ptr) {
  struct arts_db *db = (struct arts_db *)ptr;
#ifdef USE_GPU
  if (arts_node_info.gpu &&
      (db->header.type == ARTS_DB_GPU_READ ||
       db->header.type == ARTS_DB_GPU_WRITE || db->header.type == ARTS_DB_LC)) {
    arts_cuda_free_host(ptr);
    ptr = NULL;
  }
#endif
  if (ptr) {
    arts_free(ptr);
}
}

void arts_db_create_internal(arts_guid_t guid, void *addr, uint64_t size,
                          uint64_t packet_size, arts_type_t mode,
                          uint64_t arts_id) {
  struct arts_header *header = (struct arts_header *)addr;
  header->type = mode;
  header->size = packet_size;

  struct arts_db *db_res = (struct arts_db *)header;
  db_res->arts_id = arts_id; // Set compiler-assigned arts_id (0 if not set)
  db_res->guid = guid;
  db_res->version = 0;
  db_res->reader = 0;
  db_res->writer = 0;
  db_res->copyCount = 0;
  if (mode != ARTS_DB_PIN) {
    db_res->db_list = arts_new_db_list();
  }
  if (mode == ARTS_DB_LC) {
    void *shadow_copy = (void *)(((char *)addr) + packet_size);
    memcpy(shadow_copy, addr, sizeof(struct arts_db));
  }
  db_res->event_guid = arts_persistent_event_create(arts_guid_get_rank(guid), 0, guid);
  // Record arts_id metrics via counter infrastructure
  arts_counter_record_arts_id_db(arts_id, packet_size, 0, 0);
  INCREMENT_NUM_DBS_CREATED_BY(1);
}

arts_guid_t arts_db_create_remote(unsigned int route, uint64_t size,
                              arts_type_t mode) {
  DB_CREATE_COUNTER_START();
  if (route == -1) {
    route = arts_global_rank_id;
}
  arts_guid_t guid = arts_guid_create_for_rank(route, mode);
  void *ptr = arts_db_malloc(mode, sizeof(struct arts_db));
  struct arts_db *db = (struct arts_db *)ptr;
  db->header.size = size + sizeof(struct arts_db);
  db->db_list = (mode == ARTS_DB_PIN) ? (void *)0 : (void *)1;

  arts_remote_memory_move(route, guid, ptr, sizeof(struct arts_db),
                       ARTS_REMOTE_DB_SEND_MSG, arts_db_free);
  DB_CREATE_COUNTER_STOP();
  return guid;
}

// Creates a local DB only
arts_guid_t arts_db_create(void **addr, uint64_t size, arts_type_t mode) {
  DB_CREATE_COUNTER_START();
  arts_guid_t guid = NULL_GUID;
  uint64_t db_size = size + sizeof(struct arts_db);

  void *ptr = ARTS_MALLOC_WITH_TYPE(db_size, ARTS_METRIC_DB_MEMORY_SIZE);
  if (ptr) {
    guid = arts_guid_create_for_rank(arts_global_rank_id, mode);
    arts_db_create_internal(guid, ptr, size, db_size, mode, 0);
    // change false to true to force a manual DB delete
    arts_route_table_add_item(ptr, guid, arts_global_rank_id, false);
    *addr = (void *)((struct arts_db *)ptr + 1);
  }
  DB_CREATE_COUNTER_STOP();
  return guid;
}

arts_guid_t arts_db_create_ptr(arts_ptr_t *addr, uint64_t size, arts_type_t mode) {
  DB_CREATE_COUNTER_START();
  arts_guid_t guid = NULL_GUID;
  uint64_t db_size = size + sizeof(struct arts_db);

  void *ptr = ARTS_MALLOC_WITH_TYPE(db_size, ARTS_METRIC_DB_MEMORY_SIZE);
  if (ptr) {
    guid = arts_guid_create_for_rank(arts_global_rank_id, mode);
    arts_db_create_internal(guid, ptr, size, db_size, mode, 0);
    // change false to true to force a manual DB delete
    arts_route_table_add_item(ptr, guid, arts_global_rank_id, false);
    *addr = (arts_ptr_t)((struct arts_db *)ptr + 1);
  }
  DB_CREATE_COUNTER_STOP();
  return guid;
}

arts_guid_t arts_db_create_with_arts_id(void **addr, uint64_t size, arts_type_t mode,
                                  uint64_t arts_id) {
  DB_CREATE_COUNTER_START();
  arts_guid_t guid = NULL_GUID;
  uint64_t db_size = size + sizeof(struct arts_db);

  void *ptr = ARTS_MALLOC_WITH_TYPE(db_size, ARTS_METRIC_DB_MEMORY_SIZE);
  if (ptr) {
    guid = arts_guid_create_for_rank(arts_global_rank_id, mode);
    arts_db_create_internal(guid, ptr, size, db_size, mode, arts_id);
    // change false to true to force a manual DB delete
    arts_route_table_add_item(ptr, guid, arts_global_rank_id, false);
    *addr = (void *)((struct arts_db *)ptr + 1);
  }
  DB_CREATE_COUNTER_STOP();
  return guid;
}

void *arts_db_create_with_guid_and_arts_id(arts_guid_t guid, uint64_t size,
                                    uint64_t arts_id) {
  DB_CREATE_COUNTER_START();
  arts_type_t mode = arts_guid_get_type(guid);

  void *ptr = NULL;
  if (arts_is_guid_local(guid)) {
    uint64_t db_size = size + sizeof(struct arts_db);

    ptr = ARTS_MALLOC_WITH_TYPE(db_size, ARTS_METRIC_DB_MEMORY_SIZE);
    if (ptr) {
      arts_db_create_internal(guid, ptr, size, db_size, mode, arts_id);
      if (arts_route_table_add_item_race(ptr, guid, arts_global_rank_id, false)) {
        arts_route_table_fire_oo(guid, arts_out_of_order_handler);
      }
      ptr = (void *)((struct arts_db *)ptr + 1);
    }
  }
  ARTS_INFO("Creating DB[Id:%lu, Guid:%lu, Mode:%s, Ptr:%p, Route:%d, "
            "Size:%lu]",
            arts_id, guid, GET_TYPE_NAME(mode), ptr, arts_guid_get_rank(guid), size);
  DB_CREATE_COUNTER_STOP();
  return ptr;
}

// Guid must be for a local DB only
void *arts_db_create_with_guid(arts_guid_t guid, uint64_t size) {
  DB_CREATE_COUNTER_START();
  arts_type_t mode = arts_guid_get_type(guid);

  void *ptr = NULL;
  uint64_t arts_id = 0;
  if (arts_is_guid_local(guid)) {
    uint64_t db_size = size + sizeof(struct arts_db);

    ptr = ARTS_MALLOC_WITH_TYPE(db_size, ARTS_METRIC_DB_MEMORY_SIZE);
    if (ptr) {
      struct arts_db *db_header = (struct arts_db *)ptr;
      arts_db_create_internal(guid, db_header, size, db_size, mode, 0);
      arts_id = db_header->arts_id;
      if (arts_route_table_add_item_race(db_header, guid, arts_global_rank_id, false)) {
        arts_route_table_fire_oo(guid, arts_out_of_order_handler);
      }
      ptr = (void *)(db_header + 1);
    }
  }
  ARTS_INFO("Creating DB[Id:%lu, Guid:%lu, Mode:%s, Ptr:%p, Route:%d, "
            "Size:%lu]",
            arts_id, guid, GET_TYPE_NAME(mode), ptr, arts_guid_get_rank(guid), size);
  DB_CREATE_COUNTER_STOP();
  return ptr;
}

void *arts_db_create_with_guid_and_data(arts_guid_t guid, void *data, uint64_t size) {
  DB_CREATE_COUNTER_START();
  arts_type_t mode = arts_guid_get_type(guid);
  void *ptr = NULL;
  uint64_t arts_id = 0;
  if (arts_is_guid_local(guid)) {
    uint64_t db_size = size + sizeof(struct arts_db);

    ptr = ARTS_MALLOC_WITH_TYPE(db_size, ARTS_METRIC_DB_MEMORY_SIZE);

    if (ptr) {
      struct arts_db *db_header = (struct arts_db *)ptr;
      arts_db_create_internal(guid, db_header, size, db_size, mode, 0);
      arts_id = db_header->arts_id;
      void *db_data = (void *)(db_header + 1);
      memcpy(db_data, data, size);
      if (arts_route_table_add_item_race(db_header, guid, arts_global_rank_id, false)) {
        arts_route_table_fire_oo(guid, arts_out_of_order_handler);
}
      ptr = db_data;
    }
  }
  ARTS_INFO("Creating DB[Id:%lu, Guid:%lu, Mode:%s, Ptr:%p, Route:%d, "
            "Size:%lu]",
            arts_id, guid, GET_TYPE_NAME(mode), ptr, arts_guid_get_rank(guid), size);
  DB_CREATE_COUNTER_STOP();
  return ptr;
}

void *arts_db_resize_ptr(struct arts_db *db_res, unsigned int size, bool copy) {
  if (db_res) {
    unsigned int old_size = db_res->header.size;
    unsigned int new_size = size + sizeof(struct arts_db);
    struct arts_db *ptr = (struct arts_db *)ARTS_CALLOC_ALIGN_WITH_TYPE(
        1, new_size, 16, ARTS_METRIC_DB_MEMORY_SIZE);
    if (ptr) {
      if (copy) {
        memcpy(ptr, db_res, old_size);
      } else {
        memcpy(ptr, db_res, sizeof(struct arts_db));
}
      arts_free(db_res);
      ptr->header.size = size + sizeof(struct arts_db);
      return (void *)(ptr + 1);
    }
  }
  return NULL;
}

// Must be in write mode (or only copy) to update and alloced (no NO_ACQUIRE
// nonsense), otherwise will be racy...
void *arts_db_resize(arts_guid_t guid, unsigned int size, bool copy) {
  struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(guid);
  void *ptr = arts_db_resize_ptr(db_res, size, copy);
  if (ptr) {
    db_res = ((struct arts_db *)ptr) - 1;
  }
  return ptr;
}

void arts_db_move(arts_guid_t db_guid, unsigned int rank) {
  unsigned int guid_rank = arts_guid_get_rank(db_guid);
  if (guid_rank != rank) {
    if (guid_rank != arts_global_rank_id) {
      arts_db_move_request(db_guid, rank);
    } else {
      struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(db_guid);
      if (db_res) {
        arts_remote_memory_move(rank, db_guid, db_res, db_res->header.size,
                             ARTS_REMOTE_DB_MOVE_MSG, arts_db_free);
      } else {
        arts_out_of_order_db_move(db_guid, rank);
      }
    }
  }
}

void arts_db_destroy(arts_guid_t guid) {
  arts_type_t mode = arts_guid_get_type(guid);
  struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(guid);
  if (db_res != NULL) {
    arts_remote_db_destroy(guid, arts_global_rank_id, 0);
    arts_db_free(db_res);
    arts_route_table_remove_item(guid);
  } else {
    arts_remote_db_destroy(guid, arts_global_rank_id, 0);
}
}

bool arts_db_rename_with_guid(arts_guid_t new_guid, arts_guid_t old_guid) {
  bool ret = false;
  unsigned int rank = arts_guid_get_rank(old_guid);
  if (rank == arts_global_rank_id) {
    struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(old_guid);
    if (db_res != NULL) {
      db_res->guid = new_guid;
      // This is only being done by the owner...
      arts_route_table_hide_item(old_guid);
      if (arts_route_table_add_item_race(db_res, new_guid, arts_global_rank_id, false)) {
        arts_route_table_fire_oo(new_guid, arts_out_of_order_handler);
      }
      ret = true;
    }
  } else {
    arts_remote_db_rename(new_guid, old_guid);
  }
  return ret;
}

arts_guid_t arts_db_copy_to_new_type(arts_guid_t old_guid, arts_type_t new_type) {
  arts_guid_t ret = NULL_GUID;
  unsigned int rank = arts_guid_get_rank(old_guid);
  if (rank == arts_global_rank_id) {
    arts_guid_t new_guid = arts_guid_create_for_rank(rank, arts_guid_get_type(new_type));
    struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(old_guid);
    if (db_res != NULL) {
      arts_atomic_add(&db_res->copyCount, 1);
      db_res->guid = new_guid;
      if (arts_route_table_add_item_race(db_res, new_guid, arts_global_rank_id, false)) {
        arts_route_table_fire_oo(new_guid, arts_out_of_order_handler);
      }
      ret = new_guid;
    }
  }
  return ret;
}

arts_guid_t arts_db_rename(arts_guid_t guid) {
  arts_guid_t new_guid =
      arts_guid_create_for_rank(arts_guid_get_rank(guid), arts_guid_get_type(guid));
  return (arts_db_rename_with_guid(new_guid, guid)) ? new_guid : NULL_GUID;
}

void arts_db_destroy_safe(arts_guid_t guid, bool remote) {
  struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(guid);
  if (db_res != NULL) {
    if (remote) {
      arts_remote_db_destroy(guid, arts_global_rank_id, 0);
}
    arts_db_free(db_res);
    arts_route_table_remove_item(guid);
  } else if (remote) {
    arts_remote_db_destroy(guid, arts_global_rank_id, 0);
}
}

void arts_db_increment_latch(arts_guid_t guid) {
  struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(guid);
  if (db_res != NULL) {
    arts_persistent_event_increment_latch(db_res->event_guid);
  } else {
    arts_remote_db_increment_latch(guid);
}
}

void arts_db_decrement_latch(arts_guid_t guid) {
  struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(guid);
  if (db_res != NULL) {
    arts_persistent_event_decrement_latch(db_res->event_guid);
  } else {
    arts_remote_db_decrement_latch(guid);
}
}

void arts_db_add_dependence(arts_guid_t db_src, arts_guid_t edt_dest,
                         uint32_t edt_slot) {
  struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(db_src);
  if (db_res != NULL) {
    arts_add_dependence_to_persistent_event(db_res->event_guid, edt_dest, edt_slot);
  } else {
    arts_remote_db_add_dependence(db_src, edt_dest, edt_slot);
}
}

void arts_db_add_dependence_with_mode(arts_guid_t db_src, arts_guid_t edt_dest,
                                 uint32_t edt_slot, arts_type_t acquire_mode) {
  arts_db_add_dependence_with_mode_and_diff(db_src, edt_dest, edt_slot, acquire_mode);
}

void arts_db_add_dependence_with_mode_and_diff(arts_guid_t db_src, arts_guid_t edt_dest,
                                        uint32_t edt_slot,
                                        arts_type_t acquire_mode) {
  struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(db_src);
  if (db_res != NULL) {
    arts_add_dependence_to_persistent_event_with_mode_and_diff(
        db_res->event_guid, edt_dest, edt_slot, acquire_mode);
  } else {
    arts_remote_db_add_dependence_with_hints(db_src, edt_dest, edt_slot, acquire_mode);
}
}

// Auto-increments latch for WRITE mode, records dependency with compiler hints
void arts_record_dep(arts_guid_t db_src, arts_guid_t edt_dest, uint32_t edt_slot,
                   arts_type_t acquire_mode) {
  arts_db_add_dependence_with_mode_and_diff(db_src, edt_dest, edt_slot, acquire_mode);
  if (acquire_mode == ARTS_DB_WRITE) {
    arts_db_increment_latch(db_src);
}
}

void arts_record_dep_at(arts_guid_t db_src, arts_guid_t edt_dest, uint32_t edt_slot,
                     arts_type_t acquire_mode, uint64_t byte_offset,
                     uint64_t size) {
  // If no byte offset, use the standard path
  if (byte_offset == 0 && size == 0) {
    arts_record_dep(db_src, edt_dest, edt_slot, acquire_mode);
    return;
  }

  // Use extended dependency registration with byte offset info
  struct arts_db *db_res = (struct arts_db *)arts_route_table_lookup_item(db_src);
  if (db_res != NULL) {
    arts_add_dependence_to_persistent_event_with_byte_offset(
        db_res->event_guid, edt_dest, edt_slot, acquire_mode, byte_offset, size);
  } else {
    arts_remote_db_add_dependence_with_byte_offset(db_src, edt_dest, edt_slot,
                                            acquire_mode, byte_offset, size);
}

  if (acquire_mode == ARTS_DB_WRITE) {
    arts_db_increment_latch(db_src);
}
}

/**********************DB MEMORY MODEL*************************************/
// Side Effects: edt depcNeeded will be incremented, ptr will be updated,
//   and launches out of order handleReadyEdt
// Returns false on out of order and true otherwise
void acquire_dbs(struct arts_edt *edt) {
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  edt->depcNeeded = edt->depc + 1;
  ARTS_INFO("Acquiring %u DBs for EDT[Id:%lu, Guid:%lu], depcNeeded "
            "initialized to %u",
            edt->depc, edt->arts_id, edt->current_edt, edt->depcNeeded);
  for (int i = 0; i < edt->depc; i++) {
    if (depv[i].guid && depv[i].ptr == NULL) {
      struct arts_db *db_found = NULL;
      int owner = arts_guid_get_rank(depv[i].guid);

      // Calculate effective acquire mode (override if compiler provided hint)
      arts_type_t effective_mode = arts_get_effective_mode(&depv[i]);

      // Update acquire-mode counters
      if (effective_mode == ARTS_DB_READ) {
        INCREMENT_ACQUIRE_READ_MODE_BY(1);
        // Track owner update savings when override changes WRITE -> READ
        if (depv[i].mode == ARTS_DB_WRITE && owner == arts_global_rank_id) {
          INCREMENT_OWNER_UPDATES_SAVED_BY(1);
}
      } else if (effective_mode == ARTS_DB_WRITE) {
        INCREMENT_ACQUIRE_WRITE_MODE_BY(1);
        if (owner == arts_global_rank_id) {
          INCREMENT_OWNER_UPDATES_PERFORMED_BY(1);
}
      }

      ARTS_INFO("Acquiring DB[Guid:%lu, Mode:%u, Owner:%d, Rank:%u] "
                "in EDT[Id:%lu, Guid:%lu, Slot:%u]",
                depv[i].guid, depv[i].mode, owner, arts_global_rank_id,
                edt->arts_id, edt->current_edt, i);
      switch (depv[i].mode) {
      // This case assumes that the guid exists only on the owner
      case ARTS_DB_ONCE: {
        if (owner != arts_global_rank_id) {
          arts_out_of_order_handle_db_request(depv[i].guid, edt, i, false);
          arts_db_move(depv[i].guid, arts_global_rank_id);
          break;
        }
        // else fall through to the local case :-p
      }
      case ARTS_DB_ONCE_LOCAL: {
        struct arts_db *db_temp =
            (struct arts_db *)arts_route_table_lookup_item(depv[i].guid);
        if (db_temp) {
          db_found = db_temp;
          arts_atomic_sub(&edt->depcNeeded, 1U);
        } else {
          arts_out_of_order_handle_db_request(depv[i].guid, edt, i, false);
}
        break;
      }
      case ARTS_DB_PIN: {
        int valid_rank = -1;
        struct arts_db *db_temp = (struct arts_db *)arts_route_table_lookup_db(
            depv[i].guid, &valid_rank, true);
        if (db_temp) {
          db_found = db_temp;
          arts_atomic_sub(&edt->depcNeeded, 1U);
        } else {
          arts_out_of_order_handle_db_request(depv[i].guid, edt, i, true);
        }
        break;
      }
      case ARTS_DB_LC_SYNC: {
        // Owner Rank
        if (owner == arts_global_rank_id) {
          int valid_rank = -1;
          struct arts_db *db_temp = (struct arts_db *)arts_route_table_lookup_db(
              depv[i].guid, &valid_rank, false);
          // We have found an entry
          if (db_temp) {
            ARTS_DEBUG("MODE: %s -> %p", GET_TYPE_NAME(depv[i].mode), db_temp);
            db_found = db_temp;
            arts_atomic_sub(&edt->depcNeeded, 1U);
          }
          // The Db hasn't been created yet
          else {
            // TODO: Create an out-of-order sync
            ARTS_DEBUG("DB[Guid:%lu] out of order request for LC_SYNC not "
                       "supported yet",
                       depv[i].guid);
            // arts_out_of_order_handle_db_request(depv[i].guid, edt, i, true);
          }
        }
        break;
      }
      case ARTS_DB_LC_NO_COPY:
      case ARTS_DB_GPU_MEMSET:
      case ARTS_DB_GPU_READ:
      case ARTS_DB_GPU_WRITE:
      case ARTS_DB_LC:
      case ARTS_DB_READ:
      case ARTS_DB_WRITE: {
        // Owner Rank
        if (owner == arts_global_rank_id) {
          int valid_rank = -1;
          struct arts_db *db_temp = (struct arts_db *)arts_route_table_lookup_db(
              depv[i].guid, &valid_rank, true);
          // We have found an entry
          if (db_temp) {
            bool duplicate_added =
                arts_add_db_duplicate(db_temp, arts_global_rank_id, edt,
                                   edt->current_edt, i, effective_mode);
            if (duplicate_added) {
              ARTS_DEBUG("Adding duplicate DB[Guid:%lu]", depv[i].guid);
            } else {
              ARTS_DEBUG(
                  "Duplicate not added DB[Guid:%lu] (rank already tracked)",
                  depv[i].guid);
}

            // If the owner (this rank) still has the valid copy, hand it out
            // regardless of whether we inserted a new duplicate entry.
            if (valid_rank == arts_global_rank_id) {
              db_found = db_temp;
              arts_atomic_sub(&edt->depcNeeded, 1U);
            }
            // Owner rank but another rank currently has the valid copy; request
            // it.
            else {
              if (depv[i].mode == ARTS_DB_READ ||
                  depv[i].mode == ARTS_DB_GPU_READ ||
                  depv[i].mode == ARTS_DB_GPU_WRITE ||
                  depv[i].mode == ARTS_DB_LC ||
                  depv[i].mode == ARTS_DB_LC_NO_COPY ||
                  depv[i].mode == ARTS_DB_GPU_MEMSET) {
                arts_remote_db_request(depv[i].guid, valid_rank, edt, i,
                                    depv[i].mode, true, depv[i].acquire_mode);
              } else {
                arts_remote_db_full_request(depv[i].guid, valid_rank,
                                        edt->current_edt, i, depv[i].mode);
}
            }
          }
          // The Db hasn't been created yet
          else {
            ARTS_DEBUG("DB[Guid:%lu] out of order request slot %u",
                       depv[i].guid, i);
            arts_out_of_order_handle_db_request(depv[i].guid, edt, i, true);
          }
        } else {
          int valid_rank = -1;
          struct arts_db *db_temp = (struct arts_db *)arts_route_table_lookup_db(
              depv[i].guid, &valid_rank, true);
          // We have found an entry
          ARTS_INFO("[AcquireDbs] Non-owner case for DB[Guid:%lu, Mode:%u, "
                    "Owner:%d, ValidRank: %d, DbTemp: %p]",
                    depv[i].guid, depv[i].mode, owner, valid_rank,
                    (void *)db_temp);
          bool local_valid = (db_temp && valid_rank == arts_global_rank_id);
          if (db_temp) {
            ARTS_INFO("[AcquireDbs] Non-owner cache state DB[Guid:%lu, "
                      "ArtsId:%lu, Mode:%s, Effective:%s, ValidRank:%d, "
                      "LocalValid:%d, Version:%u]",
                      depv[i].guid, db_temp->arts_id, GET_TYPE_NAME(depv[i].mode),
                      GET_TYPE_NAME(effective_mode), valid_rank, local_valid,
                      db_temp->version);
          } else {
            ARTS_INFO("[AcquireDbs] Non-owner cache miss DB[Guid:%lu, "
                      "Mode:%s, Effective:%s, ValidRank:%d]",
                      depv[i].guid, GET_TYPE_NAME(depv[i].mode),
                      GET_TYPE_NAME(effective_mode), valid_rank);
          }
          if (effective_mode == ARTS_DB_WRITE && depv[i].mode == ARTS_DB_WRITE &&
              local_valid) {
            // Conservative: avoid using possibly stale cached WRITE copies.
            ARTS_INFO("  Non-owner WRITE acquire: invalidating local cached "
                      "copy for DB_WRITE to avoid stale data");
            arts_route_table_invalidate_item(depv[i].guid);
            db_temp = NULL;
            valid_rank = -1;
            local_valid = false;
          }
          if (effective_mode == ARTS_DB_READ && depv[i].mode == ARTS_DB_WRITE &&
              local_valid) {
            // Drop cached READ copies for DB_WRITE types to avoid stale data.
            arts_route_table_invalidate_item(depv[i].guid);
            db_temp = NULL;
            valid_rank = -1;
            local_valid = false;
          }
          if (local_valid && effective_mode != ARTS_DB_WRITE) {
            db_found = db_temp;
            arts_atomic_sub(&edt->depcNeeded, 1U);
            ARTS_INFO("  Found local valid copy, decremented depcNeeded");
          } else if (db_temp && effective_mode == ARTS_DB_WRITE) {
            if (local_valid) {
              db_found = db_temp;
              arts_atomic_sub(&edt->depcNeeded, 1U);
              ARTS_INFO("  Found local valid copy for WRITE; using it");
            } else {
              // For WRITE, non-owner cached copies may be stale; wait for owner.
              ARTS_INFO("  Local copy ignored for WRITE; waiting for owner");
            }
          }
          if (effective_mode == ARTS_DB_WRITE) {
            if (!db_found) {
              // Full DB transfer for exclusive WRITE
              ARTS_INFO("  WRITE mode - sending full DB request to rank %d",
                        owner);
              arts_remote_db_full_request(depv[i].guid, owner, edt->current_edt, i,
                                      effective_mode);
            } else {
              ARTS_INFO("  WRITE mode with local valid copy - no remote request "
                        "needed");
            }
          } else if (!db_temp || !local_valid) {
            // We can aggregate read requests for reads
            ARTS_INFO("  READ mode, no local copy - sending aggregated request "
                      "to rank %d",
                      owner);
            int request_rank = owner;
            arts_remote_db_request(depv[i].guid, request_rank, edt, i, depv[i].mode,
                                true, ARTS_NULL);
          } else {
            ARTS_INFO("  READ mode with local copy - no remote request needed");
          }
        }
      } break;

      case ARTS_NULL:
      default:
        arts_atomic_sub(&edt->depcNeeded, 1U);
        break;
      }

      if (db_found) {
        depv[i].ptr = db_found + 1;
}
      ARTS_DEBUG("DB[Guid:%lu, Ptr:%p] acquired", depv[i].guid, depv[i].ptr);
    } else {
      arts_atomic_sub(&edt->depcNeeded, 1U);
    }
  }
  ARTS_INFO("EDT[Id:%lu, Guid:%lu] has finished acquiring DBs", edt->arts_id,
            edt->current_edt);
}

void prep_dbs(unsigned int depc, arts_edt_dep_t *depv, bool gpu) {
  for (unsigned int i = 0; i < depc; i++) {
    arts_type_t effective_mode = arts_get_effective_mode(&depv[i]);
    if (depv[i].guid != NULL_GUID) {
      if (depv[i].acquire_mode == ARTS_DB_READ) {
        INCREMENT_ACQUIRE_READ_MODE_BY(1);
      } else if (depv[i].acquire_mode == ARTS_DB_WRITE) {
        INCREMENT_ACQUIRE_WRITE_MODE_BY(1);
}
    }

    if (depv[i].guid != NULL_GUID && effective_mode == ARTS_DB_WRITE) {
      if (depv[i].mode != ARTS_DB_PIN) {
        arts_remote_update_route_table(depv[i].guid, -1);
}
      struct arts_db *db = ((struct arts_db *)depv[i].ptr) - 1;
      ARTS_DEBUG("[prep_dbs] DB[Id:%lu, Guid:%lu] ptr=%p, db=%p", db->arts_id,
                 depv[i].guid, depv[i].ptr, db);
    }
#ifdef USE_GPU
    if (!gpu && depv[i].mode == ARTS_DB_LC) {
      struct arts_db *db = ((struct arts_db *)depv[i].ptr) - 1;
      arts_reader_lock(&db->reader, &db->writer);
      internal_inc_db_version(&db->version);
    }

    if (!gpu && depv[i].mode == ARTS_DB_LC_SYNC) {
      struct arts_db *db = ((struct arts_db *)depv[i].ptr) - 1;
      ARTS_DEBUG("internalLCSync %lu %p", depv[i].guid, db);
      internal_lc_sync_gpu(depv[i].guid, db);
    }
#endif
  }
}

void release_dbs(unsigned int depc, arts_edt_dep_t *depv, bool gpu) {
  for (int i = 0; i < depc; i++) {
    ARTS_DEBUG("Releasing DB[Guid:%lu] [Mode:%s]", depv[i].guid,
               GET_TYPE_NAME(depv[i].mode));
    unsigned int owner = arts_guid_get_rank(depv[i].guid);

    /// Check compiler-inferred acquire_mode if provided, otherwise use
    /// allocation mode
    arts_type_t effective_mode = arts_get_effective_mode(&depv[i]);

    if (depv[i].guid != NULL_GUID && effective_mode == ARTS_DB_WRITE) {
      if (depv[i].mode == ARTS_DB_PIN) {
        ARTS_DEBUG("Pinned DB write release (no frontier update)");
        arts_db_decrement_latch(depv[i].guid);
      } else if (owner == arts_global_rank_id) {
        struct arts_db *db = ((struct arts_db *)depv[i].ptr - 1);
        arts_progress_frontier(db, arts_global_rank_id);
        arts_db_decrement_latch(depv[i].guid);
      } else {
        struct arts_db *db = ((struct arts_db *)depv[i].ptr) - 1;
        if (db && (db->header.size - sizeof(struct arts_db)) == 176128) {
          ARTS_INFO("Release WRITE non-owner DB[Id:%lu, Guid:%lu, Size:%lu] "
                    "sending update to owner %u",
                    db->arts_id, depv[i].guid, db->header.size, owner);
        }
        arts_remote_update_db(depv[i].guid, true);
        INCREMENT_OWNER_UPDATES_PERFORMED_BY(1);
      }
    } else if (depv[i].guid != NULL_GUID && effective_mode == ARTS_DB_READ) {
      // READ mode: NO latch decrement, NO owner update
      ARTS_DEBUG("DB[Guid:%lu] released in READ mode (no owner update, no "
                 "latch decrement)",
                 depv[i].guid);
      INCREMENT_OWNER_UPDATES_SAVED_BY(1);
    } else if (depv[i].mode == ARTS_DB_PIN) {
      arts_db_decrement_latch(depv[i].guid);
    } else if (depv[i].mode == ARTS_DB_ONCE_LOCAL ||
               depv[i].mode == ARTS_DB_ONCE) {
      arts_route_table_invalidate_item(depv[i].guid);
    } else if (depv[i].mode == ARTS_PTR) {
      // Only free explicit buffers (guid == NULL). ESD slices point into DBs.
      if (depv[i].guid == NULL_GUID) {
        arts_free(depv[i].ptr);
}
    } else if (!gpu && depv[i].mode == ARTS_DB_LC) {
      struct arts_db *db = ((struct arts_db *)depv[i].ptr) - 1;
      arts_reader_unlock(&db->reader);
    } else {
      if (arts_route_table_return_db(depv[i].guid, depv[i].mode != ARTS_DB_PIN)) {
        ARTS_DEBUG("FREED A COPY - DB[Guid:%lu]", depv[i].guid);
      }
    }
  }
}

bool arts_add_db_duplicate(struct arts_db *db, unsigned int rank,
                        struct arts_edt *edt, arts_guid_t edt_guid,
                        unsigned int slot, arts_type_t mode) {
  bool write = (mode == ARTS_DB_WRITE);
  bool exclusive = false;
  if (edt && edt_guid == NULL_GUID) {
    edt_guid = edt->current_edt;
}
  return arts_push_db_to_list((struct arts_db_list *)db->db_list, rank, write,
                          exclusive, arts_guid_get_rank(db->guid) == rank, false,
                          edt, edt_guid, slot, mode);
}

void internal_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid, unsigned int slot,
                       unsigned int offset, unsigned int size,
                       unsigned int rank) {
  if (rank == arts_global_rank_id) {
    struct arts_db *db = (struct arts_db *)arts_route_table_lookup_item(db_guid);
    if (db) {
      void *data = (void *)(((char *)(db + 1)) + offset);
      void *ptr = arts_malloc(size);
      memcpy(ptr, data, size);
      ARTS_INFO("Getting DB[Guid:%lu] From: %p", db_guid, data);
      if (edt_guid != NULL_GUID) {
        arts_signal_edt_ptr(edt_guid, slot, ptr, size);
}
      ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GET_BW, ARTS_METRIC_THREAD, size);
    } else {
      assert(edt_guid != NULL_GUID && "DB not found and no EDT to signal");
      ARTS_INFO("Getting OO-DB[Guid:%lu] From: %p", db_guid, NULL);
      arts_out_of_order_get_from_db(edt_guid, db_guid, slot, offset, size);
    }
  } else {
    ARTS_DEBUG("Sending DB[Guid:%lu] to Rank %u", db_guid, rank);
    assert(edt_guid != NULL_GUID && "DB not found and no EDT to signal");
    arts_remote_get_from_db(edt_guid, db_guid, slot, offset, size, rank);
  }
}

void arts_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid, unsigned int slot,
                   unsigned int offset, unsigned int size) {
  GET_DB_COUNTER_START();
  unsigned int rank = arts_guid_get_rank(db_guid);
  internal_get_from_db(edt_guid, db_guid, slot, offset, size, rank);
  GET_DB_COUNTER_STOP();
}

void arts_get_from_db_at(arts_guid_t edt_guid, arts_guid_t db_guid, unsigned int slot,
                     unsigned int offset, unsigned int size,
                     unsigned int rank) {
  GET_DB_COUNTER_START();
  internal_get_from_db(edt_guid, db_guid, slot, offset, size, rank);
  GET_DB_COUNTER_STOP();
}

void internal_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                     unsigned int slot, unsigned int offset, unsigned int size,
                     arts_guid_t epoch_guid, unsigned int rank) {
  if (rank == arts_global_rank_id) {
    struct arts_db *db = (struct arts_db *)arts_route_table_lookup_item(db_guid);
    if (db) {
      // Do this so when we increment finished we can check the term status
      increment_queue_epoch(epoch_guid);
      global_shutdown_guid_inc_queue();
      void *data = (void *)(((char *)(db + 1)) + offset);
      memcpy(data, ptr, size);
      if (edt_guid != NULL_GUID) {
        arts_signal_edt(edt_guid, slot, db_guid);
}
      increment_finished_epoch(epoch_guid);
      global_shutdown_guid_inc_finished();
      ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_PUT_BW, ARTS_METRIC_THREAD, size);
    } else {
      void *cpy_ptr = arts_malloc(size);
      memcpy(cpy_ptr, ptr, size);
      arts_out_of_order_put_in_db(cpy_ptr, edt_guid, db_guid, slot, offset, size,
                            epoch_guid);
    }
  } else {
    void *cpy_ptr = arts_malloc(size);
    memcpy(cpy_ptr, ptr, size);
    arts_remote_put_in_db(cpy_ptr, edt_guid, db_guid, slot, offset, size, epoch_guid,
                      rank);
  }
}

void arts_put_in_db_at(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                   unsigned int slot, unsigned int offset, unsigned int size,
                   unsigned int rank) {
  PUT_DB_COUNTER_START();
  arts_guid_t epoch_guid = arts_get_current_epoch_guid();
  ARTS_DEBUG("Epoch [Guid:%lu]", epoch_guid);
  increment_active_epoch(epoch_guid);
  global_shutdown_guid_inc_active();
  internal_put_in_db(ptr, edt_guid, db_guid, slot, offset, size, epoch_guid, rank);
  PUT_DB_COUNTER_STOP();
}

void arts_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                 unsigned int slot, unsigned int offset, unsigned int size) {
  PUT_DB_COUNTER_START();
  unsigned int rank = arts_guid_get_rank(db_guid);
  arts_guid_t epoch_guid = arts_get_current_epoch_guid();
  ARTS_DEBUG("Epoch [Guid:%lu]", epoch_guid);
  increment_active_epoch(epoch_guid);
  global_shutdown_guid_inc_active();
  internal_put_in_db(ptr, edt_guid, db_guid, slot, offset, size, epoch_guid, rank);
  PUT_DB_COUNTER_STOP();
}

void arts_put_in_db_epoch(void *ptr, arts_guid_t epoch_guid, arts_guid_t db_guid,
                      unsigned int offset, unsigned int size) {
  PUT_DB_COUNTER_START();
  unsigned int rank = arts_guid_get_rank(db_guid);
  increment_active_epoch(epoch_guid);
  global_shutdown_guid_inc_active();
  internal_put_in_db(ptr, NULL_GUID, db_guid, 0, offset, size, epoch_guid, rank);
  PUT_DB_COUNTER_STOP();
}
