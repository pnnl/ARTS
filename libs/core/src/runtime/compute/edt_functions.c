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
#include "arts/runtime/compute/edt_functions.h"
#include "arts/utils/malloc.h"

#include <string.h>

#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/introspection/metrics.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/runtime/sync/termination_detection.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/array_list.h"
#include "arts/utils/atomics.h"

#ifdef USE_GPU
#include "arts/gpu/gpu_runtime.cuh"
#endif

#define MAX_EPOCH_ARRAY_LIST 32

extern unsigned int num_numa_domains;

__thread arts_array_list_t *epoch_list = NULL;
__thread struct arts_edt_s *current_edt = NULL;

bool arts_set_current_epoch_guid(arts_guid_t epoch_guid) {
  if (epoch_guid) {
    if (!epoch_list) {
      epoch_list = arts_new_array_list(sizeof(arts_guid_t), 8);
}
    arts_push_to_array_list(epoch_list, &epoch_guid);
    if (current_edt) {
      current_edt->epoch_guid = epoch_guid;
      return true;
    }
  }
  return false;
}

arts_guid_t arts_get_current_epoch_guid() {
  if (epoch_list) {
    uint64_t length = arts_length_array_list(epoch_list);
    if (length) {
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, length - 1);
      return *guid;
    }
  }
  return NULL_GUID;
}

arts_guid_t *arts_check_epoch_is_root(arts_guid_t to_check) {
  if (epoch_list) {
    uint64_t length = arts_length_array_list(epoch_list);
    for (uint64_t i = 0; i < length; i++) {
      arts_guid_t *guid = (arts_guid_t *)arts_get_from_array_list(epoch_list, i);
      if (*guid == to_check) {
        return guid;
}
    }
  }
  ARTS_INFO("ERROR %lu is not a valid epoch", to_check);
  return NULL;
}

void arts_set_thread_local_edt_info(struct arts_edt_s *edt) {
  arts_thread_info.current_edt_guid = edt->current_edt;
  current_edt = edt;

  if (epoch_list) {
    arts_reset_array_list(epoch_list);
}

  arts_set_current_epoch_guid(current_edt->epoch_guid);
}

void arts_save_thread_local(thread_local_t *tl) {
  if (current_edt) {
    EDT_COUNTER_STOP();
  }

  CONTEXT_SWITCH_START();
  tl->current_edt_guid = arts_thread_info.current_edt_guid;
  tl->current_edt = current_edt;
  tl->epoch_list = (void *)epoch_list;

  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
  epoch_list = NULL;
  CONTEXT_SWITCH_STOP();
  ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_YIELD_BW, ARTS_METRIC_THREAD, 1);
}

void arts_restore_thread_local(thread_local_t *tl) {
  CONTEXT_SWITCH_START();
  arts_thread_info.current_edt_guid = tl->current_edt_guid;
  current_edt = tl->current_edt;
  if (epoch_list) {
    arts_delete_array_list(epoch_list);
}
  epoch_list = (arts_array_list_t *)tl->epoch_list;
  CONTEXT_SWITCH_STOP();

  EDT_COUNTER_START();
}

void arts_increment_finished_epoch_list() {
  if (epoch_list) {

    unsigned int epoch_array_length = arts_length_array_list(epoch_list);
    for (unsigned int i = 0; i < epoch_array_length; i++) {
      arts_guid_t *guid = (arts_guid_t *)arts_get_from_array_list(epoch_list, i);
      uint64_t current_id = current_edt ? current_edt->arts_id : 0;
      ARTS_INFO("Current EDT[Id:%lu, Guid:%lu] - Unsetting Epoch [Guid:%lu]",
                current_id, arts_thread_info.current_edt_guid, *guid);
      if (*guid) {
        increment_finished_epoch(*guid);
}
    }

    if (epoch_array_length > MAX_EPOCH_ARRAY_LIST) {
      arts_delete_array_list(epoch_list);
      epoch_list = NULL;
    } else {
      arts_reset_array_list(epoch_list);
}
  }
  global_shutdown_guid_inc_finished();
}

void arts_unset_thread_local_edt_info() {
  arts_increment_finished_epoch_list();
  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
}

bool arts_edt_create_internal(struct arts_edt_s *edt, arts_type_t mode,
                           arts_guid_t *guid, unsigned int route,
                           unsigned int cluster, unsigned int edt_space,
                           arts_guid_t output_buffer, arts_edt_t func_ptr,
                           uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                           bool use_epoch, arts_guid_t epoch_guid, bool has_depv,
                           uint64_t arts_id) {
  if (!edt) {
    edt = (struct arts_edt_s *)ARTS_CALLOC_ALIGN_WITH_TYPE(1, edt_space, 16,
                                                    ARTS_METRIC_EDT_MEMORY_SIZE);
}
  edt->header.type = mode;
  edt->header.size = edt_space;
  edt->arts_id = arts_id; // Set compiler-assigned arts_id (0 if not set)
  if (edt) {
    bool created_guid = false;
    if (*guid == NULL_GUID) {
      created_guid = true;
      edt->current_edt = *guid = arts_guid_create_for_rank(route, mode);
    } else {
      edt->current_edt = *guid;
}

    edt->func_ptr = func_ptr;
    edt->depc = (has_depv) ? depc : 0;
    edt->paramc = paramc;
    edt->output_buffer = output_buffer;
    edt->epoch_guid = NULL_GUID;
    edt->cluster = cluster;
    edt->depcNeeded = depc;

    if (use_epoch) {
      arts_guid_t current_epoch_guid = NULL_GUID;
      if (epoch_guid && arts_check_epoch_is_root(epoch_guid)) {
        current_epoch_guid = epoch_guid;
      } else {
        current_epoch_guid = arts_get_current_epoch_guid();
}

      if (current_epoch_guid) {
        edt->epoch_guid = current_epoch_guid;
        increment_active_epoch(current_epoch_guid);
      }
    }
    global_shutdown_guid_inc_active();

    if (paramc) {
      unsigned int offset =
          edt_space - (depc * sizeof(arts_type_t) + depc * sizeof(arts_edt_dep_t) + paramc * sizeof(uint64_t));
      char *tmp = (char *)edt + offset;
      memcpy(tmp, paramv, sizeof(uint64_t) * paramc);
    }

    /// DEBUG
    if (use_epoch) {
      ARTS_INFO("Creating EDT[Id:%lu, Guid:%lu, Epoch:%lu, Deps:%u, Route:%d]",
                edt->arts_id, *guid, edt->epoch_guid, edt->depc, route);
    } else {
      ARTS_INFO("Created EDT[Id:%lu, Guid:%lu, Route:%d]", edt->arts_id, *guid,
                route);
    }

    if (route != arts_global_rank_id) {
      arts_remote_memory_move(route, *guid, (void *)edt,
                           (unsigned int)edt->header.size,
                           ARTS_REMOTE_EDT_MOVE_MSG, arts_free);
    } else {
      INC_OUSTANDING_EDTS(1);
      if (created_guid) {
        arts_route_table_add_item(edt, *guid, arts_global_rank_id, false);
        if (edt->depcNeeded == 0) {
          arts_handle_ready_edt(edt);
}
      } else {
        arts_route_table_add_item_race(edt, *guid, arts_global_rank_id, false);
        if (edt->depcNeeded) {
          arts_route_table_fire_oo(*guid, arts_out_of_order_handler);
        } else {
          arts_handle_ready_edt(edt);
}
      }
    }

    INCREMENT_NUM_EDTS_CREATED_BY(1);
    return true;
  }
  return false;
}

arts_guid_t arts_edt_create_dep(arts_edt_t func_ptr, uint32_t paramc,
                            const uint64_t *paramv, uint32_t depc,
                            bool has_depv, const arts_hint_t *hint) {
  EDT_CREATE_COUNTER_START();
  unsigned int route =
      (hint && hint->route != ARTS_HINT_CURRENT_NODE) ? hint->route : arts_global_rank_id;
  uint64_t arts_id = hint ? hint->id : 0;
  unsigned int dep_space = (has_depv) ? depc * sizeof(arts_edt_dep_t) : 0;
  unsigned int mode_space = (has_depv) ? depc * sizeof(arts_type_t) : 0;
  unsigned int edt_space =
      sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t)) + dep_space + mode_space;
  arts_guid_t guid = NULL_GUID;
  arts_guid_t *guid_ptr = &guid;
  bool created = arts_edt_create_internal(
      NULL, ARTS_EDT, guid_ptr, route, arts_thread_info.cluster_id, edt_space,
      NULL_GUID, func_ptr, paramc, paramv, depc, true, NULL_GUID, has_depv, arts_id);
  EDT_CREATE_COUNTER_STOP();
  return guid;
}

arts_guid_t arts_edt_create_with_guid_dep(arts_edt_t func_ptr, arts_guid_t guid,
                                    uint32_t paramc, const uint64_t *paramv,
                                    uint32_t depc, bool has_depv) {
  EDT_CREATE_COUNTER_START();
  unsigned int route = arts_guid_get_rank(guid);
  unsigned int dep_space = (has_depv) ? depc * sizeof(arts_edt_dep_t) : 0;
  unsigned int mode_space = (has_depv) ? depc * sizeof(arts_type_t) : 0;
  unsigned int edt_space =
      sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t)) + dep_space + mode_space;
  bool ret = arts_edt_create_internal(
      NULL, ARTS_EDT, &guid, route, arts_thread_info.cluster_id, edt_space,
      NULL_GUID, func_ptr, paramc, paramv, depc, true, NULL_GUID, has_depv, 0);
  EDT_CREATE_COUNTER_STOP();
  return (ret) ? guid : NULL_GUID;
}

arts_guid_t arts_edt_create_with_epoch_dep(arts_edt_t func_ptr,
                                     uint32_t paramc, const uint64_t *paramv,
                                     uint32_t depc, arts_guid_t epoch_guid,
                                     bool has_depv, const arts_hint_t *hint) {
  EDT_CREATE_COUNTER_START();
  unsigned int route =
      (hint && hint->route != ARTS_HINT_CURRENT_NODE) ? hint->route : arts_global_rank_id;
  uint64_t arts_id = hint ? hint->id : 0;
  unsigned int dep_space = (has_depv) ? depc * sizeof(arts_edt_dep_t) : 0;
  unsigned int mode_space = (has_depv) ? depc * sizeof(arts_type_t) : 0;
  unsigned int edt_space =
      sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t)) + dep_space + mode_space;
  arts_guid_t guid = NULL_GUID;
  bool created = arts_edt_create_internal(
      NULL, ARTS_EDT, &guid, route, arts_thread_info.cluster_id, edt_space,
      NULL_GUID, func_ptr, paramc, paramv, depc, true, epoch_guid, has_depv, arts_id);
  EDT_CREATE_COUNTER_STOP();
  return guid;
}

arts_guid_t arts_edt_create(arts_edt_t func_ptr, uint32_t paramc,
                         const uint64_t *paramv, uint32_t depc,
                         const arts_hint_t *hint) {
  return arts_edt_create_dep(func_ptr, paramc, paramv, depc, true, hint);
}

arts_guid_t arts_edt_create_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                 uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc) {
  return arts_edt_create_with_guid_dep(func_ptr, guid, paramc, paramv, depc, true);
}

arts_guid_t arts_edt_create_with_epoch(arts_edt_t func_ptr, uint32_t paramc,
                                  const uint64_t *paramv, uint32_t depc,
                                  arts_guid_t epoch_guid,
                                  const arts_hint_t *hint) {
  return arts_edt_create_with_epoch_dep(func_ptr, paramc, paramv, depc,
                                   epoch_guid, true, hint);
}


void arts_edt_free(struct arts_edt_s *edt) {
  arts_thread_info.edt_free = 1;
  arts_free(edt);
  arts_thread_info.edt_free = 0;
}

void arts_edt_delete(struct arts_edt_s *edt) {
  if (!edt) {
    ARTS_INFO("EDT delete called with NULL edt on rank %u", arts_global_rank_id);
    return;
  }
  ARTS_INFO("EDT delete [Guid:%lu, Id:%lu, Depc:%u, DepcNeeded:%u] on rank %u",
            edt->current_edt, edt->arts_id, edt->depc, edt->depcNeeded,
            arts_global_rank_id);
  arts_route_table_remove_item(edt->current_edt);
  arts_edt_free(edt);
}

void arts_edt_destroy(arts_guid_t guid) {
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_route_table_lookup_item(guid);
  if (!edt) {
    ARTS_INFO("EDT destroy missing [Guid:%lu] on rank %u", guid,
              arts_global_rank_id);
    return;
  }
  ARTS_INFO("EDT destroy [Guid:%lu, Id:%lu, Depc:%u, DepcNeeded:%u] on rank %u",
            edt->current_edt, edt->arts_id, edt->depc, edt->depcNeeded,
            arts_global_rank_id);
  arts_route_table_remove_item(guid);
  arts_edt_free(edt);
}

void *arts_get_depv(void *edt_ptr) {
  struct arts_edt_s *edt = (struct arts_edt_s *)edt_ptr;
  unsigned int paramc = edt->paramc;
  if (edt->header.type == ARTS_EDT) {
    struct arts_edt_s *edt = (struct arts_edt_s *)edt_ptr;
    return (void *)((uint64_t *)(edt + 1) + paramc);
  }
#ifdef USE_GPU
  if (edt->header.type == ARTS_GPU_EDT) {
    arts_gpu_edt_t *edtGpu = (arts_gpu_edt_t *)edt_ptr;
    return (void *)((uint64_t *)(edtGpu + 1) + paramc);
  }
#endif
  return NULL;
}

arts_type_t *arts_get_dep_modes(void *edt_ptr) {
  struct arts_edt_s *edt = (struct arts_edt_s *)edt_ptr;
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt_ptr);
  if (!depv) {
    return NULL;
  }
  return (arts_type_t *)(depv + edt->depc);
}

void internal_signal_edt(arts_guid_t edt_packet, uint32_t slot, arts_guid_t data_guid,
                       arts_type_t mode, void *ptr, unsigned int size) {
  SIGNAL_EDT_COUNTER_START();
  // This is old CDAG code...
  if (current_edt && current_edt->invalidateCount > 0) {
    if (mode == ARTS_PTR) {
      arts_out_of_order_signal_edt_with_ptr(edt_packet, data_guid, ptr, size, slot);
    } else {
      arts_out_of_order_signal_edt(current_edt->current_edt, edt_packet, data_guid, slot,
                              mode, true);
}
  } else {
    unsigned int rank = arts_guid_get_rank(edt_packet);
    if (rank == arts_global_rank_id) {
      struct arts_edt_s *edt =
          (struct arts_edt_s *)arts_route_table_lookup_item(edt_packet);
      if (edt) {
        arts_edt_dep_t *edt_dep = (arts_edt_dep_t *)arts_get_depv(edt);
        arts_type_t *modes = arts_get_dep_modes(edt);
        if (slot < edt->depc) {
          edt_dep[slot].guid = data_guid;
          edt_dep[slot].ptr = ptr;
          modes[slot] = mode;
        }
        unsigned int res = arts_atomic_sub(&edt->depcNeeded, 1U);
        ARTS_INFO("Signal DB[Guid:%lu] to EDT[Guid:%lu, Slot:%u, "
                  "DepCount:%d]",
                  data_guid, edt->current_edt, slot, res);
        if (res == 0) {
          arts_handle_ready_edt(edt);
}
      } else {
        if (mode == ARTS_PTR) {
          arts_out_of_order_signal_edt_with_ptr(edt_packet, data_guid, ptr, size, slot);
        } else {
          arts_out_of_order_signal_edt(edt_packet, edt_packet, data_guid, slot, mode,
                                  false);
}
      }
    } else {
      if (mode == ARTS_PTR) {
        arts_remote_signal_edt_with_ptr(edt_packet, data_guid, ptr, size, slot);
      } else {
        arts_remote_signal_edt(edt_packet, data_guid, slot, mode);
}
    }
  }
  ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_EDT_SIGNAL_THROUGHPUT, ARTS_METRIC_THREAD, 1);
  SIGNAL_EDT_COUNTER_STOP();
}

void arts_signal_edt(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid) {
  internal_signal_edt(edt_guid, slot, data_guid, ARTS_DB_WRITE, NULL, 0);
}

// Internal function to signal EDT with explicit access mode
void internal_signal_edt_with_mode(arts_guid_t edt_packet, uint32_t slot,
                               arts_guid_t data_guid, arts_type_t mode) {
  SIGNAL_EDT_COUNTER_START();
  // This is old CDAG code...
  if (current_edt && current_edt->invalidateCount > 0) {
    if (mode == ARTS_PTR) {
      arts_out_of_order_signal_edt_with_ptr(edt_packet, data_guid, NULL, 0, slot);
    } else {
      arts_out_of_order_signal_edt(current_edt->current_edt, edt_packet, data_guid, slot,
                              mode, true);
}
  } else {
    unsigned int rank = arts_guid_get_rank(edt_packet);
    if (rank == arts_global_rank_id) {
      struct arts_edt_s *edt =
          (struct arts_edt_s *)arts_route_table_lookup_item(edt_packet);
      if (edt) {
        arts_edt_dep_t *edt_dep = (arts_edt_dep_t *)arts_get_depv(edt);
        arts_type_t *modes = arts_get_dep_modes(edt);
        if (slot < edt->depc) {
          edt_dep[slot].guid = data_guid;
          edt_dep[slot].ptr = NULL;
          modes[slot] = mode;
        }
        unsigned int res = arts_atomic_sub(&edt->depcNeeded, 1U);
        ARTS_INFO("Signal DB[Guid:%lu] to EDT[Guid:%lu, Slot:%u, "
                  "DepCount:%d, Mode:%s]",
                  data_guid, edt->current_edt, slot, res,
                  GET_TYPE_NAME(mode));
        if (res == 0) {
          arts_handle_ready_edt(edt);
}
      } else {
        if (mode == ARTS_PTR) {
          arts_out_of_order_signal_edt_with_ptr(edt_packet, data_guid, NULL, 0, slot);
        } else {
          arts_out_of_order_signal_edt(edt_packet, edt_packet, data_guid, slot, mode,
                                  false);
}
      }
    } else {
      if (mode == ARTS_PTR) {
        arts_remote_signal_edt_with_ptr(edt_packet, data_guid, NULL, 0, slot);
      } else {
        arts_remote_signal_edt(edt_packet, data_guid, slot, mode);
}
    }
  }
  ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_EDT_SIGNAL_THROUGHPUT, ARTS_METRIC_THREAD, 1);
  SIGNAL_EDT_COUNTER_STOP();
}

void arts_signal_edt_value(arts_guid_t edt_guid, uint32_t slot, uint64_t value) {
  internal_signal_edt(edt_guid, slot, (arts_guid_t)value, ARTS_SINGLE_VALUE, NULL, 0);
}

void arts_signal_edt_ptr(arts_guid_t edt_guid, uint32_t slot, void *ptr,
                      unsigned int size) {
  internal_signal_edt(edt_guid, slot, NULL_GUID, ARTS_PTR, ptr, size);
}

void arts_signal_edt_ptr_with_guid(arts_guid_t edt_guid, uint32_t slot,
                              arts_guid_t db_guid, void *ptr, unsigned int size) {
  internal_signal_edt(edt_guid, slot, db_guid, ARTS_PTR, ptr, size);
}

void arts_signal_edt_null(arts_guid_t edt_guid, uint32_t slot) {
  internal_signal_edt(edt_guid, slot, NULL_GUID, ARTS_NULL, NULL, 0);
}

arts_guid_t arts_allocate_local_buffer(void **buffer, unsigned int size,
                                   unsigned int uses, arts_guid_t epoch_guid) {
  if (epoch_guid) {
    increment_active_epoch(epoch_guid);
}
  global_shutdown_guid_inc_active();

  // unsigned int alloc = 0;
  if (size) {
    if (*buffer == NULL) {
      *buffer = (char *)arts_malloc(sizeof(char) * size);
      // alloc = 1;
    }
  }

  arts_buffer_t *stub = (arts_buffer_t *)arts_malloc(sizeof(arts_buffer_t));
  stub->buffer = (buffer) ? *buffer : NULL;
  stub->size_to_write = NULL;
  stub->size = size;
  stub->uses = uses;
  stub->epoch_guid = epoch_guid;

  arts_guid_t guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_BUFFER);
  arts_route_table_add_item(stub, guid, arts_global_rank_id, false);
  return guid;
}

void *arts_set_buffer(arts_guid_t buffer_guid, void *buffer, unsigned int size) {
  void *ret = NULL;
  unsigned int rank = arts_guid_get_rank(buffer_guid);
  if (rank == arts_global_rank_id) {
    arts_buffer_t *stub = (arts_buffer_t *)arts_route_table_lookup_item(buffer_guid);
    if (stub) {
      arts_guid_t epoch_guid = stub->epoch_guid;
      if (epoch_guid) {
        increment_queue_epoch(epoch_guid);
}
      global_shutdown_guid_inc_queue();

      if (size > stub->size) {
        if (stub->size) {
          ARTS_INFO("Truncating buffer data buffer size: %u stub size: %u",
                    size, stub->size);
          arts_debug_print_stack();
        } else if (stub->buffer == NULL) {
          stub->buffer = (char *)arts_malloc(sizeof(char) * size);
          stub->size = size;
        } else {
          stub->size = size;
}
      }

      if (stub->size_to_write) {
        *stub->size_to_write = (uint32_t)size;
}

      if (stub->buffer) {
        memcpy(stub->buffer, buffer, stub->size);
        ARTS_DEBUG("Set buffer [Ptr:%p, Size:%u, Uses: %u]", stub->buffer,
                   *((unsigned int *)stub->buffer), stub->size);
        ret = stub->buffer;
      } else {
        ret = NULL;
}

      if (!arts_atomic_sub(&stub->uses, 1)) {
        arts_route_table_remove_item(buffer_guid);
        arts_free(stub);
      }

      if (epoch_guid) {
        increment_finished_epoch(epoch_guid);
}
      global_shutdown_guid_inc_finished();
    } else {
      ARTS_INFO("Out-of-order buffers not supported");
    }
  } else {
    arts_remote_memory_move(rank, buffer_guid, buffer, size,
                         ARTS_REMOTE_BUFFER_SEND_MSG, arts_free);
  }
  return ret;
}

void *arts_get_buffer(arts_guid_t buffer_guid) {
  void *buffer = NULL;
  if (arts_guid_is_local(buffer_guid)) {
    arts_buffer_t *stub = (arts_buffer_t *)arts_route_table_lookup_item(buffer_guid);
    buffer = stub->buffer;
    if (!arts_atomic_sub(&stub->uses, 1)) {
      arts_route_table_remove_item(buffer_guid);
      arts_free(stub);
    }
  }
  return buffer;
}

void *arts_block_for_buffer(arts_guid_t buffer_guid) {
  void *buffer = NULL;
  if (arts_guid_is_local(buffer_guid)) {
    arts_buffer_t *stub = (arts_buffer_t *)arts_route_table_lookup_item(buffer_guid);
    while (stub->uses > 1) {
      ARTS_DEBUG("Yield: [Uses: %u]", stub->uses);
      arts_yield();
    }
    buffer = stub->buffer;
    if (!arts_atomic_sub(&stub->uses, 1)) {
      arts_route_table_remove_item(buffer_guid);
      arts_free(stub);
    }
  }
  return buffer;
}

volatile uint64_t outstanding_edts = 0;
void check_out_edts(uint64_t threashold) {
  static uint64_t count = 0;
  if (arts_atomic_fetch_add_u64(&count, 1) + 1 == threashold) {
    arts_atomic_fetch_sub_u64(&count, threashold);
  }
}

void arts_lc_sync(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid) {
  arts_type_t mode = arts_guid_get_type(data_guid);
  if (mode == ARTS_DB_LC) {
    mode = ARTS_DB_LC_SYNC;
}
  internal_signal_edt(edt_guid, slot, data_guid, ARTS_DB_LC_SYNC, NULL, 0);
}

void arts_gpu_signal_edt_memset(arts_guid_t edt_guid, uint32_t slot,
                            arts_guid_t data_guid) {
  arts_type_t mode = arts_guid_get_type(data_guid);
  if (mode == ARTS_DB_GPU_WRITE) {
    mode = ARTS_DB_GPU_MEMSET;
  } else if (mode == ARTS_DB_LC) {
    mode = ARTS_DB_LC_NO_COPY;
}
  internal_signal_edt(edt_guid, slot, data_guid, mode, NULL, 0);
}
