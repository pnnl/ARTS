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
#include "arts/compute/edt.h"
#include "arts/utils/malloc.h"

#include <string.h>

#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/remote/handler.h"
#include "arts/runtime_state.h"
#include "arts/sync/termination.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/array_list.h"
#include "arts/utils/atomics.h"

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#endif

#ifdef ARTS_USE_CXL
#include "arts/cxl/wrapper.h"
#endif

#define MAX_EPOCH_ARRAY_LIST 32

extern unsigned int num_numa_domains;

ARTS_THREAD_LOCAL arts_array_list_t *epoch_list = NULL;
ARTS_THREAD_LOCAL struct arts_edt_s *current_edt = NULL;
ARTS_THREAD_LOCAL arts_array_list_t *created_db_list = NULL;

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
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, i);
      if (*guid == to_check) {
        return guid;
      }
    }
  }
  ARTS_INFO("ERROR %lu is not a valid epoch", to_check);
  return NULL;
}

void arts_track_created_db(arts_guid_t guid) {
  if (!created_db_list) {
    created_db_list = arts_new_array_list(sizeof(arts_guid_t), 65536);
  }
  arts_push_to_array_list(created_db_list, &guid);
}

arts_array_list_t *arts_get_created_db_list(void) { return created_db_list; }

void arts_set_thread_local_edt_info(struct arts_edt_s *edt) {
  arts_thread_info.current_edt_guid = edt->current_edt;
  current_edt = edt;

  if (epoch_list) {
    arts_reset_array_list(epoch_list);
  }

  if (created_db_list) {
    arts_reset_array_list(created_db_list);
  }

  arts_set_current_epoch_guid(current_edt->epoch_guid);
}

void arts_save_thread_local(thread_local_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  tl->current_edt_guid = arts_thread_info.current_edt_guid;
  tl->current_edt = current_edt;
  tl->epoch_list = (void *)epoch_list;
  tl->created_db_list = (void *)created_db_list;

  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
  epoch_list = NULL;
  created_db_list = NULL;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_restore_thread_local(thread_local_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  arts_thread_info.current_edt_guid = tl->current_edt_guid;
  current_edt = tl->current_edt;
  if (epoch_list) {
    arts_delete_array_list(epoch_list);
  }
  epoch_list = (arts_array_list_t *)tl->epoch_list;
  if (created_db_list) {
    arts_delete_array_list(created_db_list);
  }
  created_db_list = (arts_array_list_t *)tl->created_db_list;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_cleanup_edt_tls() {
  if (epoch_list) {
    arts_delete_array_list(epoch_list);
    epoch_list = NULL;
  }
  if (created_db_list) {
    arts_delete_array_list(created_db_list);
    created_db_list = NULL;
  }
}

void arts_increment_finished_epoch_list() {
  if (epoch_list) {

    unsigned int epoch_array_length = arts_length_array_list(epoch_list);
    for (unsigned int i = 0; i < epoch_array_length; i++) {
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, i);
#if ARTS_LOG_LEVEL >= 2
      uint64_t current_id = current_edt ? current_edt->arts_id : 0;
      ARTS_INFO("Current EDT[Id:%lu, Guid:%lu] - Unsetting Epoch [Guid:%lu]",
                current_id, arts_thread_info.current_edt_guid, *guid);
#endif
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
  arts_shutdown_epoch_inc_finished();
}

void arts_unset_thread_local_edt_info() {
  arts_increment_finished_epoch_list();
  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
}

/*
 * arts_edt_create_internal — Core EDT allocation and registration.
 *
 * Allocates the EDT struct (header + paramv + depv + modes), assigns its GUID,
 * copies parameters, registers with the epoch system, and places the EDT into
 * the route table so that incoming signals can find it.
 *
 * Two paths exist depending on whether a GUID was pre-reserved:
 *   1. New GUID (created_guid == true):
 *        - arts_route_table_add_item (no race — nobody else knows the GUID
 * yet).
 *        - If depc == 0, the EDT is immediately ready.
 *   2. Pre-reserved GUID (created_guid == false):
 *        - The GUID may already have received out-of-order signals while it was
 *          in RESERVED state. A sentinel (+1 on depc_needed) prevents premature
 *          firing during the race window between route-table insertion and
 *          OOO replay.  See the inline comments for the full protocol.
 *
 * Concurrency notes:
 *   - depc_needed is the primary synchronisation counter.  Every satisfied
 *     dependency atomically decrements it; exactly one thread observes 0
 *     and calls arts_handle_ready_edt.
 *   - The EDT must NOT be visible (in the route table) while its fields
 *     are still being written.
 */
bool arts_edt_create_internal(struct arts_edt_s *edt, arts_type_t mode,
                              arts_guid_t *guid, unsigned int route,
                              unsigned int numa_domain, unsigned int edt_space,
                              arts_guid_t output_buffer, arts_edt_t func_ptr,
                              uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, bool use_epoch,
                              arts_guid_t epoch_guid, bool has_depv,
                              uint64_t arts_id) {
  if (!edt) {
    edt = (struct arts_edt_s *)arts_calloc_align(1, edt_space, 16);
  }
  if (!edt) {
    ARTS_ERROR("EDT allocation failed (size=%u)", edt_space);
  }

  edt->header.type = mode;
  edt->header.size = edt_space;
  edt->arts_id = arts_id;

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
  edt->numa_domain = numa_domain;
  edt->depc_needed = depc;

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
  arts_shutdown_epoch_inc_active();

  /* Copy inline parameter values into the EDT's trailing storage.
   * Layout: [arts_edt_s | paramv[paramc] | depv[depc]]
   * paramv starts immediately after the struct header. */
  if (paramc) {
    unsigned int offset = sizeof(struct arts_edt_s);
    ARTS_DEBUG("EDT paramv copy: edt=%p offset=%u paramc=%u depc=%u "
               "edt_space=%u dep_size=%zu",
               (void *)edt, offset, paramc, depc, edt_space,
               depc * sizeof(arts_edt_dep_t));
    char *tmp = (char *)edt + offset;
    memcpy(tmp, paramv, sizeof(uint64_t) * paramc);
  }

  ARTS_INFO("EDT create [Guid:%lu, Id:%lu, Depc:%u, Route:%u, "
            "PreReserved:%s, Epoch:%lu, FuncPtr:%p]",
            *guid, edt->arts_id, edt->depc, route, created_guid ? "no" : "yes",
            edt->epoch_guid, (void *)func_ptr);

  if (route != arts_global_rank_id) {
    /* Remote EDT: serialise and send to the target node. */
    ARTS_INFO("EDT[Guid:%lu] remote move to rank %u", *guid, route);
    arts_remote_memory_move(route, *guid, (void *)edt,
                            (unsigned int)edt->header.size,
                            ARTS_REMOTE_EDT_MOVE_MSG, arts_free);
  } else {
    /* Local EDT: register in the route table and check readiness. */
    INC_OUTSTANDING_EDTS(1);
    if (created_guid) {
      /* New GUID path — no race, safe non-atomic insert. */
      arts_route_table_add_item(edt, *guid, arts_global_rank_id, false);
      if (edt->depc_needed == 0) {
        ARTS_INFO("EDT[Guid:%lu] immediately ready (depc=0)", *guid);
        arts_handle_ready_edt(edt);
      } else {
        ARTS_DEBUG("EDT[Guid:%lu] waiting for %u deps", *guid,
                   edt->depc_needed);
      }
    } else {
      /*
       * Pre-reserved GUID path — other threads may already hold this GUID
       * and could have queued out-of-order (OOO) signals.
       *
       * Protocol:
       *   1. Set depc_needed = depc + 1  (sentinel prevents premature 0)
       *   2. Insert into route table (EDT is now globally visible)
       *   3. Replay any queued OOO signals (they decrement depc_needed)
       *   4. Atomically remove sentinel (-1); if result is 0, all deps
       *      were already satisfied and we fire the EDT.
       *
       * Exactly one thread (either the OOO replay callback or us at step 4)
       * will observe depc_needed == 0 and call arts_handle_ready_edt.
       */
      edt->depc_needed = depc + 1;
      ARTS_INFO("EDT[Guid:%lu] pre-reserved path: sentinel depc_needed=%u",
                *guid, edt->depc_needed);
      arts_route_table_add_item_race(edt, *guid, arts_global_rank_id, false);
      arts_route_table_fire_oo(*guid, arts_out_of_order_handler);
      unsigned int remaining = arts_atomic_sub(&edt->depc_needed, 1U);
      ARTS_INFO("EDT[Guid:%lu] sentinel removed: depc_needed=%u", *guid,
                remaining);
      if (remaining == 0) {
        arts_handle_ready_edt(edt);
      }
    }
  }

  INCREMENT_NUM_EDT_CREATE_BY(1);
  return true;
}

arts_guid_t arts_edt_create_dep(arts_edt_t func_ptr, uint32_t paramc,
                                const uint64_t *paramv, uint32_t depc,
                                bool has_depv, const arts_hint_t *hint) {
  TIME_EDT_CREATE_START();
  unsigned int route = (hint && hint->route != ARTS_HINT_CURRENT_NODE)
                           ? hint->route
                           : arts_global_rank_id;
  uint64_t arts_id = hint ? hint->id : 0;
  unsigned int dep_space = (has_depv) ? depc * sizeof(arts_edt_dep_t) : 0;
  unsigned int edt_space =
      sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t)) + dep_space;
  arts_guid_t guid = NULL_GUID;
  arts_guid_t *guid_ptr = &guid;
  bool created = arts_edt_create_internal(
      NULL, ARTS_EDT, guid_ptr, route, arts_thread_info.numa_domain_id,
      edt_space, NULL_GUID, func_ptr, paramc, paramv, depc, true, NULL_GUID,
      has_depv, arts_id);
  TIME_EDT_CREATE_STOP();
  return guid;
}

arts_guid_t arts_edt_create_with_guid_dep(arts_edt_t func_ptr, arts_guid_t guid,
                                          uint32_t paramc,
                                          const uint64_t *paramv, uint32_t depc,
                                          bool has_depv) {
  TIME_EDT_CREATE_START();
  unsigned int route = arts_guid_get_rank(guid);
  unsigned int dep_space = (has_depv) ? depc * sizeof(arts_edt_dep_t) : 0;
  unsigned int edt_space =
      sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t)) + dep_space;
  bool ret = arts_edt_create_internal(
      NULL, ARTS_EDT, &guid, route, arts_thread_info.numa_domain_id, edt_space,
      NULL_GUID, func_ptr, paramc, paramv, depc, true, NULL_GUID, has_depv, 0);
  TIME_EDT_CREATE_STOP();
  return (ret) ? guid : NULL_GUID;
}

arts_guid_t arts_edt_create_with_epoch_dep(
    arts_edt_t func_ptr, uint32_t paramc, const uint64_t *paramv, uint32_t depc,
    arts_guid_t epoch_guid, bool has_depv, const arts_hint_t *hint) {
  TIME_EDT_CREATE_START();
  unsigned int route = (hint && hint->route != ARTS_HINT_CURRENT_NODE)
                           ? hint->route
                           : arts_global_rank_id;
  uint64_t arts_id = hint ? hint->id : 0;
  unsigned int dep_space = (has_depv) ? depc * sizeof(arts_edt_dep_t) : 0;
  unsigned int edt_space =
      sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t)) + dep_space;
  arts_guid_t guid = NULL_GUID;
  bool created = arts_edt_create_internal(
      NULL, ARTS_EDT, &guid, route, arts_thread_info.numa_domain_id, edt_space,
      NULL_GUID, func_ptr, paramc, paramv, depc, true, epoch_guid, has_depv,
      arts_id);
  TIME_EDT_CREATE_STOP();
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
  return arts_edt_create_with_guid_dep(func_ptr, guid, paramc, paramv, depc,
                                       true);
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
    ARTS_INFO("EDT delete called with NULL edt on rank %u",
              arts_global_rank_id);
    return;
  }
  ARTS_INFO("EDT delete [Guid:%lu, Id:%lu, Depc:%u, DepcNeeded:%u] on rank %u",
            edt->current_edt, edt->arts_id, edt->depc, edt->depc_needed,
            arts_global_rank_id);
  arts_route_table_remove_item(edt->current_edt);
  arts_edt_free(edt);
}

void arts_edt_destroy(arts_guid_t guid) {
  struct arts_edt_s *edt =
      (struct arts_edt_s *)arts_route_table_lookup_item(guid);
  if (!edt) {
    ARTS_INFO("EDT destroy missing [Guid:%lu] on rank %u", guid,
              arts_global_rank_id);
    return;
  }
  ARTS_INFO("EDT destroy [Guid:%lu, Id:%lu, Depc:%u, DepcNeeded:%u] on rank %u",
            edt->current_edt, edt->arts_id, edt->depc, edt->depc_needed,
            arts_global_rank_id);
  arts_route_table_remove_item(guid);
  arts_edt_free(edt);
}

void *arts_get_depv(void *edt_ptr) {
  struct arts_edt_s *edt = (struct arts_edt_s *)edt_ptr;
  unsigned int paramc = edt->paramc;
  if (edt->edt_type == ARTS_EDT_GPU) {
#ifdef ARTS_USE_GPU
    arts_gpu_edt_t *edtGpu = (arts_gpu_edt_t *)edt_ptr;
    return (void *)((uint64_t *)(edtGpu + 1) + paramc);
#else
    return NULL;
#endif
  }
  return (void *)((uint64_t *)(edt + 1) + paramc);
}

/* arts_get_dep_modes removed — mode now lives in arts_edt_dep_t.mode */

/*
 * arts_set_dep_mode — Write access mode to an EDT dep slot without signaling.
 *
 * This is the "mode-set" half of the two-message add_dependence pattern.
 * It sets depv[slot].mode on the target EDT.  It does NOT decrement
 * depc_needed and does NOT deliver data.
 *
 * Local EDT: direct write.  Remote EDT: forward via network message.
 * Not-yet-created EDT: queue via OOO (OO_SIGNAL_EDT with NULL data —
 * the OOO replay will call internal_signal_edt which writes mode).
 */
void arts_set_dep_mode(arts_guid_t edt_guid, uint32_t slot,
                       arts_db_access_mode_t mode) {
  unsigned int rank = arts_guid_get_rank(edt_guid);
  if (rank == arts_global_rank_id) {
    struct arts_edt_s *edt =
        (struct arts_edt_s *)arts_route_table_lookup_item(edt_guid);
    if (edt) {
      arts_edt_dep_t *edt_dep = (arts_edt_dep_t *)arts_get_depv(edt);
      if (slot < edt->depc) {
        edt_dep[slot].mode = mode;
      }
    }
    /* If EDT not yet in route table, the mode will be delivered by
       the add_dependence OOO replay, which stores mode and calls
       arts_add_dependence → arts_set_dep_mode again when the EDT
       exists. */
  } else {
    /* Remote EDT — send a lightweight mode-set message.
       We reuse the signal packet with NULL_GUID data; the receiver
       will call arts_set_dep_mode locally. */
    arts_remote_set_dep_mode(edt_guid, slot, mode);
  }
}

/*
 * internal_signal_edt — Satisfy one dependency slot on an EDT.
 *
 * Four dispatch paths:
 *   1. CDAG invalidation (current EDT has pending invalidations) →
 *      route through OOO to preserve ordering.
 *   2. Local EDT found in route table → write the dep slot and
 *      atomically decrement depc_needed.  If this was the last
 *      dependency (depc_needed hits 0), call arts_handle_ready_edt.
 *   3. Local EDT NOT found (still RESERVED or not yet created) →
 *      enqueue in the OOO list; will be replayed when the EDT
 *      transitions to AVAILABLE via arts_route_table_fire_oo.
 *   4. Remote EDT → forward the signal over the network.
 */
void internal_signal_edt(arts_guid_t edt_packet, uint32_t slot,
                         arts_guid_t data_guid, arts_db_access_mode_t mode,
                         void *ptr, unsigned int size) {
  TIME_EDT_SIGNAL_START();
  INCREMENT_NUM_EDT_SIGNAL_BY(1);

  if (current_edt && current_edt->invalidate_count > 0) {
    /* CDAG path: defer signal to maintain write-ordering invariants. */
    ARTS_DEBUG("Signal EDT[Guid:%lu] Slot:%u deferred (CDAG invalidation)",
               edt_packet, slot);
    if (mode == DB_MODE_PTR) {
      arts_out_of_order_signal_edt_with_ptr(edt_packet, data_guid, ptr, size,
                                            slot);
    } else {
      arts_out_of_order_signal_edt(current_edt->current_edt, edt_packet,
                                   data_guid, slot, mode, true);
    }
  } else {
    unsigned int rank = arts_guid_get_rank(edt_packet);
    if (rank == arts_global_rank_id) {
      /* Local signal path. */
      struct arts_edt_s *edt =
          (struct arts_edt_s *)arts_route_table_lookup_item(edt_packet);
      if (edt) {
        /* EDT exists in route table — write dep slot. */
        arts_edt_dep_t *edt_dep = (arts_edt_dep_t *)arts_get_depv(edt);
        if (slot < edt->depc) {
          edt_dep[slot].guid = data_guid;
          if (mode == DB_MODE_PTR && size > 0) {
            void *copy = arts_malloc(size);
            memcpy(copy, ptr, size);
            edt_dep[slot].ptr = copy;
          } else {
            edt_dep[slot].ptr = ptr;
          }
          if (mode != DB_MODE_NULL) {
            edt_dep[slot].mode = mode;
          }
        }
        unsigned int res = arts_atomic_sub(&edt->depc_needed, 1U);
        ARTS_INFO("Signal EDT[Guid:%lu, Slot:%u] DB[Guid:%lu] "
                  "depc_needed=%u→%u",
                  edt->current_edt, slot, data_guid, res + 1, res);
        if (res == 0) {
          ARTS_INFO("EDT[Guid:%lu] all deps satisfied — firing",
                    edt->current_edt);
          arts_handle_ready_edt(edt);
        }
      } else {
        /* EDT not yet in route table — queue as OOO. */
        ARTS_DEBUG("Signal EDT[Guid:%lu, Slot:%u] OOO (not in route table yet)",
                   edt_packet, slot);
        if (mode == DB_MODE_PTR) {
          arts_out_of_order_signal_edt_with_ptr(edt_packet, data_guid, ptr,
                                                size, slot);
        } else {
          arts_out_of_order_signal_edt(edt_packet, edt_packet, data_guid, slot,
                                       mode, false);
        }
      }
    } else {
      /* Remote signal — forward over the network. */
      ARTS_DEBUG("Signal EDT[Guid:%lu, Slot:%u] remote to rank %u", edt_packet,
                 slot, rank);
      if (mode == DB_MODE_PTR) {
        arts_remote_signal_edt_with_ptr(edt_packet, data_guid, ptr, size, slot);
      } else {
        arts_remote_signal_edt(edt_packet, data_guid, slot, mode);
      }
    }
  }
  TIME_EDT_SIGNAL_STOP();
}

void arts_signal_edt(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid,
                     arts_db_access_mode_t mode) {
  ARTS_DEBUG("arts_signal_edt [EDT:%lu, Slot:%u, DB:%lu, Mode:%u]", edt_guid,
             slot, data_guid, mode);
  void *db_ptr = NULL;
#ifdef ARTS_USE_CXL
  if (arts_guid_is_cxl(data_guid)) {
    db_ptr = (void *)((struct arts_db_s *)arts_cxl_get_ptr(data_guid) + 1);
  }
#endif
  internal_signal_edt(edt_guid, slot, data_guid, mode, db_ptr, 0);
}

// Internal function to signal EDT with explicit access mode
void internal_signal_edt_with_mode(arts_guid_t edt_packet, uint32_t slot,
                                   arts_guid_t data_guid,
                                   arts_db_access_mode_t mode) {
  TIME_EDT_SIGNAL_START();
  // This is old CDAG code...
  if (current_edt && current_edt->invalidate_count > 0) {
    if (mode == DB_MODE_PTR) {
      arts_out_of_order_signal_edt_with_ptr(edt_packet, data_guid, NULL, 0,
                                            slot);
    } else {
      arts_out_of_order_signal_edt(current_edt->current_edt, edt_packet,
                                   data_guid, slot, mode, true);
    }
  } else {
    unsigned int rank = arts_guid_get_rank(edt_packet);
    if (rank == arts_global_rank_id) {
      struct arts_edt_s *edt =
          (struct arts_edt_s *)arts_route_table_lookup_item(edt_packet);
      if (edt) {
        arts_edt_dep_t *edt_dep = (arts_edt_dep_t *)arts_get_depv(edt);
        if (slot < edt->depc) {
          edt_dep[slot].guid = data_guid;
          edt_dep[slot].ptr = NULL;
          if (mode != DB_MODE_NULL) {
            edt_dep[slot].mode = mode;
          }
        }
        unsigned int res = arts_atomic_sub(&edt->depc_needed, 1U);
        ARTS_INFO("Signal DB[Guid:%lu] to EDT[Guid:%lu, Slot:%u, "
                  "DepCount:%d, Mode:%s]",
                  data_guid, edt->current_edt, slot, res,
                  GET_DB_MODE_NAME(mode));
        if (res == 0) {
          arts_handle_ready_edt(edt);
        }
      } else {
        if (mode == DB_MODE_PTR) {
          arts_out_of_order_signal_edt_with_ptr(edt_packet, data_guid, NULL, 0,
                                                slot);
        } else {
          arts_out_of_order_signal_edt(edt_packet, edt_packet, data_guid, slot,
                                       mode, false);
        }
      }
    } else {
      if (mode == DB_MODE_PTR) {
        arts_remote_signal_edt_with_ptr(edt_packet, data_guid, NULL, 0, slot);
      } else {
        arts_remote_signal_edt(edt_packet, data_guid, slot, mode);
      }
    }
  }
  TIME_EDT_SIGNAL_STOP();
}

void arts_signal_edt_value(arts_guid_t edt_guid, uint32_t slot,
                           uint64_t value) {
  ARTS_DEBUG("arts_signal_edt_value [EDT:%lu, Slot:%u, Value:%lu]", edt_guid,
             slot, value);
  internal_signal_edt(edt_guid, slot, (arts_guid_t)value, DB_MODE_VALUE, NULL,
                      0);
}

void arts_signal_edt_ptr(arts_guid_t edt_guid, uint32_t slot, void *ptr,
                         unsigned int size) {
  internal_signal_edt(edt_guid, slot, NULL_GUID, DB_MODE_PTR, ptr, size);
}

void arts_signal_edt_ptr_with_guid(arts_guid_t edt_guid, uint32_t slot,
                                   arts_guid_t db_guid, void *ptr,
                                   unsigned int size) {
  internal_signal_edt(edt_guid, slot, db_guid, DB_MODE_PTR, ptr, size);
}

void arts_signal_edt_null(arts_guid_t edt_guid, uint32_t slot) {
  internal_signal_edt(edt_guid, slot, NULL_GUID, DB_MODE_NULL, NULL, 0);
}

arts_guid_t arts_allocate_local_buffer(void **buffer, unsigned int size,
                                       unsigned int uses,
                                       arts_guid_t epoch_guid) {
  if (epoch_guid) {
    increment_active_epoch(epoch_guid);
  }
  arts_shutdown_epoch_inc_active();

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

  arts_guid_t guid =
      arts_guid_create_for_rank(arts_global_rank_id, ARTS_BUFFER);
  arts_route_table_add_item(stub, guid, arts_global_rank_id, false);
  return guid;
}

void *arts_set_buffer(arts_guid_t buffer_guid, void *buffer,
                      unsigned int size) {
  void *ret = NULL;
  unsigned int rank = arts_guid_get_rank(buffer_guid);
  if (rank == arts_global_rank_id) {
    arts_buffer_t *stub =
        (arts_buffer_t *)arts_route_table_lookup_item(buffer_guid);
    if (stub) {
      arts_guid_t epoch_guid = stub->epoch_guid;
      if (epoch_guid) {
        increment_queue_epoch(epoch_guid);
      }
      arts_shutdown_epoch_inc_queue();

      if (size > stub->size) {
        if (stub->size) {
          ARTS_INFO("Truncating buffer data buffer size: %u stub size: %u",
                    size, stub->size);
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
      arts_shutdown_epoch_inc_finished();
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
    arts_buffer_t *stub =
        (arts_buffer_t *)arts_route_table_lookup_item(buffer_guid);
    if (stub == NULL) {
      return NULL;
    }
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
    arts_buffer_t *stub =
        (arts_buffer_t *)arts_route_table_lookup_item(buffer_guid);
    if (stub == NULL) {
      return NULL;
    }
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
void check_out_edts(uint64_t threshold) {
  static uint64_t count = 0;
  if (arts_atomic_fetch_add_u64(&count, 1) + 1 == threshold) {
    arts_atomic_fetch_sub_u64(&count, threshold);
  }
}

void arts_lc_sync(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid) {
  arts_type_t type = arts_guid_get_type(data_guid);
  (void)type;
  internal_signal_edt(edt_guid, slot, data_guid, DB_MODE_LC_SYNC, NULL, 0);
}

void arts_gpu_signal_edt_memset(arts_guid_t edt_guid, uint32_t slot,
                                arts_guid_t data_guid) {
  arts_db_access_mode_t mode = DB_MODE_MEMSET;
  struct arts_db_s *db =
      (struct arts_db_s *)arts_route_table_lookup_db(data_guid, NULL, false);
  if (db && db->db_type == ARTS_DB_LC) {
    mode = DB_MODE_LC_NO_COPY;
  }
  if (db) {
    arts_route_table_return_db(data_guid, false);
  }
  internal_signal_edt(edt_guid, slot, data_guid, mode, NULL, 0);
}
