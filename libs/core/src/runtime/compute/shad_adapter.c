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

#include "arts/runtime/compute/shad_adapter.h"

#include <string.h>

#include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/introspection/counter.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/runtime/compute/edt_functions.h"
#include "arts/runtime/sync/termination_detection.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/system/tmt_lite.h"
#include "arts/utils/atomics.h"
#include "arts/utils/queue.h"

arts_guid_t arts_edt_create_shad(arts_edt_t func_ptr, unsigned int route,
                             uint32_t paramc, const uint64_t *paramv) {
  unsigned int edt_space = sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t));
  arts_guid_t guid = NULL_GUID;
  arts_edt_create_internal(NULL, ARTS_EDT, &guid, route, arts_thread_info.cluster_id,
                        edt_space, NULL_GUID, func_ptr, paramc, paramv, 0, false,
                        NULL_GUID, false, 0);
  return guid;
}

arts_guid_t arts_active_message_shad(arts_edt_t func_ptr, unsigned int route,
                                 uint32_t paramc, const uint64_t *paramv, void *data,
                                 unsigned int size, arts_guid_t epoch_guid) {
  unsigned int rank = route; // route / num_numa_domains;
  unsigned int cluster = 0;  // route % num_numa_domains;
  arts_guid_t guid = NULL_GUID;
  bool use_epoch = (epoch_guid != NULL_GUID);

  if (size) {
    unsigned int dep_space = sizeof(arts_edt_dep_t);
    unsigned int edt_space =
        sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t)) + dep_space;
    arts_edt_create_internal(NULL, ARTS_EDT, &guid, rank, cluster, edt_space,
                          NULL_GUID, func_ptr, paramc, paramv, 1, use_epoch,
                          epoch_guid, true, 0);

    void *ptr = arts_malloc(size);
    memcpy(ptr, data, size);
    arts_signal_edt_ptr(guid, 0, ptr, size);
  } else {
    unsigned int edt_space = sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t));
    arts_edt_create_internal(NULL, ARTS_EDT, &guid, rank, cluster, edt_space,
                          NULL_GUID, func_ptr, paramc, paramv, 0, use_epoch,
                          epoch_guid, false, 0);
  }
  return guid;
}

void arts_synchronous_active_message_shad(arts_edt_t func_ptr, unsigned int route,
                                      uint32_t paramc, const uint64_t *paramv,
                                      void *data, unsigned int size) {
  unsigned int rank = route; // route / num_numa_domains;
  unsigned int cluster = 0;  // route % num_numa_domains;
  unsigned int wait_flag = 1;
  void *wait_ptr = &wait_flag;
  arts_guid_t wait_guid = arts_allocate_local_buffer(
      (&wait_ptr), sizeof(unsigned int), 1, NULL_GUID);

  arts_guid_t guid = NULL_GUID;
  if (size) {
    unsigned int dep_space = sizeof(arts_edt_dep_t);
    unsigned int edt_space =
        sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t)) + dep_space;
    arts_edt_create_internal(NULL, ARTS_EDT, &guid, rank, cluster, edt_space,
                          wait_guid, func_ptr, paramc, paramv, 1, false,
                          NULL_GUID, true, 0);

    void *ptr = arts_malloc(size);
    memcpy(ptr, data, size);
    arts_signal_edt_ptr(guid, 0, ptr, size);
  } else {
    unsigned int edt_space = sizeof(struct arts_edt_s) + (paramc * sizeof(uint64_t));
    arts_edt_create_internal(NULL, ARTS_EDT, &guid, rank, cluster, edt_space,
                          wait_guid, func_ptr, paramc, paramv, 0, false,
                          NULL_GUID, false, 0);
  }

  while (wait_flag) {
    arts_yield();
  }
}

void arts_inc_lock_shad() { arts_thread_info.shad_lock++; }

void arts_dec_lock_shad() { arts_thread_info.shad_lock--; }

void arts_check_lock_shad() {
  if (arts_thread_info.shad_lock) {
    ARTS_INFO("ARTS: Cannot perform synchronous call under lock Worker: %u "
              "ShadLock: %u",
              arts_thread_info.group_id, arts_thread_info.shad_lock);
    arts_debug_generate_seg_fault();
  }
}

void arts_start_intro_shad(unsigned int start) {
  // arts_counter_capture_start(start);
}

void arts_stop_intro_shad() { arts_counter_capture_stop(); }

unsigned int arts_get_shad_loop_stride() { return arts_node_info.shad_loop_stride; }

arts_guid_t arts_allocate_local_buffer_shad(void **buffer, uint32_t *size_to_write,
                                       arts_guid_t epoch_guid) {
  if (epoch_guid) {
    increment_active_epoch(epoch_guid);
  } else {
    ARTS_INFO("No EPOCH!!!");
  }
  global_shutdown_guid_inc_active();

  arts_buffer_t *stub = (arts_buffer_t *)arts_malloc(sizeof(arts_buffer_t));
  stub->buffer = *buffer;
  stub->size_to_write = size_to_write;
  stub->size = 0;
  stub->uses = 1;
  stub->epoch_guid = epoch_guid;

  arts_guid_t guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_BUFFER);
  arts_route_table_add_item(stub, guid, arts_global_rank_id, false);
  return guid;
}

arts_shad_lock_t *arts_shad_create_lock() {
  arts_shad_lock_t *lock =
      (arts_shad_lock_t *)arts_calloc(1, sizeof(arts_shad_lock_t));
  lock->queue = arts_new_queue();
  return lock;
}

void arts_shad_lock(arts_shad_lock_t *lock) {
  unsigned int res = arts_atomic_fetch_add(&lock->size, 1);
  if (res) {
    enqueue(arts_get_context_ticket(), lock->queue);
    arts_context_switch(1);
  }
}

void arts_shad_unlock(arts_shad_lock_t *lock) {
  unsigned int res = arts_atomic_sub(&lock->size, 1);
  if (res) {
    while (1) {
      arts_ticket_t ticket = dequeue(lock->queue);
      if (ticket) {
        arts_signal_context(ticket);
        return;
      }
    }
  }
}

#define ALIASOWNERMAP 0xF000000000000000
#define ALIASCOUNTMAP 0x0FFFFFFFFFFFFFFF
#define ALIASGETOWNER(x) (((x) & ALIASOWNERMAP) >> 60)
#define ALIASGETCOUNT(x) ((x) & ALIASCOUNTMAP)
#define ALIASEMPTY (((((uint64_t)arts_thread_info.group_id) + 1) << 60) + 1)

bool arts_shad_alias_try_lock(volatile uint64_t *lock) {
  uint64_t dirty_read = *lock;
  uint64_t owner = ALIASGETOWNER(dirty_read);
  while (!owner || owner == arts_thread_info.group_id + 1) {
    uint64_t new_value = (!owner) ? ALIASEMPTY : dirty_read + 1;
    uint64_t res = arts_atomic_cswap_u64(lock, dirty_read, new_value);
    if (res == dirty_read) {
      return true;
}
    dirty_read = res;
    owner = ALIASGETOWNER(dirty_read);
  }
  return false;
}

void arts_shad_alias_unlock(volatile uint64_t *lock) {
  uint64_t dirty_read = *lock;
  while (1) {
    uint64_t new_value = (ALIASGETCOUNT(dirty_read) == 1) ? 0 : dirty_read - 1;
    uint64_t res = arts_atomic_cswap_u64(lock, dirty_read, new_value);
    if (res == dirty_read) {
      // ARTS_INFO("RES: %lu", res);
      return;
    }
    dirty_read = res;
  }
}

#define LITEOWNERMAP 0x8000000000000000ULL
#define LITECOUNTMAP 0x7FFFFFFFFFFFFFFFULL
#define LITEGETOWNER(x) ((x) & LITEOWNERMAP)
#define LITEGETCOUNT(x) ((x) & ALIASCOUNTMAP)

static inline bool arts_tmt_lite_try_lock(volatile uint64_t *lock) {
  uint64_t local = *lock;
  if (!LITEGETOWNER(local)) {
    if (local == arts_atomic_cswap_u64(lock, local, local | LITEOWNERMAP)) {
      return true;
}
  }
  return false;
}

static inline void arts_tmt_lite_unlock(volatile uint64_t *lock) {
  arts_atomic_fetch_and_u64(lock, LITECOUNTMAP);
}

// Returns if you should dec on unlock
bool arts_shad_tmt_lock2(volatile uint64_t *lock) {
  // We should have the execution lock on entry
  if (!arts_tmt_lite_try_lock(lock)) // Try lock but fail
  {
    // arts_atomic_add_u64(lock, 1); //Inc the counter that we have created thread
    arts_create_lite_contexts(lock); // Still have execution lock at end
    uint64_t counter = 0;
    while (1) {
      if (arts_tmt_lite_try_lock(lock)) {
        break;
}
      arts_yield_lite_context();  // Give up execution lock
      arts_resume_lite_context(); // Has execution lock at end
      if (counter > 1000000) {
        arts_atomic_add_u64(lock, 1); // Inc the counter that we have created
                                   // thread
        arts_create_lite_contexts(lock);
        ARTS_INFO("STUPID CREATE!!!");
        counter = 0;
      }
      counter++;
    }
    return true;
  }
  return false;
}

void arts_shad_tmt_unlock(volatile uint64_t *lock) {
  // ARTS_INFO("Unlock: %p %u:%u", lock, arts_get_current_worker(),
  // arts_tmt_lite_get_alias());
  arts_tmt_lite_unlock(lock);
}

void arts_shad_tmt_lock(volatile uint64_t *lock) {
  while (arts_thread_info.alive) {
    if (arts_tmt_lite_try_lock(lock)) { // Try lock but fail
      break;
    }
    struct arts_edt_s *edt = arts_find_edt();
    if (edt) {
      arts_atomic_add_u64(lock, 1); // Inc the counter that we have created
                                 // thread
      arts_create_lite_contexts2(lock, edt); // Still have execution lock at end
    } else {
      CHECK_OUTSTANDING_EDTS(10000000);
    }
    arts_yield_lite_context(); // Give up execution lock
    arts_resume_lite_context();
  }
  // ARTS_INFO("Lock: %p %u:%u", lock, arts_get_current_worker(),
  // arts_tmt_lite_get_alias());
}
