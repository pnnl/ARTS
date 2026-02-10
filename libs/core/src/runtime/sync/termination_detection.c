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
#include "arts/runtime/sync/termination_detection.h"

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/rt.h"
#include "arts/runtime/compute/edt_functions.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/system/arts_print.h"
#include "arts/utils/atomics.h"

#define EPOCH_MASK 0x7FFFFFFFFFFFFFFF
#define EPOCH_BIT 0x8000000000000000

#define DEFAULT_EPOCH_POOL_SIZE 4096
__thread arts_epoch_pool_t *epoch_thread_pool;

void global_shutdown_guid_inc_active() {
  if (arts_node_info.shutdown_epoch) {
    increment_active_epoch(arts_node_info.shutdown_epoch);
}
}

void global_shutdown_guid_inc_queue() {
  if (arts_node_info.shutdown_epoch) {
    increment_queue_epoch(arts_node_info.shutdown_epoch);
}
}

void global_shutdown_guid_inc_finished() {
  if (arts_node_info.shutdown_epoch) {
    increment_finished_epoch(arts_node_info.shutdown_epoch);
}
}

void global_guid_shutdown(arts_guid_t guid) {
  if (arts_node_info.shutdown_epoch == guid) {
    arts_shutdown();
  }
}

bool decrement_queue_epoch(arts_epoch_t *epoch) {
  uint64_t local;
  while (1) {
    local = epoch->queued;
    if (local == 1) {
      if (1 == arts_atomic_cswap_u64(&epoch->queued, 1, EPOCH_BIT)) {
        return true;
}
    } else {
      if (local == arts_atomic_cswap_u64(&epoch->queued, local, local - 1)) {
        return false;
}
    }
  }
}

void increment_queue_epoch(arts_guid_t epoch_guid) {
  if (epoch_guid != NULL_GUID) {
    arts_epoch_t *epoch = (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
    if (epoch) {
      arts_atomic_add_u64(&epoch->queued, 1);
    } else {
      arts_out_of_order_inc_queue_epoch(epoch_guid);
    }
  }
}

void increment_active_epoch(arts_guid_t epoch_guid) {
  arts_epoch_t *epoch = (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
  if (epoch) {
    arts_atomic_add(&epoch->activeCount, 1);
  } else {
    arts_out_of_order_inc_active_epoch(epoch_guid);
  }
}

void increment_finished_epoch(arts_guid_t epoch_guid) {
  if (epoch_guid != NULL_GUID) {
    arts_epoch_t *epoch = (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
    if (epoch) {
      arts_atomic_add(&epoch->finishedCount, 1);
      if (arts_global_rank_count == 1) {
        if (!check_epoch(epoch, epoch->activeCount, epoch->finishedCount)) {
          if (epoch->phase == PHASE_3) {
            delete_epoch(epoch_guid, epoch);
}
        }
      } else {
        unsigned int rank = arts_guid_get_rank(epoch_guid);
        if (rank == arts_global_rank_id) {
          if (!arts_atomic_sub_u64(&epoch->queued, 1)) {
            if (!arts_atomic_cswap_u64(&epoch->outstanding, 0,
                                    arts_global_rank_count)) {
              broadcast_epoch_request(epoch_guid);
            }
          }
        } else {
          if (decrement_queue_epoch(epoch)) {
            arts_remote_epoch_send(rank, epoch_guid, epoch->activeCount,
                                epoch->finishedCount);
          }
        }
      }
    } else {
      arts_out_of_order_inc_finished_epoch(epoch_guid);
    }
  }
}

void send_epoch(arts_guid_t epoch_guid, unsigned int source, unsigned int dest) {
  arts_epoch_t *epoch = (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
  if (epoch) {
    ARTS_DEBUG("Sending epoch [Guid:%lu] to rank %u", epoch_guid, dest);
    arts_atomic_fetch_and_u64(&epoch->queued, EPOCH_MASK);
    if (!arts_atomic_cswap_u64(&epoch->queued, 0, EPOCH_BIT)) {
      arts_remote_epoch_send(dest, epoch_guid, epoch->activeCount,
                          epoch->finishedCount);
    }
  } else {
    arts_out_of_order_send_epoch(epoch_guid, source, dest);
}
}

arts_epoch_t *create_epoch(arts_guid_t *guid, arts_guid_t edt_guid,
                         unsigned int slot) {
  if (*guid == NULL_GUID) {
    *guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_EDT);
}

  arts_epoch_t *epoch = (arts_epoch_t *)arts_calloc(1, sizeof(arts_epoch_t));
  epoch->phase = PHASE_1;
  epoch->terminationExitGuid = edt_guid;
  epoch->terminationExitSlot = slot;
  epoch->guid = *guid;
  epoch->pool_guid = NULL_GUID;
  epoch->queued = (arts_is_guid_local(*guid)) ? 0 : EPOCH_BIT;
  arts_route_table_add_item_race(epoch, *guid, arts_global_rank_id, false);
  arts_route_table_fire_oo(*guid, arts_out_of_order_handler);
  return epoch;
}

bool create_shutdown_epoch() {
  if (arts_node_info.shutdown_epoch) {
    arts_node_info.shutdown_epoch = arts_guid_create_for_rank(0, ARTS_EDT);
    arts_epoch_t *epoch = create_epoch(&arts_node_info.shutdown_epoch, NULL_GUID, 0);
    arts_atomic_add(&epoch->activeCount, arts_get_total_workers());
    arts_atomic_add_u64(&epoch->queued, arts_get_total_workers());
    return true;
  }
  return false;
}

void arts_add_edt_to_epoch(arts_guid_t edt_guid, arts_guid_t epoch_guid) {
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_route_table_lookup_item(edt_guid);
  if (edt) {
    edt->epoch_guid = epoch_guid;
    increment_active_epoch(epoch_guid);
    return;
  }
  }

void broadcast_epoch_request(arts_guid_t epoch_guid) {
  unsigned int origin_rank = arts_guid_get_rank(epoch_guid);
  for (unsigned int i = 0; i < arts_global_rank_count; i++) {
    if (i != origin_rank) {
      arts_remote_epoch_req(i, epoch_guid);
    }
  }
}

arts_guid_t arts_initialize_and_start_epoch(arts_guid_t finish_edt_guid,
                                       unsigned int slot) {
  arts_epoch_t *epoch = get_pool_epoch(finish_edt_guid, slot);

  arts_set_current_epoch_guid(epoch->guid);
  arts_atomic_add(&epoch->activeCount, 1);
  arts_atomic_add_u64(&epoch->queued, 1);
  ARTS_INFO("Creating and Initializing Epoch [Guid:%lu]", epoch->guid);
  return epoch->guid;
}

arts_guid_t arts_initialize_epoch(unsigned int rank, arts_guid_t finish_edt_guid,
                               unsigned int slot) {
  arts_guid_t guid = NULL_GUID;
  // I think the idea is this is that during parallel start
  // (arts_node_info.ready_to_execute > 0) This means that the epoch will be created
  // on all nodes assuming that each node goes through the initializeEpoch code
  // path. Pool assume the current host...
  if (!arts_node_info.ready_to_execute || rank != arts_global_rank_id) {
    guid = arts_guid_create_for_rank(rank, ARTS_EDT);
    create_epoch(&guid, finish_edt_guid, slot);
    if (!arts_node_info.ready_to_execute) {
      for (unsigned int i = 0; i < arts_global_rank_count; i++) {
        if (i != arts_global_rank_id) {
          arts_remote_epoch_init_send(i, guid, finish_edt_guid, slot);
}
      }
    }
  } else // Lets get it from the pool...
  {
    arts_epoch_t *epoch = get_pool_epoch(finish_edt_guid, slot);
    guid = epoch->guid;
  }
  return guid;
}

void arts_start_epoch(arts_guid_t epoch_guid) {
  arts_epoch_t *epoch = (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
  if (epoch) {
    arts_set_current_epoch_guid(epoch->guid);
    arts_atomic_add(&epoch->activeCount, 1);
    arts_atomic_add_u64(&epoch->queued, 1);
  } else {
    ARTS_ERROR("Epoch [Guid:%lu] doesn't exist in the Route table", epoch_guid);
  }
}

bool check_epoch(arts_epoch_t *epoch, unsigned int total_active,
                unsigned int total_finish) {
  unsigned int diff = total_active - total_finish;
  ARTS_INFO("Checking Epoch [Guid:%lu, TotalActive:%u, TotalFinish:%u, "
            "Diff:%u, Phase:%u, LastActive:%u, LastFinished:%u]",
            epoch->guid, total_active, total_finish, diff, epoch->phase,
            epoch->lastActiveCount, epoch->lastFinishedCount);
  // We have a zero
  if (total_finish && !diff) {
    // Lets check the phase and if we have the same counts as before
    if (epoch->phase == PHASE_2 && epoch->lastActiveCount == total_active &&
        epoch->lastFinishedCount == total_finish) {
      ARTS_DEBUG(
          "check_epoch: Advancing to PHASE_3 - epoch termination complete!");
      epoch->phase = PHASE_3;
      if (epoch->wait_ptr) {
        *epoch->wait_ptr = 0;
}
      if (epoch->ticket) {
        arts_signal_context(epoch->ticket);
      }
      if (epoch->terminationExitGuid) {
        arts_signal_edt_value(epoch->terminationExitGuid,
                           epoch->terminationExitSlot, total_finish);
      } else {
        global_guid_shutdown(epoch->guid);
      }
      return false;
    }
    // We didn't match the last one so lets try again
    epoch->lastActiveCount = total_active;
    epoch->lastFinishedCount = total_finish;
    epoch->phase = PHASE_2;
    if (arts_global_rank_count == 1) {
      epoch->phase = PHASE_3;
      if (epoch->wait_ptr) {
        *epoch->wait_ptr = 0;
}
      if (epoch->ticket) {
        arts_signal_context(epoch->ticket);
}
      if (epoch->terminationExitGuid) {
        arts_signal_edt_value(epoch->terminationExitGuid,
                           epoch->terminationExitSlot, total_finish);
      } else {
        global_guid_shutdown(epoch->guid);
      }
      return false;
    }
    return true;
  }
  epoch->phase = PHASE_1;
  return (epoch->queued == 0);
}

void reduce_epoch(arts_guid_t epoch_guid, unsigned int active,
                 unsigned int finish) {
  arts_epoch_t *epoch = (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
  if (epoch) {
    unsigned int total_active = arts_atomic_add(&epoch->globalActiveCount, active);
    unsigned int total_finish =
        arts_atomic_add(&epoch->globalFinishedCount, finish);
    uint64_t outstanding_before = epoch->outstanding;
    if (arts_atomic_sub_u64(&epoch->outstanding, 1) == 1) {
      total_active += epoch->activeCount;
      total_finish += epoch->finishedCount;

      ARTS_DEBUG("reduce_epoch [Guid:%lu]: total_active=%u, total_finish=%u, "
                 "phase=%u, outstanding_before=%lu, queued=%lu",
                 epoch_guid, total_active, total_finish, epoch->phase,
                 outstanding_before, epoch->queued);

      // Reset for the next round
      epoch->globalActiveCount = 0;
      epoch->globalFinishedCount = 0;

      if (check_epoch(epoch, total_active, total_finish)) {
        ARTS_DEBUG("  check_epoch returned TRUE - broadcasting new request");
        arts_atomic_add_u64(&epoch->outstanding, arts_global_rank_count - 1);
        broadcast_epoch_request(epoch_guid);
        // A better idea will be to know when to kick off a new round
        // the checkinCount == 0 indicates there is a new round can be kicked
        // off
        //                arts_atomic_sub(&epoch->checkinCount, 1);
      } else {
        ARTS_DEBUG("  check_epoch returned FALSE - epoch completed or advancing "
                   "to phase %u",
                   epoch->phase);
        arts_atomic_sub_u64(&epoch->outstanding, 1);
      }

      if (epoch->phase == PHASE_3) {
        ARTS_DEBUG("  Deleting epoch [Guid:%lu] - termination complete",
                   epoch_guid);
        delete_epoch(epoch_guid, epoch);
      }
    } else {
      ARTS_DEBUG("reduce_epoch [Guid:%lu]: outstanding=%lu (still waiting for "
                 "more responses)",
                 epoch_guid, outstanding_before - 1);
    }
  }
}

arts_epoch_pool_t *create_epoch_pool(arts_guid_t *epoch_pool_guid,
                                 unsigned int pool_size, arts_guid_t *start_guid) {
  if (*epoch_pool_guid == NULL_GUID) {
    *epoch_pool_guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_EDT);
}

  bool new_range = (*start_guid == NULL_GUID);
  arts_guid_range_t temp;
  arts_guid_range_t *range;
  if (new_range) {
    range = arts_new_guid_range_node(ARTS_EDT, pool_size, arts_global_rank_id);
    *start_guid = arts_get_guid(range, 0);
  } else {
    temp.size = pool_size;
    temp.index = 0;
    temp.start_guid = *start_guid;
    range = &temp;
  }

  arts_epoch_pool_t *epoch_pool = (arts_epoch_pool_t *)arts_calloc(
      1, sizeof(arts_epoch_pool_t) + (sizeof(arts_epoch_t) * pool_size));
  epoch_pool->index = 0;
  epoch_pool->outstanding = pool_size;
  epoch_pool->size = pool_size;

  arts_route_table_add_item(epoch_pool, *epoch_pool_guid, arts_global_rank_id, false);
  for (unsigned int i = 0; i < pool_size; i++) {
    epoch_pool->pool[i].phase = PHASE_1;
    epoch_pool->pool[i].pool_guid = *epoch_pool_guid;
    epoch_pool->pool[i].guid = arts_get_guid(range, i);
    epoch_pool->pool[i].queued =
        (arts_is_guid_local(*epoch_pool_guid)) ? 0 : EPOCH_BIT;
    if (!arts_is_guid_local(*epoch_pool_guid)) {
      arts_route_table_add_item_race(&epoch_pool->pool[i], epoch_pool->pool[i].guid,
                                arts_global_rank_id, false);
      arts_route_table_fire_oo(epoch_pool->pool[i].guid, arts_out_of_order_handler);
    }
  }

  if (new_range) {
    arts_free(range);
}

  return epoch_pool;
}

void delete_epoch(arts_guid_t epoch_guid, arts_epoch_t *epoch) {
  // Can't call delete unless we already hit two barriers thus it must exit
  if (!epoch) {
    epoch = (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
}

  if (epoch->pool_guid) {
    arts_epoch_pool_t *pool =
        (arts_epoch_pool_t *)arts_route_table_lookup_item(epoch->pool_guid);
    if (arts_is_guid_local(epoch->pool_guid)) {
      arts_route_table_remove_item(epoch_guid);
      if (!arts_atomic_sub(&pool->outstanding, 1)) {
        arts_route_table_remove_item(epoch->pool_guid);
        //                arts_free(pool);  //Free in the next get_pool_epoch
        for (unsigned int i = 0; i < arts_global_rank_count; i++) {
          if (i != arts_global_rank_id) {
            arts_remote_epoch_delete(i, epoch_guid);
}
        }
      }
    } else {
      for (unsigned int i = 0; i < pool->size; i++) {
        arts_route_table_remove_item(pool->pool[i].guid);
}
      arts_route_table_remove_item(epoch->pool_guid);
      arts_free(pool);
    }
  } else {
    arts_route_table_remove_item(epoch_guid);
    arts_free(epoch);

    if (arts_is_guid_local(epoch_guid)) {
      for (unsigned int i = 0; i < arts_global_rank_count; i++) {
        if (i != arts_global_rank_id) {
          arts_remote_epoch_delete(i, epoch_guid);
}
      }
    }
  }
}

void clean_epoch_pool() {
  arts_epoch_pool_t *trail_pool = NULL;
  arts_epoch_pool_t *pool = epoch_thread_pool;

  while (pool) {
    if (pool->index == epoch_thread_pool->size && !pool->outstanding) {
      arts_epoch_pool_t *to_free = pool;

      pool = pool->next;

      if (trail_pool) {
        trail_pool->next = pool;
      } else {
        epoch_thread_pool = pool;
}

      arts_free(to_free);
    } else {
      trail_pool = pool;
      pool = pool->next;
    }
  }
}

arts_epoch_t *get_pool_epoch(arts_guid_t edt_guid, unsigned int slot) {
  //    clean_epoch_pool();
  arts_epoch_pool_t *trail_pool = NULL;
  arts_epoch_pool_t *pool = epoch_thread_pool;
  arts_epoch_t *epoch = NULL;
  while (!epoch) {
    if (!pool) {
      arts_guid_t pool_guid = NULL_GUID;
      arts_guid_t start_guid = NULL_GUID;
      pool = create_epoch_pool(&pool_guid, DEFAULT_EPOCH_POOL_SIZE, &start_guid);

      if (trail_pool) {
        trail_pool->next = pool;
      } else {
        epoch_thread_pool = pool;
}

      for (unsigned int i = 0; i < arts_global_rank_count; i++) {
        if (i != arts_global_rank_id) {
          arts_remote_epoch_init_pool_send(i, DEFAULT_EPOCH_POOL_SIZE, start_guid,
                                      pool_guid);
}
      }
    }

    if (pool->index < pool->size) {
      epoch = &pool->pool[pool->index++];
    } else {
      trail_pool = pool;
      pool = pool->next;
    }
  }

  epoch->terminationExitGuid = edt_guid;
  epoch->terminationExitSlot = slot;
  arts_route_table_add_item_race(epoch, epoch->guid, arts_global_rank_id, false);
  arts_route_table_fire_oo(epoch->guid, arts_out_of_order_handler);
  return epoch;
}

void arts_yield() {
  EDT_RUNNING_TIME_STOP();
  INCREMENT_YIELD_BY(1);
  thread_local_t tl;
  arts_save_thread_local(&tl);
  arts_node_info.scheduler();
  arts_restore_thread_local(&tl);
  EDT_RUNNING_TIME_START();
}

bool arts_wait_on_handle(arts_guid_t epoch_guid) {
  EDT_RUNNING_TIME_STOP();
  arts_guid_t *guid = arts_check_epoch_is_root(epoch_guid);
  ARTS_DEBUG("Waiting on epoch [Guid:%lu]", epoch_guid);
  // For now lets leave this rule here
  if (guid) {
    arts_guid_t local = *guid;
    *guid = NULL_GUID; // Unset
    unsigned int flag = 1;
    arts_epoch_t *epoch = (arts_epoch_t *)arts_route_table_lookup_item(local);
    if (!epoch) {
      // Epoch may still be in reserved state in route table; spin briefly
      for (int retries = 0; !epoch && retries < 1000; retries++) {
        epoch = (arts_epoch_t *)arts_route_table_lookup_item(local);
      }
      if (!epoch) {
        ARTS_ERROR("arts_wait_on_handle: Epoch [Guid:%lu] not found in route table",
                   local);
        EDT_RUNNING_TIME_START();
        return false;
      }
    }
    epoch->ticket = arts_get_context_ticket();
    if (arts_node_info.tmt && epoch->ticket) {
      increment_finished_epoch(local);
      arts_context_switch(1);
      // Drain any work enqueued during the context switch so DB releases finish
      while (arts_node_info.scheduler()) {
        ;
}
      clean_epoch_pool();
      EDT_RUNNING_TIME_START();
      return true;
    }
    if (!epoch->ticket) {
      epoch->wait_ptr = &flag;
      increment_finished_epoch(local);
      //        global_shutdown_guid_inc_finished();

      INCREMENT_YIELD_BY(1);
      thread_local_t tl;
      arts_save_thread_local(&tl);
      while (flag) {
        arts_node_info.scheduler();
}
      // Continue running until the scheduler reports no more ready work
      while (arts_node_info.scheduler()) {
        ;
}
      arts_restore_thread_local(&tl);

      clean_epoch_pool();

      EDT_RUNNING_TIME_START();
      return true;
    }
  }
  EDT_RUNNING_TIME_START();
  return false;
}
