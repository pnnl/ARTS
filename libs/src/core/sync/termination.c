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
#include "arts/sync/termination.h"

#include "arts.h"
#include "arts/compute/edt.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/remote/handler.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

#define EPOCH_MASK 0x7FFFFFFFFFFFFFFF
#define EPOCH_BIT 0x8000000000000000

#define DEFAULT_EPOCH_POOL_SIZE 4096
ARTS_THREAD_LOCAL arts_epoch_pool_t *epoch_thread_pool;

/*
 * Shutdown-epoch helpers.
 *
 * The shutdown epoch tracks all EDTs globally.  When the epoch completes
 * (active == finished, no outstanding), arts_shutdown_epoch_fire is called,
 * which triggers arts_shutdown().
 *
 * Three counters are incremented at different EDT lifecycle stages:
 *   inc_active  — when an EDT is created (globally visible).
 *   inc_queue   — when an EDT becomes ready (all deps satisfied).
 *   inc_finished — when an EDT completes execution.
 */
void arts_shutdown_epoch_inc_active() {
  if (arts_node_info.auto_shutdown_guid) {
    ARTS_DEBUG("shutdown_epoch: inc_active [Epoch:%lu]",
               arts_node_info.auto_shutdown_guid);
    increment_active_epoch(arts_node_info.auto_shutdown_guid);
  }
}

void arts_shutdown_epoch_inc_queue() {
  if (arts_node_info.auto_shutdown_guid) {
    ARTS_DEBUG("shutdown_epoch: inc_queue [Epoch:%lu]",
               arts_node_info.auto_shutdown_guid);
    increment_queue_epoch(arts_node_info.auto_shutdown_guid);
  }
}

void arts_shutdown_epoch_inc_finished() {
  if (arts_node_info.auto_shutdown_guid) {
    ARTS_DEBUG("shutdown_epoch: inc_finished [Epoch:%lu]",
               arts_node_info.auto_shutdown_guid);
    increment_finished_epoch(arts_node_info.auto_shutdown_guid);
  }
}

void arts_shutdown_epoch_fire(arts_guid_t guid) {
  if (arts_node_info.auto_shutdown_guid == guid) {
    ARTS_INFO(
        "arts_shutdown_epoch_fire: Epoch[Guid:%lu] matched shutdown epoch — "
        "calling arts_shutdown()",
        guid);
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
    arts_epoch_t *epoch =
        (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
    if (epoch) {
      arts_atomic_add_u64(&epoch->queued, 1);
    } else {
      arts_out_of_order_inc_queue_epoch(epoch_guid);
    }
  }
}

void increment_active_epoch(arts_guid_t epoch_guid) {
  arts_epoch_t *epoch =
      (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
  if (epoch) {
    arts_atomic_add(&epoch->active_count, 1);
  } else {
    arts_out_of_order_inc_active_epoch(epoch_guid);
  }
}

/*
 * increment_finished_epoch — Called when an EDT finishes execution.
 *
 * Single-node fast path: the atomic increment of finished_count returns
 * the new value.  Because arts_atomic_add uses __sync_add_and_fetch
 * (full barrier), the subsequent load of active_count sees all prior
 * increments.  Exactly one thread can observe new_finished == cur_active
 * (each new_finished value is unique), so only that thread calls
 * check_epoch.  check_epoch's internal CAS on the phase field provides
 * defense-in-depth against concurrent fire attempts.
 *
 * Multi-node: owner rank collects responses; non-owner ranks decrement
 * their queued counter and, when it hits 1, send their active/finished
 * counts to the owner for global reduction.
 */
void increment_finished_epoch(arts_guid_t epoch_guid) {
  if (epoch_guid != NULL_GUID) {
    arts_epoch_t *epoch =
        (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
    if (epoch) {
      unsigned int new_finished = arts_atomic_add(&epoch->finished_count, 1);
      ARTS_DEBUG("increment_finished_epoch[Guid:%lu]: finished_count=%u, "
                 "active_count=%u, phase=%u",
                 epoch_guid, new_finished, epoch->active_count, epoch->phase);
      if (arts_global_rank_count == 1) {
        /*
         * Single-node fast path: new_finished is unique per thread
         * (from arts_atomic_add's __sync_add_and_fetch), so at most
         * one thread sees equality with cur_active.  check_epoch's
         * internal CAS provides defense-in-depth for the fire decision.
         */
        unsigned int cur_active = epoch->active_count;
        if (new_finished > 0 && new_finished == cur_active) {
          check_epoch(epoch, cur_active, new_finished);
          if (epoch->phase == (unsigned int)PHASE_3) {
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
            arts_remote_epoch_send(rank, epoch_guid, epoch->active_count,
                                   epoch->finished_count);
          }
        }
      }
    } else {
      arts_out_of_order_inc_finished_epoch(epoch_guid);
    }
  }
}

void send_epoch(arts_guid_t epoch_guid, unsigned int source,
                unsigned int dest) {
  arts_epoch_t *epoch =
      (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
  if (epoch) {
    ARTS_DEBUG("Sending epoch [Guid:%lu] to rank %u", epoch_guid, dest);
    arts_atomic_fetch_and_u64(&epoch->queued, EPOCH_MASK);
    if (!arts_atomic_cswap_u64(&epoch->queued, 0, EPOCH_BIT)) {
      arts_remote_epoch_send(dest, epoch_guid, epoch->active_count,
                             epoch->finished_count);
    }
  } else {
    arts_out_of_order_send_epoch(epoch_guid, source, dest);
  }
}

arts_epoch_t *create_epoch(arts_guid_t *guid, arts_guid_t edt_guid,
                           unsigned int slot) {
  INCREMENT_NUM_EPOCH_CREATE_BY(1);
  if (*guid == NULL_GUID) {
    *guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_EDT);
  }

  arts_epoch_t *epoch = (arts_epoch_t *)arts_calloc(1, sizeof(arts_epoch_t));
  epoch->phase = PHASE_1;
  epoch->termination_exit_guid = edt_guid;
  epoch->termination_exit_slot = slot;
  epoch->guid = *guid;
  epoch->pool_guid = NULL_GUID;
  epoch->queued = (arts_guid_is_local(*guid)) ? 0 : EPOCH_BIT;
  arts_route_table_add_item_race(epoch, *guid, arts_global_rank_id, false);
  arts_route_table_fire_oo(*guid, arts_out_of_order_handler);
  return epoch;
}

/*
 * arts_shutdown_epoch_create — Initialize the global termination epoch.
 *
 * Pre-seeds active_count and queued with the total number of workers,
 * since each worker thread will call increment_finished_epoch when it
 * finishes its initialization sequence.
 */
bool arts_shutdown_epoch_create() {
  if (arts_node_info.auto_shutdown_guid) {
    arts_node_info.auto_shutdown_guid = arts_guid_create_for_rank(0, ARTS_EDT);
    arts_epoch_t *epoch =
        create_epoch(&arts_node_info.auto_shutdown_guid, NULL_GUID, 0);
    unsigned int total_workers = arts_get_total_workers();
    arts_atomic_add(&epoch->active_count, total_workers);
    arts_atomic_add_u64(&epoch->queued, total_workers);
    ARTS_INFO(
        "arts_shutdown_epoch_create: Epoch[Guid:%lu] created with %u workers",
        arts_node_info.auto_shutdown_guid, total_workers);
    return true;
  }
  return false;
}

void arts_add_edt_to_epoch(arts_guid_t edt_guid, arts_guid_t epoch_guid) {
  struct arts_edt_s *edt =
      (struct arts_edt_s *)arts_route_table_lookup_item(edt_guid);
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
  arts_atomic_add(&epoch->active_count, 1);
  arts_atomic_add_u64(&epoch->queued, 1);
  ARTS_INFO("Creating and Initializing Epoch [Guid:%lu]", epoch->guid);
  return epoch->guid;
}

arts_guid_t arts_initialize_epoch(unsigned int rank,
                                  arts_guid_t finish_edt_guid,
                                  unsigned int slot) {
  arts_guid_t guid = NULL_GUID;
  // I think the idea is this is that during parallel start
  // (arts_node_info.ready_to_execute > 0) This means that the epoch will be
  // created on all nodes assuming that each node goes through the
  // initializeEpoch code path. Pool assume the current host...
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
  arts_epoch_t *epoch =
      (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
  if (epoch) {
    arts_set_current_epoch_guid(epoch->guid);
    arts_atomic_add(&epoch->active_count, 1);
    arts_atomic_add_u64(&epoch->queued, 1);
  } else {
    ARTS_WARN("Epoch [Guid:%lu] doesn't exist in the Route table", epoch_guid);
  }
}

bool check_epoch(arts_epoch_t *epoch, unsigned int total_active,
                 unsigned int total_finish) {
  unsigned int diff = total_active - total_finish;
  ARTS_INFO("Checking Epoch [Guid:%lu, TotalActive:%u, TotalFinish:%u, "
            "Diff:%u, Phase:%u, LastActive:%u, LastFinished:%u]",
            epoch->guid, total_active, total_finish, diff, epoch->phase,
            epoch->last_active_count, epoch->last_finished_count);

  if (total_finish && !diff) {
    if (arts_global_rank_count == 1) {
      /*
       * Single-node: skip the two-phase confirmation protocol (no network
       * round needed).  CAS ensures exactly one thread transitions to
       * PHASE_3 and fires the epoch, even under concurrent callers.
       */
      unsigned int old_phase = arts_atomic_cswap(
          &epoch->phase, (unsigned int)PHASE_1, (unsigned int)PHASE_3);
      if (old_phase == (unsigned int)PHASE_1) {
        ARTS_DEBUG(
            "check_epoch: CAS won PHASE_1->PHASE_3, firing epoch [Guid:%lu]",
            epoch->guid);
        if (epoch->termination_exit_guid) {
          arts_signal_edt_value(epoch->termination_exit_guid,
                                epoch->termination_exit_slot, total_finish);
        } else {
          arts_shutdown_epoch_fire(epoch->guid);
        }
      }
      return false;
    }

    /*
     * Multi-node: two-phase confirmation protocol.
     * Protected by the outstanding gate in reduce_epoch (single entrant),
     * but CAS on the PHASE_2 -> PHASE_3 transition provides defense-in-depth.
     */
    if (epoch->phase == (unsigned int)PHASE_2 &&
        epoch->last_active_count == total_active &&
        epoch->last_finished_count == total_finish) {
      unsigned int old_phase = arts_atomic_cswap(
          &epoch->phase, (unsigned int)PHASE_2, (unsigned int)PHASE_3);
      if (old_phase == (unsigned int)PHASE_2) {
        ARTS_DEBUG(
            "check_epoch: CAS won PHASE_2->PHASE_3, firing epoch [Guid:%lu]",
            epoch->guid);
        if (epoch->termination_exit_guid) {
          arts_signal_edt_value(epoch->termination_exit_guid,
                                epoch->termination_exit_slot, total_finish);
        } else {
          arts_shutdown_epoch_fire(epoch->guid);
        }
      }
      return false;
    }
    // Counts differ from last round or first time: record and request another.
    epoch->last_active_count = total_active;
    epoch->last_finished_count = total_finish;
    epoch->phase = (unsigned int)PHASE_2;
    return true;
  }
  epoch->phase = (unsigned int)PHASE_1;
  return (epoch->queued == 0);
}

void reduce_epoch(arts_guid_t epoch_guid, unsigned int active,
                  unsigned int finish) {
  arts_epoch_t *epoch =
      (arts_epoch_t *)arts_route_table_lookup_item(epoch_guid);
  if (epoch) {
    unsigned int total_active =
        arts_atomic_add(&epoch->global_active_count, active);
    unsigned int total_finish =
        arts_atomic_add(&epoch->global_finished_count, finish);
    uint64_t outstanding_before = epoch->outstanding;
    if (arts_atomic_sub_u64(&epoch->outstanding, 1) == 1) {
      total_active += epoch->active_count;
      total_finish += epoch->finished_count;

      ARTS_DEBUG("reduce_epoch [Guid:%lu]: total_active=%u, total_finish=%u, "
                 "phase=%u, outstanding_before=%lu, queued=%lu",
                 epoch_guid, total_active, total_finish, epoch->phase,
                 outstanding_before, epoch->queued);

      // Reset for the next round
      epoch->global_active_count = 0;
      epoch->global_finished_count = 0;

      if (check_epoch(epoch, total_active, total_finish)) {
        ARTS_DEBUG("  check_epoch returned TRUE - broadcasting new request");
        arts_atomic_add_u64(&epoch->outstanding, arts_global_rank_count - 1);
        broadcast_epoch_request(epoch_guid);
        // A better idea will be to know when to kick off a new round
        // the checkinCount == 0 indicates there is a new round can be kicked
        // off
        //                arts_atomic_sub(&epoch->checkinCount, 1);
      } else {
        ARTS_DEBUG(
            "  check_epoch returned FALSE - epoch completed or advancing "
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
                                     unsigned int pool_size,
                                     arts_guid_t *start_guid) {
  if (*epoch_pool_guid == NULL_GUID) {
    *epoch_pool_guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_EDT);
  }

  if (*start_guid == NULL_GUID) {
    *start_guid =
        arts_guid_reserve_range(ARTS_EDT, pool_size, arts_global_rank_id);
  }

  arts_epoch_pool_t *epoch_pool = (arts_epoch_pool_t *)arts_calloc(
      1, sizeof(arts_epoch_pool_t) + (sizeof(arts_epoch_t) * pool_size));
  epoch_pool->index = 0;
  epoch_pool->outstanding = pool_size;
  epoch_pool->size = pool_size;

  arts_route_table_add_item(epoch_pool, *epoch_pool_guid, arts_global_rank_id,
                            false);
  for (unsigned int i = 0; i < pool_size; i++) {
    epoch_pool->pool[i].phase = PHASE_1;
    epoch_pool->pool[i].pool_guid = *epoch_pool_guid;
    epoch_pool->pool[i].guid = arts_guid_from_index(*start_guid, i);
    epoch_pool->pool[i].queued =
        (arts_guid_is_local(*epoch_pool_guid)) ? 0 : EPOCH_BIT;
    if (!arts_guid_is_local(*epoch_pool_guid)) {
      arts_route_table_add_item_race(&epoch_pool->pool[i],
                                     epoch_pool->pool[i].guid,
                                     arts_global_rank_id, false);
      arts_route_table_fire_oo(epoch_pool->pool[i].guid,
                               arts_out_of_order_handler);
    }
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
    if (arts_guid_is_local(epoch->pool_guid)) {
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

    if (arts_guid_is_local(epoch_guid)) {
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

void arts_link_epoch_pool_to_tls(arts_epoch_pool_t *pool) {
  pool->next = epoch_thread_pool;
  epoch_thread_pool = pool;
}

void arts_cleanup_epoch_pools(void) {
  arts_epoch_pool_t *pool = epoch_thread_pool;
  while (pool) {
    arts_epoch_pool_t *next = pool->next;
    arts_free(pool);
    pool = next;
  }
  epoch_thread_pool = NULL;
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
      pool =
          create_epoch_pool(&pool_guid, DEFAULT_EPOCH_POOL_SIZE, &start_guid);

      if (trail_pool) {
        trail_pool->next = pool;
      } else {
        epoch_thread_pool = pool;
      }

      for (unsigned int i = 0; i < arts_global_rank_count; i++) {
        if (i != arts_global_rank_id) {
          arts_remote_epoch_init_pool_send(i, DEFAULT_EPOCH_POOL_SIZE,
                                           start_guid, pool_guid);
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

  epoch->termination_exit_guid = edt_guid;
  epoch->termination_exit_slot = slot;
  arts_route_table_add_item_race(epoch, epoch->guid, arts_global_rank_id,
                                 false);
  arts_route_table_fire_oo(epoch->guid, arts_out_of_order_handler);
  return epoch;
}

void arts_yield() {
  TIME_EDT_EXEC_STOP();
  INCREMENT_NUM_YIELD_BY(1);
  thread_local_t tl;
  arts_save_thread_local(&tl);
  TIME_YIELD_START();
  arts_node_info.scheduler();
  TIME_YIELD_STOP();
  arts_restore_thread_local(&tl);
  TIME_EDT_EXEC_START();
}

/*
 * arts_wait_on_handle — Block current EDT until the given epoch completes.
 *
 * Increments the epoch's finished counter (this EDT is now "done" from the
 * epoch's perspective), then spin-polls the scheduler loop until the epoch
 * is removed from the route table (by delete_epoch).  Also breaks out if
 * the thread's alive flag becomes false (e.g., arts_shutdown was called).
 *
 * Uses route-table lookup instead of a volatile flag to avoid accessing
 * the epoch struct after delete_epoch frees it.
 */
bool arts_wait_on_handle(arts_guid_t epoch_guid) {
  TIME_EDT_EXEC_STOP();
  arts_guid_t *guid = arts_check_epoch_is_root(epoch_guid);
  ARTS_INFO("arts_wait_on_handle: Waiting on epoch [Guid:%lu]", epoch_guid);
  // For now lets leave this rule here
  if (guid) {
    arts_guid_t local = *guid;
    *guid = NULL_GUID; // Unset
    arts_epoch_t *epoch = (arts_epoch_t *)arts_route_table_lookup_item(local);
    if (!epoch) {
      // Epoch may still be in reserved state in route table; spin briefly
      for (int retries = 0; !epoch && retries < 1000; retries++) {
        epoch = (arts_epoch_t *)arts_route_table_lookup_item(local);
      }
      if (!epoch) {
        ARTS_WARN(
            "arts_wait_on_handle: Epoch [Guid:%lu] not found in route table",
            local);
        TIME_EDT_EXEC_START();
        return false;
      }
    }
    increment_finished_epoch(local);

    // Release all DB frontier locks before blocking so consumer EDTs can
    // proceed while this EDT waits on the epoch.
    arts_wait_release_dbs();

    INCREMENT_NUM_YIELD_BY(1);
    thread_local_t tl;
    arts_save_thread_local(&tl);
    TIME_YIELD_START();
    while (arts_thread_info.alive) {
      arts_epoch_t *e = (arts_epoch_t *)arts_route_table_lookup_item(local);
      if (!e || e->phase == PHASE_3) {
        break;
      }
      arts_node_info.scheduler();
    }
    // Continue running until the scheduler reports no more ready work
    while (arts_node_info.scheduler()) {
      ;
    }
    TIME_YIELD_STOP();
    arts_restore_thread_local(&tl);

    // Re-acquire all DB frontier locks after the epoch completes.
    arts_wait_reacquire_dbs();

    clean_epoch_pool();

    TIME_EDT_EXEC_START();
    return true;
  }
  TIME_EDT_EXEC_START();
  return false;
}
