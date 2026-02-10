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

/*
 * arts_tMT.c
 *
 *  Created on: March 30, 2018
 *      Author: Andres Marquez (@awmm)
 *
 *
 * This file is subject to the license agreement located in the file LICENSE
 * and cannot be distributed without it. This notice cannot be
 * removed or modified.
 *
 *
 *
 */

#define PT_CONTEXTS // maintain contexts via PThreads

#include "arts/system/tmt.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <unistd.h>

#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/system/arts_print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"

#define ONLY_ONE_THREAD
// while(!arts_test_state_one_left(local_pool->alias_running) &&
// arts_thread_info.alive)

msi_t *arts_t_mt_msi = NULL; // tmt shared data structure
__thread unsigned int alias_id = 0;
__thread msi_t *local_pool = NULL;
__thread internal_msi_t *local_internal = NULL;
__thread bool tmt_shutdown_flag = false;

static inline internal_msi_t *arts_get_msi_offset_ptr(internal_msi_t *head,
                                                 unsigned int thread) {
  unsigned int num_internal = (thread) / arts_node_info.tmt;
  internal_msi_t *ptr = head;
  for (unsigned int i = 0; i < num_internal; i++) {
    ptr = ptr->next;
}
  return ptr;
}

static inline arts_ticket_t gen_ticket() {
  arts_ticket_bits_t ticket;
  ticket.fields.rank = arts_global_rank_id;
  ticket.fields.unit = arts_thread_info.group_id;
  ticket.fields.thread = alias_id;
  ticket.fields.valid = 1;
  return (arts_ticket_t)ticket.bits;
}

static inline bool arts_accessor_state(volatile accst_t *all_states,
                                     unsigned int start, bool flipstate) {
  uint64_t uint64_t_state = *all_states;
  uint64_t current_pos = 1UL << start;
  bool b_state = (uint64_t_state & current_pos) ? true : false;

  if (flipstate) {
    arts_atomic_fetch_x_or_u64(all_states, current_pos);
}
  return b_state;
}

static inline unsigned int arts_next_candidate(const volatile accst_t *all_states) {
  return ffsll((long long)*all_states);
}

static inline bool arts_test_state_empty(const volatile accst_t *all_states) {
  return !*all_states;
}

static inline bool arts_test_state_one_left(const volatile accst_t *all_states) {
  return *all_states && !(*all_states & (*all_states - 1));
}

static inline void arts_put_to_work(unsigned int rank, internal_msi_t *ptr,
                                 unsigned int thread, bool avail) {
  if (rank == arts_global_rank_id) {
    arts_accessor_state(&ptr->alias_running, thread, true);
    if (avail) {
      arts_accessor_state(&ptr->alias_avail, thread, true);
}

#ifdef PT_CONTEXTS
    pthread_mutex_lock(&ptr->mutex);
    pthread_cond_signal(&ptr->cond[thread]);
    pthread_mutex_unlock(&ptr->mutex);
#endif
  }
}

static inline void arts_put_to_sleep(unsigned int rank, internal_msi_t *ptr,
                                  unsigned int thread, bool avail) {
  if (rank == arts_global_rank_id) {
    arts_accessor_state(&ptr->alias_running, thread, true);
    if (avail) {
      arts_accessor_state(&ptr->alias_avail, thread, true);
}

#ifdef PT_CONTEXTS
    pthread_mutex_lock(&ptr->mutex);
    pthread_cond_wait(&ptr->cond[thread], &ptr->mutex);
    pthread_mutex_unlock(&ptr->mutex);
#endif
  }
}

static void *arts_alias_thread_loop(void *arg) {
  tmask_t *t_args = (tmask_t *)arg;

  // set thread local vars
  alias_id = t_args->alias_id;
  memcpy(&arts_thread_info, t_args->tlToCopy, sizeof(struct arts_runtime_private_s));

  unsigned int unit_id = arts_thread_info.group_id;
  unsigned int num_at = arts_node_info.tmt;
  ARTS_DEBUG("Alias: %u", alias_id);

  local_pool = &arts_t_mt_msi[arts_thread_info.group_id];
  local_internal = t_args->local_internal;
  local_internal->alive[(alias_id % num_at)] = &arts_thread_info.alive;
  local_internal->initShutdown[(alias_id % num_at)] = &tmt_shutdown_flag;
  if (arts_node_info.pin_threads) {
    ARTS_DEBUG("PINNING to %u:%u", arts_thread_info.group_id, alias_id);
    arts_pthread_affinity(arts_thread_info.core_id, true);
    //        arts_abstract_machine_model_pin_thread(arts_thread_info.core_id);
  }

  arts_accessor_state(&local_internal->alias_running, alias_id % num_at, true);
  arts_accessor_state(&local_internal->alias_avail, alias_id % num_at, true);

  if (sem_post(t_args->startUpSem) == -1) { // finished  mask copy
    ARTS_INFO("FAILED SEMI INIT POST %u %u", arts_thread_info.group_id, alias_id);
    //        exit(EXIT_FAILURE);
  }

  arts_atomic_sub(&local_internal->startUpCount, 1);
  arts_put_to_sleep(arts_global_rank_id, local_internal, alias_id % num_at,
                 true); // Toggle availability
  ONLY_ONE_THREAD;

  arts_runtime_loop();
  arts_atomic_sub(&local_internal->shutDownCount, 1);

  return NULL;
}

static inline void arts_create_contexts(struct arts_runtime_private_s *semi_private,
                                      internal_msi_t *ptr, unsigned int offset) {
#ifdef PT_CONTEXTS
  tmask_t tmask;
  pthread_attr_t attr;
  long page_size = sysconf(_SC_PAGESIZE);
  size_t size = page_size;

  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, size);

  // Init semiphores
  unsigned int num_at = arts_node_info.tmt;
  for (int i = 0; i < num_at; ++i) {
    if (sem_init(&ptr->sem[i], 0, 0) == -1) {
      ARTS_INFO("FAILED SEMI INIT %u %u", arts_thread_info.group_id, i);
      //            exit(EXIT_FAILURE);
    }
  }

  pthread_mutex_init(&ptr->mutex, NULL);
  for (int i = 0; i < arts_node_info.tmt; i++) {
    pthread_cond_init(&ptr->cond[i], NULL);
  }

  tmask.tlToCopy = semi_private;
  tmask.local_internal = ptr;
  tmask.startUpSem = (num_at > 0) ? &local_internal->sem[alias_id % num_at] : NULL;

  unsigned int end = num_at - 1;
  if (offset) {
    end = num_at;
  } else {
    offset = 1;
}

  for (int i = 0; i < end; ++i) {
    tmask.alias_id = i + offset;
    ARTS_DEBUG("Creating %u : %u of %u", tmask.alias_id, i, end);
    if (pthread_create(&ptr->aliasThreads[i], &attr, &arts_alias_thread_loop,
                       &tmask)) {
      ARTS_INFO("FAILED ALIAS THREAD CREATION %u %u", arts_thread_info.group_id,
                i + offset);
      //            exit(EXIT_FAILURE);
    }
    ARTS_DEBUG("Master %u: Waiting in thread creation %d",
               arts_thread_info.group_id, i + offset);
    if (num_at == 0 ||
        sem_wait(&local_internal->sem[alias_id % num_at]) ==
        -1) { // wait to finish mask copy
      ARTS_INFO("FAILED SEMI INIT WAIT %u %u", arts_thread_info.group_id,
                i + offset);
      //            exit(EXIT_FAILURE);
    }
  }
#endif
}

static inline void arts_destroy_contexts(internal_msi_t *ptr, bool head) {
#ifdef PT_CONTEXTS
  unsigned int num_at = arts_node_info.tmt;
  unsigned int end = (head) ? num_at - 1 : num_at;

  ARTS_DEBUG("ALIAS JOIN: %u", arts_thread_info.group_id);
  for (unsigned int i = 0; i < end; i++) {
    ARTS_DEBUG("Joining %u %u", i, head);
    pthread_join(ptr->aliasThreads[i], NULL);
  }

  ARTS_DEBUG("SEM DESTROY: %u", arts_thread_info.group_id);
  for (unsigned int i = 0; i < num_at; i++) {
    sem_destroy(&ptr->sem[i]);
}
  for (int i = 0; i < arts_node_info.tmt; i++) {
    pthread_cond_destroy(&ptr->cond[i]);
  }
  pthread_mutex_destroy(&ptr->mutex);
#endif
}

// RT visible functions
// COMMENT: MasterThread (MT) is the original thread
void arts_tmt_node_init(unsigned int num_threads) {
  if (num_threads > 64) {
    arts_printf("Temporal multi-threading can't run more than 64 threads per core");
    num_threads = 64;
  }

  if (arts_node_info.tmt) {
    arts_t_mt_msi = (msi_t *)arts_calloc_align(num_threads, sizeof(msi_t), 64);
  }
}

void arts_tmt_construct_new_internal_msi(msi_t *root, unsigned int num_at,
                                    struct arts_runtime_private_s *semi_private) {
  // Move to the last one...
  unsigned int offset = 0;
  internal_msi_t *ptr = root->head;
  if (!root->head) {
    root->head = ptr = (internal_msi_t *)arts_calloc(1, sizeof(internal_msi_t));
  } else {
    offset = num_at;
    while (ptr->next) {
      offset += num_at;
      ptr = ptr->next;
    }
    ptr->next = (internal_msi_t *)arts_calloc(1, sizeof(internal_msi_t));
    ptr = ptr->next;
  }

  unsigned int total = (offset) ? num_at : num_at - 1;
  ptr->aliasThreads = (pthread_t *)arts_malloc(sizeof(pthread_t) * (total));
  ptr->sem = (sem_t *)arts_malloc(sizeof(sem_t) * (num_at));
  ptr->alive = (volatile bool **)arts_calloc(num_at, sizeof(bool *));
  ptr->initShutdown = (volatile bool **)arts_calloc(num_at, sizeof(bool *));
  ptr->alias_running = (offset) ? 0UL : 1U; // MT is running on thread 0

  if (!offset) {
    ptr->initShutdown[0] = &tmt_shutdown_flag;
}

  // More clever ways break for 64 alias
  // Start at 1 since MT is bit 0 and is running
  unsigned int start = (offset) ? 0 : 1;
  for (unsigned int i = start; i < num_at; i++) {
    ptr->alias_avail |= 1UL << i;
}

  ptr->startUpCount = ptr->shutDownCount = total;
  ptr->next = NULL;

  if (!offset) { // Thread zero needs to get initilaized...
    local_internal = ptr;
}

  arts_create_contexts(semi_private, ptr, offset);
  while (ptr->startUpCount) {
    ;
}

  arts_atomic_add(&root->total, arts_node_info.tmt);
}

void arts_tmt_runtime_private_init(struct thread_mask_s *unit,
                               struct arts_runtime_private_s *semi_private) {
  (void)unit;
  local_pool = &arts_t_mt_msi[arts_thread_info.group_id];
  local_pool->wakeUpNext = 0;
  local_pool->wakeQueue = arts_new_queue();
  arts_tmt_construct_new_internal_msi(local_pool, arts_node_info.tmt, semi_private);
}

// Shutdown is painful...  We use a two phased approach.
// 1. Indicate we need to shut down using initShutdown
// 2. Whatever thread wakes up next will see it is time to close and switch to
//    thread alias_id = 0 and then turn off alive flag for the rest of the alias
//    threads.  Next we wait of a single threads aliases to exit then we turn
//    off our local spin flag, and the thread will go into rt cleanup mode.
bool arts_tmt_runtime_stop() {
  if (arts_node_info.tmt) {
    ARTS_DEBUG("SETTING STOP FLAG: %u %u", arts_thread_info.group_id, alias_id);
    for (unsigned int j = 0; j < arts_node_info.worker_thread_count; j++) {
      for (internal_msi_t *ptr = arts_t_mt_msi[j].head; ptr != NULL;
           ptr = ptr->next) {
        for (unsigned int i = 0; i < arts_node_info.tmt; i++) {
          *(ptr->initShutdown[i]) = true;
        }
      }
    }
    return false;
  }
  return true;
}

bool arts_tmt_check_shutdown() {
  if (tmt_shutdown_flag) {
    if (alias_id) {
      arts_put_to_work(arts_global_rank_id, local_pool->head, 0,
                    true); // available so flip
      arts_put_to_sleep(arts_global_rank_id, local_internal,
                     alias_id % arts_node_info.tmt, true);
    } else {
      ARTS_DEBUG("THE STOP %u %u", arts_thread_info.group_id, alias_id);
      for (internal_msi_t *ptr = local_pool->head; ptr != NULL; ptr = ptr->next) {
        for (unsigned int i = 0; i < arts_node_info.tmt; i++) {
          if (ptr->alive[i]) {
            *ptr->alive[i] = false;
}
        }

        while (ptr->shutDownCount) {
          for (unsigned int i = 0; i < arts_node_info.tmt; i++) {
            sem_post(&ptr->sem[i]);
}
        }
      }

      while (!arts_node_info.local_spin[arts_thread_info.group_id]) {
        ;
}
      (*arts_node_info.local_spin[arts_thread_info.group_id]) = false;

      return true;
    }
  }
  return false;
}

void arts_tmt_runtime_private_cleanup() {
  if (arts_node_info.tmt) {
    bool head = true;
    internal_msi_t *trail = NULL;
    internal_msi_t *ptr = local_pool->head;
    while (ptr) {
      trail = ptr;
      ptr = ptr->next;
      arts_destroy_contexts(trail, head);
      head = false;
    }
  }
}

void arts_next_context() {
  if (arts_node_info.tmt && arts_thread_info.alive) {
    unsigned int cand = arts_atomic_swap(&local_pool->wakeUpNext, 0);
    if (!cand) {
      cand = dequeue(local_pool->wakeQueue);
}
    if (cand) {
      cand--;
      arts_put_to_work(
          arts_global_rank_id, arts_get_msi_offset_ptr(local_pool->head, cand),
          cand % arts_node_info.tmt, false); // already blocked don't flip
    } else {
      cand = (alias_id + 1) % arts_node_info.tmt;
      internal_msi_t *ptr = local_internal;
      if (!cand) {
        ptr = (local_internal->next) ? local_internal->next : local_pool->head;
      }
      ARTS_DEBUG("%u link NEXT: %u total: %u %p %u next: %p head: %p",
                 arts_thread_info.group_id, alias_id, local_pool->total, ptr, cand,
                 local_internal->next, local_pool->head);
      arts_put_to_work(arts_global_rank_id, ptr, cand, true); // available so flip
    }

    arts_put_to_sleep(arts_global_rank_id, local_internal, alias_id % arts_node_info.tmt,
                   true);
    ONLY_ONE_THREAD;
    arts_tmt_check_shutdown();
  }
}

void arts_wake_up_context() {
  if (arts_node_info.tmt && arts_thread_info.alive) {
    unsigned int cand = arts_atomic_swap(&local_pool->wakeUpNext, 0);
    if (!cand) {
      cand = dequeue(local_pool->wakeQueue);
}
    if (cand) {
      cand--;
      arts_put_to_work(arts_global_rank_id,
                    arts_get_msi_offset_ptr(local_pool->head, cand),
                    cand % arts_node_info.tmt, false);
      arts_put_to_sleep(arts_global_rank_id, local_internal,
                     alias_id % arts_node_info.tmt, true);
      ONLY_ONE_THREAD;
      arts_tmt_check_shutdown();
    }
  }
}
// End of RT visible functions

void arts_context_switch_internal() {
  unsigned int cand = arts_atomic_swap(&local_pool->wakeUpNext, 0);
  if (!cand) {
    cand = dequeue(local_pool->wakeQueue);
}
  if (!cand) {
    cand = arts_next_candidate(&local_internal->alias_avail);
}
  if (cand) {
    cand--;
    arts_put_to_work(arts_global_rank_id, local_internal, cand % arts_node_info.tmt,
                  true);
    arts_put_to_sleep(arts_global_rank_id, local_internal, alias_id % arts_node_info.tmt,
                   false); // do not change availability
  } else {
    internal_msi_t *last = NULL;
    for (internal_msi_t *ptr = local_pool->head; ptr != NULL; ptr = ptr->next) {
      if ((cand = arts_next_candidate(&ptr->alias_avail))) {
        cand--;
        arts_put_to_work(arts_global_rank_id, ptr, cand % arts_node_info.tmt, true);
        arts_put_to_sleep(arts_global_rank_id, local_internal,
                       alias_id % arts_node_info.tmt,
                       false); // do not change availability
        return;
      }

      if (!ptr->next) {
        last = ptr;
}
    }

    arts_tmt_construct_new_internal_msi(local_pool, arts_node_info.tmt,
                                   &arts_thread_info);
    if (last) {
      arts_put_to_work(arts_global_rank_id, last->next, 0, true);
    }
    arts_put_to_sleep(arts_global_rank_id, local_internal, alias_id % arts_node_info.tmt,
                   false);
  }
  ONLY_ONE_THREAD;
  arts_tmt_check_shutdown();
}

bool arts_context_switch(unsigned int wait_count) {
  ARTS_DEBUG("CONTEXT SWITCH");
  if (arts_node_info.tmt && arts_thread_info.alive) {
    bool first_flag = true;
    if (wait_count) {
      arts_atomic_add(&local_pool->blocked, 1);
}
    volatile unsigned int *wait_flag =
        &local_internal->ticket_counter[alias_id % arts_node_info.tmt];
    arts_atomic_add(wait_flag, wait_count);
    while (*wait_flag && arts_thread_info.alive) {
      if (first_flag) {
        arts_context_switch_internal();
        first_flag = false;
      } else {
        arts_next_context();
}
    }
    return true;
  }
  return false;
}

void arts_open_context_switch() {
  if (arts_node_info.tmt && arts_thread_info.alive) {
    arts_context_switch_internal();
  }
}

bool arts_signal_context(arts_ticket_t wait_ticket) {
  ARTS_DEBUG("SIGNAL CONTEXT %u", arts_node_info.tmt);
  arts_ticket_bits_t ticket = (arts_ticket_bits_t){.bits = wait_ticket};
  unsigned int rank = (unsigned int)ticket.fields.rank;
  unsigned int unit = (unsigned int)ticket.fields.unit;
  unsigned int thread = (unsigned int)ticket.fields.thread;

  if (arts_node_info.tmt) {
    if (ticket.bits) {
      if (rank == arts_global_rank_id) {
        internal_msi_t *ptr =
            arts_get_msi_offset_ptr(arts_t_mt_msi[unit].head, thread);
        if (!arts_atomic_sub(&ptr->ticket_counter[thread % arts_node_info.tmt],
                           1)) {
          arts_atomic_sub(&arts_t_mt_msi[unit].blocked, 1);
          unsigned int alias = thread + 1;
          if (arts_atomic_cswap(&arts_t_mt_msi[unit].wakeUpNext, 0, alias) != 0) {
            enqueue(alias, arts_t_mt_msi[unit].wakeQueue);
}
        }
      } else {
        arts_remote_signal_context(rank, wait_ticket);
      }
      return true;
    }
  }
  return false;
}

bool arts_avail_context() {
  return (arts_node_info.tmt && local_pool->total < MAX_TOTAL_THREADS_PER_MAX &&
          local_pool->blocked < MAX_TOTAL_THREADS_PER_MAX);
}

arts_ticket_t arts_get_context_ticket() {
  arts_ticket_bits_t ticket;
  ticket.bits = 0;
  if (arts_node_info.tmt) {
    ticket.bits = gen_ticket();
}
  return (arts_ticket_t)ticket.bits;
}

unsigned int arts_get_context_id() { return alias_id; }