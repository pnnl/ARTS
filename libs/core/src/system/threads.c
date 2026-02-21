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
#include "arts/system/threads.h"
#include "arts/counter/Preamble.h"

#include <limits.h>
#include <stdlib.h>

#include <pthread.h>
#include <unistd.h>
#ifndef __APPLE__
#include <sched.h>
#endif

#include "arts.h"
#include "arts/counter/counter.h"
#include "arts/network/remote.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/system/arts_print.h"
#include "arts/system/config.h"
#include "arts/utils/malloc.h"

unsigned int arts_global_rank_id;
unsigned int arts_global_rank_count;
unsigned int arts_global_master_rank_id;
struct arts_config_s *g_config;

struct thread_mask_s *mask;
pthread_t *node_thread_list;

void *arts_thread_loop(void *data) {
  struct thread_mask_s *thread = (struct thread_mask_s *)data;
  arts_runtime_private_init(thread, g_config);
  arts_runtime_loop();
  arts_runtime_private_cleanup();

  // Save final counter values to saved_counters before thread exits
  // This must happen after cleanup but before thread terminates
  unsigned int thread_id = thread->id;
  arts_counter_t *saved = arts_node_info.saved_counters[thread_id];
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    saved[i].count = arts_thread_local_counters[i].count;
    saved[i].start = 0;
  }
  arts_object_save_thread_data(thread_id);
  // Mark thread as closed by clearing live_counters pointer
  // Capture thread will skip threads with NULL live_counters
  arts_node_info.live_counters[thread_id] = NULL;

  return NULL;
  // pthread_exit(NULL);
}

/*
 * arts_thread_main_join — Main thread (thread 0) entry after initialization.
 *
 * Runs the runtime loop until alive==false, then cleans up and joins all
 * other pthreads.  Counter data is saved before joining to ensure no
 * data loss.
 */
void arts_thread_main_join() {
  ARTS_DEBUG("arts_thread_main_join: main thread entering runtime_loop");
  arts_runtime_loop();
  ARTS_DEBUG("arts_thread_main_join: main thread exited runtime_loop, joining "
             "%u threads",
             arts_node_info.total_thread_count - 1);
  TIME_TOTAL_STOP();
  arts_runtime_private_cleanup();

  // Save main thread's final counter values before joining other threads
  arts_counter_t *saved = arts_node_info.saved_counters[0];
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    saved[i].count = arts_thread_local_counters[i].count;
    saved[i].start = 0;
  }
  arts_object_save_thread_data(0);
  // Mark main thread as closed
  arts_node_info.live_counters[0] = NULL;

  // File-based counter aggregation: no socket synchronization needed
  // Each node writes its own JSON file independently, master polls filesystem
  // Join ALL threads (workers and network threads)
  for (int i = 1; i < arts_node_info.total_thread_count; i++) {
    pthread_join(node_thread_list[i], NULL);
  }

  arts_runtime_global_cleanup();
  // arts_free(args);
  arts_free(mask);
  arts_free(node_thread_list);
}

void arts_thread_init(struct arts_config_s *config) {
  g_config = config;

  /* Validate/adjust network thread counts now that rank_count is known. */
  if (arts_global_rank_count == 1) {
    config->sender_thread_count = 0;
    config->receiver_thread_count = 0;
  } else {
    unsigned int max_net = (arts_global_rank_count - 1) * config->port_count;
    if (config->sender_thread_count > max_net) {
      ARTS_ERROR(
          "sender_threads (%u) exceeds node*port limit (%u nodes * %u ports)",
          config->sender_thread_count, arts_global_rank_count - 1,
          config->port_count);
    }
    if (config->receiver_thread_count > max_net) {
      ARTS_ERROR(
          "receiver_threads (%u) exceeds node*port limit (%u nodes * %u ports)",
          config->receiver_thread_count, arts_global_rank_count - 1,
          config->port_count);
    }
  }
  config->worker_thread_count = config->thread_count -
                                config->sender_thread_count -
                                config->receiver_thread_count;

  mask =
      (struct thread_mask_s *)arts_malloc(sizeof(*mask) * config->thread_count);
  get_thread_mask(config, mask);
  arts_runtime_node_init(config);
  print_mask(mask, config->thread_count);

  node_thread_list =
      (pthread_t *)arts_malloc(sizeof(pthread_t) * config->thread_count);
  unsigned int thread_count = config->thread_count;

  /* Compute page-aligned stack size (0 = use default). */
  size_t stack_size = 0;
  if (config->stack_size) {
    long page_size = sysconf(_SC_PAGESIZE);
    stack_size = ((config->stack_size % page_size > 0) +
                  (config->stack_size / page_size)) *
                 (size_t)page_size;
  }

  /* Create worker and network threads with optional pinning. */
  for (unsigned int i = 1; i < thread_count; i++) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    if (stack_size) {
      pthread_attr_setstacksize(&attr, stack_size);
    }
#ifndef __APPLE__
    if (config->pin_threads) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(mask[i].pu_id, &set);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
    }
#endif
    pthread_create(&node_thread_list[i], &attr, &arts_thread_loop, &mask[i]);
    pthread_attr_destroy(&attr);
  }

  /* Pin main thread (thread 0). */
#ifndef __APPLE__
  if (config->pin_threads) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(mask[0].pu_id, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
  }
#endif
  arts_runtime_private_init(&mask[0], config);
}

/*
 * arts_shutdown — Initiate global shutdown of the ARTS runtime.
 *
 * Multi-node: delegates to arts_remote_shutdown() which broadcasts the
 *   shutdown message and waits for acknowledgements.  The send thread
 *   then calls arts_runtime_stop() after the timeout.
 * Single-node: directly calls arts_runtime_stop() to signal all threads.
 *
 * Called from:
 *   - arts_shutdown_epoch_fire() when the shutdown epoch completes.
 *   - User code via the arts_shutdown() public API.
 */
void arts_shutdown() {
  ARTS_INFO("arts_shutdown: rank_count=%u, rank_id=%u", arts_global_rank_count,
            arts_global_rank_id);
  if (arts_global_rank_count > 1) {
    arts_remote_shutdown();
  }

  if (arts_global_rank_count == 1) {
    arts_runtime_stop();
  }

  (void)fflush(stdout);
}

_Noreturn void arts_abort(uint8_t error_code) {
  (void)fflush(stdout);
  (void)fflush(stderr);
  exit(error_code);
}
