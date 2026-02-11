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
#include "arts/introspection/Preamble.h"
#define GNU_SOURCE
#include "arts/system/threads.h"

#include <limits.h>

#include <pthread.h>
#include <unistd.h>

#include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/introspection/counter.h"
#include "arts/network/remote.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/system/arts_print.h"
#include "arts/system/config.h"

unsigned int arts_global_rank_id;
unsigned int arts_global_rank_count;
unsigned int arts_global_master_rank_id;
struct arts_config_s *g_config;
struct arts_config_s *config;

struct thread_mask_s *mask;
pthread_t *node_thread_list;


void *arts_thread_loop(void *data) {
  struct thread_mask_s *unit = (struct thread_mask_s *)data;
  if (unit->pin) {
    arts_abstract_machine_model_pin_thread(&unit->core_info);
}
  arts_runtime_private_init(unit, g_config);
  arts_runtime_loop();
  arts_runtime_private_cleanup();

  // Save final counter values to saved_counters before thread exits
  // This must happen after cleanup but before thread terminates
  unsigned int thread_id = unit->id;
  arts_counter_t *saved = arts_node_info.saved_counters[thread_id];
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    saved[i].count = arts_thread_local_counters[i].count;
    saved[i].start = 0;
  }
  // Mark thread as closed by clearing live_counters pointer
  // Capture thread will skip threads with NULL live_counters
  arts_node_info.live_counters[thread_id] = NULL;

  return NULL;
  // pthread_exit(NULL);
}

void arts_thread_main_join() {
  arts_runtime_loop();
  END_TO_END_TIME_STOP();
  arts_runtime_private_cleanup();

  // Save main thread's final counter values before joining other threads
  arts_counter_t *saved = arts_node_info.saved_counters[0];
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    saved[i].count = arts_thread_local_counters[i].count;
    saved[i].start = 0;
  }
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
  destroy_thread_mask(mask);
  arts_free(node_thread_list);
}

void arts_thread_init(struct arts_config_s *config) {
  g_config = config;
  mask = get_thread_mask(config);
  node_thread_list = (pthread_t *)arts_malloc(sizeof(pthread_t) *
                                           arts_node_info.total_thread_count);
  unsigned int i = 0;
  unsigned int thread_count = arts_node_info.total_thread_count;

  if (config->stack_size) {
    void *stack;
    pthread_attr_t attr;
    long page_size = sysconf(_SC_PAGESIZE);
    size_t size =
        ((config->stack_size % page_size > 0) + (config->stack_size / page_size)) *
        page_size;
    for (i = 1; i < thread_count; i++) {
      pthread_attr_init(&attr);
      pthread_attr_setstacksize(&attr, size);
      pthread_create(&node_thread_list[i], &attr, &arts_thread_loop, &mask[i]);
    }
  } else {
    for (i = 1; i < thread_count; i++) {
      pthread_create(&node_thread_list[i], NULL, &arts_thread_loop, &mask[i]);
}
  }
  if (mask->pin) {
    arts_abstract_machine_model_pin_thread(&mask->core_info);
}
  arts_runtime_private_init(&mask[0], config);
}

void arts_shutdown() {
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

void arts_thread_set_os_thread_count(unsigned int threads) {
  pthread_setconcurrency((int)threads);
}

void arts_pthread_affinity(unsigned int cpu_core_id, bool verbose) {
#ifdef __APPLE__
  (void)cpu_core_id;
  (void)verbose;
  return;
#else
  cpu_set_t cpuset;
  pthread_t thread;
  thread = pthread_self();
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_core_id, &cpuset);
  if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) && verbose) {
    ARTS_INFO("Failed to set affinity %u", cpu_core_id);
  }
#endif
}

int *arts_valid_pthread_affinity(unsigned int *size) {
#ifdef __APPLE__
  // macOS doesn't support CPU affinity, return a simple array
  *size = 1;
  int *affin = (int *)arts_malloc(sizeof(int));
  affin[0] = 0; // Just return core 0
  return affin;
#else
  unsigned int count = 0;
  cpu_set_t cpuset;
  pthread_t thread = pthread_self();

  int *affin = (int *)arts_malloc(sizeof(int) * CPU_SETSIZE);
  for (int i = 0; i < CPU_SETSIZE; i++) {
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) {
      affin[i] = -1;
    } else {
      affin[i] = i;
      count++;
    }
  }
  *size = count;
  return affin;
#endif
}
