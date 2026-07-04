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
#include <time.h>

#include <pthread.h>
#include <unistd.h>
#ifndef __APPLE__
#include <sched.h>
#endif

#include "arts/counter/counter.h"
#include "arts/runtime_state.h"
#include "arts/system/config.h"
#include "arts/system/print.h"
#include "arts/transport/dispatcher.h"
#include "arts/utils/malloc.h"

unsigned int arts_global_rank_id;
unsigned int arts_global_rank_count;
unsigned int arts_global_master_rank_id;

/* Local lifecycle controls (moved here from the former utils/introspect.c:
 * they mutate runtime state, so they belong with the thread/shutdown logic,
 * not with read-only introspection). */
void arts_stop_local_worker(void) { arts_thread_info.alive = false; }

void arts_stop_local_node(void) { arts_runtime_stop(); }
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

  /* End-to-end marker (rank 0): both stamps are set — e2e_start_stamp at the
   * application start (runtime.c) and e2e_end_stamp at shutdown recognition
   * (shutdown.c).  Print the span on stderr, gated by $ARTS_E2E_MARKER, in the
   * same "[E2E] <ns>" form the reference runtimes use, so the harness parses
   * all three runtimes identically. */
  if (!arts_global_rank_id && arts_node_info.e2e_marker_enabled) {
    fprintf(stderr, "[E2E] %lu\n",
            (unsigned long)(arts_node_info.e2e_end_stamp -
                            arts_node_info.e2e_start_stamp));
    fflush(stderr);
  }

  /* Stop the progress threads.  After the transport cutover they no longer
   * block in a socket poll (they reap the fabric with a short bounded wait), so
   * clearing alive is enough — no socket shutdown is needed to wake them, and
   * the data sockets were already closed once bootstrap finished. */
  arts_runtime_stop_network();

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
  // Join ALL threads (workers and network threads) with timeout + cancel
  // escalation, so a single stuck thread never wedges the entire process.
  //
  // Budget (from the shutdown protocol plan):
  //   JOIN_DEADLINE_MS   = 1500  per-thread cooperative join deadline
  //   CANCEL_DEADLINE_MS = 500   post-cancel grace window
  //
  // Escalation: timed join → pthread_cancel → timed join → give up and
  // move on (we are about to exit the process anyway).
  {
    const long join_deadline_ns = 1500L * 1000000L;  /* 1.5 s */
    const long cancel_deadline_ns = 500L * 1000000L; /* 0.5 s */
    for (int i = 1; i < arts_node_info.total_thread_count; i++) {
      struct timespec deadline;
      (void)clock_gettime(CLOCK_REALTIME, &deadline);
      deadline.tv_nsec += join_deadline_ns;
      while (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_nsec -= 1000000000L;
        deadline.tv_sec += 1;
      }
      int rc = pthread_timedjoin_np(node_thread_list[i], NULL, &deadline);
      if (rc == 0) {
        continue;
      }
      ARTS_INFO("arts_thread_main_join: thread %d did not join within "
                "%ld ms (rc=%d), cancelling",
                i, join_deadline_ns / 1000000L, rc);
      pthread_cancel(node_thread_list[i]);
      (void)clock_gettime(CLOCK_REALTIME, &deadline);
      deadline.tv_nsec += cancel_deadline_ns;
      while (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_nsec -= 1000000000L;
        deadline.tv_sec += 1;
      }
      rc = pthread_timedjoin_np(node_thread_list[i], NULL, &deadline);
      if (rc != 0) {
        ARTS_INFO("arts_thread_main_join: thread %d did not join after "
                  "cancel (rc=%d); leaking and continuing",
                  i, rc);
      }
    }
  }
  arts_runtime_global_cleanup();
  // arts_free(args);
  arts_free(mask);
  arts_free(node_thread_list);
}

void arts_thread_init(struct arts_config_s *config) {
  g_config = config;

  /* Adjust the progress-thread count now that rank_count is known.  The sender
   * role is gone (its cfg count is folded into workers at config time), and a
   * single-node run has no progress thread.  Progress threads no longer own a
   * per-socket partition — they all reap the shared fabric completion queue —
   * so the old node*port ceiling no longer applies. */
  if (arts_global_rank_count == 1) {
    config->progress_thread_count = 0;
  }
  config->worker_thread_count =
      config->thread_count - config->progress_thread_count;

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
