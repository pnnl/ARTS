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

#include "arts.h"
#include "arts/counter/counter.h"
#include "arts/runtime_state.h"
#include "arts/system/config.h"
#include "arts/system/print.h"
#include "arts/transport/dispatcher.h"
#include "arts/utils/atomics.h"
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

  /* Phase C: close the network layer so receivers wake up from RPOLL
   * (they would otherwise block up to 300 s). Uses SHUT_WR on send
   * sockets so any buffered SHUTDOWN_MSG broadcast bytes still get
   * delivered via FIN, and SHUT_RD on recv sockets. Receivers see EOF
   * on the next RPOLL iteration and exit their loop. */
  if (arts_global_rank_count > 1) {
    arts_ll_server_shutdown();
  }
  /* Belt-and-braces: explicitly clear alive on network threads too, so
   * any sender that is not currently inside a socket call also exits
   * promptly. Idempotent with respect to the EOF/EPIPE path. */
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
    const long JOIN_DEADLINE_NS = 1500L * 1000000L;  /* 1.5 s */
    const long CANCEL_DEADLINE_NS = 500L * 1000000L; /* 0.5 s */
    for (int i = 1; i < arts_node_info.total_thread_count; i++) {
      struct timespec deadline;
      clock_gettime(CLOCK_REALTIME, &deadline);
      deadline.tv_nsec += JOIN_DEADLINE_NS;
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
                i, JOIN_DEADLINE_NS / 1000000L, rc);
      pthread_cancel(node_thread_list[i]);
      clock_gettime(CLOCK_REALTIME, &deadline);
      deadline.tv_nsec += CANCEL_DEADLINE_NS;
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
 * Multi-node: delegates to arts_remote_shutdown() which shuts down all
 *   send and receive sockets.  Remote nodes detect the socket closure
 *   in their receive path and call arts_runtime_stop() themselves.
 * Single-node: directly calls arts_runtime_stop() to signal all threads.
 *
 * Called from:
 *   - arts_shutdown_epoch_fire() when the shutdown epoch completes.
 *   - User code via the arts_shutdown() public API.
 */
void arts_shutdown() {
  ARTS_INFO("arts_shutdown: rank_count=%u, rank_id=%u", arts_global_rank_count,
            arts_global_rank_id);
  /* Phase A entry — arts_enter_shutdown_state handles both the
   * multi-node broadcast + drain and the local worker-thread stop. */
  arts_enter_shutdown_state(/* initiator = */ true);
  (void)fflush(stdout);
}

_Noreturn void arts_abort(uint8_t error_code) {
  (void)fflush(stdout);
  (void)fflush(stderr);
  exit(error_code);
}

/*
 * wait_for_outbox_drain — Phase A helper.
 *
 * Poll arts_node_info.outbox_pending until it reaches zero or the
 * deadline elapses. Used by the initiator of a shutdown to guarantee
 * that the broadcast MSG_SHUTDOWN packets have been fully
 * handed off to the kernel TCP buffer before the initiator tears down
 * sockets during cleanup.
 */
static void wait_for_outbox_drain(unsigned int deadline_ms) {
  struct timespec start, now;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (;;) {
    unsigned int pending =
        arts_atomic_fetch_add(&arts_node_info.outbox_pending, 0U);
    if (pending == 0U) {
      return;
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    long elapsed_ms = (now.tv_sec - start.tv_sec) * 1000L +
                      (now.tv_nsec - start.tv_nsec) / 1000000L;
    if ((unsigned long)elapsed_ms >= (unsigned long)deadline_ms) {
      ARTS_INFO("shutdown drain timeout: %u messages still pending", pending);
      return;
    }
    /* Short backoff so we don't hog the CPU while the sender thread
     * drains the outbox. */
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 1000000L /* 1 ms */};
    nanosleep(&ts, NULL);
  }
}

/*
 * arts_enter_shutdown_state — the single internal entry point for
 * transitioning a rank into SHUTTING_DOWN state.
 *
 * Called from:
 *   - arts_shutdown() on the user EDT path (initiator = true)
 *   - the MSG_SHUTDOWN handler in dispatcher.c
 *     (initiator = false)
 *   - legacy EOF-detection paths in socket.c's recv logic
 *     (initiator = false) — defense in depth
 *
 * Idempotent: repeated calls after the first are no-ops. The CAS on
 * shutdown_state ensures exactly one caller performs the broadcast and
 * stop-workers step.
 */
void arts_enter_shutdown_state(bool initiator) {
  if (arts_atomic_cswap(&arts_node_info.shutdown_state, 0U, 1U) != 0U) {
    return; /* another thread / handler already started shutdown */
  }
  ARTS_INFO("arts_enter_shutdown_state: rank=%u initiator=%d",
            arts_global_rank_id, (int)initiator);
  if (initiator && arts_global_rank_count > 1) {
    /* Phase A.1: broadcast SHUTDOWN_MSG to every other rank. */
    arts_remote_send_shutdown_broadcast();
    /* Phase A.2: wait for our own outbox to drain so the broadcast
     * bytes are in the kernel TCP buffer before we tear down. */
    wait_for_outbox_drain(500U /* SHUTDOWN_DRAIN_MS */);
  }
  /* Phase A.3: stop worker threads. Network threads (senders,
   * receivers) remain alive so they can flush any in-flight traffic
   * and deliver any inbound SHUTDOWN_MSG that helps with defense in
   * depth. */
  arts_runtime_stop_workers();
}
