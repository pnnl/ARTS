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

#ifdef ARTS_WATCHDOG_ENABLED

#include "arts/runtime/watchdog.h"

#include "arts/arts_defs.h"
#include <stdbool.h>
#include <time.h>

#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/system/arts_print.h"
#include "arts/utils/deque.h"

/*
 * TLS-based watchdog state.  Each worker thread maintains its own last-tick
 * timestamp.  No inter-thread synchronization is needed — each thread only
 * reads/writes its own TLS variables.
 */
static uint64_t watchdog_timeout_ns = 0;
static ARTS_THREAD_LOCAL uint64_t watchdog_last_tick_ns = 0;
static ARTS_THREAD_LOCAL bool watchdog_triggered = false;

static inline uint64_t get_monotonic_ns(void) {
  struct timespec ts;
  (void)clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void arts_watchdog_init(uint64_t timeout_sec) {
  watchdog_timeout_ns = timeout_sec * 1000000000ULL;
  ARTS_INFO("Watchdog initialized: timeout=%lu sec", timeout_sec);
}

void arts_watchdog_tick(void) {
  watchdog_last_tick_ns = get_monotonic_ns();
  watchdog_triggered = false;
}

/*
 * arts_watchdog_check — Called from the scheduler idle path.
 *
 * If the timeout has elapsed since the last tick (EDT completion or
 * initial tick), dump diagnostic state.  The dump fires at most once
 * per stall episode (reset by arts_watchdog_tick).
 */
void arts_watchdog_check(void) {
  if (watchdog_timeout_ns == 0 || watchdog_triggered) {
    return;
  }

  /* First call: initialize the tick so we don't false-trigger */
  if (watchdog_last_tick_ns == 0) {
    watchdog_last_tick_ns = get_monotonic_ns();
    return;
  }

  uint64_t now = get_monotonic_ns();
  uint64_t elapsed = now - watchdog_last_tick_ns;

  if (elapsed >= watchdog_timeout_ns) {
    watchdog_triggered = true;

    uint64_t elapsed_sec = elapsed / 1000000000ULL;

    ARTS_WARN("===== WATCHDOG TIMEOUT =====");
    ARTS_WARN("Thread %u: no progress for %lu seconds",
              arts_thread_info.thread_id, elapsed_sec);
    ARTS_WARN("  thread_id=%u, pu_id=%u, role=%d, alive=%d",
              arts_thread_info.thread_id, arts_thread_info.pu_id,
              arts_thread_info.role, arts_thread_info.alive);

    /* Deque sizes (may be NULL for network threads) */
    if (arts_thread_info.my_deque) {
      ARTS_WARN("  deque_size=%u", arts_deque_size(arts_thread_info.my_deque));
    }

    /* Global state snapshot */
    ARTS_WARN("  shutdown_started=%u, total_threads=%u",
              arts_node_info.shutdown_started,
              arts_node_info.total_thread_count);

    /* Thread registration status */
    for (unsigned int i = 0; i < arts_node_info.total_thread_count; i++) {
      volatile bool *spin = arts_node_info.local_spin[i];
      ARTS_WARN("  thread[%u] local_spin=%p alive=%d", i, (void *)spin,
                spin ? *spin : -1);
    }

    ARTS_WARN("===== END WATCHDOG DUMP =====");
  }
}

#endif /* ARTS_WATCHDOG_ENABLED */
