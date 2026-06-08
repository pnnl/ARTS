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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "arts.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/transport/dispatcher.h"
#include "arts/utils/atomics.h"

/*
 * arts_shutdown — Initiate global shutdown of the ARTS runtime.
 *
 * Multi-node: delegates to arts_remote_shutdown() which shuts down all
 *   send and receive sockets.  Remote nodes detect the socket closure
 *   in their receive path and call arts_runtime_stop() themselves.
 * Single-node: directly calls arts_runtime_stop() to signal all threads.
 *
 * Called from:
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
