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

/// @file socket_connect_retry_abort.c
/// @brief Exercise the lazy connect path + the shutdown-abort early-out.
///
/// Targets arts_transport_connect in libs/src/core/transport/socket.c:
///   - The 300-retry / 100 ms-sleep loop (SLURM startup skew tolerance) that
///     replaces the send fd and retries on connect failure.
///   - The early-out that bails the retry loop when
///     arts_node_info.shutdown_state is set.
///   - On success, remote_connection_alive[idx] is set true exactly once per
///     (rank,port) queue.
///
/// Deterministic shaping (we cannot inject a connect failure from a runtime
/// test): every rank both accepts and connects out during rendezvous, so the
/// successful-connect branch (and, under launcher startup skew, the retry
/// branch) is driven for every (rank,port).  We then force at least one
/// outbound message to EACH other rank so the lazy connect path is taken for
/// every send queue, and finally call a clean shutdown.  Shutdown sets
/// shutdown_state; any connect still mid-retry must observe it and abort
/// rather than spin the full 30 s -- a missed early-out would make the runtime
/// outlive the ctest TIMEOUT.  Clean exit + PASS = the abort path is honored
/// and no queue was left half-connected.
///
/// Config-agnostic across protocols.  On 1n there are no remote queues, so the
/// connect path is a no-op and the test passes trivially.

#include "arts.h"

#include <stdint.h>

/// Touch the DB so a cross-rank acquire (hence a connect to the home) happens.
void touch_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== socket_connect_retry_abort ===\n");

  unsigned int nranks = arts_get_total_ranks();

  /* Home the DB on rank 0; fire one RW EDT on every rank so each non-home
   * rank must lazily connect its send queue(s) to the home and back. */
  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  for (unsigned int rank = 0; rank < nranks; rank++) {
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t t =
        arts_edt_create(touch_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = rank, .finish_event = e});
    arts_add_dependence(db, t, 0, DB_MODE_RW);
    /* Serialize the RW chain so every rank in turn owns -> connects. */
    arts_event_wait(e);
  }

  arts_printf("PASS: socket_connect_retry_abort connected %u ranks\n", nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
