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

/// @file socket_remote_alive_owner.c
/// @brief Single-owner-per-(rank,port) send-queue invariant for the non-atomic
///        remote_connection_alive[] flag.
///
/// Targets remote_connection_alive in libs/src/core/transport/socket.c: a plain
/// (non-atomic) bool array, read-then-written in arts_transport_connect.  Its
/// correctness depends ENTIRELY on each (rank,port) send queue being serviced
/// by exactly one sender thread (queues are partitioned disjointly across
/// senders in runtime.c).  If two sender threads ever drove connect for the
/// same idx, the read-then-write would race AND double-connect (replacing a
/// live fd) -- corrupting the send socket.  This single-owner-per-queue
/// invariant is load-bearing and untested.
///
/// Shaping: hammer concurrent outbound traffic from MANY EDTs that all target
/// the same small set of remote send queues at once (a tight fan-in: every
/// non-home rank repeatedly RW-acquires DBs homed on rank 0, and rank 0 fans
/// RW work out to every other rank).  This drives the lazy connect path on the
/// same (rank,port) queue from concurrent worker EDTs while the wire is busy.
/// If the single-owner partition were violated, the racy bool would let a
/// second connect close+replace a live send fd -> an in-flight message is sent
/// on a dead/duplicated socket -> a dropped transfer -> coherence hang
/// (ctest TIMEOUT) or a wrong reader value (FAIL, nonzero exit).  All values
/// correct + clean drain asserts the partition (one owner per queue) held under
/// concurrent connect pressure.  No in-test spin; ctest TIMEOUT reaps a hang.
///
/// Config-agnostic across protocols; ports>1 (2n_io) multiplies the queues
/// hammered.  On 1n there are no remote queues -> trivial pass.
/// exposes_runtime_bug: B-remote-alive-nonatomic (the latent race surfaces only
/// if the single-owner partition is ever broken).

#include "arts.h"

#include <stdint.h>

#define ROUNDS 12u
#define WIDTH 16u

void rw_bump(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

void final_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d != NULL && d[0] == expect) {
    arts_printf("  bumps ok (%u)\n", expect);
  } else {
    arts_printf("  FAIL: expected %u got %d\n", expect, d ? (int)d[0] : -1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== socket_remote_alive_owner ===\n");

  unsigned int nranks = arts_get_total_ranks();

  /* WIDTH distinct DBs homed on rank 0.  Each round drives a remote RW bump on
   * every DB from a rotating remote rank, hammering the SAME send queues from
   * many concurrent worker EDTs (the connect path / remote_connection_alive
   * read-then-write for those (rank,port) idx).  A per-DB RW chain per round
   * keeps the counts deterministic for the final check. */
  arts_guid_t dbs[WIDTH];
  for (unsigned int w = 0; w < WIDTH; w++) {
    void *ptr = NULL;
    dbs[w] = arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB,
                            ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
    if (ptr != NULL) {
      ((unsigned int *)ptr)[0] = 0u;
    }
    arts_db_release(dbs[w], DB_MODE_RW);
  }

  for (unsigned int round = 0; round < ROUNDS; round++) {
    arts_guid_t events[WIDTH];
    for (unsigned int w = 0; w < WIDTH; w++) {
      unsigned int target =
          (nranks > 1) ? (1u + ((round + w) % (nranks - 1))) : 0u;
      events[w] = arts_event_create(&ARTS_EVENT_HINT_FINISH);
      arts_guid_t e = arts_edt_create(
          rw_bump, 0, NULL, 1,
          &(arts_edt_hint_t){.rank = target, .finish_event = events[w]});
      arts_add_dependence(dbs[w], e, 0, DB_MODE_RW);
    }
    /* Wait for the whole concurrent batch before issuing the next round so the
     * per-DB bump count stays deterministic (one bump per DB per round). */
    for (unsigned int w = 0; w < WIDTH; w++) {
      arts_event_wait(events[w]);
    }
  }

  /* Verify every DB was bumped exactly ROUNDS times -> no transfer dropped by a
   * raced/double connect. */
  for (unsigned int w = 0; w < WIDTH; w++) {
    uint64_t expect = (uint64_t)ROUNDS;
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t c =
        arts_edt_create(final_check, 1, &expect, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(dbs[w], c, 0, DB_MODE_RO);
    arts_event_wait(fe);
  }

  arts_printf("PASS: socket_remote_alive_owner %u dbs x %u rounds\n", WIDTH,
              ROUNDS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
