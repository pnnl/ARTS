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

/// @file socket_shutdown_index_domains.c
/// @brief Off-by-one consistency between socket setup and shutdown index
///        domains over the full receive/send socket mesh.
///
/// Targets the two functions in libs/src/core/transport/socket.c that walk the
/// per-(rank,port) socket tables with DIFFERENT meanings of `count`:
///   - arts_transport_setup_incoming: sizes remote_socket_receive_list as
///     (count+1)*ports where count = table_length-1, and fills accepts into
///     [0, count*ports).
///   - arts_socket_shutdown: SHUT_RDs the receive list over (count-1)*ports and
///     SHUT_WRs the send list over count*ports (skipping self-rank), where
///     count = table_length.
/// The receive-list size (table_length*ports) and shutdown's receive bound
/// ((table_length-1)*ports) line up only by careful arithmetic -- a prime
/// off-by-one regression surface (an over-run shuts down the +1 listener slot
/// or walks past the array; an under-run leaves a peer socket un-FIN'd so the
/// peer's recv never returns 0 and its receiver thread never stops).
///
/// Shaping: establish the FULL mesh by routing real cross-rank traffic to and
/// from every rank (every rank both sends to and receives from rank 0 and a
/// neighbor), so every receive/send socket index is live before shutdown.  Then
/// call a clean shutdown.  If shutdown's index domain is off by one, either a
/// peer is never FIN'd -> its receiver thread hangs -> ctest TIMEOUT, or it
/// touches an out-of-domain fd/array slot (ASan/UBSan build flags it).  Clean
/// exit + PASS asserts the two index domains are mutually consistent.
///
/// Config-agnostic across protocols; ports>1 (2n_io) widens each index domain.
/// On 1n there is exactly one (self) entry and the loops are degenerate but
/// must still not under/over-run.

#include "arts.h"

#include <stdint.h>

void poke_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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

  arts_printf("=== socket_shutdown_index_domains ===\n");

  unsigned int nranks = arts_get_total_ranks();

  /* One DB homed on each rank in turn; every other rank does an RW acquire,
   * so receive AND send sockets light up for every (peer,port) pair.  This
   * makes the full mesh live -> shutdown must walk exactly the live domain. */
  for (unsigned int home = 0; home < nranks; home++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = home});
    if (ptr != NULL) {
      ((unsigned int *)ptr)[0] = 0u;
    }
    arts_db_release(db, DB_MODE_RW);

    for (unsigned int rank = 0; rank < nranks; rank++) {
      arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
      arts_guid_t p =
          arts_edt_create(poke_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = rank, .finish_event = e});
      arts_add_dependence(db, p, 0, DB_MODE_RW);
      arts_event_wait(e);
    }
  }

  arts_printf("PASS: socket_shutdown_index_domains mesh of %u ranks\n", nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
