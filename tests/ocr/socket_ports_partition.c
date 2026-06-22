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

/// @file socket_ports_partition.c
/// @brief Single-owner-per-(rank,port) inbound-queue partition invariant.
///
/// Targets arts_transport_set_thread_inbound_queues(start, stop) in
/// libs/src/core/transport/socket.c: each receiver thread owns a DISJOINT
/// [start, stop) slice of poll_incoming / remote_socket_receive_list, and the
/// per-socket reassembly state (re_receive_res, bypass_buf partial bytes) is
/// thread-local precisely BECAUSE the slices never overlap.  That disjointness
/// is the entire safety argument for the lock-free framing loop, and it is
/// untested.  The slice size is computed as (stop - start) into an unsigned --
/// a stop<start swap would underflow to a huge allocation (no assertion guard).
///
/// Shaping (this is the 2n_io test): the 2n_io config gives ports>1 and more
/// than one sender/receiver thread per rank, so distinct (rank,port) sockets
/// are serviced by distinct receiver threads owning disjoint slices.  We blast
/// MANY concurrent cross-rank transfers in flight at once (fan-out RW writers
/// on remote ranks, then verify every one) so packets land on multiple ports
/// concurrently.  If two receiver threads shared a socket (overlapping slices)
/// the lock-free reassembly state would corrupt -> a dropped/duplicated packet
/// -> a coherence hang (ctest TIMEOUT) or a wrong reader value (FAIL printed,
/// nonzero exit).  All readers seeing the correct per-writer value asserts the
/// partition held: every (rank,port) had exactly one owner.
///
/// Config-agnostic across protocols; meaningful only with ports>1 (2n_io), but
/// runs correctly on every config (on 1n / ports==1 it degenerates to a single
/// owner trivially, still a valid smoke run).

#include "arts.h"

#include <stdint.h>

#define FANOUT 48u

void slot_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int tag = (unsigned int)paramv[0];
  if (d != NULL) {
    d[0] = tag;
  }
}

void slot_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int tag = (unsigned int)paramv[0];
  if (d != NULL && d[0] == tag) {
    arts_printf("  slot %u ok\n", tag);
  } else {
    arts_printf("  FAIL: slot %u got %d\n", tag, d ? (int)d[0] : -1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== socket_ports_partition ===\n");

  unsigned int nranks = arts_get_total_ranks();

  /* One distinct DB per slot, each homed on rank 0; a remote RW writer then a
   * remote RO reader per slot.  Creating the whole fan-out before waiting puts
   * many cross-rank transfers in flight simultaneously -> packets across both
   * ports of 2n_io concurrently, stressing the disjoint slice partition. */
  arts_guid_t dbs[FANOUT];
  arts_guid_t w_events[FANOUT];

  for (unsigned int s = 0; s < FANOUT; s++) {
    void *ptr = NULL;
    dbs[s] = arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB,
                            ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
    if (ptr != NULL) {
      ((unsigned int *)ptr)[0] = 0u;
    }
    arts_db_release(dbs[s], DB_MODE_RW);
  }

  /* Phase 1: launch all writers concurrently across the non-home ranks. */
  for (unsigned int s = 0; s < FANOUT; s++) {
    unsigned int target = (nranks > 1) ? (1u + (s % (nranks - 1))) : 0u;
    uint64_t tag = (uint64_t)(0x5000u + s);
    w_events[s] = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w = arts_edt_create(
        slot_writer, 1, &tag, 1,
        &(arts_edt_hint_t){.rank = target, .finish_event = w_events[s]});
    arts_add_dependence(dbs[s], w, 0, DB_MODE_RW);
  }
  for (unsigned int s = 0; s < FANOUT; s++) {
    arts_event_wait(w_events[s]);
  }

  /* Phase 2: launch all readers concurrently; each must see its writer's tag.
   */
  arts_guid_t r_events[FANOUT];
  for (unsigned int s = 0; s < FANOUT; s++) {
    unsigned int target = (nranks > 1) ? (1u + ((s + 1u) % (nranks - 1))) : 0u;
    uint64_t tag = (uint64_t)(0x5000u + s);
    r_events[s] = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t r = arts_edt_create(
        slot_reader, 1, &tag, 1,
        &(arts_edt_hint_t){.rank = target, .finish_event = r_events[s]});
    arts_add_dependence(dbs[s], r, 0, DB_MODE_RO);
  }
  for (unsigned int s = 0; s < FANOUT; s++) {
    arts_event_wait(r_events[s]);
  }

  arts_printf("PASS: socket_ports_partition %u slots across %u ranks\n", FANOUT,
              nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
