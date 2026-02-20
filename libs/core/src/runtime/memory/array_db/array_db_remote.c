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

#include "arts/runtime/memory/array_db.h"

#include "arts/gas/route_table.h"
#include "arts/network/remote_protocol.h"
#include "arts/runtime/globals.h"

void arts_remote_atomic_add_in_array_db(unsigned int rank, arts_guid_t db_guid,
                                  unsigned int index, unsigned int to_add,
                                  arts_guid_t edt_guid, unsigned int slot,
                                  arts_guid_t epoch_guid) {
  struct arts_remote_atomic_add_in_array_db_packet_s packet;
  packet.db_guid = db_guid;
  packet.edt_guid = edt_guid;
  packet.epoch_guid = epoch_guid;
  packet.slot = slot;
  packet.index = index;
  packet.to_add = to_add;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                       ARTS_ATOMIC_ADD_ARRAYDB_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_atomic_add_in_array_db(void *pack) {
  struct arts_remote_atomic_add_in_array_db_packet_s *packet =
      (struct arts_remote_atomic_add_in_array_db_packet_s *)pack;
  internal_atomic_add_in_array_db(packet->db_guid, packet->index, packet->to_add,
                             packet->edt_guid, packet->slot, packet->epoch_guid);
}

void arts_remote_atomic_compare_and_swap_in_array_db(
    unsigned int rank, arts_guid_t db_guid, unsigned int index,
    unsigned int old_value, unsigned int new_value, arts_guid_t edt_guid,
    unsigned int slot, arts_guid_t epoch_guid) {
  struct arts_remote_atomic_compare_and_swap_in_array_db_packet_s packet;
  packet.db_guid = db_guid;
  packet.edt_guid = edt_guid;
  packet.epoch_guid = epoch_guid;
  packet.slot = slot;
  packet.index = index;
  packet.old_value = old_value;
  packet.new_value = new_value;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                       ARTS_ATOMIC_CAS_ARRAYDB_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_atomic_compare_and_swap_in_array_db(void *pack) {
  struct arts_remote_atomic_compare_and_swap_in_array_db_packet_s *packet =
      (struct arts_remote_atomic_compare_and_swap_in_array_db_packet_s *)pack;
  internal_atomic_compare_and_swap_in_array_db(
      packet->db_guid, packet->index, packet->old_value, packet->new_value,
      packet->edt_guid, packet->slot, packet->epoch_guid);
}
