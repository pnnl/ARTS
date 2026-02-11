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
#include "arts.h"

arts_guid_t guid[4];
arts_guid_t shutdown_guid = NULL_GUID;

void check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  for (unsigned int i = 0; i < depc; i++) {
    for (unsigned int j = 0; j < 4; j++) {
      if (guid[j] == depv[i].guid) {
        unsigned int *data = (unsigned int *)depv[i].ptr;
        arts_printf("j: %u %lu: %u from %u\n", j, depv[i].guid, *data,
               arts_guid_get_rank(depv[i].guid));
      }
    }
  }
  arts_signal_edt_value(shutdown_guid, -1, 0);
}

void shut_down_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_shutdown();
}

void node_setup(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int node_id = (unsigned int)paramv[0];

  if (node_id == 0) {
    // Local to local
    unsigned int *a_ptr =
        (unsigned int *)arts_db_create_with_guid(guid[0], sizeof(unsigned int), NULL);
    *a_ptr = 1;
    arts_db_move(guid[0], 0);

    // Local to remote
    unsigned int *a_ptr2 =
        (unsigned int *)arts_db_create_with_guid(guid[1], sizeof(unsigned int), NULL);
    *a_ptr2 = 2;
    arts_db_move(guid[1], 1);

    // Remote to local
    arts_db_move(guid[2], 0);

    // Remote to remote
    arts_db_move(guid[3], 2);
  }

  if (node_id == 1) {
    unsigned int *b_ptr =
        (unsigned int *)arts_db_create_with_guid(guid[2], sizeof(unsigned int), NULL);
    *b_ptr = 3;

    unsigned int *c_ptr =
        (unsigned int *)arts_db_create_with_guid(guid[3], sizeof(unsigned int), NULL);
    *c_ptr = 4;
  }

  if (node_id == 0) {
    arts_guid_t edt_guid = arts_edt_create(check, 0, NULL, 2, &(arts_hint_t){.route = node_id});
    arts_signal_edt(edt_guid, 0, guid[0]);
    arts_signal_edt(edt_guid, 1, guid[2]);

    arts_edt_create_with_guid(shut_down_edt, shutdown_guid, 0, NULL, 3);
  }

  if (node_id == 1) {
    arts_guid_t edt_guid = arts_edt_create(check, 0, NULL, 1, &(arts_hint_t){.route = node_id});
    arts_signal_edt(edt_guid, 0, guid[1]);
  }

  if (node_id == 2) {
    arts_guid_t edt_guid = arts_edt_create(check, 0, NULL, 1, &(arts_hint_t){.route = node_id});
    arts_signal_edt(edt_guid, 0, guid[3]);
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  guid[0] = arts_guid_reserve(ARTS_DB_ONCE_LOCAL, 0);
  guid[1] = arts_guid_reserve(ARTS_DB_ONCE_LOCAL, 0);
  guid[2] = arts_guid_reserve(ARTS_DB_ONCE_LOCAL, 1);
  guid[3] = arts_guid_reserve(ARTS_DB_ONCE_LOCAL, 1);
  shutdown_guid = arts_guid_reserve(ARTS_EDT, 0);

  for (unsigned int n = 0; n < arts_get_total_nodes(); n++) {
    uint64_t args = n;
    arts_edt_create(node_setup, 1, &args, 0, &(arts_hint_t){.route = n});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
