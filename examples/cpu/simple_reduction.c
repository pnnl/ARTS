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

unsigned int num_dbs = 0;
arts_guid_t reduction_guid = NULL_GUID;

void reduction(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  uint64_t total = 0;
  for (unsigned int i = 0; i < depc; i++) {
    int *db_ptr = depv[i].ptr;
    total += db_ptr[0];
  }
  arts_signal_edt_value((arts_guid_t)paramv[0], 0, total);
}

void shut_down(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  arts_printf("Result %lu\n", depv[0].guid);
  arts_shutdown();
}

void node_setup(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int node_id = (unsigned int)paramv[0];
  int *ptr;
  arts_guid_t db_guid =
      arts_db_create((void **)&ptr, sizeof(unsigned int), NULL);
  *ptr = (int)node_id;
  arts_signal_edt(reduction_guid, node_id, db_guid);
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  num_dbs = arts_get_total_nodes();
  reduction_guid = arts_guid_reserve(ARTS_EDT, 0);

  arts_guid_t guid = arts_edt_create(shut_down, 0, NULL, 1, &(arts_hint_t){.route = 0});
  arts_edt_create_with_guid(reduction, reduction_guid, 1, (uint64_t *)&guid,
                        num_dbs);

  for (unsigned int n = 0; n < num_dbs; n++) {
    uint64_t args = n;
    arts_edt_create(node_setup, 1, &args, 0, &(arts_hint_t){.route = n});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
