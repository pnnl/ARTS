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
#include <stdio.h>

#include "arts.h"
#include <stdlib.h>

arts_guid_t shutdown_guid;
arts_guid_t *guids;

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_shutdown();
}

void acquire_test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  for (unsigned int i = 0; i < depc; i++) {
    unsigned int *num = (unsigned int *)depv[i].ptr;
    printf("%u %u i: %u %u\n", arts_get_current_node(), arts_get_current_worker(), i,
           *num);
  }
  arts_signal_edt_value(shutdown_guid, 0, 0);
}

void node_setup(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int node_id = (unsigned int)paramv[0];
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    if (arts_guid_is_local(guids[i])) {
      unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(
          guids[i], sizeof(unsigned int), NULL);
      *ptr = i;
      arts_printf("Created i: %u guid: %ld\n", i, guids[i]);
    }
  }

  unsigned int num_workers = arts_get_total_workers();
  for (unsigned int w = 0; w < num_workers; w++) {
    arts_guid_t edt_guid =
        arts_edt_create(acquire_test, 0, NULL, arts_get_total_nodes(), &(arts_hint_t){.route = node_id});
    for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
      arts_signal_edt(edt_guid, i, guids[i]);
    }
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  guids = (arts_guid_t *)malloc(sizeof(arts_guid_t) * arts_get_total_nodes());
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    guids[i] = arts_guid_reserve(ARTS_DB, i);
    arts_printf("i: %u guid: %ld\n", i, guids[i]);
  }
  shutdown_guid = arts_guid_reserve(ARTS_EDT, 0);

  arts_edt_create_with_guid(shutdown_edt, shutdown_guid, 0, NULL,
                        arts_get_total_nodes() * arts_get_total_workers());

  for (unsigned int n = 0; n < arts_get_total_nodes(); n++) {
    uint64_t args = n;
    arts_edt_create(node_setup, 1, &args, 0, &(arts_hint_t){.route = n});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
