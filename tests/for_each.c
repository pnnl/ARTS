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

unsigned int elems_per_node = 4;
arts_array_db_t *array = NULL;

void shutdown(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  arts_printf("Depc: %u\n", depc);
  for (unsigned int i = 0; i < depc; i++) {
    unsigned int *data = (unsigned int *)depv[i].ptr;
    for (unsigned int j = 0; j < elems_per_node; j++) {
      arts_printf("%u: %u\n", (i * elems_per_node) + j, data[j]);
    }
  }
  arts_shutdown();
}

void check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_gather_array_db(array, shutdown, 0, 0, NULL, 0);
}

void edt_func(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  unsigned int index = paramv[0];
  arts_guid_t check_guid = (arts_guid_t)paramv[1];
  unsigned int *value = (unsigned int *)depv[0].ptr;
  *value = index;
  arts_printf("%u:  %u %p\n", index, *value, value);
  arts_signal_edt_value(check_guid, 0, 0);
}

void init_per_node(unsigned int node_id, int argc, char **argv) {}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {
  (void)argc;
  (void)argv;
  if (!node_id && !worker_id) {
    arts_guid_t check_guid =
        arts_edt_create(check, 0, 0, NULL, elems_per_node * arts_get_total_nodes());
    arts_guid_t guid = arts_new_array_db(&array, sizeof(unsigned int),
                                     elems_per_node * arts_get_total_nodes());
    arts_for_each_in_array_db_at_data(array, 1, edt_func, 1, (uint64_t *)&check_guid);
    //        arts_for_each_in_array_db(array, edt_func, 1, &check_guid);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
