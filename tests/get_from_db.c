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
#include <stdlib.h>

#include "arts.h"

arts_guid_t db_guid = NULL_GUID;
arts_guid_t shutdown_guid = NULL_GUID;
arts_guid_t edt_guid_fixed = NULL_GUID;
unsigned int num_elements = 0;
unsigned int block_size = 0;
unsigned int stride = 0;

void getter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  unsigned int sum = 0;
  for (unsigned int i = 0; i < depc; i++) {
    unsigned int *data = (unsigned int *)depv[i].ptr;
    for (unsigned int j = 0; j < stride; j++) {
      sum += data[j];
    }
  }
  arts_signal_edt_value(shutdown_guid, arts_get_current_node(), sum);
}

void creater(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  unsigned int *data =
      (unsigned int *)arts_malloc(sizeof(unsigned int) * num_elements);
  for (unsigned int i = 0; i < num_elements; i++) {
    data[i] = i;
  }
  arts_db_create_with_guid_and_data(db_guid, data, sizeof(unsigned int) * num_elements);
  arts_edt_create_with_guid(getter, edt_guid_fixed, 0, NULL, block_size / stride);
}

void shut_down_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  unsigned int sum = 0;
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    sum += (unsigned int)depv[i].guid;
}

  unsigned int compare = 0;
  for (unsigned int i = 0; i < num_elements; i++) {
    compare += i;
}

  if (sum == compare) {
    arts_printf("CHECK SUM: %u vs %u\n", sum, compare);
  } else {
    arts_printf("FAIL SUM: %u vs %u\n", sum, compare);
}
  arts_shutdown();
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  (void)argc;
  db_guid = arts_reserve_guid_route(ARTS_DB_PIN, 0);
  shutdown_guid = arts_reserve_guid_route(ARTS_EDT, 0);
  edt_guid_fixed = arts_reserve_guid_route(ARTS_EDT, 0);
  num_elements = strtol(argv[1], NULL, 10);
  block_size = num_elements / arts_get_total_nodes();
  stride = strtol(argv[2], NULL, 10);
  if (!node_id) {
    arts_printf("num_elements: %u block_size: %u stride: %u\n", num_elements, block_size,
           stride);
}
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {
  (void)argc;
  (void)argv;
  if (block_size % stride) {
    if (!node_id && !worker_id) {
      arts_shutdown();
    }
    return;
  }

  if (!worker_id) {
    if (!node_id) {
      arts_edt_create(creater, 0, 0, NULL, 0);
      arts_edt_create_with_guid(shut_down_edt, shutdown_guid, 0, NULL,
                            arts_get_total_nodes());
    }

    unsigned int deps = block_size / stride;
    if (!node_id) {
      for (unsigned int j = 0; j < deps; j++) {
        arts_get_from_db(edt_guid_fixed, db_guid, j,
                      sizeof(unsigned int) * (node_id * block_size + j * stride),
                      sizeof(unsigned int) * stride);
      }
    } else {
      arts_guid_t edt_guid = arts_edt_create(getter, node_id, 0, NULL, deps);
      for (unsigned int j = 0; j < deps; j++) {
        arts_get_from_db(edt_guid, db_guid, j,
                      sizeof(unsigned int) * (node_id * block_size + j * stride),
                      sizeof(unsigned int) * stride);
      }
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
