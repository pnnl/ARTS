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
unsigned int num_elements = 0;
unsigned int block_size = 0;
unsigned int stride = 0;

void shut_down_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  bool pass = true;
  if (arts_is_guid_local(depv[0].guid)) {
    unsigned int *data = (unsigned int *)depv[0].ptr;
    for (unsigned int i = 0; i < num_elements; i++) {
      if (data[i] != i) {
        ARTS_PRINTF("FAIL %u vs %u\n", i, data[i]);
        pass = false;
      }
    }
  }
  if (pass) {
    ARTS_PRINTF("CHECK\n");
}
  arts_shutdown();
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  db_guid = arts_reserve_guid_route(ARTS_DB_PIN, 0);
  shutdown_guid = arts_reserve_guid_route(ARTS_EDT, 0);
  num_elements = atoi(argv[1]);
  block_size = num_elements / arts_get_total_nodes();
  stride = atoi(argv[2]);
  if (!node_id) {
    ARTS_PRINTF("num_elements: %u block_size: %u stride: %u\n", num_elements, block_size,
           stride);
}
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {
  if (block_size % stride) {
    if (!node_id && !worker_id) {
      arts_shutdown();
    }
    return;
  }

  if (!worker_id) {
    unsigned int deps = block_size / stride;
    for (unsigned int j = 0; j < deps; j++) {
      unsigned int *data =
          (unsigned int *)arts_malloc(sizeof(unsigned int) * stride);
      for (unsigned int i = 0; i < stride; i++) {
        data[i] = node_id * block_size + j * stride + i;
}
      //            ARTS_PRINTF("PUT: index: %u slot: %u\n", node_id*block_size +
      //            j*stride, node_id*deps + j);
      arts_put_in_db(data, shutdown_guid, db_guid, (node_id * deps) + j,
                  sizeof(unsigned int) * (node_id * block_size + j * stride),
                  sizeof(unsigned int) * stride);
      arts_free(data);
    }

    if (!node_id) {
      arts_db_create_with_guid(db_guid, sizeof(unsigned int) * num_elements);
      arts_edt_create_with_guid(shut_down_edt, shutdown_guid, 0, NULL,
                            num_elements / stride);
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
