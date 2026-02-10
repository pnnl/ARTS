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

unsigned int elements_per_block = 0;
unsigned int blocks = 0;
unsigned int num_add = 0;
arts_array_db_t *array = NULL;
arts_guid_t array_guid = NULL_GUID;

void end(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
         arts_edt_dep_t depv[]) {
  for (unsigned int i = 0; i < depc - 1; i++) {
    unsigned int data = depv[i].guid;
    ARTS_PRINTF("updates: %u\n", data);
  }
  arts_shutdown();
}

// Created by the epoch_end via gather will signal end
void check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  for (unsigned int i = 0; i < blocks; i++) {
    unsigned int *data = (unsigned int *)depv[i].ptr;
    for (unsigned int j = 0; j < elements_per_block; j++) {
      ARTS_PRINTF("i: %u j: %u %u\n", i, j, data[j]);
    }
  }
  arts_signal_edt_value(paramv[0], num_add * elements_per_block * blocks, 0);
}

// This is run at the end of the epoch
void epoch_end(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  unsigned int num_in_epoch = depv[0].guid;
  ARTS_PRINTF("%u in Epoch\n", num_in_epoch);
  arts_gather_array_db(array, check, 0, 1, paramv, 0);
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  elements_per_block = atoi(argv[1]);
  blocks = arts_get_total_nodes();
  num_add = atoi(argv[2]);
  array_guid = arts_reserve_guid_route(ARTS_DB_PIN, 0);
  if (!node_id) {
    ARTS_PRINTF("ElementsPerBlock: %u Blocks: %u\n", elements_per_block, blocks);
}
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {

  if (!worker_id && !node_id) {
    // The end will get all the updates and a signal from the gather
    arts_guid_t end_guid =
        arts_edt_create(end, 0, 0, NULL, (num_add * elements_per_block * blocks) + 1);

    arts_guid_t end_epoch_guid =
        arts_edt_create(epoch_end, 0, 1, (uint64_t *)&end_guid, 1);
    arts_initialize_and_start_epoch(end_epoch_guid, 0);

    array = arts_new_array_db_with_guid(array_guid, sizeof(unsigned int),
                                   elements_per_block * blocks);

    for (unsigned int j = 0; j < num_add; j++) {
      for (unsigned int i = 0; i < elements_per_block * blocks; i++) {
        ARTS_PRINTF("i: %u Slot:%u edt: %lu\n", i,
               (j * elements_per_block * blocks) + i, end_guid);
        arts_atomic_add_in_array_db(array, i, 1, end_guid,
                               (j * elements_per_block * blocks) + i);
      }
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
