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

unsigned int num_writes = 0;
arts_guid_t db_guid;
arts_guid_t *write_guids;

void write_test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  unsigned int index = paramv[0];
  unsigned int *array = (unsigned int *)depv[0].ptr;
  //    if(array)
  //    {
  for (unsigned int i = index; i < num_writes; i++) {
    array[i] = index;
}
  //    }
  if (paramc > 1) {
    ARTS_PRINTF("-----------------SIGNALLING NEXT %u\n", index);
    arts_signal_edt_value((arts_guid_t)paramv[1], -1, 0);
  } else {
    for (unsigned int i = 0; i < num_writes; i++) {
      ARTS_PRINTF("i: %u %u\n", i, array[i]);
    }
    arts_shutdown();
  }
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  db_guid = arts_reserve_guid_route(ARTS_DB_READ, 0);

  num_writes = atoi(argv[1]);
  write_guids = (arts_guid_t *)arts_malloc(sizeof(arts_guid_t) * num_writes);
  for (unsigned int i = 0; i < num_writes; i++) {
    write_guids[i] = arts_reserve_guid_route(ARTS_EDT, i % arts_get_total_nodes());
}
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {
  if (!worker_id) {
    if (!node_id) {
      unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(
          db_guid, sizeof(unsigned int) * num_writes);
      for (unsigned int i = 0; i < num_writes; i++) {
        ptr[i] = 0;
}
    }

    uint64_t args[2];
    for (uint64_t i = 0; i < num_writes; i++) {
      if (arts_is_guid_local(write_guids[i])) {
        args[0] = i;

        if (i < num_writes - 1) {
          args[1] = write_guids[i + 1];
          arts_edt_create_with_guid(write_test, write_guids[i], 2, args, 2);
        } else {
          arts_edt_create_with_guid(write_test, write_guids[i], 1, args, 2);
        }
        arts_signal_edt(write_guids[i], 0, arts_guid_cast(db_guid, ARTS_DB_WRITE));
      }
    }
    if (!node_id) {
      arts_signal_edt_value(write_guids[0], -1, 0);
}
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
