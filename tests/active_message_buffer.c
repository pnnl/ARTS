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

arts_guid_t db_dest_guid = NULL_GUID;
arts_guid_t shutdown_guid = NULL_GUID;
unsigned int num_elements = 0;
unsigned int block_size = 0;

void setter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {

(void)depc;

(void)paramc;

  unsigned int id = paramv[0];
  unsigned int *buffer = (unsigned int *)depv[0].ptr;
  unsigned int *dest = (unsigned int *)depv[1].ptr;
  for (unsigned int i = 0; i < block_size; i++) {
    dest[(id * block_size) + i] = buffer[i];
  }
  arts_printf("Setter: %u\n", id);
  arts_signal_edt(shutdown_guid, id, db_dest_guid);
}

void getter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)depc;
  unsigned int id = paramv[0];
  unsigned int *source = (unsigned int *)depv[0].ptr;
  unsigned int *buffer = &source[(size_t)id * block_size];
  arts_printf("Getter: %u\n", id);
  // This one actually sends to a remote node... yea for testing!
  arts_guid_t am = arts_active_message_with_buffer(setter, arts_get_total_nodes() - 1,
                                              paramc, paramv, 1, buffer,
                                              sizeof(unsigned int) * (size_t)block_size);
  arts_signal_edt(am, 1, db_dest_guid);
}

void shut_down_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  bool pass = true;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  for (unsigned int i = 0; i < num_elements; i++) {
    if (data[i] != i) {
      arts_printf("I: %u vs %u\n", i, data[i]);
      pass = false;
    }
  }

  if (pass) {
    arts_printf("CHECK\n");
}
  arts_shutdown();
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  (void)argc;
  (void)node_id;
  block_size = strtol(argv[1], NULL, 10);
  num_elements = block_size * arts_get_total_nodes();
  db_dest_guid = arts_reserve_guid_route(ARTS_DB_PIN, arts_get_total_nodes() - 1);
  shutdown_guid = arts_reserve_guid_route(ARTS_EDT, arts_get_total_nodes() - 1);
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {
  (void)argc;
  (void)argv;
  if (!worker_id) {
    uint64_t id = node_id;
    unsigned int *data =
        (unsigned int *)arts_malloc(sizeof(unsigned int) * num_elements);
    for (unsigned int i = 0; i < num_elements; i++) {
      data[i] = i;
    }
    // This is kinda dumb since it is sending to itself, but hey lets check
    // it...
    arts_active_message_with_buffer(getter, node_id, 1, &id, 0, data,
                                sizeof(unsigned int) * num_elements);

    if (!node_id) {
      arts_edt_create_with_guid(shut_down_edt, shutdown_guid, 0, NULL,
                            arts_get_total_nodes());
}

    if (node_id == arts_get_total_nodes() - 1) {
      arts_db_create_with_guid(db_dest_guid, sizeof(unsigned int) * num_elements);
}
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
