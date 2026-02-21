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
#include <string.h>

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
  arts_signal_edt(shutdown_guid, id, db_dest_guid, ARTS_MODE_EW);
}

void getter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)depc;
  unsigned int id = paramv[0];
  unsigned int *source = (unsigned int *)depv[0].ptr;
  unsigned int *buffer = &source[(size_t)id * block_size];
  arts_printf("Getter: %u\n", id);
  // This one actually sends to a remote node... yea for testing!
  unsigned int buf_size = sizeof(unsigned int) * (size_t)block_size;
  void *buf_copy = malloc(buf_size);
  memcpy(buf_copy, buffer, buf_size);
  arts_guid_t am =
      arts_edt_create(setter, paramc, paramv, 2,
                      &(arts_hint_t){.route = arts_get_total_nodes() - 1});
  arts_signal_edt_ptr(am, 0, buf_copy, buf_size);
  arts_signal_edt(am, 1, db_dest_guid, ARTS_MODE_EW);
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

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  block_size = strtol(argv[1], NULL, 10);
  num_elements = block_size * arts_get_total_nodes();
  db_dest_guid = arts_guid_reserve(ARTS_DB_LOCAL, arts_get_total_nodes() - 1);
  shutdown_guid = arts_guid_reserve(ARTS_EDT, arts_get_total_nodes() - 1);

  unsigned int node_id = arts_get_current_node();
  uint64_t id = node_id;
  unsigned int *data =
      (unsigned int *)malloc(sizeof(unsigned int) * num_elements);
  for (unsigned int i = 0; i < num_elements; i++) {
    data[i] = i;
  }
  // This is kinda dumb since it is sending to itself, but hey lets check
  // it...
  unsigned int data_size = sizeof(unsigned int) * num_elements;
  void *data_copy = malloc(data_size);
  memcpy(data_copy, data, data_size);
  free(data);
  arts_guid_t getter_edt =
      arts_edt_create(getter, 1, &id, 1, &(arts_hint_t){.route = node_id});
  arts_signal_edt_ptr(getter_edt, 0, data_copy, data_size);

  arts_edt_create_with_guid(shut_down_edt, shutdown_guid, 0, NULL,
                            arts_get_total_nodes());

  if (node_id == arts_get_total_nodes() - 1) {
    arts_db_create_with_guid(db_dest_guid, sizeof(unsigned int) * num_elements,
                             NULL);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
