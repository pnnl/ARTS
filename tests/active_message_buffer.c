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

/*
 * All cross-node values are passed via paramv (no shared globals).
 *
 * getter/setter paramv layout (paramc=4):
 *   [0] = id (node index)
 *   [1] = shutdown_guid
 *   [2] = block_size
 *   [3] = db_dest_guid
 *
 * getter depv (depc=1):
 *   [0] = source data (PTR mode, from arts_signal_edt_ptr)
 *
 * setter depv (depc=2):
 *   [0] = buffer data (PTR mode)
 *   [1] = db_dest_guid (EW mode)
 *
 * shut_down_edt paramv (paramc=1):
 *   [0] = num_elements
 * shut_down_edt depv (depc=total_nodes):
 *   [0..N-1] = db_dest_guid (RO mode)
 */

void setter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;

  unsigned int id = (unsigned int)paramv[0];
  arts_guid_t sd_guid = (arts_guid_t)paramv[1];
  unsigned int bs = (unsigned int)paramv[2];
  arts_guid_t dest_guid = (arts_guid_t)paramv[3];

  unsigned int *buffer = (unsigned int *)depv[0].ptr;
  unsigned int *dest = (unsigned int *)depv[1].ptr;
  for (unsigned int i = 0; i < bs; i++) {
    dest[(id * bs) + i] = buffer[i];
  }
  arts_signal_edt(sd_guid, id, dest_guid, DB_MODE_RO);
}

void getter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)depc;
  unsigned int id = (unsigned int)paramv[0];
  unsigned int bs = (unsigned int)paramv[2];
  arts_guid_t dest_guid = (arts_guid_t)paramv[3];

  unsigned int *source = (unsigned int *)depv[0].ptr;
  unsigned int *buffer = &source[(size_t)id * bs];
  unsigned int buf_size = sizeof(unsigned int) * (size_t)bs;

  arts_guid_t am =
      arts_edt_create(setter, paramc, paramv, 2,
                      &(arts_hint_t){.route = arts_guid_get_rank(dest_guid)});
  arts_signal_edt_ptr(am, 0, buffer, buf_size);
  arts_signal_edt(am, 1, dest_guid, DB_MODE_EW);
}

void shut_down_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  unsigned int ne = (unsigned int)paramv[0];
  bool pass = true;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  for (unsigned int i = 0; i < ne; i++) {
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

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  unsigned int block_size = strtol(argv[1], NULL, 10);
  unsigned int num_elements = block_size * arts_get_total_nodes();
  unsigned int last_node = arts_get_total_nodes() - 1;

  arts_guid_t shutdown_guid = arts_guid_reserve(ARTS_EDT, last_node);

  unsigned int *data =
      (unsigned int *)malloc(sizeof(unsigned int) * num_elements);
  for (unsigned int i = 0; i < num_elements; i++) {
    data[i] = i;
  }
  unsigned int data_size = sizeof(unsigned int) * num_elements;

  void *tmp;
  arts_guid_t db_dest_guid =
      arts_db_create(&tmp, sizeof(unsigned int) * num_elements, ARTS_DB_DEFAULT,
                     &(arts_hint_t){.route = last_node});

  uint64_t ne = num_elements;
  arts_edt_create_with_guid(shut_down_edt, shutdown_guid, 1, &ne,
                            arts_get_total_nodes());

  for (unsigned int r = 0; r < arts_get_total_nodes(); r++) {
    uint64_t getter_params[4] = {r, shutdown_guid, block_size, db_dest_guid};
    arts_guid_t getter_edt = arts_edt_create(getter, 4, getter_params, 1,
                                             &(arts_hint_t){.route = 0});
    arts_signal_edt_ptr(getter_edt, 0, data, data_size);
  }
  free(data);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
