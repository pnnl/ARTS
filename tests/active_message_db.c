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
 *   [0] = db_source_guid (EW mode)
 *
 * setter depv (depc=2):
 *   [0] = db_dest_guid (EW mode)
 *   [1] = cpy_db (EW mode)
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

  unsigned int *dest = (unsigned int *)depv[0].ptr;
  unsigned int *buffer = (unsigned int *)depv[1].ptr;
  for (unsigned int i = 0; i < bs; i++) {
    dest[(id * bs) + i] = buffer[i];
  }
  arts_add_dependence(dest_guid, sd_guid, id, DB_MODE_RO);
}

void getter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  (void)depc;
  unsigned int id = (unsigned int)paramv[0];
  unsigned int bs = (unsigned int)paramv[2];
  arts_guid_t dest_guid = (arts_guid_t)paramv[3];

  unsigned int *buffer;
  arts_guid_t cpy_db =
      arts_db_create((void **)&buffer, sizeof(unsigned int) * bs,
                     ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);

  unsigned int *source = (unsigned int *)depv[0].ptr;
  for (unsigned int i = 0; i < bs; i++) {
    buffer[i] = source[(id * bs) + i];
  }
  arts_guid_t am =
      arts_edt_create(setter, paramc, paramv, 2,
                      &(arts_edt_hint_t){.rank = arts_guid_get_rank(dest_guid)});
  arts_add_dependence(dest_guid, am, 0, DB_MODE_RW);
  arts_add_dependence(cpy_db, am, 1, DB_MODE_RW);
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
  unsigned int num_elements = block_size * arts_get_total_ranks();
  unsigned int last_node = arts_get_total_ranks() - 1;

  arts_guid_t db_source_guid = arts_guid_reserve(ARTS_GUID_DB, 0);
  arts_guid_t shutdown_guid = arts_guid_reserve(ARTS_GUID_EDT, last_node);

  unsigned int *db_data = (unsigned int *)arts_db_create_with_guid(
      db_source_guid, sizeof(unsigned int) * num_elements, ARTS_DB_DEFAULT,
      ARTS_DB_PROP_NONE, NULL);
  for (unsigned int i = 0; i < num_elements; i++) {
    db_data[i] = i;
  }

  void *tmp;
  arts_guid_t db_dest_guid =
      arts_db_create(&tmp, sizeof(unsigned int) * num_elements, ARTS_DB_DEFAULT,
                     ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = last_node});

  uint64_t ne = num_elements;
  arts_edt_create(shut_down_edt, 1, &ne, arts_get_total_ranks(), &(arts_edt_hint_t){.guid = shutdown_guid});

  for (unsigned int r = 0; r < arts_get_total_ranks(); r++) {
    uint64_t getter_params[4] = {r, shutdown_guid, block_size, db_dest_guid};
    arts_guid_t getter_edt = arts_edt_create(
        getter, 4, getter_params, 1,
        &(arts_edt_hint_t){.rank = arts_guid_get_rank(db_source_guid)});
    arts_add_dependence(db_source_guid, getter_edt, 0, DB_MODE_RW);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
