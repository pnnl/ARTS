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

/* write_test paramv: {index, num_writes, next_guid_or_NULL}
 * depc=2: slot 0 = DB_MODE_RW dep, slot 1 = chain-ordering null signal
 * from predecessor (or main_edt for index 0). */
void write_test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int index = (unsigned int)paramv[0];
  unsigned int num_writes = (unsigned int)paramv[1];
  arts_guid_t next_guid = (arts_guid_t)paramv[2];
  unsigned int *array = (unsigned int *)depv[0].ptr;

  for (unsigned int i = index; i < num_writes; i++) {
    array[i] = index;
  }

  if (next_guid != NULL_GUID) {
    arts_printf("-----------------SIGNALLING NEXT %u\n", index);
    arts_add_dependence(NULL_GUID, next_guid, 1, DB_MODE_NULL);
  } else {
    for (unsigned int i = 0; i < num_writes; i++) {
      arts_printf("i: %u %u\n", i, array[i]);
    }
    arts_shutdown();
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  unsigned int num_writes = (unsigned int)strtol(argv[1], NULL, 10);
  arts_guid_t db_guid = arts_guid_reserve(ARTS_GUID_DB, 0);

  arts_guid_t *write_guids =
      (arts_guid_t *)malloc(sizeof(arts_guid_t) * num_writes);
  for (unsigned int i = 0; i < num_writes; i++) {
    write_guids[i] = arts_guid_reserve(ARTS_GUID_EDT, i % arts_get_total_ranks());
  }

  unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(db_guid, sizeof(unsigned int) * num_writes, ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  for (unsigned int i = 0; i < num_writes; i++) {
    ptr[i] = 0;
  }

  /* Create every write_test EDT from rank 0; arts_edt_create_with_guid
   * auto-routes to arts_guid_get_rank(guid) so each lands on its owner.
   * paramv carries (index, num_writes, next_guid_or_NULL). */
  for (unsigned int i = 0; i < num_writes; i++) {
    arts_guid_t next = (i + 1 < num_writes) ? write_guids[i + 1] : NULL_GUID;
    uint64_t args[3] = {(uint64_t)i, (uint64_t)num_writes, (uint64_t)next};
    arts_edt_create(write_test, 3, args, 2, &(arts_edt_hint_t){.guid = write_guids[i]});
    arts_add_dependence(db_guid, write_guids[i], 0, DB_MODE_RW);
  }

  arts_add_dependence(NULL_GUID, write_guids[0], 1, DB_MODE_NULL);
  free(write_guids);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
