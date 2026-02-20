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

arts_array_db_t *array = NULL;
arts_guid_t array_guid = NULL_GUID;
unsigned int elements = 32;

void check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  for (unsigned int i = 0; i < depc; i++) {
    unsigned int *data = (unsigned int *)depv[i].ptr;
    arts_printf("%u: %u\n", i, *data);
  }

  arts_shutdown();
}

void gather_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_printf("Gather task\n");
  arts_guid_t edt_guid =
      arts_edt_create(check, 0, NULL, elements, &(arts_hint_t){.route = 0});
  for (unsigned int i = 0; i < elements; i++) {
    arts_get_from_array_db(edt_guid, i, array, i);
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  elements = (unsigned int)strtol(argv[1], NULL, 10);
  arts_guid_t gather_guid =
      arts_edt_create(gather_task, 0, 0, 1, &(arts_hint_t){.route = 0});
  arts_initialize_and_start_epoch(gather_guid, 0);
  array_guid = arts_new_array_db(&array, sizeof(unsigned int), elements);
  for (unsigned int i = 0; i < elements; i++) {
    arts_put_in_array_db(&i, NULL_GUID, 0, array, i);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
