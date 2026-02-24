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

unsigned int counter = 0;
uint64_t num_dummy = 0;
arts_guid_t exit_guid = NULL_GUID;

void dummytask(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {}

void sync_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  arts_printf("Guid:%lu Sync %lu: %lu\n", arts_get_current_guid(), paramv[0],
              depv[0].guid);
}

void exit_program(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  arts_printf("Exit: %lu\n", depv[0].guid);
  arts_shutdown();
}

void root_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  uint64_t dep = paramv[0];
  if (dep) {
    dep--;
    //        arts_guid_t guid = arts_edt_create(sync_task, 1,
    //        &dep, 1, &(arts_hint_t){.route = arts_get_current_node()});
    //        arts_guid_t epoch_guid = arts_initialize_and_start_epoch(guid,
    //        0);
    arts_guid_t epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
    arts_printf("Guid:%lu Root: %lu sync: %lu epoch: %lu\n",
                arts_get_current_guid(), dep, NULL_GUID, epoch_guid);

    unsigned int num_nodes = arts_get_total_nodes();
    for (unsigned int rank = 0; rank < num_nodes; rank++) {
      arts_edt_create(root_task, 1, &dep, 0,
                      &(arts_hint_t){.route = rank % num_nodes});
    }

    for (uint64_t rank = 0; rank < num_nodes * num_dummy; rank++) {
      arts_edt_create(dummytask, 0, NULL, 0,
                      &(arts_hint_t){.route = rank % num_nodes});
    }

    arts_wait_on_handle(epoch_guid);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  num_dummy = (uint64_t)strtol(argv[1], NULL, 10);
  exit_guid = arts_guid_reserve(ARTS_EDT, 0);
  arts_printf("Starting\n");
  arts_edt_create_with_guid(exit_program, exit_guid, 0, NULL, 1);
  arts_initialize_and_start_epoch(exit_guid, 0);
  arts_edt_create(root_task, 1, &num_dummy, 0, &(arts_hint_t){.route = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
