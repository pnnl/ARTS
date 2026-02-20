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
#include "arts/utils/atomics.h"

unsigned int counter = 0;
unsigned int num_dummy = 0;
arts_guid_t exit_guid = NULL_GUID;

void dummytask(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_atomic_add(&counter, 1);
}

void exit_program(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  unsigned int num_nodes = arts_get_total_nodes();
  for (unsigned int i = 0; i < depc; i++) {
    unsigned int num_edts = depv[i].guid;
    if (num_edts != (num_nodes * num_dummy) + 2) {
      arts_printf("Error: %u vs %u\n", num_edts, (num_nodes * num_dummy) + 2);
    }
  }
  arts_printf("Exit %u\n", counter);
  arts_shutdown();
}

void root_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_guid_t guid = arts_get_current_epoch_guid();
  arts_printf("Starting %lu %u\n", guid, arts_guid_get_rank(guid));
  unsigned int num_nodes = arts_get_total_nodes();
  for (unsigned int rank = 0; rank < num_nodes * num_dummy; rank++) {
    arts_edt_create(dummytask, 0, 0, 0,
                    &(arts_hint_t){.route = rank % num_nodes});
  }
}

void node_setup(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int node_id = (unsigned int)paramv[0];
  arts_initialize_and_start_epoch(exit_guid, node_id);
  arts_edt_create(root_task, 0, NULL, 0, &(arts_hint_t){.route = node_id});
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  num_dummy = (unsigned int)strtol(argv[1], NULL, 10);
  exit_guid = arts_guid_reserve(ARTS_EDT, 0);
  arts_edt_create_with_guid(exit_program, exit_guid, 0, NULL,
                            arts_get_total_nodes());

  for (unsigned int n = 0; n < arts_get_total_nodes(); n++) {
    uint64_t args = n;
    arts_edt_create(node_setup, 1, &args, 0, &(arts_hint_t){.route = n});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
