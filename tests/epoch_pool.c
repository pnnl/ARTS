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
#include "arts/runtime/compute/shad_adapter.h"

uint64_t num_dummy = 0;

void dummytask(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  uint64_t index = paramv[0];
  uint64_t dep = paramv[1];
  arts_printf("Dep: %lu ID: %lu Current Node: %u Current Worker: %u\n", dep,
              index, arts_get_current_node(), arts_get_current_worker());
}

void root_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  uint64_t dep = paramv[0];
  arts_printf("Root: %lu\n", dep);
  if (dep) {
    arts_guid_t pool_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);

    dep--;
    unsigned int num_nodes = arts_get_total_nodes();
    //        arts_edt_create_shad(root_task,
    //        (arts_get_current_node()+1)%num_nodes, 1, &dep);
    arts_edt_create_dep(
        root_task, 1, &dep, 0, false,
        &(arts_hint_t){.route = (arts_get_current_node() + 1) % num_nodes});

    //        uint64_t args[2];
    //        args[0] = dep;
    //
    //        for(uint64_t i=0; i<num_dummy; i++)
    //        {
    //            args[1] = i;
    //            arts_edt_create_dep(dummytask, 2, args, 0, false,
    //            &(arts_hint_t){.route = i%num_nodes});
    //        }
    arts_printf("Waiting on %lu\n", pool_guid);
    if (arts_wait_on_handle(pool_guid)) {
      arts_printf("Done waiting on %lu dep: %lu\n", pool_guid, dep);
    }
  }

  if (dep + 1 == num_dummy) {
    arts_shutdown();
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  num_dummy = (uint64_t)strtol(argv[1], NULL, 10);
  arts_printf("Starting\n");
  uint64_t arg = num_dummy;
  arts_edt_create_shad(root_task, 0, 1, &arg);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
