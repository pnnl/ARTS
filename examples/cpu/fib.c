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

uint64_t start = 0;

void fib_join(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  unsigned int x = (unsigned int)depv[0].guid;
  unsigned int y = (unsigned int)depv[1].guid;
  arts_signal_edt_value((arts_guid_t)paramv[0], paramv[1], x + y);
}

void fib_fork(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  unsigned int next = (arts_get_current_node() + 1) % arts_get_total_nodes();
  //    arts_printf("NODE: %u WORKER: %u NEXT: %u\n", arts_get_current_node(),
  //    arts_get_current_worker(), next);

  arts_guid_t guid = (arts_guid_t)paramv[0];
  unsigned int slot = paramv[1];
  unsigned int num = paramv[2];
  if (num < 2) {
    arts_signal_edt_value(guid, slot, num);
  } else {
    arts_guid_t join_guid =
        arts_edt_create(fib_join, paramc - 1, paramv, 2,
                        &(arts_hint_t){.route = arts_get_current_node()});

    uint64_t args[3] = {(uint64_t)join_guid, 0, num - 1};
    arts_edt_create(fib_fork, 3, args, 0, &(arts_hint_t){.route = next});

    args[1] = 1;
    args[2] = num - 2;
    arts_edt_create(fib_fork, 3, args, 0, &(arts_hint_t){.route = next});
  }
}

void fib_done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  uint64_t time = arts_get_time_stamp() - start;
  arts_printf("Fib %u: %u time: %lu nodes: %u workers: %u\n", paramv[0],
              depv[0].guid, time, arts_get_total_nodes(),
              arts_get_total_workers());
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  uint64_t num = strtol(argv[1], NULL, 10);
  arts_guid_t done_guid =
      arts_edt_create(fib_done, 1, &num, 1, &(arts_hint_t){.route = 0});
  uint64_t args[3] = {(uint64_t)done_guid, 0, num};
  start = arts_get_time_stamp();
  arts_guid_t guid =
      arts_edt_create(fib_fork, 3, args, 0, &(arts_hint_t){.route = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
