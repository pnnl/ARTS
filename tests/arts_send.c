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
#include "arts/runtime/network/remote_functions.h"

unsigned int num_elements = 0;

void send_handler(void *args) {
  bool pass = true;
  unsigned int *data = (unsigned int *)args;
  for (unsigned int i = 0; i < num_elements; i++) {
    if (data[i] != i) {
      pass = false;
}
  }
  if (pass) {
    arts_printf("CHECK %u of %u\n", arts_get_current_node(), arts_get_total_nodes());
}

  if (arts_get_current_node() + 1 == arts_get_total_nodes()) {
    arts_printf("Shutdown\n");
    arts_shutdown();
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  num_elements = strtol(argv[1], NULL, 10);
  unsigned int size = sizeof(unsigned int) * num_elements;
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    unsigned int *data = (unsigned int *)malloc(size);
    for (unsigned int j = 0; j < num_elements; j++) {
      data[j] = j;
    }
    arts_remote_send(i, send_handler, data, size, true);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
