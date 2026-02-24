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
#include "arts.h"
#include <stdlib.h>
#include <string.h>

arts_guid_t db_dest_guid = NULL_GUID;
arts_guid_t shutdown_guid = NULL_GUID;
unsigned int num_elements = 0;
unsigned int block_size = 0;

void dummy(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  arts_guid_t result_guid = (arts_guid_t)paramv[0];
  unsigned int result_size = paramv[1];
  unsigned int buffer_size = paramv[2] / sizeof(unsigned int);
  unsigned int *buffer = (unsigned int *)depv[0].ptr;
  arts_printf("%lu %u %u %p\n", result_guid, result_size, buffer_size, buffer);
  unsigned int *sum = (unsigned int *)calloc(1, result_size);
  for (unsigned int i = 0; i < buffer_size; i++) {
    arts_printf("%u\n", buffer[i]);
    *sum += buffer[i];
  }
  arts_printf("Sum before: %u\n", *sum);
  arts_set_buffer(result_guid, sum, result_size);
  free(sum);
}

void start_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  uint64_t args[3];

  unsigned int result = 0;
  unsigned int *data_ptr = &result;
  args[0] = arts_allocate_local_buffer((void **)&data_ptr, sizeof(unsigned int),
                                       1, NULL_GUID);
  args[1] = sizeof(unsigned int);

  unsigned int buffer_size = sizeof(unsigned int) * 5;
  unsigned int *data = (unsigned int *)calloc(1, buffer_size);
  for (unsigned int i = 0; i < 5; i++) {
    data[i] = i;
  }
  args[2] = buffer_size;

  void *data_copy = malloc(buffer_size);
  memcpy(data_copy, data, buffer_size);
  unsigned int target = (arts_get_current_node() + 1) % arts_get_total_nodes();
  arts_guid_t am =
      arts_edt_create(dummy, 3, args, 1, &(arts_hint_t){.route = target});
  arts_signal_edt_ptr(am, 0, data_copy, buffer_size);
  free(data);
  free(data_copy);

  while (!result) {
    arts_yield();
    arts_printf("Did a YIELD\n");
  }

  arts_printf("Sum: %u\n", result);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("Starting\n");
  arts_edt_create(start_edt, 0, NULL, 0, &(arts_hint_t){.route = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
