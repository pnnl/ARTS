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
#include "bins.h"

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "arts.h"
#include "arts/utils/array_list.h"

static arts_array_list_t **list;

void init_list_record() {
  unsigned int size = arts_get_total_gpus() + 1;
  list = (arts_array_list_t **)arts_calloc(size, sizeof(arts_array_list_t *));
  for (unsigned int i = 0; i < size; i++) {
    list[i] = arts_new_array_list(sizeof(unsigned int), 32);
}
}

void add_to_list(unsigned int frontier_size, unsigned int index) {
  arts_push_to_array_list(list[index], &frontier_size);
}

void write_bins_to_file(unsigned int index) {
  // Do the CPU bins
  if (index + 1 == arts_get_total_gpus()) {
    write_bins_to_file(index + 1);
}

  char filename[1024];
  (void)sprintf(filename, "bins_%u_%u.ct", arts_get_current_node(), index);
  if (arts_length_array_list(list[index])) {
    FILE *fp = fopen(filename, "w");
    if (fp) {
      arts_printf("Writing to %s\n", filename);
      arts_array_list_iterator_t *iter = arts_new_array_list_iterator(list[index]);
      unsigned int *bin;
      while (arts_array_list_has_next(iter)) {
        bin = (unsigned int *)arts_array_list_next(iter);
        (void)fprintf(fp, "%u\n", *bin);
      }
      arts_delete_array_list_iterator(iter);
      (void)fclose(fp);
    } else {
      arts_printf("Couldn't open %s\n", filename);
}
  }
}
