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

#include <stdio.h>

#include "arts.h"
#include "arts/gas/route_table.h"
#include "arts/runtime/globals.h"

#define MYSIZE 10

// Run with only 1 node 1 worker!

void print_rt(const char *message) {
  arts_printf("Start: %s\n", message);
  arts_route_table_iterator_t *iter =
      arts_new_route_table_iterator(arts_node_info.route_table[0]);
  arts_route_item_t *item = arts_route_table_iterate(iter);
  while (item) {
    arts_print_item(item);
    item = arts_route_table_iterate(iter);
  }
  arts_printf("End: %s\n", message);
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int node_id = arts_get_current_node();
  int dummy_rank;

  printf("Start\n");
  arts_guid_range_t *range = arts_guid_range_create(ARTS_DB, MYSIZE, node_id);
  for (uint64_t i = 0; i < MYSIZE; i++) {
    arts_db_create_with_guid(arts_guid_range_next(range), 1024 * sizeof(char),
                             NULL);
  }
  print_rt("After DB Init");

  for (uint64_t i = 0; i < MYSIZE; i++) {
    arts_route_table_lookup_db(arts_guid_range_get(range, i), &dummy_rank,
                               true);
  }
  print_rt("After DB Lookup");

  for (uint64_t i = 0; i < MYSIZE; i++) {
    internal_route_table_return_db(arts_node_info.route_table[0],
                                   arts_guid_range_get(range, i), false, false);
  }
  print_rt("After DB Return with Mark");

  arts_clean_up_route_table(arts_node_info.route_table[0]);
  print_rt("After GC");

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
