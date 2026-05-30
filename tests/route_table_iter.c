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
#include "arts/runtime_state.h"
#include "arts/utils/atomics.h"

#define MYSIZE 10

void print_rt() {
  arts_route_table_iterator_t iter;
  arts_reset_route_table_iterator(&iter, arts_node_info.route_table[0]);
  arts_route_item_t *item = arts_route_table_iterate(&iter);
  while (item) {
    arts_print_item(item);
    item = arts_route_table_iterate(&iter);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int node_id = arts_get_current_rank();
  printf("Init per node\n");
  arts_guid_t range_start =
      arts_guid_reserve_range(ARTS_GUID_EDT, MYSIZE, node_id);
  for (uint64_t i = 0; i < MYSIZE; i++) {
    /* Install a non-owned sentinel (fake integer ptr) for the iterator-walk
     * exercise.  NULL deleter so the cb does not free this bogus address at
     * shutdown (deleter-by-kind would pick arts_edt_deleter and crash). */
    void *location = arts_route_table_add_item_with_deleter(
        (void *)(uintptr_t)range_start, arts_guid_from_index(range_start, i),
        NULL);
    (void)location;
  }

  print_rt();

  arts_guid_t guid = arts_guid_from_index(range_start, 0);

  /* Legacy arts_route_table_lookup_db(guid, &rank, mark_to_delete) is gone
   * post-Phase-6.  The replacement split: lookup_item / lookup_data return
   * the data pointer atomically; lookup_rank returns the rank.  Neither
   * carries a "mark to delete" flag — destruction lives on the
   * acquire_item / release_item / mark_delete API.  The iterator-walk and
   * post-walk add_item exercise here verifies the data path only. */

  void *ptr = arts_route_table_lookup_data(guid);
  int rank = arts_route_table_lookup_rank(guid);
  arts_printf("Lookup %lu %p (rank %d)\n", guid, ptr, rank);

  ptr = arts_route_table_lookup_data(guid);
  arts_printf("DB Lookup %lu %p\n", guid, ptr);

  /* Install a sentinel (non-owned integer) for the data-path exercise.  Use
   * the explicit-deleter install with NULL so the cb does NOT try to free this
   * fake pointer as a real DB at shutdown (deleter-by-kind would pick
   * arts_db_deleter and crash on the bogus address). */
  (void)node_id;
  void *location = arts_route_table_add_item_with_deleter(
      (void *)(uintptr_t)range_start, guid, NULL);
  (void)location;

  ptr = arts_route_table_lookup_data(guid);
  arts_printf("Lookup2 %lu %p\n", guid, ptr);

  ptr = arts_route_table_lookup_data(guid);
  arts_printf("DB Lookup2 %lu %p\n", guid, ptr);

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
