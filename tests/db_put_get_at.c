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

/// @file db_put_get_at.c
/// @brief Tests arts_put_in_db_at and arts_get_from_db_at (rank-targeted).
///        Requires multi-node (node_count > 1).

#include "arts.h"
#include <string.h>

/// Verify data received from arts_get_from_db_at.
void check_get_at(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 42 && data[1] == 43);
  if (ok) {
    arts_printf("  PASS: put_in_db_at + get_from_db_at round-trip OK\n");
  } else {
    if (data) {
      arts_printf("  FAIL: got [%d, %d] expected [42, 43]\n", data[0], data[1]);
    } else {
      arts_printf("  FAIL: null ptr from get_from_db_at\n");
    }
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_put_get_at (multi-node) ===\n");

  unsigned int total = arts_get_total_nodes();
  if (total < 2) {
    arts_printf("  SKIP: need node_count >= 2 (have %u)\n", total);
    arts_shutdown();
    return;
  }

  unsigned int target = 1;

  // Create DB on remote rank.
  arts_guid_t db = arts_db_create_remote(target, 2 * sizeof(int));

  // Put data at rank=target.
  int send[2] = {42, 43};
  arts_put_in_db_at(send, NULL_GUID, db, 0, 0, 2 * sizeof(int), target);

  // Get data from rank=target.
  arts_guid_t reader =
      arts_edt_create(check_get_at, 0, NULL, 1, &(arts_hint_t){.route = 0});
  arts_get_from_db_at(reader, db, 0, 0, 2 * sizeof(int), target);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
