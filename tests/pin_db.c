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

unsigned int node = 0;
arts_guid_t some_db_guid = NULL_GUID;

// This will hang but print a warning if the edt is not on the same node as the
// pinned DBs
void edt_func(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  unsigned int *ptr = (unsigned int *)depv[0].ptr;
  unsigned int *ptr2 = (unsigned int *)depv[1].ptr;

  if (*ptr == 1234) {
    arts_printf("arts_db_create Check\n");
  } else {
    arts_printf("artsDBCreate Fail\n");
  }

  if (*ptr2 == 9876) {
    arts_printf("arts_db_create_with_guid Check\n");
  } else {
    arts_printf("arts_db_create_with_guid Fail\n");
  }

  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  // This is the node we are going to pin to
  node = strtol(argv[1], NULL, 10);
  // Allocate some DB to test arts_db_create_with_guid
  some_db_guid = arts_guid_reserve(ARTS_DB, node);

  unsigned int node_id = arts_get_current_node();
  if (node_id == node) {
    int *ptr = NULL;
    // Set pin to true to pin to node given by command line
    // It is pinned to the node creating the DB
    arts_guid_t db_guid = arts_guid_reserve(ARTS_DB, 0);
    ptr = (int *)arts_db_create_with_guid(db_guid, sizeof(unsigned int), ARTS_DB_LOCAL, NULL, NULL);
    *ptr = 1234;

    // EDT is going to run on node given by command line
    arts_guid_t edt_guid =
        arts_edt_create(edt_func, 0, NULL, 2, &(arts_hint_t){.route = node});

    // Put both signals up front forcing one to be out of order to test the OO
    // code path
    arts_signal_edt(edt_guid, 0, db_guid, DB_MODE_EW);       // Note the mode
    arts_signal_edt(edt_guid, 1, some_db_guid, DB_MODE_EW);  // Note the mode

    // This is the delayed DB
    int *ptr2 = (int *)arts_db_create_with_guid(some_db_guid,
                                                sizeof(unsigned int), ARTS_DB_LOCAL, NULL, NULL);
    *ptr2 = 9876;
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
