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

arts_guid_t shutdown_guid;
arts_guid_t edt_guid;
arts_guid_t db_guid;

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  arts_shutdown();
}

void acquire_test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  unsigned int *num = (unsigned int *)depv[0].ptr;
  ARTS_PRINTF("%u %u i: %u %u\n", arts_get_current_node(), arts_get_current_worker(), 0,
         *num);
  arts_signal_edt_value(shutdown_guid, 0, 0);
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  edt_guid = arts_reserve_guid_route(ARTS_EDT, 0);
  shutdown_guid = arts_reserve_guid_route(ARTS_EDT, 0);
  db_guid = arts_reserve_guid_route(ARTS_DB_READ, 1);
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {
  if (!worker_id) {
    if (node_id) {
      unsigned int *ptr =
          (unsigned int *)arts_db_create_with_guid(db_guid, sizeof(unsigned int));
      *ptr = 999;
      arts_signal_edt(edt_guid, 0, db_guid);
    } else {
      arts_edt_create_with_guid(shutdown_edt, shutdown_guid, 0, NULL, 1);
      arts_edt_create_with_guid(acquire_test, edt_guid, 0, NULL, 1);
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
