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

arts_guid_t db_guid = NULL_GUID;
arts_guid_t a_guid = NULL_GUID;
arts_guid_t b_guid = NULL_GUID;

void check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  unsigned int *ptr;
  arts_guid_t guid = arts_guid_reserve(ARTS_DB_ONCE, 0);
  ptr = (unsigned int *)arts_db_create_with_guid(guid, sizeof(unsigned int), NULL);
  *ptr = 2;

  arts_printf("Check: %lu %u new_guid: %lu\n", depv[0].guid,
         *((unsigned int *)depv[0].ptr), guid);
  arts_signal_edt(b_guid, 0, guid, ARTS_DB_WRITE);
}

void shut_down_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  arts_printf("ShutdownEdt: %lu %u\n", depv[0].guid, *((unsigned int *)depv[0].ptr));
  arts_shutdown();
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  db_guid = arts_guid_reserve(ARTS_DB_ONCE, 0);
  a_guid = arts_guid_reserve(ARTS_EDT, 1);
  b_guid = arts_guid_reserve(ARTS_EDT, 2);

  unsigned int node_id = arts_get_current_node();
  if (node_id == 0) {
    unsigned int *a_ptr =
        (unsigned int *)arts_db_create_with_guid(db_guid, sizeof(unsigned int), NULL);
    *a_ptr = 1;
  }

  if (node_id == 1) {
    arts_edt_create_with_guid(check, a_guid, 0, NULL, 1);
    arts_signal_edt(a_guid, 0, db_guid, ARTS_DB_WRITE);
  }

  if (node_id == 2) {
    arts_edt_create_with_guid(shut_down_edt, b_guid, 0, NULL, 1);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
