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

/// @file route_table_remote_guid.c
/// @brief Tests route table operations with remote GUIDs (DB on rank 1,
///        EDT on rank 0). Requires multi-node (node_count > 1).

#include "arts.h"

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_shutdown();
}

void acquire_test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  arts_guid_t shut_guid = (arts_guid_t)paramv[0];
  unsigned int *num = (unsigned int *)depv[0].ptr;
  arts_printf("%u %u i: %u %u\n", arts_get_current_rank(),
              arts_get_current_worker(), 0, *num);
  arts_add_dependence((arts_guid_t)(0), shut_guid, 0, DB_MODE_VAL);
}

/// node_setup paramv: [0]=node_id, [1]=edt_guid, [2]=db_guid
void node_setup(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int node_id = (unsigned int)paramv[0];
  arts_guid_t local_edt_guid = (arts_guid_t)paramv[1];
  arts_guid_t local_db_guid = (arts_guid_t)paramv[2];
  if (node_id) {
    unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(
        local_db_guid, sizeof(unsigned int), ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE,
        NULL);
    *ptr = 999;
    arts_add_dependence(local_db_guid, local_edt_guid, 0, DB_MODE_RW);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_guid_t local_edt_guid = arts_guid_reserve(ARTS_GUID_EDT, 0);
  arts_guid_t local_shutdown_guid = arts_guid_reserve(ARTS_GUID_EDT, 0);
  arts_guid_t local_db_guid = arts_guid_reserve(ARTS_GUID_DB, 1);

  uint64_t acq_params[1];
  acq_params[0] = (uint64_t)local_shutdown_guid;
  arts_edt_create(acquire_test, 1, acq_params, 1,
                  &(arts_edt_hint_t){.guid = local_edt_guid});
  arts_edt_create(shutdown_edt, 0, NULL, 1,
                  &(arts_edt_hint_t){.guid = local_shutdown_guid});

  for (unsigned int n = 0; n < arts_get_total_ranks(); n++) {
    uint64_t args[3];
    args[0] = (uint64_t)n;
    args[1] = (uint64_t)local_edt_guid;
    args[2] = (uint64_t)local_db_guid;
    arts_edt_create(node_setup, 3, args, 0, &(arts_edt_hint_t){.rank = n});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
