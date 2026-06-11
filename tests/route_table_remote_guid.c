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
/// @brief Tests route table operations with remote pre-reserved GUIDs: one DB
///        per non-zero rank (each homed on its rank, created remotely with a
///        guid rank 0 reserved up front) all wired into one rank-0 EDT.
///        Scales to any rank count — each rank owns a distinct guid and a
///        distinct dependence slot, so no guid is double-created and no slot
///        is double-signaled.  Requires multi-node (node_count > 1).

#include "arts.h"

#include <stdio.h>

/// Per-rank payload: rank n writes BASE_VALUE + n so the reader can verify
/// each remote-created DB routed back its own (correct) buffer.
#define BASE_VALUE 999u

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_shutdown();
}

/// depv[i] = the DB created on rank i+1; hard-gates each payload.
void acquire_test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  arts_guid_t shut_guid = (arts_guid_t)paramv[0];
  for (uint32_t i = 0; i < depc; i++) {
    unsigned int *num = (unsigned int *)depv[i].ptr;
    unsigned int expected = BASE_VALUE + (i + 1);
    if (num == NULL || *num != expected) {
      (void)fprintf(stderr,
                    "FAIL: slot %u: ptr=%p value=%u expected=%u (remote-guid "
                    "DB from rank %u routed wrong data)\n",
                    i, (void *)num, num ? *num : 0u, expected, i + 1);
      arts_abort(1);
    }
    arts_printf("%u %u i: %u %u\n", arts_get_current_rank(),
                arts_get_current_worker(), i, *num);
  }
  arts_add_dependence((arts_guid_t)(0), shut_guid, 0, DB_MODE_VAL);
}

/// node_setup paramv: [0]=node_id, [1]=edt_guid, [2]=db_guid.  Runs on rank
/// node_id (>= 1): creates THAT rank's DB under the pre-reserved guid and
/// wires it to its own slot (node_id - 1) of the rank-0 EDT.
void node_setup(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int node_id = (unsigned int)paramv[0];
  arts_guid_t local_edt_guid = (arts_guid_t)paramv[1];
  arts_guid_t local_db_guid = (arts_guid_t)paramv[2];
  unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(
      local_db_guid, sizeof(unsigned int), ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE,
      NULL);
  *ptr = BASE_VALUE + node_id;
  arts_add_dependence(local_db_guid, local_edt_guid, node_id - 1, DB_MODE_RW);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int nranks = arts_get_total_ranks();
  arts_guid_t local_edt_guid = arts_guid_reserve(ARTS_GUID_EDT, 0);
  arts_guid_t local_shutdown_guid = arts_guid_reserve(ARTS_GUID_EDT, 0);

  uint64_t acq_params[1];
  acq_params[0] = (uint64_t)local_shutdown_guid;
  arts_edt_create(acquire_test, 1, acq_params, nranks - 1,
                  &(arts_edt_hint_t){.guid = local_edt_guid});
  arts_edt_create(shutdown_edt, 0, NULL, 1,
                  &(arts_edt_hint_t){.guid = local_shutdown_guid});

  /* One pre-reserved DB guid per non-zero rank, homed on that rank; each
   * rank creates exactly its own guid and signals exactly its own slot. */
  for (unsigned int n = 1; n < nranks; n++) {
    uint64_t args[3];
    args[0] = (uint64_t)n;
    args[1] = (uint64_t)local_edt_guid;
    args[2] = (uint64_t)arts_guid_reserve(ARTS_GUID_DB, n);
    arts_edt_create(node_setup, 3, args, 0, &(arts_edt_hint_t){.rank = n});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
