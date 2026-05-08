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

/// @file db_local_create.c
/// @brief Tests arts_db_local_create() public API.

#include "arts.h"
#include <string.h>

#define NUM_ELEMS (128 / sizeof(unsigned int))

/// Test 1: Local creation with explicit hint — verify data and GUID type.
void check_local(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t guid = (arts_guid_t)paramv[0];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  bool ok = (data != NULL);
  if (ok) {
    for (unsigned int i = 0; i < NUM_ELEMS; i++) {
      if (data[i] != i + 1) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    ok = (arts_guid_get_type(guid) == ARTS_DB);
  }
  arts_printf("  %s: arts_db_local_create local path\n", ok ? "PASS" : "FAIL");
}

/// Test 2: NULL hint => current node.
void check_null_hint(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t guid = (arts_guid_t)paramv[0];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  bool ok = (data != NULL && arts_guid_get_type(guid) == ARTS_DB);
  if (ok) {
    ok = (*data == 42);
  }
  arts_printf("  %s: arts_db_local_create with NULL hint\n",
              ok ? "PASS" : "FAIL");
}

/// Test 3a: EW writer — multiply by 3.
void ew_modify(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data) {
    for (unsigned int i = 0; i < NUM_ELEMS; i++) {
      data[i] *= 3;
    }
  }
}

/// Test 3b: EW verifier — check values are (i+1)*3.
void ew_verify(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  bool ok = (data != NULL);
  if (ok) {
    for (unsigned int i = 0; i < NUM_ELEMS; i++) {
      if (data[i] != (i + 1) * 3) {
        ok = false;
        break;
      }
    }
  }
  arts_printf("  %s: arts_db_local_create EW ordering\n", ok ? "PASS" : "FAIL");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_local_create ===\n");

  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  // Test 1: Local creation with explicit route = current node.
  void *ptr1 = NULL;
  arts_db_hint_t hint1 = {.rank = arts_get_current_rank()};
  arts_guid_t g1 = arts_db_create(&ptr1, NUM_ELEMS * sizeof(unsigned int),
                                  ARTS_DB_PIN, ARTS_DB_PROP_NONE, &hint1);
  unsigned int *d1 = (unsigned int *)ptr1;
  for (unsigned int i = 0; i < NUM_ELEMS; i++) {
    d1[i] = i + 1;
  }
  arts_db_release(g1);
  uint64_t p1 = (uint64_t)g1;
  arts_guid_t e1 =
      arts_edt_create(check_local, 1, &p1, 1, &(arts_edt_hint_t){.epoch = epoch});
  arts_add_dependence(g1, e1, 0, DB_MODE_RO);

  // Test 2: NULL hint.
  void *ptr2 = NULL;
  arts_guid_t g2 = arts_db_create(&ptr2, sizeof(unsigned int), ARTS_DB_PIN,
                                  ARTS_DB_PROP_NONE, NULL);
  *(unsigned int *)ptr2 = 42;
  arts_db_release(g2);
  uint64_t p2 = (uint64_t)g2;
  arts_guid_t e2 =
      arts_edt_create(check_null_hint, 1, &p2, 1, &(arts_edt_hint_t){.epoch = epoch});
  arts_add_dependence(g2, e2, 0, DB_MODE_RO);

  // Test 3: EW ordering — writer then verifier.
  void *ptr3 = NULL;
  arts_guid_t g3 = arts_db_create(&ptr3, NUM_ELEMS * sizeof(unsigned int),
                                  ARTS_DB_PIN, ARTS_DB_PROP_NONE, NULL);
  unsigned int *d3 = (unsigned int *)ptr3;
  for (unsigned int i = 0; i < NUM_ELEMS; i++) {
    d3[i] = i + 1;
  }
  arts_db_release(g3);

  arts_guid_t e3b =
      arts_edt_create(ew_verify, 0, NULL, 1, &(arts_edt_hint_t){.epoch = epoch});
  arts_guid_t e3a =
      arts_edt_create(ew_modify, 0, NULL, 1, &(arts_edt_hint_t){.epoch = epoch});
  arts_add_dependence(g3, e3a, 0, DB_MODE_RW);
  arts_add_dependence(g3, e3b, 0, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
