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

/// @file edt_create_basic.c
/// @brief Tests basic EDT creation variants: arts_edt_create, with_guid,
///        with finish scope, _dep, with multiple paramv sizes, zero depc, etc.

#include "arts.h"
#include <string.h>

/// Number of sub-tests.
#define NUM_SUBTESTS 5

static volatile unsigned int passed = 0;
static volatile unsigned int failed = 0;

/// Collector GUID: each subtest signals this after completing.
static arts_guid_t coll_guid = NULL_GUID;

static void report(const char *name, bool ok, unsigned int slot) {
  if (ok) {
    __sync_fetch_and_add((unsigned int *)&passed, 1);
    arts_printf("  PASS: %s\n", name);
  } else {
    __sync_fetch_and_add((unsigned int *)&failed, 1);
    arts_printf("  FAIL: %s\n", name);
  }
  arts_add_dependence((arts_guid_t)(1), coll_guid, slot, DB_MODE_VAL);
}

/// 1) EDT with zero params and zero deps fires immediately.
void zero_dep_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depv;
  (void)paramv;
  report("zero_dep_edt fires", paramc == 0 && depc == 0, 0);
}

/// 2) EDT with params verifies param delivery.
void param_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depv;
  (void)depc;
  bool ok = (paramc == 3 && paramv[0] == 42 && paramv[1] == 0xDEADBEEF &&
             paramv[2] == 99);
  report("param delivery", ok, 1);
}

/// 3) EDT created with pre-reserved GUID.
void guid_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depv;
  (void)depc;
  (void)paramc;
  arts_guid_t my_guid = arts_edt_get_current_guid();
  arts_guid_t expected = (arts_guid_t)paramv[0];
  report("create_with_guid matches", my_guid == expected, 2);
}

/// 4) EDT created in a finish scope.
void scope_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depv;
  (void)depc;
  (void)paramc;
  (void)paramv;
  report("scope_edt fires in finish scope", true, 3);
}

/// 5) EDT with has_depv=true — verify depv storage exists.
void dep_true_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramv;
  (void)paramc;
  bool ok = (depc == 1 && depv[0].guid != NULL_GUID);
  report("create_dep has_depv=true delivers data", ok, 4);
}

/// Collector EDT: once all sub-tests have signaled, print summary and shutdown.
void collector_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depv;
  (void)depc;
  arts_printf("=== edt_create_basic: %u passed, %u failed ===\n", passed,
              failed);
  if (failed == 0) {
    arts_printf("ALL TESTS PASSED\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_create_basic ===\n");

  // Collector: each subtest signals one slot when done.
  coll_guid = arts_edt_create(collector_edt, 0, NULL, NUM_SUBTESTS,
                              &(arts_edt_hint_t){.rank = 0});

  // 1) Zero deps — fires immediately.
  arts_edt_create(zero_dep_edt, 0, NULL, 0, &(arts_edt_hint_t){.rank = 0});

  // 2) Param delivery.
  uint64_t params[3] = {42, 0xDEADBEEF, 99};
  arts_edt_create(param_edt, 3, params, 0, &(arts_edt_hint_t){.rank = 0});

  // 3) Create with pre-reserved GUID.
  arts_guid_t reserved = arts_guid_reserve(ARTS_GUID_EDT, 0);
  uint64_t guid_param = (uint64_t)reserved;
  arts_edt_create(guid_edt, 1, &guid_param, 0,
                  &(arts_edt_hint_t){.guid = reserved});

  // 4) Create with finish scope.
  arts_guid_t fe_guid = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_edt_create(scope_edt, 0, NULL, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe_guid});

  // 5) Create dep with has_depv=true — signal with DB.
  void *db_ptr = NULL;
  arts_guid_t db =
      arts_db_create(&db_ptr, 64, ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  arts_guid_t dep_t =
      arts_edt_create(dep_true_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(db, dep_t, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
