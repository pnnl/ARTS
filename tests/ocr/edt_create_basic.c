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
#include <stdatomic.h>
#include <string.h>

/// Number of sub-tests.
#define NUM_SUBTESTS 5

/// Counter DB layout: [passed, failed] as two consecutive _Atomic unsigned int.
typedef struct {
  _Atomic unsigned int passed;
  _Atomic unsigned int failed;
} counter_db_t;

/// paramv layout shared by all subtest EDTs:
///   paramv[0] = coll_guid  (uint64_t cast of arts_guid_t)
///   paramv[1] = counter_db (uint64_t cast of arts_guid_t)
#define PV_COLL 0
#define PV_CTR  1

/// Increment the appropriate counter in the counter DB (via DB_MODE_RW dep in
/// depv[0]), print the result, then signal the collector via DB_MODE_VAL.
/// Each subtest EDT acquires counter_db as depv[0] with DB_MODE_RW.
static void report(arts_edt_dep_t depv[], const uint64_t *paramv,
                   const char *name, bool ok, unsigned int slot) {
  counter_db_t *ctr = (counter_db_t *)depv[0].ptr;
  if (ok) {
    atomic_fetch_add_explicit(&ctr->passed, 1u, memory_order_relaxed);
    arts_printf("  PASS: %s\n", name);
  } else {
    atomic_fetch_add_explicit(&ctr->failed, 1u, memory_order_relaxed);
    arts_printf("  FAIL: %s\n", name);
  }
  arts_guid_t coll_guid = (arts_guid_t)paramv[PV_COLL];
  arts_add_dependence((arts_guid_t)(1), coll_guid, slot, DB_MODE_VAL);
}

/// 1) EDT with zero params and zero deps fires immediately.
///    depv[0] = counter_db (RW)
void zero_dep_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  /* depc==1 here because of counter_db dep; original test checked paramc==0
   * and that the EDT fires (zero *user* deps); we verify those properties. */
  report(depv, paramv, "zero_dep_edt fires",
         paramc == 2 /* coll_guid + ctr */ && depv[0].ptr != NULL, 0);
}

/// 2) EDT with params verifies param delivery.
///    depv[0] = counter_db (RW)
void param_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  /* paramv[0]=coll, [1]=ctr, [2..4]=test params */
  bool ok = (paramc == 5 && paramv[2] == 42 && paramv[3] == 0xDEADBEEF &&
             paramv[4] == 99);
  report(depv, paramv, "param delivery", ok, 1);
}

/// 3) EDT created with pre-reserved GUID.
///    depv[0] = counter_db (RW)
void guid_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  arts_guid_t my_guid = arts_edt_get_current_guid();
  arts_guid_t expected = (arts_guid_t)paramv[2];
  report(depv, paramv, "create_with_guid matches", my_guid == expected, 2);
}

/// 4) EDT created in a finish scope.
///    depv[0] = counter_db (RW)
void scope_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  report(depv, paramv, "scope_edt fires in finish scope", true, 3);
}

/// 5) EDT with has_depv=true — verify depv storage exists.
///    depv[0] = counter_db (RW), depv[1] = the test DB (RO)
void dep_true_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramv;
  (void)paramc;
  bool ok = (depc == 2 && depv[1].guid != NULL_GUID);
  report(depv, paramv, "create_dep has_depv=true delivers data", ok, 4);
}

/// Collector EDT: once all sub-tests have signaled, print summary and shutdown.
/// paramv[0] = counter_db guid
/// depv[0..NUM_SUBTESTS-1] = DB_MODE_VAL slots from each subtest
/// depv[NUM_SUBTESTS] = counter_db (RO) to read totals
void collector_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  /* Counter DB arrives in depv[NUM_SUBTESTS] as RO. */
  counter_db_t *ctr = (counter_db_t *)depv[NUM_SUBTESTS].ptr;
  unsigned int p = atomic_load_explicit(&ctr->passed, memory_order_relaxed);
  unsigned int f = atomic_load_explicit(&ctr->failed, memory_order_relaxed);
  arts_printf("=== edt_create_basic: %u passed, %u failed ===\n", p, f);
  if (f == 0) {
    arts_printf("ALL TESTS PASSED\n");
  }
  /* Suppress unused-paramv warning: paramv[0] held counter_db guid for
   * identification purposes; it is read via the dep slot instead. */
  (void)paramv;
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_create_basic ===\n");

  /* Counter DB: holds [passed, failed] as atomics. */
  void *ctr_ptr = NULL;
  arts_guid_t ctr_db = arts_db_create(&ctr_ptr, sizeof(counter_db_t),
                                       ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  counter_db_t *ctr = (counter_db_t *)ctr_ptr;
  atomic_init(&ctr->passed, 0u);
  atomic_init(&ctr->failed, 0u);
  arts_db_release(ctr_db, DB_MODE_RW);

  /* Collector: NUM_SUBTESTS DB_MODE_VAL slots + 1 RO dep for the counter DB. */
  arts_guid_t coll_guid =
      arts_edt_create(collector_edt, 0, NULL, NUM_SUBTESTS + 1,
                      &(arts_edt_hint_t){.rank = 0});
  /* Wire counter DB as the last dep of collector (RO read after all subtests). */
  arts_add_dependence(ctr_db, coll_guid, NUM_SUBTESTS, DB_MODE_RO);

  /* Common paramv prefix for all subtest EDTs: [coll_guid, ctr_db]. */
  uint64_t base[2] = {(uint64_t)coll_guid, (uint64_t)ctr_db};

  /* 1) Zero user deps — the only dep is counter_db. */
  {
    arts_guid_t e =
        arts_edt_create(zero_dep_edt, 2, base, 1, &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(ctr_db, e, 0, DB_MODE_RW);
  }

  /* 2) Param delivery: paramv = [coll, ctr, 42, 0xDEADBEEF, 99]. */
  {
    uint64_t pv[5] = {base[0], base[1], 42, 0xDEADBEEF, 99};
    arts_guid_t e =
        arts_edt_create(param_edt, 5, pv, 1, &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(ctr_db, e, 0, DB_MODE_RW);
  }

  /* 3) Create with pre-reserved GUID: paramv = [coll, ctr, reserved_guid]. */
  {
    arts_guid_t reserved = arts_guid_reserve(ARTS_GUID_EDT, 0);
    uint64_t pv[3] = {base[0], base[1], (uint64_t)reserved};
    arts_guid_t e = arts_edt_create(guid_edt, 3, pv, 1,
                                    &(arts_edt_hint_t){.guid = reserved});
    arts_add_dependence(ctr_db, e, 0, DB_MODE_RW);
  }

  /* 4) Create in a finish scope: paramv = [coll, ctr]. */
  {
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t e =
        arts_edt_create(scope_edt, 2, base, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ctr_db, e, 0, DB_MODE_RW);
  }

  /* 5) has_depv=true: depv[0]=counter_db(RW), depv[1]=test DB(RO). */
  {
    void *db_ptr = NULL;
    arts_guid_t test_db =
        arts_db_create(&db_ptr, 64, ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
    arts_guid_t e =
        arts_edt_create(dep_true_edt, 2, base, 2, &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(ctr_db, e, 0, DB_MODE_RW);
    arts_add_dependence(test_db, e, 1, DB_MODE_RO);
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
