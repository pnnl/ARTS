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

/// @file multinode_coherence.c
/// @brief Tests RW/RO ordering across nodes: sequential RW writers on different
///        nodes, concurrent RO readers, and RW ping-pong.  Requires multi-node
///        (node_count > 1).
///
/// Cross-node RW grant order is NOT implied by dependence-registration order —
/// the runtime serializes same-DB RW but does not impose a global cross-node
/// sequence (an unordered program is racy by design).  So each stage that must
/// observe the previous stage's write is ordered EXPLICITLY by gating it on the
/// previous EDT's finish event; only then is the final value deterministic and
/// worth asserting.

#include "arts.h"

/// EW writer: writes paramv[0] into data[0].
void coh_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  int value = (int)paramv[0];
  if (data) {
    data[0] = value;
  }
}

/// RO reader: asserts data[0] == paramv[0].
void coh_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  int expected = (int)paramv[0];
  unsigned int test_id = (unsigned int)paramv[1];
  bool ok = (data != NULL && data[0] == expected);
  if (ok) {
    arts_printf("  PASS: test %u reader saw %d\n", test_id, expected);
  } else {
    arts_printf("  FAIL: test %u reader expected %d got %d\n", test_id,
                expected, data ? data[0] : -1);
  }
}

/// Test 2: RO reader that checks a 3-element array.
void coh_reader_3(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  unsigned int reader_id = (unsigned int)paramv[0];
  bool ok = (data != NULL && data[0] == 42 && data[1] == 84 && data[2] == 126);
  if (ok) {
    arts_printf("  PASS: test 2 reader %u saw {42, 84, 126}\n", reader_id);
  } else {
    arts_printf("  FAIL: test 2 reader %u data mismatch {%d, %d, %d}\n",
                reader_id, data ? data[0] : -1, data ? data[1] : -1,
                data ? data[2] : -1);
  }
}

/// Test 2: EW writer that writes {42, 84, 126}.
void coh_writer_3(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = 42;
    data[1] = 84;
    data[2] = 126;
  }
}

/// Test 3: EW writer that increments data[0].
void coh_incrementer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = data[0] + 1;
  }
}

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== multinode_coherence ===\n");

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  // Test 1: Cross-node EW chain.
  // writer1(EW, node 0) writes 100 -> writer2(EW, node 1) writes 200
  // -> reader(RO, node 0) asserts 200.
  {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((int *)ptr)[0] = 0;
    arts_db_release(db, DB_MODE_RW);

    arts_guid_t e1 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t val1 = 100;
    arts_guid_t w1 =
        arts_edt_create(coh_writer, 1, &val1, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = e1});
    arts_add_dependence(db, w1, 0, DB_MODE_RW);

    arts_guid_t e2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t val2 = 200;
    arts_guid_t w2 =
        arts_edt_create(coh_writer, 1, &val2, 2,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = e2});
    arts_add_dependence(db, w2, 0, DB_MODE_RW);
    arts_add_dependence(e1, w2, 1, DB_MODE_NULL); /* w2 after w1 completes */

    uint64_t rparams[2] = {200, 1};
    arts_guid_t r =
        arts_edt_create(coh_reader, 2, rparams, 2,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, r, 0, DB_MODE_RO);
    arts_add_dependence(e2, r, 1, DB_MODE_NULL); /* r after w2 completes */
  }

  // Test 2: Cross-node concurrent RO readers after remote EW writer.
  // writer(EW, node 1) writes {42, 84, 126}
  // -> reader_a(RO, node 0) and reader_b(RO, node 1) both verify.
  {
    void *ptr2 = NULL;
    arts_guid_t db2 =
        arts_db_create(&ptr2, 3 * sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((int *)ptr2)[0] = 0;
    ((int *)ptr2)[1] = 0;
    ((int *)ptr2)[2] = 0;
    arts_db_release(db2, DB_MODE_RW);

    arts_guid_t ew = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w =
        arts_edt_create(coh_writer_3, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = ew});
    arts_add_dependence(db2, w, 0, DB_MODE_RW);

    /* Both readers gated on the writer's completion so they observe its data,
     * but run concurrently with each other (shared RO). */
    uint64_t id0 = 0;
    arts_guid_t ra =
        arts_edt_create(coh_reader_3, 1, &id0, 2,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db2, ra, 0, DB_MODE_RO);
    arts_add_dependence(ew, ra, 1, DB_MODE_NULL);

    uint64_t id1 = 1;
    arts_guid_t rb =
        arts_edt_create(coh_reader_3, 1, &id1, 2,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    arts_add_dependence(db2, rb, 0, DB_MODE_RO);
    arts_add_dependence(ew, rb, 1, DB_MODE_NULL);
  }

  // Test 3: EW ping-pong across nodes.
  // inc_a(EW, node 0, +1) -> inc_b(EW, node 1, +1)
  // -> inc_c(EW, node 0, +1) -> reader(RO, node 0) asserts value == 3.
  {
    void *ptr3 = NULL;
    arts_guid_t db3 =
        arts_db_create(&ptr3, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((int *)ptr3)[0] = 0;
    arts_db_release(db3, DB_MODE_RW);

    arts_guid_t ea = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t ia =
        arts_edt_create(coh_incrementer, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = ea});
    arts_add_dependence(db3, ia, 0, DB_MODE_RW);

    arts_guid_t eb = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t ib =
        arts_edt_create(coh_incrementer, 0, NULL, 2,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = eb});
    arts_add_dependence(db3, ib, 0, DB_MODE_RW);
    arts_add_dependence(ea, ib, 1, DB_MODE_NULL); /* ib after ia */

    arts_guid_t ec = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t ic =
        arts_edt_create(coh_incrementer, 0, NULL, 2,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = ec});
    arts_add_dependence(db3, ic, 0, DB_MODE_RW);
    arts_add_dependence(eb, ic, 1, DB_MODE_NULL); /* ic after ib */

    uint64_t rparams3[2] = {3, 3};
    arts_guid_t r3 =
        arts_edt_create(coh_reader, 2, rparams3, 2,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db3, r3, 0, DB_MODE_RO);
    arts_add_dependence(ec, r3, 1, DB_MODE_NULL); /* r3 after ic */
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
