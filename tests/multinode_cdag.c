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

/// @file multinode_cdag.c
/// @brief Tests CDAG EW/RO ordering across nodes: sequential EW writers on
///        different nodes, concurrent RO readers, and EW ping-pong.
///        Requires multi-node (node_count > 1).

#include "arts.h"

/// EW writer: writes paramv[0] into data[0].
void cdag_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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
void cdag_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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
void cdag_reader_3(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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
void cdag_writer_3(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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
void cdag_incrementer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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

  arts_printf("=== multinode_cdag ===\n");

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t epoch = arts_initialize_and_start_epoch(shut, 0);

  // Test 1: Cross-node EW chain.
  // writer1(EW, node 0) writes 100 -> writer2(EW, node 1) writes 200
  // -> reader(RO, node 0) asserts 200.
  {
    void *ptr = NULL;
    arts_guid_t db = arts_db_create(&ptr, sizeof(int), ARTS_DB_DEFAULT,
                                    &(arts_hint_t){.route = 0});
    ((int *)ptr)[0] = 0;
    arts_db_release(db);

    uint64_t val1 = 100;
    arts_guid_t w1 = arts_edt_create_with_epoch(cdag_writer, 1, &val1, 1, epoch,
                                                &(arts_hint_t){.route = 0});
    arts_record_dep(db, w1, 0, DB_MODE_EW);

    uint64_t val2 = 200;
    arts_guid_t w2 = arts_edt_create_with_epoch(cdag_writer, 1, &val2, 1, epoch,
                                                &(arts_hint_t){.route = 1});
    arts_record_dep(db, w2, 0, DB_MODE_EW);

    uint64_t rparams[2] = {200, 1};
    arts_guid_t r = arts_edt_create_with_epoch(
        cdag_reader, 2, rparams, 1, epoch, &(arts_hint_t){.route = 0});
    arts_record_dep(db, r, 0, DB_MODE_RO);
  }

  // Test 2: Cross-node concurrent RO readers after remote EW writer.
  // writer(EW, node 1) writes {42, 84, 126}
  // -> reader_a(RO, node 0) and reader_b(RO, node 1) both verify.
  {
    void *ptr2 = NULL;
    arts_guid_t db2 = arts_db_create(&ptr2, 3 * sizeof(int), ARTS_DB_DEFAULT,
                                     &(arts_hint_t){.route = 0});
    ((int *)ptr2)[0] = 0;
    ((int *)ptr2)[1] = 0;
    ((int *)ptr2)[2] = 0;
    arts_db_release(db2);

    arts_guid_t w = arts_edt_create_with_epoch(cdag_writer_3, 0, NULL, 1, epoch,
                                               &(arts_hint_t){.route = 1});
    arts_record_dep(db2, w, 0, DB_MODE_EW);

    uint64_t id0 = 0;
    arts_guid_t ra = arts_edt_create_with_epoch(
        cdag_reader_3, 1, &id0, 1, epoch, &(arts_hint_t){.route = 0});
    arts_record_dep(db2, ra, 0, DB_MODE_RO);

    uint64_t id1 = 1;
    arts_guid_t rb = arts_edt_create_with_epoch(
        cdag_reader_3, 1, &id1, 1, epoch, &(arts_hint_t){.route = 1});
    arts_record_dep(db2, rb, 0, DB_MODE_RO);
  }

  // Test 3: EW ping-pong across nodes.
  // inc_a(EW, node 0, +1) -> inc_b(EW, node 1, +1)
  // -> inc_c(EW, node 0, +1) -> reader(RO, node 0) asserts value == 3.
  {
    void *ptr3 = NULL;
    arts_guid_t db3 = arts_db_create(&ptr3, sizeof(int), ARTS_DB_DEFAULT,
                                     &(arts_hint_t){.route = 0});
    ((int *)ptr3)[0] = 0;
    arts_db_release(db3);

    arts_guid_t ia = arts_edt_create_with_epoch(
        cdag_incrementer, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
    arts_record_dep(db3, ia, 0, DB_MODE_EW);

    arts_guid_t ib = arts_edt_create_with_epoch(
        cdag_incrementer, 0, NULL, 1, epoch, &(arts_hint_t){.route = 1});
    arts_record_dep(db3, ib, 0, DB_MODE_EW);

    arts_guid_t ic = arts_edt_create_with_epoch(
        cdag_incrementer, 0, NULL, 1, epoch, &(arts_hint_t){.route = 0});
    arts_record_dep(db3, ic, 0, DB_MODE_EW);

    uint64_t rparams3[2] = {3, 3};
    arts_guid_t r3 = arts_edt_create_with_epoch(
        cdag_reader, 2, rparams3, 1, epoch, &(arts_hint_t){.route = 0});
    arts_record_dep(db3, r3, 0, DB_MODE_RO);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
