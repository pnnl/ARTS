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

/// @file multinode_db_advanced.c
/// @brief Tests advanced DB operations across nodes: create_with_guid + data,
///        signal_edt_ptr cross-node, and db_rename cross-node access.
///        Requires multi-node (node_count > 1).

#include "arts.h"
#include <string.h>

#define DB_ELEMS 8
#define PTR_SIZE 256

// ---------------------------------------------------------------------------
// Test 1: arts_db_create_with_guid + initial data on remote node.
// ---------------------------------------------------------------------------

/// Reader on node 1: verifies initial data pattern.
void check_create_with_data(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL);
  for (int i = 0; i < DB_ELEMS && ok; i++) {
    if (data[i] != i * 7) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: create_with_guid + data on remote node\n");
  } else {
    arts_printf("  FAIL: create_with_guid data mismatch\n");
  }
}

// ---------------------------------------------------------------------------
// Test 2: arts_signal_edt_ptr cross-node data verification.
// ---------------------------------------------------------------------------

/// Remote EDT: verifies byte pattern.
void check_signal_ptr(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int size = (unsigned int)paramv[0];
  uint8_t *data = (uint8_t *)depv[0].ptr;
  bool ok = (data != NULL);
  for (unsigned int i = 0; i < size && ok; i++) {
    if (data[i] != (uint8_t)(i & 0xFF)) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: cross-node signal_edt_ptr %u bytes\n", size);
  } else {
    arts_printf("  FAIL: cross-node signal_edt_ptr data mismatch\n");
  }
}

// ---------------------------------------------------------------------------
// Test 3: arts_db_rename cross-node access.
// ---------------------------------------------------------------------------

/// Reader on node 1: verifies renamed DB data.
void check_rename_get(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  bool ok = (data != NULL);
  for (int i = 0; i < DB_ELEMS && ok; i++) {
    if (data[i] != i * 13) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("  PASS: db_rename cross-node access\n");
  } else {
    arts_printf("  FAIL: db_rename data mismatch\n");
  }
}

// ---------------------------------------------------------------------------

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

  arts_printf("=== multinode_db_advanced ===\n");

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t epoch = arts_initialize_and_start_epoch(shut, 0);

  // Test 1: Create DB locally with GUID + initial data, read from remote node.
  // arts_db_create_with_guid is local-only, so create on node 0 and get
  // from node 1 via arts_get_from_db.
  {
    int init_data[DB_ELEMS];
    for (int i = 0; i < DB_ELEMS; i++) {
      init_data[i] = i * 7;
    }
    arts_guid_t reserved = arts_guid_reserve(ARTS_DB, 0);
    arts_db_create_with_guid(reserved, DB_ELEMS * sizeof(int), ARTS_DB_DEFAULT,
                             init_data, NULL);
    arts_db_release(reserved);
    arts_guid_t reader = arts_edt_create_with_epoch(
        check_create_with_data, 0, NULL, 1, epoch, &(arts_hint_t){.route = 1});
    arts_get_from_db(reader, reserved, 0, 0, DB_ELEMS * sizeof(int));
  }

  // Test 2: Signal ptr (256 bytes) from node 0 to EDT on node 1.
  {
    uint8_t buf[PTR_SIZE];
    for (unsigned int i = 0; i < PTR_SIZE; i++) {
      buf[i] = (uint8_t)(i & 0xFF);
    }
    uint64_t size_param = PTR_SIZE;
    arts_guid_t reader = arts_edt_create_with_epoch(
        check_signal_ptr, 1, &size_param, 1, epoch, &(arts_hint_t){.route = 1});
    arts_signal_edt_ptr(reader, 0, buf, PTR_SIZE);
  }

  // Test 3: Create DB on node 0, fill, rename, then get from node 1.
  {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, DB_ELEMS * sizeof(int), ARTS_DB_DEFAULT,
                       &(arts_hint_t){.route = 0});
    int *data = (int *)ptr;
    for (int i = 0; i < DB_ELEMS; i++) {
      data[i] = i * 13;
    }
    arts_db_release(db);

    arts_guid_t renamed = arts_db_rename(db);
    arts_db_release(renamed);

    arts_guid_t reader = arts_edt_create_with_epoch(
        check_rename_get, 0, NULL, 1, epoch, &(arts_hint_t){.route = 1});
    arts_get_from_db(reader, renamed, 0, 0, DB_ELEMS * sizeof(int));
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
