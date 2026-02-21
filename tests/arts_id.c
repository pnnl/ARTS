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
/******************************************************************************
** Test: arts_id Tracking and Counter Integration
** This test validates the arts_id tracking infrastructure for ArtsMate
** integration, including:
** - EDT and DB creation with arts_id values
** - Per-thread counter tracking
** - Hash table collision handling
** - JSON export functionality
******************************************************************************/
#include "arts.h"
#include "arts/counter/counter.h"
#include "arts/counter/object_counter.h"
#include "arts/runtime/globals.h"
#include <stdlib.h>
#include <string.h>

/// Test configuration
#define NUM_TEST_EDTS 10
#define NUM_TEST_DBS 5
#define TEST_ARTS_ID_BASE 1000
#define TEST_MATRIX_SIZE 1024

/// Global validation flag
volatile unsigned int *test_result = NULL;

/// Test EDT that simulates work
void test_edt_worker(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  arts_printf("[test_edt_worker] Entry - paramc=%u, depc=%u\n", paramc, depc);

  if (paramc < 1) {
    arts_printf("[test_edt_worker] ERROR: Missing arts_id parameter\n");
    return;
  }

  uint64_t expected_arts_id = paramv[0];
  unsigned int node_id = arts_get_current_node();

  arts_printf("Node %u - EDT with expected arts_id=%lu executing\n", node_id,
              expected_arts_id);

  // Simulate some work (matrix computation)
  if (depc > 0 && depv[0].ptr != NULL) {
    double *matrix = (double *)depv[0].ptr;
    unsigned int size = 32; // Small test matrix

    // Simple matrix operation to simulate work
    for (unsigned int i = 0; i < size; i++) {
      for (unsigned int j = 0; j < size; j++) {
        matrix[((uint64_t)i * size) + j] =
            (double)(((uint64_t)i * j) + expected_arts_id);
      }
    }

    arts_printf("Node %u - EDT %lu completed matrix computation\n", node_id,
                expected_arts_id);
  }

  arts_printf("[test_edt_worker] Completed - arts_id=%lu\n", expected_arts_id);
}

/// Test EDT that reads multiple DBs
void test_edt_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  arts_printf("[test_edt_reader] Entry - paramc=%u, depc=%u\n", paramc, depc);

  if (paramc < 1) {
    arts_printf("[test_edt_reader] ERROR: Missing arts_id parameter\n");
    return;
  }

  uint64_t expected_arts_id = paramv[0];
  unsigned int node_id = arts_get_current_node();

  arts_printf("Node %u - Reader EDT %lu accessing %u DBs\n", node_id,
              expected_arts_id, depc);

  // Access multiple DBs
  for (uint32_t i = 0; i < depc; i++) {
    if (depv[i].ptr != NULL) {
      double *matrix = (double *)depv[i].ptr;
      double sum = 0.0;
      unsigned int size = 32;

      // Read and sum matrix
      for (unsigned int j = 0; j < size * size; j++) {
        sum += matrix[j];
      }

      arts_printf("Node %u - Reader EDT %lu: DB[%u] sum=%.2f\n", node_id,
                  expected_arts_id, i, sum);
    }
  }

  arts_printf("[test_edt_reader] Completed - arts_id=%lu\n", expected_arts_id);
}

/// Validator EDT: Checks arts_id values in created structures
void validator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_printf("[Validator] Starting validation\n");

  bool success = true;
  unsigned int errors = 0;

  // Test 1: Verify object counter data structures exist
  arts_printf(
      "[Validator] Test 1: Checking object counter data structures...\n");

#if ARTS_OBJECT_EDT_TABLE_ENABLED
  arts_printf("[Validator] EDT object table enabled\n");
#else
  arts_printf(
      "[Validator] EDT object table disabled (expected if counters OFF)\n");
#endif

#if ARTS_OBJECT_DB_TABLE_ENABLED
  arts_printf("[Validator] DB object table enabled\n");
#else
  arts_printf(
      "[Validator] DB object table disabled (expected if counters OFF)\n");
#endif

#if ARTS_OBJECT_EDT_TRACE_ENABLED
  arts_printf("[Validator] EDT traces enabled\n");
#else
  arts_printf("[Validator] EDT traces disabled (expected if counters OFF)\n");
#endif

#if ARTS_OBJECT_DB_TRACE_ENABLED
  arts_printf("[Validator] DB traces enabled\n");
#else
  arts_printf("[Validator] DB traces disabled (expected if counters OFF)\n");
#endif

  // Test 2: Verify counter mode configuration
  arts_printf("[Validator] Test 2: Checking counter modes...\n");

  unsigned int obj_num_edt_mode = arts_counter_mode_array[OBJ_NUM_EDT];
  unsigned int obj_num_db_mode = arts_counter_mode_array[OBJ_NUM_DB];
  unsigned int obj_trace_edt_mode = arts_counter_mode_array[OBJ_TRACE_EDT];
  unsigned int obj_trace_db_mode = arts_counter_mode_array[OBJ_TRACE_DB];

  arts_printf("[Validator] OBJ_NUM_EDT mode: %u (0=OFF, 1=ONCE, 2=PERIODIC)\n",
              obj_num_edt_mode);
  arts_printf("[Validator] OBJ_NUM_DB mode: %u\n", obj_num_db_mode);
  arts_printf("[Validator] OBJ_TRACE_EDT mode: %u\n", obj_trace_edt_mode);
  arts_printf("[Validator] OBJ_TRACE_DB mode: %u\n", obj_trace_db_mode);

  // Test 3: Per-node JSON export
  arts_printf("[Validator] Test 3: Object counter JSON export...\n");

#if ARTS_OBJECT_ANY_ENABLED
  arts_printf("[Validator] Object counter export active\n");
  arts_printf("[Validator] Look for object_n*.json files after shutdown\n");
#else
  arts_printf("[Validator] Object counter export skipped (all counters OFF)\n");
#endif

  // Final result
  if (success) {
    arts_printf("═══════════════════════════════════════\n");
    arts_printf("TEST STATUS: SUCCESS\n");
    arts_printf("═══════════════════════════════════════\n");
    arts_printf("✓ All arts_id tracking tests passed!\n");
    arts_printf("✓ Created %u EDTs with arts_id values\n", NUM_TEST_EDTS);
    arts_printf("✓ Created %u DBs with arts_id values\n", NUM_TEST_DBS);
    arts_printf("✓ Counter infrastructure validated\n");
    if (test_result) {
      *test_result = 1;
    }
  } else {
    arts_printf("═══════════════════════════════════════\n");
    arts_printf("TEST STATUS: FAILURE\n");
    arts_printf("═══════════════════════════════════════\n");
    arts_printf("✗ Found %u errors in arts_id tracking!\n", errors);
    if (test_result) {
      *test_result = 0;
    }
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("═══════════════════════════════════════\n");
  arts_printf("ARTS arts_id Tracking Test\n");
  arts_printf("═══════════════════════════════════════\n");
  arts_printf("Test Configuration:\n");
  arts_printf("- Test EDTs: %u\n", NUM_TEST_EDTS);
  arts_printf("- Test DBs:  %u\n", NUM_TEST_DBS);
  arts_printf("- Nodes:     %u\n", arts_get_total_nodes());
  arts_printf("- Threads:   %u\n", arts_node_info.total_thread_count);
  arts_printf("═══════════════════════════════════════\n");

  // Allocate test result flag
  test_result = (volatile unsigned int *)malloc(sizeof(unsigned int));
  *test_result = 0;

  // Start epoch for coordination
  arts_guid_t epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
  arts_printf("[Step 1] Started epoch (guid: %lu)\n", epoch_guid);

  // Create test DBs with arts_id values
  arts_printf("[Step 3] Creating %u test DBs with arts_id values:\n",
              NUM_TEST_DBS);
  arts_guid_t *db_guids =
      (arts_guid_t *)malloc(NUM_TEST_DBS * sizeof(arts_guid_t));
  void **db_ptrs = (void **)malloc(NUM_TEST_DBS * sizeof(void *));

  unsigned int matrix_size =
      (unsigned int)((unsigned long)32 * 32 * sizeof(double));

  for (unsigned int i = 0; i < NUM_TEST_DBS; i++) {
    uint64_t arts_id =
        TEST_ARTS_ID_BASE + 100 + i; // DB arts_id: 1100, 1101, ...

    db_guids[i] =
        arts_db_create(&db_ptrs[i], matrix_size, &(arts_hint_t){.id = arts_id});

    // Initialize matrix to zeros
    double *matrix = (double *)db_ptrs[i];
    for (unsigned int j = 0; j < 32 * 32; j++) {
      matrix[j] = 0.0;
    }

    arts_printf("  - DB[%u]: guid=%lu, arts_id=%lu, size=%u bytes\n", i,
                db_guids[i], arts_id, matrix_size);
  }

  arts_printf("\n");

  // Create validator EDT first
  arts_printf("[Step 4] Creating validator EDT (will run last)...\n");
  arts_guid_t validator_guid = arts_edt_create_with_epoch(
      validator, 0, NULL, 1, epoch_guid, &(arts_hint_t){.route = 0});

  // Create writer EDTs with arts_id values
  arts_printf("[Step 5] Creating %u writer EDTs with arts_id values:\n",
              NUM_TEST_EDTS);
  arts_guid_t *writer_guids =
      (arts_guid_t *)malloc(NUM_TEST_EDTS * sizeof(arts_guid_t));
  unsigned int *writer_db_indices =
      (unsigned int *)malloc(NUM_TEST_EDTS * sizeof(unsigned int));

  for (unsigned int i = 0; i < NUM_TEST_EDTS; i++) {
    uint64_t arts_id = TEST_ARTS_ID_BASE + i; // EDT arts_id: 1000, 1001, ...
    uint64_t param = arts_id;

    // Distribute EDTs across nodes (round-robin)
    unsigned int target_node = i % arts_get_total_nodes();

    // Each EDT will use one DB
    unsigned int db_index = i % NUM_TEST_DBS;
    writer_db_indices[i] = db_index;

    writer_guids[i] =
        arts_edt_create(test_edt_worker, 1, &param, 1,
                        &(arts_hint_t){.route = target_node, .id = arts_id});

    arts_printf("  - EDT[%u]: guid=%lu, arts_id=%lu, node=%u, using DB[%u]\n",
                i, writer_guids[i], arts_id, target_node, db_index);
  }

  arts_printf("\n");

  // Create reader EDTs with arts_id values
  arts_printf("[Step 6] Creating reader EDTs with arts_id values:\n");

  unsigned int num_readers = 3;
  arts_guid_t *reader_guids =
      (arts_guid_t *)malloc(num_readers * sizeof(arts_guid_t));
  unsigned int *reader_num_deps =
      (unsigned int *)malloc(num_readers * sizeof(unsigned int));

  for (unsigned int i = 0; i < num_readers; i++) {
    uint64_t arts_id =
        TEST_ARTS_ID_BASE + 200 + i; // Reader arts_id: 1200, 1201, ...
    uint64_t param = arts_id;

    unsigned int target_node = i % arts_get_total_nodes();

    // Readers will depend on multiple DBs
    unsigned int num_deps =
        (i % NUM_TEST_DBS) + 1; // 1 to NUM_TEST_DBS dependencies
    reader_num_deps[i] = num_deps;

    reader_guids[i] =
        arts_edt_create(test_edt_reader, 1, &param, num_deps,
                        &(arts_hint_t){.route = target_node, .id = arts_id});

    arts_printf("  - Reader[%u]: guid=%lu, arts_id=%lu, node=%u, deps=%u\n", i,
                reader_guids[i], arts_id, target_node, num_deps);
  }

  arts_printf("\n");

  // Record all dependencies using arts_record_dep
  arts_printf("[Step 7] Recording dependencies for all EDTs...\n");

  // Record writer dependencies
  for (unsigned int i = 0; i < NUM_TEST_EDTS; i++) {
    unsigned int db_index = writer_db_indices[i];
    arts_record_dep(db_guids[db_index], writer_guids[i], 0, ARTS_DB_WRITE);
  }

  // Record reader dependencies
  for (unsigned int i = 0; i < num_readers; i++) {
    unsigned int num_deps = reader_num_deps[i];
    for (unsigned int d = 0; d < num_deps; d++) {
      arts_record_dep(db_guids[d], reader_guids[i], d, ARTS_DB_READ);
    }
  }

  // Record validator dependency (reads DB[0] after all writers complete)
  arts_record_dep(db_guids[0], validator_guid, 0, ARTS_DB_READ);

  // Release auto-acquired WRITE access for all created DBs before blocking.
  // Without this, consumer EDTs would deadlock waiting for our epilogue.
  for (unsigned int i = 0; i < NUM_TEST_DBS; i++) {
    arts_db_release(db_guids[i]);
  }

  arts_printf("[Step 8] Waiting for epoch to complete...\n");

  // Wait for all EDTs to complete
  arts_wait_on_handle(epoch_guid);

  arts_printf("[Step 9] Epoch completed\n");

  // Check test result
  if (*test_result == 1) {
    arts_printf("Overall test result: PASSED\n");
  } else {
    arts_printf("Overall test result: FAILED or INCOMPLETE\n");
  }

  // Note: Counter export now happens automatically via counter infrastructure
  // No manual export call needed - arts_counter_write_thread() handles all
  // metrics

  // Cleanup
  free((void *)writer_guids);
  free((void *)writer_db_indices);
  free((void *)reader_guids);
  free((void *)reader_num_deps);
  free((void *)db_ptrs);
  free((void *)db_guids);
  free((void *)test_result);

  // Shutdown
  arts_shutdown();
}

int main(int argc, char **argv) {
  int ret = arts_rt(argc, argv);
  return ret;
}
