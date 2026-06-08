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
#include <stdlib.h>

/// Configurable test parameters
unsigned int array_size = 1024 * 1024;
unsigned int num_readers = 16;

arts_guid_t data_guid;
volatile unsigned int *validation_result = NULL;

/// Writer EDT: Initializes the array (WRITE mode)
void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  unsigned int node_id = arts_get_current_rank();
  uint64_t *data = (uint64_t *)depv[0].ptr;

  arts_printf("Writer (Node %u): Initializing array with sequential values\n",
              node_id);

  // Initialize array with i * 7
  for (unsigned int i = 0; i < array_size; i++) {
    data[i] = (uint64_t)i * 7;
  }

  arts_printf("Writer (Node %u): Completed initialization of %u elements\n",
              node_id, array_size);
  // dec_latch happens automatically on release (WRITE mode)
}

/// Reader EDT: Reads and validates a portion of the array (READ mode override)
void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  unsigned int reader_id = (unsigned int)paramv[0];
  unsigned int node_id = arts_get_current_rank();
  uint64_t *data = (uint64_t *)depv[0].ptr;

  arts_printf("Reader %u (Node %u): Reading array (mode=READ)\n", reader_id,
              node_id);

  // Read and validate a stripe of the array
  unsigned int start_idx = reader_id * (array_size / num_readers);
  unsigned int end_idx = (reader_id + 1) * (array_size / num_readers);

  uint64_t sum = 0;
  unsigned int errors = 0;

  for (unsigned int i = start_idx; i < end_idx; i++) {
    uint64_t expected = (uint64_t)i * 7;
    uint64_t actual = data[i];

    if (actual != expected) {
      if (errors < 3) {
        arts_printf("Reader %u ERROR at index %u: expected %lu, got %lu\n",
                    reader_id, i, expected, actual);
      }
      errors++;
    }
    sum += actual;
  }

  if (errors == 0) {
    arts_printf("Reader %u (Node %u): SUCCESS - all values correct (sum=%lu)\n",
                reader_id, node_id, sum);
  } else {
    arts_printf("Reader %u (Node %u): FAILURE - %u errors found\n", reader_id,
                node_id, errors);
  }
}

/// Validator EDT: Final check after all readers (READ mode override)
void validator_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  unsigned int errors = 0;

  arts_printf("=== Validator: Performing final validation ===\n");

  // Full array validation
  for (unsigned int i = 0; i < array_size; i++) {
    uint64_t expected = (uint64_t)i * 7;
    uint64_t actual = data[i];

    if (actual != expected) {
      if (errors < 5) {
        arts_printf("Validator ERROR at index %u: expected %lu, got %lu\n", i,
                    expected, actual);
      }
      errors++;
    }
  }

  if (errors == 0) {
    arts_printf("=== ACQUIRE-MODE TEST PASSED ===\n");
    arts_printf("SUCCESS: All %u elements validated correctly!\n", array_size);
    arts_printf("All %u readers completed with READ mode (no owner updates).\n",
                num_readers);
    if (validation_result) {
      *validation_result = 1;
    }
  } else {
    arts_printf("=== ACQUIRE-MODE TEST FAILED ===\n");
    arts_printf("FAILURE: Found %u errors!\n", errors);
    if (validation_result) {
      *validation_result = 0;
    }
  }

  arts_printf("Acquire-Mode Test Complete\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  /// Parse command line arguments for parametric testing
  /// Usage: ./testAcquireModeCompilerPattern [array_size] [num_readers]
  ///  - Parse array_size
  if (argc > 1 && argv[1]) {
    array_size = (unsigned int)strtol(argv[1], NULL, 10);
  }
  ///  - Parse num_readers
  if (argc > 2 && argv[2]) {
    num_readers = (unsigned int)strtol(argv[2], NULL, 10);
  }

  /// Validate parameters
  if (array_size == 0 || num_readers == 0) {
    arts_printf("ERROR: array_size (%u) and num_readers (%u) must be > 0\n",
                array_size, num_readers);
    arts_shutdown();
    return;
  }
  /// Validate that array_size is divisible by num_readers
  if (array_size % num_readers != 0) {
    arts_printf(
        "ERROR: array_size (%u) must be divisible by num_readers (%u)\n",
        array_size, num_readers);
    arts_shutdown();
    return;
  }

  arts_printf("Acquire-Mode Test\n");
  arts_printf("- Array:   %u elements (%zu MB)\n", array_size,
              ((unsigned long)array_size * sizeof(uint64_t)) /
                  (1024UL * 1024UL));
  arts_printf("- Readers: %u concurrent reader EDTs\n", num_readers);
  arts_printf("- Nodes:   %u\n", arts_get_total_ranks());

  /// Reserve GUID for data DB

  /// Create data DB and initialize to zeros
  data_guid = arts_guid_reserve(ARTS_GUID_DB, 0);
  size_t db_size = array_size * sizeof(uint64_t);
  arts_printf("Creating Data DB (guid: %lu, size: %zu bytes = %.2lf MB)\n",
              data_guid, db_size, (double)db_size / (1024.0 * 1024.0));
  uint64_t *data_ptr = (uint64_t *)arts_db_create_with_guid(
      data_guid, db_size, ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  for (size_t i = 0; i < array_size; i++) {
    data_ptr[i] = 0;
  }
  arts_printf("Data DB initialized to zeros\n");

  /// Allocate validation result flag (shared with validator)
  validation_result = (volatile unsigned int *)malloc(sizeof(unsigned int));
  *validation_result = 0;

  /// Start finish scope
  arts_guid_t fe_guid = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_printf("[Step 1] Started finish scope (guid: %lu)\n", fe_guid);

  /// Create writer EDT
  arts_guid_t writer_edt_guid = arts_edt_create(writer_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe_guid});
  arts_printf("[Step 2] Created writer EDT (guid: %lu)\n", writer_edt_guid);

  /// Create reader EDTs
  arts_guid_t *reader_edt_guids =
      (arts_guid_t *)calloc(num_readers, sizeof(arts_guid_t));
  for (unsigned int i = 0; i < num_readers; i++) {
    unsigned int target_node = (i % arts_get_total_ranks());
    uint64_t param = i;
    reader_edt_guids[i] =
        arts_edt_create(reader_edt, 1, &param, 1, &(arts_edt_hint_t){.rank = target_node, .finish_event = fe_guid});

    if ((i + 1) % 4 == 0 || i == num_readers - 1) {
      unsigned int range_start = (i / 4) * 4;
      unsigned int range_end = i;
      arts_printf("Created readers %u-%u\n", range_start, range_end);
    }
  }

  arts_guid_t validator_edt_guid = arts_edt_create(validator_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe_guid});
  arts_printf("[Step 4] Created validator EDT (guid: %lu)\n",
              validator_edt_guid);

  /// Record ALL dependencies
  arts_printf(
      "[Step 5] Recording dependencies: 1 writer (WRITE) + %u readers \n"
      "(READ) + 1 validator (READ)",
      num_readers);
  arts_add_dependence(data_guid, writer_edt_guid, 0, DB_MODE_RW);
  for (unsigned int i = 0; i < num_readers; i++) {
    arts_add_dependence(data_guid, reader_edt_guids[i], 0, DB_MODE_RO);
  }

  arts_add_dependence(data_guid, validator_edt_guid, 0, DB_MODE_RO);

  arts_printf("  All dependencies recorded (latch=1, only writer)\n");

  // Release auto-acquired WRITE access before blocking on finish scope.
  arts_db_release(data_guid, DB_MODE_RW);

  /// Wait for finish scope to complete
  arts_printf("[Step 6] Waiting for finish scope to complete\n");
  arts_event_wait(fe_guid);

  /// Free reader EDTs
  free(reader_edt_guids);

  /// Print final status
  if (validation_result && *validation_result == 1) {
    arts_printf("TEST STATUS: SUCCESS\n");
    arts_printf("Acquire-mode override successfully reduced owner updates.\n");
    arts_printf(
        "Expected: 1 writer update + 0 reader updates = ~%u%% reduction\n",
        (100 * num_readers) / (num_readers + 1));
    arts_printf("(vs %u updates if all were WRITE mode)\n", num_readers + 1);
  } else {
    arts_printf("TEST STATUS: FAILURE\n");
  }

  /// Shutdown runtime
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
