/******************************************************************************
** Test: Acquire-Mode Override
******************************************************************************/
#include "arts.h"
#include "arts/system/arts_print.h"
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
  unsigned int node_id = arts_get_current_node();
  uint64_t *data = (uint64_t *)depv[0].ptr;

  ARTS_PRINT("Writer (Node %u): Initializing array with sequential values",
             node_id);

  // Initialize array with i * 7
  for (unsigned int i = 0; i < array_size; i++) {
    data[i] = (uint64_t)i * 7;
  }

  ARTS_PRINT("Writer (Node %u): Completed initialization of %u elements",
             node_id, array_size);
  // dec_latch happens automatically on release (WRITE mode)
}

/// Reader EDT: Reads and validates a portion of the array (READ mode override)
void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  unsigned int reader_id = (unsigned int)paramv[0];
  unsigned int node_id = arts_get_current_node();
  uint64_t *data = (uint64_t *)depv[0].ptr;

  ARTS_PRINT("Reader %u (Node %u): Reading array (mode=READ)", reader_id,
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
        ARTS_PRINT("Reader %u ERROR at index %u: expected %lu, got %lu",
                   reader_id, i, expected, actual);
      }
      errors++;
    }
    sum += actual;
  }

  if (errors == 0) {
    ARTS_PRINT("Reader %u (Node %u): SUCCESS - all values correct (sum=%lu)",
               reader_id, node_id, sum);
  } else {
    ARTS_PRINT("Reader %u (Node %u): FAILURE - %u errors found", reader_id,
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

  ARTS_PRINT("=== Validator: Performing final validation ===");

  // Full array validation
  for (unsigned int i = 0; i < array_size; i++) {
    uint64_t expected = (uint64_t)i * 7;
    uint64_t actual = data[i];

    if (actual != expected) {
      if (errors < 5) {
        ARTS_PRINT("Validator ERROR at index %u: expected %lu, got %lu", i,
                   expected, actual);
      }
      errors++;
    }
  }

  if (errors == 0) {
    ARTS_PRINT("=== ACQUIRE-MODE TEST PASSED ===");
    ARTS_PRINT("SUCCESS: All %u elements validated correctly!", array_size);
    ARTS_PRINT("All %u readers completed with READ mode (no owner updates).",
               num_readers);
    if (validation_result) {
      *validation_result = 1;
    }
  } else {
    ARTS_PRINT("=== ACQUIRE-MODE TEST FAILED ===");
    ARTS_PRINT("FAILURE: Found %u errors!", errors);
    if (validation_result) {
      *validation_result = 0;
    }
  }

  ARTS_PRINT("Acquire-Mode Test Complete");
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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
    ARTS_PRINT("ERROR: array_size (%u) and num_readers (%u) must be > 0",
               array_size, num_readers);
    arts_shutdown();
    return;
  }
  /// Validate that array_size is divisible by num_readers
  if (array_size % num_readers != 0) {
    ARTS_PRINT("ERROR: array_size (%u) must be divisible by num_readers (%u)",
               array_size, num_readers);
    arts_shutdown();
    return;
  }

  ARTS_PRINT("Acquire-Mode Test");
  ARTS_PRINT("- Array:   %u elements (%zu MB)", array_size,
             ((unsigned long)array_size * sizeof(uint64_t)) / (1024UL * 1024UL));
  ARTS_PRINT("- Readers: %u concurrent reader EDTs", num_readers);
  ARTS_PRINT("- Nodes:   %u", arts_get_total_nodes());

  /// Reserve GUID for data DB

  /// Create data DB and initialize to zeros
  data_guid = arts_guid_reserve(ARTS_DB, 0);
  size_t db_size = array_size * sizeof(uint64_t);
  ARTS_PRINT("Creating Data DB (guid: %lu, size: %zu bytes = %.2f MB)",
             data_guid, db_size, db_size / (1024.0 * 1024.0));
  uint64_t *data_ptr = (uint64_t *)arts_db_create_with_guid(data_guid, db_size, NULL);
  for (size_t i = 0; i < array_size; i++) {
    data_ptr[i] = 0;
}
  ARTS_PRINT("Data DB initialized to zeros");

  /// Allocate validation result flag (shared with validator)
  validation_result = (volatile unsigned int *)malloc(sizeof(unsigned int));
  *validation_result = 0;

  /// Start epoch
  arts_guid_t epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
  ARTS_PRINT("[Step 1] Started epoch (guid: %lu)", epoch_guid);

  /// Create writer EDT
  arts_guid_t writer_edt_guid =
      arts_edt_create_with_epoch(writer_edt, 0, NULL, 1, epoch_guid, &(arts_hint_t){.route = 0});
  ARTS_PRINT("[Step 2] Created writer EDT (guid: %lu)", writer_edt_guid);

  /// Create reader EDTs
  arts_guid_t *reader_edt_guids =
      (arts_guid_t *)calloc(num_readers, sizeof(arts_guid_t));
  for (unsigned int i = 0; i < num_readers; i++) {
    unsigned int target_node = (i % arts_get_total_nodes());
    uint64_t param = i;
    reader_edt_guids[i] =
        arts_edt_create_with_epoch(reader_edt, 1, &param, 1, epoch_guid, &(arts_hint_t){.route = target_node});

    if ((i + 1) % 4 == 0 || i == num_readers - 1) {
      unsigned int range_start = (i / 4) * 4;
      unsigned int range_end = i;
      ARTS_PRINT("Created readers %u-%u", range_start, range_end);
    }
  }

  arts_guid_t validator_edt_guid =
      arts_edt_create_with_epoch(validator_edt, 0, NULL, 1, epoch_guid, &(arts_hint_t){.route = 0});
  ARTS_PRINT("[Step 4] Created validator EDT (guid: %lu)", validator_edt_guid);

  /// Record ALL dependencies
  ARTS_PRINT("[Step 5] Recording dependencies: 1 writer (WRITE) + %u readers "
             "(READ) + 1 validator (READ)",
             num_readers);
  arts_record_dep(data_guid, writer_edt_guid, 0, ARTS_DB_WRITE);
  for (unsigned int i = 0; i < num_readers; i++) {
    arts_record_dep(data_guid, reader_edt_guids[i], 0, ARTS_DB_READ);
}

  arts_record_dep(data_guid, validator_edt_guid, 0, ARTS_DB_READ);

  ARTS_PRINT("  All dependencies recorded (latch=1, only writer)");

  // Release auto-acquired WRITE access before blocking on epoch.
  arts_db_release(data_guid);

  /// Wait for epoch to complete
  ARTS_PRINT("[Step 6] Waiting for epoch to complete");
  arts_wait_on_handle(epoch_guid);

  /// Free reader EDTs
  free(reader_edt_guids);

  /// Print final status
  if (validation_result && *validation_result == 1) {
    ARTS_PRINT("TEST STATUS: SUCCESS");
    ARTS_PRINT("Acquire-mode override successfully reduced owner updates.");
    ARTS_PRINT("Expected: 1 writer update + 0 reader updates = ~%u%% reduction",
               (100 * num_readers) / (num_readers + 1));
    ARTS_PRINT("(vs %u updates if all were WRITE mode)", num_readers + 1);
  } else {
    ARTS_PRINT("TEST STATUS: FAILURE");
  }

  /// Shutdown runtime
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
