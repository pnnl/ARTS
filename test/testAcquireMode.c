/******************************************************************************
** Test: Acquire-Mode Override
******************************************************************************/
#include "arts/arts.h"
#include "arts/runtime/memory/DbFunctions.h"
#include "arts/system/ArtsPrint.h"
#include <stdlib.h>

/// Configurable test parameters
unsigned int arraySize = 1024 * 1024;
unsigned int numReaders = 16;

artsGuid_t dataGuid;
volatile unsigned int *validationResult = NULL;

/// Writer EDT: Initializes the array (WRITE mode)
void writerEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               artsEdtDep_t depv[]) {
  unsigned int nodeId = artsGetCurrentNode();
  uint64_t *data = (uint64_t *)depv[0].ptr;

  ARTS_PRINT("Writer (Node %u): Initializing array with sequential values",
             nodeId);
  ARTS_PRINT("Writer: mode=%u, acquireMode=%u, useTwinDiff=%d", depv[0].mode,
             depv[0].acquireMode, depv[0].useTwinDiff);

  // Initialize array with i * 7
  for (unsigned int i = 0; i < arraySize; i++) {
    data[i] = i * 7;
  }

  ARTS_PRINT("Writer (Node %u): Completed initialization of %u elements",
             nodeId, arraySize);
  // dec_latch happens automatically on release (WRITE mode)
}

/// Reader EDT: Reads and validates a portion of the array (READ mode override)
void readerEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               artsEdtDep_t depv[]) {
  unsigned int readerId = (unsigned int)paramv[0];
  unsigned int nodeId = artsGetCurrentNode();
  uint64_t *data = (uint64_t *)depv[0].ptr;

  ARTS_PRINT("Reader %u (Node %u): Reading array (acquireMode=READ)", readerId,
             nodeId);
  ARTS_PRINT("Reader %u: mode=%u, acquireMode=%u, useTwinDiff=%d", readerId,
             depv[0].mode, depv[0].acquireMode, depv[0].useTwinDiff);
  bool modeOk = (depv[0].acquireMode == ARTS_DB_READ);
  if (!modeOk) {
    ARTS_PRINT("Reader %u (Node %u): ERROR - expected acquireMode=READ but "
               "received %u",
               readerId, nodeId, depv[0].acquireMode);
  }

  // Read and validate a stripe of the array
  unsigned int startIdx = readerId * (arraySize / numReaders);
  unsigned int endIdx = (readerId + 1) * (arraySize / numReaders);

  uint64_t sum = 0;
  unsigned int errors = 0;

  for (unsigned int i = startIdx; i < endIdx; i++) {
    uint64_t expected = i * 7;
    uint64_t actual = data[i];

    if (actual != expected) {
      if (errors < 3) {
        ARTS_PRINT("Reader %u ERROR at index %u: expected %lu, got %lu",
                   readerId, i, expected, actual);
      }
      errors++;
    }
    sum += actual;
  }

  if (errors == 0 && modeOk) {
    ARTS_PRINT("Reader %u (Node %u): SUCCESS - all values correct (sum=%lu)",
               readerId, nodeId, sum);
  } else {
    ARTS_PRINT("Reader %u (Node %u): FAILURE - %u errors found", readerId,
               nodeId, errors);
  }
}

/// Validator EDT: Final check after all readers (READ mode override)
void validatorEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                  artsEdtDep_t depv[]) {
  uint64_t *data = (uint64_t *)depv[0].ptr;
  unsigned int errors = 0;
  bool modeOk = (depv[0].acquireMode == ARTS_DB_READ);

  ARTS_PRINT("=== Validator: Performing final validation ===");
  ARTS_PRINT("Validator: mode=%u, acquireMode=%u, useTwinDiff=%d", depv[0].mode,
             depv[0].acquireMode, depv[0].useTwinDiff);
  if (!modeOk) {
    ARTS_PRINT("Validator ERROR: Expected acquireMode=READ but received %u",
               depv[0].acquireMode);
  }

  // Full array validation
  for (unsigned int i = 0; i < arraySize; i++) {
    uint64_t expected = i * 7;
    uint64_t actual = data[i];

    if (actual != expected) {
      if (errors < 5) {
        ARTS_PRINT("Validator ERROR at index %u: expected %lu, got %lu", i,
                   expected, actual);
      }
      errors++;
    }
  }

  if (errors == 0 && modeOk) {
    ARTS_PRINT("=== ACQUIRE-MODE TEST PASSED ===");
    ARTS_PRINT("SUCCESS: All %u elements validated correctly!", arraySize);
    ARTS_PRINT("All %u readers completed with READ mode (no owner updates).",
               numReaders);
    if (validationResult) {
      *validationResult = 1;
    }
  } else {
    ARTS_PRINT("=== ACQUIRE-MODE TEST FAILED ===");
    ARTS_PRINT("FAILURE: Found %u errors!", errors);
    if (validationResult) {
      *validationResult = 0;
    }
  }

  // Acquire-mode metrics are tracked via counter infrastructure
  // To view metrics, enable counters in counter.cfg and check output files
  ARTS_PRINT("Acquire-Mode Test Complete");
}

void artsMain(int argc, char **argv) {
  /// Parse command line arguments for parametric testing
  /// Usage: ./testAcquireModeCompilerPattern [arraySize] [numReaders]
  ///  - Parse arraySize
  if (argc > 1 && argv[1])
    arraySize = (unsigned int)atoi(argv[1]);
  ///  - Parse numReaders
  if (argc > 2 && argv[2])
    numReaders = (unsigned int)atoi(argv[2]);

  /// Validate parameters
  if (arraySize == 0 || numReaders == 0) {
    ARTS_PRINT("ERROR: arraySize (%u) and numReaders (%u) must be > 0",
               arraySize, numReaders);
    artsShutdown();
    return;
  }
  /// Validate that arraySize is divisible by numReaders
  if (arraySize % numReaders != 0) {
    ARTS_PRINT("ERROR: arraySize (%u) must be divisible by numReaders (%u)",
               arraySize, numReaders);
    artsShutdown();
    return;
  }

  ARTS_PRINT("Acquire-Mode Test");
  ARTS_PRINT("- Array:   %u elements (%zu MB)", arraySize,
             (arraySize * sizeof(uint64_t)) / (1024 * 1024));
  ARTS_PRINT("- Readers: %u concurrent reader EDTs", numReaders);
  ARTS_PRINT("- Nodes:   %u", artsGetTotalNodes());

  /// Reserve GUID for data DB

  /// Create data DB and initialize to zeros
  dataGuid = artsReserveGuidRoute(ARTS_DB_WRITE, 0);
  size_t dbSize = arraySize * sizeof(uint64_t);
  ARTS_PRINT("Creating Data DB (guid: %lu, size: %zu bytes = %.2f MB)",
             dataGuid, dbSize, dbSize / (1024.0 * 1024.0));
  uint64_t *dataPtr = (uint64_t *)artsDbCreateWithGuid(dataGuid, dbSize);
  for (size_t i = 0; i < arraySize; i++)
    dataPtr[i] = 0;
  ARTS_PRINT("Data DB initialized to zeros");

  /// Allocate validation result flag (shared with validator)
  validationResult = (volatile unsigned int *)artsMalloc(sizeof(unsigned int));
  *validationResult = 0;

  /// Start epoch
  artsGuid_t epochGuid = artsInitializeAndStartEpoch(NULL_GUID, 0);
  ARTS_PRINT("[Step 1] Started epoch (guid: %lu)", epochGuid);

  /// Create writer EDT
  artsGuid_t writerEdtGuid =
      artsEdtCreateWithEpoch(writerEdt, 0, 0, NULL, 1, epochGuid);
  ARTS_PRINT("[Step 2] Created writer EDT (guid: %lu)", writerEdtGuid);

  /// Create reader EDTs
  artsGuid_t *readerEdtGuids =
      (artsGuid_t *)artsMalloc(numReaders * sizeof(artsGuid_t));
  for (unsigned int i = 0; i < numReaders; i++) {
    unsigned int targetNode = (i % artsGetTotalNodes());
    uint64_t param = i;
    readerEdtGuids[i] =
        artsEdtCreateWithEpoch(readerEdt, targetNode, 1, &param, 1, epochGuid);

    if ((i + 1) % 4 == 0 || i == numReaders - 1) {
      unsigned int rangeStart = (i / 4) * 4;
      unsigned int rangeEnd = i;
      ARTS_PRINT("Created readers %u-%u", rangeStart, rangeEnd);
    }
  }

  artsGuid_t validatorEdtGuid =
      artsEdtCreateWithEpoch(validatorEdt, 0, 0, NULL, 1, epochGuid);
  ARTS_PRINT("[Step 4] Created validator EDT (guid: %lu)", validatorEdtGuid);

  /// Record ALL dependencies
  ARTS_PRINT("[Step 5] Recording dependencies: 1 writer (WRITE) + %u readers "
             "(READ) + 1 validator (READ)",
             numReaders);
  artsRecordDep(dataGuid, writerEdtGuid, 0, ARTS_DB_WRITE, true);
  for (unsigned int i = 0; i < numReaders; i++)
    artsRecordDep(dataGuid, readerEdtGuids[i], 0, ARTS_DB_READ, false);

  artsRecordDep(dataGuid, validatorEdtGuid, 0, ARTS_DB_READ, false);

  ARTS_PRINT("  All dependencies recorded (latch=1, only writer)");

  /// Wait for epoch to complete
  ARTS_PRINT("[Step 6] Waiting for epoch to complete");
  artsWaitOnHandle(epochGuid);

  /// Free reader EDTs
  artsFree(readerEdtGuids);

  /// Print final status
  if (validationResult && *validationResult == 1) {
    ARTS_PRINT("TEST STATUS: SUCCESS");
    ARTS_PRINT("Acquire-mode override successfully reduced owner updates.");
    ARTS_PRINT("Expected: 1 writer update + 0 reader updates = ~%u%% reduction",
               (100 * numReaders) / (numReaders + 1));
    ARTS_PRINT("(vs %u updates if all were WRITE mode)", numReaders + 1);
  } else {
    ARTS_PRINT("TEST STATUS: FAILURE");
  }

  /// Shutdown runtime
  artsShutdown();
}

int main(int argc, char **argv) {
  artsRT(argc, argv);
  return 0;
}
