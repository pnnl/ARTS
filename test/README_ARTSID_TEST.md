# arts_id Tracking Test Guide

This document explains how to build, run, and validate the `testArtsId` test for the arts_id tracking infrastructure.

## Overview

The `testArtsId` test validates the complete arts_id tracking implementation for ArtsMate integration, including:

- Creating EDTs and DBs with compiler-assigned arts_id values
- Per-thread counter tracking using hash tables
- Collision handling in the hash table
- JSON export functionality
- Multi-node distributed tracking

## Prerequisites

- ARTS runtime built with introspection support
- CMake build system configured
- Optional: MPI for multi-node testing

## Building the Test

### Step 1: Configure Counter Tracking (Optional)

By default, all arts_id counters are disabled (`OFF`) for zero overhead. To enable tracking:

```bash
cd /path/to/arts
vi counter.cfg
```

Enable the desired counters:

```cfg
# Aggregate mode (low overhead: ~15-18 cycles)
artsIdEdtMetrics=THREAD         # Per-arts_id EDT metrics
artsIdDbMetrics=THREAD          # Per-arts_id DB metrics

# Detailed mode (medium overhead: ~35 cycles)
artsIdEdtCaptures=OFF           # Individual EDT captures
artsIdDbCaptures=OFF            # Individual DB captures
```

**Counter Modes**:
- `OFF`: Disabled (zero overhead)
- `ONCE`: Single capture at shutdown
- `THREAD`: Per-thread capture without reduction
- `NODE`: Capture and reduce per node

### Step 2: Build ARTS and Test

```bash
# From ARTS root directory
mkdir -p build
cd build

# Configure (this regenerates Preamble.h from counter.cfg)
cmake ..

# Build the test
make testArtsId

# Verify the test binary was created
ls -lh test/testArtsId
```

## Running the Test

### Single Node Execution

```bash
# From build directory
./test/testArtsId
```

### Multi-Node Execution (MPI)

```bash
# Run with 4 MPI processes
mpirun -np 4 ./test/testArtsId

# Run with 8 processes on specific hosts
mpirun -np 8 -host node1,node2,node3,node4 ./test/testArtsId
```

### With Runtime Arguments

```bash
# ARTS runtime accepts various arguments
./test/testArtsId --help                    # Show ARTS options
./test/testArtsId --threads 8               # Use 8 threads per node
./test/testArtsId --counter-folder ./output # Set counter output folder
```

## Expected Output

### Test Execution Log

```
═══════════════════════════════════════
ARTS arts_id Tracking Test
═══════════════════════════════════════
Test Configuration:
- Test EDTs: 10
- Test DBs:  5
- Nodes:     1
- Threads:   4
═══════════════════════════════════════

[Step 1] Started epoch (guid: 12345)
[Step 2] Created validator EDT (guid: 12346)

[Step 3] Creating 5 test DBs with arts_id values:
  - DB[0]: guid=12347, arts_id=1100, size=8192 bytes
  - DB[1]: guid=12348, arts_id=1101, size=8192 bytes
  - DB[2]: guid=12349, arts_id=1102, size=8192 bytes
  - DB[3]: guid=12350, arts_id=1103, size=8192 bytes
  - DB[4]: guid=12351, arts_id=1104, size=8192 bytes

[Step 4] Creating 10 writer EDTs with arts_id values:
  - EDT[0]: guid=12352, arts_id=1000, node=0, using DB[0]
  - EDT[1]: guid=12353, arts_id=1001, node=0, using DB[1]
  ...

[Step 5] Creating reader EDTs with arts_id values:
  - Reader[0]: guid=12362, arts_id=1200, node=0, deps=1
  ...

[Validator] Starting validation
[Validator] Test 1: Checking arts_id data structures...
[Validator] Test 2: Checking counter modes...
[Validator] artsIdEdtMetrics mode: 2 (0=OFF, 1=ONCE, 2=THREAD, 3=NODE)
[Validator] Test 3: Exporting counters.json...
[Validator] ✓ Exported counters.json
[Validator] ✓ counters.json appears valid (starts with '{')

═══════════════════════════════════════
TEST STATUS: SUCCESS
═══════════════════════════════════════
✓ All arts_id tracking tests passed!
✓ Created 10 EDTs with arts_id values
✓ Created 5 DBs with arts_id values
✓ Counter infrastructure validated
```

### Generated counters.json (when counters enabled)

```json
{
  "version": "1.0",
  "timestamp": "2025-01-13T22:30:15Z",
  "total_runtime_ms": 0.00,
  "edts": [
    {
      "arts_id": 1000,
      "invocations": 1,
      "exec_ms": 2.345,
      "stall_ms": 0.123
    },
    {
      "arts_id": 1001,
      "invocations": 1,
      "exec_ms": 2.156,
      "stall_ms": 0.089
    }
  ],
  "dbs": [
    {
      "arts_id": 1100,
      "bytes_local": 8192,
      "bytes_remote": 0,
      "cache_misses": 0
    }
  ]
}
```

## Validating the Implementation

### 1. Verify Zero Overhead When Disabled

```bash
# Ensure counter.cfg has all arts_id counters set to OFF
grep artsId counter.cfg
# Should show:
# artsIdEdtMetrics=OFF
# artsIdDbMetrics=OFF
# artsIdEdtCaptures=OFF
# artsIdDbCaptures=OFF

# Rebuild and check generated code
cmake ..
cat build/core/inc/arts/introspection/Preamble.h | grep -A5 artsIdEdtMetrics

# Should see:
# #define ENABLE_artsIdEdtMetrics 0
# (macros compile to ((void)0) - zero overhead)

# Run test
./test/testArtsId
# Should complete successfully even with counters disabled
```

### 2. Verify Tracking with Aggregate Mode

```bash
# Edit counter.cfg
artsIdEdtMetrics=THREAD
artsIdDbMetrics=THREAD

# Rebuild
cmake ..
make testArtsId

# Run test
./test/testArtsId

# Verify counters.json was created
ls -lh counters.json
cat counters.json

# Should contain arts_id entries matching test EDTs (1000-1009) and DBs (1100-1104)
```

### 3. Verify Hash Table Statistics

The test prints collision statistics when enabled:

```
[Validator] Test 4: Checking hash table statistics...
[Validator] Total EDT hash collisions: 0
[Validator] Total DB hash collisions: 0
```

With 10 EDTs and 5 DBs and a 256-entry hash table, collisions should be rare (< 5%).

### 4. Verify Multi-Node Distribution

```bash
# Run with 4 MPI processes
mpirun -np 4 ./test/testArtsId

# Check that EDTs are distributed across nodes
# Look for log messages showing different node IDs:
# Node 0 - EDT with expected arts_id=1000 executing
# Node 1 - EDT with expected arts_id=1001 executing
# Node 2 - EDT with expected arts_id=1002 executing
# Node 3 - EDT with expected arts_id=1003 executing
```

## Performance Validation

### Measure Overhead

Create a simple benchmark:

```bash
# Disable counters
artsIdEdtMetrics=OFF
artsIdDbMetrics=OFF

# Rebuild and run
cmake .. && make testArtsId
time ./test/testArtsId
# Baseline: X seconds

# Enable aggregate mode
artsIdEdtMetrics=THREAD
artsIdDbMetrics=THREAD

# Rebuild and run
cmake .. && make testArtsId
time ./test/testArtsId
# Should be: X + ~0.02% seconds (minimal overhead)
```

## Troubleshooting

### Test Fails with "counters.json not found"

**Cause**: Counters are disabled in counter.cfg

**Solution**: Enable at least one arts_id counter:
```cfg
artsIdEdtMetrics=THREAD
```

### Hash Table Collisions High (> 20%)

**Cause**: Hash table too small for workload

**Solution**: Increase `ARTS_ID_HASH_SIZE` in ArtsIdCounter.h:
```c
#ifndef ARTS_ID_HASH_SIZE
#define ARTS_ID_HASH_SIZE 512  // Increased from 256
#endif
```

### Test Hangs or Deadlocks

**Cause**: Possible EDT dependency issue

**Solution**: Check ARTS configuration:
- Ensure `USE_SMART_DB` is defined if using smart DBs
- Verify epoch handling is correct
- Check MPI configuration for multi-node runs

### Compile Errors about Missing Types

**Cause**: Build system not configured correctly

**Solution**:
```bash
# Clean build
rm -rf build
mkdir build && cd build
cmake ..
make testArtsId
```

## Integration with ArtsMate

This test demonstrates the runtime side of ArtsMate integration. For complete integration:

1. **Compiler Side (CARTS)**:
   - Assign unique arts_id values to loops and memrefs
   - Generate calls to `artsEdtCreateWithArtsId()` and `artsDbCreateWithArtsId()`
   - Export `entities.json` with static metadata

2. **Runtime Side (ARTS)** - This test validates:
   - Storage of arts_id in EDT/DB structures
   - Counter tracking during execution
   - Export of `counters.json`

3. **Analysis Side (ArtsMate)**:
   - Read `entities.json` (static metadata)
   - Read `counters.json` (runtime measurements)
   - Merge by arts_id to create complete performance profile

## Next Steps

After validating this test:

1. **Integrate with CARTS compiler** to generate arts_id values
2. **Run on real applications** to validate overhead and scalability
3. **Tune hash table size** based on typical program characteristics
4. **Add detailed capture mode** tests for deep performance analysis
5. **Implement full ArtsMate pipeline** for end-to-end testing

## References

- `ARTS_RUNTIME_COUNTER_INTEGRATION.md` - Detailed implementation plan
- `counter.cfg` - Counter configuration reference
- `ArtsIdCounter.h` - API documentation
- `testTwinDiff.c` - Reference test structure
- `testAcquireMode.c` - Reference test patterns
