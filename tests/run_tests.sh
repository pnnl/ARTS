#!/usr/bin/env bash
###############################################################################
# ARTS Test Runner
#
# Builds the project, runs all registered tests, classifies results, and
# writes a timestamped log to tests/logs/.
#
# Usage:
#   bash tests/run_tests.sh              # Run from project root
#   bash run_tests.sh                    # Run from tests/ directory
#   bash tests/run_tests.sh --no-build   # Skip the build step
#   bash tests/run_tests.sh --multinode  # Also run multi-node tests
#   bash tests/run_tests.sh --gpu        # Also run GPU tests (requires CUDA)
###############################################################################
set -euo pipefail

#===============================================================================
# Configuration
#===============================================================================
TIMEOUT_DEFAULT=10  # seconds
TIMEOUT_LONG=30     # for stress tests

#===============================================================================
# Resolve directories
#===============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
TEST_BIN_DIR="$BUILD_DIR/tests"
LOG_DIR="$SCRIPT_DIR/logs"

#===============================================================================
# Parse arguments
#===============================================================================
DO_BUILD=1
DO_MULTINODE=0
DO_GPU=0
for arg in "$@"; do
  case "$arg" in
    --no-build)   DO_BUILD=0 ;;
    --multinode)  DO_MULTINODE=1 ;;
    --gpu)        DO_GPU=1 ;;
    *)            echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

#===============================================================================
# Build
#===============================================================================
if [ "$DO_BUILD" -eq 1 ]; then
  echo "=== Building project ==="
  cd "$BUILD_DIR"
  cmake -GNinja .. -DCMAKE_BUILD_TYPE=Debug 2>&1 | tail -5
  ninja 2>&1 | tail -5
  echo ""
fi

#===============================================================================
# Setup
#===============================================================================
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="$LOG_DIR/${TIMESTAMP}.log"

# Copy config files
cp "$PROJECT_ROOT/configs/local/1n.cfg"       "$TEST_BIN_DIR/arts.cfg"
if [ -f "$PROJECT_ROOT/configs/local/2n.cfg" ]; then
  cp "$PROJECT_ROOT/configs/local/2n.cfg"       "$TEST_BIN_DIR/arts_2node.cfg"
fi
if [ -f "$PROJECT_ROOT/configs/local/gpu/1n.cfg" ]; then
  cp "$PROJECT_ROOT/configs/local/gpu/1n.cfg"  "$TEST_BIN_DIR/arts_gpu.cfg"
fi

# Counters
PASS=0; FAIL=0; LEAK=0; HANG=0; SKIP=0; TOTAL=0

#===============================================================================
# Test registry
#
# Format: "binary_name|args|timeout|description"
# Leave args empty for no arguments.
#===============================================================================
SINGLE_NODE_TESTS=(
  # --- EDT tests ---
  "edt_create_basic||$TIMEOUT_DEFAULT|EDT create basic"
  "edt_chain||$TIMEOUT_DEFAULT|EDT chain"
  "edt_fan_out||$TIMEOUT_DEFAULT|EDT fan out"
  "edt_signal||$TIMEOUT_DEFAULT|EDT signal (ptr/value)"
  "edt_dep_variants||$TIMEOUT_DEFAULT|EDT dep variants"
  "edt_destroy||$TIMEOUT_DEFAULT|EDT destroy"
  "edt_epoch_wait||$TIMEOUT_DEFAULT|EDT epoch wait"
  "stress_edt||$TIMEOUT_LONG|EDT stress test"
  "paramv_memcpy||$TIMEOUT_DEFAULT|paramv memcpy"
  # --- DB tests ---
  "db_create||$TIMEOUT_DEFAULT|DB create"
  "db_create_with_data||$TIMEOUT_DEFAULT|DB create with data"
  "db_dependence||$TIMEOUT_DEFAULT|DB dependence"
  "db_destroy||$TIMEOUT_DEFAULT|DB destroy"
  "db_local||$TIMEOUT_DEFAULT|DB local"
  "db_local_create||$TIMEOUT_DEFAULT|DB local create"
  "db_put_get||$TIMEOUT_DEFAULT|DB put/get"
  "db_rename||$TIMEOUT_DEFAULT|DB rename"
  "pin_db|0|$TIMEOUT_DEFAULT|PIN DB (node 0)"
  "get_from_db|16 1|$TIMEOUT_DEFAULT|get from DB"
  "put_in_db|16 1|$TIMEOUT_DEFAULT|put in DB"
  # --- Array DB tests ---
  "array_db||$TIMEOUT_DEFAULT|Array DB"
  "array_db_advanced||$TIMEOUT_DEFAULT|Array DB advanced"
  "array_db_for_each||$TIMEOUT_DEFAULT|Array DB for_each"
  "array_db_with_guid||$TIMEOUT_DEFAULT|Array DB with GUID"
  "local_array_db||$TIMEOUT_DEFAULT|Local array DB"
  "for_each||$TIMEOUT_DEFAULT|for_each"
  "put_array_db_epoch|16|$TIMEOUT_DEFAULT|put array DB epoch"
  "put_array_db_epoch_direct|16|$TIMEOUT_DEFAULT|put array DB epoch direct"
  "gather_array_db_epoch||$TIMEOUT_DEFAULT|gather array DB epoch"
  "termination_detection_array_db|16|$TIMEOUT_DEFAULT|termination detection array DB"
  # --- Epoch / sync tests ---
  "epoch_basic||$TIMEOUT_DEFAULT|Epoch basic"
  "epoch_deferred_start||$TIMEOUT_DEFAULT|Epoch deferred start"
  "epoch_finish_edt||$TIMEOUT_DEFAULT|Epoch finish EDT"
  "epoch_pool|3|$TIMEOUT_DEFAULT|Epoch pool"
  "rec_epoch|3|$TIMEOUT_DEFAULT|Recursive epoch"
  "termination_detection|3|$TIMEOUT_DEFAULT|Termination detection"
  # --- Event tests ---
  "event_basic||$TIMEOUT_DEFAULT|Event basic"
  "event_chain||$TIMEOUT_DEFAULT|Event chain"
  "persistent_event||$TIMEOUT_DEFAULT|Persistent event"
  "persistent_event_advanced||$TIMEOUT_DEFAULT|Persistent event advanced"
  # --- GUID tests ---
  "guid_basic||$TIMEOUT_DEFAULT|GUID basic"
  "guid_range||$TIMEOUT_DEFAULT|GUID range"
  # --- Routing / misc ---
  "hint_routing||$TIMEOUT_DEFAULT|Hint routing"
  "node_query||$TIMEOUT_DEFAULT|Node query"
  "record_dep_at||$TIMEOUT_DEFAULT|Record dep at"
  "utility_api||$TIMEOUT_DEFAULT|Utility API"
  "route_table||$TIMEOUT_DEFAULT|Route table"
  "route_table_iter||$TIMEOUT_DEFAULT|Route table iter"
  "route_table_iter_destroy||$TIMEOUT_DEFAULT|Route table iter destroy"
  "acquire_mode||$TIMEOUT_DEFAULT|Acquire mode"
  "arts_id||$TIMEOUT_DEFAULT|Arts ID (object counters)"
  "out_of_order_list||$TIMEOUT_DEFAULT|Out-of-order list"
  # --- Graph tests ---
  "csr||$TIMEOUT_DEFAULT|CSR graph"
  "distribution||$TIMEOUT_DEFAULT|Distribution"
)

MULTI_NODE_TESTS=(
  "multinode_db||$TIMEOUT_LONG|Multi-node DB"
  "multinode_edt||$TIMEOUT_LONG|Multi-node EDT"
  "multinode_epoch||$TIMEOUT_LONG|Multi-node epoch"
  "multinode_event||$TIMEOUT_LONG|Multi-node event"
  "active_message_db|64|$TIMEOUT_LONG|Active message DB"
  "arts_send|16|$TIMEOUT_LONG|arts_send"
  "route_table_remote_guid||$TIMEOUT_LONG|Route table remote GUID"
  "db_put_get_at||$TIMEOUT_LONG|DB put/get at"
  "db_remote||$TIMEOUT_LONG|DB remote"
  "remote_db_event||$TIMEOUT_LONG|Remote DB event"
  "multinode_event_types||$TIMEOUT_LONG|Multi-node event types"
  "multinode_array_db||$TIMEOUT_LONG|Multi-node array DB"
  "multinode_db_advanced||$TIMEOUT_LONG|Multi-node DB advanced"
)

GPU_TESTS=(
  # --- GPU EDT tests ---
  "gpu_edt_basic||$TIMEOUT_DEFAULT|GPU EDT basic"
  "gpu_edt_dep||$TIMEOUT_DEFAULT|GPU EDT dep variants"
  "gpu_edt_with_guid||$TIMEOUT_DEFAULT|GPU EDT with GUID"
  "gpu_edt_passthrough||$TIMEOUT_DEFAULT|GPU EDT passthrough"
  # --- GPU memory tests ---
  "gpu_memory||$TIMEOUT_DEFAULT|GPU memory"
  "gpu_memset||$TIMEOUT_DEFAULT|GPU memset"
  # --- GPU DB tests ---
  "gpu_db||$TIMEOUT_DEFAULT|GPU DB transfer"
  # --- GPU LC tests ---
  "gpu_lc_sync_basic||$TIMEOUT_DEFAULT|GPU LC sync basic"
  "lc_sync||$TIMEOUT_DEFAULT|LC sync"
  # --- GPU multi-kernel ---
  "gpu_multi_kernel||$TIMEOUT_DEFAULT|GPU multi-kernel"
  # --- GPU library tests (thrust/cuBLAS) ---
  "gpu_for_all||$TIMEOUT_DEFAULT|GPU for_all (thrust)"
  "gpu_lib||$TIMEOUT_DEFAULT|GPU lib (cuBLAS)"
)

#===============================================================================
# Run a single test
#
# Args: binary args timeout description config_file
# Sets global: last_result (PASS|FAIL|LEAK|HANG|SKIP)
#===============================================================================
run_test() {
  local binary="$1"
  local args="$2"
  local tout="$3"
  local desc="$4"
  local config="${5:-}"

  TOTAL=$((TOTAL + 1))
  local bin_path="$TEST_BIN_DIR/$binary"

  if [ ! -x "$bin_path" ]; then
    last_result="SKIP"
    SKIP=$((SKIP + 1))
    printf "  %-40s  %s\n" "$desc" "[SKIP] (not built)"
    echo "=== $desc ($binary $args) === SKIP (not built)" >> "$LOG_FILE"
    return
  fi

  # Use multinode config if specified
  local env_prefix=""
  if [ -n "$config" ]; then
    env_prefix="ARTS_CONFIG=$config"
  fi

  # Run with timeout, capture output
  local output
  local exit_code=0
  echo "=== $desc ($binary $args) ===" >> "$LOG_FILE"

  if [ -n "$config" ]; then
    # Relax ASan for GPU/multinode tests: protect_shadow_gap=0 prevents ASan
    # shadow memory from blocking CUDA driver VA mappings; detect_leaks=0
    # suppresses false-positive leak reports from libcuda.so internals.
    output=$(cd "$TEST_BIN_DIR" && timeout "$tout" env \
      ARTS_CONFIG="$config" \
      ASAN_OPTIONS="${ASAN_OPTIONS:-}:protect_shadow_gap=0:detect_leaks=0:alloc_dealloc_mismatch=0" \
      "./$binary" $args 2>&1) || exit_code=$?
  else
    output=$(cd "$TEST_BIN_DIR" && timeout "$tout" "./$binary" $args 2>&1) || exit_code=$?
  fi

  echo "$output" >> "$LOG_FILE"
  echo "" >> "$LOG_FILE"

  # Classify result
  # Check sanitizer output FIRST — LeakSanitizer exits with code 1,
  # same as arts_abort(), so exit code alone can't distinguish them.
  if [ "$exit_code" -eq 124 ]; then
    # timeout returns 124 when the command times out
    last_result="HANG"
    HANG=$((HANG + 1))
    printf "  %-40s  %s\n" "$desc" "[HANG] (timeout ${tout}s)"
  elif echo "$output" | grep -qE "ERROR:.*Sanitizer|LeakSanitizer"; then
    last_result="LEAK"
    LEAK=$((LEAK + 1))
    printf "  %-40s  %s\n" "$desc" "[LEAK]"
  elif [ "$exit_code" -ne 0 ]; then
    last_result="FAIL"
    FAIL=$((FAIL + 1))
    printf "  %-40s  %s\n" "$desc" "[FAIL] (exit $exit_code)"
  else
    last_result="PASS"
    PASS=$((PASS + 1))
    printf "  %-40s  %s\n" "$desc" "[PASS]"
  fi
}

#===============================================================================
# Main
#===============================================================================
echo "=== ARTS Test Runner ==="
echo "    Build dir:  $BUILD_DIR"
echo "    Log file:   $LOG_FILE"
echo ""
echo "--- Single-node tests ---"

echo "=== ARTS Test Run: $TIMESTAMP ===" > "$LOG_FILE"
echo "" >> "$LOG_FILE"

for entry in "${SINGLE_NODE_TESTS[@]}"; do
  IFS='|' read -r binary args tout desc <<< "$entry"
  run_test "$binary" "$args" "$tout" "$desc"
done

if [ "$DO_MULTINODE" -eq 1 ]; then
  echo ""
  echo "" >> "$LOG_FILE"
  echo "=== Multi-node tests ===" >> "$LOG_FILE"

  for variant in 2n 4n 5n; do
    MULTINODE_CFG="$TEST_BIN_DIR/arts_${variant}ode.cfg"
    if [ ! -f "$MULTINODE_CFG" ]; then
      echo "  WARNING: arts_${variant}ode.cfg not found, skipping ${variant} multi-node tests"
      continue
    fi
    echo ""
    echo "--- Multi-node tests (${variant} on localhost) ---"
    for entry in "${MULTI_NODE_TESTS[@]}"; do
      IFS='|' read -r binary args tout desc <<< "$entry"
      run_test "$binary" "$args" "$tout" "${desc} (${variant})" "$MULTINODE_CFG"
    done
  done
fi

if [ "$DO_GPU" -eq 1 ]; then
  echo ""
  echo "--- GPU tests (requires CUDA) ---"
  echo "" >> "$LOG_FILE"
  echo "=== GPU tests ===" >> "$LOG_FILE"

  GPU_CFG="$TEST_BIN_DIR/arts_gpu.cfg"
  if [ ! -f "$GPU_CFG" ]; then
    echo "  ERROR: arts_gpu.cfg not found at $GPU_CFG"
  else
    for entry in "${GPU_TESTS[@]}"; do
      IFS='|' read -r binary args tout desc <<< "$entry"
      run_test "$binary" "$args" "$tout" "$desc" "$GPU_CFG"
    done
  fi
fi

#===============================================================================
# Summary
#===============================================================================
echo ""
echo "==============================================================================="
echo "  SUMMARY"
echo "==============================================================================="
printf "  PASS: %3d\n" "$PASS"
printf "  FAIL: %3d\n" "$FAIL"
printf "  LEAK: %3d\n" "$LEAK"
printf "  HANG: %3d\n" "$HANG"
printf "  SKIP: %3d\n" "$SKIP"
echo   "  -----------"
printf "  TOTAL: %2d\n" "$TOTAL"
echo "==============================================================================="
echo "  Log: $LOG_FILE"
echo "==============================================================================="

# Append summary to log
echo "" >> "$LOG_FILE"
echo "=== SUMMARY ===" >> "$LOG_FILE"
echo "PASS: $PASS  FAIL: $FAIL  LEAK: $LEAK  HANG: $HANG  SKIP: $SKIP  TOTAL: $TOTAL" >> "$LOG_FILE"

# Exit with failure if any tests failed
if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
exit 0
