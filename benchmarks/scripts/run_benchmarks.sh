#!/bin/bash
# Benchmark smoke test script
# Runs each benchmark app with correct arguments and reports pass/fail.
# Writes timestamped logs to benchmarks/scripts/logs/.
#
# Usage:
#   cd build/benchmarks
#   ../../benchmarks/scripts/run_benchmarks.sh [options] [backend ...]
#
# Options:
#   --multinode   Run apps in 2-node mode (simulated via 2 localhost processes)
#   --no-build    Skip the build step
#
# Backends: arts, xsocr, baseline (default: all three, run sequentially)
#
# Environment variables:
#   BENCH_MEM_LIMIT_KB  - Virtual memory limit per process in KB (default: 8388608 = 8GB)
#                         Set to 0 to disable. Prevents OOM from apps with huge allocations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMEOUT=60  # seconds per app
DO_MULTINODE=0
DO_BUILD=0

# Parse options and backends
BACKENDS=()
for arg in "$@"; do
    case "$arg" in
        --multinode)  DO_MULTINODE=1 ;;
        --no-build)   DO_BUILD=0 ;;
        arts|xsocr|baseline) BACKENDS+=("$arg") ;;
        all) BACKENDS=(arts xsocr baseline) ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--multinode] [--no-build] [arts|xsocr|baseline|all]"
            exit 1
            ;;
    esac
done
# Default: run all backends sequentially
if [ ${#BACKENDS[@]} -eq 0 ]; then
    BACKENDS=(arts xsocr baseline)
fi

# Memory limit in KB for child processes (virtual address space).
# Default 8GB. Set BENCH_MEM_LIMIT_KB=0 to disable.
BENCH_MEM_LIMIT_KB="${BENCH_MEM_LIMIT_KB:-8388608}"

if [ "$BENCH_MEM_LIMIT_KB" -gt 0 ] 2>/dev/null; then
    ulimit -v "$BENCH_MEM_LIMIT_KB" 2>/dev/null || true
    MEM_LIMIT_MB=$((BENCH_MEM_LIMIT_KB / 1024))
else
    MEM_LIMIT_MB="unlimited"
fi

# ==========================================================================
# Logging setup (matches tests/run_tests.sh format)
# ==========================================================================
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="$LOG_DIR/${TIMESTAMP}.log"

# Counters
PASS=0
FAIL=0
SKIP=0
HANG=0
TOTAL=0

# Colors (if terminal supports it)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' NC=''
fi

log() {
    echo "$@" >> "$LOG_FILE"
}

run_app() {
    local name="$1"
    local dir="$2"
    shift 2
    local args=("$@")

    TOTAL=$((TOTAL + 1))

    local exe="$dir/$name"
    if [ ! -x "$exe" ]; then
        printf "  ${YELLOW}SKIP${NC}  %-45s (not built)\n" "$name"
        log "=== $name ${args[*]:-} === SKIP (not built)"
        log ""
        SKIP=$((SKIP + 1))
        return
    fi

    log "=== $name ${args[*]:-} ==="
    local output
    local rc=0
    output=$(timeout "$TIMEOUT" "$exe" "${args[@]}" 2>&1) || rc=$?
    log "$output"
    log ""

    if [ $rc -eq 0 ]; then
        printf "  ${GREEN}PASS${NC}  %-45s\n" "$name ${args[*]:-}"
        PASS=$((PASS + 1))
    elif [ $rc -eq 124 ]; then
        printf "  ${RED}HANG${NC}  %-45s (timeout ${TIMEOUT}s)\n" "$name ${args[*]:-}"
        log "--- HANG (timeout ${TIMEOUT}s) ---"
        HANG=$((HANG + 1))
        FAIL=$((FAIL + 1))
    else
        printf "  ${RED}FAIL${NC}  %-45s (exit $rc)\n" "$name ${args[*]:-}"
        log "--- FAIL (exit $rc) ---"
        FAIL=$((FAIL + 1))
    fi
}

run_mpi_app() {
    local name="$1"
    local dir="$2"
    local np="$3"
    shift 3
    local args=("$@")

    TOTAL=$((TOTAL + 1))

    local exe="$dir/$name"
    if [ ! -x "$exe" ]; then
        printf "  ${YELLOW}SKIP${NC}  %-45s (not built)\n" "$name"
        log "=== $name (np=$np) ${args[*]:-} === SKIP (not built)"
        log ""
        SKIP=$((SKIP + 1))
        return
    fi

    if ! command -v mpirun &>/dev/null; then
        printf "  ${YELLOW}SKIP${NC}  %-45s (no mpirun)\n" "$name"
        log "=== $name (np=$np) ${args[*]:-} === SKIP (no mpirun)"
        log ""
        SKIP=$((SKIP + 1))
        return
    fi

    log "=== $name (np=$np) ${args[*]:-} ==="
    local output
    local rc=0
    output=$(timeout "$TIMEOUT" mpirun --oversubscribe -np "$np" "$exe" "${args[@]}" 2>&1) || rc=$?
    log "$output"
    log ""

    if [ $rc -eq 0 ]; then
        printf "  ${GREEN}PASS${NC}  %-45s\n" "$name (np=$np) ${args[*]:-}"
        PASS=$((PASS + 1))
    elif [ $rc -eq 124 ]; then
        printf "  ${RED}HANG${NC}  %-45s (timeout ${TIMEOUT}s)\n" "$name (np=$np) ${args[*]:-}"
        log "--- HANG (timeout ${TIMEOUT}s) ---"
        HANG=$((HANG + 1))
        FAIL=$((FAIL + 1))
    else
        printf "  ${RED}FAIL${NC}  %-45s (exit $rc)\n" "$name (np=$np) ${args[*]:-}"
        log "--- FAIL (exit $rc) ---"
        FAIL=$((FAIL + 1))
    fi
}

# ==========================================================================
# ARTS and XSOCR backend apps
# ==========================================================================

run_ocr_apps() {
    local suffix="$1"
    local dir="apps"

    echo ""
    echo "===== OCR apps ($suffix backend) ====="
    log ""
    log "===== OCR apps ($suffix backend) ====="

    # Ensure arts.cfg exists for ARTS backend
    if [ "$suffix" = "arts" ]; then
        if [ "$DO_MULTINODE" -eq 1 ]; then
            for cfg in "$REPO_ROOT/sample_configs/arts_multinode.cfg"; do
                if [ -f "$cfg" ]; then
                    cp "$cfg" arts.cfg
                    break
                fi
            done
        elif [ ! -f arts.cfg ]; then
            for cfg in "$REPO_ROOT/sample_configs/arts.cfg"; do
                if [ -f "$cfg" ]; then
                    cp "$cfg" .
                    break
                fi
            done
        fi
        if [ ! -f arts.cfg ]; then
            echo "WARNING: No arts.cfg found, ARTS apps may fail"
        fi
    fi

    # Set OCR_CONFIG for XSOCR backend
    if [ "$suffix" = "xsocr" ]; then
        export OCR_CONFIG="${REPO_ROOT}/benchmarks/xsocr/default.cfg"
    fi

    # Tier 1: Simple apps (minimal parameters for smoke testing)
    run_app "fibonacci_${suffix}"       "$dir" 10
    run_app "printf_${suffix}"          "$dir"
    run_app "quicksort_${suffix}"       "$dir"
    run_app "nqueens_${suffix}"         "$dir" 6 2

    # smithwaterman needs input files (not self-generating on x86)
    local sw_dir="/tmp/arts_sw_test"
    mkdir -p "$sw_dir"
    echo "ACGTACGTACGT" > "$sw_dir/str1.txt"
    echo "ACGTACGT" > "$sw_dir/str2.txt"
    echo "8" > "$sw_dir/score.txt"
    run_app "smithwaterman_${suffix}"   "$dir" 2 2 "$sw_dir/str1.txt" "$sw_dir/str2.txt" "$sw_dir/score.txt"

    # basicIO: mode=0 (write), count=10, filename
    run_app "basicIO_${suffix}"         "$dir" 0 10 /tmp/arts_basicIO_test.dat

    # Tier 2: Examples
    run_app "cache_offset_${suffix}"    "$dir"
    run_app "highbw_${suffix}"          "$dir"
    run_app "multigen_${suffix}"        "$dir"
    run_app "multigen_2_${suffix}"      "$dir"
    run_app "task_priorities_${suffix}"  "$dir"
    run_app "testlibs_${suffix}"        "$dir"

    # Tier 3: Kernels and utilities
    run_app "dbctrl_${suffix}"          "$dir" 5 5 256
    run_app "prodcon_${suffix}"         "$dir"
    run_app "curvefit_${suffix}"        "$dir"
    run_app "triangle_${suffix}"        "$dir"
    run_app "tempest_${suffix}"         "$dir"

    # Tier 4: Medium-complexity
    run_app "fft_${suffix}"             "$dir" 6
    run_app "graph500_${suffix}"        "$dir" 6 8 1 1

    # Tier 5: CoMD variants (small domain + few timesteps for smoke test)
    run_app "CoMD_intel_chandra_${suffix}"       "$dir" -x 4 -y 4 -z 4 -N 2
    run_app "CoMD_intel_chandra_tiled_${suffix}" "$dir" -x 4 -y 4 -z 4 -N 2
    run_app "CoMD_sdsc_${suffix}"                "$dir" -x 4 -y 4 -z 4 -N 2
    run_app "CoMD_sdsc2_${suffix}"               "$dir" -x 4 -y 4 -z 4 -N 2

    # Tier 6: HPCG variants
    run_app "hpcg_intel_${suffix}"              "$dir"
    run_app "hpcg_intel_Eager_${suffix}"        "$dir"

    # Tier 7: Stencil variants
    run_app "Stencil1D_intel_chandra_${suffix}" "$dir"
    run_app "stencil1D_sticky_${suffix}"        "$dir"
    run_app "Stencil2D_intel_chandra_${suffix}" "$dir"
    run_app "Stencil2D_intel_channelEVTs_${suffix}" "$dir"

    # Tier 8: P2P and Reduction
    run_app "p2p_${suffix}"                     "$dir"
    run_app "reduction_intel_${suffix}"         "$dir"
    run_app "reduction_intel_chandra_${suffix}" "$dir" 10

    # Tier 9: HPGMG
    run_app "hpgmg_${suffix}"                   "$dir"

    # Tier 10: MiniAMR (small mesh + few timesteps)
    run_app "miniAMR_forkbomb_${suffix}"        "$dir" --num_tsteps 3
    run_app "miniAMR_intel_${suffix}"           "$dir"
    run_app "miniAMR_intel_bryan_${suffix}"     "$dir"
    run_app "miniAMR_intel_chandra_${suffix}"   "$dir" --nx 4 --ny 4 --nz 4 --num_tsteps 2 --num_refine 3

    # Tier 11: Nekbone, NPB-CG
    run_app "nekbone_${suffix}"                 "$dir"
    run_app "npb_cg_${suffix}"                  "$dir"

    # Tier 12: RSBench, XSBench (tiny lookup count for smoke test)
    run_app "RSBench_intel_${suffix}"           "$dir" -l 100
    run_app "RSBench_intel_sharedDB_${suffix}"  "$dir" -l 100
    run_app "XSBench_intel_${suffix}"           "$dir" -s small -g 10 -l 100
    run_app "XSBench_intel_sharedDB_${suffix}"  "$dir" -s small -g 10 -l 100

    # Tier 13: LCS
    run_app "LCS_distributed_ST_${suffix}"      "$dir"
    run_app "LCS_shared_${suffix}"              "$dir"

    # Tier 14: SAR (tiny only for smoke test)
    run_app "sar_tiny_${suffix}"                "$dir"

    # Tier 15: Stream, UTS
    run_app "stream_${suffix}"                  "$dir"
    run_app "uts_${suffix}"                     "$dir"
}

# ==========================================================================
# Baseline apps (native OMP/MPI)
# ==========================================================================

run_baseline_apps() {
    local dir="baseline"

    echo ""
    echo "===== Baseline apps (OMP/MPI) ====="
    log ""
    log "===== Baseline apps (OMP/MPI) ====="

    # Pure OpenMP (minimal sizes for smoke testing)
    run_app "XSBench_omp"   "$dir" -s small -l 10000
    run_app "RSBench_omp"   "$dir" -l 10000
    run_app "Stencil2D_omp" "$dir" 2 5 64
    run_app "nqueens_omp"   "$dir" 6
    run_app "npb_cg_omp"    "$dir"
    run_app "lulesh_omp"    "$dir" -s 5 -i 2

    # MPI apps (run with 1 or 2 ranks, small domains)
    run_mpi_app "CoMD_mpi_omp"   "$dir" 1 -x 4 -y 4 -z 4 -N 2
    run_mpi_app "miniAMR_mpi"    "$dir" 1 --nx 4 --ny 4 --nz 4 --num_tsteps 2
    # Stencil1D_mpi: computes correctly but missing MPI_Finalize() in third-party
    # code causes Open MPI 5 to report non-zero exit. Known issue, not our bug.
    run_mpi_app "Stencil1D_mpi"  "$dir" 2 102 5
    run_mpi_app "Stencil2D_mpi"  "$dir" 1 5 64
    run_mpi_app "hpgmg_mpi_omp"  "$dir" 1 3
    run_mpi_app "lulesh_mpi_omp" "$dir" 1 -s 5 -i 2
    run_mpi_app "hpcg_mpi"       "$dir" 1
}

# ==========================================================================
# Main
# ==========================================================================

MODE_LABEL="single-node"
if [ "$DO_MULTINODE" -eq 1 ]; then
    MODE_LABEL="multi-node (2 localhost)"
fi

echo "=== ARTS Benchmark Smoke Test ==="
echo "    Backends: ${BACKENDS[*]}"
echo "    Mode:     $MODE_LABEL"
echo "    Timeout:  ${TIMEOUT}s per app"
echo "    Memory:   ${MEM_LIMIT_MB}MB per process"
echo "    Log:      $LOG_FILE"

log "=== ARTS Benchmark Smoke Test: $TIMESTAMP ==="
log "Backends: ${BACKENDS[*]}"
log "Mode:     $MODE_LABEL"
log "Timeout:  ${TIMEOUT}s per app"
log "Memory:   ${MEM_LIMIT_MB}MB per process"
log ""

for backend in "${BACKENDS[@]}"; do
    case "$backend" in
        arts)     run_ocr_apps arts ;;
        xsocr)    run_ocr_apps xsocr ;;
        baseline) run_baseline_apps ;;
    esac
done

echo ""
echo "==============================================================================="
echo "  SUMMARY"
echo "==============================================================================="
printf "  PASS: %3d\n" "$PASS"
printf "  FAIL: %3d  (includes %d hangs)\n" "$FAIL" "$HANG"
printf "  SKIP: %3d\n" "$SKIP"
echo   "  -----------"
printf "  TOTAL: %2d\n" "$TOTAL"
echo "==============================================================================="
echo "  Log: $LOG_FILE"
echo "==============================================================================="

log ""
log "=== SUMMARY ==="
log "PASS: $PASS  FAIL: $FAIL  HANG: $HANG  SKIP: $SKIP  TOTAL: $TOTAL"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
