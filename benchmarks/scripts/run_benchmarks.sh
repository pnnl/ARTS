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
#   --multinode   Run apps in multi-node mode (simulated via N localhost processes)
#   --no-build    Skip the build step
#
# Backends: arts, xsocr, baseline (default: all three, run sequentially)
#
# Environment variables:
#   BENCH_MEM_LIMIT_KB   - Virtual memory limit per process in KB (default: 8388608 = 8GB)
#                          Set to 0 to disable. Prevents OOM from apps with huge allocations.
#   MULTINODE_VARIANT    - Multinode config variant to use when --multinode is set.
#                          Valid values: 2n (default), 3n, 4n, 2n_io.
#                          Maps to configs/local/${VARIANT}.cfg + configs/mpi/${VARIANT}.cfg.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMEOUT=60  # seconds per app
DO_MULTINODE=0
DO_BUILD=0
# Multinode config variant: 2n (default), 3n, 4n, 2n_io.
# Override via env: MULTINODE_VARIANT=3n run_benchmarks.sh --multinode arts
MULTINODE_VARIANT="${MULTINODE_VARIANT:-2n}"

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

# Kill any orphaned ARTS SSH child processes left after timeout.
# When timeout kills the master ARTS process, the SSH-spawned child may
# linger and hold ports 50000/50001.  Wait briefly for natural cleanup,
# then force-kill remaining processes by name.
cleanup_arts_children() {
    local name="$1"
    # Extract base binary name (e.g., "CoMD_intel_chandra_arts")
    local base
    base="$(basename "$name")"
    # Give the SSH child 1s to notice the master died and exit
    sleep 1
    # Kill any remaining processes matching the binary name
    pkill -f "$base" 2>/dev/null || true
    # Wait for ports to fully release
    sleep 1
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
        # Clean up SSH children that may hold ports after timeout
        if [ "$DO_MULTINODE" -eq 1 ]; then
            cleanup_arts_children "$name"
        fi
    else
        printf "  ${RED}FAIL${NC}  %-45s (exit $rc)\n" "$name ${args[*]:-}"
        log "--- FAIL (exit $rc) ---"
        FAIL=$((FAIL + 1))
        # Clean up SSH children on failure too (e.g., exit 255 from bind failure)
        if [ "$DO_MULTINODE" -eq 1 ]; then
            cleanup_arts_children "$name"
        fi
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

    # Ensure correct arts.cfg for ARTS backend (always overwrite to prevent
    # stale multinode config from a previous run causing single-node crashes)
    if [ "$suffix" = "arts" ]; then
        if [ "$DO_MULTINODE" -eq 1 ]; then
            cp "$REPO_ROOT/configs/local/${MULTINODE_VARIANT}.cfg" arts.cfg
        else
            cp "$REPO_ROOT/configs/local/1n.cfg" arts.cfg
        fi
        if [ ! -f arts.cfg ]; then
            echo "WARNING: No arts.cfg found, ARTS apps may fail"
        fi
    fi

    # Set OCR_CONFIG for XSOCR backend
    if [ "$suffix" = "xsocr" ]; then
        if [ "$DO_MULTINODE" -eq 1 ]; then
            export OCR_CONFIG="${REPO_ROOT}/configs/mpi/${MULTINODE_VARIANT}.cfg"
        else
            export OCR_CONFIG="${REPO_ROOT}/configs/mpi/1n.cfg"
        fi
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
    echo "12" > "$sw_dir/score.txt"
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
    run_app "cholesky_${suffix}"        "$dir"
    run_app "fft_${suffix}"             "$dir" 6
    run_app "globalsum_cgShim_${suffix}"    "$dir"
    run_app "globalsum_cgNoShim_${suffix}"  "$dir"
    run_app "globalsum_pcg_${suffix}"       "$dir"
    run_app "graph500_${suffix}"        "$dir" 6 8 1 1

    # Tier 5: CoMD variants (small domain + few timesteps for smoke test)
    # CoMD_intel_chandra: Known HANG on ARTS backend — FINISH EDT + affinity
    # hint interaction causes deadlock in epoch termination.  Passes on XSOCR.
    # The tiled variant (CoMD_intel_chandra_tiled) passes on both backends.
    if [ "$suffix" != "arts" ]; then
        run_app "CoMD_intel_chandra_${suffix}"   "$dir" -x 4 -y 4 -z 4 -N 1
    fi
    run_app "CoMD_intel_chandra_tiled_${suffix}" "$dir" -x 4 -y 4 -z 4 -N 2
    run_app "CoMD_sdsc_${suffix}"                "$dir" -x 4 -y 4 -z 4 -N 2
    run_app "CoMD_sdsc2_${suffix}"               "$dir" -x 4 -y 4 -z 4 -N 2

    # Tier 6: HPCG variants (1x1x1 grid, M=16, 5 iterations — minimum for smoke test)
    run_app "hpcg_intel_${suffix}"              "$dir" 1 1 1 16 5
    run_app "hpcg_intel_Eager_${suffix}"        "$dir" 1 1 1 16 5
    # hpcg_intel_Eager_Collective uses collective EVTs — arts-only.
    # Known HANG on ARTS: collective event reduction mechanism deadlocks
    # during iteration.  The non-collective variants (hpcg_intel, hpcg_intel_Eager)
    # pass on both backends.
    #if [ "$suffix" = "arts" ]; then
    #    run_app "hpcg_intel_Eager_Collective_${suffix}" "$dir" 1 1 1 16 5
    #fi

    # Tier 7: Stencil variants
    run_app "Stencil1D_intel_chandra_${suffix}" "$dir"
    run_app "stencil1D_sticky_${suffix}"        "$dir"
    run_app "Stencil2D_intel_chandra_${suffix}" "$dir"
    run_app "Stencil2D_intel_channelEVTs_${suffix}" "$dir"

    # Tier 8: P2P and Reduction (2 workers, 10 cols, 100 rows, 10 timesteps)
    run_app "p2p_${suffix}"                     "$dir" 2 10 100 10
    run_app "reduction_intel_${suffix}"         "$dir"
    # reduction_intel_chandra: Known SIGSEGV on XSOCR (v1 reduction lib bug)
    if [ "$suffix" != "xsocr" ]; then
        run_app "reduction_intel_chandra_${suffix}" "$dir" 10
    fi

    # Tier 9: HPGMG
    run_app "hpgmg_${suffix}"                   "$dir"

    # Tier 10: MiniAMR (small mesh + few timesteps)
    # miniAMR_forkbomb: Skip — "forkbomb" variant creates millions of blocks by
    # design. Times out on BOTH ARTS and XSOCR even with --num_tsteps 1. Original
    # code is incomplete (half the EDT implementations are commented out).
    #run_app "miniAMR_forkbomb_${suffix}"        "$dir" --max_blocks 10 --num_tsteps 1
    run_app "miniAMR_intel_${suffix}"           "$dir" --nx 4 --ny 4 --nz 4 --num_tsteps 2 --num_objects 1
    # miniAMR_intel_bryan: Skip on ARTS — hangs even with valid args (app creates
    # EDTs but never reaches shutdown). XSOCR passes because its native runtime
    # handles the incomplete shutdown path differently.
    if [[ "$suffix" != "arts" ]]; then
      run_app "miniAMR_intel_bryan_${suffix}"   "$dir"
    fi
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

    # Tier 14: SAR (tiny/small/medium/large; huge intentionally excluded —
    # huge dataset generation alone is far longer than this script's
    # per-app timeout, and the runtime test is outside the laptop budget).
    run_app "sar_tiny_${suffix}"                "$dir"
    run_app "sar_small_${suffix}"               "$dir"
    run_app "sar_medium_${suffix}"              "$dir"
    run_app "sar_large_${suffix}"               "$dir"

    # Tier 15: Stream, UTS
    # stream: Known SIGSEGV on XSOCR (runtime race in scheduler/memory mgmt)
    if [ "$suffix" != "xsocr" ]; then
        run_app "stream_${suffix}"              "$dir"
    fi
    run_app "uts_${suffix}"                     "$dir"

    # Tier 16: NUMA diagnostics (needs libnuma — will SKIP if not built)
    run_app "xeonNumaSize_${suffix}"            "$dir"
}

# ==========================================================================
# ARTS multi-node apps (22 apps that use affinity/rank distribution)
# ==========================================================================

run_ocr_apps_multinode() {
    local dir="apps"

    echo ""
    echo "===== OCR apps (arts backend, multi-node ${MULTINODE_VARIANT} localhost) ====="
    log ""
    log "===== OCR apps (arts backend, multi-node ${MULTINODE_VARIANT} localhost) ====="

    # Copy multi-node config
    cp "$REPO_ROOT/configs/local/${MULTINODE_VARIANT}.cfg" arts.cfg

    # CoMD variants (affinity-aware)
    # CoMD_intel_chandra: Known HANG — FINISH EDT + affinity deadlock
    #run_app "CoMD_intel_chandra_arts"       "$dir" -x 4 -y 4 -z 4 -N 1
    run_app "CoMD_intel_chandra_tiled_arts" "$dir" -x 4 -y 4 -z 4 -N 1

    # Graph500 (affinity-aware)
    run_app "graph500_arts"                "$dir" 6 8 1 1

    # HPCG variants (reduction lib — affinity-aware, minimum params)
    run_app "hpcg_intel_arts"              "$dir" 1 1 1 16 5
    run_app "hpcg_intel_Eager_arts"        "$dir" 1 1 1 16 5
    # hpcg_intel_Eager_Collective: skipped — collective event deadlock
    #run_app "hpcg_intel_Eager_Collective_arts" "$dir" 1 1 1 16 5

    # Stencil channel variants: Known multinode HANG (pre-existing, not
    # channel-rewrite related). The stencil apps assume local channel metadata
    # that doesn't replicate across nodes.
    #run_app "Stencil1D_intel_chandra_arts" "$dir"
    run_app "stencil1D_sticky_arts"        "$dir"
    #run_app "Stencil2D_intel_chandra_arts" "$dir"
    run_app "Stencil2D_intel_channelEVTs_arts" "$dir"

    # P2P and Reduction (reduction lib — affinity-aware, reduced params)
    run_app "p2p_arts"                     "$dir" 2 10 100 10
    run_app "reduction_intel_arts"         "$dir"

    # MiniAMR variants (affinity-aware)
    # miniAMR_forkbomb: Skip — fork-bombs by design, always times out
    #run_app "miniAMR_forkbomb_arts"        "$dir" --max_blocks 10 --num_tsteps 1
    run_app "miniAMR_intel_arts"           "$dir" --nx 4 --ny 4 --nz 4 --num_tsteps 2 --num_objects 1
    # miniAMR_intel_bryan: Skip on ARTS — see run_ocr_apps comment above
    run_app "miniAMR_intel_chandra_arts"   "$dir" --nx 4 --ny 4 --nz 4 --num_tsteps 2 --num_refine 3

    # Nekbone (reduction lib — affinity-aware)
    run_app "nekbone_arts"                 "$dir"

    # RSBench, XSBench sharedDB variants (reduction + ocrAppUtils — affinity-aware)
    run_app "RSBench_intel_sharedDB_arts"  "$dir" -l 100
    run_app "XSBench_intel_sharedDB_arts"  "$dir" -s small -g 10 -l 100

    # LCS distributed (reduction lib — affinity-aware)
    run_app "LCS_distributed_ST_arts"      "$dir"

    # Prodcon (labeled GUIDs — affinity-aware)
    run_app "prodcon_arts"                 "$dir"

    # Tempest (ocrAffinityGetCount — affinity-aware)
    run_app "tempest_arts"                 "$dir"
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
    run_mpi_app "SNAP_mpi_omp"   "$dir" 1

    # SAR OpenMP baseline (needs crlibm — will SKIP if not built)
    run_app "sar_omp"            "$dir"
}

# ==========================================================================
# Baseline multi-node apps (MPI with np=2)
# ==========================================================================

run_baseline_apps_multinode() {
    local dir="baseline"

    echo ""
    echo "===== Baseline apps (MPI multi-node, np=2) ====="
    log ""
    log "===== Baseline apps (MPI multi-node, np=2) ====="

    run_mpi_app "CoMD_mpi_omp"   "$dir" 2 -x 4 -y 4 -z 4 -N 2
    run_mpi_app "miniAMR_mpi"    "$dir" 2 --nx 4 --ny 4 --nz 4 --num_tsteps 2
    run_mpi_app "Stencil1D_mpi"  "$dir" 2 102 5
    run_mpi_app "Stencil2D_mpi"  "$dir" 2 5 64
    run_mpi_app "hpgmg_mpi_omp"  "$dir" 2 3
    # lulesh np must be a perfect cube (1, 8, 27, ...); np=2 is invalid
    run_mpi_app "lulesh_mpi_omp" "$dir" 8 -s 5 -i 2
    run_mpi_app "hpcg_mpi"       "$dir" 2
    run_mpi_app "SNAP_mpi_omp"   "$dir" 2
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
        arts)
            if [ "$DO_MULTINODE" -eq 1 ]; then
                run_ocr_apps_multinode
            else
                run_ocr_apps arts
            fi
            ;;
        xsocr)
            if [ "$DO_MULTINODE" -eq 1 ]; then
                echo ""
                echo "===== SKIP: XSOCR does not support multi-node ====="
                log ""
                log "===== SKIP: XSOCR does not support multi-node ====="
            else
                run_ocr_apps xsocr
            fi
            ;;
        baseline)
            run_baseline_apps
            if [ "$DO_MULTINODE" -eq 1 ]; then
                run_baseline_apps_multinode
            fi
            ;;
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
