#!/bin/bash
# Run the ARTS test suites across ALL FIVE build configurations
# (<memory model>_<protocol>_<timing>).  Each config has its own build trees,
# build_<config> (ctest, Debug) and build_release_<config> (harness):
#
#   config             cmake flags
#   -----------------  -------------------------------------------------------
#   ocr_rcu_eager      -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=RCU    -DARTS_PROTOCOL_TIMING=EAGER
#   ocr_rcu_lazy       -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=RCU    -DARTS_PROTOCOL_TIMING=LAZY
#   ocr_rwlock_eager   -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=RWLOCK -DARTS_PROTOCOL_TIMING=EAGER
#   ocr_rwlock_lazy    -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=RWLOCK -DARTS_PROTOCOL_TIMING=LAZY
#   wrf_rcu_eager      -DARTS_MEMORY_MODEL=DB_WRF -DARTS_COHERENCE_PROTOCOL=RCU    -DARTS_PROTOCOL_TIMING=EAGER
#
# DB_WRF (wrf_rcu_eager) requires program-ordered write-write conflicts, so some
# correctness deviations are EXPECTED there — they are reported, not silently
# treated as regressions.  OCR-model builds must be clean.
#
# Usage:
#   bash tests/run_all_models.sh                                  # all protocols, ctest + harness
#   bash tests/run_all_models.sh --models ocr_rcu_eager,ocr_rcu_lazy  # subset
#   bash tests/run_all_models.sh --no-harness                     # ctest only
#   bash tests/run_all_models.sh --no-ctest                       # harness only
#   bash tests/run_all_models.sh --no-build                       # skip reconfigure/rebuild
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

MODELS="ocr_rcu_eager ocr_rcu_lazy ocr_rwlock_eager ocr_rwlock_lazy wrf_rcu_eager"
DO_CTEST=1
DO_HARNESS=1
DO_BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --models)     MODELS="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
    --no-harness) DO_HARNESS=0; shift ;;
    --no-ctest)   DO_CTEST=0; shift ;;
    --no-build)   DO_BUILD=0; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

# protocol → ctest build dir / harness build dir / cmake flags
ctest_dir()   { echo "build_$1"; }
harness_dir() { echo "build_release_$1"; }
model_label() { echo "$1" | tr '[:lower:]' '[:upper:]'; }
model_cmake_flags() {
  case "$1" in
    ocr_rcu_eager)    echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=RCU -DARTS_PROTOCOL_TIMING=EAGER" ;;
    ocr_rcu_lazy)     echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=RCU -DARTS_PROTOCOL_TIMING=LAZY" ;;
    ocr_rwlock_eager) echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=RWLOCK -DARTS_PROTOCOL_TIMING=EAGER" ;;
    ocr_rwlock_lazy)  echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=RWLOCK -DARTS_PROTOCOL_TIMING=LAZY" ;;
    wrf_rcu_eager)    echo "-DARTS_MEMORY_MODEL=DB_WRF -DARTS_COHERENCE_PROTOCOL=RCU -DARTS_PROTOCOL_TIMING=EAGER" ;;
  esac
}
# expected CMakeCache values per config
model_model()  { case "$1" in wrf_rcu_eager) echo DB_WRF;; *) echo OCR;; esac; }
model_proto()  { case "$1" in ocr_rwlock_*) echo RWLOCK;; *) echo RCU;; esac; }
model_timing() { case "$1" in *_eager) echo EAGER;; *_lazy) echo LAZY;; esac; }

# Configure a build dir to the requested configuration if its cache does not
# match, then build.  Reconfigure forces a full rebuild (compile-flag change).
# The cache check compares ARTS_MEMORY_MODEL, ARTS_COHERENCE_PROTOCOL, and
# ARTS_PROTOCOL_TIMING.
ensure_build() {
  local dir="$1" model="$2" wantgpu="$3" extra="${4:-}"
  local want_model; want_model="$(model_model "$model")"
  local want_proto; want_proto="$(model_proto "$model")"
  local want_timing; want_timing="$(model_timing "$model")"
  local have_model; have_model="$(grep -E '^ARTS_MEMORY_MODEL:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local have_proto; have_proto="$(grep -E '^ARTS_COHERENCE_PROTOCOL:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local have_timing; have_timing="$(grep -E '^ARTS_PROTOCOL_TIMING:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local mismatch=0
  [ "$have_model" != "$want_model" ] && mismatch=1
  [ "$have_proto" != "$want_proto" ] && mismatch=1
  [ -n "$want_timing" ] && [ "$have_timing" != "$want_timing" ] && mismatch=1
  if [ ! -d "$dir" ] || [ "$mismatch" = 1 ]; then
    echo "  [cfg] $dir → $(model_cmake_flags "$model") (was MODEL='${have_model:-none}' PROTO='${have_proto:-none}' TIMING='${have_timing:-none}')"
    # shellcheck disable=SC2086
    cmake -GNinja -B "$dir" -DCMAKE_BUILD_TYPE=Release \
          $(model_cmake_flags "$model") -DARTS_USE_GPU="$wantgpu" $extra >/dev/null 2>&1 \
      || { echo "  [cfg] FAILED for $dir"; return 1; }
  fi
  ninja -C "$dir" >/dev/null 2>&1 || { echo "  [build] FAILED for $dir"; return 1; }
  return 0
}

# drain TCP TIME_WAIT/LISTEN leftovers on the ctest port range (bases 20000+,
# below the kernel ephemeral range) before a multinode ctest run
drain_ports() { local n=0; until ! ss -tan 2>/dev/null | grep -qE 'LISTEN.*:2[0-9]{4}\b'; do sleep 0.2; n=$((n+1)); [ $n -gt 100 ] && break; done; }

LOGDIR="$REPO/tests/logs"
mkdir -p "$LOGDIR"

declare -A RESULT
for m in $MODELS; do
  echo "================= PROTOCOL: $(model_label "$m") ================="

  if [ "$DO_CTEST" = 1 ]; then
    cd="$(ctest_dir "$m")"
    [ "$DO_BUILD" = 1 ] && ensure_build "$cd" "$m" OFF
    # No config copying: CTest sets each test's ARTS_CONFIG env straight at
    # the source cfg (tests/CMakeLists.txt).
    s=$( cd "$cd" && ctest -L single_node 2>&1 | grep -oE '[0-9]+% tests passed[^.]*' | head -1 )
    drain_ports
    mn=$( cd "$cd" && ctest -L multinode 2>&1 | grep -oE '[0-9]+% tests passed[^.]*' | head -1 )
    RESULT["$m,ctest_single"]="${s:-NORUN}"
    RESULT["$m,ctest_multi"]="${mn:-NORUN}"
    echo "  ctest single : ${s:-NORUN}"
    echo "  ctest multi  : ${mn:-NORUN}"
  fi

  if [ "$DO_HARNESS" = 1 ]; then
    hd="$(harness_dir "$m")"
    [ "$DO_BUILD" = 1 ] && ensure_build "$hd" "$m" OFF "-DARTS_BUILD_BENCHMARKS=ON"
    drain_ports
    timeout -k 30 2400 python3 "$REPO/benchmarks/scripts/correctness_harness.py" \
        --build-dir "$hd" --no-baseline >"$LOGDIR/harness_$m.log" 2>&1
    tally=$( grep -E '^Tier [ABM]:' "$LOGDIR/harness_$m.log" | tr '\n' ' ' )
    RESULT["$m,harness"]="${tally:-NORUN}"
    echo "  harness      : ${tally:-NORUN}  (full log: $LOGDIR/harness_$m.log)"
  fi
done

echo
echo "===================== SUMMARY (ocr_rcu_eager/ocr_rcu_lazy must be clean; WRF_RCU DB-WRF deviations annotated) ====================="
for m in $MODELS; do
  M="$(model_label "$m")"
  echo "[$M]"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest single : ${RESULT[$m,ctest_single]:-skip}"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest multi  : ${RESULT[$m,ctest_multi]:-skip}"
  [ "$DO_HARNESS" = 1 ] && echo "   harness      : ${RESULT[$m,harness]:-skip}"
done
