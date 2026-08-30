#!/bin/bash
# Run the ARTS ctest suites across ALL NINE build configurations
# (<memory model>_<family>_<live second axis>), each in its own build_<config>
# tree, then run the application matrix once over the single benchmark build
# (which holds every coherence configuration at once):
#
#   config             cmake flags
#   -----------------  ---------------------------------------------------------
#   ocr_val_wt         -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=VAL  -DARTS_WRITE_POLICY=WT
#   ocr_val_wb         -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=VAL  -DARTS_WRITE_POLICY=WB
#   ocr_val_wt_purge   -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=VAL  -DARTS_WRITE_POLICY=WT -DARTS_RELEASE_POLICY=PURGE
#   ocr_excl_purge     -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=EXCL -DARTS_RELEASE_POLICY=PURGE
#   ocr_excl_retain    -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=EXCL -DARTS_RELEASE_POLICY=RETAIN
#   ocr_inv_wt         -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=INV  -DARTS_WRITE_POLICY=WT
#   ocr_inv_wb         -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=INV  -DARTS_WRITE_POLICY=WB
#   ocr_inv_wt_purge   -DARTS_MEMORY_MODEL=OCR    -DARTS_COHERENCE_PROTOCOL=INV  -DARTS_WRITE_POLICY=WT -DARTS_RELEASE_POLICY=PURGE
#   wrf_val_wt         -DARTS_MEMORY_MODEL=DB_WRF -DARTS_COHERENCE_PROTOCOL=VAL  -DARTS_WRITE_POLICY=WT
#
# DB_WRF (wrf_val_wt) requires program-ordered write-write conflicts, so some
# correctness deviations are EXPECTED there — they are reported, not silently
# treated as regressions.  OCR-model builds must be clean.
#
# Usage:
#   bash tests/run_all_models.sh                                  # ctest + applications
#   bash tests/run_all_models.sh --models ocr_val_wt,ocr_val_wb  # subset
#   bash tests/run_all_models.sh --no-harness                     # ctest only
#   bash tests/run_all_models.sh --no-ctest                       # applications only
#   bash tests/run_all_models.sh --no-build                       # skip reconfigure/rebuild
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

MODELS="ocr_val_wt ocr_val_wb ocr_val_wt_purge ocr_excl_purge ocr_excl_retain ocr_inv_wt ocr_inv_wb ocr_inv_wt_purge wrf_val_wt"
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

# protocol → ctest build dir / cmake flags
ctest_dir()   { echo "build_$1"; }
model_label() { echo "$1" | tr '[:lower:]' '[:upper:]'; }
model_cmake_flags() {
  case "$1" in
    ocr_val_wt)       echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=VAL -DARTS_WRITE_POLICY=WT" ;;
    ocr_val_wb)       echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=VAL -DARTS_WRITE_POLICY=WB" ;;
    ocr_val_wt_purge) echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=VAL -DARTS_WRITE_POLICY=WT -DARTS_RELEASE_POLICY=PURGE" ;;
    ocr_excl_purge)   echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=EXCL -DARTS_RELEASE_POLICY=PURGE" ;;
    ocr_excl_retain)  echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=EXCL -DARTS_RELEASE_POLICY=RETAIN" ;;
    ocr_inv_wt)       echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=INV -DARTS_WRITE_POLICY=WT" ;;
    ocr_inv_wb)       echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=INV -DARTS_WRITE_POLICY=WB" ;;
    ocr_inv_wt_purge) echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=INV -DARTS_WRITE_POLICY=WT -DARTS_RELEASE_POLICY=PURGE" ;;
    wrf_val_wt)       echo "-DARTS_MEMORY_MODEL=DB_WRF -DARTS_COHERENCE_PROTOCOL=VAL -DARTS_WRITE_POLICY=WT" ;;
  esac
}
# expected CMakeCache values per config
model_model()  { case "$1" in wrf_val_wt) echo DB_WRF;; *) echo OCR;; esac; }
model_proto()  { case "$1" in ocr_excl_*) echo EXCL;; ocr_inv_*) echo INV;; *) echo VAL;; esac; }
# The live second axis, read off the suffix.  ocr_{val,inv}_wt_purge pin BOTH
# ARTS_WRITE_POLICY and ARTS_RELEASE_POLICY away from their defaults, so their
# "timing" is the pair, not a single value.
model_timing() { case "$1" in *_wt_purge) echo "WT+PURGE" ;; *_wt) echo WT;; *_wb) echo WB;; ocr_excl_purge) echo PURGE;; ocr_excl_retain) echo RETAIN;; esac; }

# Configure a build dir to the requested configuration if its cache does not
# match, then build.  Reconfigure forces a full rebuild (compile-flag change).
# The cache check compares ARTS_MEMORY_MODEL, ARTS_COHERENCE_PROTOCOL, and
# the live second axis (ARTS_WRITE_POLICY / ARTS_RELEASE_POLICY, or both for
# the ocr_{val,inv}_wt_purge configs, which pin both away from default).
ensure_build() {
  local dir="$1" model="$2" wantgpu="$3" extra="${4:-}"
  local want_model; want_model="$(model_model "$model")"
  local want_proto; want_proto="$(model_proto "$model")"
  local want_timing; want_timing="$(model_timing "$model")"
  local have_model; have_model="$(grep -E '^ARTS_MEMORY_MODEL:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local have_proto; have_proto="$(grep -E '^ARTS_COHERENCE_PROTOCOL:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local have_timing
  case "$model" in
    ocr_excl_*) have_timing="$(grep -E '^ARTS_RELEASE_POLICY:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)" ;;
    *_wt_purge)
      local have_write have_release
      have_write="$(grep -E '^ARTS_WRITE_POLICY:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
      have_release="$(grep -E '^ARTS_RELEASE_POLICY:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
      have_timing="${have_write}+${have_release}"
      ;;
    *)          have_timing="$(grep -E '^ARTS_WRITE_POLICY:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)" ;;
  esac
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

done

# The application matrix is no longer per-configuration: one benchmark build
# holds every coherence configuration, so it runs once for all of them rather
# than once inside the loop above.
if [ "$DO_HARNESS" = 1 ]; then
  hd="build_release"
  [ "$DO_BUILD" = 1 ] && ensure_build "$hd" ocr_val_wb OFF "-DARTS_BUILD_BENCHMARKS=ON"
  drain_ports
  echo "================= APPLICATION MATRIX ================="
  timeout -k 30 2400 artsrun run -p bentley -b paper-main --nodes 1 \
      --build-dir "$hd" >"$LOGDIR/apps.log" 2>&1
  tally=$( grep -E 'MINORITY REPORT|No disagreement' "$LOGDIR/apps.log" | head -1 )
  APPS_RESULT="${tally:-NORUN}"
  echo "  applications : ${APPS_RESULT}  (full log: $LOGDIR/apps.log)"
fi

echo
echo "===================== SUMMARY (OCR-model configs must be clean; WRF_VAL DB-WRF deviations annotated) ====================="
for m in $MODELS; do
  M="$(model_label "$m")"
  echo "[$M]"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest single : ${RESULT[$m,ctest_single]:-skip}"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest multi  : ${RESULT[$m,ctest_multi]:-skip}"
done
