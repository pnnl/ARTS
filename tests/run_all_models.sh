#!/bin/bash
# Run the ARTS test suites across ALL THREE host memory-model/protocol
# configurations (eager / lazy / relaxed).  Each has its own build trees:
#
#   model    ctest build     harness build           cmake flags
#   -------  -------------   ---------------------   ------------------------------
#   eager    build_eager     build_release_eager     -DARTS_MEMORY_MODEL=OCR
#                                                    -DARTS_COHERENCE_PROTOCOL=EAGER
#   lazy     build_lazy      build_release_lazy      -DARTS_MEMORY_MODEL=OCR
#                                                    -DARTS_COHERENCE_PROTOCOL=LAZY
#   relaxed  build_relaxed   build_release_relaxed   -DARTS_MEMORY_MODEL=RELAXED
#
# RELAXED is the DB-DRF model: DB-level data races are undefined, so some
# correctness deviations are EXPECTED there — they are reported, not silently
# treated as regressions.  OCR builds (eager/lazy) must be clean.
#
# Usage:
#   bash tests/run_all_models.sh                       # all models, ctest + harness
#   bash tests/run_all_models.sh --models eager,lazy   # subset
#   bash tests/run_all_models.sh --no-harness          # ctest only
#   bash tests/run_all_models.sh --no-ctest            # harness only
#   bash tests/run_all_models.sh --no-build            # skip reconfigure/rebuild
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

MODELS="eager lazy relaxed"
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

# model → ctest build dir / harness build dir / cmake model flags
ctest_dir() { case "$1" in eager) echo build_eager;; lazy) echo build_lazy;; relaxed) echo build_relaxed;; esac; }
harness_dir() { case "$1" in eager) echo build_release_eager;; lazy) echo build_release_lazy;; relaxed) echo build_release_relaxed;; esac; }
model_label() { echo "$1" | tr '[:lower:]' '[:upper:]'; }
model_cmake_flags() {
  case "$1" in
    eager)   echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=EAGER" ;;
    lazy)    echo "-DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=LAZY" ;;
    relaxed) echo "-DARTS_MEMORY_MODEL=RELAXED" ;;
  esac
}
# expected CMakeCache values per model (protocol empty = don't care)
model_mm() { case "$1" in eager|lazy) echo OCR;; relaxed) echo RELAXED;; esac; }
model_proto() { case "$1" in eager) echo EAGER;; lazy) echo LAZY;; relaxed) echo "";; esac; }

# Configure a build dir to the requested model if its cache does not match,
# then build.  Reconfigure forces a full rebuild (compile-flag change).
# The cache check compares ARTS_MEMORY_MODEL AND (for OCR configurations)
# ARTS_COHERENCE_PROTOCOL.
ensure_build() {
  local dir="$1" model="$2" wantgpu="$3" extra="${4:-}"
  local want_mm; want_mm="$(model_mm "$model")"
  local want_proto; want_proto="$(model_proto "$model")"
  local have_mm; have_mm="$(grep -E '^ARTS_MEMORY_MODEL:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local have_proto; have_proto="$(grep -E '^ARTS_COHERENCE_PROTOCOL:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local mismatch=0
  [ "$have_mm" != "$want_mm" ] && mismatch=1
  [ -n "$want_proto" ] && [ "$have_proto" != "$want_proto" ] && mismatch=1
  if [ ! -d "$dir" ] || [ "$mismatch" = 1 ]; then
    echo "  [cfg] $dir → $(model_cmake_flags "$model") (was MM='${have_mm:-none}' PROTO='${have_proto:-none}')"
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
  echo "================= MODEL: $(model_label "$m") ================="

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
echo "===================== SUMMARY (eager/lazy must be clean; RELAXED DB-DRF deviations annotated) ====================="
for m in $MODELS; do
  M="$(model_label "$m")"
  echo "[$M]"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest single : ${RESULT[$m,ctest_single]:-skip}"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest multi  : ${RESULT[$m,ctest_multi]:-skip}"
  [ "$DO_HARNESS" = 1 ] && echo "   harness      : ${RESULT[$m,harness]:-skip}"
done
