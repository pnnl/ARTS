#!/bin/bash
# Run the ARTS test suites across ALL THREE host protocol configurations
# (mrnew_eager / mrnew_lazy / mrmw).  Each has its own build trees:
#
#   protocol     ctest build          harness build                cmake flags
#   -----------  -------------------  ---------------------------  ----------------------------------
#   mrnew_eager  build_mrnew_eager    build_release_mrnew_eager    -DARTS_COHERENCE_PROTOCOL=MRNEW
#                                                                   -DARTS_PROTOCOL_TIMING=EAGER
#   mrnew_lazy   build_mrnew_lazy     build_release_mrnew_lazy     -DARTS_COHERENCE_PROTOCOL=MRNEW
#                                                                   -DARTS_PROTOCOL_TIMING=LAZY
#   mrmw         build_mrmw           build_release_mrmw           -DARTS_COHERENCE_PROTOCOL=MRMW
#
# MRMW is the DB-DRF protocol: DB-level data races are undefined, so some
# correctness deviations are EXPECTED there — they are reported, not silently
# treated as regressions.  MRNEW builds (mrnew_eager/mrnew_lazy) must be clean.
#
# Usage:
#   bash tests/run_all_models.sh                                  # all protocols, ctest + harness
#   bash tests/run_all_models.sh --models mrnew_eager,mrnew_lazy  # subset
#   bash tests/run_all_models.sh --no-harness                     # ctest only
#   bash tests/run_all_models.sh --no-ctest                       # harness only
#   bash tests/run_all_models.sh --no-build                       # skip reconfigure/rebuild
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

MODELS="mrnew_eager mrnew_lazy mrmw"
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
ctest_dir()   { case "$1" in mrnew_eager) echo build_mrnew_eager;; mrnew_lazy) echo build_mrnew_lazy;; mrmw) echo build_mrmw;; esac; }
harness_dir() { case "$1" in mrnew_eager) echo build_release_mrnew_eager;; mrnew_lazy) echo build_release_mrnew_lazy;; mrmw) echo build_release_mrmw;; esac; }
model_label() { echo "$1" | tr '[:lower:]' '[:upper:]'; }
model_cmake_flags() {
  case "$1" in
    mrnew_eager) echo "-DARTS_COHERENCE_PROTOCOL=MRNEW -DARTS_PROTOCOL_TIMING=EAGER" ;;
    mrnew_lazy)  echo "-DARTS_COHERENCE_PROTOCOL=MRNEW -DARTS_PROTOCOL_TIMING=LAZY" ;;
    mrmw)        echo "-DARTS_COHERENCE_PROTOCOL=MRMW" ;;
  esac
}
# expected CMakeCache values per protocol (timing empty for MRMW = don't care)
model_proto()  { case "$1" in mrnew_eager|mrnew_lazy) echo MRNEW;; mrmw) echo MRMW;; esac; }
model_timing() { case "$1" in mrnew_eager) echo EAGER;; mrnew_lazy) echo LAZY;; mrmw) echo "";; esac; }

# Configure a build dir to the requested protocol if its cache does not match,
# then build.  Reconfigure forces a full rebuild (compile-flag change).
# The cache check compares ARTS_COHERENCE_PROTOCOL AND (for MRNEW configurations)
# ARTS_PROTOCOL_TIMING.
ensure_build() {
  local dir="$1" model="$2" wantgpu="$3" extra="${4:-}"
  local want_proto; want_proto="$(model_proto "$model")"
  local want_timing; want_timing="$(model_timing "$model")"
  local have_proto; have_proto="$(grep -E '^ARTS_COHERENCE_PROTOCOL:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local have_timing; have_timing="$(grep -E '^ARTS_PROTOCOL_TIMING:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  local mismatch=0
  [ "$have_proto" != "$want_proto" ] && mismatch=1
  [ -n "$want_timing" ] && [ "$have_timing" != "$want_timing" ] && mismatch=1
  if [ ! -d "$dir" ] || [ "$mismatch" = 1 ]; then
    echo "  [cfg] $dir → $(model_cmake_flags "$model") (was PROTO='${have_proto:-none}' TIMING='${have_timing:-none}')"
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
echo "===================== SUMMARY (mrnew_eager/mrnew_lazy must be clean; MRMW DB-DRF deviations annotated) ====================="
for m in $MODELS; do
  M="$(model_label "$m")"
  echo "[$M]"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest single : ${RESULT[$m,ctest_single]:-skip}"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest multi  : ${RESULT[$m,ctest_multi]:-skip}"
  [ "$DO_HARNESS" = 1 ] && echo "   harness      : ${RESULT[$m,harness]:-skip}"
done
