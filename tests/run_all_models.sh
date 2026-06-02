#!/bin/bash
# Run the ARTS test suites across ALL THREE host memory-consistency models
# (RC / LRC / LC).  Each model has its own build trees:
#
#   model  ctest build   harness build        notes
#   -----  -----------   ------------------   ----------------------------------
#   RC     build_cut     build_release_rc     Release Consistency (sync writeback)
#   LRC    build_lrc     build_release_lrc    Lazy Release (defer writeback)
#   LC     build_lc      build_release_lc     Location Consistency (WEAK)
#
# LC is a WEAK model: a DB-level data race may expose a partial write, so some
# LC correctness deviations are EXPECTED.  They are reported, not silently
# treated as regressions — RC/LRC must be clean; LC deviations are annotated.
#
# Usage:
#   bash tests/run_all_models.sh                 # all models, ctest + harness
#   bash tests/run_all_models.sh --models rc,lrc # subset
#   bash tests/run_all_models.sh --no-harness    # ctest only
#   bash tests/run_all_models.sh --no-ctest      # harness only
#   bash tests/run_all_models.sh --no-build      # skip reconfigure/rebuild
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

MODELS="rc lrc lc"
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

# model → ctest build dir / harness build dir / ARTS_MEMORY_MODEL value
ctest_dir() { case "$1" in rc) echo build_cut;; lrc) echo build_lrc;; lc) echo build_lc;; esac; }
harness_dir() { case "$1" in rc) echo build_release_rc;; lrc) echo build_release_lrc;; lc) echo build_release_lc;; esac; }
model_val() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

# Configure a build dir to the requested model if its cache does not match,
# then build.  Reconfigure forces a full rebuild (compile-flag change).
ensure_build() {
  local dir="$1" model="$2" wantgpu="$3"
  local want; want="$(model_val "$model")"
  local have; have="$(grep -E '^ARTS_MEMORY_MODEL:STRING=' "$dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
  if [ ! -d "$dir" ] || [ "$have" != "$want" ]; then
    echo "  [cfg] $dir → ARTS_MEMORY_MODEL=$want (was '${have:-none}')"
    cmake -GNinja -B "$dir" -DCMAKE_BUILD_TYPE=Release \
          -DARTS_MEMORY_MODEL="$want" -DARTS_USE_GPU="$wantgpu" >/dev/null 2>&1 \
      || { echo "  [cfg] FAILED for $dir"; return 1; }
  fi
  ninja -C "$dir" >/dev/null 2>&1 || { echo "  [build] FAILED for $dir"; return 1; }
  return 0
}

# drain TCP TIME_WAIT on the test ports before a multinode ctest run
drain_ports() { local n=0; until ! ss -tan 2>/dev/null | grep -qE 'LISTEN.*:5000[0-9]'; do sleep 0.2; n=$((n+1)); [ $n -gt 100 ] && break; done; }

declare -A RESULT
for m in $MODELS; do
  echo "================= MODEL: $(model_val "$m") ================="

  if [ "$DO_CTEST" = 1 ]; then
    cd="$(ctest_dir "$m")"
    [ "$DO_BUILD" = 1 ] && ensure_build "$cd" "$m" OFF
    # Single-node arts.cfg + the multinode variant cfgs that
    # register_multinode_test wires via ARTS_CONFIG=arts_<variant>.cfg.
    # A freshly (re)configured build dir does not have these until copied.
    cp "$REPO/configs/local/1n.cfg" "$cd/tests/arts.cfg" 2>/dev/null
    for v in 2n 3n 4n 2n_io; do
      cp "$REPO/configs/local/$v.cfg" "$cd/tests/arts_$v.cfg" 2>/dev/null
    done
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
    [ "$DO_BUILD" = 1 ] && ensure_build "$hd" "$m" OFF
    cp "$REPO/configs/local/1n.cfg" "$hd/benchmarks/apps/arts.cfg" 2>/dev/null
    drain_ports
    timeout -k 30 2400 python3 "$REPO/benchmarks/scripts/correctness_harness.py" \
        --build-dir "$hd" --no-baseline >/tmp/harness_$m.log 2>&1
    tally=$( grep -E '^Tier [ABM]:' /tmp/harness_$m.log | tr '\n' ' ' )
    RESULT["$m,harness"]="${tally:-NORUN}"
    echo "  harness      : ${tally:-NORUN}  (full log: /tmp/harness_$m.log)"
  fi
done

echo
echo "===================== SUMMARY (RC/LRC must be clean; LC weak-model deviations annotated) ====================="
for m in $MODELS; do
  M="$(model_val "$m")"
  echo "[$M]"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest single : ${RESULT[$m,ctest_single]:-skip}"
  [ "$DO_CTEST" = 1 ]   && echo "   ctest multi  : ${RESULT[$m,ctest_multi]:-skip}"
  [ "$DO_HARNESS" = 1 ] && echo "   harness      : ${RESULT[$m,harness]:-skip}"
done
