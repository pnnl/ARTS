#!/bin/bash
# Cross-mode binary mismatch detection — exercises 3 build pairs.
set -uo pipefail
ROOT=$(git rev-parse --show-toplevel)

declare -A BIN
BIN[RCU_EAGER]="${ROOT}/build_release_ocr_rcu_eager/tests/coherence_mode_mismatch"
BIN[RCU_LAZY]="${ROOT}/build_release_ocr_rcu_lazy/tests/coherence_mode_mismatch"
BIN[WRF_RCU]="${ROOT}/build_release_wrf_rcu_eager/tests/coherence_mode_mismatch"

for mode in RCU_EAGER RCU_LAZY WRF_RCU; do
  if [ ! -x "${BIN[$mode]}" ]; then
    echo "Build for ${mode} missing — run benchmarks/scripts/run_three_builds.sh first."
    exit 1
  fi
done

PAIRS=(
  "RCU_EAGER RCU_LAZY"
  "RCU_EAGER WRF_RCU"
  "RCU_LAZY WRF_RCU"
)

OVERALL=0
for pair in "${PAIRS[@]}"; do
  A=$(echo "$pair" | awk '{print $1}')
  B=$(echo "$pair" | awk '{print $2}')
  echo "=== Pair: rank0=${A}, rank1=${B} ==="
  WORKDIR=$(mktemp -d)
  trap 'rm -rf "$WORKDIR"' EXIT
  cp "${ROOT}/configs/local/test/2n.cfg" "${WORKDIR}/arts.cfg"
  ( cd "${WORKDIR}" && timeout 30s "${BIN[$A]}" ) &
  P0=$!
  sleep 1
  ( cd "${WORKDIR}" && timeout 30s "${BIN[$B]}" ) &
  P1=$!
  set +e
  wait $P0; R0=$?
  wait $P1; R1=$?
  set -e
  if [ "${R0}" != 0 ] || [ "${R1}" != 0 ]; then
    echo "  PASS: mismatch detected (${A}<->${B}: rank0=${R0} rank1=${R1})"
  else
    echo "  FAIL: ${A}<->${B} ranks completed normally"
    OVERALL=1
  fi
done

exit $OVERALL
