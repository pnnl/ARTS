#!/bin/bash
# Cross-mode binary mismatch detection — exercises 3 build pairs.
set -uo pipefail
ROOT=$(git rev-parse --show-toplevel)

declare -A BIN
BIN[EAGER]="${ROOT}/build_release_eager/tests/coherence_mode_mismatch"
BIN[LAZY]="${ROOT}/build_release_lazy/tests/coherence_mode_mismatch"
BIN[RELAXED]="${ROOT}/build_release_relaxed/tests/coherence_mode_mismatch"

for mode in EAGER LAZY RELAXED; do
  if [ ! -x "${BIN[$mode]}" ]; then
    echo "Build for ${mode} missing — run benchmarks/scripts/run_three_builds.sh first."
    exit 1
  fi
done

PAIRS=(
  "EAGER LAZY"
  "EAGER RELAXED"
  "LAZY RELAXED"
)

OVERALL=0
for pair in "${PAIRS[@]}"; do
  A=$(echo "$pair" | awk '{print $1}')
  B=$(echo "$pair" | awk '{print $2}')
  echo "=== Pair: rank0=${A}, rank1=${B} ==="
  WORKDIR=$(mktemp -d)
  trap 'rm -rf "$WORKDIR"' EXIT
  cp "${ROOT}/configs/local/2n.cfg" "${WORKDIR}/arts.cfg"
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
