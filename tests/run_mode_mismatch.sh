#!/bin/bash
# Cross-mode binary mismatch detection — exercises 3 build pairs.
set -uo pipefail
ROOT=$(git rev-parse --show-toplevel)

declare -A BIN
BIN[MRNEW_EAGER]="${ROOT}/build_release_mrnew_eager/tests/coherence_mode_mismatch"
BIN[MRNEW_LAZY]="${ROOT}/build_release_mrnew_lazy/tests/coherence_mode_mismatch"
BIN[MRMW]="${ROOT}/build_release_mrmw/tests/coherence_mode_mismatch"

for mode in MRNEW_EAGER MRNEW_LAZY MRMW; do
  if [ ! -x "${BIN[$mode]}" ]; then
    echo "Build for ${mode} missing — run benchmarks/scripts/run_three_builds.sh first."
    exit 1
  fi
done

PAIRS=(
  "MRNEW_EAGER MRNEW_LAZY"
  "MRNEW_EAGER MRMW"
  "MRNEW_LAZY MRMW"
)

OVERALL=0
for pair in "${PAIRS[@]}"; do
  A=$(echo "$pair" | awk '{print $1}')
  B=$(echo "$pair" | awk '{print $2}')
  echo "=== Pair: rank0=${A}, rank1=${B} ==="
  WORKDIR=$(mktemp -d)
  trap 'rm -rf "$WORKDIR"' EXIT
  cp "${ROOT}/configs/local/laptop/2n.cfg" "${WORKDIR}/arts.cfg"
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
