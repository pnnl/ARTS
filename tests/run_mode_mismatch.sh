#!/bin/bash
# Launches rank 0 from build_release_rc and rank 1 from build_release_lrc.
# Expects a fatal error within 30s due to wire-protocol mode mismatch.
set -uo pipefail
ROOT=$(git rev-parse --show-toplevel)
RC_BIN="${ROOT}/build_release_rc/tests/coherence_mode_mismatch"
LRC_BIN="${ROOT}/build_release_lrc/tests/coherence_mode_mismatch"

if [ ! -x "$RC_BIN" ] || [ ! -x "$LRC_BIN" ]; then
  echo "Both RC + LRC builds must exist first.  Run run_both_builds.sh."
  exit 1
fi

# Use 2-node config in cwd
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
cp "${ROOT}/configs/local/2n.cfg" "${WORKDIR}/arts.cfg"

echo "Launching rank 0 (RC) and rank 1 (LRC) — expect FATAL within 30s..."
( cd "${WORKDIR}" && timeout 30s "$RC_BIN" ) &
PID0=$!
sleep 1
( cd "${WORKDIR}" && timeout 30s "$LRC_BIN" ) &
PID1=$!

set +e
wait $PID0; RC0=$?
wait $PID1; RC1=$?
set -e

if [ "${RC0}" != 0 ] || [ "${RC1}" != 0 ]; then
  echo "PASS: mismatch detected (rank0=$RC0, rank1=$RC1)"
  exit 0
fi
echo "FAIL: ranks completed normally despite mode mismatch"
exit 1
