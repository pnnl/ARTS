#!/bin/bash
# Configures, builds, and runs harness for RC, LRC, and LC.
set -uo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

OVERALL_FAIL=0

for mode in RC LRC LC; do
  builddir="build_release_${mode,,}"
  echo "=== ${mode} build ==="
  cmake -GNinja "-B${builddir}" -DCMAKE_BUILD_TYPE=Release \
        -DARTS_USE_GPU=OFF "-DARTS_MEMORY_MODEL=${mode}" 2>&1 | tail -3
  ninja -C "${builddir}" 2>&1 | tail -3 || { echo "FAIL: ${mode} build"; OVERALL_FAIL=1; }
done

for mode in RC LRC LC; do
  builddir="build_release_${mode,,}"
  echo "=== ${mode} ctest single_node ==="
  ( cd "${builddir}" && ctest -L single_node --output-on-failure ) || \
    { echo "FAIL: ${mode} single_node"; OVERALL_FAIL=1; }
  echo "=== ${mode} ctest multinode (excl 4n/5n / lock_req_before_create) ==="
  ( cd "${builddir}" && ctest -L multinode --output-on-failure \
        -E "(_4n|_5n|coherence_lock_req_before_create)" ) || \
    { echo "FAIL: ${mode} multinode"; OVERALL_FAIL=1; }
  if [ -d "${builddir}/benchmarks/apps" ]; then
    echo "=== ${mode} correctness harness ==="
    ( cd "${builddir}/benchmarks" && \
      python3 "${ROOT}/benchmarks/scripts/correctness_harness.py" \
        --build-dir "${ROOT}/${builddir}" ) || \
      { echo "FAIL: ${mode} harness"; OVERALL_FAIL=1; }
  fi
done

if [ "${OVERALL_FAIL}" -ne 0 ]; then
  echo "=== OVERALL: FAIL ==="
  exit 1
else
  echo "=== OVERALL: PASS ==="
fi
