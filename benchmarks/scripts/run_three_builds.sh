#!/bin/bash
# Configures, builds, and runs harness for MRNEW+EAGER, MRNEW+LAZY, and MRMW.
set -uo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

OVERALL_FAIL=0

# mrnew_eager: MRNEW protocol with EAGER timing
cmake -GNinja -Bbuild_release_mrnew_eager -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_COHERENCE_PROTOCOL=MRNEW -DARTS_PROTOCOL_TIMING=EAGER \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_mrnew_eager 2>&1 | tail -3 || { echo "FAIL: mrnew_eager build"; OVERALL_FAIL=1; }

# mrnew_lazy: MRNEW protocol with LAZY timing (default)
cmake -GNinja -Bbuild_release_mrnew_lazy -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_COHERENCE_PROTOCOL=MRNEW -DARTS_PROTOCOL_TIMING=LAZY \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_mrnew_lazy 2>&1 | tail -3 || { echo "FAIL: mrnew_lazy build"; OVERALL_FAIL=1; }

# mrmw: MRMW protocol (DB-DRF, no per-DB coherence)
cmake -GNinja -Bbuild_release_mrmw -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_COHERENCE_PROTOCOL=MRMW \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_mrmw 2>&1 | tail -3 || { echo "FAIL: mrmw build"; OVERALL_FAIL=1; }

for cfg in mrnew_eager mrnew_lazy mrmw; do
  builddir="build_release_${cfg}"
  echo "=== ${cfg} ctest single_node ==="
  ( cd "${builddir}" && ctest -L single_node --output-on-failure ) || \
    { echo "FAIL: ${cfg} single_node"; OVERALL_FAIL=1; }
  echo "=== ${cfg} ctest multinode (excl 4n/5n / lock_req_before_create) ==="
  ( cd "${builddir}" && ctest -L multinode --output-on-failure \
        -E "(_4n|_5n|coherence_lock_req_before_create)" ) || \
    { echo "FAIL: ${cfg} multinode"; OVERALL_FAIL=1; }
  if [ -d "${builddir}/benchmarks/apps" ]; then
    echo "=== ${cfg} correctness harness ==="
    ( cd "${builddir}/benchmarks" && \
      python3 "${ROOT}/benchmarks/scripts/correctness_harness.py" \
        --build-dir "${ROOT}/${builddir}" ) || \
      { echo "FAIL: ${cfg} harness"; OVERALL_FAIL=1; }
  fi
done

if [ "${OVERALL_FAIL}" -ne 0 ]; then
  echo "=== OVERALL: FAIL ==="
  exit 1
else
  echo "=== OVERALL: PASS ==="
fi
