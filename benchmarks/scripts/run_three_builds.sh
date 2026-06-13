#!/bin/bash
# Configures, builds, and runs harness for EAGER, LAZY, and RELAXED.
set -uo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

OVERALL_FAIL=0

# eager: OCR model with EAGER coherence protocol
cmake -GNinja -Bbuild_release_eager -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=EAGER \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_eager 2>&1 | tail -3 || { echo "FAIL: eager build"; OVERALL_FAIL=1; }

# lazy: OCR model with LAZY coherence protocol
cmake -GNinja -Bbuild_release_lazy -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=LAZY \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_lazy 2>&1 | tail -3 || { echo "FAIL: lazy build"; OVERALL_FAIL=1; }

# relaxed: RELAXED model (no per-DB coherence protocol)
cmake -GNinja -Bbuild_release_relaxed -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_MEMORY_MODEL=RELAXED \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_relaxed 2>&1 | tail -3 || { echo "FAIL: relaxed build"; OVERALL_FAIL=1; }

for cfg in eager lazy relaxed; do
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
