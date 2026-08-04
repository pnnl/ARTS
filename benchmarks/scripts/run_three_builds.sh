#!/bin/bash
# Configures, builds, and runs harness for ocr_rcu_{home,owner} and wrf_val_wt.
set -uo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

OVERALL_FAIL=0

# ocr_val_wt: VAL protocol, HOME placement
cmake -GNinja -Bbuild_release_ocr_val_wt -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=VAL -DARTS_WRITE_POLICY=WT \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_ocr_val_wt 2>&1 | tail -3 || { echo "FAIL: ocr_val_wt build"; OVERALL_FAIL=1; }

# ocr_val_wb: VAL protocol, OWNER placement (default)
cmake -GNinja -Bbuild_release_ocr_val_wb -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_MEMORY_MODEL=OCR -DARTS_COHERENCE_PROTOCOL=VAL -DARTS_WRITE_POLICY=WB \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_ocr_val_wb 2>&1 | tail -3 || { echo "FAIL: ocr_val_wb build"; OVERALL_FAIL=1; }

# wrf_val_wt: VAL protocol under the DB-WRF memory model
cmake -GNinja -Bbuild_release_wrf_val_wt -DCMAKE_BUILD_TYPE=Release \
      -DARTS_USE_GPU=OFF -DARTS_MEMORY_MODEL=DB_WRF -DARTS_COHERENCE_PROTOCOL=VAL -DARTS_WRITE_POLICY=WT \
      -DARTS_BUILD_BENCHMARKS=ON 2>&1 | tail -3
ninja -C build_release_wrf_val_wt 2>&1 | tail -3 || { echo "FAIL: wrf_val_wt build"; OVERALL_FAIL=1; }

for cfg in ocr_val_wt ocr_val_wb wrf_val_wt; do
  builddir="build_release_${cfg}"
  echo "=== ${cfg} ctest single_node ==="
  ( cd "${builddir}" && ctest -L single_node --output-on-failure ) || \
    { echo "FAIL: ${cfg} single_node"; OVERALL_FAIL=1; }
  echo "=== ${cfg} ctest multinode (excl 4n/5n / excl_req_before_create) ==="
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
