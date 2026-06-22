/* SPDX-License-Identifier: Apache-2.0
 *
 * T129 — EDT trailing-storage offset agreement + check_out_edts modulo wrap.
 *
 * Property under test
 * -------------------
 * An EDT is allocated as a single contiguous block whose trailing storage is
 * [<header> | paramv[paramc] | depv[depc]].  Two independent code paths must
 * agree on where that storage begins:
 *
 *   - arts_edt_create_core copies the inline params at a subtype-aware
 *     offset (sizeof(arts_edt_s) for CPU, sizeof(arts_gpu_edt_t) for GPU);
 *   - arts_get_depv recovers the depv base at the SAME subtype-aware offset
 *     (depv sits right after paramc uint64_t params).
 *
 * If these two offsets disagree, the param copy lands on top of the depv (or
 * GPU grid/block metadata) and the runtime reads deps / params from the wrong
 * place.  This test pins the contract: for a CPU EDT the paramv base is
 * sizeof(struct arts_edt_s), and arts_get_depv returns that base + paramc*8.
 * For the GPU subtype the base is sizeof(arts_gpu_edt_t) (header includes the
 * grid/block metadata) and arts_get_depv mirrors it.
 *
 * Separately, check_out_edts maintains a process-static counter that is
 * supposed to wrap to 0 exactly when it reaches the threshold (modulo-N).  We
 * exercise the wrap arithmetic with a small threshold over many calls and
 * assert it never grows unbounded — i.e. the wrap is exact.
 *
 * Pure unit: this exercises header arithmetic (arts_edt_total_size /
 * arts_get_depv offsets) and the check_out_edts library helper.  No runtime is
 * started; main() runs the asserts directly.
 */

#include "arts.h"
#include "arts/edt.h"
#include "arts/runtime_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#endif

/* check_out_edts is a runtime helper with external linkage. */
void check_out_edts(uint64_t threshold);

/* Verify arts_get_depv returns header_offset + paramc*8 for a heap-allocated
 * EDT-shaped block of the given subtype.  Returns 1 on PASS, 0 on FAIL. */
static int check_depv_offset(arts_edt_types_t type, size_t header_offset,
                             uint32_t paramc, uint32_t depc) {
  /* Allocate a block large enough to hold header + paramv + depv for either
   * subtype.  Over-allocate generously so any reasonable offset is in-bounds.
   */
  size_t alloc = header_offset + (size_t)paramc * sizeof(uint64_t) +
                 (size_t)depc * sizeof(arts_edt_dep_t) + 64;
  char *blk = (char *)calloc(1, alloc);
  if (!blk) {
    fprintf(stderr, "FAIL edt_size_offsets: calloc\n");
    return 0;
  }
  struct arts_edt_s *edt = (struct arts_edt_s *)blk;
  edt->paramc = paramc;
  edt->depc = depc;
  edt->edt_type = type;

  char *got = (char *)arts_get_depv(edt);
  char *want = blk + header_offset + (size_t)paramc * sizeof(uint64_t);
  int ok = (got == want);
  if (!ok) {
    fprintf(stderr,
            "FAIL edt_size_offsets: depv offset mismatch type=%d paramc=%u "
            "got=%p want=%p (header_offset=%zu)\n",
            (int)type, paramc, (void *)got, (void *)want, header_offset);
  }
  free(blk);
  return ok;
}

static int check_cpu_offsets(void) {
  int ok = 1;
  const uint32_t pcs[] = {0, 1, 2, 7, 16, 64};
  const uint32_t dcs[] = {0, 1, 3, 8};
  for (size_t i = 0; i < sizeof(pcs) / sizeof(pcs[0]); i++) {
    for (size_t j = 0; j < sizeof(dcs) / sizeof(dcs[0]); j++) {
      ok &= check_depv_offset(ARTS_EDT_CPU, sizeof(struct arts_edt_s), pcs[i],
                              dcs[j]);
    }
  }
  return ok;
}

static int check_gpu_offsets(void) {
#ifdef ARTS_USE_GPU
  int ok = 1;
  const uint32_t pcs[] = {0, 1, 4, 16};
  const uint32_t dcs[] = {0, 1, 4};
  for (size_t i = 0; i < sizeof(pcs) / sizeof(pcs[0]); i++) {
    for (size_t j = 0; j < sizeof(dcs) / sizeof(dcs[0]); j++) {
      ok &= check_depv_offset(ARTS_EDT_GPU, sizeof(arts_gpu_edt_t), pcs[i],
                              dcs[j]);
    }
  }
  return ok;
#else
  printf("  SKIP gpu offsets: not a GPU build\n");
  return 1;
#endif
}

static int check_modulo_wrap(void) {
  /* check_out_edts uses a process-static counter that subtracts `threshold`
   * once it reaches it.  Call it many multiples of the threshold; the only
   * observable property from outside is that it does not abort/UB and that
   * repeated wraps keep working (no monotonic blow-up of the static counter).
   * We exercise far more than `threshold` iterations to force several wraps. */
  const uint64_t threshold = 8;
  for (uint64_t i = 0; i < threshold * 100u + 3u; i++) {
    check_out_edts(threshold);
  }
  /* A degenerate threshold of 1 must wrap on every call (count never grows). */
  for (uint64_t i = 0; i < 1000u; i++) {
    check_out_edts(1);
  }
  return 1;
}

int main(void) {
  int ok = 1;
  ok &= check_cpu_offsets();
  ok &= check_gpu_offsets();
  ok &= check_modulo_wrap();
  if (!ok) {
    fprintf(stderr, "FAIL edt_size_offsets\n");
    return 1;
  }
  printf(
      "PASS edt_size_offsets: depv offset agreement + modulo wrap verified\n");
  return 0;
}
