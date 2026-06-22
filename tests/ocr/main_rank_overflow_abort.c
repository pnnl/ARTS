/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file main_rank_overflow_abort.c
/// @brief arts_rt aborts (exit 1) when the configured rank count exceeds the
///        GUID layout limit (ARTS_GUID_RANK_MASK - 1 == 16382).
///
/// main.c sets arts_global_rank_count = config.table_length and, BEFORE any
/// transport setup or process launch, guards it:
///   if (rank_count > ARTS_GUID_RANK_MASK - 1U) ARTS_ERROR(...);
/// ARTS_ERROR calls arts_abort(1) (process exits 1).  The check sits at the
/// very top so an over-limit node count aborts before the launcher would ever
/// fork that many processes.
///
/// ARTS reads each config variable from the environment when present
/// (arts_config_find_variable does getenv(name)), so this test sets
/// node_count=99999 in its own main() before calling arts_rt.  99999 >> 16382,
/// so the overflow guard fires and the process must exit with status 1.  The
/// guard runs before transport setup, so no processes are launched.  ctest
/// expects exit code 1 (FAIL_REGULAR_EXPRESSION on a clean token, or a
/// WILL_FAIL / non-zero exit expectation set by the integrator).
///
/// runtime_single, all configs (the guard sits above the coherence layer).
/// exposes_runtime_bug: no — this asserts the guard works (clean abort exit 1).

#include "arts.h"

#include <stdio.h>
#include <stdlib.h>

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Never reached: the rank-count overflow guard aborts during arts_rt bring-up
   * before main_edt is ever scheduled. */
  arts_printf("UNEXPECTED: main_edt ran despite rank overflow\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Force an over-limit configured rank count via the env override path.  The
   * GUID rank field is 14 bits (max usable 16382); 99999 exceeds it. */
  setenv("node_count", "99999", 1);

  arts_rt(argc, argv);

  /* If arts_rt returned, the overflow guard did NOT abort -> regression. */
  printf("RANK_OVERFLOW_NO_ABORT\n");
  return 0;
}
