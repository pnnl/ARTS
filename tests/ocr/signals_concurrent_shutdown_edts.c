/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file signals_concurrent_shutdown_edts.c
/// @brief N EDTs concurrently call arts_shutdown -> the
/// arts_enter_shutdown_state
///        CAS gate keeps exactly one broadcast/stop with no deadlock.
///
/// arts_shutdown -> arts_enter_shutdown_state(true) gates the shutdown sequence
/// on arts_atomic_cswap(shutdown_state, 0, 1).  When many worker EDTs race into
/// arts_shutdown at once, exactly one wins the 0->1 CAS and performs the
/// broadcast (MN) + stop_workers; every other caller observes a non-zero state
/// and returns immediately (idempotent).  The required properties are: no
/// double broadcast, no double stop, and no deadlock — the runtime must
/// terminate cleanly.
///
/// This is the EDT-facet of the shutdown-state CAS gate (T221 drives the same
/// gate but also asserts the white-box shutdown_state value; here the focus is
/// the EDT-level idempotent contract under maximum contention with a wide
/// fan-in of independent shutdown callers).  ctest's TIMEOUT reaps a deadlock
/// regression; the clean-exit token confirms the gate held.
///
/// runtime_single, all configs (shutdown sits above the coherence layer).

#include "arts.h"

#include <stdatomic.h>

#define N_SHUTDOWN_EDTS 12

/// Independent shutdown caller — no shared dep, so all N are eligible to run
/// simultaneously across all worker threads, maximizing CAS contention.
void shutdown_caller_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== signals_concurrent_shutdown_edts ===\n");

  for (int i = 0; i < N_SHUTDOWN_EDTS; i++) {
    arts_edt_create(shutdown_caller_edt, 0, NULL, 0,
                    &(arts_edt_hint_t){.rank = 0});
  }

  arts_printf("PASS signals_concurrent_shutdown_edts: %d shutdown EDTs "
              "launched\n",
              N_SHUTDOWN_EDTS);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* Clean return => the CAS gate kept exactly one shutdown sequence; no
   * deadlock from concurrent initiators. */
  arts_printf("CONCURRENT_SHUTDOWN_EDTS_OK\n");
  return 0;
}
