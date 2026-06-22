/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file shutdown_abort_eof.c
/// @brief A non-master rank arts_abort()s; peers detect the socket EOF and exit
///        cleanly via the (non-CAS-gated) arts_runtime_stop EOF path.
///
/// arts_abort does a hard exit(code) with no broadcast/drain — other ranks
/// learn of the departure only through socket EOF.  The transport RX EOF path
/// calls arts_runtime_stop() directly (NOT the CAS-gated
/// arts_enter_shutdown_state), so a peer that learns of shutdown via EOF stops
/// its threads but leaves shutdown_state == 0.  The properties under test: (a)
/// a non-master abort makes every surviving rank detect EOF and exit cleanly
/// (no hang), and (b) the EOF-driven stop leaving shutdown_state == 0 is
/// harmless to a clean exit.
///
/// An EDT placed on a NON-master rank (the last rank) calls arts_abort with a
/// sentinel code.  That rank exits with the code; the master (and any other
/// peers) detect EOF and return 0 from arts_rt.  Each surviving rank prints a
/// clean token.  ctest TIMEOUT reaps an EOF-detection hang regression.
///
/// runtime_multinode (2n/3n/4n/2n_io).  Self-skips cleanly to a single-node
/// clean shutdown when only one rank is present.
/// all configs (shutdown sits above the coherence layer).

#include "arts.h"

#include <stdio.h>

#define ABORT_CODE 37u

void abort_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  non-master rank %u aborting (code %u)\n",
              arts_get_current_rank(), ABORT_CODE);
  /* Hard exit: no broadcast/drain; peers discover via socket EOF. */
  arts_abort((uint8_t)ABORT_CODE);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== shutdown_abort_eof ===\n");

  unsigned int ranks = arts_get_total_ranks();
  if (ranks < 2) {
    arts_printf("SKIP shutdown_abort_eof: requires 2+ ranks (got %u)\n", ranks);
    arts_shutdown();
    return;
  }

  unsigned int origin = ranks - 1; /* a non-master rank */
  arts_printf(
      "PASS shutdown_abort_eof: %u ranks, aborting non-master rank %u\n", ranks,
      origin);
  arts_edt_create(abort_edt, 0, NULL, 0, &(arts_edt_hint_t){.rank = origin});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* Surviving ranks reach here after detecting EOF and exiting cleanly. */
  printf("[rank %u] ABORT_EOF_EXIT\n", arts_get_current_rank());
  return 0;
}
