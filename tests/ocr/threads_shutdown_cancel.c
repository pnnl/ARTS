/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file threads_shutdown_cancel.c
/// @brief Drives arts_thread_main_join's timed-join -> pthread_cancel -> leak
///        escalation by leaving one worker busy past alive=false.
///
/// arts_thread_main_join joins workers 1..N-1 with a 1.5s pthread_timedjoin_np;
/// on timeout it pthread_cancels the thread and tries a second 0.5s timed join;
/// on a second failure it leaks the thread and continues (the process is about
/// to exit anyway).  A worker that is stuck inside a long EDT cannot observe
/// alive=false until the EDT returns, so a deliberately slow EDT forces the
/// timed-join to expire and the cancel escalation to fire.  The property under
/// test is liveness: the process must still terminate within the join+cancel
/// budget and free mask/node_thread_list (no hang).
///
/// We request shutdown, then busy-spin one worker EDT past the join deadline.
/// The spin is BOUNDED (a wall-clock budget, not an infinite loop), comfortably
/// shorter than the ctest TIMEOUT but longer than the 1.5s join deadline so the
/// cancel path is taken.  ctest's per-test TIMEOUT is the backstop for a real
/// hang regression.
///
/// runtime_single, all configs (protocol-agnostic).

#include "arts.h"

#include <time.h>

/// Wall-clock budget the slow worker spins after shutdown is requested.  Must
/// exceed the 1.5s join deadline (so the cancel escalation fires) yet stay well
/// under any reasonable ctest TIMEOUT.
#define SLOW_SPIN_NS (2200ull * 1000ull * 1000ull) /* 2.2 s */

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/// Requests shutdown, then busy-spins so this worker stays inside an EDT past
/// alive=false, forcing arts_thread_main_join's timed-join to expire and the
/// pthread_cancel escalation to run.
void slow_worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  slow worker: requesting shutdown then spinning\n");
  arts_shutdown();
  uint64_t deadline = now_ns() + SLOW_SPIN_NS;
  volatile uint64_t sink = 0;
  while (now_ns() < deadline) {
    sink += 1;
  }
  (void)sink;
  arts_printf("  slow worker: spin budget elapsed, returning\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== threads_shutdown_cancel ===\n");
  arts_printf("PASS threads_shutdown_cancel: launching slow worker\n");

  arts_edt_create(slow_worker_edt, 0, NULL, 0, &(arts_edt_hint_t){.rank = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* arts_rt returned => join (and possibly cancel) completed and the process is
   * exiting cleanly; mask/node_thread_list were freed in global cleanup. */
  arts_printf("SHUTDOWN_CANCEL_DONE\n");
  return 0;
}
