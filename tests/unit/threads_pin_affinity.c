/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file threads_pin_affinity.c
/// @brief Verifies arts_thread_init applied the per-thread CPU affinity: each
///        worker thread's effective affinity matches its assigned pu_id.
///
/// When config->pin_threads is on (non-Apple), arts_thread_init builds a
/// cpu_set_t with mask[i].pu_id for each spawned thread via
/// pthread_attr_setaffinity_np (and pins thread 0 via pthread_setaffinity_np).
/// The observable invariant: each worker thread, queried with
/// pthread_getaffinity_np, is bound to exactly the single PU recorded in its
/// arts_thread_info.pu_id.  We assert this in the per-worker startup hook
/// init_per_worker, which the runtime calls on each worker thread after
/// affinity has been applied and before scheduler execution.
///
/// config_specific: pin_threads=on (Linux only).  When pin is off, the affinity
/// mask covers more than one CPU and we cannot pin a single pu_id -> that
/// worker reports a SKIP for its check (the test still passes).  On Apple
/// (__APPLE__) there is no pthread_getaffinity_np and pinning is a no-op -> the
/// whole hook self-skips.  Uses only the public init_per_worker hook +
/// arts_thread_info white-box pu_id read.
///
/// runtime_single, all configs (protocol-agnostic; affinity behavior is
/// config).

#define _GNU_SOURCE
#include "arts.h"
#include "arts/runtime_state.h"

#include <stdatomic.h>

#if !defined(__APPLE__)
#include <pthread.h>
#include <sched.h>
#endif

/// Per-worker hook: runs on each worker thread after affinity is applied.
/// Asserts the thread is bound to exactly its assigned pu_id when pinned.
void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                     char **argv) {
  (void)node_id;
  (void)worker_id;
  (void)argc;
  (void)argv;
#if defined(__APPLE__)
  arts_printf("  SKIP pin affinity: Apple (no pthread_getaffinity_np)\n");
#else
  cpu_set_t set;
  CPU_ZERO(&set);
  unsigned int my_pu = arts_thread_info.pu_id;
  if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    arts_printf("  WARN pin affinity: getaffinity failed (worker pu_id=%u)\n",
                my_pu);
    return;
  }
  int ncpus = CPU_COUNT(&set);
  if (ncpus == 1) {
    /* Pinned: the single set CPU must be this thread's pu_id. */
    if (CPU_ISSET((int)my_pu, &set)) {
      arts_printf("  PASS pin affinity: worker bound to pu_id=%u\n", my_pu);
    } else {
      arts_printf("  FAIL pin affinity: worker pu_id=%u not the single bound "
                  "CPU\n",
                  my_pu);
    }
  } else {
    /* Not pinned (pin_threads off): affinity spans %d CPUs -> nothing to pin.
     */
    arts_printf("  SKIP pin affinity: not pinned (worker pu_id=%u, %d CPUs in "
                "affinity)\n",
                my_pu, ncpus);
  }
#endif
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("=== threads_pin_affinity ===\n");
  arts_printf(
      "PASS threads_pin_affinity: per-worker affinity hook installed\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
