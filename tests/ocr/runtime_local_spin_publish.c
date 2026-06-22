/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file runtime_local_spin_publish.c
/// @brief TSan-oriented smoke for the local_spin publish-vs-alive ordering in
///        arts_runtime_private_init, and the stop-via-local_spin read in
///        arts_runtime_stop_by_role.
///
/// In arts_runtime_private_init the pointer publish (local_spin[id] = &alive)
/// precedes the alive store (alive = true).  A stopper in
/// arts_runtime_stop_by_role that observes local_spin[id] != NULL between those
/// two stores could write *local_spin[id] = false only to have alive = true
/// overwrite it -> a lost stop (B-local-spin-ordering).  The two stores are
/// also cross-thread accesses to `alive` via the local_spin indirection with no
/// acquire/release fence (volatile only), which TSan should report as a data
/// race if the ordering regresses.
///
/// The race window is closed in practice by the startup barriers (a stop this
/// early is improbable), so this is not a deterministic failure; it is a
/// TSan-instrumented exercise of the publish + stop path.  Under a TSan build
/// it surfaces the cross-thread local_spin/alive access; under non-TSan builds
/// it is a plain clean startup+shutdown smoke.  We drive many worker EDTs so
/// every worker thread runs the loop (reading its own `alive`) while the
/// shutdown stop walk writes `alive=false` through local_spin from the main
/// thread.
///
/// runtime_single (TSan), all configs (protocol-agnostic).
/// exposes_runtime_bug: B-local-spin-ordering (TSan-detected; not
/// deterministic).

#include "arts.h"

#include <stdatomic.h>

#define N_BUSY 16

/// Busy EDTs keep every worker thread spinning in the scheduler loop (reading
/// its own thread-local `alive`) so the cross-thread stop write through
/// local_spin races against the loop read under TSan instrumentation.
void busy_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic unsigned int *done = (_Atomic unsigned int *)depv[0].ptr;
  unsigned int total = (unsigned int)paramv[0];
  /* Last EDT to finish requests shutdown so the stop walk runs concurrently
   * with any still-draining workers. */
  if (done &&
      atomic_fetch_add_explicit(done, 1u, memory_order_relaxed) + 1u == total) {
    arts_shutdown();
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== runtime_local_spin_publish ===\n");

  void *done_ptr = NULL;
  arts_guid_t done_db =
      arts_db_create(&done_ptr, sizeof(_Atomic unsigned int), ARTS_DB_DEFAULT,
                     ARTS_DB_PROP_NONE, NULL);
  atomic_init((_Atomic unsigned int *)done_ptr, 0u);
  arts_db_release(done_db, DB_MODE_RW);

  uint64_t total = N_BUSY;
  for (int i = 0; i < N_BUSY; i++) {
    arts_guid_t e =
        arts_edt_create(busy_edt, 1, &total, 1, &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(done_db, e, 0, DB_MODE_RW);
  }

  arts_printf("PASS runtime_local_spin_publish: %d busy EDTs launched\n",
              N_BUSY);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  arts_printf("LOCAL_SPIN_PUBLISH_DONE\n");
  return 0;
}
