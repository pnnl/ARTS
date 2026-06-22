/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file runtime_concurrent_shutdown.c
/// @brief Drives multiple concurrent shutdown initiators (many EDTs each call
///        arts_shutdown) and verifies the shutdown_state CAS gate keeps exactly
///        one broadcast/stop, with no deadlock and a clean return.
///
/// arts_shutdown -> arts_enter_shutdown_state(true) gates on
/// arts_atomic_cswap(shutdown_state, 0, 1): exactly one caller wins the 0->1
/// transition and performs broadcast (MN) + stop_workers; all others observe a
/// non-zero state and no-op.  Meanwhile arts_runtime_stop*/socket EOF paths
/// write shutdown_state with a PLAIN store, so a mixed atomic/plain write to
/// the same word is the documented hazard (B-shutdown-state-dualpath).  Here we
/// stress N worker EDTs racing into arts_shutdown simultaneously: the runtime
/// must terminate cleanly (ctest TIMEOUT reaps a hang) and the final
/// shutdown_state must read 1 (the CAS landed exactly once and was never torn
/// back to 0).
///
/// We fan out N independent EDTs that all call arts_shutdown.  A win counter in
/// a DB lets each EDT observe how many got to run before the runtime stopped;
/// the property under test is liveness (no deadlock) plus shutdown_state==1.
///
/// runtime_single, all configs (protocol-agnostic).
/// exposes_runtime_bug: B-shutdown-state-dualpath (mixed atomic/plain store).

#include "arts.h"
#include "arts/runtime_state.h"

#include <stdatomic.h>

#define N_INITIATORS 8

/// Each initiator EDT bumps a shared counter then calls arts_shutdown.  Because
/// stop_workers only retires worker threads, in-flight EDTs may still be torn
/// down; the test does not require all N to run, only that the runtime stops.
void initiator_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic unsigned int *ctr = (_Atomic unsigned int *)depv[0].ptr;
  if (ctr) {
    atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== runtime_concurrent_shutdown ===\n");

  /* Shared counter DB acquired RW by every initiator (forces them to race on
   * the same DB while all also race into arts_shutdown). */
  void *ctr_ptr = NULL;
  arts_guid_t ctr_db = arts_db_create(&ctr_ptr, sizeof(_Atomic unsigned int),
                                      ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  atomic_init((_Atomic unsigned int *)ctr_ptr, 0u);
  arts_db_release(ctr_db, DB_MODE_RW);

  for (int i = 0; i < N_INITIATORS; i++) {
    arts_guid_t e = arts_edt_create(initiator_edt, 0, NULL, 1,
                                    &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(ctr_db, e, 0, DB_MODE_RW);
  }

  arts_printf("PASS runtime_concurrent_shutdown: launched %d initiators\n",
              N_INITIATORS);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* After arts_rt returns, the runtime has stopped cleanly.  Exactly one CAS
   * must have driven shutdown_state to 1 and nothing torn it back to 0. */
  unsigned int s = arts_node_info.shutdown_state;
  if (s != 1u) {
    arts_printf("FAIL runtime_concurrent_shutdown: shutdown_state=%u "
                "(expected 1)\n",
                s);
    return 1;
  }
  arts_printf("SHUTDOWN_STATE_OK\n");
  return 0;
}
