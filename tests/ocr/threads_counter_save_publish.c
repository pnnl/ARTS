/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file threads_counter_save_publish.c
/// @brief TSan-oriented smoke for the counter-save publish ordering at thread
///        exit in arts_thread_loop / arts_thread_main_join.
///
/// On thread exit each thread copies its live counter totals into
/// arts_node_info.saved_counters[tid][...] and then publishes exit by storing
/// arts_node_info.live_counters[tid] = NULL (the signal a concurrent
/// counter-capture/aggregation reader uses to know saved_counters[tid] is
/// final).  Both the saved[] writes and the NULL store are plain (non-atomic,
/// non-fenced) stores, so a concurrent reader observing
/// live_counters[tid]==NULL is not guaranteed under the C memory model to see
/// the completed saved[] writes (B-counter-save-publish; counters are
/// diagnostic so severity is low, but it is a genuine data race the missing
/// release fence introduces).
///
/// Counters are diagnostic, so there is no deterministic correctness assertion;
/// this is a TSan-instrumented exercise of the per-thread counter save + exit
/// publish across many worker threads.  We run a multi-EDT workload (every
/// worker thread accumulates counter events) then shut down so each thread runs
/// the save+NULL-publish epilogue while aggregation reads live/saved counters.
///
/// runtime_single (TSan), all configs (protocol-agnostic).
/// exposes_runtime_bug: B-counter-save-publish (TSan-detected; not
/// deterministic).

#include "arts.h"

#include <stdatomic.h>

#define N_WORK 24

void work_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic unsigned int *done = (_Atomic unsigned int *)depv[0].ptr;
  unsigned int total = (unsigned int)paramv[0];
  /* Touch the DB so per-thread counters advance, then the last EDT requests
   * shutdown so every worker runs its counter-save + live_counters[tid]=NULL
   * exit publish concurrently with counter aggregation. */
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

  arts_printf("=== threads_counter_save_publish ===\n");

  void *done_ptr = NULL;
  arts_guid_t done_db =
      arts_db_create(&done_ptr, sizeof(_Atomic unsigned int), ARTS_DB_DEFAULT,
                     ARTS_DB_PROP_NONE, NULL);
  atomic_init((_Atomic unsigned int *)done_ptr, 0u);
  arts_db_release(done_db, DB_MODE_RW);

  uint64_t total = N_WORK;
  for (int i = 0; i < N_WORK; i++) {
    arts_guid_t e =
        arts_edt_create(work_edt, 1, &total, 1, &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(done_db, e, 0, DB_MODE_RW);
  }

  arts_printf("PASS threads_counter_save_publish: %d work EDTs launched\n",
              N_WORK);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  arts_printf("COUNTER_SAVE_PUBLISH_DONE\n");
  return 0;
}
