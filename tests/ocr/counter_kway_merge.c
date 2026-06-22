/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/// @file counter_kway_merge.c
/// @brief EXPOSES B135: the PERIODIC capture k-way merge under-allocates its
///        output arrays to max_captures (largest single source) while it emits
///        one entry per DISTINCT epoch across all sources.  Disjoint per-thread
///        epoch sets make the distinct count exceed max_captures and the merge
///        writes past the allocation (heap overflow).
///
/// Targets arts_compute_node_reduced_captures (counter.c) [node-level] and, on
/// multinode, arts_merge_capture_histories.  Both size out_epochs/out_values to
/// max_captures = max over sources of that source's capture count, then write
/// (*out)[captures_written++] once per distinct epoch.  If thread A captured
/// epochs {1,3,5} and thread B {2,4,6} (each count 3, max 3) the union has 6
/// distinct epochs -> writes indices 0..5 into a length-3 buffer.
///
/// Disjoint epoch sets arise naturally when worker threads are live during
/// DIFFERENT capture passes: the capture thread skips a thread whose
/// live_counters[t] is NULL (not yet registered / already closed) for that
/// pass, so threads with staggered activity windows accumulate complementary
/// epoch sets.  This workload maximises that by issuing many short, staggered
/// bursts across all workers interleaved with sub-interval and super-interval
/// stalls (forcing both on-time and late capture branches) over many capture
/// intervals.
///
/// The overflow occurs in the shutdown counter-write path
/// (arts_counter_write_node).  Under ASan it is a heap-buffer-overflow ->
/// abort/crash; this test is registered WITHOUT a PASS_REGULAR_EXPRESSION so
/// the crash surfaces as a failure documenting the defect (per the
/// bug-exposing-test convention).  It is NOT masked.
///
/// Config-specific on the counter axis: needs PERIODIC counters enabled
/// (default counters.cfg and full_counters.cfg both enable PERIODIC CLUSTER
/// counters, so the merge path runs).  Config-agnostic across coherence
/// protocols.  A stranded run is reaped by the ctest TIMEOUT.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define BURSTS 24
#define BURST_EDTS 16

static void sleep_ns(uint64_t ns) {
  struct timespec ts = {(time_t)(ns / 1000000000ULL),
                        (long)(ns % 1000000000ULL)};
  (void)nanosleep(&ts, NULL);
}

/* Leaf does a short variable stall so different workers stay live across
   different capture passes, producing disjoint per-thread epoch sets. */
void burst_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  uint64_t spin = (paramc > 0) ? paramv[0] : 0;
  /* Stagger per-EDT work so workers' live windows differ. */
  sleep_ns(spin);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  for (int b = 0; b < BURSTS; b++) {
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (int i = 0; i < BURST_EDTS; i++) {
      /* Per-EDT stall ramps so workers desynchronise their live windows. */
      uint64_t stall =
          (uint64_t)((i % 8) + 1) * 3ULL * 1000000ULL; /* 3..24 ms */
      arts_edt_create(burst_edt, 1, &stall, 0,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    }
    arts_event_wait(fe);

    /* Alternate sub-interval and super-interval main stalls so the capture
       thread alternates on-time and late branches (epoch jumps), widening the
       set of distinct epochs that appear across threads. */
    if (b % 2 == 0) {
      sleep_ns(70ULL * 1000000ULL); /* super-interval: forces late branch */
    } else {
      sleep_ns(5ULL * 1000000ULL); /* sub-interval */
    }
  }

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* If we reach here the merge did not overflow on this run.  No
     PASS_REGULAR_EXPRESSION is set: under ASan the under-allocation surfaces as
     a heap-buffer-overflow abort, which the harness reports as a failure
     documenting B135. */
  printf("counter_kway_merge: completed (no overflow observed this run)\n");
  return 0;
}
