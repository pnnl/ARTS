/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/// @file counter_time_sync.c
/// @brief Validates the RTT counter time-sync handshake
///        (arts_send_time_sync_request / arts_handler_time_sync_request /
///        arts_handler_time_sync_response in counter.c).
///
/// When PERIODIC counters are enabled, arts_counter_capture_start performs an
/// RTT handshake before spawning the capture thread:
///   - master rank: sets offset 0, received = true immediately.
///   - worker rank: sends MSG_TIME_SYNC_REQUEST, then spin-waits up to 500ms
///     for MSG_TIME_SYNC_RESPONSE; on receipt computes
///     offset = (T1 + T3)/2 - T2 and sets received = true; on timeout falls
///     back to offset 0 with received = true (degraded, not a failure).
///
/// After arts_rt returns, every rank that needed a capture thread must have
/// received == true.  On a worker, the computed offset must lie within the
/// handshake's bound: |offset| <= 500ms (the spin-wait window; one-way delay is
/// RTT/2 and the fallback is exactly 0).  This pins both the success path and
/// the timeout-fallback contract.
///
/// The sync globals (arts_counter_time_offset /
/// arts_counter_time_sync_received) have no public/internal header declaration
/// (file-defined in counter.c); they are non-static, so we re-declare the
/// externs here for white-box inspection.
///
/// Multinode: needs node_count > 1 to exercise the worker request/response
/// path; SKIPs on a single node.  If no PERIODIC counter is enabled in this
/// build's counter config the handshake never runs (received stays false) and
/// the test SKIPs.  Config-agnostic across coherence protocols.  Reaped by the
/// ctest TIMEOUT if stranded.

#include "arts.h"
#include "arts/system/identity.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Defined (non-static, volatile) in libs/src/core/counter/counter.c. */
extern volatile int64_t arts_counter_time_offset;
extern volatile bool arts_counter_time_sync_received;

void busy_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  unsigned int ranks = arts_get_total_ranks();
  for (unsigned int r = 0; r < ranks; r++) {
    arts_edt_create(busy_edt, 0, NULL, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = fe});
  }
  arts_event_wait(fe);

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);

  if (arts_global_rank_count < 2) {
    printf("SKIP counter_time_sync: single node (no worker handshake)\n");
    return 0;
  }

  bool received = arts_counter_time_sync_received;
  int64_t offset = arts_counter_time_offset;

  if (!received) {
    /* No PERIODIC counter -> no capture thread -> no handshake on this build.
     */
    printf("SKIP counter_time_sync: handshake not exercised (PERIODIC counters "
           "disabled)\n");
    return 0;
  }

  if (arts_global_rank_id == arts_global_master_rank_id) {
    /* Master must report offset 0. */
    if (offset != 0) {
      printf("FAIL counter_time_sync: master rank %u offset %lld != 0\n",
             arts_global_master_rank_id, (long long)offset);
      return 1;
    }
    printf("PASS counter_time_sync: master rank %u offset 0\n",
           arts_global_master_rank_id);
    return 0;
  }

  /* Worker: offset must be within the 500ms handshake window (success path
     bounds one-way delay by RTT/2; timeout fallback is exactly 0). */
  const int64_t bound_ns = 500LL * 1000000LL;
  if (offset > bound_ns || offset < -bound_ns) {
    printf("FAIL counter_time_sync: worker rank %u offset %lld ns outside "
           "+/-500ms window\n",
           arts_global_rank_id, (long long)offset);
    return 1;
  }

  printf(
      "PASS counter_time_sync: worker rank %u offset %lld ns within window\n",
      arts_global_rank_id, (long long)offset);
  return 0;
}
