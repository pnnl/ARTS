/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/// @file counter_capture_epoch_monotonic.c
/// @brief Forces capture-thread lag and asserts epoch numbering stays strictly
///        monotonic across the on-time vs late (drift) branches of
///        arts_counter_capture_thread (counter.c).
///
/// The capture thread numbers each sample with an epoch.  Two branches assign
/// the epoch differently:
///   - on-time  (sleep_ns > 0):  current_epoch = capture_epoch + 1; then
///     capture_epoch++.
///   - late     (sleep_ns <= 0):  capture_epoch += intervals_behind;
///     current_epoch = capture_epoch.
/// The invariant under test: across a run that mixes both branches (induced by
/// stalling the workload longer than the capture interval), the emitted epochs
/// in captureHistory are strictly increasing and never repeat.
///
/// Strategy: run a workload that spans many capture intervals and deliberately
/// blocks the main EDT for several intervals at a stretch (via nanosleep) so
/// the capture thread takes the late branch at least once.  After shutdown,
/// read the per-node JSON, extract a PERIODIC counter's captureHistory, and
/// verify the epoch column is strictly increasing.
///
/// Requires PERIODIC counters to be enabled (so a capture thread exists and
/// captureHistory is written).  Both the default counters.cfg and
/// full_counters.cfg enable PERIODIC CLUSTER counters, so n0.json carries a
/// captureHistory.  If no captureHistory is found (PERIODIC disabled in this
/// build's counter config) the test SKIPs cleanly.
/// Config-agnostic across coherence protocols.

#include "arts.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Number of stall rounds; each stall is longer than a typical capture
   interval so the capture thread is forced through its late branch. */
#define STALL_ROUNDS 6
#define STALL_NS (60ULL * 1000000ULL) /* 60 ms per stall round */

static void sleep_ns(uint64_t ns) {
  struct timespec ts = {(time_t)(ns / 1000000000ULL),
                        (long)(ns % 1000000000ULL)};
  (void)nanosleep(&ts, NULL);
}

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

  /* Generate enough counter activity (EDT creates) spread over time that the
     capture thread takes several samples, and stall between bursts so it is
     forced late at least once. */
  for (int round = 0; round < STALL_ROUNDS; round++) {
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (int i = 0; i < 32; i++) {
      arts_edt_create(busy_edt, 0, NULL, 0,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    }
    arts_event_wait(fe);
    /* Stall the worker (and thus counter activity) longer than one interval. */
    sleep_ns(STALL_NS);
  }

  arts_shutdown();
}

/* Parse a captureHistory string of the form [[e,v],[e,v],...] and verify the
   epoch column is strictly increasing.  Returns:
     1  = found a history and epochs are strictly increasing (PASS)
     0  = found a history but epochs were NOT strictly increasing (FAIL)
    -1  = no captureHistory present (SKIP) */
static int check_epoch_monotonic(const char *json) {
  const char *key = "\"captureHistory\"";
  const char *p = strstr(json, key);
  if (!p) {
    return -1;
  }
  /* Advance to the value's opening '['. */
  p = strchr(p + strlen(key), '[');
  if (!p) {
    return -1;
  }
  p++; /* skip outer '[' */

  bool have_prev = false;
  uint64_t prev_epoch = 0;
  uint64_t entries = 0;
  while (*p && *p != ']') {
    while (*p && (*p == ' ' || *p == ',')) {
      p++;
    }
    if (*p != '[') {
      break;
    }
    p++; /* skip inner '[' */
    char *end = NULL;
    uint64_t epoch = strtoull(p, &end, 10);
    if (end == p) {
      break;
    }
    p = end;
    /* skip to inner ']' */
    while (*p && *p != ']') {
      p++;
    }
    if (*p == ']') {
      p++;
    }
    entries++;
    if (have_prev && epoch <= prev_epoch) {
      arts_printf("FAIL counter_capture_epoch_monotonic: epoch not strictly "
                  "increasing (%llu after %llu)\n",
                  (unsigned long long)epoch, (unsigned long long)prev_epoch);
      return 0;
    }
    prev_epoch = epoch;
    have_prev = true;
  }

  if (entries == 0) {
    return -1;
  }
  arts_printf(
      "counter_capture_epoch_monotonic: %llu epochs strictly increasing\n",
      (unsigned long long)entries);
  return 1;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);

  FILE *fp = fopen("./counters/n0.json", "r");
  if (!fp) {
    /* No node file: NODE/CLUSTER counters disabled in this build. */
    printf("SKIP counter_capture_epoch_monotonic: no ./counters/n0.json\n");
    return 0;
  }
  (void)fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  (void)fseek(fp, 0, SEEK_SET);
  if (sz <= 0) {
    (void)fclose(fp);
    printf("SKIP counter_capture_epoch_monotonic: empty n0.json\n");
    return 0;
  }
  char *buf = (char *)malloc((size_t)sz + 1);
  size_t got = fread(buf, 1, (size_t)sz, fp);
  (void)fclose(fp);
  buf[got] = '\0';

  int r = check_epoch_monotonic(buf);
  free(buf);

  if (r == 1) {
    printf("PASS counter_capture_epoch_monotonic\n");
    return 0;
  }
  if (r == -1) {
    printf("SKIP counter_capture_epoch_monotonic: no captureHistory (PERIODIC "
           "disabled)\n");
    return 0;
  }
  printf("FAIL counter_capture_epoch_monotonic\n");
  return 1;
}
