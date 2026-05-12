/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/// @file event_sticky_oversatisfy_error.c
/// @brief Verifies that a STICKY event raises ARTS_ERROR on over-satisfy.
///
/// This is a death test: the second arts_event_satisfy_slot drives the latch
/// past zero with error_on_neg_latch=true, which triggers ARTS_ERROR ->
/// process abort.  CTest registers the test with a bash wrapper +
/// PASS_REGULAR_EXPRESSION that matches the error banner (same pattern used
/// by term_sigfpe / term_sigsegv).

#include "arts.h"

#include <stdio.h>

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== event_sticky_oversatisfy_error ===\n");

  /* STICKY: latch=1, life_count=INT32_MAX, error_on_neg_latch=true. */
  arts_event_hint_t hint = ARTS_EVENT_HINT_STICKY;
  arts_guid_t ev = arts_event_create(&hint);

  /* First satisfy: latch 1 -> 0, fires.  No error. */
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  arts_printf("  before second satisfy (expected to ARTS_ERROR)\n");
  fflush(stdout);

  /* Second satisfy: latch would drop below zero -> ARTS_ERROR -> abort. */
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  /* If we reach here the test failed (no ARTS_ERROR was raised). */
  arts_printf("FAIL: STICKY did not raise ARTS_ERROR on over-satisfy\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
