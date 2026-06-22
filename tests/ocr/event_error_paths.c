/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_error_paths — C15 / T158.
 *
 * Target: handler_event_satisfy_slot error rejection (event.c).
 *
 * Two invalid-slot rejections share one mechanism (ARTS_ERROR, which always
 * aborts):
 *   - CHANNEL: only ARTS_EVENT_LATCH_DECR_SLOT (0) is a legal satisfy slot;
 *     any other slot is rejected ("CHANNEL: only DECR (slot 0) satisfy
 *     supported").
 *   - simple: only DECR (0) / INCR (1) are legal; any other slot is rejected
 *     ("Event latch invalid slot").
 *
 * Because ARTS_ERROR aborts the process, a single run can exercise exactly one
 * rejection.  This test drives the CHANNEL non-DECR rejection: create a CHANNEL
 * event and satisfy slot INCR (1).  The runtime MUST reject it — the CORRECT
 * behavior is the abort with the documented error message.  ctest matches that
 * message via PASS_REGULAR_EXPRESSION, so the test passes precisely when the
 * runtime correctly rejects the invalid slot.
 *
 * If the runtime ever silently ACCEPTED the bad slot, no error line would be
 * printed and the marker-only output below would not match the pass regex —
 * the test would fail, flagging the missing guard.
 */

#include <stdio.h>

#include "arts.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* CHANNEL event, home-local. */
  arts_event_hint_t ch = ARTS_EVENT_HINT_CHANNEL;
  ch.rank = arts_get_current_rank();
  arts_guid_t ev = arts_event_create(&ch);
  if (ev == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: CHANNEL create returned NULL_GUID\n");
    arts_abort(1);
  }

  (void)fprintf(stderr,
                "event_error_paths: satisfying CHANNEL on INCR slot (must be "
                "rejected)\n");
  fflush(stderr);

  /* Illegal: CHANNEL accepts only DECR (slot 0).  Expect ARTS_ERROR abort. */
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_INCR_SLOT);

  /* Unreachable if the guard is correct.  If we get here the runtime wrongly
   * accepted the bad slot — fail loudly (no pass-regex match). */
  (void)fprintf(stderr,
                "FAIL: CHANNEL accepted an illegal non-DECR satisfy slot\n");
  arts_abort(1);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
