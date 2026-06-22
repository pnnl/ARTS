/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
#include <stdio.h>
#include <unistd.h>

#include "arts/transport/stdio_forward.h"

/* The launcher forwards every non-master rank's stdout AND stderr to the
 * master's stdout/stderr.  A star launcher (rank 0 spawns all peers) needs
 * 2*(N-1) live forwarders for an N-node run.  Any fixed internal ceiling
 * would silently drop a rank's output once exceeded — output must never be
 * discarded — so this test opens far more forwarders than any historical
 * fixed cap and requires every one to be created.  (Plain assert() is a
 * no-op under -DNDEBUG in Release builds, so the check uses an explicit
 * nonzero exit instead.) */
enum { FORWARDERS = 300 };

int main(void) {
  int wfd[FORWARDERS];
  int made = 0;
  for (int i = 0; i < FORWARDERS; i++) {
    wfd[i] = arts_stdio_forwarder_make_pipe((unsigned int)i, "stdout", stdout);
    if (wfd[i] >= 0) {
      made++;
    }
  }
  /* Close every write-end so the reader threads observe EOF and exit; the
   * shutdown call then joins them. */
  for (int i = 0; i < FORWARDERS; i++) {
    if (wfd[i] >= 0) {
      close(wfd[i]);
    }
  }
  arts_stdio_forwarder_shutdown_all();

  if (made != FORWARDERS) {
    (void)fprintf(stderr,
                  "STDIO_FORWARD_SCALE FAIL: only %d/%d forwarders created — "
                  "a fixed cap silently drops output beyond it\n",
                  made, FORWARDERS);
    return 1;
  }
  printf("STDIO_FORWARD_SCALE made=%d/%d PASS\n", made, FORWARDERS);
  return 0;
}
