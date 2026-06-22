/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file signals_double_sigterm.c
/// @brief Two SIGTERMs force the signal watcher's second-hit hard exit
///        (_exit(128+sig) == _exit(143) for SIGTERM).
///
/// arts_signal_watcher_main stays in its sigwait loop after the first term
/// signal (cooperative shutdown).  A SECOND term signal makes it print "forcing
/// exit" and call _exit(128+sig) immediately — bypassing normal teardown so a
/// hung graceful path is always escapable.  For SIGTERM (15) that is
/// _exit(143).  We raise two process-directed SIGTERMs so the watcher takes the
/// escalation arm; the process must terminate with status 143 (ctest matches
/// exit code 143 and that the normal clean-exit token is NOT printed).
///
/// Both raises happen from one EDT, back-to-back, so the second arrives while
/// the watcher is still in its loop after the first hit (the first only stops
/// workers; network threads + the watcher stay alive).
///
/// runtime_single, all configs (signals sit above the coherence layer).
/// Expected process exit status: 143.

#include "arts.h"

#include <signal.h>
#include <time.h>
#include <unistd.h>

void raise2_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  raising two SIGTERMs\n");
  /* First hit: watcher initiates cooperative shutdown, stays in loop.
   * Second hit: watcher forces _exit(143).  Both must be PROCESS-directed
   * (kill(getpid()), not raise()/pthread_kill(self)): the watcher consumes term
   * signals via sigwait() on its own thread, and a thread-directed signal
   * raised on this worker (which blocks the term set) would stay pending here
   * instead. */
  kill(getpid(), SIGTERM);
  kill(getpid(), SIGTERM);
  /* Bounded wait so this EDT does not return (and let the runtime reach a clean
   * shutdown) before the watcher's _exit(143) lands.  This is NOT an infinite
   * spin: it elapses well under the ctest TIMEOUT, and the expected outcome is
   * the watcher pre-empting the wait with _exit(143). */
  struct timespec budget = {.tv_sec = 5, .tv_nsec = 0};
  nanosleep(&budget, NULL);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("=== signals_double_sigterm ===\n");
  arts_edt_create(raise2_edt, 0, NULL, 0, &(arts_edt_hint_t){.rank = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* The watcher's _exit(143) should pre-empt this; if we ever reach here the
   * escalation did not fire (a regression). */
  arts_printf("DOUBLE_SIGTERM_UNEXPECTED_CLEAN_EXIT\n");
  return 0;
}
