/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file signals_sigterm_graceful.c
/// @brief A single SIGTERM/SIGINT to the process triggers the signal watcher's
///        first-hit cooperative shutdown and a clean return from arts_rt.
///
/// arts_signal_watcher_main is a dedicated thread that sigwaits the term set
/// (SIGTERM/SIGINT/SIGALRM/SIGHUP) + SIGRTMIN; every other thread inherits a
/// block mask so only the watcher receives term signals.  On the first term hit
/// the watcher prints "initiating graceful shutdown", calls
/// arts_enter_shutdown_state(true), and STAYS in the loop (a second hit would
/// force exit).  A single SIGTERM must therefore drive a cooperative shutdown:
/// arts_rt returns normally and main() prints the clean-exit token.
///
/// An EDT raises one process-directed term signal (default SIGTERM; argv[1] may
/// select SIGINT) via raise(); the watcher consumes it and shuts down.  We do
/// NOT raise a second signal, so the _exit(128+sig) escalation must not fire.
/// ctest matches the clean token; a regression that breaks the watcher would
/// hang (ctest TIMEOUT) or exit with 128+sig.
///
/// runtime_single, all configs (signals sit above the coherence layer).

#include "arts.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void raise_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int sig = (int)paramv[0];
  arts_printf("  raising single %s\n", sig == SIGINT ? "SIGINT" : "SIGTERM");
  /* Must be PROCESS-directed (kill(getpid()), not raise()/pthread_kill(self)):
   * the watcher consumes term signals via sigwait() on its own thread, and a
   * thread-directed signal raised on this worker (which blocks the term set)
   * would stay pending here forever instead of reaching the watcher. */
  kill(getpid(), sig);
  /* Watcher's first hit -> graceful shutdown; this EDT returns normally. */
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_printf("=== signals_sigterm_graceful ===\n");

  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  int sig = SIGTERM;
  if (argc > 1 && strcmp(argv[1], "int") == 0) {
    sig = SIGINT;
  }

  uint64_t pv[1] = {(uint64_t)sig};
  arts_edt_create(raise_edt, 1, pv, 0, &(arts_edt_hint_t){.rank = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  /* Clean return => single term signal drove a cooperative shutdown. */
  arts_printf("SIGTERM_GRACEFUL_OK\n");
  return 0;
}
