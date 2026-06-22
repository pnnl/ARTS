/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/// @file counter_timer_balance.c
/// @brief Pins the start/end balance contract of arts_counter_timer_start /
///        arts_counter_timer_end (counter.c).
///
/// Contract (counter.c):
///   - timer_start: CAS counter->start 0 -> now; ERROR (abort) if already set
///     (double-start).
///   - timer_end:   swap counter->start -> 0; ERROR (abort) if it was already 0
///     (end-without-start); then fold (end-start) into counter->count.
///
/// The two unbalanced operations call ARTS_ERROR which always aborts
/// (arts_abort(1)).  We exercise both the legal balanced path (observable:
/// count grows, start resets to 0) and the two illegal paths in forked
/// children, each of which MUST die by abort (nonzero / signalled exit).  The
/// timer functions only touch atomics + arts_get_time_stamp + (on error)
/// arts_abort, so they are safe to call directly without arts_rt — keeping the
/// contract check runtime-free and deterministic.  arts_rt is still driven (a
/// trivial EDT + finish event) so this registers as a runtime single-node test.
///
/// Config-agnostic: the timer functions behave identically under every
/// coherence and counter configuration (atomic CAS/swap on a single uint64
/// field).

#include "arts.h"
#include "arts/counter/counter.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

/* Spawn a child running fn; return true iff the child died abnormally
   (nonzero exit code or killed by a signal) — i.e. the abort fired. */
static bool child_aborts(void (*fn)(void)) {
  pid_t pid = fork();
  if (pid == 0) {
    fn();
    /* If we reach here the abort did NOT fire — report clean exit. */
    _exit(0);
  }
  int status = 0;
  (void)waitpid(pid, &status, 0);
  bool exited_nonzero = WIFEXITED(status) && WEXITSTATUS(status) != 0;
  bool signalled = WIFSIGNALED(status);
  return exited_nonzero || signalled;
}

static void do_double_start(void) {
  arts_counter_t c = {0, 0};
  arts_counter_timer_start(&c); /* legal */
  arts_counter_timer_start(&c); /* double-start -> ERROR/abort */
}

static void do_end_without_start(void) {
  arts_counter_t c = {0, 0};
  arts_counter_timer_end(&c); /* end-without-start -> ERROR/abort */
}

static bool check_timer_contract(void) {
  /* Legal balanced path: start then end -> start back to 0, count accumulated.
   */
  arts_counter_t c = {0, 0};
  arts_counter_timer_start(&c);
  if (c.start == 0) {
    arts_printf(
        "FAIL counter_timer_balance: start not armed after timer_start\n");
    return false;
  }
  /* burn a little wall time so end-start is observable */
  for (volatile uint64_t spin = 0; spin < 100000; spin++) {
  }
  arts_counter_timer_end(&c);
  if (c.start != 0) {
    arts_printf(
        "FAIL counter_timer_balance: start not cleared after timer_end\n");
    return false;
  }
  /* A second balanced cycle must accumulate, not reset. */
  uint64_t after_first = c.count;
  arts_counter_timer_start(&c);
  arts_counter_timer_end(&c);
  if (c.count < after_first) {
    arts_printf("FAIL counter_timer_balance: count went backwards across "
                "balanced cycles (%llu -> %llu)\n",
                (unsigned long long)after_first, (unsigned long long)c.count);
    return false;
  }

  /* Illegal paths must abort. */
  if (!child_aborts(do_double_start)) {
    arts_printf("FAIL counter_timer_balance: double-start did NOT abort\n");
    return false;
  }
  if (!child_aborts(do_end_without_start)) {
    arts_printf(
        "FAIL counter_timer_balance: end-without-start did NOT abort\n");
    return false;
  }
  return true;
}

void leaf_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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
  arts_edt_create(leaf_edt, 0, NULL, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Validate the timer balance contract before bringing the runtime up so the
     forked children do not inherit a live runtime. */
  bool ok = check_timer_contract();

  arts_rt(argc, argv);

  if (ok) {
    printf("PASS counter_timer_balance\n");
    return 0;
  }
  printf("FAIL counter_timer_balance\n");
  return 1;
}
