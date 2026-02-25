/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#include "arts/system/debug.h"

#include <execinfo.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#include "arts/system/print.h"
#include "arts/system/threads.h"

// Async-signal-safe signal name lookup (strsignal() is NOT safe).
static const char *signal_name(int sig) {
  switch (sig) {
  case SIGSEGV:
    return "SIGSEGV";
  case SIGBUS:
    return "SIGBUS";
  case SIGFPE:
    return "SIGFPE";
  case SIGTERM:
    return "SIGTERM";
  case SIGINT:
    return "SIGINT";
  case SIGALRM:
    return "SIGALRM";
  case SIGHUP:
    return "SIGHUP";
  default:
    return "UNKNOWN";
  }
}

// Async-signal-safe: write an unsigned int as decimal digits to stderr.
static void write_uint(unsigned int val) {
  char buf[16];
  int pos = (int)sizeof(buf);
  if (val == 0) {
    buf[--pos] = '0';
  } else {
    while (val > 0) {
      buf[--pos] = (char)('0' + (val % 10));
      val /= 10;
    }
  }
  (void)write(STDERR_FILENO, buf + pos, (size_t)(sizeof(buf) - (size_t)pos));
}

static void write_backtrace(void) {
  void *frames[32];
  int depth = backtrace(frames, 32);
  backtrace_symbols_fd(frames, depth, STDERR_FILENO);
}

// Crash signals (SIGSEGV, SIGBUS, SIGFPE) — unrecoverable, re-raise for core.
static void arts_crash_signal_handler(int sig) {
  const char *pre = "\n[ARTS] Crashed: ";
  (void)write(STDERR_FILENO, pre, strlen(pre));
  const char *name = signal_name(sig);
  (void)write(STDERR_FILENO, name, strlen(name));
  const char *mid = " (rank ";
  (void)write(STDERR_FILENO, mid, strlen(mid));
  write_uint(arts_global_rank_id);
  const char *post = ") — stack trace:\n";
  (void)write(STDERR_FILENO, post, strlen(post));

  write_backtrace();

  struct sigaction sa;
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(sig, &sa, NULL);
  (void)raise(sig);
}

// Termination signals (SIGTERM, SIGINT, SIGALRM, SIGHUP) — graceful exit.
static void arts_term_signal_handler(int sig) {
  const char *pre = "\n[ARTS] Killed by ";
  (void)write(STDERR_FILENO, pre, strlen(pre));
  const char *name = signal_name(sig);
  (void)write(STDERR_FILENO, name, strlen(name));
  const char *mid = " (rank ";
  (void)write(STDERR_FILENO, mid, strlen(mid));
  write_uint(arts_global_rank_id);
  const char *post = ") — stack trace:\n";
  (void)write(STDERR_FILENO, post, strlen(post));

  write_backtrace();

  struct sigaction sa;
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(sig, &sa, NULL);
  (void)raise(sig);
}

void arts_install_signal_handlers(void) {
  // Crash handlers — backtrace + re-raise for core dump.
  struct sigaction crash_sa;
  crash_sa.sa_handler = arts_crash_signal_handler;
  sigemptyset(&crash_sa.sa_mask);
  crash_sa.sa_flags = 0;
  int crash_sigs[] = {SIGSEGV, SIGBUS, SIGFPE};
  for (int i = 0; i < 3; i++) {
    sigaction(crash_sigs[i], &crash_sa, NULL);
  }

  // Termination handlers — backtrace + re-raise for clean exit.
  struct sigaction term_sa;
  term_sa.sa_handler = arts_term_signal_handler;
  sigemptyset(&term_sa.sa_mask);
  term_sa.sa_flags = 0;
  int term_sigs[] = {SIGTERM, SIGINT, SIGALRM, SIGHUP};
  for (int i = 0; i < 4; i++) {
    sigaction(term_sigs[i], &term_sa, NULL);
  }

  // Ignore SIGPIPE — let socket writes fail with EPIPE instead of killing us.
  signal(SIGPIPE, SIG_IGN); // NOLINT(cert-err33-c)
}

#ifndef __APPLE__

#include <sys/prctl.h>
#include <sys/resource.h>

void arts_turn_on_core_dumps(void) {
  (void)prctl(PR_SET_DUMPABLE, 1);

  struct rlimit limit;
  limit.rlim_cur = RLIM_INFINITY;
  limit.rlim_max = RLIM_INFINITY;
  if (setrlimit(RLIMIT_CORE, &limit) != 0) {
    ARTS_INFO("Failed to force core dumps");
  }
}

#else

void arts_turn_on_core_dumps(void) {
  ARTS_INFO("Core dumps not supported on OS X.");
}

#endif
