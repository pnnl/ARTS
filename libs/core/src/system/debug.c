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

#include "arts/system/arts_print.h"

static void arts_crash_signal_handler(int sig) {
  const char *msg = "\n[ARTS] Fatal signal — stack trace:\n";
  (void)write(STDERR_FILENO, msg, strlen(msg));

  void *frames[32];
  int n = backtrace(frames, 32);
  backtrace_symbols_fd(frames, n, STDERR_FILENO);

  struct sigaction sa;
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(sig, &sa, NULL);
  (void)raise(sig);
}

static void arts_install_crash_handlers(void) {
  struct sigaction sa;
  sa.sa_handler = arts_crash_signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  int sigs[] = {SIGSEGV, SIGBUS, SIGFPE};
  for (int i = 0; i < 3; i++) {
    sigaction(sigs[i], &sa, NULL);
  }
}

#if !defined(__APPLE__)

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

  arts_install_crash_handlers();
}

#else

void arts_turn_on_core_dumps(void) {
  ARTS_INFO("Core dumps not supported on OS X.");
  arts_install_crash_handlers();
}

#endif
