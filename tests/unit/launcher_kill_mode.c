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

/// @file launcher_kill_mode.c
/// @brief SSH-launcher kill_mode contract: master builds `pkill <basename>`
///        per node and then exit(0)s the whole master process.
///
/// kill_mode (config kill_mode != 0) is the "kill stale instances then quit"
/// utility path of the SSH launcher.  In kill_mode, `arts_launcher_ssh_startup_
/// processes` SSHes a `pkill <basename>` to each non-master node, dup's the
/// children's stdio to /dev/null, and after the loop calls exit(0) — the master
/// process terminates INSIDE launch_processes(), before thread init.  As a
/// direct consequence the master never schedules main_edt: a kill_mode run does
/// no runtime work at all, it just fans out pkills and exits 0.
///
/// This is fundamentally not exercisable on the local CI box:
///   1. There is no SSH (the test box has no SSH launcher / remote nodes), and
///   2. even under SSH, the kill_mode master exit(0)s before main_edt, so a
///      body that reaches main_edt is by definition NOT a kill_mode launch.
///
/// Hence this is a config_specific test gated on the SSH-kill-mode build define
/// the SSH harness would set (ARTS_TEST_SSH_KILL_MODE).  Without it (every CI
/// build) the test self-skips cleanly with exit 0.  With it, the real body
/// documents the contract: reaching main_edt under kill_mode is a contradiction
/// (the master should have exit(0)'d), so its presence is itself the failure
/// signal — under a correct kill_mode launch the body never runs and the master
/// exits 0 on its own.

#include "arts.h"

#include <stdio.h>

#if !defined(ARTS_TEST_SSH_KILL_MODE)

/* Default (all CI builds): SSH kill_mode is not runnable here — skip cleanly.
 */
int main(void) {
  printf("SKIP launcher_kill_mode: SSH kill_mode (no SSH / no remote nodes on "
         "this host; kill_mode master exit(0)s before main_edt)\n");
  return 0;
}

#else

/// In a correct kill_mode launch the master exit(0)s inside launch_processes
/// and main_edt is NEVER scheduled.  If we ever reach this body, the kill_mode
/// early-exit contract was violated.
void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  fprintf(stderr, "FAIL: main_edt scheduled under kill_mode — master should "
                  "have exit(0)'d in launch_processes before thread init\n");
  arts_abort(1);
}

int main(int argc, char **argv) {
  /* arts_rt drives the master through launch_processes(); in kill_mode that
   * calls exit(0) and this return is never reached.  The harness asserts the
   * process exits 0 and that no main_edt FAIL line was printed.  If arts_rt
   * returns normally, kill_mode did not exit(0) as required. */
  arts_rt(argc, argv);
  fprintf(stderr, "FAIL: arts_rt returned under kill_mode — expected the "
                  "master to exit(0) inside launch_processes\n");
  return 1;
}

#endif
