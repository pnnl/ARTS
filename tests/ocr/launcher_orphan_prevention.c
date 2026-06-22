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

/// @file launcher_orphan_prevention.c
/// @brief Validates the orphan-prevention precondition of the local launcher:
///        PR_SET_PDEATHSIG(SIGKILL) + the getppid()==1 recheck.
///
/// The launcher closes the fork->prctl race window with two guards in the
/// child:
///   1. prctl(PR_SET_PDEATHSIG, SIGKILL) — kernel kills the child when the
///      master process dies, so no rank survives as a 599%-CPU orphan.
///   2. getppid() == 1 recheck — if the master died between fork() and prctl(),
///      the child has already been reparented to init (pid 1) and _exit(0)s
///      immediately rather than living on.
///
/// The actual kill-master-mid-startup scenario is driven externally by the test
/// harness (it reaps any survivor by exe path); a self-test cannot kill its own
/// master without poisoning the run.  What IS deterministically verifiable from
/// inside a healthy run is the *precondition* these guards rely on: every
/// non-master rank is still parented to a live, non-init master process while
/// the cluster runs.  If getppid() returned 1 here, the child should already
/// have _exit(0)'d in startup and never reached an EDT — so observing getppid()
/// != 1 on every rank confirms the parent-link the PDEATHSIG arms against.
///
/// Each rank reports via a finish event; main_edt waits on each so completion
/// is deterministic.  A stranded rank is caught by the ctest TIMEOUT.
/// Protocol-agnostic: launcher carries no coherence state, runs in all configs.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

/// Per-rank probe: confirm this rank is still parented to a non-init master.
void parent_probe_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int expected_rank = (unsigned int)paramv[0];

  unsigned int me = arts_get_current_rank();
  if (me != expected_rank) {
    arts_printf("FAIL: parent probe ran on %u, expected %u\n", me,
                expected_rank);
    arts_abort(1);
    return;
  }

  pid_t ppid = getppid();
  if (me == 0) {
    /* The master is not spawned by the launcher; its parent is whatever shell
     * or harness launched it.  Only assert it is a real running process. */
    if (ppid <= 0) {
      arts_printf("FAIL: master rank 0 has bad ppid %d\n", (int)ppid);
      arts_abort(1);
      return;
    }
    arts_printf("  PASS: rank 0 (master) ppid=%d\n", (int)ppid);
    return;
  }

  /* A spawned rank whose master had died before prctl() armed would have been
   * reparented to init (pid 1) and the getppid() recheck would have made it
   * _exit(0) in startup — it would never have reached this EDT.  Reaching here
   * with ppid != 1 means the PDEATHSIG-armed parent link is intact. */
  if (ppid == 1) {
    arts_printf("FAIL: rank %u reparented to init (orphan) — getppid()==1\n",
                me);
    arts_abort(1);
    return;
  }
  if (ppid <= 0) {
    arts_printf("FAIL: rank %u has bad ppid %d\n", me, (int)ppid);
    arts_abort(1);
    return;
  }
  arts_printf("  PASS: rank %u parented to live master ppid=%d\n", me,
              (int)ppid);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== launcher_orphan_prevention ===\n");

  unsigned int total = arts_get_total_ranks();
  for (unsigned int r = 0; r < total; r++) {
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t params[1] = {(uint64_t)r};
    arts_edt_create(parent_probe_edt, 1, params, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_event_wait(fe);
  }

  arts_printf("PASS: launcher_orphan_prevention (%u ranks parented)\n", total);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
