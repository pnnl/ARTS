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

/// @file launcher_cleanup_escalation.c
/// @brief Exercises the graceful first stage of the local launcher's cleanup
///        escalation ladder (waitpid 5s -> SIGTERM 2s -> SIGKILL).
///
/// `arts_launcher_local_cleanup_processes` reaps each spawned rank with a
/// three-stage ladder: a 5 s graceful waitpid(WNOHANG) poll, then SIGTERM with
/// a 2 s poll, then a final SIGKILL backstop.  Under a normal shutdown every
/// rank exits on its own (it observed the runtime stop), so all ranks are
/// reaped in the FIRST graceful stage — the launcher never needs to send
/// SIGTERM/SIGKILL.
///
/// Deliberately injecting a hung rank to force the SIGTERM/SIGKILL rungs would
/// leave a survivor that has to be killed externally (poisoning ports for the
/// next run); the abnormal rungs are validated by the test harness's external
/// reap-and-escalate logic, not by a self-test.  What this test pins down is
/// the healthy contract: a real multinode run that does cross-rank work on
/// every rank and then calls arts_shutdown() reaches the launcher's cleanup
/// with all ranks already exited, so the graceful waitpid stage drains them and
/// the process group tears down within the ctest TIMEOUT.  If the graceful reap
/// path were broken (e.g. a rank that never exits, or a launcher that never
/// reaped), the run would hang and the ctest TIMEOUT would catch it.
///
/// A finish event per rank gates main_edt so completion is deterministic; there
/// is no in-test spin/watchdog.
/// Protocol-agnostic: launcher carries no coherence state, runs in all configs.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

/// Trivial per-rank work so every rank is genuinely scheduled (and thus must be
/// reaped at cleanup), then returns normally so it can exit gracefully.
void work_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int expected_rank = (unsigned int)paramv[0];
  unsigned int me = arts_get_current_rank();
  if (me != expected_rank) {
    arts_printf("FAIL: work ran on rank %u, expected %u\n", me, expected_rank);
    arts_abort(1);
    return;
  }
  arts_printf("  rank %u did work; will exit gracefully\n", me);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== launcher_cleanup_escalation ===\n");

  unsigned int total = arts_get_total_ranks();
  for (unsigned int r = 0; r < total; r++) {
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t params[1] = {(uint64_t)r};
    arts_edt_create(work_edt, 1, params, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_event_wait(fe);
  }

  /* All ranks have finished their work and are idling on the runtime loop.
   * arts_shutdown() drives every rank to exit on its own, so the master's
   * launcher cleanup reaps them all in the graceful waitpid stage — no SIGTERM
   * or SIGKILL needed.  A broken graceful reap would hang here -> TIMEOUT. */
  arts_printf("PASS: launcher_cleanup_escalation (%u ranks, graceful reap)\n",
              total);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
