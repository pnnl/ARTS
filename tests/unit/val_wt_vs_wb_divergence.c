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

/// @file val_wt_vs_wb_divergence.c
/// @brief Differential: same RW->RW->RO transfer scenario, asserted correct
///        under BOTH VAL placements (HOME and OWNER).
///
/// The two VAL placement TUs (home.c / owner.c) drive the SAME owner->owner
/// ownership transfer with divergent INTERNAL plumbing:
///   - HOME: arts_handler_db_grant_response drains pending_rw + runs the
///     newly-granted RW EDT IMMEDIATELY at the OWNERSHIP_RESPONSE (no stale-RO
///     window; home serves RO synchronously).
///   - OWNER: the same RESPONSE only installs the buffer + sentinel and sets
///     grant_unconfirmed=1; the RW drain is DEFERRED to the CONFIRM_ACK
///     round, and a fresh RW acquire during that window must PARK.
/// Both paths MUST produce identical externally-observable results.  This test
/// pins the contrast as ONE scenario built and run under each placement (each
/// build dir defines exactly one of ARTS_WRITE_POLICY_WT / ARTS_WRITE_POLICY_WB) and
/// asserts the end-to-end correctness that both plumbings must satisfy: a
/// transferred-then-read value is exact.  The pass token records which placement
/// ran so the integrator sees both arms execute.
///
/// A chain of RW writers on alternating ranks each stamp a per-step value and
/// then a final RO reader must observe the LAST writer's value — the transfer
/// must carry the latest buffer in either placement.  Mismatch => arts_abort.
///
/// VAL-only (home.c/owner.c are not compiled under WRF_VAL/EXCL).  Self
/// skips cleanly elsewhere.  On 1n there is no actual transfer but the chain is
/// still correct (passes trivially); the divergence is physically exercised at
/// 2n+.  A stranded waiter is caught by the ctest TIMEOUT (no in-test spin).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_VAL)

int main(void) {
  printf("SKIP val_wt_vs_wb_divergence: VAL-only\n");
  return 0;
}

#else

#define STEPS 64u

/// RW writer: stamp data[0] with this step's value.
static void wr_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/// RO reader: must observe the last writer's stamp.
static void rd_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(
        stderr, "FAIL val_wt_vs_wb_divergence: expected 0x%x got 0x%x\n",
        expect, d ? d[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

#if defined(ARTS_WRITE_POLICY_WT)
  const char *placement = "HOME";
#else
  const char *placement = "OWNER";
#endif
  arts_printf("=== val_wt_vs_wb_divergence (%s) ===\n", placement);

  unsigned int nranks = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  unsigned int last = 0u;
  for (unsigned int s = 0; s < STEPS; s++) {
    uint64_t v = (uint64_t)(0xA0000000u + s);
    last = (unsigned int)v;
    unsigned int rank = (nranks > 1) ? (s % nranks) : 0u;
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w = arts_edt_create(
        wr_edt, 1, &v, 1, &(arts_edt_hint_t){.rank = rank, .finish_event = e});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
    arts_event_wait(e);
  }

  uint64_t expect = (uint64_t)last;
  arts_guid_t er = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t r = arts_edt_create(
      rd_edt, 1, &expect, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = er});
  arts_add_dependence(db, r, 0, DB_MODE_RO);
  arts_event_wait(er);

  arts_printf("PASS val_wt_vs_wb_divergence (%s) %u steps\n", placement,
              STEPS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif
