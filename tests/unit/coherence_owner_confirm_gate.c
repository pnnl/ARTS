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

/// @file coherence_owner_confirm_gate.c
/// @brief Exploit + regression for the WB ownership-confirm gate.
///
/// Reproduces the stale-RO window: when ownership transfers home(0)->W, the
/// home updates rw_holder only at INSTALL_ACK, after W has already run its RW
/// EDT.  An RO acquire that is causally-ordered AFTER W's write but races W's
/// INSTALL_ACK is redirected by home to the OLD owner and reads a stale value.
///
/// Per iteration (a clean write -> read happens-before chain):
///   reset (rank 0, RW)  : re-own on home, write vprev      [finish_event
///   e_reset] writer(rank W, RW)  : gated on e_reset; transfer 0->W;
///                         write vnew                        [finish_event
///                         e_writer]
///   reader(rank 0, RO)  : gated on e_writer; MUST read vnew [finish_event
///   e_reader]
/// main_edt waits on each finish event in order (e_reset, e_writer, e_reader),
/// releasing the creator-token so each scope can drain and the next EDT can
/// run.
///
/// Config-agnostic: runs under configs/local/{1n,2n,3n,4n,2n_io}.cfg.  The
/// 2n_io config (multiple sender/receiver threads -> wire reorder) is the one
/// that exposes the race.  On 1n there is no transfer and the test passes
/// trivially.  WRF_VAL has no ownership transfer -> SKIP.
/// A stranded waiter is caught by the ctest TIMEOUT (no in-test watchdog).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ITERS 500u
#define RESET_BASE 0x10000000u
#define WRITE_BASE 0x20000000u

/// reset: re-establish ownership on home (rank 0) and stamp a per-iter
/// sentinel.
void reset_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/// writer: RW acquire forces the home->W ownership transfer; write the new
/// value.
void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/// reader: RO acquire, causally after the writer; MUST observe the writer's
/// value.  arts_abort(1) on any mismatch so the exit code is non-zero.
void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr, "FAIL: stale RO read — expected 0x%x got 0x%x\n",
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

  arts_printf("=== coherence_owner_confirm_gate ===\n");

#ifdef ARTS_PROTOCOL_WRF_VAL
  arts_printf("SKIP: WRF_VAL has no ownership transfer\n");
  arts_shutdown();
  return;
#endif

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W =
      (nranks > 1) ? 1u : 0u; /* writer rank (remote when possible) */

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  for (unsigned int it = 0; it < ITERS; it++) {
    uint64_t vprev = (uint64_t)(RESET_BASE + it);
    uint64_t vnew = (uint64_t)(WRITE_BASE + it);

    /* --- Phase 1: reset EDT re-establishes ownership on home (rank 0). --- */
    arts_guid_t e_reset = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t rst =
        arts_edt_create(reset_edt, 1, &vprev, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = e_reset});
    arts_add_dependence(db, rst, 0, DB_MODE_RW);
    /* Release the e_reset creator-token and wait for the reset EDT to finish.
     * When this returns, the reset EDT has completed and e_reset has fired,
     * meaning ownership is back on rank 0. */
    arts_event_wait(e_reset);

    /* --- Phase 2: writer EDT on rank W; forces 0->W ownership transfer. --- */
    arts_guid_t e_writer = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t wr = arts_edt_create(
        writer_edt, 1, &vnew, 1,
        &(arts_edt_hint_t){.rank = W, .finish_event = e_writer});
    arts_add_dependence(db, wr, 0, DB_MODE_RW);
    /* Release the e_writer creator-token and wait for the writer EDT to finish.
     * While this call is spinning (DB released for member EDT progress), the
     * writer EDT acquires RW ownership (home redirects 0->W) and writes vnew.
     * The race window: home updates rw_holder only on INSTALL_ACK, which may
     * arrive AFTER the writer EDT has completed and e_writer has fired here. */
    arts_event_wait(e_writer);

    /* --- Phase 3: reader EDT on home (rank 0); MUST see vnew. ---
     * Causally ordered AFTER the writer (we only create this EDT after
     * arts_event_wait(e_writer) returns), but the RO acquire's LOCK_REQ
     * can race the in-flight INSTALL_ACK at the home — the bug window. */
    arts_guid_t e_reader = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t rd = arts_edt_create(
        reader_edt, 1, &vnew, 1,
        &(arts_edt_hint_t){.rank = 0, .finish_event = e_reader});
    arts_add_dependence(db, rd, 0, DB_MODE_RO);
    arts_event_wait(e_reader);
  }

  arts_printf("PASS: coherence_owner_confirm_gate %u iterations\n", ITERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
