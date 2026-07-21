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

/// @file mrsw_lazy_confirm_gate.c
/// @brief MRSW LAZY ownership_unconfirmed gate (config_specific, LAZY-only).
///
/// In MRSW LAZY a TRANSFER (OWNERSHIP_RESPONSE) install sets
/// ownership_unconfirmed=1 BEFORE the writer_count 0->2 bump and DEFERS the RW
/// drain to CONFIRM_ACK (home has not yet flipped rw_holder).  A fresh RW
/// acquire arriving in the [TRANSFER..CONFIRM_ACK] window must PARK on the gate
/// and only run once CONFIRM_ACK opens it.  The CONFIRM_ACK ship-decision: if
/// the deferred drain finds pending_rw empty it drops the orphan token, and a
/// piggybacked INVALIDATE that takes writer_count to 0 makes THIS handler ship
/// the transfer onward.
///
/// Construction: a strict write->read happens-before chain repeated many times.
/// Per iteration: reset on home (rank 0) re-owns + stamps vprev; writer on rank
/// W forces the 0->W transfer and writes vnew; reader on home, causally after
/// the writer, MUST observe vnew.  A fresh RW that runs while the gate is set
/// would write data observable before the directory names the rank (stale-RO),
/// surfacing as the reader seeing vprev/garbage.  main_edt serializes the three
/// phases with finish-event waits, exactly as the stale-RO window demands.
///
/// LAZY-only: EAGER has no ownership_unconfirmed gate (it WRITEBACKs
/// synchronously and the home always holds current data), so the test
/// self-skips at compile time under EAGER (and under any non-MRSW protocol).  A
/// stranded waiter (gate never opened) is caught by the ctest TIMEOUT.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW) || !defined(ARTS_TIMING_LAZY)

int main(void) {
  printf("SKIP mrsw_lazy_confirm_gate: MRSW+LAZY-only\n");
  return 0;
}

#else

#define ITERS 400u
#define RESET_BASE 0x30000000u
#define WRITE_BASE 0x40000000u

static void reset_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const unsigned int *d = (const unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL: stale RO read past confirm gate — expected 0x%x got "
                  "0x%x\n",
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

  arts_printf("=== mrsw_lazy_confirm_gate ===\n");

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W = (nranks > 1u) ? 1u : 0u;

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  for (unsigned int it = 0; it < ITERS; it++) {
    uint64_t vprev = (uint64_t)(RESET_BASE + it);
    uint64_t vnew = (uint64_t)(WRITE_BASE + it);

    arts_guid_t e_reset = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t rst =
        arts_edt_create(reset_edt, 1, &vprev, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = e_reset});
    arts_add_dependence(db, rst, 0, DB_MODE_RW);
    arts_event_wait(e_reset);

    arts_guid_t e_writer = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t wr = arts_edt_create(
        writer_edt, 1, &vnew, 1,
        &(arts_edt_hint_t){.rank = W, .finish_event = e_writer});
    arts_add_dependence(db, wr, 0, DB_MODE_RW);
    arts_event_wait(e_writer);

    arts_guid_t e_reader = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t rd = arts_edt_create(
        reader_edt, 1, &vnew, 1,
        &(arts_edt_hint_t){.rank = 0, .finish_event = e_reader});
    arts_add_dependence(db, rd, 0, DB_MODE_RO);
    arts_event_wait(e_reader);
  }

  arts_printf("PASS: mrsw_lazy_confirm_gate %u iterations\n", ITERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW && ARTS_TIMING_LAZY */
