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

/// @file val_writer_count_nonneg.c
/// @brief Hammer the VAL writer_count non-negativity / +2 drain-guard window
///        under adversarial INVALIDATE(-1) vs OWNERSHIP_RESPONSE(+2) reorder.
///
/// writer_count is the central VAL ownership counter: +1 (acquire fast /
/// GRANT drain), +2 (OWNERSHIP_RESPONSE install guard: 0->2 in one RMW so a
/// racing INVALIDATE never catches an intermediate 1), -1 (INVALIDATE /
/// release).  The signed-cast `(int)writer_count > 0` owner test rests on the
/// count NEVER wrapping a huge unsigned and NEVER reporting false-owner.  The
/// dangerous window: a next-round INVALIDATE(-1) lands mid-install on a SECOND
/// receiver thread before the +2 completes; the +2 (rather than swap(1)) keeps
/// the edge commutative so no transfer is lost and no count underflows.
///
/// This test drives the window the same way coherence_owner_grant_reorder
/// does, but as an VAL-both-placements gate (HOME exercises the HOME
/// INVALIDATE path; OWNER exercises the TRANSFER+INVALIDATE two-wire reorder).
/// Many short RW-incrementer EDTs are spawned round-robin across ALL ranks, all
/// gated only on the SAME DB inside one finish scope per batch.  Mutually
/// unordered, they build a deep home pending_rw queue so home fires a tight
/// GRANT-then-INVALIDATE stream — a fresh +2/-1 reorder opportunity per
/// transfer.  progress_threads>=2 (the 2n_io config) makes the physical reorder
/// possible.
///
/// ASSERTS: (1) PROGRESS — every batch quiesces (a lost transfer from an
/// underflowed/wrapped count strands the next RW acquirer; caught by ctest
/// TIMEOUT).  (2) NO DATA LOSS — every increment survives so the canonical sum
/// is EXACTLY `total` (a false-owner serving a stale buffer would drop counts).
/// Same-node concurrent RW EDTs share the buffer under HW coherence, so the
/// increment is atomic (DB_MODE_RW is per-NODE exclusive only).
///
/// VAL-only; self-skips elsewhere.  At 1n there is no transfer (degenerate
/// pass).  Targets the writer_count-wrap / false-owner bug class.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_VAL)

int main(void) {
  printf("SKIP val_writer_count_nonneg: VAL-only\n");
  return 0;
}

#else

#define BATCHES 40u
#define PER_BATCH 24u

/// RW incrementer: atomic +1 on the shared owned buffer.
static void inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic unsigned int *d = (_Atomic unsigned int *)depv[0].ptr;
  if (d != NULL) {
    atomic_fetch_add_explicit(d, 1u, memory_order_relaxed);
  }
}

/// RO reader: assert the canonical sum equals the total increments shipped.
static void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL val_writer_count_nonneg: sum expected %u got %u\n",
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

  arts_printf("=== val_writer_count_nonneg ===\n");

  unsigned int nranks = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  unsigned int total = 0u;
  for (unsigned int b = 0; b < BATCHES; b++) {
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int i = 0; i < PER_BATCH; i++) {
      unsigned int rank = (nranks > 1) ? ((b + i) % nranks) : 0u;
      arts_guid_t w =
          arts_edt_create(inc_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = rank, .finish_event = e});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
      total++;
    }
    arts_event_wait(e);
  }

  uint64_t expect = (uint64_t)total;
  arts_guid_t ec = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t c =
      arts_edt_create(check_edt, 1, &expect, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = ec});
  arts_add_dependence(db, c, 0, DB_MODE_RO);
  arts_event_wait(ec);

  arts_printf("PASS val_writer_count_nonneg sum=%u\n", total);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif
