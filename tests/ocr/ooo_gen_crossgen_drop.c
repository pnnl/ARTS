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

/// @file ooo_gen_crossgen_drop.c
/// @brief Targets the documented-but-UNIMPLEMENTED OoO `gen` cross-generation
///        drop (route_table.h gen field + ooo.h comments vs ooo.c).
///
/// route_table.h and ooo.h describe a defer-time `gen` snapshot stored in the
/// OoO payload, dropped at drain-replay when the slot's `gen` has advanced past
/// it — so a stale coherence op (notably an HOME OWNERSHIP_INVALIDATE) that
/// outlives a destroy + labeled-GUID re-create is NOT replayed against the NEW
/// generation's object.  But `arts_ooo_payload_s` has no gen field and ooo.c
/// never reads `item->gen`; cross-generation suppression relies ENTIRELY on
/// each handler's own `was_destroyed`-style guard.
///
/// This test drives a destroy + same-GUID re-create across the generation
/// boundary while a coherence op for the OLD generation is in flight, and
/// asserts the NEW generation reads its OWN value — never a value corrupted by
/// a stale cross-generation replay.  If the per-handler guards suppress the
/// stale op, the test PASSES; if a stale cross-gen op leaks through (the B-gen
/// bug), the new generation's reader observes a wrong value and the test FAILS.
///
/// Construction (labeled GUID; ownership transfer forces deferrable coherence
/// ops):
///   Generation A:
///     - DB created at a fixed labeled GUID, home rank 0, value VAL_A.
///     - A remote RW writer on rank W forces a home(0)->W ownership transfer
///       (HOME may defer OWNERSHIP_INVALIDATE; OWNER/WRF_VAL defer SNAPSHOT /
///       OWNERSHIP / PUBLISH on the before-install path).
///     - The DB is destroyed.
///   Generation B:
///     - The SAME labeled GUID is re-created (gen bumped), value VAL_B.
///     - A reader on home asserts it sees VAL_B, not VAL_A and not garbage.
///
/// Multinode-only (the cross-gen + ownership-transfer race needs >1 rank); on a
/// single rank the test self-skips.  2n_io (multi sender/receiver -> wire
/// reorder) is the configuration most likely to actually float a stale op
/// across the boundary.  RELAXED (WRF_VAL) has no ownership transfer but still
/// has destroy/publish defer paths, so it is exercised too.
///
/// exposes_runtime_bug: targets the B-gen unimplemented cross-generation drop.
/// A stranded waiter is reaped by the ctest TIMEOUT (no in-test spin).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ITERS 64u
#define VAL_A_BASE 0xA0000000u
#define VAL_B_BASE 0xB0000000u

/// Gen-A remote RW writer: forces the ownership transfer away from home and
/// stamps the gen-A value (deferrable coherence ops are emitted by the
/// acquire).
void cg_writer_a(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/// Gen-B reader on home: MUST observe the gen-B value.  A stale cross-gen op
/// replayed against this fresh object would corrupt the value; abort so the
/// exit code is non-zero and ctest reports the bug.
void cg_reader_b(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL: cross-gen stale replay — gen-B reader expected 0x%x "
                  "got 0x%x\n",
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

  arts_printf("=== ooo_gen_crossgen_drop ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    arts_printf("SKIP: ooo_gen_crossgen_drop needs >1 rank\n");
    arts_shutdown();
    return;
  }
  unsigned int W = 1u; /* remote writer rank */

  /* One labeled GUID, re-used across generations (home rank 0). */
  arts_guid_t db = arts_guid_reserve(ARTS_GUID_DB, 0);

  for (unsigned int it = 0; it < ITERS; it++) {
    unsigned int va = VAL_A_BASE + it;
    unsigned int vb = VAL_B_BASE + it;

    /* ---- Generation A ---- */
    void *ptr = arts_db_create_with_guid(db, sizeof(unsigned int), ARTS_DB,
                                         ARTS_DB_PROP_NONE,
                                         &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = va;
    arts_db_release(db, DB_MODE_RW);

    /* Remote RW writer forces the ownership transfer (deferrable coherence
     * ops emitted toward home).  Wait for it via its finish scope so the
     * destroy below is causally after the writer ran. */
    arts_guid_t e_a = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t pva = (uint64_t)va;
    arts_guid_t wa =
        arts_edt_create(cg_writer_a, 1, &pva, 1,
                        &(arts_edt_hint_t){.rank = W, .finish_event = e_a});
    arts_add_dependence(db, wa, 0, DB_MODE_RW);
    arts_event_wait(e_a);

    /* Destroy generation A (bumps the slot gen; any still-in-flight gen-A
     * coherence op for this label is now stale). */
    arts_db_destroy(db);

    /* ---- Generation B (same labeled GUID) ---- */
    void *ptr2 = arts_db_create_with_guid(db, sizeof(unsigned int), ARTS_DB,
                                          ARTS_DB_PROP_NONE,
                                          &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr2)[0] = vb;
    arts_db_release(db, DB_MODE_RW);

    /* Reader on home MUST see the gen-B value — a stale cross-gen replay would
     * corrupt it. */
    arts_guid_t e_b = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t pvb = (uint64_t)vb;
    arts_guid_t rb =
        arts_edt_create(cg_reader_b, 1, &pvb, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = e_b});
    arts_add_dependence(db, rb, 0, DB_MODE_RO);
    arts_event_wait(e_b);

    /* Destroy generation B before re-using the label next iteration. */
    arts_db_destroy(db);
  }

  arts_printf("PASS: ooo_gen_crossgen_drop %u generations, no stale replay\n",
              ITERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
