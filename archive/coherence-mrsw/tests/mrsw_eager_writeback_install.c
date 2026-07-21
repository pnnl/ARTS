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

/// @file mrsw_eager_writeback_install.c
/// @brief MRSW EAGER synchronous WRITEBACK + ack-on-MISS (config_specific).
///
/// In MRSW EAGER, arts_db_release_rw on a non-home token holder performs a
/// SYNCHRONOUS WRITEBACK: it sends the buffer to home with a stack-local sem
/// and blocks in await_writeback_ack.  The home handler buf_installs
/// monotonically (stale out-of-order WRITEBACKs are ignored) and posts
/// WRITEBACK_ACK exactly once.  The dispatcher posts the ACK on BOTH a HIT and
/// a MISS, so a home cache torn down between the WRITEBACK send and the ACK
/// never strands the blocked releaser.
///
/// Two facets driven here:
///   (A) Monotonic install / exactly-once ACK: a chain of remote RW writers
///   each
///       WRITEBACK to home in order; the final home-side reader must observe
///       the last writer's value (no stale install winning, no lost ACK -> no
///       hang).
///   (B) Destroy-of-home AFTER writer quiesces: each iteration gates the
///       destroyer on the writer's per-iteration finish event (feB) via
///       DB_MODE_NULL, so destroy is causally after the synchronous WRITEBACK
///       completes.  Destroy-in-use is OCR undefined behaviour; this facet
///       verifies the runtime stays free of hangs and crashes on the legal
///       path.
///
/// EAGER + MRSW only (LAZY has no synchronous WRITEBACK).  Requires >= 2 ranks
/// so the writer is non-home (home releases skip the WRITEBACK entirely); SKIPs
/// on 1n.  A stranded releaser is caught by the ctest TIMEOUT.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW) || !defined(ARTS_TIMING_EAGER)

int main(void) {
  printf("SKIP mrsw_eager_writeback_install: MRSW+EAGER-only\n");
  return 0;
}

#else

#define CHAIN_LEN 32u /* facet A: ordered WRITEBACK chain */
#define RACE_ITERS 20 /* facet B: destroy-vs-ACK races */

/* ---- facet A: ordered WRITEBACK chain to home ---- */

static void rw_set_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    d[0] = paramv[0]; /* stamp a strictly increasing value */
  }
}

static void chain_reader_edt(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t expect = paramv[0];
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v != expect) {
    (void)fprintf(stderr,
                  "FAIL: home read %llu != %llu (stale WRITEBACK install)\n",
                  (unsigned long long)v, (unsigned long long)expect);
    arts_abort(1);
  }
}

/* ---- facet B: destroy racing the synchronous WRITEBACK ACK ---- */

static void race_writer_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
  /* On return the runtime performs the synchronous WRITEBACK; a destroyer EDT
   * may tear down the home cache concurrently, exercising ack-on-MISS. */
}

static void destroyer_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_db_destroy((arts_guid_t)paramv[0]);
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: mrsw_eager_writeback_install (chain + %d race iters, no "
              "hang)\n",
              RACE_ITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_eager_writeback_install ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2u) {
    arts_printf("SKIP: mrsw_eager_writeback_install requires >= 2 ranks\n");
    arts_shutdown();
    return;
  }

  arts_guid_t shut =
      arts_edt_create(shutdown_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  /* ---- facet A ---- */
  {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((uint64_t *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* Facet A's own finish scope: gates the home reader AFTER the whole write
     * chain.  (Previously chain_reader was merely a MEMBER of fe and depended
     * only on db RO — the passive immediate db satisfy fired it before any
     * writer ran, so it read the initial 0.  Being a finish-scope member does
     * NOT order an EDT after the scope's other members; an explicit dependence
     * on the scope's finish event does.) */
    arts_guid_t feA = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    /* A REAL ordered chain: each writer runs only after the previous writer's
     * output event fires, which the runtime satisfies AFTER that writer's
     * release (its synchronous WRITEBACK to home has completed).  This makes
     * the execution order equal the creation order, so the last writer (highest
     * value) is well-defined.  Without this edge the writers would be
     * concurrent RW acquirers serialized by the single-writer TOKEN in
     * ownership-grant order (per-rank, non-deterministic) — the final value
     * would then be the last ownership-rank's last writer, NOT the last-created
     * one, and the reader's check below would spuriously fail at higher rank
     * counts. */
    uint64_t last = 0u;
    arts_guid_t prev_oe = NULL_GUID;
    for (unsigned int i = 0; i < CHAIN_LEN; i++) {
      uint64_t val = (uint64_t)(0x1000u + i);
      last = val;
      arts_guid_t oe = arts_event_create(&ARTS_EVENT_HINT_LATCH(1));
      uint32_t depc = (i == 0u) ? 1u : 2u;
      arts_guid_t w =
          arts_edt_create(rw_set_edt, 1, &val, depc,
                          &(arts_edt_hint_t){.rank = (i % (nranks - 1u)) + 1u,
                                             .finish_event = feA,
                                             .output_event = oe});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
      if (i > 0u) {
        arts_add_dependence(prev_oe, w, 1,
                            DB_MODE_NULL); /* run after prev WB */
      }
      prev_oe = oe;
    }
    /* Home reader gated on feA (whole chain done) via slot 1, reads db (slot 0,
     * RO); joins the outer fe so the shutdown EDT still waits for it.  Must see
     * the final value, proving the synchronous WRITEBACKs installed in order
     * and every ACK landed. */
    arts_guid_t rd =
        arts_edt_create(chain_reader_edt, 1, &last, 2,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, rd, 0, DB_MODE_RO);
    arts_add_dependence(feA, rd, 1, DB_MODE_NULL);
  }

  /* ---- facet B ---- */
  for (int it = 0; it < RACE_ITERS; it++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((uint64_t *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* Dedicated finish scope for this iteration's writer.  The destroyer
     * depends on feB so it only runs after the synchronous WRITEBACK + release
     * completes — destroy-in-use is OCR undefined behaviour. */
    arts_guid_t feB = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    /* Remote writer triggers a synchronous WRITEBACK on release. */
    arts_guid_t w =
        arts_edt_create(race_writer_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 1u, .finish_event = feB});
    arts_add_dependence(db, w, 0, DB_MODE_RW);

    /* Destroyer on the DB home rank (rank 0), gated after the writer. */
    uint64_t dbv = (uint64_t)db;
    arts_guid_t dz =
        arts_edt_create(destroyer_edt, 1, &dbv, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(feB, dz, 0, DB_MODE_NULL);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW && ARTS_TIMING_EAGER */
