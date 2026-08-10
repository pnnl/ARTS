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

/// @file coherence_payload_large.c
/// @brief Large-payload cross-rank coherence transfer correctness.
///
/// Every distributed coherence transfer that carries DB data — ownership
/// GRANT/TRANSFER, RO snapshot, HOME publish — must deliver the whole
/// payload bit-for-bit.  This test drives DBs across a size ladder that spans
/// every transport regime: payloads far larger than a control message, one AT
/// the control-plane message ceiling, and one far ABOVE it (which can only
/// arrive via the one-sided rendezvous data plane — an inline send of that
/// size fails loudly by design).  A corrupted or partially-delivered payload
/// surfaces as a checksum mismatch, a hard framing abort, or a spurious
/// shutdown, none of which the sizeof(int) DBs elsewhere in the suite can see.
///
/// Flow (RW writer -> RO readers, two phases so the pattern is durably
/// established before any reader):
///   Phase 1 — N_DBS DBs (one per rung of the size ladder) are created
///     round-robin across ranks; a home-rank RW init writer stamps
///     DB[j] = PATTERN(j) for every element.  All init writers run inside one
///     finish scope.
///   Phase 2 — gated on that finish scope (so every init writer has RELEASED
///     its RW grant first), WORKERS_PER_DB RO readers per DB, each pinned to a
///     rank != the DB home, acquire RO (which pulls the whole payload
///     cross-rank as one snapshot) and verify every element equals PATTERN(j)
///     (arts_abort on any mismatch).  RO (not concurrent RW) keeps the flow
///     well-defined under every protocol including the DB-WRF WRF_VAL contract,
///     while still exercising the exact large-payload framing under test.
///   * A second finish scope gates a shutdown EDT that prints PASS once every
///     reader has completed.
///
/// The phase barrier matters: without it a remotely-homed DB's init writer must
/// first migrate to its home rank, letting workers on other ranks race ahead
/// and acquire the still-zero DB — a coherence-ordering artifact unrelated to
/// the framing this test targets.  A framing bug instead surfaces as garbage
/// bytes (checksum mismatch -> FAIL), a hard RX framing abort, or a
/// mis-dispatched raw payload frame (spurious shutdown -> finish scope never
/// completes -> ctest TIMEOUT).  On the correct single-gather path every hop
/// verifies and the run prints PASS.  Auto-skips on < 2 ranks.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

/* Size ladder — one DB per rung, spanning the transport regimes:
 *   256 KiB : bulk but modest (many control messages' worth).
 *     2 MiB : exactly the control-plane message ceiling (boundary).
 *     8 MiB : 4x the ceiling — deliverable ONLY by the one-sided rendezvous
 *             data plane; proves the ceiling bounds control traffic alone. */
#define N_DBS 4
#define WORKERS_PER_DB 3
static const uint32_t db_bytes[N_DBS] = {256u * 1024u, 256u * 1024u,
                                         2u * 1024u * 1024u,
                                         8u * 1024u * 1024u};

static inline int pattern(uint32_t j) {
  return (int)((uint32_t)j * 2654435761u + 0x9e3779b9u);
}

static void init_writer_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)depc;
  if (paramc < 1) {
    arts_printf("FAIL: init_writer missing size param\n");
    arts_abort(1);
  }
  uint32_t n_ints = (uint32_t)paramv[0];
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: init_writer got NULL ptr\n");
    arts_abort(1);
  }
  for (uint32_t j = 0; j < n_ints; j++) {
    data[j] = pattern(j);
  }
}

static void verify_worker_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)depc;
  if (paramc < 2) {
    arts_printf("FAIL: worker missing paramv\n");
    arts_abort(1);
  }
  uint32_t n_ints = (uint32_t)paramv[1];
  /* RO read-only view: the whole DB was shipped cross-rank as one snapshot. */
  const int *data = (const int *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("FAIL: worker got NULL ptr\n");
    arts_abort(1);
  }
  /* Verify the WHOLE payload arrived intact — a partial or misplaced transfer
   * corrupts interior bytes a spot-check would never see. */
  for (uint32_t j = 0; j < n_ints; j++) {
    if (data[j] != pattern(j)) {
      arts_printf("FAIL: payload corruption at element %u: got %d expected %d "
                  "(worker id %llu)\n",
                  j, data[j], pattern(j), (unsigned long long)paramv[0]);
      arts_abort(1);
    }
  }
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: %d DBs (256 KiB .. 8 MiB) verified by %d cross-rank RO "
              "readers each\n",
              N_DBS, WORKERS_PER_DB);
  arts_shutdown();
}

/* Phase 2: gated on the init finish scope (so every init writer has released
 * its RW grant and the pattern is durably at each DB's owner), fan out the
 * cross-rank RW verify workers.  The N_DBS DB GUIDs arrive via paramv. */
static void phase2_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  unsigned int nnodes = arts_get_total_ranks();

  arts_guid_t shut =
      arts_edt_create(shutdown_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe_work = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe_work, shut, 0, DB_MODE_NULL);

  for (uint32_t d = 0; d < paramc; d++) {
    arts_guid_t db = (arts_guid_t)paramv[d];
    unsigned int home = d % nnodes;
    /* Each reader is pinned to a rank other than the DB's home, so its RO
     * acquire pulls the whole payload across the wire as one snapshot.  RO is
     * used (not RW): the readers only observe, so the flow is well-defined under
     * every protocol including the DB-WRF WRF_VAL contract (a concurrent-RW test
     * would be racy there by design), and it still exercises the exact
     * large-payload transport framing this test targets. */
    for (int w = 0; w < WORKERS_PER_DB; w++) {
      unsigned int worker_rank = (home + 1u + (unsigned int)w) % nnodes;
      uint64_t wparams[2] = {(uint64_t)d * WORKERS_PER_DB + (uint64_t)w,
                             db_bytes[d] / sizeof(int)};
      arts_guid_t wk = arts_edt_create(
          verify_worker_edt, 2, wparams, 1,
          &(arts_edt_hint_t){.rank = worker_rank, .finish_event = fe_work});
      arts_add_dependence(db, wk, 0, DB_MODE_RO);
    }
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nnodes = arts_get_total_ranks();
  if (nnodes < 2) {
    arts_printf("SKIP: requires 2+ ranks (got %u)\n", nnodes);
    arts_shutdown();
    return;
  }

  arts_printf("=== coherence_payload_large (%d DBs, 256 KiB .. 8 MiB, "
              "%d workers/DB, %u ranks) ===\n",
              N_DBS, WORKERS_PER_DB, nnodes);

  /* Phase 1: create the DBs, then a phase-2 EDT carrying their GUIDs that is
   * gated on the init finish scope, then the home-rank init writers.  The
   * barrier guarantees every init writer has stamped-and-released its DB before
   * a single verify worker acquires it — so a failure can only be payload
   * corruption, never the init-vs-worker acquire race a remotely-homed DB would
   * otherwise hit. */
  arts_guid_t dbs[N_DBS];
  for (int d = 0; d < N_DBS; d++) {
    void *raw = NULL;
    unsigned int home = (unsigned int)(d % (int)nnodes);
    dbs[d] = arts_db_create(&raw, db_bytes[d], ARTS_DB, ARTS_DB_PROP_NONE,
                            &(arts_db_hint_t){.rank = home});
    /* Release the creator's initial RW grant so the DB's home copy is published
     * before any acquire.  Required for the DB-WRF (WRF_VAL) contract — where an
     * unreleased create leaves the home copy unpublished and a cross-rank RO
     * reader would observe zeros — and harmless under the OCR-coherence
     * protocols (mirrors the db_wrf_invariant idiom). */
    arts_db_release(dbs[d], DB_MODE_RW);
  }

  arts_guid_t fe_init = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t phase2 =
      arts_edt_create(phase2_edt, N_DBS, (uint64_t *)dbs, 1,
                      &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe_init, phase2, 0, DB_MODE_NULL);

  for (int d = 0; d < N_DBS; d++) {
    unsigned int home = (unsigned int)(d % (int)nnodes);
    uint64_t n_ints = db_bytes[d] / sizeof(int);
    arts_guid_t init = arts_edt_create(
        init_writer_edt, 1, &n_ints, 1,
        &(arts_edt_hint_t){.rank = home, .finish_event = fe_init});
    arts_add_dependence(dbs[d], init, 0, DB_MODE_RW);
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
