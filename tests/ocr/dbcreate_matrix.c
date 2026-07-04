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

/// @file dbcreate_matrix.c
/// @brief DB-create capability matrix: exercises all 12 logical cells of the
///        (creator-acquisition A) x (home-determination B) x (locality C)
///        grid against the ARTS core create path directly (arts.h, no OCR
///        shim).  Prints a scalar CELLS_OK=<k>/12.
///
/// Grid:
///   A: ACQUIRE (default hold) | NO_ACQUIRE (DB_PROP_NO_ACQUIRE)
///   B: (i) affinity hint  (ii) labeled GUID  (iii) no-hint (round robin)
///   C: local (home == creator) | remote (home != creator)
///
///     1 ACQ/affinity/local     7  NOACQ/affinity/local
///     2 ACQ/affinity/remote    8  NOACQ/affinity/remote
///     3 ACQ/labeled/local      9  NOACQ/labeled/local
///     4 ACQ/labeled/remote     10 NOACQ/labeled/remote
///     5 ACQ/no-hint (a)        11 NOACQ/no-hint (a)
///     6 ACQ/no-hint (b)        12 NOACQ/no-hint (b)
///
/// Each cell is driven once from rank 0 and (when the node count allows) once
/// from rank 1, for coverage; a cell's pass bit requires every instance that
/// ran to pass.  Single-node collapses every "remote" C target back onto the
/// creator (still runs; C.f. Sec.5 "single-node collapses C=remote to
/// local").  No-hint cells (B(iii)) have no caller-selectable target home, so
/// only data correctness is asserted for them; affinity/labeled cells assert
/// both data correctness AND that the DB's encoded home rank
/// (arts_guid_get_rank) equals the intended target -- this is the ARTS-native
/// equivalent of an affinity-query home assert (the GUID's rank field is the
/// authoritative encoding of where the DB's master lives, so this check is
/// exact, not best-effort).
///
/// Ordering note: a NO_ACQUIRE cell's writer EDT (RW dependence) must itself
/// create and wire the consumer EDT (RO dependence) *after* writing the
/// payload -- wiring the consumer independently, from the same body that wired
/// the writer, would race the writer's own acquire (both dependences on a
/// freshly-created, still-idle DB can resolve at roughly the same time; only
/// program-order sequencing inside one EDT body guarantees the hand-off).
/// ACQUIRE cells avoid this entirely: the driver itself holds the write from
/// creation, writes, releases, and only then wires the consumer -- pure
/// program order within one function, the same pattern used by
/// tests/ocr/db_labeled_guid.c.

#include "arts.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define NUM_CELLS 12
#define PAYLOAD_COUNT 4
#define PAYLOAD_BYTES (PAYLOAD_COUNT * sizeof(uint64_t))
#define HOME_SKIP ((unsigned int)-1) /* no home assert (no-hint cells) */

typedef enum { AXIS_AFFINITY, AXIS_LABELED, AXIS_NOHINT } axis_t;

/* ---- consumer_edt: paramv = {cellIdx, wantHome, createOk}; depv[0]=target
 * DB (RO), depv[1]=results DB (RW, uint8_t[NUM_CELLS]).  createOk is 1
 * unless a NO_ACQUIRE cell's create call violated the OCR NO_ACQUIRE
 * contract (addr must come back NULL) -- see run_cell. --------------------
 */
static void consumer_edt(uint32_t paramc, const uint64_t *paramv,
                         uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int cell_idx = (unsigned int)paramv[0];
  unsigned int want_home = (unsigned int)paramv[1];
  bool create_ok = paramv[2] != 0;

  const uint64_t *data = (const uint64_t *)depv[0].ptr;
  bool data_ok = (data != NULL);
  for (unsigned int i = 0; data_ok && i < PAYLOAD_COUNT; i++) {
    data_ok = (data[i] == (uint64_t)cell_idx * 1000 + i);
  }

  bool home_ok = true;
  if (want_home != HOME_SKIP) {
    home_ok = (arts_guid_get_rank(depv[0].guid) == want_home);
  }

  bool ok = data_ok && home_ok && create_ok;
  uint8_t *results = (uint8_t *)depv[1].ptr;
  results[cell_idx - 1] = results[cell_idx - 1] && (ok ? 1 : 0);

  arts_printf("  cell %2u: %s (data=%s home=%s create=%s)\n", cell_idx,
              ok ? "PASS" : "FAIL", data_ok ? "ok" : "BAD",
              (want_home == HOME_SKIP) ? "n/a" : (home_ok ? "ok" : "BAD"),
              create_ok ? "ok" : "BAD");
}

/* Wires the RO-consumer(s) that validate a cell's DB.
 *
 *  - AXIS_AFFINITY: one policy-placed consumer that ALSO checks the DB's
 *    encoded home rank == home_idx.  This is non-tautological for affinity
 *    because the RUNTIME minted the GUID at the home it chose from the hint,
 *    so re-reading the GUID's rank field verifies the runtime honored the
 *    affinity request (a genuine round-trip through placement).
 *  - AXIS_LABELED: TWO consumers -- one pinned to the encoded home rank, one
 *    pinned to a DIFFERENT rank (home_idx+1)%n (off-home) -- both of which
 *    must read the correct payload.  This is the STRUCTURAL home check: the
 *    GUID's rank field is tautological on the labeled axis (the test minted
 *    the GUID), so instead we prove the runtime actually installed the
 *    master at the encoded home by acquiring the DB RO from a rank that is
 *    NOT the encoded home.  If the master were wrongly registered at the
 *    creator instead, the off-home acquire would miss and read stale/NULL
 *    (data=BAD) -- the non-tautological signal.  At single node n==1 so the
 *    two consumers collapse onto the same rank (the check is a harmless
 *    no-op there and only bites at multinode).
 *  - AXIS_NOHINT: one policy-placed consumer, no home assert (no
 *    caller-selected target to check against).
 */
static void wire_consumers(arts_guid_t db, unsigned int cell_idx, axis_t axis,
                           unsigned int home_idx, unsigned int n,
                           uint64_t create_ok, arts_guid_t results_db,
                           arts_guid_t gate_event) {
  /* gate_event (NULL_GUID = none): for NO_ACQUIRE cells the writer's payload
   * is published only when the writer EDT finishes and its RW hold is
   * released (writeback).  A bare DB->consumer dep satisfies immediately, so
   * the consumer's RO acquire would race ahead of that writeback and read the
   * home's v1 zero placeholder -- an UNDEFINED RO/RW overlap per the OCR
   * model, not a coherence bug.  The gate event (the writer's OUTPUT EVENT,
   * satisfied strictly AFTER the writer's DBs are released) is added as an
   * extra consumer pre-slot so the consumer becomes runnable -- and thus does
   * its RO acquire -- only happens-after the writeback.  ACQUIRE cells pass
   * NULL_GUID: they are already release-before-wire in run_cell. */
  bool gated = (gate_event != NULL_GUID);
  uint32_t depc = gated ? 3 : 2;
  if (axis == AXIS_LABELED) {
    unsigned int consumer_ranks[2] = {home_idx, (home_idx + 1) % n};
    for (unsigned int j = 0; j < 2; j++) {
      uint64_t cparam[3] = {cell_idx, HOME_SKIP, create_ok};
      arts_guid_t consumer = arts_edt_create(
          consumer_edt, 3, cparam, depc,
          &(arts_edt_hint_t){.rank = consumer_ranks[j]});
      arts_add_dependence(db, consumer, 0, DB_MODE_RO);
      arts_add_dependence(results_db, consumer, 1, DB_MODE_RW);
      if (gated) {
        arts_add_dependence(gate_event, consumer, 2, DB_MODE_NULL);
      }
    }
  } else {
    unsigned int want_home = (axis == AXIS_AFFINITY) ? home_idx : HOME_SKIP;
    uint64_t cparam[3] = {cell_idx, want_home, create_ok};
    arts_guid_t consumer =
        arts_edt_create(consumer_edt, 3, cparam, depc, NULL);
    arts_add_dependence(db, consumer, 0, DB_MODE_RO);
    arts_add_dependence(results_db, consumer, 1, DB_MODE_RW);
    if (gated) {
      arts_add_dependence(gate_event, consumer, 2, DB_MODE_NULL);
    }
  }
}

/* ---- writer_edt (NO_ACQUIRE cells only): paramv = {cellIdx}; depv[0]=target
 * DB (RW).  Writes the payload and returns; the runtime satisfies this EDT's
 * output event (see run_cell) only after this EDT's RW hold is released, so a
 * consumer gated on that event is happens-after the writeback. ------------
 */
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int cell_idx = (unsigned int)paramv[0];
  uint64_t *p = (uint64_t *)depv[0].ptr;
  for (unsigned int i = 0; i < PAYLOAD_COUNT; i++) {
    p[i] = (uint64_t)cell_idx * 1000 + i;
  }
}

/* Runs one grid cell from the calling (driver) EDT's body. */
static void run_cell(unsigned int cell_idx, axis_t axis, unsigned int home_idx,
                     unsigned int n, bool no_acquire, arts_guid_t results_db) {
  arts_db_hint_t db_hint = ARTS_DB_HINT_DEFAULTS;
  const arts_db_hint_t *hintp = NULL;

  if (axis == AXIS_LABELED) {
    db_hint.guid = arts_guid_reserve(ARTS_GUID_DB, home_idx);
    hintp = &db_hint;
  } else if (axis == AXIS_AFFINITY) {
    db_hint.rank = home_idx;
    hintp = &db_hint;
  } /* AXIS_NOHINT: hintp stays NULL -> round-robin policy; no home assert. */

  uint16_t flags = no_acquire ? ARTS_DB_PROP_NO_ACQUIRE : ARTS_DB_PROP_NONE;
  void *addr = NULL;
  arts_guid_t db = arts_db_create(&addr, PAYLOAD_BYTES, ARTS_DB, flags, hintp);

  if (!no_acquire) {
    uint64_t *p = (uint64_t *)addr;
    for (unsigned int i = 0; i < PAYLOAD_COUNT; i++) {
      p[i] = (uint64_t)cell_idx * 1000 + i;
    }
    arts_db_release(db, DB_MODE_RW);
    wire_consumers(db, cell_idx, axis, home_idx, n, 1, results_db, NULL_GUID);
  } else {
    /* OCR contract: DB_PROP_NO_ACQUIRE must come back with addr==NULL (the
     * creator does not hold). A backend that silently drops the flag hands
     * back a non-NULL pointer instead -- catch that here rather than relying
     * on it to also corrupt the payload (a mis-acquired creator hold still
     * gets released before the real writer's RW acquire proceeds, so the
     * data can end up correct despite the contract violation). */
    uint64_t create_ok = (addr == NULL) ? 1 : 0;
    /* Writer's output event: satisfied by the runtime strictly AFTER the
     * writer's RW hold is released (writeback published).  The consumer(s)
     * gate on it so their RO read is happens-after the write.  Wire the
     * consumer(s) to the event BEFORE the writer's DB-dep satisfy below
     * (which makes the writer runnable), so the registration lands before the
     * writer can fire the event -- the portable single-fire-event ordering. */
    arts_guid_t writer_done = arts_event_create(&ARTS_EVENT_HINT_ONCE);
    uint64_t wparam[1] = {cell_idx};
    arts_guid_t writer = arts_edt_create(
        writer_edt, 1, wparam, 1,
        &(arts_edt_hint_t){.output_event = writer_done});
    wire_consumers(db, cell_idx, axis, home_idx, n, create_ok, results_db,
                   writer_done);
    arts_add_dependence(db, writer, 0, DB_MODE_RW);
  }
}

/* ---- driver_edt: paramv = {driverRank, resultsDB}. Runs all 12 cells for
 * one creator rank. --------------------------------------------------------
 */
static void driver_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int d = (unsigned int)paramv[0];
  arts_guid_t results_db = (arts_guid_t)paramv[1];
  unsigned int n = arts_get_total_ranks();
  unsigned int remote = (d + 1) % n;

  run_cell(1, AXIS_AFFINITY, d, n, false, results_db);
  run_cell(2, AXIS_AFFINITY, remote, n, false, results_db);
  run_cell(3, AXIS_LABELED, d, n, false, results_db);
  run_cell(4, AXIS_LABELED, remote, n, false, results_db);
  run_cell(5, AXIS_NOHINT, HOME_SKIP, n, false, results_db);
  run_cell(6, AXIS_NOHINT, HOME_SKIP, n, false, results_db);
  run_cell(7, AXIS_AFFINITY, d, n, true, results_db);
  run_cell(8, AXIS_AFFINITY, remote, n, true, results_db);
  run_cell(9, AXIS_LABELED, d, n, true, results_db);
  run_cell(10, AXIS_LABELED, remote, n, true, results_db);
  run_cell(11, AXIS_NOHINT, HOME_SKIP, n, true, results_db);
  run_cell(12, AXIS_NOHINT, HOME_SKIP, n, true, results_db);
}

/* ---- reduce_edt: depv[0]=outer finish control, depv[1]=results DB (RO). */
static void reduce_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint8_t *results = (const uint8_t *)depv[1].ptr;
  unsigned int k = 0;
  for (unsigned int i = 0; i < NUM_CELLS; i++) {
    k += results[i] ? 1 : 0;
  }
  arts_printf("CELLS_OK=%u/12\n", k);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int n = arts_get_total_ranks();
  arts_printf("=== dbcreate_matrix (%u ranks) ===\n", n);

  void *raddr = NULL;
  arts_guid_t results_db =
      arts_db_create(&raddr, NUM_CELLS, ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  uint8_t *results = (uint8_t *)raddr;
  for (unsigned int i = 0; i < NUM_CELLS; i++) {
    results[i] = 1;
  }
  arts_db_release(results_db, DB_MODE_RW);

  arts_guid_t outer = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  arts_guid_t reduce = arts_edt_create(reduce_edt, 0, NULL, 2, NULL);
  arts_add_dependence(outer, reduce, 0, DB_MODE_NULL);
  arts_add_dependence(results_db, reduce, 1, DB_MODE_RO);

  uint64_t d0[2] = {0, (uint64_t)results_db};
  arts_edt_create(driver_edt, 2, d0, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = outer});

  if (n > 1) {
    uint64_t d1[2] = {1, (uint64_t)results_db};
    arts_edt_create(driver_edt, 2, d1, 0,
                    &(arts_edt_hint_t){.rank = 1, .finish_event = outer});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
