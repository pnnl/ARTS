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

/// @file snapshot_response_3case.c
/// @brief T108 — SNAPSHOT_RESPONSE three-case classifier (B014/B028).
///
/// The 1:1 SNAPSHOT_RESPONSE handler classifies an incoming reply by a
/// monotonic version against the locally-installed buffer:
///   case 1 (a->version <= buf): resume the parked reader against the live
///          buffer (nothing newer to install);
///   case 2 (data + a->version > buf): install + drain-all the reorder buffer
///          + resume self;
///   case 3 (NO_DATA + a->version > buf): the with-data reply was reordered
///          behind a NO_DATA reply — push onto pending_snapshot, then re-check
///          (race recovery) so a concurrent case-2 install does not strand us.
/// The zero-size / NULL-data branch selection is the `total` (header) vs the
/// payload predicate (B028): a sentinel (db_size==0) DB drives a snapshot whose
/// header.total uses `data?data_size:0` while the payload branch additionally
/// requires `data && data_size>0`.
///
/// This is a black-box runtime exercise of those paths.  It builds, per
/// iteration, a clean producer-write / many-readers happens-before chain so
/// remote RO acquires generate SNAPSHOT_REQUEST→SNAPSHOT_RESPONSE rounds whose
/// wire reorder (multiple sender/receiver threads in 2n_io) drives case 2 vs
/// case 3.  It runs both a sized DB (case-1/2/3 with payload) and a sentinel
/// zero-size DB (B028 total-vs-payload predicate).  Every reader asserts it
/// sees the writer's value; a stranded case-3 reader that never installs is
/// caught by the ctest TIMEOUT (no in-test spin).
///
/// Config gate: SNAPSHOT_RESPONSE is a snapshot-bearing protocol message that
/// exists only in the non-LOCK builds (MRNEW/MRSW/MRMW route RO through the
/// snapshot path).  LOCK has no snapshot machinery → compile-time self-skip.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if defined(ARTS_PROTOCOL_LOCK)
int main(void) {
  printf("SKIP snapshot_response_3case: non-LOCK (snapshot-bearing) only\n");
  return 0;
}
#else

#define ITERS 200u
#define WRITE_BASE 0x30000000u

/// writer: RW acquire; stamp the per-iter sentinel into the (sized) DB.
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/// reader: RO acquire, causally after the writer; MUST observe the new value.
/// arts_abort(1) on any mismatch so the process exits non-zero.
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr, "FAIL: stale snapshot — expected 0x%x got 0x%x\n",
                  expect, d ? d[0] : 0u);
    arts_abort(1);
  }
}

/// sentinel reader: zero-size DB has no payload; depv[0].ptr is NULL by design.
/// The point is that the parked reader WAKES (case-1 path on an empty buffer),
/// not that it reads anything.
static void sentinel_reader_edt(uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== snapshot_response_3case ===\n");

  unsigned int nranks = arts_get_total_ranks();
  /* The reordering that drives case 2 vs case 3 needs cross-rank readers; on a
   * single rank the snapshot still flows through the self-send Cat-C path
   * (case-1 resume), exercising the classifier without reorder. */
  unsigned int W = (nranks > 1) ? 1u : 0u;

  /* ----- Part A: sized DB — case 1/2/3 with payload (B014) ----- */
  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  for (unsigned int it = 0; it < ITERS; it++) {
    uint64_t vnew = (uint64_t)(WRITE_BASE + it);

    arts_guid_t e_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t wr =
        arts_edt_create(writer_edt, 1, &vnew, 1,
                        &(arts_edt_hint_t){.rank = W, .finish_event = e_w});
    arts_add_dependence(db, wr, 0, DB_MODE_RW);
    arts_event_wait(e_w);

    /* Fan out concurrent RO readers across all ranks: each foreign reader
     * issues a SNAPSHOT_REQUEST whose 1:1 SNAPSHOT_RESPONSE the classifier
     * must resolve.  Multiple readers per generation maximise the chance a
     * NO_DATA reply lands ahead of a with-data reply (case 3 push+recheck). */
    arts_guid_t e_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int r = 0; r < nranks; r++) {
      arts_guid_t rd =
          arts_edt_create(reader_edt, 1, &vnew, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = e_r});
      arts_add_dependence(db, rd, 0, DB_MODE_RO);
    }
    arts_event_wait(e_r);
  }

  /* ----- Part B: sentinel zero-size DB — total-vs-payload predicate (B028) ---
   * The snapshot sender's header `total` uses `data?data_size:0` while the
   * payload branch requires `data && data_size>0`; a zero-size DB makes those
   * two predicates differ.  A cross-rank RO acquire of a sentinel DB must still
   * wake its parked reader (no payload underflow / corrupt install). */
  void *sptr = NULL;
  arts_guid_t sdb = arts_db_create(&sptr, 0, ARTS_DB, ARTS_DB_PROP_NONE,
                                   &(arts_db_hint_t){.rank = 0});
  arts_db_release(sdb, DB_MODE_RW);

  for (unsigned int it = 0; it < ITERS; it++) {
    arts_guid_t e_sw = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t sw =
        arts_edt_create(sentinel_reader_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = W, .finish_event = e_sw});
    arts_add_dependence(sdb, sw, 0, DB_MODE_RW);
    arts_event_wait(e_sw);

    arts_guid_t e_sr = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int r = 0; r < nranks; r++) {
      arts_guid_t srd =
          arts_edt_create(sentinel_reader_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = e_sr});
      arts_add_dependence(sdb, srd, 0, DB_MODE_RO);
    }
    arts_event_wait(e_sr);
  }

  arts_printf("PASS: snapshot_response_3case %u iters x %u ranks\n", ITERS,
              nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_LOCK */
