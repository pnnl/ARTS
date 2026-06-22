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

/// @file mrsw_writer_count_state.c
/// @brief MRSW writer_count {2,1,0} state-machine stress (config_specific).
///
/// The MRSW cache encodes ownership in a single writer_count with states
/// {2,1,0}: 2 = sentinel+token (one local RW writer in charge), 1 = idle owner
/// (sentinel only) OR invalidated/draining (token only), 0 = not owner.  Two
/// invariants this test protects black-box:
///   (1) writer_count never goes below 0 / never skips the token floor during
///       install — exposed as a hang or wrong final value if the {2,1,0}
///       machine underflows.
///   (2) An RO local-hit in a non-invalidated epoch toggles 1<->2 only and
///       never reaches 0, so concurrent RO readers interleaved with the RW
///       chain always observe a consistent (token-floor) value, never a
///       half-installed one.
///
/// Construction (single home rank, but config-agnostic): a chain of RW EDTs
/// each increment a counter DB by 1; between consecutive RW writes a fan of RO
/// readers acquire the SAME DB and must observe a monotonically non-decreasing
/// value (the idle-owner 1<->2 toggle must never expose 0 / stale).  Each RW
/// runs exactly once -> final == N.  The interleaved RO readers assert
/// value >= their causal lower bound and value <= N.  A mismatch -> arts_abort.
///
/// MRSW-only: under MRNEW/MRMW/LOCK the writer_count {2,1,0} encoding does not
/// exist, so the test self-skips at compile time and prints SKIP.  A stranded
/// waiter (underflow -> lost token -> latch never fires) is caught by the ctest
/// TIMEOUT (no in-test watchdog).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW)

int main(void) {
  printf("SKIP mrsw_writer_count_state: MRSW-only\n");
  return 0;
}

#else

#define N_WRITERS 64u
#define N_RO_PER_GAP 4u

/// RW writer: increment the shared counter by exactly 1.
static void rw_inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

/// RO reader: causal lower bound in paramv[0], upper bound N in paramv[1].
/// The value must be within [lo, hi]; a transient 0 / stale read (writer_count
/// dropping below the token floor mid-epoch) shows up as an out-of-range value.
static void ro_check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t lo = paramv[0];
  uint64_t hi = paramv[1];
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v < lo || v > hi) {
    (void)fprintf(
        stderr, "FAIL: RO read out of range — got %llu, expected [%llu,%llu]\n",
        (unsigned long long)v, (unsigned long long)lo, (unsigned long long)hi);
    arts_abort(1);
  }
}

static void final_check_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v != (uint64_t)N_WRITERS) {
    (void)fprintf(stderr, "FAIL: final value %llu != %u\n",
                  (unsigned long long)v, N_WRITERS);
    arts_abort(1);
  }
  arts_printf("PASS: mrsw_writer_count_state final=%u\n", N_WRITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_writer_count_state ===\n");

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: every RW writer + interleaved RO fan under one finish event.  The
   * DB RW chain provides happens-before between writers; the RO fans are
   * unordered vs the chain (each carries its causal bounds and legitimately
   * races mid-flight values).  Waiting on the finish event blocks main_edt
   * until every writer has run AND written back, so the phase-2 checker
   * observes the committed final value. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int i = 0; i < N_WRITERS; i++) {
    arts_guid_t w =
        arts_edt_create(rw_inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, w, 0, DB_MODE_RW);

    /* RO fan after writer i: causal lower bound is 0 (RO is unordered vs the
     * RW chain in OCR), upper bound is N.  The point is that no RO ever sees a
     * value above N or a torn/transient state. */
    for (unsigned int r = 0; r < N_RO_PER_GAP; r++) {
      uint64_t bounds[2] = {0u, (uint64_t)N_WRITERS};
      arts_guid_t ro =
          arts_edt_create(ro_check_edt, 2, bounds, 1,
                          &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
      arts_add_dependence(db, ro, 0, DB_MODE_RO);
    }
  }
  arts_event_wait(fe); /* blocks until ALL writers ran + wrote back */

  /* Phase 2: the final checker, created only now that phase 1 has fully
   * quiesced, so its RO dependency is registered after the committed value is
   * in place and its snapshot observes every increment. */
  arts_guid_t fe2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t fin =
      arts_edt_create(final_check_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe2});
  arts_add_dependence(db, fin, 0, DB_MODE_RO);
  arts_event_wait(fe2);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW */
