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

/// @file socket_outbox_pending_parity.c
/// @brief Drive the outbox_pending +1/-1 accounting dance to net balance.
///
/// Targets the load-bearing counter invariant in
/// libs/src/core/transport/socket.c:
///   - arts_actual_send: exactly one outbox_pending decrement per call that
///     completes (full send -> 1 sub; hard error -> 1 sub; EAGAIN partial ->
///     NO sub, caller retries).
///   - arts_transport_send_payload: when the header is fully sent it
///     PRE-INCREMENTs outbox_pending to compensate for the second
///     arts_actual_send call (one increment per logical message, one decrement
///     per arts_actual_send).  An off-by-one here makes shutdown-drain hang
///     (outbox_pending never reaches 0) or wrap.
///
/// Scenario shaping: this exercises both wire send paths that touch
/// outbox_pending.
///   (a) Large cross-rank DB payloads (>1 MiB) force the header-then-payload
///       two-send path (arts_transport_send_payload) and EAGAIN partials under
///       socket back-pressure -- the pre-increment compensation path.
///   (b) Many small cross-rank RW transfers force the single-send path
///       (arts_actual_send) with high message churn.
/// After all traffic completes we run a clean shutdown.  The whole point is the
/// shutdown DRAIN: if outbox_pending is miscounted the runtime never observes
/// pending==0 and shutdown hangs forever -- caught by the ctest TIMEOUT (no
/// in-test spin/watchdog).  A clean exit + the PASS token is the assertion that
/// the counter returned to exactly 0.
///
/// Config-agnostic across protocols.  On 1n there is no cross-rank wire
/// traffic, so the test passes trivially (still a valid smoke run).  2n_io
/// (ports>1, multiple sender threads) maximizes the partial-send churn.
/// exposes_runtime_bug: B-outbox-pending-parity (shared with C17/T176).

#include "arts.h"

#include <stdint.h>
#include <stdlib.h>

#define BIG_ELEMS (300u * 1024u) /* 300K * 4B ~= 1.2 MiB -> payload path */
#define SMALL_ROUNDS 64u         /* high small-message churn */

/* ---- Large-payload chain: RW writer (remote) -> RO reader (home). ---- */

void big_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int seed = (unsigned int)paramv[0];
  if (d != NULL) {
    for (unsigned int i = 0; i < BIG_ELEMS; i++) {
      d[i] = seed + i;
    }
  }
}

void big_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int seed = (unsigned int)paramv[0];
  bool ok = (d != NULL);
  if (ok) {
    /* spot-check first/middle/last so the whole payload had to arrive */
    ok = (d[0] == seed) && (d[BIG_ELEMS / 2] == seed + BIG_ELEMS / 2) &&
         (d[BIG_ELEMS - 1] == seed + BIG_ELEMS - 1);
  }
  if (ok) {
    arts_printf("  big chain ok (seed %u)\n", seed);
  } else {
    arts_printf("  FAIL: big chain mismatch (seed %u)\n", seed);
  }
}

/* ---- Small-message churn: RW incrementer ping-pong across ranks. ---- */

void small_inc(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== socket_outbox_pending_parity ===\n");

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W = (nranks > 1) ? 1u : 0u; /* remote when possible */

  /* --- Phase 1: large payload transfer (header+payload two-send dance). --- */
  {
    void *ptr = NULL;
    arts_guid_t db = arts_db_create(
        &ptr, (uint64_t)BIG_ELEMS * sizeof(unsigned int), ARTS_DB,
        ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
    for (unsigned int i = 0; i < BIG_ELEMS; i++) {
      ((unsigned int *)ptr)[i] = 0u;
    }
    arts_db_release(db, DB_MODE_RW);

    unsigned int seed = 0xABCDu;
    arts_guid_t e_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t pw = (uint64_t)seed;
    arts_guid_t w =
        arts_edt_create(big_writer, 1, &pw, 1,
                        &(arts_edt_hint_t){.rank = W, .finish_event = e_w});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
    arts_event_wait(e_w);

    arts_guid_t e_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t r =
        arts_edt_create(big_reader, 1, &pw, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = e_r});
    arts_add_dependence(db, r, 0, DB_MODE_RO);
    arts_event_wait(e_r);
  }

  /* --- Phase 2: many small cross-rank RW transfers (single-send churn). --- */
  {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    for (unsigned int round = 0; round < SMALL_ROUNDS; round++) {
      unsigned int target = (nranks > 1) ? (round % nranks) : 0u;
      arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
      arts_guid_t inc = arts_edt_create(
          small_inc, 0, NULL, 1,
          &(arts_edt_hint_t){.rank = target, .finish_event = e});
      arts_add_dependence(db, inc, 0, DB_MODE_RW);
      arts_event_wait(e);
    }
  }

  arts_printf("PASS: socket_outbox_pending_parity drained clean\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
