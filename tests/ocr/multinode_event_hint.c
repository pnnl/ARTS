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

/// @file multinode_event_hint.c
/// @brief Tests cross-rank event delivery for ONCE and IDEMPOTENT hints.
///
/// Sub-test A (ONCE): rank 0 creates a ONCE event; rank-1 waiter adds a dep
/// BEFORE the satisfier EDT is dispatched (ordering guaranteed by dispatch
/// sequence in main_edt), then a satisfier EDT on rank 0 fires it.
///
/// Sub-test B (IDEMPOTENT): rank 0 creates and fires an IDEM event via a
/// satisfier EDT; an idem_setup EDT on rank 1 depends on the IDEM event
/// (so it runs after it fires) and then registers a second late dependent
/// on the same persistent event — which should deliver immediately.
///
/// Exits cleanly on single-node runs (prints SKIP).

#include "arts.h"
#include <stdint.h>
#include <stdio.h>

/* ----------------------------- Sub-test A: ONCE --------------------------- */

/// Satisfier: fires the ONCE event on rank 0.
static void once_satisfier(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t ev = (arts_guid_t)paramv[0];
  arts_event_satisfy(ev, NULL_GUID);
}

/// Waiter: runs on rank 1 after the ONCE event fires.
static void once_waiter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: ONCE cross-rank event delivered to rank-1 waiter\n");
}

/* ------------------------- Sub-test B: IDEMPOTENT ------------------------- */

/// Satisfier: fires the IDEM event on rank 0.
static void idem_satisfier(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t ev = (arts_guid_t)paramv[0];
  arts_event_satisfy(ev, NULL_GUID);
}

/// Late waiter: runs on rank 1 after the late dep on the already-fired IDEM
/// event delivers immediately.
static void idem_late_waiter(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf(
      "  PASS: IDEMPOTENT late dep on rank-1 fires after cross-rank fire\n");
}

/// Setup EDT: runs on rank 1 AFTER the IDEM event has fired (it is wired as
/// a dep on the IDEM event itself), then registers a second late dependent
/// on the same persistent event.
static void idem_setup_on_rank1(uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t ev = (arts_guid_t)paramv[0];
  arts_guid_t fe = (arts_guid_t)paramv[1];
  arts_guid_t late =
      arts_edt_create(idem_late_waiter, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
  arts_add_dependence(ev, late, 0, DB_MODE_RW);
}

/* ----------------------------- Shutdown EDT ------------------------------- */

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_shutdown();
}

/* ================================ main_edt ================================ */

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int ranks = arts_get_total_ranks();
  if (ranks < 2) {
    arts_printf("SKIP: multinode_event_hint requires node_count >= 2\n");
    arts_shutdown();
    return;
  }

  arts_printf("=== multinode_event_hint (%u ranks) ===\n", ranks);

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  /* Sub-test A: ONCE — rank-1 waiter dep registered before satisfier fires. */
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_ONCE;
    arts_guid_t ev_a = arts_event_create(&h);
    uint64_t ev_param = (uint64_t)ev_a;

    /* Register waiter on rank 1 first (before dispatching satisfier). */
    arts_guid_t waiter =
        arts_edt_create(once_waiter, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    arts_add_dependence(ev_a, waiter, 0, DB_MODE_RW);

    /* Satisfier on rank 0 fires the event. */
    arts_edt_create(once_satisfier, 1, &ev_param, 0,
                    &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  }

  /* Sub-test B: IDEMPOTENT — setup EDT on rank 1 wired as dep on IDEM event,
   * so it runs after the event fires, then registers a further late waiter. */
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_IDEMPOTENT;
    arts_guid_t ev_b = arts_event_create(&h);
    uint64_t params[2] = {(uint64_t)ev_b, (uint64_t)fe};

    /* idem_setup_on_rank1 itself depends on the IDEM event (slot 0), so it
     * is guaranteed to run only after idem_satisfier has fired ev_b. */
    arts_guid_t setup =
        arts_edt_create(idem_setup_on_rank1, 2, params, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    arts_add_dependence(ev_b, setup, 0, DB_MODE_RW);

    /* Satisfier on rank 0 fires the IDEM event. */
    arts_edt_create(idem_satisfier, 1, params, 0,
                    &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
