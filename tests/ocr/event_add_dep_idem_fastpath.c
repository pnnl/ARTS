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

/// @file event_add_dep_idem_fastpath.c
/// @brief Single-node test for the IDEM late-add fast path: a waiter
/// registered via arts_add_dependence AFTER the event has fired must run
/// immediately (no hang, no lost wakeup).

#include "arts.h"
#include <assert.h>
#include <stdio.h>

void late_waiter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: late_waiter_edt fired via add_dependence fast path\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== event_add_dep_idem_fastpath ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* IDEM-equivalent: persistent (life_count=INT32_MAX).  The event
   * survives the first fire so a late add_dependence can observe
   * `fired=true` and take the fast-path: data delivered immediately,
   * no enqueue. */
  arts_event_hint_t hint = ARTS_EVENT_HINT_IDEMPOTENT;
  arts_guid_t ev = arts_event_create(&hint);
  assert(ev != NULL_GUID);

  /* Fire the event before any waiter exists. */
  arts_event_satisfy(ev, NULL_GUID);

  /* Register the waiter AFTER the satisfy.  The IDEM fast path in
   * arts_add_dependence must observe fired=true and signal slot 0
   * inline, so the waiter EDT runs with no hang. */
  arts_guid_t waiter =
      arts_edt_create(late_waiter_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(ev, waiter, 0, DB_MODE_RW);

  arts_event_wait(fe);

  /* Cleanup the IDEM-style event we kept alive. */
  arts_event_destroy(ev);

  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
