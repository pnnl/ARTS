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

/// @file event_satisfy_once.c
/// @brief Single-node test for arts_event_satisfy on a default
/// (ONCE-equivalent)
///        event: register one waiter, satisfy slot 0, waiter EDT must run.

#include "arts.h"
#include <assert.h>
#include <stdio.h>

void waiter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: waiter EDT fired from event_satisfy\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== event_satisfy_once ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Default hint = OCR ONCE_T: latch=1, auto_destroy=true,
   * nb_deps_required=1, single-fire. */
  arts_guid_t ev = arts_event_create(NULL);
  assert(ev != NULL_GUID);

  /* Register one waiter EDT.  Wired via arts_add_dependence — when the
   * event fires, the waiter slot 0 is satisfied and the EDT runs. */
  arts_guid_t waiter = arts_edt_create(waiter_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(ev, waiter, 0, DB_MODE_RW);

  /* Slot-0 satisfy via the OCR-aligned wrapper. */
  arts_event_satisfy(ev, NULL_GUID);

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
