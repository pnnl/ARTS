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
#include <stdio.h>

#define USE_GNU
#include <sched.h>

#include "arts.h"
#include "arts/runtime/rt.h"

void test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
#ifdef __linux__
  arts_printf("Running edt %u on %u %u, %u\n", arts_get_current_guid(),
         arts_get_current_node(), arts_get_current_worker(), sched_getcpu());
#else
  arts_printf("Running edt %u on %u %u\n", arts_get_current_guid(),
         arts_get_current_node(), arts_get_current_worker());
#endif
}

void exit_program(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  arts_printf("Exit: %u\n", depv[0].guid);
  arts_shutdown();
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("Main EDT %u\n", arts_get_current_guid());
  arts_printf("Starting\n");
  arts_guid_t exit_guid = arts_guid_reserve(ARTS_EDT, 0);
  arts_edt_create_with_guid(exit_program, exit_guid, 0, NULL, 1);
  arts_guid_t epoch_guid = arts_initialize_and_start_epoch(exit_guid, 0);

  int number_of_workers = (int)arts_get_total_workers();
  for (int i = 0; i < number_of_workers; i++) {
    uint64_t args[3];
    arts_guid_t guid = arts_edt_create_with_epoch(test, 3, args, 0, epoch_guid, &(arts_hint_t){.route = 0});
  }
  for (int i = 0; i < 100000000; i++) {
    // Simulate some work
    if (i % 10000000 == 0) {
      printf("Thread %d is working on iteration %d\n", arts_get_current_worker(),
             i);
    }
  }
  arts_wait_on_handle(epoch_guid);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
