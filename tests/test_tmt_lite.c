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
#include "arts.h"
#include "arts/runtime/compute/shad_adapter.h"
#include "arts/system/tmt_lite.h"

#define EDTCOUNT 2
volatile uint64_t ulock = 0x8000000000000000ULL;
volatile unsigned int lock = 0;
unsigned int count = 0;

void locker(volatile unsigned int *lock) {
  // if(!arts_try_lock(lock))
  // {
  //     arts_create_lite_contexts(lock);
  //     while(!arts_try_lock(lock))
  //     {
  //         arts_resume_lite_context();
  //     }
  // }
  // arts_resume_lite_context();
  // if(!arts_try_lock(lock))
  // {
  //     arts_create_lite_contexts();
  //     while(!arts_try_lock(lock)) { pthread_yield(); }
  // }
  // arts_resume_lite_context();
}

void tester(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
            arts_edt_dep_t depv[]) {
  // locker(&lock);
  arts_shad_tmt_lock(&ulock);
  unsigned int local = ++count;
  ARTS_PRINTF("Done  %u:%u Local: %u\n", arts_get_current_worker(),
         arts_tmt_lite_get_alias(), local);
  // if(!arts_get_current_worker() && !arts_tmt_lite_get_alias())
  // sleep(5);
  arts_tmt_lite_get_alias();
  arts_shad_tmt_unlock(&ulock);
  // arts_unlock(&lock);

  if (local == EDTCOUNT) {
    ARTS_PRINTF("SHUTTING DOWN %u\n", count);
    arts_shutdown();
  }
}

void init_per_node(unsigned int node_id, unsigned int worker_id, int argc,
                 char **argv) {
  ARTS_PRINTF("%u -- %u\n", arts_get_total_workers(), arts_get_current_worker());
}

void init_per_worker(unsigned int node_id, unsigned int worker_id, int argc,
                   char **argv) {
  for (unsigned int i = 0; i < EDTCOUNT; i++) {
    if (i % arts_get_total_workers() == worker_id) {
      arts_edt_create(tester, 0, 0, NULL, 0);
}
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
