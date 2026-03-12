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

#define __USE_GNU
#include <sched.h>

#include "arts/arts.h"
#include "arts/runtime/RT.h"

void test(uint32_t paramc, uint64_t *paramv, uint32_t depc,
          artsEdtDep_t depv[]) {
#ifdef __linux__
  PRINTF("Running edt %u on %u %u, %u\n", artsGetCurrentGuid(),
         artsGetCurrentNode(), artsGetCurrentWorker(), sched_getcpu());
#else
  PRINTF("Running edt %u on %u %u\n", artsGetCurrentGuid(),
         artsGetCurrentNode(), artsGetCurrentWorker());
#endif
}

void initPerNode(unsigned int nodeId, int argc, char **argv) {
  PRINTF("Node %u argc %u\n", nodeId, argc);
}

void exitProgram(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                 artsEdtDep_t depv[]) {
  PRINTF("Exit: %u\n", depv[0].guid);
  artsShutdown();
}

void artsMain(int argc, char **argv) {
  PRINTF("Main EDT %u\n", artsGetCurrentGuid());
  PRINTF("Starting\n");
  artsGuid_t exitGuid = artsReserveGuidRoute(ARTS_EDT, 0);
  artsEdtCreateWithGuid(exitProgram, exitGuid, 0, NULL, 1);
  artsGuid_t epochGuid = artsInitializeAndStartEpoch(exitGuid, 0);

  int numberOfWorkers = artsGetTotalWorkers();
  for (int i = 0; i < numberOfWorkers; i++) {
    uint64_t args[3];
    artsGuid_t guid = artsEdtCreateWithEpoch(test, 0, 3, args, 0, epochGuid);
  }
  for (int i = 0; i < 100000000; i++) {
    // Simulate some work
    if (i % 10000000 == 0) {
      printf("Thread %d is working on iteration %d\n", artsGetCurrentWorker(),
             i);
    }
  }
  artsWaitOnHandle(epochGuid);
}

int main(int argc, char **argv) {
  artsRT(argc, argv);
  return 0;
}
