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
#include "arts/arts.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "arts/runtime/Globals.h"
#include "arts/runtime/Runtime.h"

extern __thread struct artsEdt *currentEdt;
extern unsigned int numNumaDomains;

artsGuid_t artsGetCurrentGuid() {
  if (currentEdt) {
    return currentEdt->currentEdt;
  }
  return NULL_GUID;
}

unsigned int artsGetCurrentNode() { return artsGlobalRankId; }

unsigned int artsGetTotalNodes() { return artsGlobalRankCount; }

unsigned int artsGetTotalWorkers() { return artsNodeInfo.workerThreadCount; }

unsigned int artsGetCurrentWorker() { return artsThreadInfo.groupId; }

unsigned int artsGetCurrentCluster() { return artsThreadInfo.clusterId; }

unsigned int artsGetTotalClusters() { return numNumaDomains; }

void artsStopLocalWorker() { artsThreadInfo.alive = false; }

void artsStopLocalNode() { artsRuntimeStop(); }

uint64_t artsThreadSafeRandom() {
  long int temp = jrand48(artsThreadInfo.drand_buf);
  return (uint64_t)temp;
}

unsigned int artsGetTotalGpus() { return artsNodeInfo.gpu; }
