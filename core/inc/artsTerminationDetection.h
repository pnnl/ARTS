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

#ifndef ARTS_TERMINATION_DETECTION_H
#define  ARTS_TERMINATION_DETECTION_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"

artsEpoch_t * createEpoch(artsGuid_t * guid, artsGuid_t edtGuid, unsigned int slot);
void incrementQueueEpoch(artsGuid_t epochGuid);
void incrementActiveEpoch(artsGuid_t epochGuid);
void incrementFinishedEpoch(artsGuid_t epochGuid);
void sendEpoch(artsGuid_t epochGuid, unsigned int source, unsigned int dest);
void broadcastEpochRequest(artsGuid_t epochGuid);
bool checkEpoch(artsEpoch_t * epoch, unsigned int totalActive, unsigned int totalFinish);
void reduceEpoch(artsGuid_t epochGuid, unsigned int active, unsigned int finish);
void deleteEpoch(artsGuid_t epochGuid, artsEpoch_t * epoch);

typedef struct artsEpochPool {
    struct artsEpochPool * next;
    unsigned int size;
    unsigned int index;
    volatile unsigned int outstanding;
    artsEpoch_t pool[];
} artsEpochPool_t;

artsEpochPool_t * createEpochPool(artsGuid_t * epochPoolGuid, unsigned int poolSize, artsGuid_t * startGuid);
artsEpoch_t * getPoolEpoch(artsGuid_t edtGuid, unsigned int slot);

void globalShutdownGuidIncActive();
void globalShutdownGuidIncQueue();
void globalShutdownGuidIncFinished();
bool createShutdownEpoch();

#ifdef __cplusplus
}
#endif
#endif
