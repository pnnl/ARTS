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
#ifndef ARTSRUNTIME_H
#define ARTSRUNTIME_H
#ifdef __cplusplus
extern "C" {
#endif
#include "artsAbstractMachineModel.h"

#define NODEDEQUESIZE 8

enum artsInitType{ 
    artsWorkerThread, 
    artsReceiverThread, 
    artsRemoteStealThread,
    artsCounterThread,
    artsOtherThread
};

void artsRuntimeNodeInit(unsigned int workerThreads, unsigned int receivingThreads, unsigned int senderThreads, unsigned int receiverThreads, unsigned int totalThreads, bool remoteStealingOn, struct artsConfig * config);
void artsRuntimeGlobalCleanup();
void artsRuntimePrivateCleanup();
void artsRuntimeStop();
void artsHandleReadyEdt(struct artsEdt *edt);
void artsRehandleReadyEdt(struct artsEdt *edt);
void artsHandleRemoteStolenEdt(struct artsEdt *edt);
bool artsRuntimeSchedulerLoop();
void artsThreadZeroNodeStart();
void artsThreadZeroPrivateInit(struct threadMask * unit, struct artsConfig * config);
void artsRuntimePrivateInit(struct threadMask * unit, struct artsConfig * config);
int artsRuntimeLoop();
int artsRuntimeSchedulerLoopWait( volatile bool * waitForMe );
bool artsDefaultSchedulerLoop();

bool artsRuntimeEdtLockDb (artsGuid_t dbGuid, struct artsDb * db, void * edtPacket, bool shared);
void artsRuntimeEdtLockDbSignalNext (struct artsDb * db, artsGuid_t dbGuid, bool remote);
struct artsEdt * artsRuntimeStealFromWorker();
struct artsEdt * artsRuntimeStealFromNetwork();
void artsDbUnlock (struct artsDb * db, artsGuid_t dbGuid, bool write);
bool artsDbLockAllDbs( struct artsEdt * edt );
bool artsDbLock (artsGuid_t dbGuid, void * edtPacket, unsigned int rank, bool shared);

bool artsNetworkFirstSchedulerLoop();
bool artsNetworkBeforeStealSchedulerLoop();
bool artsDefaultSchedulerLoop();
bool artsGpuSchedulerLoop();
#ifdef __cplusplus
}
#endif

#endif
