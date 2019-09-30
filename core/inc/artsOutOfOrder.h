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
#ifndef ARTSOUTOFORDER_H
#define ARTSOUTOFORDER_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "artsOutOfOrderList.h"
void artsOutOfOrderSignalEdt ( artsGuid_t waitOn, artsGuid_t edtPacket, artsGuid_t dataGuid, uint32_t slot, artsType_t mode, bool force);
void artsOutOfOrderEventSatisfy( artsGuid_t waitOn, artsGuid_t eventGuid, artsGuid_t dataGuid );
void artsOutOfOrderEventSatisfySlot( artsGuid_t waitOn, artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot, bool force);
void artsOutOfOrderAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot, artsType_t mode, artsGuid_t waitOn);
void artsOutOfOrderHandleReadyEdt(artsGuid_t triggerGuid, struct artsEdt *edt);
void artsOutOfOrderHandleRemoteDbSend(int rank, artsGuid_t dbGuid, artsType_t mode);
void artsOutOfOrderHandleDbRequestWithOOList(struct artsOutOfOrderList * addToMe, void ** data, struct artsEdt *edt, unsigned int slot);
void artsOutOfOrderHandleDbRequest(artsGuid_t dbGuid, struct artsEdt *edt, unsigned int slot, bool inc);
void artsOutOfOrderHandleRemoteDbExclusiveRequest(artsGuid_t dbGuid, int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsOutOfOrderHandleRemoteDbFullSend(artsGuid_t dbGuid, int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsOutOfOrderGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void artsOutOfOrderSignalEdtWithPtr(artsGuid_t edtGuid, artsGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot);
void artsOutOfOrderPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epcohGuid);
void artsOutOfOrderIncActiveEpoch(artsGuid_t epochGuid);
void artsOutOfOrderIncFinishedEpoch(artsGuid_t epochGuid);
void artsOutOfOrderSendEpoch(artsGuid_t epochGuid, unsigned int source, unsigned int dest);
void artsOutOfOrderIncQueueEpoch(artsGuid_t epochGuid);
void artsOutOfOrderAtomicAddInArrayDb(artsGuid_t dbGuid,  unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsOutOfOrderAtomicCompareAndSwapInArrayDb(artsGuid_t dbGuid,  unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsOutOfOrderDbMove(artsGuid_t dataGuid, unsigned int rank);

void artsOutOfOrderHandler( void * handleMe, void * memoryPtr );

#ifdef __cplusplus
}
#endif

#endif
