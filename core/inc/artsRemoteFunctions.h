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
#ifndef ARTSEMOTEFUNCTIONS_H
#define ARTSEMOTEFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
#include "artsRemoteProtocol.h"

void artsRemoteAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot, artsType_t mode, unsigned int rank);
void artsRemoteUpdateRouteTable(artsGuid_t guid, unsigned int rank);
void artsRemoteHandleUpdateDbGuid(void * ptr);
void artsRemoteHandleInvalidateDb(void * ptr);
void artsRemoteDbDestroy(artsGuid_t guid, unsigned int originRank, bool clean);
void artsRemoteHandleDbDestroyForward(void * ptr);
void artsRemoteHandleDbCleanForward(void * ptr);
void artsRemoteHandleDbDestroy(void * ptr);
void artsRemoteUpdateDb(artsGuid_t guid, bool sendDb);
void artsRemoteHandleUpdateDb(void * ptr);
void artsRemoteMemoryMove(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType, void(*freeMethod)(void*));
void artsRemoteMemoryMoveNoFree(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType);
void artsRemoteHandleEdtMove(void * ptr);
void artsRemoteHandleDbMove(void * ptr);
void artsRemoteHandleEventMove(void * ptr);
void artsRemoteSignalEdt(artsGuid_t edt, artsGuid_t db, uint32_t slot, artsType_t mode);
void artsRemoteEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot);
void artsDbRequestCallback(struct artsEdt *edt, unsigned int slot, struct artsDb * dbRes);
bool artsRemoteDbRequest(artsGuid_t dataGuid, int rank, struct artsEdt * edt, int pos, artsType_t mode, bool aggRequest);
void artsRemoteDbForward(int destRank, int sourceRank, artsGuid_t dataGuid, artsType_t mode);
void artsRemoteDbSendNow(int rank, struct artsDb * db);
void artsRemoteDbSendCheck(int rank, struct artsDb * db, artsType_t mode);
void artsRemoteDbSend(struct artsRemoteDbRequestPacket * pack);
void artsRemoteHandleDbRecieved(struct artsRemoteDbSendPacket * packet);
void artsRemoteDbFullRequest(artsGuid_t dataGuid, int rank, struct artsEdt * edt, int pos, artsType_t mode);
void artsRemoteDbForwardFull(int destRank, int sourceRank, artsGuid_t dataGuid, struct artsEdt * edt, int pos, artsType_t mode);
void artsRemoteDbFullSendNow(int rank, struct artsDb * db, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsRemoteDbFullSendCheck(int rank, struct artsDb * db, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsRemoteDbFullSend(struct artsRemoteDbFullRequestPacket * pack);
void artsRemoteHandleDbFullRecieved(struct artsRemoteDbFullSendPacket * packet);
void artsRemoteSendAlreadyLocal(int rank, artsGuid_t guid, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsRemoteHandleSendAlreadyLocal(void * pack);
void artsRemoteGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);
void artsRemoteHandleGetFromDb(void * pack);
void artsRemotePutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid, unsigned int rank);
void artsRemoteHandlePutInDb(void * pack);
void artsRemoteSignalEdtWithPtr(artsGuid_t edtGuid, artsGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot);
void artsRemoteHandleSignalEdtWithPtr(void * pack);
void artsRemoteMetricUpdate(int rank, int type, int level, uint64_t timeStamp, uint64_t toAdd, bool sub);
void artsRemoteHandleSend(void * pack);
void artsRemoteEpochInitSend(unsigned int rank, artsGuid_t epochGuid, artsGuid_t edtGuid, unsigned int slot);
void artsRemoteHandleEpochInitSend(void * pack);
void artsRemoteEpochInitPoolSend(unsigned int rank, unsigned int poolSize, artsGuid_t startGuid, artsGuid_t poolGuid);
void artsRemoteHandleEpochInitPoolSend(void * pack);
void artsRemoteEpochReq(unsigned int rank, artsGuid_t guid);
void artsRemoteHandleEpochReq(void * pack);
void artsRemoteEpochSend(unsigned int rank, artsGuid_t guid, unsigned int active, unsigned int finish);
void artsRemoteHandleEpochSend(void * pack);
void artsRemoteAtomicAddInArrayDb(unsigned int rank, artsGuid_t dbGuid, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsRemoteHandleAtomicAddInArrayDb(void * pack);
void artsRemoteAtomicCompareAndSwapInArrayDb(unsigned int rank, artsGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsRemoteHandleAtomicCompareAndSwapInArrayDb(void * pack);
void artsRemoteEpochDelete(unsigned int rank, artsGuid_t epochGuid);
void artsRemoteHandleEpochDelete(void * pack);
void artsDbMoveRequest(artsGuid_t dbGuid, unsigned int destRank);
void artsDbMoveRequestHandle(void * pack);
void artsRemoteHandleBufferSend(void * pack);
void artsRemoteHandleDbDestroy(void * ptr);
void artsRemoteSignalContext(unsigned int rank, uint64_t ticket);
void artsRemoteHandleSignalContext(void * pack);
#ifdef __cplusplus
}
#endif

#endif
