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
#include "arts/network/Server.h"

#include <unistd.h>

#include "arts/arts.h"
#include "arts/introspection/Metrics.h"
#include "arts/network/Remote.h"
#include "arts/network/RemoteProtocol.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/Runtime.h"
#include "arts/runtime/compute/EdtFunctions.h"
#include "arts/runtime/memory/DbFunctions.h"
#include "arts/runtime/network/RemoteFunctions.h"
#include "arts/runtime/sync/EventFunctions.h"
#include "arts/system/ArtsPrint.h"

#define EDT_MUG_SIZE 32

bool artsGlobalIWillPrint = false;
FILE *lockPrintFile;
extern bool serverEnd;

#ifdef SEQUENCENUMBERS
uint64_t *recSeqNumbers;
#endif

void artsRemoteTryToBecomePrinter() {
  lockPrintFile = fopen(".artsPrintLock", "wx");
  if (lockPrintFile)
    artsGlobalIWillPrint = true;
}

void artsRemoteTryToClosePrinter() {
  if (lockPrintFile)
    fclose(lockPrintFile);
  remove(".artsPrintLock");
}

void artsRemoteShutdown() { artsLLServerShutdown(); }

void artsServerSetup(struct artsConfig *config) {
  // ASYNC Message Deque Init
  artsLLServerSetup(config);
  outInit(artsGlobalRankCount * config->ports);
#ifdef SEQUENCENUMBERS
  recSeqNumbers = (uint64_t *)artsCalloc(artsGlobalRankCount, sizeof(uint64_t));
#endif
}

void artsServerProcessPacket(struct artsRemotePacket *packet) {
  if (packet->messageType != ARTS_REMOTE_METRIC_UPDATE_MSG &&
      packet->messageType != ARTS_REMOTE_SHUTDOWN_MSG) {
    artsMetricsTriggerEvent(artsNetworkRecieveBW, artsThread, packet->size);
    artsMetricsTriggerEvent(artsFreeBW + packet->messageType, artsThread,
                            packet->size);
    artsMetricsUpdatePacketInfo(packet->size);
  }
#ifdef SEQUENCENUMBERS
  uint64_t expSeqNumber =
      __sync_fetch_and_add(&recSeqNumbers[packet->seqRank], 1U);
  if (expSeqNumber != packet->seqNum) {
    ARTS_DEBUG(
        "MESSAGE RECIEVED OUT OF ORDER exp: %lu rec: %lu source: %u type: %d",
        expSeqNumber, packet->seqNum, packet->rank, packet->messageType);
  }
//    else
//        ARTS_INFO("Recv: %lu -> %lu = %lu", packet->seqRank, artsGlobalRankId,
//        packet->seqNum);
#endif

  switch (packet->messageType) {
  case ARTS_REMOTE_SHUTDOWN_MSG: {
    ARTS_DEBUG("Remote Shutdown Request");
    artsRuntimeStop();
    break;
  }
  case ARTS_REMOTE_EDT_SIGNAL_MSG: {
    struct artsRemoteEdtSignalPacket *pack =
        (struct artsRemoteEdtSignalPacket *)(packet);
    internalSignalEdtWithMode(pack->edt, pack->slot, pack->db, pack->mode,
                              pack->acquireMode, pack->useTwinDiff);
    break;
  }
  case ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG: {
    struct artsRemoteEventSatisfySlotPacket *pack =
        (struct artsRemoteEventSatisfySlotPacket *)(packet);
    artsEventSatisfySlot(pack->event, pack->db, pack->slot);
    break;
  }
  case ARTS_REMOTE_PERSISTENT_EVENT_SATISFY_SLOT_MSG: {
    struct artsRemotePersistentEventSatisfySlotPacket *pack =
        (struct artsRemotePersistentEventSatisfySlotPacket *)(packet);

    artsPersistentEventSatisfy(pack->event, pack->action, pack->lock);
    break;
  }
#ifdef USE_SMART_DB
  case ARTS_REMOTE_DB_INCREMENT_LATCH_MSG: {
    struct artsRemoteGuidOnlyPacket *pack =
        (struct artsRemoteGuidOnlyPacket *)(packet);
    artsDbIncrementLatch(pack->guid);
    break;
  }
  case ARTS_REMOTE_DB_DECREMENT_LATCH_MSG: {
    struct artsRemoteGuidOnlyPacket *pack =
        (struct artsRemoteGuidOnlyPacket *)(packet);
    artsDbDecrementLatch(pack->guid);
    break;
  }
  case ARTS_REMOTE_DB_ADD_DEPENDENCE_MSG: {
    struct artsRemoteDbAddDependencePacket *pack =
        (struct artsRemoteDbAddDependencePacket *)(packet);
    artsDbAddDependenceWithModeAndDiff(pack->dbSrc, pack->edtDest,
                                       pack->edtSlot, pack->acquireMode,
                                       pack->useTwinDiff);
    break;
  }
#endif
  case ARTS_REMOTE_DB_REQUEST_MSG: {
    struct artsRemoteDbRequestPacket *pack =
        (struct artsRemoteDbRequestPacket *)(packet);
    if (packet->size != sizeof(*pack)) {
      ARTS_INFO("Error dbpacket insanity");
    }
    artsRemoteDbSend(pack);
    break;
  }
  case ARTS_REMOTE_DB_SEND_MSG: {
    ARTS_DEBUG("Remote Db Received");
    struct artsRemoteDbSendPacket *pack =
        (struct artsRemoteDbSendPacket *)(packet);
    artsRemoteHandleDbReceived(pack);
    break;
  }
  case ARTS_REMOTE_ADD_DEPENDENCE_MSG: {
    ARTS_DEBUG("Dependence Received");
    struct artsRemoteAddDependencePacket *pack =
        (struct artsRemoteAddDependencePacket *)(packet);
    artsAddDependence(pack->source, pack->destination, pack->slot);
    break;
  }
  case ARTS_REMOTE_ADD_DEPENDENCE_TO_PERSISTENT_EVENT_MSG: {
    ARTS_DEBUG("Persistent Dependence Received");
    struct artsRemoteAddDependencePacket *pack =
        (struct artsRemoteAddDependencePacket *)(packet);
    artsAddDependenceToPersistentEventWithModeAndDiff(
        pack->source, pack->destination, pack->slot, pack->acquireMode,
        pack->useTwinDiff);
    break;
  }
  case ARTS_REMOTE_INVALIDATE_DB_MSG: {
    ARTS_DEBUG("DB Invalidate Received");
    artsRemoteHandleInvalidateDb(packet);
    break;
  }
  case ARTS_REMOTE_DB_FULL_REQUEST_MSG: {
    struct artsRemoteDbFullRequestPacket *pack =
        (struct artsRemoteDbFullRequestPacket *)(packet);
    artsRemoteDbFullSend(pack);
    break;
  }
  case ARTS_REMOTE_DB_FULL_SEND_MSG: {
    ARTS_DEBUG("DB Full Send Received");
    struct artsRemoteDbFullSendPacket *pack =
        (struct artsRemoteDbFullSendPacket *)(packet);
    artsRemoteHandleDbFullRecieved(pack);
    break;
  }
  case ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG: {
    ARTS_DEBUG("DB Full Send Already Local Received");
    artsRemoteHandleSendAlreadyLocal(packet);
    break;
  }
  case ARTS_REMOTE_DB_DESTROY_MSG: {
    ARTS_DEBUG("DB Destroy Received");
    artsRemoteHandleDbDestroy(packet);
    break;
  }
  case ARTS_REMOTE_DB_DESTROY_FORWARD_MSG: {
    ARTS_DEBUG("DB Destroy Forward Received");
    artsRemoteHandleDbDestroyForward(packet);
    break;
  }
  case ARTS_REMOTE_DB_CLEAN_FORWARD_MSG: {
    ARTS_DEBUG("DB Clean Forward Received");
    artsRemoteHandleDbCleanForward(packet);
    break;
  }
  case ARTS_REMOTE_DB_UPDATE_GUID_MSG: {
    ARTS_DEBUG("DB Guid Update Received");
    artsRemoteHandleUpdateDbGuid(packet);
    break;
  }
  case ARTS_REMOTE_EDT_MOVE_MSG: {
    ARTS_DEBUG("EDT Move Received");
    artsRemoteHandleEdtMove(packet);
    break;
  }
  case ARTS_REMOTE_DB_MOVE_MSG: {
    ARTS_DEBUG("DB Move Received");
    artsRemoteHandleDbMove(packet);
    break;
  }
  case ARTS_REMOTE_DB_UPDATE_MSG: {
    ARTS_DEBUG("DB Update Received");
    artsRemoteHandleUpdateDb(packet);
    break;
  }
  case ARTS_REMOTE_DB_PARTIAL_UPDATE_MSG: {
    ARTS_DEBUG("DB Partial Update Received");
    artsRemoteHandlePartialUpdate(packet);
    break;
  }
  case ARTS_REMOTE_EVENT_MOVE_MSG: {
    ARTS_DEBUG("Event Move Received");
    artsRemoteHandleEventMove(packet);
    break;
  }
  case ARTS_REMOTE_PERSISTENT_EVENT_MOVE_MSG: {
    ARTS_DEBUG("Persistent Event Move Received");
    artsRemoteHandlePersistentEventMove(packet);
    break;
  }
  case ARTS_REMOTE_METRIC_UPDATE_MSG: {

    struct artsRemoteMetricUpdate *pack =
        (struct artsRemoteMetricUpdate *)(packet);
    ARTS_DEBUG("Metric update Received %u -> %d %ld", artsGlobalRankId,
               pack->type, pack->toAdd);
    artsMetricsHandleRemoteUpdate(pack->type, artsSystem, pack->toAdd,
                                  pack->sub);
    break;
  }
  case ARTS_REMOTE_GET_FROM_DB_MSG: {
    ARTS_DEBUG("Get From DB Received");
    artsRemoteHandleGetFromDb(packet);
    break;
  }
  case ARTS_REMOTE_PUT_IN_DB_MSG: {
    ARTS_DEBUG("Put In DB Received");
    artsRemoteHandlePutInDb(packet);
    break;
  }
  case ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG: {
    ARTS_DEBUG("Signal EDT With Ptr Received");
    artsRemoteHandleSignalEdtWithPtr(packet);
    break;
  }
  case ARTS_REMOTE_SEND_MSG: {
    ARTS_DEBUG("Send Received");
    artsRemoteHandleSend(packet);
    break;
  }
  case ARTS_EPOCH_INIT_MSG: {
    ARTS_DEBUG("Epoch Init Received");
    artsRemoteHandleEpochInitSend(packet);
    break;
  }
  case ARTS_EPOCH_REQ_MSG: {
    ARTS_DEBUG("Epoch Req Received");
    artsRemoteHandleEpochReq(packet);
    break;
  }
  case ARTS_EPOCH_SEND_MSG: {
    ARTS_DEBUG("Epoch Send Received");
    artsRemoteHandleEpochSend(packet);
    break;
  }
  case ARTS_ATOMIC_ADD_ARRAYDB_MSG: {
    ARTS_DEBUG("Atomic Add ArrayDB Received");
    artsRemoteHandleAtomicAddInArrayDb(packet);
    break;
  }
  case ARTS_ATOMIC_CAS_ARRAYDB_MSG: {
    ARTS_DEBUG("Atomic Compare And Swap ArrayDB Received");
    artsRemoteHandleAtomicCompareAndSwapInArrayDb(packet);
    break;
  }
  case ARTS_EPOCH_INIT_POOL_MSG: {
    ARTS_DEBUG("Epoch Init Pool Received");
    artsRemoteHandleEpochInitPoolSend(packet);
    break;
  }
  case ARTS_EPOCH_DELETE_MSG: {
    ARTS_DEBUG("Epoch Delete Received");
    artsRemoteHandleEpochDelete(packet);
    break;
  }
  case ARTS_REMOTE_BUFFER_SEND_MSG: {
    ARTS_DEBUG("Buffer Send Received");
    artsRemoteHandleBufferSend(packet);
    break;
  }
  case ARTS_REMOTE_DB_MOVE_REQ_MSG: {
    ARTS_DEBUG("DB Move Request Received");
    artsDbMoveRequestHandle(packet);
    break;
  }
  case ARTS_REMOTE_DB_RENAME_MSG: {
    ARTS_DEBUG("DB Rename Received");
    artsRemoteHandleDbRename(packet);
    break;
  }
  default: {
    ARTS_INFO("Unknown Packet %d %d %d", packet->messageType, packet->size,
              packet->rank);
    artsShutdown();
    artsRuntimeStop();
  }
  }
}
