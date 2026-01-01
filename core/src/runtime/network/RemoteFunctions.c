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
#include "arts/runtime/network/RemoteFunctions.h"

#include "arts/arts.h"
#include "arts/gas/OutOfOrder.h"
#include "arts/gas/RouteTable.h"
#include "arts/introspection/Metrics.h"
#include "arts/network/RemoteProtocol.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/Runtime.h"
#include "arts/runtime/compute/EdtFunctions.h"
#include "arts/runtime/memory/ArrayDb.h"
#include "arts/runtime/memory/DbFunctions.h"
#include "arts/runtime/memory/DbList.h"
#include "arts/runtime/memory/TwinDiff.h"
#include "arts/runtime/sync/TerminationDetection.h"
#include "arts/system/ArtsPrint.h"
#include "arts/utils/Atomics.h"

static inline void artsFillPacketHeader(struct artsRemotePacket *header,
                                        unsigned int size,
                                        unsigned int messageType) {
  header->size = size;
  header->messageType = messageType;
  header->rank = artsGlobalRankId;
}

static void sendRemoteAddDependencePacket(unsigned int messageType,
                                          artsGuid_t source,
                                          artsGuid_t destination, uint32_t slot,
                                          unsigned int rank,
                                          artsType_t acquireMode,
                                          bool useTwinDiff) {
  struct artsRemoteAddDependencePacket packet;
  packet.source = source;
  packet.destination = destination;
  packet.slot = slot;
  packet.acquireMode = acquireMode;
  packet.useTwinDiff = useTwinDiff ? 1 : 0;
  memset(packet.reserved, 0, sizeof(packet.reserved));
  artsFillPacketHeader(&packet.header, sizeof(packet), messageType);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteAddDependence(artsGuid_t source, artsGuid_t destination,
                             uint32_t slot, unsigned int rank) {
  ARTS_DEBUG("Remote Add dependence sent %d", rank);
  sendRemoteAddDependencePacket(ARTS_REMOTE_ADD_DEPENDENCE_MSG, source,
                                destination, slot, rank, ARTS_NULL, true);
}

void artsRemoteAddDependenceWithHints(artsGuid_t source, artsGuid_t destination,
                                      uint32_t slot, unsigned int rank,
                                      artsType_t acquireMode,
                                      bool useTwinDiff) {
  ARTS_DEBUG("Remote Add dependence (acquire=%u, twin=%d) sent %d", acquireMode,
             useTwinDiff, rank);
  sendRemoteAddDependencePacket(ARTS_REMOTE_ADD_DEPENDENCE_MSG, source,
                                destination, slot, rank, acquireMode,
                                useTwinDiff);
}

void artsRemoteAddDependenceToPersistentEvent(artsGuid_t source,
                                              artsGuid_t destination,
                                              uint32_t slot,
                                              unsigned int rank) {
  ARTS_DEBUG("Remote Add dependence to persistent event sent %d", rank);
  sendRemoteAddDependencePacket(
      ARTS_REMOTE_ADD_DEPENDENCE_TO_PERSISTENT_EVENT_MSG, source, destination,
      slot, rank, ARTS_NULL, true);
}

void artsRemoteAddDependenceToPersistentEventWithHints(
    artsGuid_t source, artsGuid_t destination, uint32_t slot, unsigned int rank,
    artsType_t acquireMode, bool useTwinDiff) {
  ARTS_DEBUG(
      "Remote Add dependence to persistent event (acquire=%u, twin=%d) sent %d",
      acquireMode, useTwinDiff, rank);
  sendRemoteAddDependencePacket(
      ARTS_REMOTE_ADD_DEPENDENCE_TO_PERSISTENT_EVENT_MSG, source, destination,
      slot, rank, acquireMode, useTwinDiff);
}

void artsRemoteUpdateRouteTable(artsGuid_t guid, unsigned int rank) {
  unsigned int owner = artsGuidGetRank(guid);
  if (owner == artsGlobalRankId) {
    struct artsDbFrontierIterator *iter =
        artsRouteTableGetRankDuplicates(guid, rank);
    if (iter) {
      unsigned int node;
      while (artsDbFrontierIterNext(iter, &node)) {
        if (node != artsGlobalRankId && node != rank) {
          struct artsRemoteGuidOnlyPacket outPacket;
          outPacket.guid = guid;
          artsFillPacketHeader(&outPacket.header, sizeof(outPacket),
                               ARTS_REMOTE_INVALIDATE_DB_MSG);
          artsRemoteSendRequestAsync(node, (char *)&outPacket,
                                     sizeof(outPacket));
        }
      }
      artsFree(iter);
    }
  } else {
    struct artsRemoteGuidOnlyPacket packet;
    artsFillPacketHeader(&packet.header, sizeof(packet),
                         ARTS_REMOTE_DB_UPDATE_GUID_MSG);
    packet.guid = guid;
    artsRemoteSendRequestAsync(owner, (char *)&packet, sizeof(packet));
  }
}

void artsRemoteHandleUpdateDbGuid(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  ARTS_DEBUG("Updated %ld to %d", packet->guid, packet->header.rank);
  artsRemoteUpdateRouteTable(packet->guid, packet->header.rank);
}

void artsRemoteHandleInvalidateDb(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  void *address = artsRouteTableLookupItem(packet->guid);
  artsRouteTableInvalidateItem(packet->guid);
}

// TODO: Fix this...
void artsRemoteDbDestroy(artsGuid_t guid, unsigned int originRank, bool clean) {
  //    unsigned int rank = artsGuidGetRank(guid);
  //    //ARTS_INFO("Destroy Check");
  //    if(rank == artsGlobalRankId)
  //    {
  //        struct artsRouteInvalidate * table =
  //        artsRouteTableGetRankDuplicates(guid); struct artsRouteInvalidate *
  //        next = table; struct artsRouteInvalidate * current;
  //
  //        if(next != NULL && next->used != 0)
  //        {
  //            struct artsRemoteGuidOnlyPacket outPacket;
  //            outPacket.guid = guid;
  //            artsFillPacketHeader(&outPacket.header, sizeof(outPacket),
  //            ARTS_REMOTE_DB_DESTROY_MSG);
  //
  //            int lastSend=-1;
  //            while( next != NULL)
  //            {
  //                for(int i=0; i < next->used; i++ )
  //                {
  //                    if(originRank != next->data[i] && next->data[i] !=
  //                    lastSend)
  //                    {
  ////                        ARTS_INFO("Destroy Send 1");
  //                        lastSend = next->data[i];
  //                        artsRemoteSendRequestAsync(next->data[i], (char
  //                        *)&outPacket, sizeof(outPacket));
  //                    }
  //                }
  //                next->used = 0;
  //                //current=next;
  //                next = next->next;
  //                //artsFree(current);
  //            }
  //        }
  //        if(originRank != artsGlobalRankId && !clean)
  //        {
  ////            ARTS_INFO("Origin Destroy");
  ////            artsDebugPrintStack();
  //            void * address = artsRouteTableLookupItem(guid);
  //            artsFree(address);
  //            artsRouteTableRemoveItem(guid);
  //        }
  //        //if( originRank != artsGlobalRankId )
  //        //    artsDbDestroy(guid);
  //    }
  //    else
  //    {
  //        //void * dbAddress = artsRouteTableLookupItem(  guid );
  //        //ARTS_DEBUG("depv %ld %p %p", guid, dbAddress, callBack);
  //        struct artsRemoteGuidOnlyPacket packet;
  //        if(!clean)
  //            artsFillPacketHeader(&packet.header, sizeof(packet),
  //            ARTS_REMOTE_DB_DESTROY_FORWARD_MSG);
  //        else
  //            artsFillPacketHeader(&packet.header, sizeof(packet),
  //            ARTS_REMOTE_DB_CLEAN_FORWARD_MSG);
  //        packet.guid = guid;
  ////        ARTS_INFO("Destroy Send 2");
  ////        artsDebugPrintStack();
  //        artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
  //    }
}

void artsRemoteHandleDbDestroyForward(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  artsRemoteDbDestroy(packet->guid, packet->header.rank, 0);
  artsDbDestroySafe(packet->guid, false);
}

void artsRemoteHandleDbCleanForward(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  artsRemoteDbDestroy(packet->guid, packet->header.rank, 1);
}

void artsRemoteHandleDbDestroy(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  artsDbDestroySafe(packet->guid, false);
}

void artsRemoteUpdateDb(artsGuid_t guid, bool sendDb) {
  unsigned int rank = artsGuidGetRank(guid);
  if (rank != artsGlobalRankId) {
    struct artsRemoteGuidOnlyPacket packet;
    packet.guid = guid;
    struct artsDb *db = NULL;
    if (sendDb && (db = (struct artsDb *)artsRouteTableLookupItem(guid))) {
      int size = sizeof(struct artsRemoteGuidOnlyPacket) + db->header.size;
      artsFillPacketHeader(&packet.header, size, ARTS_REMOTE_DB_UPDATE_MSG);
      artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet),
                                        (char *)db, db->header.size);
    } else {
      artsFillPacketHeader(&packet.header,
                           sizeof(struct artsRemoteGuidOnlyPacket),
                           ARTS_REMOTE_DB_UPDATE_MSG);
      artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
    }
  }
}

void artsRemoteHandleUpdateDb(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  struct artsDb *packetDb = (struct artsDb *)(packet + 1);
  unsigned int rank = artsGuidGetRank(packet->guid);
  if (rank == artsGlobalRankId) {
    struct artsDb **dataPtr;
    bool write = packet->header.size > sizeof(struct artsRemoteGuidOnlyPacket);
    itemState_t state = artsRouteTableLookupItemWithState(
        packet->guid, (void ***)&dataPtr, allocatedKey, write);
    struct artsDb *db = (dataPtr) ? *dataPtr : NULL;
    if (write) {
      void *ptr = (void *)(db + 1);
      memcpy(ptr, packetDb + 1, db->header.size - sizeof(struct artsDb));
      artsRouteTableSetRank(packet->guid, artsGlobalRankId);
      artsProgressFrontier(db, artsGlobalRankId);
    } else {
      artsProgressFrontier(db, packet->header.rank);
    }
    artsDbDecrementLatch(packet->guid);
  }
}

void artsRemotePartialUpdateDb(artsGuid_t guid, struct artsDiffList *diffs,
                               void *working) {
  unsigned int rank = artsGuidGetRank(guid);
  if (rank == artsGlobalRankId) {
    ARTS_DEBUG("artsRemotePartialUpdateDb: owner is local, no send needed");
    return;
  }

  if (!diffs || diffs->regionCount == 0) {
    ARTS_DEBUG("artsRemotePartialUpdateDb: no diffs to send");
    return;
  }

  // Calculate packet size
  size_t headerSize = sizeof(struct artsRemotePartialUpdatePacket);
  size_t regionDescSize =
      diffs->regionCount * sizeof(struct artsRemoteDiffRegion);
  size_t totalSize = headerSize + regionDescSize + diffs->totalBytes;

  // Allocate packet buffer
  char *buffer = (char *)malloc(totalSize);
  if (!buffer) {
    ARTS_INFO("Failed to allocate %zu bytes for partial update packet",
              totalSize);
    return;
  }

  // Fill packet header
  struct artsRemotePartialUpdatePacket *packet =
      (struct artsRemotePartialUpdatePacket *)buffer;
  artsFillPacketHeader(&packet->header, totalSize,
                       ARTS_REMOTE_DB_PARTIAL_UPDATE_MSG);
  packet->guid = guid;
  packet->regionCount = diffs->regionCount;
  packet->dataBytes = diffs->totalBytes;
  packet->flags = 0;
  packet->reserved = 0;

  // Fill region descriptors and data
  struct artsRemoteDiffRegion *regions =
      (struct artsRemoteDiffRegion *)(packet + 1);
  uint8_t *dataPtr = (uint8_t *)(regions + diffs->regionCount);
  uint8_t *workData = (uint8_t *)working;

  struct artsDiffRegion *region = diffs->head;
  uint32_t idx = 0;
  while (region) {
    regions[idx].offset = region->offset;
    regions[idx].length = region->length;

    // Copy data for this region
    memcpy(dataPtr, workData + region->offset, region->length);
    dataPtr += region->length;

    region = region->next;
    idx++;
  }

  ARTS_INFO("Sending partial update to rank %u: %u regions, %u bytes (%.2f%%)",
            rank, diffs->regionCount, diffs->totalBytes,
            diffs->dirtyRatio * 100.0);

  // Send packet
  artsRemoteSendRequestAsync(rank, buffer, totalSize);

  // Free buffer
  free(buffer);
}

void artsRemoteHandlePartialUpdate(void *ptr) {
  struct artsRemotePartialUpdatePacket *packet =
      (struct artsRemotePartialUpdatePacket *)ptr;

  ARTS_DEBUG("Rank %u handling partial update for DB[Guid:%lu] from rank %u",
             artsGlobalRankId, packet->guid, packet->header.rank);

  unsigned int rank = artsGuidGetRank(packet->guid);
  if (rank != artsGlobalRankId) {
    ARTS_INFO("Received partial update for non-owned DB[Guid:%lu]",
              packet->guid);
    return;
  }

  // Lookup the DB
  struct artsDb *db = (struct artsDb *)artsRouteTableLookupItem(packet->guid);
  if (!db) {
    ARTS_INFO("Partial update for unknown DB[Guid:%lu]", packet->guid);
    return;
  }

  void *dbData = (void *)(db + 1);

  // Parse and apply diff regions
  struct artsRemoteDiffRegion *regions =
      (struct artsRemoteDiffRegion *)(packet + 1);
  uint8_t *diffData = (uint8_t *)(regions + packet->regionCount);

  ARTS_INFO("Applying partial update: %u regions, %u bytes to DB[Guid:%lu]",
            packet->regionCount, packet->dataBytes, packet->guid);

  for (uint32_t i = 0; i < packet->regionCount; i++) {
    ARTS_DEBUG("  Region %u: offset=%u, length=%u", i, regions[i].offset,
               regions[i].length);
    memcpy((uint8_t *)dbData + regions[i].offset, diffData, regions[i].length);
    diffData += regions[i].length;
  }

  // Update route table and progress frontier
  artsRouteTableSetRank(packet->guid, artsGlobalRankId);
  artsProgressFrontier(db, artsGlobalRankId);

  artsDbDecrementLatch(packet->guid);

  ARTS_DEBUG("Partial update applied successfully for DB[Guid:%lu]",
             packet->guid);
}

void artsRemoteMemoryMove(unsigned int route, artsGuid_t guid, void *ptr,
                          unsigned int memSize, unsigned messageType,
                          void (*freeMethod)(void *)) {
  REMOTE_MEMORY_MOVE_START();
  struct artsRemoteGuidOnlyPacket packet;
  artsFillPacketHeader(&packet.header, sizeof(packet) + memSize, messageType);
  packet.guid = guid;
  artsRemoteSendRequestPayloadAsyncFree(route, (char *)&packet, sizeof(packet),
                                        (char *)ptr, 0, memSize, freeMethod);
  artsRouteTableRemoveItem(guid);
  REMOTE_MEMORY_MOVE_STOP();
}

void artsRemoteMemoryMoveNoFree(unsigned int route, artsGuid_t guid, void *ptr,
                                unsigned int memSize, unsigned messageType) {
  struct artsRemoteGuidOnlyPacket packet;
  artsFillPacketHeader(&packet.header, sizeof(packet) + memSize, messageType);
  packet.guid = guid;
  artsRemoteSendRequestPayloadAsync(route, (char *)&packet, sizeof(packet),
                                    (char *)ptr, memSize);
}

void artsRemoteHandleEdtMove(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  unsigned int size =
      packet->header.size - sizeof(struct artsRemoteGuidOnlyPacket);
  struct artsEdt *edt =
      (struct artsEdt *)artsMallocAlignWithType(size, 16, artsEdtMemorySize);

  memcpy(edt, packet + 1, size);
  artsRouteTableAddItemRace(edt, (artsGuid_t)packet->guid, artsGlobalRankId,
                            false);
  ARTS_INFO("EDT[Guid:%lu] Moved to Rank: %d", packet->guid, artsGlobalRankId);
  if (edt->depcNeeded == 0)
    artsHandleReadyEdt(edt);
  else
    artsRouteTableFireOO(packet->guid, artsOutOfOrderHandler);
}

void artsRemoteHandleDbMove(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  unsigned int size =
      packet->header.size - sizeof(struct artsRemoteGuidOnlyPacket);

  struct artsDb *dbHeader = (struct artsDb *)(packet + 1);
  unsigned int dbSize = dbHeader->header.size;

  struct artsHeader *memPacket = (struct artsHeader *)artsMallocAlignWithType(
      dbSize, 16, artsDbMemorySize);

  if (size == dbSize)
    memcpy(memPacket, packet + 1, size);
  else {
    memPacket->type = (unsigned int)artsGuidGetType(packet->guid);
    memPacket->size = dbSize;
  }
  // We need a local pointer for this node
  if (dbHeader->dbList) {
    struct artsDb *newDb = (struct artsDb *)memPacket;
    newDb->dbList = artsNewDbList();
  }

  ARTS_INFO("DB[Guid:%lu] Moved to Rank: %d", packet->guid, artsGlobalRankId);
  if (artsRouteTableAddItemRace(memPacket, (artsGuid_t)packet->guid,
                                artsGlobalRankId, false))
    artsRouteTableFireOO(packet->guid, artsOutOfOrderHandler);
}

void artsRemoteHandleEventMove(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  unsigned int size =
      packet->header.size - sizeof(struct artsRemoteGuidOnlyPacket);

  struct artsHeader *memPacket = (struct artsHeader *)artsMallocAlignWithType(
      size, 16, artsEventMemorySize);

  memcpy(memPacket, packet + 1, size);
  artsRouteTableAddItemRace(memPacket, (artsGuid_t)packet->guid,
                            artsGlobalRankId, false);
  artsRouteTableFireOO(packet->guid, artsOutOfOrderHandler);
}

void artsRemoteHandlePersistentEventMove(void *ptr) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)ptr;
  unsigned int size =
      packet->header.size - sizeof(struct artsRemoteGuidOnlyPacket);

  struct artsHeader *memPacket = (struct artsHeader *)artsMallocAlignWithType(
      size, 16, artsPersistentEventMemorySize);

  memcpy(memPacket, packet + 1, size);
  ARTS_INFO("Persistent Event [Guid:%lu] Moved to Rank: %d", packet->guid,
            artsGlobalRankId);
  artsRouteTableAddItemRace(memPacket, (artsGuid_t)packet->guid,
                            artsGlobalRankId, false);
  artsRouteTableFireOO(packet->guid, artsOutOfOrderHandler);
}

static void sendRemoteEdtSignalPacket(artsGuid_t edt, artsGuid_t db,
                                      uint32_t slot, artsType_t mode,
                                      artsType_t acquireMode,
                                      bool useTwinDiff) {
  struct artsRemoteEdtSignalPacket packet;
  unsigned int rank = artsGuidGetRank(edt);

  if (rank == artsGlobalRankId)
    rank = artsRouteTableLookupRank(edt);

  packet.db = db;
  packet.edt = edt;
  packet.slot = slot;
  packet.mode = mode;
  packet.dbRoute = artsGuidGetRank(db);
  packet.acquireMode = acquireMode;
  packet.useTwinDiff = useTwinDiff ? 1 : 0;
  memset(packet.signalReserved, 0, sizeof(packet.signalReserved));
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_EDT_SIGNAL_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteSignalEdt(artsGuid_t edt, artsGuid_t db, uint32_t slot,
                         artsType_t mode) {
  ARTS_INFO("Remote Signal from DB[Guid:%lu] to EDT[Guid:%lu, Slot:%d, Rank: "
            "%d]",
            db, edt, slot, artsGuidGetRank(edt));
  sendRemoteEdtSignalPacket(edt, db, slot, mode, ARTS_NULL, true);
}

void artsRemoteSignalEdtWithHints(artsGuid_t edt, artsGuid_t db, uint32_t slot,
                                  artsType_t mode, artsType_t acquireMode,
                                  bool useTwinDiff) {
  ARTS_INFO("Remote Signal from DB[Guid:%lu] to EDT[Guid:%lu, Slot:%d, "
            "Rank: %d, AcquireMode:%u, Twin=%d]",
            db, edt, slot, artsGuidGetRank(edt), acquireMode, useTwinDiff);
  sendRemoteEdtSignalPacket(edt, db, slot, mode, acquireMode, useTwinDiff);
}

void artsRemoteEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid,
                                uint32_t slot) {
  struct artsRemoteEventSatisfySlotPacket packet;
  packet.event = eventGuid;
  packet.db = dataGuid;
  packet.slot = slot;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG);
  artsRemoteSendRequestAsync(artsGuidGetRank(eventGuid), (char *)&packet,
                             sizeof(packet));
}

void artsRemotePersistentEventSatisfySlot(artsGuid_t eventGuid, uint32_t action,
                                          bool lock) {
  struct artsRemotePersistentEventSatisfySlotPacket packet;
  packet.event = eventGuid;
  packet.action = action;
  packet.lock = lock;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_PERSISTENT_EVENT_SATISFY_SLOT_MSG);
  artsRemoteSendRequestAsync(artsGuidGetRank(eventGuid), (char *)&packet,
                             sizeof(packet));
}

static void sendRemoteDbAddDependencePacket(artsGuid_t dbSrc,
                                            artsGuid_t edtDest,
                                            uint32_t edtSlot,
                                            artsType_t acquireMode,
                                            bool useTwinDiff) {
  struct artsRemoteDbAddDependencePacket packet;
  packet.dbSrc = dbSrc;
  packet.edtDest = edtDest;
  packet.edtSlot = edtSlot;
  packet.acquireMode = acquireMode;
  packet.useTwinDiff = useTwinDiff ? 1 : 0;
  memset(packet.reserved, 0, sizeof(packet.reserved));
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_DB_ADD_DEPENDENCE_MSG);
  artsRemoteSendRequestAsync(artsGuidGetRank(dbSrc), (char *)&packet,
                             sizeof(packet));
}

void artsRemoteDbAddDependence(artsGuid_t dbSrc, artsGuid_t edtDest,
                               uint32_t edtSlot) {
  sendRemoteDbAddDependencePacket(dbSrc, edtDest, edtSlot, ARTS_NULL, true);
}

void artsRemoteDbAddDependenceWithHints(artsGuid_t dbSrc, artsGuid_t edtDest,
                                        uint32_t edtSlot,
                                        artsType_t acquireMode,
                                        bool useTwinDiff) {
  sendRemoteDbAddDependencePacket(dbSrc, edtDest, edtSlot, acquireMode,
                                  useTwinDiff);
}

void artsRemoteDbAddDependenceWithByteOffset(artsGuid_t dbSrc,
                                             artsGuid_t edtDest,
                                             uint32_t edtSlot,
                                             artsType_t acquireMode,
                                             bool useTwinDiff,
                                             uint64_t byteOffset,
                                             uint64_t size) {
  struct artsRemoteDbAddDependenceWithByteOffsetPacket packet;
  packet.dbSrc = dbSrc;
  packet.edtDest = edtDest;
  packet.edtSlot = edtSlot;
  packet.acquireMode = acquireMode;
  packet.useTwinDiff = useTwinDiff ? 1 : 0;
  memset(packet.reserved, 0, sizeof(packet.reserved));
  packet.byteOffset = byteOffset;
  packet.size = size;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_DB_ADD_DEPENDENCE_WITH_BYTE_OFFSET_MSG);
  artsRemoteSendRequestAsync(artsGuidGetRank(dbSrc), (char *)&packet,
                             sizeof(packet));
}

void artsRemoteHandleDbAddDependenceWithByteOffset(void *ptr) {
  struct artsRemoteDbAddDependenceWithByteOffsetPacket *packet =
      (struct artsRemoteDbAddDependenceWithByteOffsetPacket *)ptr;

  /// Look up the local DB
  struct artsDb *dbRes =
      (struct artsDb *)artsRouteTableLookupItem(packet->dbSrc);
  if (dbRes != NULL) {
    /// DB is local - add dependency to its persistent event with byte offset
    artsAddDependenceToPersistentEventWithByteOffset(
        dbRes->eventGuid, packet->edtDest, packet->edtSlot, packet->acquireMode,
        packet->useTwinDiff != 0, packet->byteOffset, packet->size);
  } else {
    /// DB not found locally - this shouldn't happen as we routed to the owner
    ARTS_DEBUG("ESD: Remote byte-offset dep: DB %lu not found on node %u",
               packet->dbSrc, artsGlobalRankId);
  }
}

void artsRemoteDbIncrementLatch(artsGuid_t db) {
  struct artsRemoteGuidOnlyPacket packet;
  packet.guid = db;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_DB_INCREMENT_LATCH_MSG);
  artsRemoteSendRequestAsync(artsGuidGetRank(db), (char *)&packet,
                             sizeof(packet));
}

void artsRemoteDbDecrementLatch(artsGuid_t db) {
  struct artsRemoteGuidOnlyPacket packet;
  packet.guid = db;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_DB_DECREMENT_LATCH_MSG);
  artsRemoteSendRequestAsync(artsGuidGetRank(db), (char *)&packet,
                             sizeof(packet));
}

void artsDbRequestCallback(struct artsEdt *edt, unsigned int slot,
                           struct artsDb *dbRes) {
  artsEdtDep_t *depv = (artsEdtDep_t *)artsGetDepv(edt);
  depv[slot].ptr = dbRes + 1;
  unsigned int temp = artsAtomicSub(&edt->depcNeeded, 1U);
  if (temp == 0)
    artsHandleRemoteStolenEdt(edt);
}

bool artsRemoteDbRequest(artsGuid_t dataGuid, int rank, struct artsEdt *edt,
                         int pos, artsType_t mode, bool aggRequest,
                         artsType_t acquireMode, bool useTwinDiff) {
  if (artsRouteTableAddSent(dataGuid, edt, pos, aggRequest)) {
    struct artsRemoteDbRequestPacket packet;
    packet.dbGuid = dataGuid;
    packet.mode = mode;
    packet.acquireMode = acquireMode;
    packet.useTwinDiff = useTwinDiff ? 1 : 0;
    packet.reserved[0] = 0;
    packet.reserved[1] = 0;
    packet.reserved[2] = 0;
    artsFillPacketHeader(&packet.header, sizeof(packet),
                         ARTS_REMOTE_DB_REQUEST_MSG);
    ARTS_DEBUG(
        "Rank %u requesting DB[Guid:%lu] from rank %d (slot=%d, twin=%d)",
        artsGlobalRankId, dataGuid, rank, pos, packet.useTwinDiff);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
    return true;
  }
  return false;
}

void artsRemoteDbForward(int destRank, int sourceRank, artsGuid_t dataGuid,
                         artsType_t mode) {
  struct artsRemoteDbRequestPacket packet;
  packet.header.size = sizeof(packet);
  packet.header.messageType = ARTS_REMOTE_DB_REQUEST_MSG;
  packet.header.rank = destRank;
  packet.dbGuid = dataGuid;
  packet.mode = mode;
  artsRemoteSendRequestAsync(sourceRank, (char *)&packet, sizeof(packet));
}

void artsRemoteDbSendNow(int rank, struct artsDb *db) {
  struct artsRemoteDbSendPacket packet;
  int size = sizeof(struct artsRemoteDbSendPacket) + db->header.size;
  artsFillPacketHeader(&packet.header, size, ARTS_REMOTE_DB_SEND_MSG);
  artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet),
                                    (char *)db, db->header.size);
}

void artsRemoteDbSendCheck(int rank, struct artsDb *db, artsType_t mode) {
  if (!artsIsGuidLocal(db->guid)) {
    artsRouteTableReturnDb(db->guid, false);
    artsRemoteDbSendNow(rank, db);
  } else if (artsAddDbDuplicate(db, rank, NULL, 0, mode)) {
    artsRemoteDbSendNow(rank, db);
  }
}

void artsRemoteDbSend(struct artsRemoteDbRequestPacket *pack) {
  unsigned int redirected = artsRouteTableLookupRank(pack->dbGuid);
  ARTS_INFO("Remote DB Send [Guid:%lu] [Rank: %d] [Mode:%d] [AcquireMode:"
            "%d] [UseTwinDiff: %d]",
            pack->dbGuid, pack->header.rank, pack->mode, pack->acquireMode,
            pack->useTwinDiff);
  if (redirected != artsGlobalRankId && redirected != -1)
    artsRemoteSendRequestAsync(redirected, (char *)pack, pack->header.size);
  else {
    struct artsDb *db = (struct artsDb *)artsRouteTableLookupItem(pack->dbGuid);
    if (db == NULL) {
      artsOutOfOrderHandleRemoteDbSend(pack->header.rank, pack->dbGuid,
                                       pack->mode);
    } else if (!artsIsGuidLocal(db->guid) &&
               pack->header.rank == artsGlobalRankId) {
      // This is when the memory model sends a CDAG write after CDAG write to
      // the same node The artsIsGuidLocal should be an extra check, maybe not
      // required
      artsRouteTableFireOO(pack->dbGuid, artsOutOfOrderHandler);
    } else {
      // Check if twin-diff is requested for WRITE mode
      if (pack->useTwinDiff && pack->acquireMode == ARTS_DB_WRITE) {
        ARTS_INFO("  Twin-diff requested - sending full DB");
        artsRemoteDbSendCheck(pack->header.rank, db, pack->mode);
      } else {
        artsRemoteDbSendCheck(pack->header.rank, db, pack->mode);
      }
    }
  }
}

void artsRemoteHandleDbReceived(struct artsRemoteDbSendPacket *packet) {
  struct artsDb *packetDb = (struct artsDb *)(packet + 1);
  ARTS_DEBUG("Handle DB Received [Guid:%lu] on rank %u", packetDb->guid,
             artsGlobalRankId);
  struct artsDb *dbRes = NULL;
  struct artsDb **dataPtr = NULL;
  itemState_t state = artsRouteTableLookupItemWithState(
      packetDb->guid, (void ***)&dataPtr, allocatedKey, true);

  struct artsDb *tPtr = (dataPtr) ? *dataPtr : NULL;
  struct artsDbList *dbList = NULL;
  if (tPtr && artsIsGuidLocal(packetDb->guid))
    dbList = (struct artsDbList *)tPtr->dbList;
  ARTS_DEBUG("Rec DB State: %u", state);
  switch (state) {
  case requestedKey: {
    if (packetDb->header.size == tPtr->header.size) {
      void *source = (void *)((struct artsDb *)packetDb + 1);
      void *dest = (void *)((struct artsDb *)tPtr + 1);
      memcpy(dest, source, packetDb->header.size - sizeof(struct artsDb));
      tPtr->dbList = dbList;
      dbRes = tPtr;
    } else {
      ARTS_INFO("Did the DB do a remote resize...");
    }
  } break;

  case reservedKey: {
    dbRes = (struct artsDb *)artsMallocAlignWithType(packetDb->header.size, 16,
                                                     artsDbMemorySize);
    memcpy(dbRes, packetDb, packetDb->header.size);
    if (artsIsGuidLocal(packetDb->guid))
      dbRes->dbList = artsNewDbList();
    else
      dbRes->dbList = NULL;
  } break;

  default: {
    itemState_t state = artsRouteTableLookupItemWithState(
        packetDb->guid, (void ***)&tPtr, anyKey, false);
  } break;
  }

  if (dbRes && artsRouteTableUpdateItem(packetDb->guid, (void *)dbRes,
                                        artsGlobalRankId, state)) {
    artsRouteTableFireOO(packetDb->guid, artsOutOfOrderHandler);
  }
}

void artsRemoteDbFullRequest(artsGuid_t dataGuid, int rank, struct artsEdt *edt,
                             int pos, artsType_t mode) {
  // Do not try to reduce full requests since they are unique
  struct artsRemoteDbFullRequestPacket packet;
  packet.dbGuid = dataGuid;
  packet.edt = edt;
  packet.slot = pos;
  packet.mode = mode;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_DB_FULL_REQUEST_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
  ARTS_DEBUG("Request Full DB[Guid:%lu] from rank %u to rank %u, mode: %u",
             dataGuid, rank, packet.header.rank, mode);
}

void artsRemoteDbForwardFull(int destRank, int sourceRank, artsGuid_t dataGuid,
                             struct artsEdt *edt, int pos, artsType_t mode) {
  struct artsRemoteDbFullRequestPacket packet;
  packet.header.size = sizeof(packet);
  packet.header.messageType = ARTS_REMOTE_DB_FULL_REQUEST_MSG;
  packet.header.rank = destRank;
  packet.dbGuid = dataGuid;
  packet.edt = edt;
  packet.slot = pos;
  packet.mode = mode;
  artsRemoteSendRequestAsync(sourceRank, (char *)&packet, sizeof(packet));
}

void artsRemoteDbFullSendNow(int rank, struct artsDb *db, struct artsEdt *edt,
                             unsigned int slot, artsType_t mode) {
  struct artsRemoteDbFullSendPacket packet;
  packet.edt = edt;
  packet.slot = slot;
  packet.mode = mode;
  int size = sizeof(struct artsRemoteDbFullSendPacket) + db->header.size;
  artsFillPacketHeader(&packet.header, size, ARTS_REMOTE_DB_FULL_SEND_MSG);
  artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet),
                                    (char *)db, db->header.size);
}

void artsRemoteDbFullSendCheck(int rank, struct artsDb *db, struct artsEdt *edt,
                               unsigned int slot, artsType_t mode) {
  if (!artsIsGuidLocal(db->guid)) {
    artsRouteTableReturnDb(db->guid, false);
    artsRemoteDbFullSendNow(rank, db, edt, slot, mode);
  } else if (artsAddDbDuplicate(db, rank, edt, slot, mode)) {
    artsRemoteDbFullSendNow(rank, db, edt, slot, mode);
  }
}

void artsRemoteDbFullSend(struct artsRemoteDbFullRequestPacket *pack) {
  unsigned int redirected = artsRouteTableLookupRank(pack->dbGuid);
  if (redirected != artsGlobalRankId && redirected != -1) {
    artsRemoteSendRequestAsync(redirected, (char *)pack, pack->header.size);
  } else {
    struct artsDb *db = (struct artsDb *)artsRouteTableLookupItem(pack->dbGuid);
    if (db == NULL) {
      artsOutOfOrderHandleRemoteDbFullSend(pack->header.rank, pack->dbGuid,
                                           (struct artsEdt *)pack->edt,
                                           pack->slot, pack->mode);
    } else {
      artsRemoteDbFullSendCheck(pack->header.rank, db,
                                (struct artsEdt *)pack->edt, pack->slot,
                                pack->mode);
    }
  }
}

void artsRemoteHandleDbFullRecieved(struct artsRemoteDbFullSendPacket *packet) {
  bool dec;
  itemState_t state;
  struct artsDb *packetDb = (struct artsDb *)(packet + 1);
  ARTS_DEBUG("Handle Full DB Received [Guid:%lu, Slot:%u, Mode:%u]",
             packetDb->guid, packet->slot, packet->mode);
  void **dataPtr = artsRouteTableReserve(packetDb->guid, &dec, &state);
  struct artsDb *dbRes = (dataPtr) ? (struct artsDb *)*dataPtr : NULL;
  if (dbRes) {
    if (packetDb->header.size == dbRes->header.size) {
      struct artsDbList *dbList = (struct artsDbList *)dbRes->dbList;
      void *source = (void *)((struct artsDb *)packetDb + 1);
      void *dest = (void *)((struct artsDb *)dbRes + 1);
      memcpy(dest, source, packetDb->header.size - sizeof(struct artsDb));
      dbRes->dbList = dbList;
    } else {
      ARTS_INFO("Did the DB do a remote resize...");
    }
  } else {
    dbRes = (struct artsDb *)artsMallocAlignWithType(packetDb->header.size, 16,
                                                     artsDbMemorySize);
    memcpy(dbRes, packetDb, packetDb->header.size);
    if (artsIsGuidLocal(packetDb->guid))
      dbRes->dbList = artsNewDbList();
    else
      dbRes->dbList = NULL;
  }
  if (artsRouteTableUpdateItem(packetDb->guid, (void *)dbRes, artsGlobalRankId,
                               state))
    artsRouteTableFireOO(packetDb->guid, artsOutOfOrderHandler);
  artsDbRequestCallback(packet->edt, packet->slot, dbRes);
}

void artsRemoteSendAlreadyLocal(int rank, artsGuid_t guid, struct artsEdt *edt,
                                unsigned int slot, artsType_t mode) {
  struct artsRemoteDbFullRequestPacket packet;
  packet.dbGuid = guid;
  packet.edt = edt;
  packet.slot = slot;
  packet.mode = mode;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleSendAlreadyLocal(void *pack) {
  struct artsRemoteDbFullRequestPacket *packet =
      (struct artsRemoteDbFullRequestPacket *)pack;
  int rank;
  struct artsDb *dbRes =
      (struct artsDb *)artsRouteTableLookupDb(packet->dbGuid, &rank, true);
  artsDbRequestCallback((struct artsEdt *)packet->edt, packet->slot, dbRes);
}

void artsRemoteGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid,
                         unsigned int slot, unsigned int offset,
                         unsigned int size, unsigned int rank) {
  struct artsRemoteGetPutPacket packet;
  packet.edtGuid = edtGuid;
  packet.dbGuid = dbGuid;
  packet.slot = slot;
  packet.offset = offset;
  packet.size = size;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_GET_FROM_DB_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleGetFromDb(void *pack) {
  struct artsRemoteGetPutPacket *packet = (struct artsRemoteGetPutPacket *)pack;
  artsGetFromDbAt(packet->edtGuid, packet->dbGuid, packet->slot, packet->offset,
                  packet->size, artsGlobalRankId);
}

void artsRemotePutInDb(void *ptr, artsGuid_t edtGuid, artsGuid_t dbGuid,
                       unsigned int slot, unsigned int offset,
                       unsigned int size, artsGuid_t epochGuid,
                       unsigned int rank) {
  struct artsRemoteGetPutPacket packet;
  packet.edtGuid = edtGuid;
  packet.dbGuid = dbGuid;
  packet.epochGuid = epochGuid;
  packet.slot = slot;
  packet.offset = offset;
  packet.size = size;
  int totalSize = sizeof(struct artsRemoteGetPutPacket) + size;
  artsFillPacketHeader(&packet.header, totalSize, ARTS_REMOTE_PUT_IN_DB_MSG);
  //    artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet),
  //    (char *)ptr, size);
  artsRemoteSendRequestPayloadAsyncFree(rank, (char *)&packet, sizeof(packet),
                                        (char *)ptr, 0, size, artsFree);
}

void artsRemoteHandlePutInDb(void *pack) {
  struct artsRemoteGetPutPacket *packet = (struct artsRemoteGetPutPacket *)pack;
  void *data = (void *)(packet + 1);
  internalPutInDb(data, packet->edtGuid, packet->dbGuid, packet->slot,
                  packet->offset, packet->size, packet->epochGuid,
                  artsGlobalRankId);
}

void artsRemoteSignalEdtWithPtr(artsGuid_t edtGuid, artsGuid_t dbGuid,
                                void *ptr, unsigned int size,
                                unsigned int slot) {
  unsigned int rank = artsGuidGetRank(edtGuid);
  ARTS_DEBUG("SEND NOW: %u -> %u", artsGlobalRankId, rank);
  struct artsRemoteSignalEdtWithPtrPacket packet;
  packet.edtGuid = edtGuid;
  packet.dbGuid = dbGuid;
  packet.size = size;
  packet.slot = slot;
  int totalSize = sizeof(struct artsRemoteSignalEdtWithPtrPacket) + size;
  artsFillPacketHeader(&packet.header, totalSize,
                       ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG);
  artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet),
                                    (char *)ptr, size);
}

void artsRemoteHandleSignalEdtWithPtr(void *pack) {
  struct artsRemoteSignalEdtWithPtrPacket *packet =
      (struct artsRemoteSignalEdtWithPtrPacket *)pack;
  void *source = (void *)(packet + 1);
  void *dest = artsMalloc(packet->size);
  memcpy(dest, source, packet->size);
  artsSignalEdtPtr(packet->edtGuid, packet->slot, dest, packet->size);
}

void artsRemoteMetricUpdate(int rank, int type, int level, uint64_t timeStamp,
                            uint64_t toAdd, bool sub) {
  ARTS_DEBUG("Remote Metric Update");
  struct artsRemoteMetricUpdate packet;
  packet.type = type;
  packet.timeStamp = timeStamp;
  packet.toAdd = toAdd;
  packet.sub = sub;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_METRIC_UPDATE_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteSend(unsigned int rank, sendHandler_t funPtr, void *args,
                    unsigned int size, bool free) {
  if (rank == artsGlobalRankId) {
    funPtr(args);
    if (free)
      artsFree(args);
    return;
  }
  struct artsRemoteSend packet;
  packet.funPtr = funPtr;
  int totalSize = sizeof(struct artsRemoteSend) + size;
  artsFillPacketHeader(&packet.header, totalSize, ARTS_REMOTE_SEND_MSG);

  if (free)
    artsRemoteSendRequestPayloadAsyncFree(rank, (char *)&packet, sizeof(packet),
                                          (char *)args, 0, size, artsFree);
  else
    artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet),
                                      (char *)args, size);
}

void artsRemoteHandleSend(void *pack) {
  struct artsRemoteSend *packet = (struct artsRemoteSend *)pack;
  void *args = (void *)(packet + 1);
  packet->funPtr(args);
}

void artsRemoteEpochInitSend(unsigned int rank, artsGuid_t epochGuid,
                             artsGuid_t edtGuid, unsigned int slot) {
  struct artsRemoteEpochInitPacket packet;
  packet.epochGuid = epochGuid;
  packet.edtGuid = edtGuid;
  packet.slot = slot;
  artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_INIT_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochInitSend(void *pack) {
  ARTS_DEBUG("Net Epoch Init Rec");
  struct artsRemoteEpochInitPacket *packet =
      (struct artsRemoteEpochInitPacket *)pack;
  artsGuid_t localEpochGuid = packet->epochGuid;
  createEpoch(&localEpochGuid, packet->edtGuid, packet->slot);
  packet->epochGuid = localEpochGuid;
}

void artsRemoteEpochInitPoolSend(unsigned int rank, unsigned int poolSize,
                                 artsGuid_t startGuid, artsGuid_t poolGuid) {
  //    ARTS_INFO("Net Epoch Init Pool Send: %u %lu %lu", rank, startGuid,
  //    poolGuid);
  struct artsRemoteEpochInitPoolPacket packet;
  packet.poolSize = poolSize;
  packet.startGuid = startGuid;
  packet.poolGuid = poolGuid;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_EPOCH_INIT_POOL_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochInitPoolSend(void *pack) {
  //    ARTS_INFO("Net Epoch Init Pool Rec");
  struct artsRemoteEpochInitPoolPacket *packet =
      (struct artsRemoteEpochInitPoolPacket *)pack;
  //    ARTS_INFO("Net Epoch Init Pool Rec %lu %lu", packet->startGuid,
  //    packet->poolGuid);
  artsGuid_t local_poolGuid = packet->poolGuid;
  artsGuid_t local_startGuid = packet->startGuid;
  createEpochPool(&local_poolGuid, packet->poolSize, &local_startGuid);
  packet->poolGuid = local_poolGuid;
  packet->startGuid = local_startGuid;
}

void artsRemoteEpochReq(unsigned int rank, artsGuid_t guid) {
  struct artsRemoteGuidOnlyPacket packet;
  packet.guid = guid;
  artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_REQ_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochReq(void *pack) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)pack;
  // For now the source and dest are the same...
  sendEpoch(packet->guid, packet->header.rank, packet->header.rank);
}

void artsRemoteEpochSend(unsigned int rank, artsGuid_t guid,
                         unsigned int active, unsigned int finish) {
  struct artsRemoteEpochSendPacket packet;
  packet.epochGuid = guid;
  packet.active = active;
  packet.finish = finish;
  artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_SEND_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochSend(void *pack) {
  struct artsRemoteEpochSendPacket *packet =
      (struct artsRemoteEpochSendPacket *)pack;
  reduceEpoch(packet->epochGuid, packet->active, packet->finish);
}

void artsRemoteAtomicAddInArrayDb(unsigned int rank, artsGuid_t dbGuid,
                                  unsigned int index, unsigned int toAdd,
                                  artsGuid_t edtGuid, unsigned int slot,
                                  artsGuid_t epochGuid) {
  struct artsRemoteAtomicAddInArrayDbPacket packet;
  packet.dbGuid = dbGuid;
  packet.edtGuid = edtGuid;
  packet.epochGuid = epochGuid;
  packet.slot = slot;
  packet.index = index;
  packet.toAdd = toAdd;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_ATOMIC_ADD_ARRAYDB_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleAtomicAddInArrayDb(void *pack) {
  struct artsRemoteAtomicAddInArrayDbPacket *packet =
      (struct artsRemoteAtomicAddInArrayDbPacket *)pack;
  struct artsDb *db = (struct artsDb *)artsRouteTableLookupItem(packet->dbGuid);
  internalAtomicAddInArrayDb(packet->dbGuid, packet->index, packet->toAdd,
                             packet->edtGuid, packet->slot, packet->epochGuid);
}

void artsRemoteAtomicCompareAndSwapInArrayDb(
    unsigned int rank, artsGuid_t dbGuid, unsigned int index,
    unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid,
    unsigned int slot, artsGuid_t epochGuid) {
  struct artsRemoteAtomicCompareAndSwapInArrayDbPacket packet;
  packet.dbGuid = dbGuid;
  packet.edtGuid = edtGuid;
  packet.epochGuid = epochGuid;
  packet.slot = slot;
  packet.index = index;
  packet.oldValue = oldValue;
  packet.newValue = newValue;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_ATOMIC_CAS_ARRAYDB_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleAtomicCompareAndSwapInArrayDb(void *pack) {
  struct artsRemoteAtomicCompareAndSwapInArrayDbPacket *packet =
      (struct artsRemoteAtomicCompareAndSwapInArrayDbPacket *)pack;
  struct artsDb *db = (struct artsDb *)artsRouteTableLookupItem(packet->dbGuid);
  internalAtomicCompareAndSwapInArrayDb(
      packet->dbGuid, packet->index, packet->oldValue, packet->newValue,
      packet->edtGuid, packet->slot, packet->epochGuid);
}

void artsRemoteEpochDelete(unsigned int rank, artsGuid_t epochGuid) {
  struct artsRemoteGuidOnlyPacket packet;
  packet.guid = epochGuid;
  artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_DELETE_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochDelete(void *pack) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)pack;
  deleteEpoch(packet->guid, NULL);
}

void artsDbMoveRequest(artsGuid_t dbGuid, unsigned int destRank) {
  struct artsRemoteDbRequestPacket packet;
  packet.dbGuid = dbGuid;
  packet.mode = ARTS_DB_ONCE;
  packet.header.size = sizeof(packet);
  packet.header.messageType = ARTS_REMOTE_DB_MOVE_REQ_MSG;
  packet.header.rank = destRank;
  artsRemoteSendRequestAsync(artsGuidGetRank(dbGuid), (char *)&packet,
                             sizeof(packet));
}

void artsDbMoveRequestHandle(void *pack) {
  struct artsRemoteDbRequestPacket *packet =
      (struct artsRemoteDbRequestPacket *)pack;
  artsDbMove(packet->dbGuid, packet->header.rank);
}

void artsRemoteHandleBufferSend(void *pack) {
  struct artsRemoteGuidOnlyPacket *packet =
      (struct artsRemoteGuidOnlyPacket *)pack;
  unsigned int size =
      packet->header.size - sizeof(struct artsRemoteGuidOnlyPacket);
  void *buffer = (void *)(packet + 1);
  artsSetBuffer(packet->guid, buffer, size);
}

void artsRemoteSignalContext(unsigned int rank, uint64_t ticket) {
  struct artsRemoteSignalContextPacket packet;
  packet.ticket = ticket;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_ATOMIC_ADD_ARRAYDB_MSG);
  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleSignalContext(void *pack) {
  struct artsRemoteSignalContextPacket *packet =
      (struct artsRemoteSignalContextPacket *)pack;
  artsSignalContext(packet->ticket);
}

void artsRemoteDbRename(artsGuid_t newGuid, artsGuid_t oldGuid) {
  unsigned int destRank = artsGuidGetRank(oldGuid);
  struct artsRemoteDbRename packet;
  packet.oldGuid = oldGuid;
  packet.newGuid = newGuid;
  packet.header.size = sizeof(packet);
  packet.header.messageType = ARTS_REMOTE_DB_RENAME_MSG;
  packet.header.rank = destRank;
  artsRemoteSendRequestAsync(destRank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleDbRename(void *pack) {
  struct artsRemoteDbRename *packet = (struct artsRemoteDbRename *)pack;
  artsDbRenameWithGuid(packet->newGuid, packet->oldGuid);
}

// Time synchronization for counter capture alignment (RTT-based)
// External declarations for time sync state (defined in Counter.c)
extern volatile int64_t artsCounterTimeOffset;
extern volatile bool artsCounterTimeSyncReceived;

// Worker sends sync request to master
void artsRemoteTimeSyncReqSend(unsigned int masterRank) {
  struct artsRemoteTimeSyncReqPacket packet;
  uint64_t t1 = artsGetTimeStamp();
  packet.workerSendTime = t1;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_TIME_SYNC_REQ_MSG);
  artsRemoteSendRequestAsync(masterRank, (char *)&packet, sizeof(packet));
  ARTS_INFO("Time sync request sent to master (rank %u), T1=%lu", masterRank,
            t1);
}

// Master receives sync request, sends back response
void artsRemoteHandleTimeSyncReq(void *pack) {
  struct artsRemoteTimeSyncReqPacket *packet =
      (struct artsRemoteTimeSyncReqPacket *)pack;
  uint64_t t2 = artsGetTimeStamp(); // Master's receive time
  unsigned int workerRank = packet->header.rank;
  ARTS_INFO("Time sync request received from worker (rank %u), T1=%lu, T2=%lu",
            workerRank, packet->workerSendTime, t2);
  // Send response back to worker
  artsRemoteTimeSyncRespSend(workerRank, packet->workerSendTime, t2);
}

// Master sends sync response to worker
void artsRemoteTimeSyncRespSend(unsigned int workerRank,
                                uint64_t workerSendTime,
                                uint64_t masterRecvTime) {
  struct artsRemoteTimeSyncRespPacket packet;
  packet.workerSendTime = workerSendTime;
  packet.masterRecvTime = masterRecvTime;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_TIME_SYNC_RESP_MSG);
  artsRemoteSendRequestAsync(workerRank, (char *)&packet, sizeof(packet));
}

// Worker receives sync response, calculates offset
void artsRemoteHandleTimeSyncResp(void *pack) {
  struct artsRemoteTimeSyncRespPacket *packet =
      (struct artsRemoteTimeSyncRespPacket *)pack;
  uint64_t t3 = artsGetTimeStamp(); // Worker's receive time
  uint64_t t1 = packet->workerSendTime;
  uint64_t t2 = packet->masterRecvTime;

  // NTP-like offset calculation:
  // offset = ((T1 - T2) + (T3 - T2)) / 2
  //        = (T1 + T3 - 2*T2) / 2
  //        = (T1 + T3) / 2 - T2
  // This gives us: worker_time - offset = master_time
  // Positive offset means worker clock is ahead of master
  int64_t offset = ((int64_t)t1 + (int64_t)t3) / 2 - (int64_t)t2;

  // Also calculate RTT for logging
  uint64_t rtt = t3 - t1;

  // Ensure atomic and ordered publication of offset and sync flag:
  // - offset is stored atomically
  // - the flag is stored with release semantics so readers that see 'true'
  //   are guaranteed to see the corresponding offset value.
  __atomic_store_n(&artsCounterTimeOffset, offset, __ATOMIC_RELAXED);
  __atomic_store_n(&artsCounterTimeSyncReceived, true, __ATOMIC_RELEASE);

  ARTS_INFO("Time sync complete: T1=%lu, T2=%lu, T3=%lu, RTT=%lu ns (%.3f ms), "
            "offset=%ld ns (%.3f ms)",
            t1, t2, t3, rtt, (double)rtt / 1000000.0, offset,
            (double)offset / 1000000.0);
}

// Cluster counter reduction
extern uint64_t *artsClusterCounters;
extern volatile unsigned int artsClusterNodesReceived;

void artsRemoteCounterReduceSend(unsigned int rank, unsigned int counterIndex,
                                 uint64_t value) {
  struct artsRemoteCounterReducePacket packet;
  packet.counterIndex = counterIndex;
  packet.value = value;
  artsFillPacketHeader(&packet.header, sizeof(packet),
                       ARTS_REMOTE_COUNTER_REDUCE_MSG);

  artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleCounterReduce(void *pack) {
  struct artsRemoteCounterReducePacket *packet =
      (struct artsRemoteCounterReducePacket *)pack;
  unsigned int idx = packet->counterIndex;
  uint64_t value = packet->value;

  // Bounds check to prevent out-of-bounds access
  if (idx >= NUM_COUNTER_TYPES) {
    ARTS_INFO("Invalid counterIndex %u received, ignoring", idx);
    return;
  }

  // Apply reduction based on the counter's reduce type
  switch (artsCounterReduceTypes[idx]) {
  case artsCounterSum:
    __sync_fetch_and_add(&artsClusterCounters[idx], value);
    break;
  case artsCounterMax: {
    uint64_t old = artsClusterCounters[idx];
    while (value > old) {
      uint64_t result =
          __sync_val_compare_and_swap(&artsClusterCounters[idx], old, value);
      if (result == old)
        break;
      old = result;
    }
    break;
  }
  case artsCounterMin: {
    uint64_t old = artsClusterCounters[idx];
    while (value < old) {
      uint64_t result =
          __sync_val_compare_and_swap(&artsClusterCounters[idx], old, value);
      if (result == old)
        break;
      old = result;
    }
    break;
  }
  }
}

void artsRemoteCounterReduceDoneSend(unsigned int masterRank) {
  struct artsRemotePacket packet;
  artsFillPacketHeader(&packet, sizeof(packet),
                       ARTS_REMOTE_COUNTER_REDUCE_DONE_MSG);
  artsRemoteSendRequestAsync(masterRank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleCounterReduceDone(void) {
  // Increment the number of nodes that have sent their cluster counters
  __sync_fetch_and_add(&artsClusterNodesReceived, 1);
}
