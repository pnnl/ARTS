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
#include "artsRemoteFunctions.h"
#include "artsGlobals.h"
#include "artsGuid.h"
#include "artsRemoteProtocol.h"
#include "artsRouteTable.h"
#include "artsRuntime.h"
#include "artsOutOfOrder.h"
#include "artsAtomics.h"
#include "artsRemote.h"
#include "artsDbFunctions.h"
#include "artsDbList.h"
#include "artsDebug.h"
#include "artsEdtFunctions.h"
#include "artsServer.h"
#include "artsTerminationDetection.h"
#include "artsArrayDb.h"
#include "artsCounter.h"
#include "artsIntrospection.h"
#include "artsTMT.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

static inline void artsFillPacketHeader(struct artsRemotePacket * header, unsigned int size, unsigned int messageType)
{
    header->size = size;
    header->messageType = messageType;
    header->rank = artsGlobalRankId;
}

void artsRemoteAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot, artsType_t mode, unsigned int rank)
{
    DPRINTF("Remote Add dependence sent %d\n", rank);
    struct artsRemoteAddDependencePacket packet;
    packet.source = source;
    packet.destination = destination;
    packet.slot = slot;
    packet.mode = mode;
    packet.destRoute = artsGuidGetRank(destination); 
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_ADD_DEPENDENCE_MSG);
    artsRemoteSendRequestAsync( rank, (char *)&packet, sizeof(packet) );
}

void artsRemoteUpdateRouteTable(artsGuid_t guid, unsigned int rank)
{
    DPRINTF("Here Update Table %ld %u\n", guid, rank);
    unsigned int owner = artsGuidGetRank(guid);
    if(owner == artsGlobalRankId)
    {
        struct artsDbFrontierIterator * iter = artsRouteTableGetRankDuplicates(guid, rank);
        if(iter)
        {
            unsigned int node;
            while(artsDbFrontierIterNext(iter, &node))
            {
                if(node != artsGlobalRankId && node != rank)
                {
                    struct artsRemoteInvalidateDbPacket outPacket;
                    outPacket.guid = guid;
                    artsFillPacketHeader(&outPacket.header, sizeof(outPacket), ARTS_REMOTE_INVALIDATE_DB_MSG);
                    artsRemoteSendRequestAsync(node, (char *)&outPacket, sizeof(outPacket));
                }
            }
            artsFree(iter);
        }
    }
    else
    {         
        struct artsRemoteUpdateDbGuidPacket packet;
        artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_DB_UPDATE_GUID_MSG);
        packet.guid = guid;
        artsRemoteSendRequestAsync(owner, (char *)&packet, sizeof(packet));
    }

}

void artsRemoteHandleUpdateDbGuid(void * ptr)
{
    struct artsRemoteUpdateDbGuidPacket * packet = ptr;
    DPRINTF("Updated %ld to %d\n", packet->guid, packet->header.rank);
    artsRemoteUpdateRouteTable(packet->guid, packet->header.rank);
}

void artsRemoteHandleInvalidateDb(void * ptr)
{
    struct artsRemoteInvalidateDbPacket * packet = ptr;
    void * address = artsRouteTableLookupItem(packet->guid);
    artsRouteTableInvalidateItem(packet->guid);
}

//TODO: Fix this...
void artsRemoteDbDestroy(artsGuid_t guid, unsigned int originRank, bool clean)
{
//    unsigned int rank = artsGuidGetRank(guid);
//    //PRINTF("Destroy Check\n");
//    if(rank == artsGlobalRankId)
//    {
//        struct artsRouteInvalidate * table = artsRouteTableGetRankDuplicates(guid);
//        struct artsRouteInvalidate * next = table;
//        struct artsRouteInvalidate * current;
//        
//        if(next != NULL && next->used != 0)
//        {
//            struct artsRemoteGuidOnlyPacket outPacket;
//            outPacket.guid = guid;
//            artsFillPacketHeader(&outPacket.header, sizeof(outPacket), ARTS_REMOTE_DB_DESTROY_MSG);
//            
//            int lastSend=-1;
//            while( next != NULL)
//            {
//                for(int i=0; i < next->used; i++ )
//                {
//                    if(originRank != next->data[i] && next->data[i] != lastSend)
//                    {
////                        PRINTF("Destroy Send 1\n");
//                        lastSend = next->data[i];
//                        artsRemoteSendRequestAsync(next->data[i], (char *)&outPacket, sizeof(outPacket));
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
////            PRINTF("Origin Destroy\n");
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
//        //DPRINTF("depv %ld %p %p\n", guid, dbAddress, callBack);            
//        struct artsRemoteGuidOnlyPacket packet;
//        if(!clean)
//            artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_DB_DESTROY_FORWARD_MSG);
//        else 
//            artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_DB_CLEAN_FORWARD_MSG);
//        packet.guid = guid;
////        PRINTF("Destroy Send 2\n");
////        artsDebugPrintStack();
//        artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
//    }
}

void artsRemoteHandleDbDestroyForward(void * ptr)
{
    //PRINTF("Destroy\n");
    struct artsRemoteGuidOnlyPacket * packet = ptr;
    artsRemoteDbDestroy(packet->guid, packet->header.rank, 0);
    artsDbDestroySafe(packet->guid, false);
}

void artsRemoteHandleDbCleanForward(void * ptr)
{
    struct artsRemoteGuidOnlyPacket * packet = ptr;
    artsRemoteDbDestroy(packet->guid, packet->header.rank, 1);
}

void artsRemoteHandleDbDestroy(void * ptr)
{
    struct artsRemoteGuidOnlyPacket * packet = ptr;
    //PRINTF("Deleted %ld\n", packet->guid);
    //PRINTF("Destroy\n");
    artsDbDestroySafe(packet->guid, false);
}

void artsRemoteUpdateDb(artsGuid_t guid, bool sendDb)
{
//    sendDb = true;
    unsigned int rank = artsGuidGetRank(guid);
    if(rank != artsGlobalRankId)
    {
        struct artsRemoteUpdateDbPacket packet;
        packet.guid = guid;
        struct artsDb * db = NULL;
        if(sendDb && (db = artsRouteTableLookupItem(guid)))
        {
            int size = sizeof(struct artsRemoteUpdateDbPacket)+db->header.size;
            artsFillPacketHeader(&packet.header, size, ARTS_REMOTE_DB_UPDATE_MSG);
            artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)db, db->header.size);
        }
        else
        {
            artsFillPacketHeader(&packet.header, sizeof(struct artsRemoteUpdateDbPacket), ARTS_REMOTE_DB_UPDATE_MSG);
            artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
        }
    }
}

void artsRemoteHandleUpdateDb(void * ptr)
{
    struct artsRemoteUpdateDbPacket * packet = (struct artsRemoteUpdateDbPacket *) ptr;
    struct artsDb * packetDb = (struct artsDb *)(packet+1);
    unsigned int rank = artsGuidGetRank(packet->guid);
    if(rank == artsGlobalRankId)
    {
        struct artsDb ** dataPtr;
        bool write = packet->header.size > sizeof(struct artsRemoteUpdateDbPacket);
        itemState_t state = artsRouteTableLookupItemWithState(packet->guid, (void***)&dataPtr, allocatedKey, write);
        struct artsDb * db = (dataPtr) ? *dataPtr : NULL;
//        PRINTF("DB: %p %lu %u %u %d\n", db, packet->guid, packet->header.size, sizeof(struct artsRemoteUpdateDbPacket), state);
        if(write)
        {
            void * ptr = (void*)(db+1);
            memcpy(ptr, packetDb, db->header.size - sizeof(struct artsDb));
            artsRouteTableSetRank(packet->guid, artsGlobalRankId);
            artsProgressFrontier(db, artsGlobalRankId);
            // artsRouteTableDecItem(packet->guid, dataPtr);
        }
        else
        {
            artsProgressFrontier(db, packet->header.rank);
        }
    }
}

void artsRemoteMemoryMove(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType, void(*freeMethod)(void*))
{
    struct artsRemoteMemoryMovePacket packet;
    artsFillPacketHeader(&packet.header, sizeof(packet)+memSize, messageType);
    packet.guid = guid;
    artsRemoteSendRequestPayloadAsyncFree(route, (char *)&packet, sizeof(packet), ptr, 0, memSize, freeMethod);
    artsRouteTableRemoveItem(guid);
}

void artsRemoteMemoryMoveNoFree(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType)
{
    struct artsRemoteMemoryMovePacket packet;
    artsFillPacketHeader(&packet.header, sizeof(packet)+memSize, messageType);
    packet.guid = guid;
    artsRemoteSendRequestPayloadAsync(route, (char *)&packet, sizeof(packet), ptr, memSize);
}

void artsRemoteHandleEdtMove(void * ptr)
{
    struct artsRemoteMemoryMovePacket * packet = ptr ;    
    unsigned int size = packet->header.size - sizeof(struct artsRemoteMemoryMovePacket);

    ARTSSETMEMSHOTTYPE(artsEdtMemorySize);
    struct artsEdt * edt = artsMalloc(size);
    ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);

    memcpy(edt, packet+1, size);
    artsRouteTableAddItemRace(edt, (artsGuid_t) packet->guid, artsGlobalRankId, false);
    if(edt->depcNeeded == 0) 
        artsHandleReadyEdt(edt);
    else   
        artsRouteTableFireOO(packet->guid, artsOutOfOrderHandler);           
}

void artsRemoteHandleDbMove(void * ptr)
{   
    struct artsRemoteMemoryMovePacket * packet = ptr ;
    unsigned int size = packet->header.size - sizeof(struct artsRemoteMemoryMovePacket);
    
    struct artsDb * dbHeader = (struct artsDb *)(packet+1);
    unsigned int dbSize  = dbHeader->header.size;
    
    ARTSSETMEMSHOTTYPE(artsDbMemorySize);
    struct artsHeader * memPacket = artsMalloc(dbSize);
    ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);
    
    if(size == dbSize)
        memcpy(memPacket, packet+1, size);
    else
    {
        memPacket->type = (unsigned int) artsGuidGetType(packet->guid);
        memPacket->size = dbSize;
    }
    //We need a local pointer for this node
    if(dbHeader->dbList)
    {
        struct artsDb * newDb = (struct artsDb*)memPacket;
        newDb->dbList = artsNewDbList();
    }
    
    if(artsRouteTableAddItemRace(memPacket, (artsGuid_t) packet->guid, artsGlobalRankId, false))
        artsRouteTableFireOO(packet->guid, artsOutOfOrderHandler);
}

void artsRemoteHandleEventMove(void * ptr)
{
    struct artsRemoteMemoryMovePacket * packet = ptr ;
    unsigned int size = packet->header.size - sizeof(struct artsRemoteMemoryMovePacket);
    
    ARTSSETMEMSHOTTYPE(artsEventMemorySize);
    struct artsHeader * memPacket = artsMalloc(size);
    ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);
    
    memcpy(memPacket, packet+1, size);
    artsRouteTableAddItemRace(memPacket, (artsGuid_t) packet->guid, artsGlobalRankId, false);
    artsRouteTableFireOO(packet->guid, artsOutOfOrderHandler);
}

void artsRemoteSignalEdt(artsGuid_t edt, artsGuid_t db, uint32_t slot, artsType_t mode)
{
    DPRINTF("Remote Signal %ld %ld %d %d\n",edt,db,slot, artsGuidGetRank(edt));
    struct artsRemoteEdtSignalPacket packet;
    
    unsigned int rank = artsGuidGetRank(edt); 

    if(rank == artsGlobalRankId)
        rank = artsRouteTableLookupRank( edt);
    packet.db = db;
    packet.edt = edt;
    packet.slot = slot;
    packet.mode = mode;
    packet.dbRoute = artsGuidGetRank(db); 
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_EDT_SIGNAL_MSG);
    artsRemoteSendRequestAsync( rank, (char *)&packet, sizeof(packet) );
//    if(artsGlobalRankId==1) {PRINTF("Size: %u\n", sizeof(packet)); artsDebugPrintStack();}
}


void artsRemoteEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot)
{  
    DPRINTF("Remote Satisfy Slot\n");
    struct artsRemoteEventSatisfySlotPacket packet;
    packet.event = eventGuid;
    packet.db = dataGuid;
    packet.slot = slot;
    packet.dbRoute = artsGuidGetRank(dataGuid); 
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG);
    artsRemoteSendRequestAsync(artsGuidGetRank( eventGuid ), (char *)&packet, sizeof(packet) );
}

void artsDbRequestCallback(struct artsEdt *edt, unsigned int slot, struct artsDb * dbRes)
{ 
    artsEdtDep_t * depv = artsGetDepv(edt);
    
    depv[slot].ptr = dbRes + 1;
    unsigned int temp = artsAtomicSub(&edt->depcNeeded, 1U);
    if(temp == 0)
        artsHandleRemoteStolenEdt(edt);
}

bool artsRemoteDbRequest(artsGuid_t dataGuid, int rank, struct artsEdt * edt, int pos, artsType_t mode, bool aggRequest)
{
    if(artsRouteTableAddSent(dataGuid, edt, pos, aggRequest))
    {
        struct artsRemoteDbRequestPacket packet;
        packet.dbGuid = dataGuid;
        packet.mode = mode;
        artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_DB_REQUEST_MSG);
        artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
        DPRINTF("DB req send: %u -> %u mode: %u agg: %u\n", packet.header.rank, rank, mode, aggRequest);
        return true;
    }
    return false;
}

void artsRemoteDbForward(int destRank, int sourceRank, artsGuid_t dataGuid, artsType_t mode)
{
    struct artsRemoteDbRequestPacket packet;
    packet.header.size = sizeof(packet);
    packet.header.messageType = ARTS_REMOTE_DB_REQUEST_MSG;
    packet.header.rank = destRank;
    packet.dbGuid = dataGuid;
    packet.mode = mode;
    artsRemoteSendRequestAsync(sourceRank, (char *)&packet, sizeof(packet));
}

void artsRemoteDbSendNow(int rank, struct artsDb * db)
{
    DPRINTF("SEND NOW: %u -> %u\n", artsGlobalRankId, rank);
//    artsDebugPrintStack();
    struct artsRemoteDbSendPacket packet;
    int size = sizeof(struct artsRemoteDbSendPacket)+db->header.size;
    artsFillPacketHeader(&packet.header, size, ARTS_REMOTE_DB_SEND_MSG);
    artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)db, db->header.size);
}

void artsRemoteDbSendCheck(int rank, struct artsDb * db, artsType_t mode)
{
    if(!artsIsGuidLocal(db->guid))
    {
        artsRouteTableReturnDb(db->guid, false);
        artsRemoteDbSendNow(rank, db);
    }
    else if(artsAddDbDuplicate(db, rank, NULL, 0, mode))
    {
        artsRemoteDbSendNow(rank, db);
    }
}

void artsRemoteDbSend(struct artsRemoteDbRequestPacket * pack)
{
    unsigned int redirected = artsRouteTableLookupRank(pack->dbGuid);
    if(redirected != artsGlobalRankId && redirected != -1)
        artsRemoteSendRequestAsync(redirected, (char *)pack, pack->header.size);
    else
    {
        struct artsDb * db = artsRouteTableLookupItem(pack->dbGuid);
        if(db == NULL)
        {
            artsOutOfOrderHandleRemoteDbSend(pack->header.rank, pack->dbGuid, pack->mode);
        }
        else if(!artsIsGuidLocal(db->guid) && pack->header.rank == artsGlobalRankId)
        {
            //This is when the memory model sends a CDAG write after CDAG write to the same node
            //The artsIsGuidLocal should be an extra check, maybe not required
            artsRouteTableFireOO(pack->dbGuid, artsOutOfOrderHandler);
        }
        else
            artsRemoteDbSendCheck(pack->header.rank, db, pack->mode);
    }
}

void artsRemoteHandleDbRecieved(struct artsRemoteDbSendPacket * packet)
{
    struct artsDb * packetDb = (struct artsDb *)(packet+1);    
    struct artsDb * dbRes = NULL;
    struct artsDb ** dataPtr = NULL;
    itemState_t state = artsRouteTableLookupItemWithState(packetDb->guid, (void***)&dataPtr, allocatedKey, true);
    
    struct artsDb * tPtr = (dataPtr) ? *dataPtr : NULL;
    struct artsDbList * dbList = NULL;
    if(tPtr && artsIsGuidLocal(packetDb->guid))    
        dbList = tPtr->dbList;
    DPRINTF("Rec DB State: %u\n", state);
    switch(state)
    {              
        case requestedKey:
            if(packetDb->header.size == tPtr->header.size)
            {
                void * source = (void*)((struct artsDb *) packetDb + 1);
                void * dest = (void*)((struct artsDb *) tPtr + 1);
                memcpy(dest, source, packetDb->header.size - sizeof(struct artsDb));
                tPtr->dbList = dbList;
                dbRes = tPtr;
            }
            else
            {
                PRINTF("Did the DB do a remote resize...\n");
            }
            break;
            
        case reservedKey:
            ARTSSETMEMSHOTTYPE(artsDbMemorySize);
            dbRes = artsMalloc(packetDb->header.size);
            ARTSSETMEMSHOTTYPE(artsDbMemorySize);
            memcpy(dbRes, packetDb, packetDb->header.size);
            if(artsIsGuidLocal(packetDb->guid))
               dbRes->dbList = artsNewDbList(); 
            else
                dbRes->dbList = NULL;
            break;
            
        default:
            PRINTF("Got a DB but current key state is %d looking again\n", state);
            itemState_t state = artsRouteTableLookupItemWithState(packetDb->guid, (void*)&tPtr, anyKey, false);
            PRINTF("The current state after re-checking is %d\n", state);
            break;
    }
    if(dbRes && artsRouteTableUpdateItem(packetDb->guid, (void*)dbRes, artsGlobalRankId, state))
    {
        artsRouteTableFireOO(packetDb->guid, artsOutOfOrderHandler);
    }
    
    // artsRouteTableDecItem(packetDb->guid, dataPtr);
}

void artsRemoteDbFullRequest(artsGuid_t dataGuid, int rank, struct artsEdt * edt, int pos, artsType_t mode)
{
    //Do not try to reduce full requests since they are unique
    struct artsRemoteDbFullRequestPacket packet;
    packet.dbGuid = dataGuid;
    packet.edt = edt;
    packet.slot = pos;
    packet.mode = mode;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_DB_FULL_REQUEST_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteDbForwardFull(int destRank, int sourceRank, artsGuid_t dataGuid, struct artsEdt * edt, int pos, artsType_t mode)
{
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

void artsRemoteDbFullSendNow(int rank, struct artsDb * db, struct artsEdt * edt, unsigned int slot, artsType_t mode)
{
    DPRINTF("SEND FULL NOW: %u -> %u\n", artsGlobalRankId, rank);
    struct artsRemoteDbFullSendPacket packet;
    packet.edt = edt;
    packet.slot = slot;
    packet.mode = mode;
    int size = sizeof(struct artsRemoteDbFullSendPacket)+db->header.size;
    artsFillPacketHeader(&packet.header, size, ARTS_REMOTE_DB_FULL_SEND_MSG);
    artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)db, db->header.size);
}

void artsRemoteDbFullSendCheck(int rank, struct artsDb * db, struct artsEdt * edt, unsigned int slot, artsType_t mode)
{
    if(!artsIsGuidLocal(db->guid))
    {
        artsRouteTableReturnDb(db->guid, false);
        artsRemoteDbFullSendNow(rank, db, edt, slot, mode);
    }
    else if(artsAddDbDuplicate(db, rank, edt, slot, mode))
    {
        artsRemoteDbFullSendNow(rank, db, edt, slot, mode);
    }
}

void artsRemoteDbFullSend(struct artsRemoteDbFullRequestPacket * pack)
{
    unsigned int redirected = artsRouteTableLookupRank(pack->dbGuid);
    if(redirected != artsGlobalRankId && redirected != -1)
        artsRemoteSendRequestAsync(redirected, (char *)pack, pack->header.size);
    else
    {
        struct artsDb * db = artsRouteTableLookupItem(pack->dbGuid);
        if(db == NULL)
        {
            artsOutOfOrderHandleRemoteDbFullSend(pack->header.rank, pack->dbGuid, pack->edt, pack->slot, pack->mode);
        }
        else
            artsRemoteDbFullSendCheck(pack->header.rank, db, pack->edt, pack->slot, pack->mode);
    }
}

void artsRemoteHandleDbFullRecieved(struct artsRemoteDbFullSendPacket * packet)
{
    bool dec;
    itemState_t state;
    struct artsDb * packetDb = (struct artsDb *)(packet+1);    
    void ** dataPtr = artsRouteTableReserve(packetDb->guid, &dec, &state);
    struct artsDb * dbRes = (dataPtr) ? *dataPtr : NULL;    
    if(dbRes)
    {
        if(packetDb->header.size == dbRes->header.size)
        {
            struct artsDbList * dbList = dbRes->dbList;
            void * source = (void*)((struct artsDb *) packetDb + 1);
            void * dest = (void*)((struct artsDb *) dbRes + 1);
            memcpy(dest, source, packetDb->header.size - sizeof(struct artsDb));
            dbRes->dbList = dbList;
        }
        else
            PRINTF("Did the DB do a remote resize...\n");
    }
    else
    {
        ARTSSETMEMSHOTTYPE(artsDbMemorySize);
        dbRes = artsMalloc(packetDb->header.size);
        ARTSSETMEMSHOTTYPE(artsDbMemorySize);
        memcpy(dbRes, packetDb, packetDb->header.size);
        if(artsIsGuidLocal(packetDb->guid))
           dbRes->dbList = artsNewDbList();
        else
            dbRes->dbList = NULL;
    }
    if(artsRouteTableUpdateItem(packetDb->guid, (void*)dbRes, artsGlobalRankId, state))
        artsRouteTableFireOO(packetDb->guid, artsOutOfOrderHandler);
    artsDbRequestCallback(packet->edt, packet->slot, dbRes);
    // if(dec)
    //     artsRouteTableDecItem(packetDb->guid, dataPtr);
}

void artsRemoteSendAlreadyLocal(int rank, artsGuid_t guid, struct artsEdt * edt, unsigned int slot, artsType_t mode)
{
    struct artsRemoteDbFullRequestPacket packet;
    packet.dbGuid = guid;
    packet.edt = edt;
    packet.slot = slot;
    packet.mode = mode;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG);
    artsRemoteSendRequestAsync(rank, (char*)&packet, sizeof(packet));
}

void artsRemoteHandleSendAlreadyLocal(void * pack)
{
    struct artsRemoteDbFullRequestPacket * packet = pack;
    int rank;
    struct artsDb * dbRes = artsRouteTableLookupDb(packet->dbGuid, &rank);
    artsDbRequestCallback(packet->edt, packet->slot, dbRes);
}

void artsRemoteGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank)
{
    struct artsRemoteGetPutPacket packet;
    packet.edtGuid = edtGuid;
    packet.dbGuid = dbGuid;
    packet.slot = slot;
    packet.offset = offset;
    packet.size = size;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_GET_FROM_DB_MSG);
    artsRemoteSendRequestAsync(rank, (char*)&packet, sizeof(packet));
}

void artsRemoteHandleGetFromDb(void * pack)
{
    struct artsRemoteGetPutPacket * packet = pack;
    artsGetFromDbAt(packet->edtGuid, packet->dbGuid, packet->slot, packet->offset, packet->size, artsGlobalRankId);
}

void artsRemotePutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid, unsigned int rank)
{
    struct artsRemoteGetPutPacket packet;
    packet.edtGuid = edtGuid;
    packet.dbGuid = dbGuid;
    packet.epochGuid = epochGuid;
    packet.slot = slot;
    packet.offset = offset;
    packet.size = size;
    int totalSize = sizeof(struct artsRemoteGetPutPacket)+size;
    artsFillPacketHeader(&packet.header, totalSize, ARTS_REMOTE_PUT_IN_DB_MSG);
//    artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)ptr, size);
    artsRemoteSendRequestPayloadAsyncFree(rank, (char*)&packet, sizeof(packet), (char *)ptr, 0, size, artsFree);
}

void artsRemoteHandlePutInDb(void * pack)
{
    struct artsRemoteGetPutPacket * packet = pack;
    void * data = (void*)(packet+1);
    internalPutInDb(data, packet->edtGuid, packet->dbGuid, packet->slot, packet->offset, packet->size, packet->epochGuid, artsGlobalRankId);
}

void artsRemoteSignalEdtWithPtr(artsGuid_t edtGuid, artsGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot)
{
    unsigned int rank = artsGuidGetRank(edtGuid);
    DPRINTF("SEND NOW: %u -> %u\n", artsGlobalRankId, rank);
    struct artsRemoteSignalEdtWithPtrPacket packet;
    packet.edtGuid = edtGuid;
    packet.dbGuid = dbGuid;
    packet.size = size;
    packet.slot = slot;
    int totalSize = sizeof(struct artsRemoteSignalEdtWithPtrPacket)+size;
    artsFillPacketHeader(&packet.header, totalSize, ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG);
    artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)ptr, size);
}

void artsRemoteHandleSignalEdtWithPtr(void * pack)
{
    struct artsRemoteSignalEdtWithPtrPacket * packet = pack;
    void * source = (void*)(packet + 1);
    void * dest = artsMalloc(packet->size);
    memcpy(dest, source, packet->size);
    artsSignalEdtPtr(packet->edtGuid, packet->slot, dest, packet->size);
}

void artsRemoteMetricUpdate(int rank, int type, int level, uint64_t timeStamp, uint64_t toAdd, bool sub)
{
    DPRINTF("Remote Metric Update");
    struct artsRemoteMetricUpdate packet; 
    packet.type = type;
    packet.timeStamp = timeStamp;
    packet.toAdd = toAdd;
    packet.sub = sub; 
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_REMOTE_METRIC_UPDATE_MSG);
    artsRemoteSendRequestAsync( rank, (char *)&packet, sizeof(packet) );
}

void artsRemoteSend(unsigned int rank, sendHandler_t funPtr, void * args, unsigned int size, bool free)
{
    if(rank==artsGlobalRankId)
    {
        funPtr(args);
        if(free)
            artsFree(args);
        return;
    }
    struct artsRemoteSend packet;
    packet.funPtr = funPtr;
    int totalSize = sizeof(struct artsRemoteSend)+size;
    artsFillPacketHeader(&packet.header, totalSize, ARTS_REMOTE_SEND_MSG);
    
    if(free)
        artsRemoteSendRequestPayloadAsyncFree(rank, (char*)&packet, sizeof(packet), (char *)args, 0, size, artsFree);
    else
        artsRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)args, size);
}

void artsRemoteHandleSend(void * pack)
{
    struct artsRemoteSend * packet = pack;
    void * args = (void*)(packet+1);
    packet->funPtr(args);
}

void artsRemoteEpochInitSend(unsigned int rank, artsGuid_t epochGuid, artsGuid_t edtGuid, unsigned int slot)
{
    DPRINTF("Net Epoch Init Send: %u\n", rank);
    struct artsRemoteEpochInitPacket packet;
    packet.epochGuid = epochGuid;
    packet.edtGuid = edtGuid;
    packet.slot = slot;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_INIT_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochInitSend(void * pack)
{
    DPRINTF("Net Epoch Init Rec\n");
    struct artsRemoteEpochInitPacket * packet = pack;
    createEpoch(&packet->epochGuid, packet->edtGuid, packet->slot);
}

void artsRemoteEpochInitPoolSend(unsigned int rank, unsigned int poolSize, artsGuid_t startGuid, artsGuid_t poolGuid)
{
//    PRINTF("Net Epoch Init Pool Send: %u %lu %lu\n", rank, startGuid, poolGuid);
    struct artsRemoteEpochInitPoolPacket packet;
    packet.poolSize = poolSize;
    packet.startGuid = startGuid;
    packet.poolGuid = poolGuid;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_INIT_POOL_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochInitPoolSend(void * pack)
{
//    PRINTF("Net Epoch Init Pool Rec\n");
    struct artsRemoteEpochInitPoolPacket * packet = pack;
//    PRINTF("Net Epoch Init Pool Rec %lu %lu\n", packet->startGuid, packet->poolGuid);
    createEpochPool(&packet->poolGuid, packet->poolSize, &packet->startGuid);
}

void artsRemoteEpochReq(unsigned int rank, artsGuid_t guid)
{
    DPRINTF("Net Epoch Req Send: %u\n", rank);
    struct artsRemoteEpochReqPacket packet;
    packet.epochGuid = guid;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_REQ_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochReq(void * pack)
{
    DPRINTF("Net Epoch Req Rec\n");
    struct artsRemoteEpochReqPacket * packet = pack;
    //For now the source and dest are the same...
    sendEpoch(packet->epochGuid, packet->header.rank, packet->header.rank);
}

void artsRemoteEpochSend(unsigned int rank, artsGuid_t guid, unsigned int active, unsigned int finish)
{
    DPRINTF("Net Epoch Send Send: %u\n", rank);
    struct artsRemoteEpochSendPacket packet;
    packet.epochGuid = guid;
    packet.active = active;
    packet.finish = finish;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_SEND_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochSend(void * pack)
{
    DPRINTF("Net Epoch Send: Rec\n");
    struct artsRemoteEpochSendPacket * packet = pack;
    reduceEpoch(packet->epochGuid, packet->active, packet->finish);
}

void artsRemoteAtomicAddInArrayDb(unsigned int rank, artsGuid_t dbGuid, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid)
{
    struct artsRemoteAtomicAddInArrayDbPacket packet;
    packet.dbGuid = dbGuid;
    packet.edtGuid = edtGuid;
    packet.epochGuid = epochGuid;
    packet.slot = slot;
    packet.index = index;
    packet.toAdd = toAdd;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_ATOMIC_ADD_ARRAYDB_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleAtomicAddInArrayDb(void * pack)
{
    struct artsRemoteAtomicAddInArrayDbPacket * packet = pack;
    struct artsDb * db = artsRouteTableLookupItem(packet->dbGuid);
    internalAtomicAddInArrayDb(packet->dbGuid, packet->index, packet->toAdd, packet->edtGuid, packet->slot, packet->epochGuid);  
}

void artsRemoteAtomicCompareAndSwapInArrayDb(unsigned int rank, artsGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid)
{
    struct artsRemoteAtomicCompareAndSwapInArrayDbPacket packet;
    packet.dbGuid = dbGuid;
    packet.edtGuid = edtGuid;
    packet.epochGuid = epochGuid;
    packet.slot = slot;
    packet.index = index;
    packet.oldValue = oldValue;
    packet.newValue = newValue;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_ATOMIC_CAS_ARRAYDB_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleAtomicCompareAndSwapInArrayDb(void * pack)
{
    struct artsRemoteAtomicCompareAndSwapInArrayDbPacket * packet = pack;
    struct artsDb * db = artsRouteTableLookupItem(packet->dbGuid);
    internalAtomicCompareAndSwapInArrayDb(packet->dbGuid, packet->index, packet->oldValue, packet->newValue, packet->edtGuid, packet->slot, packet->epochGuid);
}

void artsRemoteEpochDelete(unsigned int rank, artsGuid_t epochGuid)
{
    struct artsRemoteEpochReqPacket packet;
    packet.epochGuid = epochGuid;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_EPOCH_DELETE_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleEpochDelete(void * pack)
{
    struct artsRemoteEpochReqPacket * packet = (struct artsRemoteEpochReqPacket*) pack;
    deleteEpoch(packet->epochGuid, NULL);
}

void artsDbMoveRequest(artsGuid_t dbGuid, unsigned int destRank)
{
    struct artsRemoteDbRequestPacket packet;
    packet.dbGuid = dbGuid;
    packet.mode = ARTS_DB_ONCE;
    packet.header.size = sizeof(packet);
    packet.header.messageType = ARTS_REMOTE_DB_MOVE_REQ_MSG;
    packet.header.rank = destRank;
    artsRemoteSendRequestAsync(artsGuidGetRank(dbGuid), (char *)&packet, sizeof(packet));
}

void artsDbMoveRequestHandle(void * pack)
{
    struct artsRemoteDbRequestPacket * packet = pack;
    artsDbMove(packet->dbGuid, packet->header.rank);
}

void artsRemoteHandleBufferSend(void * pack)
{
    struct artsRemoteMemoryMovePacket * packet = (struct artsRemoteMemoryMovePacket *) pack;
    unsigned int size = packet->header.size - sizeof(struct artsRemoteMemoryMovePacket);
    void * buffer = (void*)(packet+1);
    artsSetBuffer(packet->guid, buffer, size);
}

void artsRemoteSignalContext(unsigned int rank, uint64_t ticket)
{
    struct artsRemoteSignalContextPacket packet;
    packet.ticket = ticket;
    artsFillPacketHeader(&packet.header, sizeof(packet), ARTS_ATOMIC_ADD_ARRAYDB_MSG);
    artsRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void artsRemoteHandleSignalContext(void * pack)
{
    struct artsRemoteSignalContextPacket * packet = pack;
    artsSignalContext(packet->ticket);
}
