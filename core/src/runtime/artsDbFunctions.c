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

#include "artsDbFunctions.h"
#include "artsAtomics.h"
#include "artsGuid.h"
#include "artsGlobals.h"
#include "artsRemoteFunctions.h"
#include "artsRouteTable.h"
#include "artsOutOfOrder.h"
#include "artsDbList.h"
#include "artsDebug.h"
#include "artsEdtFunctions.h"
#include "artsTerminationDetection.h"
#include "artsCounter.h"
#include "artsIntrospection.h"

#define DPRINTF( ... )

artsTypeName;

void artsDbCreateInternal(artsGuid_t guid, void *addr, uint64_t size, uint64_t packetSize, artsType_t mode)
{
    struct artsHeader *header = (struct artsHeader*)addr;
    header->type = mode;
    header->size = packetSize;

    struct artsDb * dbRes = (struct artsDb *)header;
    dbRes->guid = guid;
    if(mode != ARTS_DB_PIN)
    {
        dbRes->dbList = artsNewDbList();
    }
}

artsGuid_t artsDbCreateRemote(unsigned int route, uint64_t size, artsType_t mode)
{
    ARTSEDTCOUNTERTIMERSTART(dbCreateCounter);
    artsGuid_t guid = artsGuidCreateForRank(route, mode);
    void * ptr = artsMalloc(sizeof(struct artsDb));
    struct artsDb * db = (struct artsDb*) ptr;
    db->header.size = size + sizeof(struct artsDb);
    db->dbList = (mode == ARTS_DB_PIN) ? (void*)0 : (void*)1;
    
    artsRemoteMemoryMove(route, guid, ptr, sizeof(struct artsDb), ARTS_REMOTE_DB_SEND_MSG, artsFree);
    ARTSEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
}

//Creates a local DB only
artsGuid_t artsDbCreate(void **addr, uint64_t size, artsType_t mode)
{
    ARTSEDTCOUNTERTIMERSTART(dbCreateCounter);
    artsGuid_t guid = NULL_GUID;
    unsigned int dbSize = size + sizeof(struct artsDb);

    ARTSSETMEMSHOTTYPE(artsDbMemorySize);
    void * ptr = artsMalloc(dbSize);
    ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);
    if(ptr)
    {
        guid = artsGuidCreateForRank(artsGlobalRankId, mode);
        artsDbCreateInternal(guid, ptr, size, dbSize, mode);
        //change false to true to force a manual DB delete
        artsRouteTableAddItem(ptr, guid, artsGlobalRankId, false);
        *addr = (void*)((struct artsDb *) ptr + 1);
    }
    ARTSEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
    return guid;
}

//Guid must be for a local DB only
void * artsDbCreateWithGuid(artsGuid_t guid, uint64_t size)
{
    ARTSEDTCOUNTERTIMERSTART(dbCreateCounter);
    artsType_t mode = artsGuidGetType(guid);
    void * ptr = NULL;
    if(artsIsGuidLocal(guid))
    {
        unsigned int dbSize = size + sizeof(struct artsDb);
        
        ARTSSETMEMSHOTTYPE(artsDbMemorySize);
        ptr = artsMalloc(dbSize);
        ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);
        if(ptr)
        {
            artsDbCreateInternal(guid, ptr, size, dbSize, mode);
            if(artsRouteTableAddItemRace(ptr, guid, artsGlobalRankId, false))
                artsRouteTableFireOO(guid, artsOutOfOrderHandler);
            ptr = (void*)((struct artsDb *) ptr + 1);
        }
    }
    ARTSEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
    return ptr;
}

void * artsDbCreateWithGuidAndData(artsGuid_t guid, void * data, uint64_t size)
{
    ARTSEDTCOUNTERTIMERSTART(dbCreateCounter);
    artsType_t mode = artsGuidGetType(guid);
    void * ptr = NULL;
    if(artsIsGuidLocal(guid))
    {
        unsigned int dbSize = size + sizeof(struct artsDb);
        
        ARTSSETMEMSHOTTYPE(artsDbMemorySize);
        ptr = artsMalloc(dbSize);
        ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);
        
        if(ptr)
        {
            artsDbCreateInternal(guid, ptr, size, dbSize, mode);
            void * dbData = (void*)((struct artsDb *) ptr + 1);
            memcpy(dbData, data, size);
            if(artsRouteTableAddItemRace(ptr, guid, artsGlobalRankId, false))
                artsRouteTableFireOO(guid, artsOutOfOrderHandler);
            ptr = dbData;
        }
    }
    ARTSEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
    return ptr;
}

void * artsDbResizePtr(struct artsDb * dbRes, unsigned int size, bool copy)
{
    if(dbRes)
    {
        unsigned int oldSize = dbRes->header.size;
        unsigned int newSize = size + sizeof(struct artsDb);
        ARTSSETMEMSHOTTYPE(artsDbMemorySize);
        struct artsDb *  ptr = artsCalloc(size + sizeof(struct artsDb));
        ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);
        if(ptr)
        {
            if(copy)
                memcpy(ptr, dbRes, oldSize);
            else
                memcpy(ptr, dbRes, sizeof(struct artsDb));
            artsFree(dbRes);
            ptr->header.size = size + sizeof(struct artsDb);
            return (void*)(ptr+1);
        }
    }
    return NULL;
}

//Must be in write mode (or only copy) to update and alloced (no NO_ACQUIRE nonsense), otherwise will be racy...
void * artsDbResize(artsGuid_t guid, unsigned int size, bool copy)
{
    struct artsDb * dbRes = artsRouteTableLookupItem(guid);
    void * ptr = artsDbResizePtr(dbRes, size, copy);
    if(ptr)
    {
        dbRes = ((struct artsDb *)ptr) - 1;
//        artsRouteTableUpdateItem(dbRes, guid, artsGlobalRankId);
    }
    return ptr;
}

void artsDbMove(artsGuid_t dbGuid, unsigned int rank)
{
    unsigned int guidRank = artsGuidGetRank(dbGuid);
    if(guidRank != rank)
    {
        if(guidRank != artsGlobalRankId)
            artsDbMoveRequest(dbGuid, rank);
        else
        {
            struct artsDb * dbRes = artsRouteTableLookupItem(dbGuid);
            if(dbRes)
                artsRemoteMemoryMove(rank, dbGuid, dbRes, dbRes->header.size, ARTS_REMOTE_DB_MOVE_MSG, artsFree);
            else
                artsOutOfOrderDbMove(dbGuid, rank);
        }
    }
}

void artsDbDestroy(artsGuid_t guid)
{
    struct artsDb * dbRes = artsRouteTableLookupItem(guid);
    if(dbRes!=NULL)
    {
        artsRemoteDbDestroy(guid, artsGlobalRankId, 0);
        artsFree(dbRes);
        artsRouteTableRemoveItem(guid);
    }
    else
        artsRemoteDbDestroy(guid, artsGlobalRankId, 0);
}

void artsDbDestroySafe(artsGuid_t guid, bool remote)
{
    struct artsDb * dbRes = artsRouteTableLookupItem(guid);
    if(dbRes!=NULL)
    {
        if(remote)
            artsRemoteDbDestroy(guid, artsGlobalRankId, 0);
        artsFree(dbRes);
        artsRouteTableRemoveItem(guid);
    }
    else if(remote)
        artsRemoteDbDestroy(guid, artsGlobalRankId, 0);
}

/**********************DB MEMORY MODEL*************************************/

//Side Effects:/ edt depcNeeded will be incremented, ptr will be updated,
//  and launches out of order handleReadyEdt
//Returns false on out of order and true otherwise
void acquireDbs(struct artsEdt * edt)
{
    artsEdtDep_t * depv = (artsEdtDep_t *)(((uint64_t *)(edt + 1)) + edt->paramc);
    edt->depcNeeded = edt->depc + 1;
    for(int i=0; i<edt->depc; i++)
    {
        DPRINTF("MODE: %s\n", getTypeName(depv[i].mode));
        if(depv[i].guid && depv[i].ptr == NULL)
        {
            struct artsDb * dbFound = NULL;
            int owner = artsGuidGetRank(depv[i].guid);
            switch(depv[i].mode)
            {
                //This case assumes that the guid exists only on the owner
                case ARTS_DB_ONCE:
                {
                    if(owner != artsGlobalRankId)
                    {
                        artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i);
                        artsDbMove(depv[i].guid, artsGlobalRankId);
                        break;
                    }
                    //else fall through to the local case :-p
                }
                case ARTS_DB_ONCE_LOCAL:
                {
                    struct artsDb * dbTemp = artsRouteTableLookupItem(depv[i].guid);
                    if(dbTemp)
                    {
                        dbFound = dbTemp;
                    artsAtomicSub(&edt->depcNeeded, 1U);
                    }
                    else
                        artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i);
                    break;
                }
                case ARTS_DB_PIN:
                {
//                    if(artsIsGuidLocal(depv[i].guid))
//                    {
                        int validRank = -1;
                        struct artsDb * dbTemp = artsRouteTableLookupDb(depv[i].guid, &validRank);
                        if(dbTemp)
                        {
                            dbFound = dbTemp;
                            artsAtomicSub(&edt->depcNeeded, 1U);
                        }
                        else
                        {
                            artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i);
                        }
//                    }
//                    else
//                    {
//                        PRINTF("Cannot acquire DB %lu because it is pinned on %u\n", depv[i].guid, artsGuidGetRank(depv[i].guid));
//                        depv[i].ptr = NULL;
//                        artsAtomicSub(&edt->depcNeeded, 1U);
//                    }
                    break;
                }
                case ARTS_DB_READ:
                case ARTS_DB_WRITE:
                    if(owner == artsGlobalRankId) //Owner Rank
                    {
                        int validRank = -1;
                        struct artsDb * dbTemp = artsRouteTableLookupDb(depv[i].guid, &validRank);
                        if(dbTemp) //We have found an entry
                        {
                        if(artsAddDbDuplicate(dbTemp, artsGlobalRankId, edt, i, depv[i].mode))
                        {
                            if(validRank == artsGlobalRankId) //Owner rank and we have the valid copy
                            {
                                dbFound = dbTemp;
                                artsAtomicSub(&edt->depcNeeded, 1U);
                            }
                            else //Owner rank but someone else has valid copy
                            {
                                if(depv[i].mode == ARTS_DB_READ)
                                    artsRemoteDbRequest(depv[i].guid, validRank, edt, i, depv[i].mode, true);
                                else
                                    artsRemoteDbFullRequest(depv[i].guid, validRank, edt, i, depv[i].mode);
                            }
                        }
//                            else  //We can't read right now due to an exclusive access or cdag write in progress
//                            {
//                                PRINTF("############### %lu Queue in frontier\n", depv[i].guid);
//                            }
                    }
                        else //The Db hasn't been created yet
                        {
                            DPRINTF("%lu out of order request\n", depv[i].guid);
                            artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i);
                        }
                    }
                    else
                    {
                        int validRank = -1;
                        struct artsDb * dbTemp = artsRouteTableLookupDb(depv[i].guid, &validRank);
                        if(dbTemp) //We have found an entry
                        {
                            dbFound = dbTemp;
                            artsAtomicSub(&edt->depcNeeded, 1U);
                        }

                        if(depv[i].mode == ARTS_DB_WRITE)
                        {
                            //We can't aggregate read requests for cdag write
                            artsRemoteDbFullRequest(depv[i].guid, owner, edt, i, depv[i].mode);
                        }
                        else if(!dbTemp)
                        {
                            //We can aggregate read requests for reads
                            artsRemoteDbRequest(depv[i].guid, owner, edt, i, depv[i].mode, true);
                        }
                    }
                    break;

                case ARTS_NULL:
                default:
                    artsAtomicSub(&edt->depcNeeded, 1U);
                    break;
            }

            if(dbFound)
            {
                depv[i].ptr = dbFound + 1;
            }
        }
        else
        {
            artsAtomicSub(&edt->depcNeeded, 1U);
        }
    }
}

void prepDbs(unsigned int depc, artsEdtDep_t * depv)
{
    for(unsigned int i=0; i<depc; i++)
    {
        if(   depv[i].guid != NULL_GUID &&
              depv[i].mode == ARTS_DB_WRITE )
        {
            artsRemoteUpdateRouteTable(depv[i].guid, -1);
        }
    }
}

void releaseDbs(unsigned int depc, artsEdtDep_t * depv)
{
    for(int i=0; i<depc; i++)
    {
//        PRINTF(">>>>>>>>>>>>>>>>>>>>>>> %lu\n", depv[i].guid);
        if(   depv[i].guid != NULL_GUID &&
              depv[i].mode == ARTS_DB_WRITE )
        {
            //Signal we finished and progress frontier
            if(artsGuidGetRank(depv[i].guid) == artsGlobalRankId)
            {
                struct artsDb * db = ((struct artsDb *)depv[i].ptr - 1);
                artsProgressFrontier(db, artsGlobalRankId);
            }
            else
            {
                artsRemoteUpdateDb(depv[i].guid, false);
            }
//            artsRouteTableReturnDb(depv[i].guid, false);
        }
        else if(depv[i].mode == ARTS_DB_ONCE_LOCAL || depv[i].mode == ARTS_DB_ONCE)
        {
            artsRouteTableInvalidateItem(depv[i].guid);
        }
        else if(depv[i].mode == ARTS_PTR)
        {
            artsFree(depv[i].ptr);
        }
        else 
        {
            if(artsRouteTableReturnDb(depv[i].guid, depv[i].mode != ARTS_DB_PIN))
                PRINTF("FREED A COPY!\n");
        }
    }
}

bool artsAddDbDuplicate(struct artsDb * db, unsigned int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode)
{
    bool write = (mode==ARTS_DB_WRITE);
    bool exclusive = false;
    return artsPushDbToList(db->dbList, rank, write, exclusive, artsGuidGetRank(db->guid) == rank, false, edt, slot, mode);
}

void internalGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank)
{
    if(rank==artsGlobalRankId)
    {
        struct artsDb * db = artsRouteTableLookupItem(dbGuid);
        if(db)
        {
            void * data = (void*)(((char*) (db+1)) + offset);
            void * ptr = artsMalloc(size);
            memcpy(ptr, data, size);
            DPRINTF("GETTING: %u From: %p\n", *(unsigned int*)ptr, data);
            artsSignalEdtPtr(edtGuid, slot, ptr, size);
            artsUpdatePerformanceMetric(artsGetBW, artsThread, size, false);
        }
        else
        {
            artsOutOfOrderGetFromDb(edtGuid, dbGuid, slot, offset, size);
        }
    }
    else
    {
        DPRINTF("Sending to %u\n", rank);
        artsRemoteGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
    }
}

void artsGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    ARTSEDTCOUNTERTIMERSTART(getDbCounter);
    unsigned int rank = artsGuidGetRank(dbGuid);
    internalGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
    ARTSEDTCOUNTERTIMERENDINCREMENT(getDbCounter);
}

void artsGetFromDbAt(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank)
{
    ARTSEDTCOUNTERTIMERSTART(getDbCounter);
    internalGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
    ARTSEDTCOUNTERTIMERENDINCREMENT(getDbCounter);
}

void internalPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid, unsigned int rank)
{
    if(rank==artsGlobalRankId)
    {
        struct artsDb * db = artsRouteTableLookupItem(dbGuid);
        if(db)
        {
            //Do this so when we increment finished we can check the term status
            incrementQueueEpoch(epochGuid);
            globalShutdownGuidIncQueue();
            void * data = (void*)(((char*) (db+1)) + offset);
            memcpy(data, ptr, size);
            DPRINTF("PUTTING %u From: %p\n", *((unsigned int *)data), data);
            if(edtGuid)
            {
                artsSignalEdt(edtGuid, slot, dbGuid);
            }
            DPRINTF("FINISHING PUT %lu\n", epochGuid);
            incrementFinishedEpoch(epochGuid);
            globalShutdownGuidIncFinished();
            artsUpdatePerformanceMetric(artsPutBW, artsThread, size, false);
        }
        else
        {
            void * cpyPtr = artsMalloc(size);
            memcpy(cpyPtr, ptr, size);
            artsOutOfOrderPutInDb(cpyPtr, edtGuid, dbGuid, slot, offset, size, epochGuid);
        }
    }
    else
    {
        void * cpyPtr = artsMalloc(size);
        memcpy(cpyPtr, ptr, size);
        artsRemotePutInDb(cpyPtr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
    }
}

void artsPutInDbAt(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank)
{
    ARTSEDTCOUNTERTIMERSTART(putDbCounter);
    artsGuid_t epochGuid = artsGetCurrentEpochGuid();
    DPRINTF("EPOCH %lu\n", epochGuid);
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    internalPutInDb(ptr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
    ARTSEDTCOUNTERTIMERENDINCREMENT(putDbCounter);
}

void artsPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    ARTSEDTCOUNTERTIMERSTART(putDbCounter);
    unsigned int rank = artsGuidGetRank(dbGuid);
    artsGuid_t epochGuid = artsGetCurrentEpochGuid();
    DPRINTF("EPOCH %lu\n", epochGuid);
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    internalPutInDb(ptr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
    ARTSEDTCOUNTERTIMERENDINCREMENT(putDbCounter);
}

void artsPutInDbEpoch(void * ptr, artsGuid_t epochGuid, artsGuid_t dbGuid, unsigned int offset, unsigned int size)
{
    ARTSEDTCOUNTERTIMERSTART(putDbCounter);
    unsigned int rank = artsGuidGetRank(dbGuid);
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    internalPutInDb(ptr, NULL_GUID, dbGuid, 0, offset, size, epochGuid, rank);
    ARTSEDTCOUNTERTIMERENDINCREMENT(putDbCounter);
}
