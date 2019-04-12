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
#include "arts.h"
#include "artsRemoteProtocol.h"
#include "artsGlobals.h"
#include "artsRuntime.h"
#include "artsServer.h"
#include "artsRemoteFunctions.h"
#include "artsEdtFunctions.h"
#include "artsEventFunctions.h"
#include "artsRouteTable.h"
#include "artsRemote.h"
#include "artsAtomics.h"
#include "artsIntrospection.h"
#include <unistd.h>
//#include "artsRemote.h"
#define DPRINTF( ... )
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

#define EDT_MUG_SIZE 32 

bool artsGlobalIWillPrint=false;
FILE * lockPrintFile;
extern bool serverEnd;

#ifdef SEQUENCENUMBERS
uint64_t * recSeqNumbers;
#endif

void artsRemoteTryToBecomePrinter()
{
    lockPrintFile = fopen(".artsPrintLock", "wx");
    if(lockPrintFile)
        artsGlobalIWillPrint=true;
}

void artsRemoteTryToClosePrinter()
{
    if(lockPrintFile)
        fclose(lockPrintFile);
    remove(".artsPrintLock");
}

void artsRemoteShutdown()
{
    artsLLServerShutdown();
}

void artsServerSetup( struct artsConfig * config)
{
    //ASYNC Message Deque Init
    artsLLServerSetup(config);
    outInit(artsGlobalRankCount*config->ports);
    #ifdef SEQUENCENUMBERS
    recSeqNumbers = artsCalloc(sizeof(uint64_t)*artsGlobalRankCount);
    #endif
}

void artsServerProcessPacket(struct artsRemotePacket * packet)
{
    if(packet->messageType!=ARTS_REMOTE_METRIC_UPDATE_MSG ||
       packet->messageType!=ARTS_REMOTE_SHUTDOWN_MSG)
    {
        artsUpdatePerformanceMetric(artsNetworkRecieveBW, artsThread, packet->size, false);
        artsUpdatePerformanceMetric(artsFreeBW + packet->messageType, artsThread, packet->size, false);
        artsUpdatePacketInfo(packet->size);
    }
#ifdef SEQUENCENUMBERS
    uint64_t expSeqNumber = __sync_fetch_and_add(&recSeqNumbers[packet->seqRank], 1U);
    if(expSeqNumber != packet->seqNum)
    {
        DPRINTF("MESSAGE RECIEVED OUT OF ORDER exp: %lu rec: %lu source: %u type: %d\n", expSeqNumber, packet->seqNum, packet->rank, packet->messageType);
    }
//    else
//        PRINTF("Recv: %lu -> %lu = %lu\n", packet->seqRank, artsGlobalRankId, packet->seqNum);
#endif
    
    switch(packet->messageType)
    {
        case ARTS_REMOTE_SHUTDOWN_MSG:
        {
            DPRINTF("Remote Shutdown Request\n");           
            artsRuntimeStop();
            break;
        }
        case ARTS_REMOTE_EDT_SIGNAL_MSG:
        {
            struct artsRemoteEdtSignalPacket *pack = (struct artsRemoteEdtSignalPacket *)(packet);
            internalSignalEdt(pack->edt, pack->slot, pack->db, pack->mode, NULL, 0);
            break;
        }   
        case ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG:
        {
            struct artsRemoteEventSatisfySlotPacket *pack = (struct artsRemoteEventSatisfySlotPacket *)(packet);
            artsEventSatisfySlot(pack->event, pack->db, pack->slot);
            break;
        }   
        case ARTS_REMOTE_DB_REQUEST_MSG:
        {
            struct artsRemoteDbRequestPacket *pack = (struct artsRemoteDbRequestPacket *)(packet);
            if(packet->size != sizeof(*pack))
                PRINTF("Error dbpacket insanity\n");
            artsRemoteDbSend(pack);
            break;
        }
        case ARTS_REMOTE_DB_SEND_MSG:  
        {
            DPRINTF("Remote Db Recieved\n");
            struct artsRemoteDbSendPacket *pack = (struct artsRemoteDbSendPacket *)(packet);
            artsRemoteHandleDbRecieved(pack);
            break;
        }
        case ARTS_REMOTE_ADD_DEPENDENCE_MSG:
        {
            DPRINTF("Dependence Recieved\n");
            struct artsRemoteAddDependencePacket *pack = (struct artsRemoteAddDependencePacket *)(packet);
            artsAddDependence( pack->source, pack->destination, pack->slot);
            break;
        }   
        
        case ARTS_REMOTE_INVALIDATE_DB_MSG:
        {
            DPRINTF("DB Invalidate Recieved\n");
            artsRemoteHandleInvalidateDb( packet );
            break;
        }
        case ARTS_REMOTE_DB_FULL_REQUEST_MSG:
        {
            struct artsRemoteDbFullRequestPacket *pack = (struct artsRemoteDbFullRequestPacket *)(packet);
            artsRemoteDbFullSend(pack);
            break;
        }
        case ARTS_REMOTE_DB_FULL_SEND_MSG:
        {
            struct artsRemoteDbFullSendPacket * pack = (struct artsRemoteDbFullSendPacket *)(packet);
            artsRemoteHandleDbFullRecieved(pack);
            break;
        }
        case ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG:
        {
            artsRemoteHandleSendAlreadyLocal(packet);
            break;
        }
        case ARTS_REMOTE_DB_DESTROY_MSG:
        {
            DPRINTF("DB Destroy Recieved\n");
            artsRemoteHandleDbDestroy( packet );
            break;
        }
        case ARTS_REMOTE_DB_DESTROY_FORWARD_MSG:
        {
            DPRINTF("DB Destroy Forward Recieved\n");
            artsRemoteHandleDbDestroyForward( packet );
            break;
        }
        case ARTS_REMOTE_DB_CLEAN_FORWARD_MSG:
        {
            DPRINTF("DB Clean Forward Recieved\n");
            artsRemoteHandleDbCleanForward( packet );
            break;
        }
        case ARTS_REMOTE_DB_UPDATE_GUID_MSG:
        {
            DPRINTF("DB Guid Update Recieved\n");
            artsRemoteHandleUpdateDbGuid( packet );
            break;
        }
        case ARTS_REMOTE_EDT_MOVE_MSG:
        {
            DPRINTF("EDT Move Recieved\n");
            artsRemoteHandleEdtMove( packet );
            break;
        }
        case ARTS_REMOTE_DB_MOVE_MSG:
        {
            DPRINTF("DB Move Recieved\n");
            artsRemoteHandleDbMove( packet );
            break;
        }
        case ARTS_REMOTE_DB_UPDATE_MSG:
        {
            artsRemoteHandleUpdateDb(packet);
            break;
        }
        case ARTS_REMOTE_EVENT_MOVE_MSG:
        {
            artsRemoteHandleEventMove(packet);
            break;
        }
        case ARTS_REMOTE_METRIC_UPDATE_MSG:
        {
            
            struct artsRemoteMetricUpdate * pack = (struct artsRemoteMetricUpdate *) (packet);
            DPRINTF("Metric update Recieved %u -> %d %ld\n", artsGlobalRankId, pack->type, pack->toAdd);
            artsHandleRemoteMetricUpdate(pack->type, artsSystem, pack->toAdd, pack->sub);
            break;    
        }
        case ARTS_REMOTE_GET_FROM_DB_MSG:
        {
            artsRemoteHandleGetFromDb(packet);
            break;
        }
        case ARTS_REMOTE_PUT_IN_DB_MSG:
        {
            artsRemoteHandlePutInDb(packet);
            break;
        }
        case ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG:
        {
            artsRemoteHandleSignalEdtWithPtr(packet);
            break;
        }
        case ARTS_REMOTE_SEND_MSG:
        {
            artsRemoteHandleSend(packet);
            break;
        }
        case ARTS_EPOCH_INIT_MSG:
        {
            artsRemoteHandleEpochInitSend(packet);
            break;
        }
        case ARTS_EPOCH_REQ_MSG:
        {
            artsRemoteHandleEpochReq(packet);
            break;
        }
        case ARTS_EPOCH_SEND_MSG:
        {
            artsRemoteHandleEpochSend(packet);
            break;
        }
        case ARTS_ATOMIC_ADD_ARRAYDB_MSG:
        {
            artsRemoteHandleAtomicAddInArrayDb(packet);
            break;
        }
        case ARTS_ATOMIC_CAS_ARRAYDB_MSG:
        {
            artsRemoteHandleAtomicCompareAndSwapInArrayDb(packet);
            break;
        }
        case ARTS_EPOCH_INIT_POOL_MSG:
        {
            artsRemoteHandleEpochInitPoolSend(packet);
            break;
        }
        case ARTS_EPOCH_DELETE_MSG:
        {
            artsRemoteHandleEpochDelete(packet);
            break;
        }
        case ARTS_REMOTE_BUFFER_SEND_MSG:
        {
            artsRemoteHandleBufferSend(packet);
            break;
        }
        case ARTS_REMOTE_DB_MOVE_REQ_MSG:
        {
            artsDbMoveRequestHandle(packet);
            break;
        }
        default:
        {
            PRINTF("Unknown Packet %d %d %d\n", packet->messageType, packet->size, packet->rank);
            artsShutdown();
            artsRuntimeStop();
        }
    }
}


