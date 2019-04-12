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
#ifndef ARTSREMOTEPROTOCOL_H
#define ARTSREMOTEPROTOCOL_H
#ifdef __cplusplus
extern "C" {
#endif
#define SEQUENCENUMBERS 1

//TODO: Switch to an enum
enum artsServerMessageType
{
    ARTS_REMOTE_SHUTDOWN_MSG=0,
    ARTS_REMOTE_EDT_SIGNAL_MSG,
    ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG,
    ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG,
    ARTS_REMOTE_ADD_DEPENDENCE_MSG,
    ARTS_REMOTE_DB_REQUEST_MSG,
    ARTS_REMOTE_DB_SEND_MSG,
    ARTS_REMOTE_INVALIDATE_DB_MSG,
    ARTS_REMOTE_DB_UPDATE_GUID_MSG,
    ARTS_REMOTE_DB_UPDATE_MSG,
    ARTS_REMOTE_DB_DESTROY_MSG,
    ARTS_REMOTE_DB_DESTROY_FORWARD_MSG,
    ARTS_REMOTE_DB_CLEAN_FORWARD_MSG,
    ARTS_REMOTE_DB_MOVE_REQ_MSG,      
    ARTS_REMOTE_EDT_MOVE_MSG,
    ARTS_REMOTE_EVENT_MOVE_MSG,
    ARTS_REMOTE_DB_MOVE_MSG,
    ARTS_REMOTE_PINGPONG_TEST_MSG,
    ARTS_REMOTE_METRIC_UPDATE_MSG,
    ARTS_REMOTE_DB_FULL_REQUEST_MSG,
    ARTS_REMOTE_DB_FULL_SEND_MSG,
    ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG,
    ARTS_REMOTE_GET_FROM_DB_MSG,
    ARTS_REMOTE_PUT_IN_DB_MSG,
    ARTS_REMOTE_SEND_MSG,
    ARTS_EPOCH_INIT_MSG,
    ARTS_EPOCH_INIT_POOL_MSG,
    ARTS_EPOCH_REQ_MSG, 
    ARTS_EPOCH_SEND_MSG,
    ARTS_EPOCH_DELETE_MSG,
    ARTS_ATOMIC_ADD_ARRAYDB_MSG,
    ARTS_ATOMIC_CAS_ARRAYDB_MSG,
    ARTS_REMOTE_BUFFER_SEND_MSG,
    ARTS_REMOTE_CONTEXT_SIG_MSG
};

//Header
struct __attribute__ ((__packed__)) artsRemotePacket
{
    unsigned int messageType;
    unsigned int size;
    unsigned int rank;
#ifdef SEQUENCENUMBERS
    unsigned int seqRank;
    uint64_t seqNum;
#endif
#ifdef COUNT
    uint64_t timeStamp;
    uint64_t procTimeStamp;
#endif
};

struct artsRemoteGuidOnlyPacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteInvalidateDbPacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteUpdateDbGuidPacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteUpdateDbPacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteMemoryMovePacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteAddDependencePacket
{
    struct artsRemotePacket header;
    artsGuid_t source;
    artsGuid_t destination;
    uint32_t slot;
    artsType_t mode;
    unsigned int destRoute;
};

struct __attribute__ ((__packed__)) artsRemoteEdtSignalPacket
{
    struct artsRemotePacket header;
    artsGuid_t edt;
    artsGuid_t db;
    uint32_t slot;
    artsType_t mode;
    //-------------------------Routing info
    unsigned int dbRoute;
};

struct __attribute__ ((__packed__)) artsRemoteEventSatisfySlotPacket
{
    struct artsRemotePacket header;
    artsGuid_t event;
    artsGuid_t db;
    uint32_t slot;
    //-------------------------Routing info
    unsigned int dbRoute;
};

struct __attribute__ ((__packed__)) artsRemoteDbRequestPacket
{
    struct artsRemotePacket header;
    artsGuid_t dbGuid;
    artsType_t mode;
};

struct __attribute__ ((__packed__)) artsRemoteDbSendPacket
{
    struct artsRemotePacket header;
};

struct __attribute__ ((__packed__)) artsRemoteDbFullRequestPacket
{
    struct artsRemotePacket header;
    artsGuid_t dbGuid;
    void * edt;
    unsigned int slot;
    artsType_t mode;
};

struct __attribute__ ((__packed__)) artsRemoteDbFullSendPacket
{
    struct artsRemotePacket header;
    struct artsEdt * edt;
    unsigned int slot;
    artsType_t mode;
};

struct __attribute__ ((__packed__)) artsRemoteMetricUpdate
{
    struct artsRemotePacket header;
    int type;
    uint64_t timeStamp;
    uint64_t toAdd;
    bool sub;
};

struct __attribute__ ((__packed__)) artsRemoteGetPutPacket
{
    struct artsRemotePacket header;
    artsGuid_t edtGuid;
    artsGuid_t dbGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int offset;
    unsigned int size;
};

struct __attribute__ ((__packed__)) artsRemoteSignalEdtWithPtrPacket
{
    struct artsRemotePacket header;
    artsGuid_t edtGuid;
    artsGuid_t dbGuid;
    unsigned int size;
    unsigned int slot;
};

struct __attribute__ ((__packed__)) artsRemoteSend
{
    struct artsRemotePacket header;
    sendHandler_t funPtr;
};

struct __attribute__ ((__packed__)) artsRemoteEpochInitPacket
{
    struct artsRemotePacket header;
    artsGuid_t epochGuid;
    artsGuid_t edtGuid;
    unsigned int slot;
};

struct __attribute__ ((__packed__)) artsRemoteEpochInitPoolPacket
{
    struct artsRemotePacket header;
    unsigned int poolSize;
    artsGuid_t startGuid;
    artsGuid_t poolGuid;
};

struct __attribute__ ((__packed__)) artsRemoteEpochReqPacket
{
    struct artsRemotePacket header;
    artsGuid_t epochGuid;
};

struct __attribute__ ((__packed__)) artsRemoteEpochSendPacket
{
    struct artsRemotePacket header;
    artsGuid_t epochGuid;
    unsigned int active;
    unsigned int finish;
};

struct __attribute__ ((__packed__)) artsRemoteAtomicAddInArrayDbPacket
{
    struct artsRemotePacket header;
    artsGuid_t dbGuid;
    artsGuid_t edtGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int index;
    unsigned int toAdd;
};

struct __attribute__ ((__packed__)) artsRemoteAtomicCompareAndSwapInArrayDbPacket
{
    struct artsRemotePacket header;
    artsGuid_t dbGuid;
    artsGuid_t edtGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int index;
    unsigned int oldValue;
    unsigned int newValue;
};

struct __attribute__ ((__packed__)) artsRemoteSignalContextPacket
{
    struct artsRemotePacket header;
    uint64_t ticket;
};

void outInit( unsigned int size );
bool artsRemoteAsyncSend();
void artsRemoteSendRequestAsync( int rank, char * message, unsigned int length );
void artsRemoteSendRequestPayloadAsync( int rank, char * message, unsigned int length, char * payload, unsigned int size );
void artsRemoteSendRequestPayloadAsyncFree( int rank, char * message, unsigned int length, char * payload, unsigned int offset, unsigned int size, void(*freeMethod)(void*));
void artsRemotSetThreadOutboundQueues(unsigned int start, unsigned int stop);
#ifdef __cplusplus
}
#endif

#endif
