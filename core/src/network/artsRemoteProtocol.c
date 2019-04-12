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
#include "artsLinkList.h"
#include "artsGlobals.h"
#include "artsAtomics.h"
#include "artsIntrospection.h"
#include "artsRemoteProtocol.h"
#include "artsCounter.h"
#include <string.h>
#include "artsDebug.h"
#include "artsRemote.h"
#include "artsServer.h"
#define DPRINTF( ... )

struct outList
{
#ifdef COUNT
    uint64_t timeStamp;
#endif
    unsigned int offset;
    unsigned int length;
    unsigned int rank;
    void * payload;
    unsigned int payloadSize;
    unsigned int offsetPayload;
    void (*freeMethod)(void *);
};

unsigned int nodeListSize;

struct artsLinkList * outHead;
extern unsigned int ports;

__thread unsigned int threadStart;
__thread unsigned int threadStop;

__thread struct outList ** outResend;

#ifdef SEQUENCENUMBERS
unsigned int * seqNumLock = NULL;
uint64_t * seqNumber = NULL;
__thread uint64_t * lastOut;
__thread uint64_t * lastSent;
#endif

void partialSendStore(struct outList * out , unsigned int lengthRemaining)
{
    //PRINTF("Payload\n");
    if (out->payload == NULL)
    {
        out->offset = out->offset+(out->length-lengthRemaining);
        out->length = lengthRemaining;
    }
    else
    {
        unsigned int sent = out->length+out->payloadSize;
        sent -= lengthRemaining; 
        //PRINTF("Here %d\n", sent);
        if(sent >= out->length )
        {
            out->length=0;
            out->offsetPayload = out->offsetPayload + (out->payloadSize-lengthRemaining);
            out->payloadSize = lengthRemaining;
            //PRINTF("Here 1 %d %d %d %p\n", out->payloadSize, out->offsetPayload, lengthRemaining, out );
            
        }
        else
        {
            //PRINTF("Here 2a %d %d %d %d\n", out->payloadSize, out->length, sent, lengthRemaining);
            out->offset = out->offset + (out->length-(lengthRemaining-out->payloadSize));
            out->length = lengthRemaining - out->payloadSize;
            //PRINTF("Here 2b %d %d\n", out->offset, out->length);
        }
    }
    
}

void artsRemotSetThreadOutboundQueues(unsigned int start, unsigned int stop)
{
    threadStart = start;
    threadStop = stop;
    
    unsigned int size = stop - start;
    outResend = artsCalloc(sizeof(struct outList *)*size);
#ifdef SEQUENCENUMBERS
    lastOut = artsCalloc(sizeof(uint64_t)*artsGlobalRankCount);
    lastSent = artsCalloc(sizeof(uint64_t)*artsGlobalRankCount);
#endif
}

void outInit( unsigned int size )
{
    nodeListSize = size;
    outHead = artsLinkListGroupNew(size);
#ifdef SEQUENCENUMBERS
    seqNumber = artsCalloc(sizeof(uint64_t)*artsGlobalRankCount);
    seqNumLock = artsCalloc(sizeof(unsigned int)*size);
#endif
}

static inline void outInsertNode( struct outList * node, unsigned int length  )
{
#ifdef COUNT
    //This is for network queue sitting time...
//    node->timeStamp = artsExtGetTimeStamp();
#endif
    
    //int listId = node->rank*ports+artsThreadInfo.threadId%ports;
    long unsigned int listId;
    //mrand48_r (&artsThreadInfo.drand_buf, &listId);
    listId = node->rank*ports + artsThreadInfo.groupId % ports;
    struct artsLinkList * list = artsLinkListGet(outHead, listId);
    struct artsRemotePacket *packet = (struct artsRemotePacket *) (node + 1);
#ifdef SEQUENCENUMBERS
    artsLock(&seqNumLock[listId]);
    packet->seqNum = artsAtomicFetchAddU64(&seqNumber[node->rank], 1U);
    packet->seqRank = artsGlobalRankId;
#endif
    artsLinkListPushBack(list, node);
#ifdef SEQUENCENUMBERS
    artsUnlock(&seqNumLock[listId]);
    //PRINTF("Push: %u -> %u = %lu %p\n", artsGlobalRankId, node->rank, packet->seqNum, list);
#endif
//    nartsUpdatePerformanceMetric(artsNetworkQueuePush, artsThread, packet->size, false);
    artsUpdatePerformanceMetric(artsNetworkQueuePush, artsThread, 1, false);
}

static inline struct outList * outPopNode( unsigned int threadId, void ** freeMe  )
{
    struct outList * out;
    struct artsLinkList * list;
    list = artsLinkListGet(outHead, threadId);
    out = artsLinkListPopFront(list, freeMe);
    if(out)
    {
        struct artsRemotePacket *packet = (struct artsRemotePacket *) (out + 1);
#ifdef SEQUENCENUMBERS
        if(lastOut[packet->seqRank] && packet->seqNum != lastOut[packet->seqRank] + 1){
            DPRINTF("POP OUT OF ORDER %u -> %u %lu vs %lu %p\n", packet->seqRank, packet->rank, lastOut[packet->seqRank], packet->seqNum, list);
        }
        lastOut[packet->seqRank] = packet->seqNum;
        //DPRINTF("Pop : %u -> %u = %lu %p\n", artsGlobalRankId, out->rank, packet->seqNum, list);
#endif
//        artsUpdatePerformanceMetric(artsNetworkQueuePop, artsThread, packet->size, false);
    }
    artsUpdatePerformanceMetric(artsNetworkQueuePop, artsThread, 1, false);
    return out;
}

bool artsRemoteAsyncSend() 
{
    bool success = false;
    
    void * freeMe;
    unsigned int lengthRemaining;
    struct outList * out;
    
    bool sent = true;
    while (sent) 
    {
        sent = false;
        //Loop over our threads
        for (int i = threadStart; i < threadStop; i++) 
        {
            out = NULL; //For looping purposes...
            if(outResend[i-threadStart])//Checking failed sends?
                out = outResend[i-threadStart];
            else //Look for new messages
                out = outPopNode(i, &freeMe);
            
            if (out) 
            {
                #ifdef SEQUENCENUMBERS
                    struct artsRemotePacket *packet = (struct artsRemotePacket *) (out + 1);
                    if(lastSent[packet->seqRank] != packet->seqNum && packet->seqNum != lastSent[packet->seqRank]+1){
                        DPRINTF("SENT OUT OF ORDER %lu vs %lu\n", lastSent[packet->seqRank], packet->seqNum);
                    }
                    lastSent[packet->seqRank] = packet->seqNum;
                #endif
                if (!out->payload) 
                    lengthRemaining = artsRemoteSendRequest(out->rank, i, ((char*) (out + 1))+out->offset, out->length);
                else 
                {
                    lengthRemaining = artsRemoteSendPayloadRequest(out->rank, i, ((char*) (out + 1))+out->offset, out->length, ((char *)out->payload)+out->offsetPayload, out->payloadSize);
                    if (out->freeMethod && !lengthRemaining)
                        out->freeMethod(out->payload);
                }
                
                if(lengthRemaining == -1)
                    return false;
                else if(lengthRemaining)
                {
                    partialSendStore(out,lengthRemaining);
                    outResend[i-threadStart] = out;
                }
                else
                {
                    struct artsRemotePacket *packet = (struct artsRemotePacket *) (out + 1);
                    if(packet->messageType!=ARTS_REMOTE_METRIC_UPDATE_MSG)
                        artsUpdatePerformanceMetric(artsNetworkSendBW, artsThread, packet->size, false);
                    
                    outResend[i-threadStart] = NULL;
                    artsLinkListDeleteItem(out);
                }
                
                sent = true;
                success = true;
            }
        }
    }
    return success;
}

static inline void selfSendCheck( unsigned int rank )
{
    if(rank == artsGlobalRankId || rank >= artsGlobalRankCount  )
    {
        PRINTF("Send error rank stack trace: %u of %u\n", rank, artsGlobalRankCount); 
        artsDebugPrintStack();
        artsDebugGenerateSegFault();
    }
}

static inline void sizeSendCheck( unsigned int size )
{
    if( size == 0 || size > 1073741824)
    {
        PRINTF("Send error size stack trace: %d\n", size); 
        artsDebugPrintStack();
        artsDebugGenerateSegFault();
    }
}

void artsRemoteSendRequestAsync(int rank, char * message, unsigned int length)
{
    selfSendCheck(rank); 
    struct outList * next = artsLinkListNewItem(length + sizeof(struct outList));
    next->offset = 0;
    next->offsetPayload = 0;
    next->length = length;
    next->rank = rank;
    next->payload=NULL;
    memcpy(next+1, message, length);
    outInsertNode(next, length+sizeof(struct outList));
}

void artsRemoteSendRequestPayloadAsync(int rank, char * message, unsigned int length, char * payload, unsigned int size)
{
    selfSendCheck(rank); 
    sizeSendCheck(length);
    sizeSendCheck(size);
    struct outList * next = artsLinkListNewItem(length + sizeof(struct outList));
    next->offset = 0;
    next->offsetPayload = 0;
    next->length = length;
    next->rank = rank;
    next->payload = payload;
    next->freeMethod = NULL;
    next->payloadSize = size;
    memcpy(next+1, message, length);
    outInsertNode(next, length+sizeof(struct outList));
}

void artsRemoteSendRequestPayloadAsyncFree(int rank, char * message, unsigned int length, char * payload, unsigned int offset, unsigned int size, void(*freeMethod)(void*))
{
    selfSendCheck(rank); 
    sizeSendCheck(length);
    sizeSendCheck(size);
    struct outList * next = artsLinkListNewItem(length + sizeof(struct outList));
    next->offset = 0;
    next->offsetPayload = offset;
    next->length = length;
    next->rank = rank;
    next->payload = payload;
    next->payloadSize = size;
    next->freeMethod = freeMethod;
    memcpy(next+1, message, length);
    outInsertNode(next, length+sizeof(struct outList));
}
