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

/*
 * arts_tMT.c
 *
 *  Created on: March 30, 2018
 *      Author: Andres Marquez (@awmm)
 *
 *
 * This file is subject to the license agreement located in the file LICENSE
 * and cannot be distributed without it. This notice cannot be
 * removed or modified.
 *
 *
 *
 */

#define PT_CONTEXTS  // maintain contexts via PThreads

#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <inttypes.h>
#define __USE_GNU
#include <string.h>

#include "artsGlobals.h"
#include "artsAtomics.h"
#include "artsDeque.h"
#include "artsDbFunctions.h"
#include "artsEdtFunctions.h"
#include "artsRemoteFunctions.h"
#include "artsThreads.h"

#include "artsTMT.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )
#define ONLY_ONE_THREAD 
//while(!artsTestStateOneLeft(localPool->alias_running) && artsThreadInfo.alive)

msi_t * _arts_tMT_msi = NULL; // tMT shared data structure
__thread unsigned int aliasId = 0;
__thread msi_t * localPool = NULL;
__thread internalMsi_t * localInternal = NULL;

static inline internalMsi_t * artsGetMsiOffsetPtr(internalMsi_t * head, unsigned int thread)
{
    unsigned int numInternal = (thread) / artsNodeInfo.tMT;
    internalMsi_t * ptr = head;
    for(unsigned int i=0; i<numInternal; i++)
        ptr = ptr->next;
    return ptr;
}

static inline artsTicket GenTicket() 
{
    artsTicket ticket;
    ticket.fields.rank = artsGlobalRankId;
    ticket.fields.unit = artsThreadInfo.groupId;
    ticket.fields.thread = aliasId;
    ticket.fields.valid = 1;
    DPRINTF("r: %u u: %u t: %u v: %u\n", (unsigned int)ticket.fields.rank, (unsigned int)ticket.fields.unit, (unsigned int)ticket.fields.thread, (unsigned int)ticket.fields.valid);
    return ticket;
}

static inline bool artsAccessorState(volatile accst_t* all_states, unsigned int start, bool flipstate) 
{
    uint64_t uint64_tState = *all_states;
    uint64_t current_pos = 1UL << start;  
    bool bState = (uint64_tState & current_pos) ? true : false;

    if(flipstate) 
        artsAtomicFetchXOrU64(all_states, current_pos);
    return bState;
}

static inline unsigned int artsNextCandidate(volatile accst_t* all_states) 
{
    return ffsll(*all_states);
}

static inline bool artsTestStateEmpty(volatile accst_t* all_states) 
{
    return !*all_states;
}

static inline bool artsTestStateOneLeft(volatile accst_t* all_states) 
{
    return *all_states && !(*all_states & (*all_states-1));
}

static inline void artsPutToWork(unsigned int rank, internalMsi_t * ptr, unsigned int thread, bool avail) 
{
    if(rank == artsGlobalRankId)
    {
        artsAccessorState(&ptr->alias_running, thread, true);
        if(avail) artsAccessorState(&ptr->alias_avail, thread, true);

    #ifdef PT_CONTEXTS
        if(sem_post(&ptr->sem[thread]) == -1) { //Wake avail thread up
            PRINTF("FAILED SEMI POST %u %u\n", artsThreadInfo.groupId, aliasId);
//            exit(EXIT_FAILURE);
        }
    #endif
    }
}

static inline void artsPutToSleep(unsigned int rank, internalMsi_t * ptr, unsigned int thread, bool avail) 
{
    if(rank == artsGlobalRankId)
    {
        artsAccessorState(&ptr->alias_running, thread, true);
        if(avail) artsAccessorState(&ptr->alias_avail, thread, true);

    #ifdef PT_CONTEXTS
        if(sem_wait(&ptr->sem[thread]) == -1) {
            PRINTF("FAILED SEMI WAIT %u %u\n", artsThreadInfo.groupId, aliasId);
//            exit(EXIT_FAILURE);
        }
    #endif
    }
}

static void* artsAliasThreadLoop(void* arg) 
{
    tmask_t * tArgs = (tmask_t*)arg;

    //set thread local vars
    aliasId = tArgs->aliasId;
    memcpy(&artsThreadInfo, tArgs->tlToCopy, sizeof(struct artsRuntimePrivate));
    
    unsigned int unitId = artsThreadInfo.groupId;
    unsigned int numAT = artsNodeInfo.tMT;
    DPRINTF("Alias: %u\n", aliasId);
    
    localPool = &_arts_tMT_msi[artsThreadInfo.groupId];
    localInternal = tArgs->localInternal;
    localInternal->alive[(aliasId % numAT)] = &artsThreadInfo.alive;
    if(artsNodeInfo.pinThreads)
    {
        DPRINTF("PINNING to %u:%u\n", artsThreadInfo.groupId, aliasId);
        artsPthreadAffinity(artsThreadInfo.coreId);
//        artsAbstractMachineModelPinThread(artsThreadInfo.coreId);
        
    }

    artsAccessorState(&localInternal->alias_running, aliasId % numAT, true);
    artsAccessorState(&localInternal->alias_avail, aliasId % numAT, true);
    
    if(sem_post(tArgs->startUpSem) == -1) {// finished  mask copy
        PRINTF("FAILED SEMI INIT POST %u %u\n", artsThreadInfo.groupId, aliasId);
//        exit(EXIT_FAILURE);
    }
    
    artsAtomicSub(&localInternal->startUpCount, 1);

    artsPutToSleep(artsGlobalRankId, localInternal, aliasId % numAT, true); //Toggle availability
    ONLY_ONE_THREAD;
    
    artsRuntimeLoop();
    
    sem_post(&localInternal->sem[0]);
    artsAtomicSub(&localInternal->shutDownCount, 1);
}

static inline void artsCreateContexts(struct artsRuntimePrivate * semiPrivate, internalMsi_t * ptr, unsigned int offset) 
{
#ifdef PT_CONTEXTS
    tmask_t tmask;
    pthread_attr_t attr;
    long pageSize = sysconf(_SC_PAGESIZE);
    size_t size = pageSize;

    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, size);

    //Init semiphores
    unsigned int numAT = artsNodeInfo.tMT;
    for(int i = 0; i < numAT; ++i) 
    {
        if(sem_init(&ptr->sem[i], 0, 0) == -1) {
            PRINTF("FAILED SEMI INIT %u %u\n", artsThreadInfo.groupId, i);
//            exit(EXIT_FAILURE);
        }
    }
    
    tmask.tlToCopy = semiPrivate;
    tmask.localInternal = ptr;
    tmask.startUpSem = &localInternal->sem[aliasId % numAT];
    
    unsigned int end = numAT-1;
    if(offset)
        end = numAT;
    else
        offset = 1;
    
    for(int i = 0; i < end; ++i) 
    {
        tmask.aliasId = i + offset;
        DPRINTF("Creating %u : %u of %u\n", tmask.aliasId, i, end);
        if(pthread_create(&ptr->aliasThreads[i], &attr, &artsAliasThreadLoop, &tmask)) {
            PRINTF("FAILED ALIAS THREAD CREATION %u %u\n", artsThreadInfo.groupId, i + offset);
//            exit(EXIT_FAILURE);
        }
        DPRINTF("Master %u: Waiting in thread creation %d\n", artsThreadInfo.groupId, i + offset);
        if(sem_wait(&localInternal->sem[aliasId % numAT]) == -1) { // wait to finish mask copy
            PRINTF("FAILED SEMI INIT WAIT %u %u\n", artsThreadInfo.groupId, i + offset);
//            exit(EXIT_FAILURE);
        }
    }
#endif
}

static inline void artsDestroyContexts(internalMsi_t * ptr, bool head) {
#ifdef PT_CONTEXTS
    unsigned int numAT = artsNodeInfo.tMT;
    unsigned int start = (head) ? 1 : 0;
    unsigned int end = (head) ? numAT-1 : numAT;
    DPRINTF("SHUTDOWN ALIAS %u: %u\n", artsThreadInfo.groupId, ptr->shutDownCount);
    while(ptr->shutDownCount) {
        for(unsigned int i=start; i<numAT; i++)
            sem_post(&ptr->sem[i]);
    }
    
    DPRINTF("ALIAS JOIN: %u\n", artsThreadInfo.groupId);
    for(unsigned int i=0; i<end; i++)
    {
        DPRINTF("Joining %u %u\n", i, head);
        pthread_join(ptr->aliasThreads[i], NULL);
    }
    
    DPRINTF("SEM DESTROY: %u\n", artsThreadInfo.groupId);
    for(unsigned int i=0; i<numAT; i++)
        sem_destroy(&ptr->sem[i]);
#endif
}

// RT visible functions
// COMMENT: MasterThread (MT) is the original thread
void artsTMTNodeInit(unsigned int numThreads)
{    
    if(artsNodeInfo.tMT > 64)
    {
        PRINTF("Temporal multi-threading can't run more than 64 threads per core\n");
        artsNodeInfo.tMT = 64;
    }
    
    if(artsNodeInfo.tMT)
    {
        _arts_tMT_msi = (msi_t *) artsCalloc(numThreads * sizeof(msi_t));
    }
}

void artsTMTConstructNewInternalMsi(msi_t * root, unsigned int numAT, struct artsRuntimePrivate * semiPrivate)
{
    //Move to the last one...
    unsigned int offset = 0;
    internalMsi_t * ptr = root->head;
    if(!root->head)
        root->head = ptr = artsCalloc(sizeof(internalMsi_t));
    else
    {
        offset = numAT;
        while(ptr->next)
        {
            offset += numAT;
            ptr = ptr->next;
        }    
        ptr->next = artsCalloc(sizeof(internalMsi_t));
        ptr = ptr->next;
    }
    
    unsigned int total = (offset) ? numAT : numAT-1;
    ptr->aliasThreads = (pthread_t*) artsMalloc(sizeof (pthread_t) * (total));
    ptr->sem = (sem_t*) artsMalloc(sizeof (sem_t) * (numAT));
    ptr->alive = (volatile bool**) artsCalloc(sizeof(bool*) * numAT);
    ptr->alias_running = (offset) ? 0UL : 1U; // MT is running on thread 0
    
    //More clever ways break for 64 alias
    //Start at 1 since MT is bit 0 and is running
    unsigned int start = (offset) ? 0 : 1;
    for(unsigned int i=start; i<numAT; i++)
        ptr->alias_avail |= 1UL << i;

    ptr->startUpCount = ptr->shutDownCount = total;
    ptr->next = NULL;
    
    if(!offset) //Thread zero needs to get initilaized...
        localInternal = ptr;
    
    artsCreateContexts(semiPrivate, ptr, offset);
    while(ptr->startUpCount);
    
    artsAtomicAdd(&root->total, artsNodeInfo.tMT);
}

void artsTMTRuntimePrivateInit(struct threadMask* unit, struct artsRuntimePrivate * semiPrivate) 
{
    localPool = &_arts_tMT_msi[artsThreadInfo.groupId];
    localPool->wakeUpNext = 0;
    localPool->wakeQueue  = artsNewQueue();
    artsTMTConstructNewInternalMsi(localPool, artsNodeInfo.tMT, semiPrivate);
}

void artsTMTRuntimeStop()
{
    if(artsNodeInfo.tMT)
    {
        for(unsigned int j=0; j<artsNodeInfo.workerThreadCount; j++)
        {
            for(internalMsi_t * ptr = _arts_tMT_msi[j].head; ptr!=NULL; ptr=ptr->next)
            {
                for(unsigned int i=0; i<artsNodeInfo.tMT; i++)
                {
                    if(ptr->alive[i])
                        *ptr->alive[i] = false;
                }
            }
        }
    }
}

void artsTMTRuntimePrivateCleanup()
{
    if(artsNodeInfo.tMT)
    {
        bool head = true;
        internalMsi_t * trail = NULL;
        internalMsi_t * ptr = localPool->head;
        while(ptr)
        {
            trail = ptr;
            ptr = ptr->next;
            artsDestroyContexts(trail, head);
            head = false;
        }
    }
}

void artsNextContext() 
{
    if(artsNodeInfo.tMT && artsThreadInfo.alive)
    {
        unsigned int cand = artsAtomicSwap(&localPool->wakeUpNext, 0);
        if(!cand)
            cand = dequeue(localPool->wakeQueue);
        if(cand)
        {
            cand--;
            artsPutToWork(artsGlobalRankId, artsGetMsiOffsetPtr(localPool->head, cand), cand % artsNodeInfo.tMT, false); //already blocked don't flip
        }
        else
        {
            if(aliasId && (aliasId + 1) % artsNodeInfo.tMT == 0 && localInternal->next)
            {
                artsPutToWork(artsGlobalRankId, localInternal->next, 0, true); //available so flip
            }
            else
            {
                cand = (aliasId + 1) % artsNodeInfo.tMT;
                artsPutToWork(artsGlobalRankId, localInternal, cand, true);  //available so flip
            }
        }

        artsPutToSleep(artsGlobalRankId, localInternal, aliasId % artsNodeInfo.tMT, true);
        ONLY_ONE_THREAD;
    }
}

void artsWakeUpContext()
{
    if(artsNodeInfo.tMT && artsThreadInfo.alive)
    {
        unsigned int cand = artsAtomicSwap(&localPool->wakeUpNext, 0);
        if(!cand)
            cand = dequeue(localPool->wakeQueue);
        if(cand)
        {
            cand--;
            artsPutToWork( artsGlobalRankId, artsGetMsiOffsetPtr(localPool->head, cand),    cand % artsNodeInfo.tMT, false);
            artsPutToSleep(artsGlobalRankId,                              localInternal, aliasId % artsNodeInfo.tMT, true);
            ONLY_ONE_THREAD;
        }
    }
}
// End of RT visible functions

bool artsContextSwitch(unsigned int waitCount) 
{
    DPRINTF("CONTEXT SWITCH\n");
    if(artsNodeInfo.tMT && artsThreadInfo.alive)
    {
        if(waitCount)
            artsAtomicAdd(&localPool->blocked, 1);
        volatile unsigned int * waitFlag = &localInternal->ticket_counter[aliasId % artsNodeInfo.tMT];
        artsAtomicAdd(waitFlag, waitCount);
        while(*waitFlag)
        {
            unsigned int cand = artsAtomicSwap(&localPool->wakeUpNext, 0);
            if(!cand)
                cand = dequeue(localPool->wakeQueue);
            if(!cand)
                cand = artsNextCandidate(&localInternal->alias_avail);
            if(cand)
            {
                cand--;
                artsPutToWork( artsGlobalRankId, localInternal,    cand % artsNodeInfo.tMT, true);
                artsPutToSleep(artsGlobalRankId, localInternal, aliasId % artsNodeInfo.tMT, false); // do not change availability
                ONLY_ONE_THREAD;
            }
            else {
                internalMsi_t * last = NULL;
                for(internalMsi_t * ptr = localPool->head; ptr!=NULL; ptr=ptr->next)
                {
                    if((cand = artsNextCandidate(&ptr->alias_avail)))
                    {
                        cand--;
                        artsPutToWork( artsGlobalRankId,           ptr,    cand % artsNodeInfo.tMT, true);
                        artsPutToSleep(artsGlobalRankId, localInternal, aliasId % artsNodeInfo.tMT, false); // do not change availability
                        return true;
                    }
                    
                    if(!ptr->next)
                        last = ptr;
                }

                artsTMTConstructNewInternalMsi(localPool, artsNodeInfo.tMT, &artsThreadInfo);
                artsPutToWork( artsGlobalRankId, last->next,    0, true);
                artsPutToSleep(artsGlobalRankId, localInternal, aliasId % artsNodeInfo.tMT, false);
            }
        }
        return true;
    }
    return false;
}

bool artsSignalContext(artsTicket_t waitTicket)
{
    DPRINTF("SIGNAL CONTEXT\n");
    artsTicket ticket = (artsTicket) waitTicket;
    unsigned int rank   = (unsigned int)ticket.fields.rank;
    unsigned int unit   = (unsigned int)ticket.fields.unit;
    unsigned int thread = (unsigned int)ticket.fields.thread;
    
    if(artsNodeInfo.tMT)
    {
        if(ticket.bits)
        {
            if(rank == artsGlobalRankId)
            {
                internalMsi_t * ptr = artsGetMsiOffsetPtr(_arts_tMT_msi[unit].head, thread);
                if(!artsAtomicSub(&ptr->ticket_counter[thread % artsNodeInfo.tMT], 1))
                {
                    artsAtomicSub(&_arts_tMT_msi[unit].blocked, 1);
                    unsigned int alias = thread + 1;
                    if(artsAtomicCswap(&_arts_tMT_msi[unit].wakeUpNext, 0, alias) != 0)
                        enqueue(alias, _arts_tMT_msi[unit].wakeQueue);
                }
            }
            else
            {
                artsRemoteSignalContext(rank, waitTicket);
            }
            return true;
        }
    }
    return false;
}

bool artsAvailContext()
{
    return (artsNodeInfo.tMT && localPool->total < MAX_TOTAL_THREADS_PER_MAX && localPool->blocked < MAX_TOTAL_THREADS_PER_MAX);
}

artsTicket_t artsGetContextTicket()
{
    artsTicket ticket;
    if(artsAvailContext())
        ticket = GenTicket();
    else
        ticket.bits = 0;
    DPRINTF("r: %u u: %u t: %u v: %u\n", (unsigned int)ticket.fields.rank, (unsigned int)ticket.fields.unit, (unsigned int)ticket.fields.thread, (unsigned int)ticket.fields.valid);
    return (artsTicket_t)ticket.bits;
}
