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

#include "artsTMT.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )
#define ONLY_ONE_THREAD 
//while(!artsTestStateOneLeft(localPool->alias_running) && artsThreadInfo.alive)

msi_t * _arts_tMT_msi = NULL; // tMT shared data structure
__thread unsigned int aliasId = 0;
__thread msi_t * localPool = NULL;

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

static inline void artsPutToWork(unsigned int rank, unsigned int unit, unsigned int thread, bool avail) 
{
    if(rank == artsGlobalRankId)
    {
        msi_t * pool = &_arts_tMT_msi[unit];
        artsAccessorState(pool->alias_running, thread, true);
        if(avail) artsAccessorState(pool->alias_avail, thread, true);

    #ifdef PT_CONTEXTS
        if(sem_post(&pool->sem[thread]) == -1) { //Wake avail thread up
            PRINTF("FAILED SEMI POST %u %u\n", artsThreadInfo.groupId, aliasId);
//            exit(EXIT_FAILURE);
        }
    #endif
    }
}

static inline void artsPutToSleep(unsigned int rank, unsigned int unit, unsigned int thread, bool avail) 
{
    if(rank == artsGlobalRankId)
    {
        msi_t * pool = &_arts_tMT_msi[unit];
        artsAccessorState(pool->alias_running, thread, true);
        if(avail) artsAccessorState(pool->alias_avail, thread, true);

    #ifdef PT_CONTEXTS
        if(sem_wait(&pool->sem[thread]) == -1) {
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
    DPRINTF("Setting: %u\n", unitId*(numAT-1)+(aliasId-1));
    artsNodeInfo.tMTLocalSpin[unitId*(numAT-1)+(aliasId-1)] = &artsThreadInfo.alive;
    
    localPool = &_arts_tMT_msi[artsThreadInfo.groupId];

    if(tArgs->unit->pin)
    {
        DPRINTF("PINNING to %u:%u\n", artsThreadInfo.groupId, aliasId);
//        artsAbstractMachineModelPinThread(tArgs->unit->coreInfo);
    }

    artsAccessorState(localPool->alias_running, aliasId, true);
    artsAccessorState(localPool->alias_avail, aliasId, true);
    
    if(sem_post(&localPool->sem[0]) == -1) {// finished  mask copy
        PRINTF("FAILED SEMI INIT POST %u %u\n", artsThreadInfo.groupId, aliasId);
//        exit(EXIT_FAILURE);
    }
    
    artsAtomicSub(&localPool->startUpCount, 1);

    artsPutToSleep(artsGlobalRankId, artsThreadInfo.groupId, aliasId, true); //Toggle availability
    ONLY_ONE_THREAD;
    
    artsRuntimeLoop();
    
    sem_post(&localPool->sem[0]);
    artsAtomicSub(&localPool->shutDownCount, 1);
}

static inline void artsCreateContexts(struct threadMask * mask, struct artsRuntimePrivate * semiPrivate) 
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
        if(sem_init(&localPool->sem[i], 0, 0) == -1) {
            PRINTF("FAILED SEMI INIT %u %u\n", artsThreadInfo.groupId, i);
//            exit(EXIT_FAILURE);
        }
    }
    
    tmask.unit = mask;
    tmask.tlToCopy = semiPrivate;
    for(int i = 1; i < numAT; ++i) 
    {
        tmask.aliasId = i;
        if(pthread_create(&localPool->aliasThreads[i-1], &attr, &artsAliasThreadLoop, &tmask)) {
            PRINTF("FAILED ALIAS THREAD CREATION %u %u\n", artsThreadInfo.groupId, i);
//            exit(EXIT_FAILURE);
        }
        DPRINTF("Master %u: Waiting in thread creation %d\n", artsThreadInfo.groupId, i);
        if(sem_wait(&localPool->sem[0]) == -1) { // wait to finish mask copy
            PRINTF("FAILED SEMI INIT WAIT %u %u\n", artsThreadInfo.groupId, i);
//            exit(EXIT_FAILURE);
        }
    }
#endif
}

static inline void artsDestroyContexts() {
#ifdef PT_CONTEXTS
    unsigned int numAT = artsNodeInfo.tMT;
    DPRINTF("SHUTDOWN ALIAS %u: %u\n", artsThreadInfo.groupId, localPool->shutDownCount);
    while(localPool->shutDownCount) {
        for(unsigned int i=1; i<numAT; i++)
            sem_post(&localPool->sem[i]);
//        PRINTF("SHUTDOWN ALIAS %u: %u\n", artsThreadInfo.groupId, localPool->shutDownCount);
    }
    
    DPRINTF("ALIAS JOIN: %u\n", artsThreadInfo.groupId);
    for(unsigned int i=0; i<numAT-1; i++)
        pthread_join(localPool->aliasThreads[i], NULL);
    
    DPRINTF("SEM DESTROY: %u\n", artsThreadInfo.groupId);
    for(unsigned int i=0; i<numAT; i++)
        sem_destroy(&localPool->sem[i]);
#endif
}

// RT visible functions
// COMMENT: MasterThread (MT) is the original thread
void artsTMTNodeInit(unsigned int numThreads)
{
    if(artsNodeInfo.tMT == 1)
    {
        PRINTF("Temporal multi-threading only running 1 thread per core.  To context switch tMT > 1\n");
        artsNodeInfo.tMT = 0;
    }
    
    if(artsNodeInfo.tMT > 64)
    {
        PRINTF("Temporal multi-threading can't run more than 64 threads per core\n");
        artsNodeInfo.tMT = 64;
    }
    
    if(artsNodeInfo.tMT)
    {
        _arts_tMT_msi = (msi_t *) artsCalloc(numThreads * sizeof(msi_t));
        artsNodeInfo.tMTLocalSpin = (volatile bool**) artsCalloc(sizeof(bool*) * numThreads * (artsNodeInfo.tMT-1));
    }
}

void artsTMTRuntimePrivateInit(struct threadMask* unit, struct artsRuntimePrivate * semiPrivate) 
{
    unsigned int numAT = artsNodeInfo.tMT;
    localPool = &_arts_tMT_msi[artsThreadInfo.groupId];
    localPool->aliasThreads = (pthread_t*) artsMalloc(sizeof (pthread_t) * (numAT-1));
    
    // FIXME: for now, we'll live with an bitmap array structure...
    // FIXME: convert SOA into AOS to avoid collisions
    localPool->alias_running = (accst_t*) artsCalloc(sizeof (accst_t));
    *localPool->alias_running = 1UL; // MT is running on thread 0
    
    localPool->alias_avail   = (accst_t*) artsCalloc(sizeof (accst_t));
    //More clever ways break for 64 alias
    //Start at 1 since MT is bit 0 and is running
    for(unsigned int i=1; i<numAT; i++)
        *localPool->alias_avail |= 1UL << i;
    
    localPool->wakeUpNext = 0;
    localPool->wakeQueue  = artsNewQueue();

    localPool->sem = (sem_t*) artsMalloc(sizeof (sem_t) * (numAT));
    
    localPool->startUpCount = localPool->shutDownCount = numAT-1;
    artsCreateContexts(unit, semiPrivate);
    
    while(localPool->startUpCount);
    ONLY_ONE_THREAD;
}

void artsTMTRuntimeStop()
{
    if(artsNodeInfo.tMT)
    {
        for(unsigned int i=0; i<artsNodeInfo.workerThreadCount * (artsNodeInfo.tMT-1); i++)
            *artsNodeInfo.tMTLocalSpin[i] = false;
    }
}

void artsTMTRuntimePrivateCleanup()
{
    if(artsNodeInfo.tMT)
        artsDestroyContexts();
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
            artsPutToWork(artsGlobalRankId, artsThreadInfo.groupId, cand, false); //already blocked don't flip
        }
        else
        {
            cand = (aliasId + 1) % artsNodeInfo.tMT;
            artsPutToWork(artsGlobalRankId, artsThreadInfo.groupId, cand, true);  //available so flip
        }

        artsPutToSleep(artsGlobalRankId, artsThreadInfo.groupId, aliasId, true);
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
            artsPutToWork( artsGlobalRankId, artsThreadInfo.groupId, cand,    false);
            artsPutToSleep(artsGlobalRankId, artsThreadInfo.groupId, aliasId, true);
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
        volatile unsigned int * waitFlag = &localPool->ticket_counter[aliasId];
        artsAtomicAdd(waitFlag, waitCount);
        while(*waitFlag)
        {
            unsigned int cand = artsAtomicSwap(&localPool->wakeUpNext, 0);
            if(!cand)
                cand = dequeue(localPool->wakeQueue);
            if(!cand)
                cand = artsNextCandidate(localPool->alias_avail);
            if(cand)
                cand--;
            else {
                cand = (aliasId + 1) % artsNodeInfo.tMT;
            }
            artsPutToWork( artsGlobalRankId, artsThreadInfo.groupId, cand,    true);
            artsPutToSleep(artsGlobalRankId, artsThreadInfo.groupId, aliasId, false); // do not change availability
            ONLY_ONE_THREAD;
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
                if(!artsAtomicSub(&_arts_tMT_msi[unit].ticket_counter[thread], 1))
                {
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
    if(artsNodeInfo.tMT)
    {
        unsigned int cand = artsNextCandidate(localPool->alias_avail);
        DPRINTF("R: %p A: %p Cand: %u\n", localPool->alias_running, localPool->alias_avail, cand);
        return cand != 0;
    }
    return false;
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
