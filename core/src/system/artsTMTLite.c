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
#include "artsDebug.h"
#include "artsTMT.h"
#include "artsArrayList.h"
#include "artsRuntime.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

__thread unsigned int tmtLiteAliasId = 0;

size_t pageSize = 0; //Got this from andres' TMT... Not sure if we need it?
artsArrayList ** threadToJoin;
unsigned int * aliasNumber;

volatile unsigned int toCreateThreads = 0;
volatile unsigned int doneCreateThreads = 0;
volatile unsigned int doneThreads = 0;
volatile uint64_t * outstanding;


volatile unsigned int * threadReaderLock;
volatile unsigned int * threadWriterLock;
volatile unsigned int * arrayListLock;

// volatile unsigned int threadCreateReader = 0;
// volatile unsigned int threadCreateWriter = 0;

void artsWriterLockYield(volatile unsigned int * readLock, volatile unsigned int * writeLock)
{
    unsigned int toSwap = tmtLiteAliasId + 1;
    while(artsAtomicCswap(writeLock, 0U, toSwap) != 0U) { 
        pthread_yield(); 
        }
    while((*readLock)) { pthread_yield(); }
    return;
}

void artsInitTMTLitePerNode(unsigned int numWorkers)
{
    long temp = sysconf(_SC_PAGESIZE);
    pageSize = temp;
    threadToJoin = (artsArrayList**) artsCalloc(sizeof(artsArrayList*) * numWorkers);
    aliasNumber = (unsigned int*) artsCalloc(sizeof(unsigned int) * numWorkers);
    threadReaderLock = (volatile unsigned int*) artsCalloc(sizeof(unsigned int) * numWorkers);
    threadWriterLock = (volatile unsigned int*) artsCalloc(sizeof(unsigned int) * numWorkers);
    arrayListLock = (volatile unsigned int*) artsCalloc(sizeof(unsigned int) * numWorkers);
    outstanding = (volatile uint64_t *) artsCalloc(sizeof(uint64_t) * numWorkers);
}

void artsInitTMTLitePerWorker(unsigned int id)
{
    artsWriterLockYield(&threadReaderLock[id], &threadWriterLock[id]);
    DPRINTF("EXECUTION LOCK: %p %u %u\n", &threadWriterLock[id], threadWriterLock[id], id);
    threadToJoin[id] = artsNewArrayList(sizeof(pthread_t), 8);
}

void artsTMTLiteShutdown() {
    DPRINTF("%u outstanding: %u\n",toCreateThreads, doneCreateThreads);
    while(toCreateThreads!=doneCreateThreads);
    
}

void artsTMTLitePrivateCleanUp(unsigned int id)
{
    DPRINTF("%u outstanding: %u\n",toCreateThreads, doneThreads);
    while(toCreateThreads!=doneThreads) { pthread_yield(); }
    uint64_t outstanding = artsLengthArrayList(threadToJoin[id]);
    for(uint64_t i=0; i<outstanding; i++)
    {
        pthread_t * thread = artsGetFromArrayList(threadToJoin[id], i);
        pthread_join(*thread, NULL);
    }
    PRINTF("%u joined: %lu threads\n", id, outstanding);
}

typedef struct
{
  uint32_t aliasId;  // alias id
  uint32_t sourceId;
  struct artsEdt * edtToRun;
  volatile unsigned int * toDec;
  struct artsRuntimePrivate * tlToCopy; // we copy the master thread's TL
} liteArgs_t;

void * artsAliasLiteThreadLoop(void * arg) 
{
    liteArgs_t * tArgs = (liteArgs_t*) arg;
    tmtLiteAliasId = tArgs->aliasId;
    uint32_t sourceId = tArgs->sourceId;
    memcpy(&artsThreadInfo, tArgs->tlToCopy, sizeof(struct artsRuntimePrivate));

    if(artsNodeInfo.pinThreads)
    {
        DPRINTF("PINNING to %u:%u\n", artsThreadInfo.groupId, aliasId);
        artsPthreadAffinity(artsThreadInfo.coreId, false);
    }

    unsigned int res = artsAtomicAdd(&doneCreateThreads, 1);
    artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);
    DPRINTF("GOT LOCK %u alias %u\n", sourceId, tmtLiteAliasId);
    if(artsThreadInfo.alive)
        artsNodeInfo.scheduler();
    artsWriterUnlock(&threadWriterLock[sourceId]);
    uint64_t tempRes = artsAtomicSub(tArgs->toDec, 1);
    artsAtomicAdd(&doneThreads, 1);
    artsFree(tArgs);
}

void artsCreateLiteContexts(volatile uint64_t * toDec) 
{
    unsigned int sourceId = artsThreadInfo.groupId;
    unsigned int res = artsAtomicAdd(&toCreateThreads, 1);
    volatile unsigned int spinFlag = 1;
    liteArgs_t * args = artsCalloc(sizeof(liteArgs_t));
    args->aliasId = ++aliasNumber[sourceId];
    args->sourceId = sourceId;
    args->toDec = toDec;
    args->tlToCopy = &artsThreadInfo; 
    DPRINTF("threadsToCreate %u args to copy: %p\n", res, args.tlToCopy);
    
    
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, pageSize);
    pthread_t * thread = (pthread_t*) artsNextFreeFromArrayList(threadToJoin[artsThreadInfo.groupId]);
    
    artsWriterUnlock(&threadWriterLock[sourceId]);

    pthread_create(thread, &attr, &artsAliasLiteThreadLoop, args);
    
    artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);
}

void * artsAliasLiteThreadLoop2(void * arg) 
{
    liteArgs_t * tArgs = (liteArgs_t*) arg;
    tmtLiteAliasId = tArgs->aliasId;
    uint32_t sourceId = tArgs->sourceId;
    memcpy(&artsThreadInfo, tArgs->tlToCopy, sizeof(struct artsRuntimePrivate));

    if(artsNodeInfo.pinThreads)
    {
        DPRINTF("PINNING to %u:%u\n", artsThreadInfo.groupId, aliasId);
        artsPthreadAffinity(artsThreadInfo.coreId, false);
    }

    DPRINTF("SourceId: %u vs %u -- %u\n", sourceId, artsThreadInfo.groupId, tmtLiteAliasId);

    artsAtomicAdd(&doneCreateThreads, 1);
    
    artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);
    
    DPRINTF("GOT LOCK %u alias %u\n", sourceId, tmtLiteAliasId);
    artsRunEdt(tArgs->edtToRun);
    
    artsWriterUnlock(&threadWriterLock[sourceId]);
    
    artsAtomicSub(tArgs->toDec, 1);
    artsAtomicSubU64(&outstanding[sourceId], 1);
    artsAtomicAdd(&doneThreads, 1);
    artsFree(tArgs);
}

void artsCreateLiteContexts2(volatile uint64_t * toDec, struct artsEdt * edt)
{
    unsigned int sourceId = artsThreadInfo.groupId;
    unsigned int res = artsAtomicAdd(&toCreateThreads, 1);
    artsAtomicAddU64(&outstanding[sourceId], 1);
    volatile unsigned int spinFlag = 1;
    liteArgs_t * args = artsCalloc(sizeof(liteArgs_t));
    args->aliasId = ++aliasNumber[sourceId];
    args->sourceId = sourceId;
    args->edtToRun = edt;
    args->toDec = toDec;
    args->tlToCopy = &artsThreadInfo; 
    DPRINTF("threadsToCreate %u args to copy: %p\n", res, args.tlToCopy);
    
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, pageSize);
    artsLock(&arrayListLock[artsThreadInfo.groupId]);
    pthread_t * thread = (pthread_t*) artsNextFreeFromArrayList(threadToJoin[artsThreadInfo.groupId]);
    artsUnlock(&arrayListLock[artsThreadInfo.groupId]);
    pthread_create(thread, &attr, &artsAliasLiteThreadLoop2, args);
}

void artsYieldLiteContext()
{
    unsigned int sourceId = artsThreadInfo.groupId;
    artsWriterUnlock(&threadWriterLock[sourceId]);
    pthread_yield();
}

void artsResumeLiteContext()
{
    unsigned int sourceId = artsThreadInfo.groupId;
    artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);
}

unsigned int artsTMTLiteGetAlias()
{
    return tmtLiteAliasId;
}

void artsTMTSchedulerYield()
{
    unsigned int sourceId = artsThreadInfo.groupId;
    if(outstanding[sourceId])
    {
        // PRINTF("Scheduler Yield %u\n", outstanding[sourceId]);
        artsWriterUnlock(&threadWriterLock[sourceId]);
        pthread_yield();
        artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);
    }
}