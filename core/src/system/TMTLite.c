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
#include "arts/system/TMTLite.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <unistd.h>

#include "arts/runtime/Globals.h"
#include "arts/runtime/Runtime.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Threads.h"
#include "arts/utils/ArrayList.h"
#include "arts/utils/Atomics.h"

__thread unsigned int tmtLiteAliasId = 0;

size_t pageSize = 0; // Got this from andres' TMT... Not sure if we need it?
artsArrayList **threadToJoin;
unsigned int *aliasNumber;

volatile unsigned int toCreateThreads = 0;
volatile unsigned int doneCreateThreads = 0;
volatile unsigned int doneThreads = 0;
volatile uint64_t *outstanding;

volatile unsigned int *threadReaderLock;
volatile unsigned int *threadWriterLock;
volatile unsigned int *arrayListLock;

// volatile unsigned int threadCreateReader = 0;
// volatile unsigned int threadCreateWriter = 0;

void artsWriterLockYield(volatile unsigned int *readLock,
                         volatile unsigned int *writeLock) {
  unsigned int toSwap = tmtLiteAliasId + 1;
  while (artsAtomicCswap(writeLock, 0U, toSwap) != 0U) {
    INCREMENT_SLEEP_COUNTER_BY(1);
    sched_yield();
  }
  while ((*readLock)) {
    INCREMENT_SLEEP_COUNTER_BY(1);
    sched_yield();
  }
  return;
}

void artsInitTMTLitePerNode(unsigned int numWorkers) {
  long temp = sysconf(_SC_PAGESIZE);
  pageSize = temp;
  threadToJoin =
      (artsArrayList **)artsCalloc(numWorkers, sizeof(artsArrayList *));
  aliasNumber = (unsigned int *)artsCalloc(numWorkers, sizeof(unsigned int));
  threadReaderLock =
      (volatile unsigned int *)artsCalloc(numWorkers, sizeof(unsigned int));
  threadWriterLock =
      (volatile unsigned int *)artsCalloc(numWorkers, sizeof(unsigned int));
  arrayListLock =
      (volatile unsigned int *)artsCalloc(numWorkers, sizeof(unsigned int));
  outstanding = (volatile uint64_t *)artsCalloc(numWorkers, sizeof(uint64_t));
}

void artsInitTMTLitePerWorker(unsigned int id) {
  artsWriterLockYield(&threadReaderLock[id], &threadWriterLock[id]);
  threadToJoin[id] = artsNewArrayList(sizeof(pthread_t), 8);
}

void artsTMTLiteShutdown() {
  while (toCreateThreads != doneCreateThreads)
    ;
}

void artsTMTLitePrivateCleanUp(unsigned int id) {
  while (toCreateThreads != doneThreads) {
    INCREMENT_SLEEP_COUNTER_BY(1);
    sched_yield();
  }
  uint64_t outstanding = artsLengthArrayList(threadToJoin[id]);
  for (uint64_t i = 0; i < outstanding; i++) {
    pthread_t *thread = (pthread_t *)artsGetFromArrayList(threadToJoin[id], i);
    pthread_join(*thread, NULL);
  }
}

typedef struct {
  uint32_t aliasId; // alias id
  uint32_t sourceId;
  struct artsEdt *edtToRun;
  volatile uint64_t *toDec;
  struct artsRuntimePrivate *tlToCopy; // we copy the master thread's TL
} liteArgs_t;

void *artsAliasLiteThreadLoop(void *arg) {
  liteArgs_t *tArgs = (liteArgs_t *)arg;
  tmtLiteAliasId = tArgs->aliasId;
  uint32_t sourceId = tArgs->sourceId;
  memcpy(&artsThreadInfo, tArgs->tlToCopy, sizeof(struct artsRuntimePrivate));

  if (artsNodeInfo.pinThreads) {
    artsPthreadAffinity(artsThreadInfo.coreId, false);
  }

  unsigned int res = artsAtomicAdd(&doneCreateThreads, 1);
  artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);
  if (artsThreadInfo.alive)
    artsNodeInfo.scheduler();
  artsWriterUnlock(&threadWriterLock[sourceId]);
  uint64_t tempRes = artsAtomicSubU64(tArgs->toDec, 1);
  artsAtomicAdd(&doneThreads, 1);
  artsFree(tArgs);
  return NULL;
}

void artsCreateLiteContexts(volatile uint64_t *toDec) {
  unsigned int sourceId = artsThreadInfo.groupId;
  unsigned int res = artsAtomicAdd(&toCreateThreads, 1);
  volatile unsigned int spinFlag = 1;
  liteArgs_t *args = (liteArgs_t *)artsCalloc(1, sizeof(liteArgs_t));
  args->aliasId = ++aliasNumber[sourceId];
  args->sourceId = sourceId;
  args->toDec = toDec;
  args->tlToCopy = &artsThreadInfo;

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, pageSize);
  pthread_t *thread = (pthread_t *)artsNextFreeFromArrayList(
      threadToJoin[artsThreadInfo.groupId]);

  artsWriterUnlock(&threadWriterLock[sourceId]);

  pthread_create(thread, &attr, &artsAliasLiteThreadLoop, args);

  artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);
}

void *artsAliasLiteThreadLoop2(void *arg) {
  liteArgs_t *tArgs = (liteArgs_t *)arg;
  tmtLiteAliasId = tArgs->aliasId;
  uint32_t sourceId = tArgs->sourceId;
  memcpy(&artsThreadInfo, tArgs->tlToCopy, sizeof(struct artsRuntimePrivate));

  if (artsNodeInfo.pinThreads) {
    artsPthreadAffinity(artsThreadInfo.coreId, false);
  }

  artsAtomicAdd(&doneCreateThreads, 1);
  artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);

  artsRunEdt(tArgs->edtToRun);

  artsWriterUnlock(&threadWriterLock[sourceId]);

  artsAtomicSubU64(tArgs->toDec, 1);
  artsAtomicSubU64(&outstanding[sourceId], 1);
  artsAtomicAdd(&doneThreads, 1);
  artsFree(tArgs);
  return NULL;
}

void artsCreateLiteContexts2(volatile uint64_t *toDec, struct artsEdt *edt) {
  unsigned int sourceId = artsThreadInfo.groupId;
  unsigned int res = artsAtomicAdd(&toCreateThreads, 1);
  artsAtomicAddU64(&outstanding[sourceId], 1);
  volatile unsigned int spinFlag = 1;
  liteArgs_t *args = (liteArgs_t *)artsCalloc(1, sizeof(liteArgs_t));
  args->aliasId = ++aliasNumber[sourceId];
  args->sourceId = sourceId;
  args->edtToRun = edt;
  args->toDec = toDec;
  args->tlToCopy = &artsThreadInfo;

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, pageSize);
  artsLock(&arrayListLock[artsThreadInfo.groupId]);
  pthread_t *thread = (pthread_t *)artsNextFreeFromArrayList(
      threadToJoin[artsThreadInfo.groupId]);
  artsUnlock(&arrayListLock[artsThreadInfo.groupId]);
  pthread_create(thread, &attr, &artsAliasLiteThreadLoop2, args);
}

void artsYieldLiteContext() {
  unsigned int sourceId = artsThreadInfo.groupId;
  artsWriterUnlock(&threadWriterLock[sourceId]);
  INCREMENT_SLEEP_COUNTER_BY(1);
  sched_yield();
}

void artsResumeLiteContext() {
  unsigned int sourceId = artsThreadInfo.groupId;
  artsWriterLockYield(&threadReaderLock[sourceId], &threadWriterLock[sourceId]);
}

unsigned int artsTMTLiteGetAlias() { return tmtLiteAliasId; }

void artsTMTSchedulerYield() {
  unsigned int sourceId = artsThreadInfo.groupId;
  if (outstanding[sourceId]) {
    // ARTS_INFO("Scheduler Yield %u", outstanding[sourceId]);
    artsWriterUnlock(&threadWriterLock[sourceId]);
    sched_yield();
    artsWriterLockYield(&threadReaderLock[sourceId],
                        &threadWriterLock[sourceId]);
  }
}