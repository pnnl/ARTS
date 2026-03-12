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
 * artsTMT
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

#ifndef ARTS_SYSTEM_TMT_H
#define ARTS_SYSTEM_TMT_H
#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>
#include <semaphore.h>

#include "arts/system/AbstractMachineModel.h"
#include "arts/utils/Queue.h"

#define MAX_TOTAL_THREADS_PER_MAX 65536
#define MAX_THREADS_PER_MASTER 64

typedef uint64_t accst_t; // accessor state

typedef union {
  uint64_t bits : 64;
  struct __attribute__((packed)) {
    uint32_t rank : 31;
    uint16_t unit : 16;
    uint16_t thread : 16;
    uint8_t valid : 1;
  } fields;
} artsTicket;

// Assume in TMT.c struct
typedef struct internalMsi_t {
  pthread_t *aliasThreads;
  sem_t *sem;
  volatile bool **alive;
  volatile bool **initShutdown;
  volatile unsigned int
      ticket_counter[MAX_THREADS_PER_MASTER]; // Fixed counters used to keep
                                              // track of outstanding promises
                                              // (we can only wait on one
                                              // context at a time)
  volatile accst_t alias_running;             // FIXME: right data structure?
  volatile accst_t alias_avail;
  volatile unsigned int startUpCount;
  volatile unsigned int shutDownCount;
  struct internalMsi_t *next;
  pthread_mutex_t mutex;
  pthread_cond_t cond[64];
} internalMsi_t;

typedef struct msi {
  struct internalMsi_t *head;
  volatile unsigned int blocked;
  volatile unsigned int total;
  volatile unsigned int wakeUpNext;
  artsQueue *wakeQueue;
} msi_t __attribute__((aligned(64))); // master shared info

typedef struct {
  uint32_t aliasId;                    // alias id
  struct artsRuntimePrivate *tlToCopy; // we copy the master thread's TL
  struct internalMsi_t *localInternal;
  sem_t *startUpSem;
} tmask_t; // per alias thread info

// RTS internal interface
void artsTMTNodeInit(unsigned int numThreads);
void artsTMTRuntimePrivateInit(struct threadMask *unit,
                               struct artsRuntimePrivate *semiPrivate);
void artsTMTRuntimePrivateCleanup();
bool artsTMTRuntimeStop();

bool artsAvailContext();
void artsNextContext();
void artsWakeUpContext();

#ifdef __cplusplus
}
#endif
#endif /* ARTS_SYSTEM_TMT_H */
