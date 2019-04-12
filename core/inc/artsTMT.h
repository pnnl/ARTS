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

#ifndef CORE_INC_ARTS_TMT_H_
#define CORE_INC_ARTS_TMT_H_

#include <semaphore.h>
#include "arts.h"
#include "artsConfig.h"
#include "artsAbstractMachineModel.h"
#include "artsRuntime.h"
#include "artsGlobals.h"

#define MAX_THREADS_PER_MASTER 64

typedef uint64_t accst_t;  // accessor state

typedef union
{
    uint64_t bits: 64;
    struct __attribute__((packed))
    {
        uint32_t   rank:   31;
        uint16_t   unit:   16; 
        uint16_t thread:   16;
        uint8_t   valid:    1;
    } fields;
} artsTicket;

typedef struct
{
    pthread_t *               aliasThreads;                           //Actual pthreads
    sem_t *                   sem;                                    //Semaphores used to wake/sleep
    volatile unsigned int     ticket_counter[MAX_THREADS_PER_MASTER]; //Fixed counters used to keep track of outstanding promises (we can only wait on one context at a time)
    volatile accst_t *        alias_running; // FIXME: right data structure?
    volatile accst_t *        alias_avail;
    volatile unsigned int     wakeUpNext;
    artsQueue *               wakeQueue;
    volatile unsigned int     startUpCount;
    volatile unsigned int     shutDownCount;
} msi_t __attribute__ ((aligned (64)));                        // master shared info

typedef struct
{
  uint32_t                    aliasId;  // alias id
  struct threadMask *         unit;     // alias shared pool info
  struct artsRuntimePrivate * tlToCopy; // we copy the master thread's TL
} tmask_t; // per alias thread info

// RTS internal interface
void artsTMTNodeInit(unsigned int numThreads);
void artsTMTRuntimePrivateInit(struct threadMask* unit, struct artsRuntimePrivate * semiPrivate);
void artsTMTRuntimePrivateCleanup();
void artsTMTRuntimeStop();

bool artsAvailContext();
void artsNextContext();
void artsWakeUpContext();



#endif /* CORE_INC_ARTS_TMT_H_ */
