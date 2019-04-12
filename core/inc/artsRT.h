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
#ifndef ARTSRT_H
#define ARTSRT_H
#ifdef __cplusplus
extern "C" {
#endif
    
//#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
    
/* boolean support in C */
#ifdef __cplusplus
#define TRUE true
#define FALSE false
#else
#define true 1
#define TRUE 1
#define false 0
#define FALSE 0
typedef uint8_t bool;
#endif /* __cplusplus */  
    
typedef intptr_t artsGuid_t; /**< GUID type */    
#define NULL_GUID ((artsGuid_t)0x0)

//This is the ticker for context switching
typedef uint64_t artsTicket_t;

typedef enum
{
    ARTS_NULL = 0,
    ARTS_EDT,
    ARTS_EVENT,
    ARTS_EPOCH,
    ARTS_CALLBACK,
    ARTS_BUFFER,
//These are the DB modes.  Allocate/cast these types of DBs!
            
//ARTS_DB_READ: This mode is write once read many.  
//Create the DB in this mode and write data before signaling the dbGuid.
//This mode aggregates requests, and caches reads in the routing table.
    ARTS_DB_READ, 
            
//ARTS_DB_WRITE: This mode is used to provide exclusive access.  
//Use this mode by casting ARTS_DB_READ DBs to ARTS_DB_WRITE and signal an EDT.
//This mode is currently broken!!!
    ARTS_DB_WRITE,
            
//ARTS_DB_PIN: This mode bypasses the memory model.
//The DB is only available on a single node.
//To interact with it remotely use put/gets
    ARTS_DB_PIN,
            
//ARTS_DB_ONCE: This mode will automatically free the DB after it is acquired.
//This is to help memory management, since we are never reusing the DB.
    ARTS_DB_ONCE,
            
//ARTS_DB_ONCE: This mode is the same as ARTS_DB_ONCE except we are guarenteing
//That the DB is local to the EDT accessing it (i.e. edtGuid and dbGuid have the same route).
    ARTS_DB_ONCE_LOCAL,
            
//End DB modes
    ARTS_LAST_TYPE,
    ARTS_SINGLE_VALUE,
    ARTS_PTR
} artsType_t;

typedef struct
{
    artsGuid_t guid;
    artsType_t mode;
    void *ptr;
} artsEdtDep_t;

//Signature of an EDT
typedef void (*artsEdt_t) (uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);

//Signature of an event callback.  The data parameter is the value of the dataGuid used to satisfy the event.
typedef void (*eventCallback_t)(artsEdtDep_t data);

//Signature of the send handler used by artsRemoteSend.
typedef void (*sendHandler_t) (void * args);

typedef enum
{
    ARTS_EVENT_LATCH_DECR_SLOT = 0,
    ARTS_EVENT_LATCH_INCR_SLOT = 1
} artsLatchEventSlot_t;

struct artsHeader
{
    uint8_t type:8;
    uint64_t size:56;
} __attribute__ ((aligned));

struct artsDb
{
    struct artsHeader header;
    artsGuid_t guid;
    void * dbList;
} __attribute__ ((aligned));

struct artsEdt
{
    struct artsHeader header;
    artsEdt_t funcPtr;
    uint32_t paramc;
    uint32_t depc;
    artsGuid_t currentEdt;
    artsGuid_t outputBuffer;
    artsGuid_t epochGuid;
    unsigned int cluster;
    volatile unsigned int depcNeeded;
    volatile unsigned int invalidateCount;
} __attribute__ ((aligned));

struct artsDependent
{
    uint8_t type;
    volatile unsigned int slot;
    volatile artsGuid_t addr;
    volatile eventCallback_t callback;
    volatile bool doneWriting;
};

struct artsDependentList
{
    unsigned int size;
    struct artsDependentList * volatile next;
    struct artsDependent dependents[];
};

struct artsEvent
{
    struct artsHeader header;
    volatile bool fired;
    volatile unsigned int destroyOnFire;
    volatile unsigned int latchCount;
    volatile unsigned int pos;
    volatile unsigned int lastKnown;
    volatile unsigned int dependentCount;
    artsGuid_t data;
    struct artsDependentList dependent;
} __attribute__ ((aligned));

typedef struct
{
    unsigned int size;
    unsigned int index;
    artsGuid_t startGuid;
} artsGuidRange;

typedef struct  artsArrayDb
{
    unsigned int elementSize;
    unsigned int elementsPerBlock;
    unsigned int numBlocks;
    char head[];
} artsArrayDb_t;

typedef enum
{
    PHASE_1,
    PHASE_2,
    PHASE_3
} TerminationDetectionPhase;

typedef struct {
    TerminationDetectionPhase phase;
    volatile unsigned int activeCount;
    volatile unsigned int finishedCount;
    volatile unsigned int globalActiveCount;
    volatile unsigned int globalFinishedCount;
    volatile unsigned int lastActiveCount;
    volatile unsigned int lastFinishedCount;
    volatile uint64_t queued;
    volatile uint64_t outstanding;
    unsigned int terminationExitSlot;
    artsGuid_t terminationExitGuid;
    artsGuid_t guid;
    artsGuid_t poolGuid;
    volatile unsigned int * waitPtr;
    volatile uint64_t ticket;
} artsEpoch_t;

void PRINTF( const char* format, ... );

#ifdef __cplusplus
}
#endif

#endif /* ARTSRT_H */

