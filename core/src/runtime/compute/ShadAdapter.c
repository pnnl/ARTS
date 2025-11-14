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

#include "arts/runtime/compute/ShadAdapter.h"

#include "arts/arts.h"
#include "arts/gas/Guid.h"
#include "arts/gas/RouteTable.h"
#include "arts/introspection/Counter.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/Runtime.h"
#include "arts/runtime/compute/EdtFunctions.h"
#include "arts/runtime/sync/TerminationDetection.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"
#include "arts/system/TMTLite.h"
#include "arts/utils/Atomics.h"
#include "arts/utils/Queue.h"

artsGuid_t artsEdtCreateShad(artsEdt_t funcPtr, unsigned int route,
                             uint32_t paramc, uint64_t *paramv) {
  unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t);
  artsGuid_t guid = NULL_GUID;
  artsEdtCreateInternal(NULL, ARTS_EDT, &guid, route, artsThreadInfo.clusterId,
                        edtSpace, NULL_GUID, funcPtr, paramc, paramv, 0, false,
                        NULL_GUID, false, 0);
  return guid;
}

artsGuid_t artsActiveMessageShad(artsEdt_t funcPtr, unsigned int route,
                                 uint32_t paramc, uint64_t *paramv, void *data,
                                 unsigned int size, artsGuid_t epochGuid) {
  unsigned int rank = route; // route / numNumaDomains;
  unsigned int cluster = 0;  // route % numNumaDomains;
  artsGuid_t guid = NULL_GUID;
  bool useEpoch = (epochGuid != NULL_GUID);

  if (size) {
    unsigned int depSpace = sizeof(artsEdtDep_t);
    unsigned int edtSpace =
        sizeof(struct artsEdt) + paramc * sizeof(uint64_t) + depSpace;
    artsEdtCreateInternal(NULL, ARTS_EDT, &guid, rank, cluster, edtSpace,
                          NULL_GUID, funcPtr, paramc, paramv, 1, useEpoch,
                          epochGuid, true, 0);

    void *ptr = artsMalloc(size);
    memcpy(ptr, data, size);
    artsSignalEdtPtr(guid, 0, ptr, size);
  } else {
    unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t);
    artsEdtCreateInternal(NULL, ARTS_EDT, &guid, rank, cluster, edtSpace,
                          NULL_GUID, funcPtr, paramc, paramv, 0, useEpoch,
                          epochGuid, false, 0);
  }
  return guid;
}

void artsSynchronousActiveMessageShad(artsEdt_t funcPtr, unsigned int route,
                                      uint32_t paramc, uint64_t *paramv,
                                      void *data, unsigned int size) {
  unsigned int rank = route; // route / numNumaDomains;
  unsigned int cluster = 0;  // route % numNumaDomains;
  unsigned int waitFlag = 1;
  void *waitPtr = &waitFlag;
  artsGuid_t waitGuid = artsAllocateLocalBuffer(
      (void **)&waitPtr, sizeof(unsigned int), 1, NULL_GUID);

  artsGuid_t guid = NULL_GUID;
  if (size) {
    unsigned int depSpace = sizeof(artsEdtDep_t);
    unsigned int edtSpace =
        sizeof(struct artsEdt) + paramc * sizeof(uint64_t) + depSpace;
    artsEdtCreateInternal(NULL, ARTS_EDT, &guid, rank, cluster, edtSpace,
                          waitGuid, funcPtr, paramc, paramv, 1, false,
                          NULL_GUID, true, 0);

    void *ptr = artsMalloc(size);
    memcpy(ptr, data, size);
    artsSignalEdtPtr(guid, 0, ptr, size);
  } else {
    unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t);
    artsEdtCreateInternal(NULL, ARTS_EDT, &guid, rank, cluster, edtSpace,
                          waitGuid, funcPtr, paramc, paramv, 0, false,
                          NULL_GUID, false, 0);
  }

  while (waitFlag) {
    artsYield();
  }
}

void artsIncLockShad() { artsThreadInfo.shadLock++; }

void artsDecLockShad() { artsThreadInfo.shadLock--; }

void artsCheckLockShad() {
  if (artsThreadInfo.shadLock) {
    ARTS_INFO("ARTS: Cannot perform synchronous call under lock Worker: %u "
              "ShadLock: %u",
              artsThreadInfo.groupId, artsThreadInfo.shadLock);
    artsDebugGenerateSegFault();
  }
}

void artsStartIntroShad(unsigned int start) { artsCounterStart(start); }

void artsStopIntroShad() { artsCounterStop(); }

unsigned int artsGetShadLoopStride() { return artsNodeInfo.shadLoopStride; }

artsGuid_t artsAllocateLocalBufferShad(void **buffer, uint32_t *sizeToWrite,
                                       artsGuid_t epochGuid) {
  if (epochGuid) {
    incrementActiveEpoch(epochGuid);
  } else {
    ARTS_INFO("No EPOCH!!!");
  }
  globalShutdownGuidIncActive();

  artsBuffer_t *stub = (artsBuffer_t *)artsMalloc(sizeof(artsBuffer_t));
  stub->buffer = *buffer;
  stub->sizeToWrite = sizeToWrite;
  stub->size = 0;
  stub->uses = 1;
  stub->epochGuid = epochGuid;

  artsGuid_t guid = artsGuidCreateForRank(artsGlobalRankId, ARTS_BUFFER);
  artsRouteTableAddItem(stub, guid, artsGlobalRankId, false);
  return guid;
}

artsShadLock_t *artsShadCreateLock() {
  artsShadLock_t *lock =
      (artsShadLock_t *)artsCalloc(1, sizeof(artsShadLock_t));
  lock->queue = artsNewQueue();
  return lock;
}

void artsShadLock(artsShadLock_t *lock) {
  unsigned int res = artsAtomicFetchAdd(&lock->size, 1);
  if (res) {
    enqueue(artsGetContextTicket(), lock->queue);
    artsContextSwitch(1);
  }
}

void artsShadUnlock(artsShadLock_t *lock) {
  unsigned int res = artsAtomicSub(&lock->size, 1);
  if (res) {
    while (1) {
      artsTicket_t ticket = dequeue(lock->queue);
      if (ticket) {
        artsSignalContext(ticket);
        return;
      }
    }
  }
}

#define ALIASOWNERMAP 0xF000000000000000
#define ALIASCOUNTMAP 0x0FFFFFFFFFFFFFFF
#define ALIASGETOWNER(x) (x & ALIASOWNERMAP) >> 60
#define ALIASGETCOUNT(x) (x & ALIASCOUNTMAP)
#define ALIASEMPTY ((((uint64_t)artsThreadInfo.groupId) + 1) << 60) + 1

bool artsShadAliasTryLock(volatile uint64_t *lock) {
  uint64_t dirtyRead = *lock;
  uint64_t owner = ALIASGETOWNER(dirtyRead);
  while (!owner || owner == artsThreadInfo.groupId + 1) {
    uint64_t newValue = (!owner) ? ALIASEMPTY : dirtyRead + 1;
    uint64_t res = artsAtomicCswapU64(lock, dirtyRead, newValue);
    if (res == dirtyRead)
      return true;
    dirtyRead = res;
    owner = ALIASGETOWNER(dirtyRead);
  }
  return false;
}

void artsShadAliasUnlock(volatile uint64_t *lock) {
  uint64_t dirtyRead = *lock;
  while (1) {
    uint64_t newValue = (ALIASGETCOUNT(dirtyRead) == 1) ? 0 : dirtyRead - 1;
    uint64_t res = artsAtomicCswapU64(lock, dirtyRead, newValue);
    if (res == dirtyRead) {
      // ARTS_INFO("RES: %lu", res);
      return;
    }
    dirtyRead = res;
  }
}

#define LITEOWNERMAP 0x8000000000000000ULL
#define LITECOUNTMAP 0x7FFFFFFFFFFFFFFFULL
#define LITEGETOWNER(x) (x & LITEOWNERMAP)
#define LITEGETCOUNT(x) (x & ALIASCOUNTMAP)

static inline bool artsTMTLiteTryLock(volatile uint64_t *lock) {
  uint64_t local = *lock;
  if (!LITEGETOWNER(local)) {
    if (local == artsAtomicCswapU64(lock, local, local | LITEOWNERMAP))
      return true;
  }
  return false;
}

static inline void artsTMTLiteUnlock(volatile uint64_t *lock) {
  artsAtomicFetchAndU64(lock, LITECOUNTMAP);
}

// Returns if you should dec on unlock
bool artsShadTMTLock2(volatile uint64_t *lock) {
  // We should have the execution lock on entry
  if (!artsTMTLiteTryLock(lock)) // Try lock but fail
  {
    // artsAtomicAddU64(lock, 1); //Inc the counter that we have created thread
    artsCreateLiteContexts(lock); // Still have execution lock at end
    uint64_t counter = 0;
    while (1) {
      if (artsTMTLiteTryLock(lock))
        break;
      artsYieldLiteContext();  // Give up execution lock
      artsResumeLiteContext(); // Has execution lock at end
      if (counter > 1000000) {
        artsAtomicAddU64(lock, 1); // Inc the counter that we have created
                                   // thread
        artsCreateLiteContexts(lock);
        ARTS_INFO("STUPID CREATE!!!");
        counter = 0;
      }
      counter++;
    }
    return true;
  }
  return false;
}

void artsShadTMTUnlock(volatile uint64_t *lock) {
  // ARTS_INFO("Unlock: %p %u:%u", lock, artsGetCurrentWorker(),
  // artsTMTLiteGetAlias());
  artsTMTLiteUnlock(lock);
}

void artsShadTMTLock(volatile uint64_t *lock) {
  while (artsThreadInfo.alive) {
    if (artsTMTLiteTryLock(lock)) // Try lock but fail
      break;
    else {
      struct artsEdt *edt = artsFindEdt();
      if (edt) {
        artsAtomicAddU64(lock, 1); // Inc the counter that we have created
                                   // thread
        artsCreateLiteContexts2(lock, edt); // Still have execution lock at end
      } else {
        checkOutstandingEdts(10000000);
      }
      artsYieldLiteContext(); // Give up execution lock
    }
    artsResumeLiteContext();
  }
  // ARTS_INFO("Lock: %p %u:%u", lock, artsGetCurrentWorker(),
  // artsTMTLiteGetAlias());
}
