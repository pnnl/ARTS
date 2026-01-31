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

#include "arts/runtime/memory/DbFunctions.h"

#include <assert.h>
#include <string.h>

#include "arts/arts.h"
#include "arts/gas/Guid.h"
#include "arts/gas/OutOfOrder.h"
#include "arts/gas/RouteTable.h"
#include "arts/introspection/ArtsIdCounter.h"
#include "arts/introspection/Counter.h"
#include "arts/introspection/Metrics.h"
#include "arts/introspection/Preamble.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/RT.h"
#include "arts/runtime/compute/EdtFunctions.h"
#include "arts/runtime/memory/DbList.h"
#include "arts/runtime/memory/TwinDiff.h"
#include "arts/runtime/network/RemoteFunctions.h"
#include "arts/runtime/sync/TerminationDetection.h"
#include "arts/system/ArtsPrint.h"
#include "arts/utils/Atomics.h"

#ifdef USE_GPU
#include "arts/gpu/GpuRuntime.cuh"
#endif

static inline artsType_t artsGetEffectiveMode(artsEdtDep_t *dep) {
  return (dep->acquireMode != ARTS_NULL) ? dep->acquireMode : dep->mode;
}

artsTypeName;

void *artsDbMalloc(artsType_t mode, unsigned int size) {
  void *ptr = NULL;
#ifdef USE_GPU
  if (artsNodeInfo.gpu) {
    if (mode == ARTS_DB_LC)
      ptr = artsCudaMallocHost(size * 2);
    else if (mode == ARTS_DB_GPU_READ || mode == ARTS_DB_GPU_WRITE)
      ptr = artsCudaMallocHost(size);
  }
#endif
  if (!ptr)
    ptr = artsMallocAlign(size, 16);
  return ptr;
}

void artsDbFree(void *ptr) {
  struct artsDb *db = (struct artsDb *)ptr;
#ifdef USE_GPU
  if (artsNodeInfo.gpu &&
      (db->header.type == ARTS_DB_GPU_READ ||
       db->header.type == ARTS_DB_GPU_WRITE || db->header.type == ARTS_DB_LC)) {
    artsCudaFreeHost(ptr);
    ptr = NULL;
  }
#endif
  if (ptr)
    artsFree(ptr);
}

void artsDbCreateInternal(artsGuid_t guid, void *addr, uint64_t size,
                          uint64_t packetSize, artsType_t mode,
                          uint64_t arts_id) {
  struct artsHeader *header = (struct artsHeader *)addr;
  header->type = mode;
  header->size = packetSize;

  struct artsDb *dbRes = (struct artsDb *)header;
  dbRes->arts_id = arts_id; // Set compiler-assigned arts_id (0 if not set)
  dbRes->guid = guid;
  dbRes->version = 0;
  dbRes->reader = 0;
  dbRes->writer = 0;
  dbRes->copyCount = 0;
  dbRes->twin = NULL;
  dbRes->twinFlags = 0;
  dbRes->twinSize = 0;
  if (mode != ARTS_DB_PIN) {
    dbRes->dbList = artsNewDbList();
  }
  if (mode == ARTS_DB_LC) {
    void *shadowCopy = (void *)(((char *)addr) + packetSize);
    memcpy(shadowCopy, addr, sizeof(struct artsDb));
  }
  dbRes->eventGuid = artsPersistentEventCreate(artsGuidGetRank(guid), 0, guid);
  // Record arts_id metrics via counter infrastructure
  artsCounterRecordArtsIdDb(arts_id, packetSize, 0, 0);
  INCREMENT_NUM_DBS_CREATED_BY(1);
}

artsGuid_t artsDbCreateRemote(unsigned int route, uint64_t size,
                              artsType_t mode) {
  DB_CREATE_COUNTER_START();
  if (route == -1)
    route = artsGlobalRankId;
  artsGuid_t guid = artsGuidCreateForRank(route, mode);
  void *ptr = artsDbMalloc(mode, sizeof(struct artsDb));
  struct artsDb *db = (struct artsDb *)ptr;
  db->header.size = size + sizeof(struct artsDb);
  db->dbList = (mode == ARTS_DB_PIN) ? (void *)0 : (void *)1;

  artsRemoteMemoryMove(route, guid, ptr, sizeof(struct artsDb),
                       ARTS_REMOTE_DB_SEND_MSG, artsDbFree);
  DB_CREATE_COUNTER_STOP();
  return guid;
}

// Creates a local DB only
artsGuid_t artsDbCreate(void **addr, uint64_t size, artsType_t mode) {
  DB_CREATE_COUNTER_START();
  artsGuid_t guid = NULL_GUID;
  uint64_t dbSize = size + sizeof(struct artsDb);

  void *ptr = artsMallocWithType(dbSize, artsDbMemorySize);
  if (ptr) {
    guid = artsGuidCreateForRank(artsGlobalRankId, mode);
    artsDbCreateInternal(guid, ptr, size, dbSize, mode, 0);
    // change false to true to force a manual DB delete
    artsRouteTableAddItem(ptr, guid, artsGlobalRankId, false);
    *addr = (void *)((struct artsDb *)ptr + 1);
  }
  DB_CREATE_COUNTER_STOP();
  return guid;
}

artsGuid_t artsDbCreatePtr(artsPtr_t *addr, uint64_t size, artsType_t mode) {
  DB_CREATE_COUNTER_START();
  artsGuid_t guid = NULL_GUID;
  uint64_t dbSize = size + sizeof(struct artsDb);

  void *ptr = artsMallocWithType(dbSize, artsDbMemorySize);
  if (ptr) {
    guid = artsGuidCreateForRank(artsGlobalRankId, mode);
    artsDbCreateInternal(guid, ptr, size, dbSize, mode, 0);
    // change false to true to force a manual DB delete
    artsRouteTableAddItem(ptr, guid, artsGlobalRankId, false);
    *addr = (artsPtr_t)((struct artsDb *)ptr + 1);
  }
  DB_CREATE_COUNTER_STOP();
  return guid;
}

artsGuid_t artsDbCreateWithArtsId(void **addr, uint64_t size, artsType_t mode,
                                  uint64_t arts_id) {
  DB_CREATE_COUNTER_START();
  artsGuid_t guid = NULL_GUID;
  uint64_t dbSize = size + sizeof(struct artsDb);

  void *ptr = artsMallocWithType(dbSize, artsDbMemorySize);
  if (ptr) {
    guid = artsGuidCreateForRank(artsGlobalRankId, mode);
    artsDbCreateInternal(guid, ptr, size, dbSize, mode, arts_id);
    // change false to true to force a manual DB delete
    artsRouteTableAddItem(ptr, guid, artsGlobalRankId, false);
    *addr = (void *)((struct artsDb *)ptr + 1);
  }
  DB_CREATE_COUNTER_STOP();
  return guid;
}

void *artsDbCreateWithGuidAndArtsId(artsGuid_t guid, uint64_t size,
                                    uint64_t arts_id) {
  DB_CREATE_COUNTER_START();
  artsType_t mode = artsGuidGetType(guid);

  void *ptr = NULL;
  if (artsIsGuidLocal(guid)) {
    uint64_t dbSize = size + sizeof(struct artsDb);

    ptr = artsMallocWithType(dbSize, artsDbMemorySize);
    if (ptr) {
      artsDbCreateInternal(guid, ptr, size, dbSize, mode, arts_id);
      if (artsRouteTableAddItemRace(ptr, guid, artsGlobalRankId, false)) {
        artsRouteTableFireOO(guid, artsOutOfOrderHandler);
      }
      ptr = (void *)((struct artsDb *)ptr + 1);
    }
  }
  ARTS_INFO("Creating DB[Id:%lu, Guid:%lu, Mode:%s, Ptr:%p, Route:%d, "
            "Size:%lu]",
            arts_id, guid, getTypeName(mode), ptr, artsGuidGetRank(guid), size);
  DB_CREATE_COUNTER_STOP();
  return ptr;
}

// Guid must be for a local DB only
void *artsDbCreateWithGuid(artsGuid_t guid, uint64_t size) {
  DB_CREATE_COUNTER_START();
  artsType_t mode = artsGuidGetType(guid);

  void *ptr = NULL;
  uint64_t arts_id = 0;
  if (artsIsGuidLocal(guid)) {
    uint64_t dbSize = size + sizeof(struct artsDb);

    ptr = artsMallocWithType(dbSize, artsDbMemorySize);
    if (ptr) {
      struct artsDb *dbHeader = (struct artsDb *)ptr;
      artsDbCreateInternal(guid, dbHeader, size, dbSize, mode, 0);
      arts_id = dbHeader->arts_id;
      if (artsRouteTableAddItemRace(dbHeader, guid, artsGlobalRankId, false)) {
        artsRouteTableFireOO(guid, artsOutOfOrderHandler);
      }
      ptr = (void *)(dbHeader + 1);
    }
  }
  ARTS_INFO("Creating DB[Id:%lu, Guid:%lu, Mode:%s, Ptr:%p, Route:%d, "
            "Size:%lu]",
            arts_id, guid, getTypeName(mode), ptr, artsGuidGetRank(guid), size);
  DB_CREATE_COUNTER_STOP();
  return ptr;
}

void *artsDbCreateWithGuidAndData(artsGuid_t guid, void *data, uint64_t size) {
  DB_CREATE_COUNTER_START();
  artsType_t mode = artsGuidGetType(guid);
  void *ptr = NULL;
  uint64_t arts_id = 0;
  if (artsIsGuidLocal(guid)) {
    uint64_t dbSize = size + sizeof(struct artsDb);

    ptr = artsMallocWithType(dbSize, artsDbMemorySize);

    if (ptr) {
      struct artsDb *dbHeader = (struct artsDb *)ptr;
      artsDbCreateInternal(guid, dbHeader, size, dbSize, mode, 0);
      arts_id = dbHeader->arts_id;
      void *dbData = (void *)(dbHeader + 1);
      memcpy(dbData, data, size);
      if (artsRouteTableAddItemRace(dbHeader, guid, artsGlobalRankId, false))
        artsRouteTableFireOO(guid, artsOutOfOrderHandler);
      ptr = dbData;
    }
  }
  ARTS_INFO("Creating DB[Id:%lu, Guid:%lu, Mode:%s, Ptr:%p, Route:%d, "
            "Size:%lu]",
            arts_id, guid, getTypeName(mode), ptr, artsGuidGetRank(guid), size);
  DB_CREATE_COUNTER_STOP();
  return ptr;
}

void *artsDbResizePtr(struct artsDb *dbRes, unsigned int size, bool copy) {
  if (dbRes) {
    unsigned int oldSize = dbRes->header.size;
    unsigned int newSize = size + sizeof(struct artsDb);
    struct artsDb *ptr = (struct artsDb *)artsCallocAlignWithType(
        1, newSize, 16, artsDbMemorySize);
    if (ptr) {
      if (copy)
        memcpy(ptr, dbRes, oldSize);
      else
        memcpy(ptr, dbRes, sizeof(struct artsDb));
      artsFree(dbRes);
      ptr->header.size = size + sizeof(struct artsDb);
      return (void *)(ptr + 1);
    }
  }
  return NULL;
}

// Must be in write mode (or only copy) to update and alloced (no NO_ACQUIRE
// nonsense), otherwise will be racy...
void *artsDbResize(artsGuid_t guid, unsigned int size, bool copy) {
  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(guid);
  void *ptr = artsDbResizePtr(dbRes, size, copy);
  if (ptr) {
    dbRes = ((struct artsDb *)ptr) - 1;
  }
  return ptr;
}

void artsDbMove(artsGuid_t dbGuid, unsigned int rank) {
  unsigned int guidRank = artsGuidGetRank(dbGuid);
  if (guidRank != rank) {
    if (guidRank != artsGlobalRankId)
      artsDbMoveRequest(dbGuid, rank);
    else {
      struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(dbGuid);
      if (dbRes) {
        artsRemoteMemoryMove(rank, dbGuid, dbRes, dbRes->header.size,
                             ARTS_REMOTE_DB_MOVE_MSG, artsDbFree);
      } else {
        artsOutOfOrderDbMove(dbGuid, rank);
      }
    }
  }
}

void artsDbDestroy(artsGuid_t guid) {
  artsType_t mode = artsGuidGetType(guid);
  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(guid);
  if (dbRes != NULL) {
    artsRemoteDbDestroy(guid, artsGlobalRankId, 0);
    artsDbFree(dbRes);
    artsRouteTableRemoveItem(guid);
  } else
    artsRemoteDbDestroy(guid, artsGlobalRankId, 0);
}

bool artsDbRenameWithGuid(artsGuid_t newGuid, artsGuid_t oldGuid) {
  bool ret = false;
  unsigned int rank = artsGuidGetRank(oldGuid);
  if (rank == artsGlobalRankId) {
    struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(oldGuid);
    if (dbRes != NULL) {
      dbRes->guid = newGuid;
      // This is only being done by the owner...
      artsRouteTableHideItem(oldGuid);
      if (artsRouteTableAddItemRace(dbRes, newGuid, artsGlobalRankId, false)) {
        artsRouteTableFireOO(newGuid, artsOutOfOrderHandler);
      }
      ret = true;
    }
  } else {
    artsRemoteDbRename(newGuid, oldGuid);
  }
  return ret;
}

artsGuid_t artsDbCopyToNewType(artsGuid_t oldGuid, artsType_t newType) {
  artsGuid_t ret = NULL_GUID;
  unsigned int rank = artsGuidGetRank(oldGuid);
  if (rank == artsGlobalRankId) {
    artsGuid_t newGuid = artsGuidCreateForRank(rank, artsGuidGetType(newType));
    struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(oldGuid);
    if (dbRes != NULL) {
      artsAtomicAdd(&dbRes->copyCount, 1);
      dbRes->guid = newGuid;
      if (artsRouteTableAddItemRace(dbRes, newGuid, artsGlobalRankId, false)) {
        artsRouteTableFireOO(newGuid, artsOutOfOrderHandler);
      }
      ret = newGuid;
    }
  }
  return ret;
}

artsGuid_t artsDbRename(artsGuid_t guid) {
  artsGuid_t newGuid =
      artsGuidCreateForRank(artsGuidGetRank(guid), artsGuidGetType(guid));
  return (artsDbRenameWithGuid(newGuid, guid)) ? newGuid : NULL_GUID;
}

void artsDbDestroySafe(artsGuid_t guid, bool remote) {
  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(guid);
  if (dbRes != NULL) {
    if (remote)
      artsRemoteDbDestroy(guid, artsGlobalRankId, 0);
    artsDbFree(dbRes);
    artsRouteTableRemoveItem(guid);
  } else if (remote)
    artsRemoteDbDestroy(guid, artsGlobalRankId, 0);
}

void artsDbIncrementLatch(artsGuid_t guid) {
  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(guid);
  if (dbRes != NULL)
    artsPersistentEventIncrementLatch(dbRes->eventGuid);
  else
    artsRemoteDbIncrementLatch(guid);
}

void artsDbDecrementLatch(artsGuid_t guid) {
  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(guid);
  if (dbRes != NULL)
    artsPersistentEventDecrementLatch(dbRes->eventGuid);
  else
    artsRemoteDbDecrementLatch(guid);
}

void artsDbAddDependence(artsGuid_t dbSrc, artsGuid_t edtDest,
                         uint32_t edtSlot) {
  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(dbSrc);
  if (dbRes != NULL)
    artsAddDependenceToPersistentEvent(dbRes->eventGuid, edtDest, edtSlot);
  else
    artsRemoteDbAddDependence(dbSrc, edtDest, edtSlot);
}

void artsDbAddDependenceWithMode(artsGuid_t dbSrc, artsGuid_t edtDest,
                                 uint32_t edtSlot, artsType_t acquireMode) {
  artsDbAddDependenceWithModeAndDiff(dbSrc, edtDest, edtSlot, acquireMode,
                                     true);
}

void artsDbAddDependenceWithModeAndDiff(artsGuid_t dbSrc, artsGuid_t edtDest,
                                        uint32_t edtSlot,
                                        artsType_t acquireMode,
                                        bool useTwinDiff) {

  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(dbSrc);
  if (dbRes != NULL)
    artsAddDependenceToPersistentEventWithModeAndDiff(
        dbRes->eventGuid, edtDest, edtSlot, acquireMode, useTwinDiff);
  else
    artsRemoteDbAddDependenceWithHints(dbSrc, edtDest, edtSlot, acquireMode,
                                       useTwinDiff);
}

// Auto-increments latch for WRITE mode, records dependency with compiler hints
void artsRecordDep(artsGuid_t dbSrc, artsGuid_t edtDest, uint32_t edtSlot,
                   artsType_t acquireMode, bool useTwinDiff) {
  artsDbAddDependenceWithModeAndDiff(dbSrc, edtDest, edtSlot, acquireMode,
                                     useTwinDiff);
  if (acquireMode == ARTS_DB_WRITE)
    artsDbIncrementLatch(dbSrc);
}

void artsRecordDepAt(artsGuid_t dbSrc, artsGuid_t edtDest, uint32_t edtSlot,
                     artsType_t acquireMode, bool useTwinDiff,
                     uint64_t byteOffset, uint64_t size) {
  // If no byte offset, use the standard path
  if (byteOffset == 0 && size == 0) {
    artsRecordDep(dbSrc, edtDest, edtSlot, acquireMode, useTwinDiff);
    return;
  }

  // Use extended dependency registration with byte offset info
  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(dbSrc);
  if (dbRes != NULL)
    artsAddDependenceToPersistentEventWithByteOffset(
        dbRes->eventGuid, edtDest, edtSlot, acquireMode, useTwinDiff,
        byteOffset, size);
  else
    artsRemoteDbAddDependenceWithByteOffset(dbSrc, edtDest, edtSlot,
                                            acquireMode, useTwinDiff,
                                            byteOffset, size);

  if (acquireMode == ARTS_DB_WRITE)
    artsDbIncrementLatch(dbSrc);
}

/**********************DB MEMORY MODEL*************************************/
// Side Effects: edt depcNeeded will be incremented, ptr will be updated,
//   and launches out of order handleReadyEdt
// Returns false on out of order and true otherwise
void acquireDbs(struct artsEdt *edt) {
  artsEdtDep_t *depv = (artsEdtDep_t *)artsGetDepv(edt);
  edt->depcNeeded = edt->depc + 1;
  ARTS_INFO("Acquiring %u DBs for EDT[Id:%lu, Guid:%lu], depcNeeded "
            "initialized to %u",
            edt->depc, edt->arts_id, edt->currentEdt, edt->depcNeeded);
  for (int i = 0; i < edt->depc; i++) {
    if (depv[i].guid && depv[i].ptr == NULL) {
      struct artsDb *dbFound = NULL;
      int owner = artsGuidGetRank(depv[i].guid);

      // Calculate effective acquire mode (override if compiler provided hint)
      artsType_t effectiveMode = artsGetEffectiveMode(&depv[i]);

      // Update acquire-mode counters
      if (effectiveMode == ARTS_DB_READ) {
        INCREMENT_ACQUIRE_READ_MODE_BY(1);
        // Track owner update savings when override changes WRITE -> READ
        if (depv[i].mode == ARTS_DB_WRITE && owner == artsGlobalRankId)
          INCREMENT_OWNER_UPDATES_SAVED_BY(1);
      } else if (effectiveMode == ARTS_DB_WRITE) {
        INCREMENT_ACQUIRE_WRITE_MODE_BY(1);
        if (owner == artsGlobalRankId)
          INCREMENT_OWNER_UPDATES_PERFORMED_BY(1);
      }

      ARTS_INFO("Acquiring DB[Guid:%lu, Mode:%u, Owner:%d, Rank:%u] "
                "in EDT[Id:%lu, Guid:%lu, Slot:%u]",
                depv[i].guid, depv[i].mode, owner, artsGlobalRankId,
                edt->arts_id, edt->currentEdt, i);
      switch (depv[i].mode) {
      // This case assumes that the guid exists only on the owner
      case ARTS_DB_ONCE: {
        if (owner != artsGlobalRankId) {
          artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i, false);
          artsDbMove(depv[i].guid, artsGlobalRankId);
          break;
        }
        // else fall through to the local case :-p
      }
      case ARTS_DB_ONCE_LOCAL: {
        struct artsDb *dbTemp =
            (struct artsDb *)artsRouteTableLookupItem(depv[i].guid);
        if (dbTemp) {
          dbFound = dbTemp;
          artsAtomicSub(&edt->depcNeeded, 1U);
        } else
          artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i, false);
        break;
      }
      case ARTS_DB_PIN: {
        int validRank = -1;
        struct artsDb *dbTemp = (struct artsDb *)artsRouteTableLookupDb(
            depv[i].guid, &validRank, true);
        if (dbTemp) {
          dbFound = dbTemp;
          artsAtomicSub(&edt->depcNeeded, 1U);
        } else {
          artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i, true);
        }
        break;
      }
      case ARTS_DB_LC_SYNC: {
        // Owner Rank
        if (owner == artsGlobalRankId) {
          int validRank = -1;
          struct artsDb *dbTemp = (struct artsDb *)artsRouteTableLookupDb(
              depv[i].guid, &validRank, false);
          // We have found an entry
          if (dbTemp) {
            ARTS_DEBUG("MODE: %s -> %p", getTypeName(depv[i].mode), dbTemp);
            dbFound = dbTemp;
            artsAtomicSub(&edt->depcNeeded, 1U);
          }
          // The Db hasn't been created yet
          else {
            // TODO: Create an out-of-order sync
            ARTS_DEBUG("DB[Guid:%lu] out of order request for LC_SYNC not "
                       "supported yet",
                       depv[i].guid);
            // artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i, true);
          }
        }
        break;
      }
      case ARTS_DB_LC_NO_COPY:
      case ARTS_DB_GPU_MEMSET:
      case ARTS_DB_GPU_READ:
      case ARTS_DB_GPU_WRITE:
      case ARTS_DB_LC:
      case ARTS_DB_READ:
      case ARTS_DB_WRITE: {
        // Owner Rank
        if (owner == artsGlobalRankId) {
          int validRank = -1;
          struct artsDb *dbTemp = (struct artsDb *)artsRouteTableLookupDb(
              depv[i].guid, &validRank, true);
          // We have found an entry
          if (dbTemp) {
            bool duplicateAdded =
                artsAddDbDuplicate(dbTemp, artsGlobalRankId, edt,
                                   edt->currentEdt, i, effectiveMode);
            if (duplicateAdded)
              ARTS_DEBUG("Adding duplicate DB[Guid:%lu]", depv[i].guid);
            else
              ARTS_DEBUG(
                  "Duplicate not added DB[Guid:%lu] (rank already tracked)",
                  depv[i].guid);

            // If the owner (this rank) still has the valid copy, hand it out
            // regardless of whether we inserted a new duplicate entry.
            if (validRank == artsGlobalRankId) {
              dbFound = dbTemp;
              artsAtomicSub(&edt->depcNeeded, 1U);
            }
            // Owner rank but another rank currently has the valid copy; request
            // it.
            else {
              if (depv[i].mode == ARTS_DB_READ ||
                  depv[i].mode == ARTS_DB_GPU_READ ||
                  depv[i].mode == ARTS_DB_GPU_WRITE ||
                  depv[i].mode == ARTS_DB_LC ||
                  depv[i].mode == ARTS_DB_LC_NO_COPY ||
                  depv[i].mode == ARTS_DB_GPU_MEMSET)
                artsRemoteDbRequest(depv[i].guid, validRank, edt, i,
                                    depv[i].mode, true, depv[i].acquireMode,
                                    depv[i].useTwinDiff);
              else
                artsRemoteDbFullRequest(depv[i].guid, validRank,
                                        edt->currentEdt, i, depv[i].mode);
            }
          }
          // The Db hasn't been created yet
          else {
            ARTS_DEBUG("DB[Guid:%lu] out of order request slot %u",
                       depv[i].guid, i);
            artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i, true);
          }
        } else {
          int validRank = -1;
          struct artsDb *dbTemp = (struct artsDb *)artsRouteTableLookupDb(
              depv[i].guid, &validRank, true);
          // We have found an entry
          ARTS_INFO("[AcquireDbs] Non-owner case for DB[Guid:%lu, Mode:%u, "
                    "Owner:%d, ValidRank: %d, DbTemp: %p]",
                    depv[i].guid, depv[i].mode, owner, validRank,
                    (void *)dbTemp);
          bool localValid = (dbTemp && validRank == artsGlobalRankId);
          if (effectiveMode == ARTS_DB_READ && depv[i].mode == ARTS_DB_WRITE &&
              localValid) {
            // Drop cached READ copies for DB_WRITE types to avoid stale data.
            artsRouteTableInvalidateItem(depv[i].guid);
            dbTemp = NULL;
            validRank = -1;
            localValid = false;
          }
          if (localValid && effectiveMode != ARTS_DB_WRITE) {
            dbFound = dbTemp;
            artsAtomicSub(&edt->depcNeeded, 1U);
            ARTS_INFO("  Found local valid copy, decremented depcNeeded");
          } else if (dbTemp && effectiveMode == ARTS_DB_WRITE) {
            if (localValid) {
              dbFound = dbTemp;
              artsAtomicSub(&edt->depcNeeded, 1U);
              ARTS_INFO("  Found local valid copy for WRITE; using it");
            } else {
              // For WRITE, non-owner cached copies may be stale; wait for owner.
              ARTS_INFO("  Local copy ignored for WRITE; waiting for owner");
            }
          }
          if (effectiveMode == ARTS_DB_WRITE) {
            if (!dbFound) {
              // Check if twin-diff is enabled for this WRITE acquire
              if (depv[i].useTwinDiff && depv[i].acquireMode == ARTS_DB_WRITE) {
                // Twin-diff enabled: use aggregated request
                // (will receive diffs, not full DB)
                ARTS_INFO(
                    "  WRITE mode with twin-diff - sending aggregated request "
                    "to rank %d (AcquireMode:%u, UseTwinDiff: %d)",
                    owner, depv[i].acquireMode, depv[i].useTwinDiff);
                artsRemoteDbRequest(depv[i].guid, owner, edt, i,
                                    depv[i].acquireMode, true,
                                    depv[i].acquireMode, depv[i].useTwinDiff);
              } else {
                // Traditional path: full DB transfer for exclusive WRITE
                ARTS_INFO("  WRITE mode - sending full DB request to rank %d",
                          owner);
                artsRemoteDbFullRequest(depv[i].guid, owner, edt->currentEdt, i,
                                        effectiveMode);
              }
            } else {
              ARTS_INFO("  WRITE mode with local valid copy - no remote request "
                        "needed");
            }
          } else if (!dbTemp || !localValid) {
            // We can aggregate read requests for reads
            ARTS_INFO("  READ mode, no local copy - sending aggregated request "
                      "to rank %d",
                      owner);
            int requestRank = owner;
            artsRemoteDbRequest(depv[i].guid, requestRank, edt, i, depv[i].mode,
                                true, ARTS_NULL, false);
          } else {
            ARTS_INFO("  READ mode with local copy - no remote request needed");
          }
        }
      } break;

      case ARTS_NULL:
      default:
        artsAtomicSub(&edt->depcNeeded, 1U);
        break;
      }

      if (dbFound)
        depv[i].ptr = dbFound + 1;
      ARTS_DEBUG("DB[Guid:%lu, Ptr:%p] acquired", depv[i].guid, depv[i].ptr);
    } else {
      artsAtomicSub(&edt->depcNeeded, 1U);
    }
  }
  ARTS_INFO("EDT[Id:%lu, Guid:%lu] has finished acquiring DBs", edt->arts_id,
            edt->currentEdt);
}

void prepDbs(unsigned int depc, artsEdtDep_t *depv, bool gpu) {
  for (unsigned int i = 0; i < depc; i++) {
    artsType_t effectiveMode = artsGetEffectiveMode(&depv[i]);
    if (depv[i].guid != NULL_GUID) {
      if (depv[i].acquireMode == ARTS_DB_READ)
        INCREMENT_ACQUIRE_READ_MODE_BY(1);
      else if (depv[i].acquireMode == ARTS_DB_WRITE)
        INCREMENT_ACQUIRE_WRITE_MODE_BY(1);
    }

    if (depv[i].guid != NULL_GUID && effectiveMode == ARTS_DB_WRITE) {
      if (depv[i].mode != ARTS_DB_PIN)
        artsRemoteUpdateRouteTable(depv[i].guid, -1);
      struct artsDb *db = ((struct artsDb *)depv[i].ptr) - 1;
      ARTS_DEBUG("[prepDbs] DB[Id:%lu, Guid:%lu] ptr=%p, db=%p, "
                 "useTwinDiff=%d",
                 db->arts_id, depv[i].guid, depv[i].ptr, db,
                 depv[i].useTwinDiff);
      // Allocate twin
      if (depv[i].useTwinDiff && !(db->twinFlags & ARTS_DB_HAS_TWIN)) {
        artsDbAllocateTwin(db);
        size_t dbSize = db->header.size - sizeof(struct artsDb);
        ARTS_DEBUG("Allocated twin for DB[Id:%lu, Guid:%lu, size: %zu] "
                   "(compiler hint)",
                   db->arts_id, depv[i].guid, dbSize);
      }
    }
#ifdef USE_GPU
    if (!gpu && depv[i].mode == ARTS_DB_LC) {
      struct artsDb *db = ((struct artsDb *)depv[i].ptr) - 1;
      artsReaderLock(&db->reader, &db->writer);
      internalIncDbVersion(&db->version);
    }

    if (!gpu && depv[i].mode == ARTS_DB_LC_SYNC) {
      struct artsDb *db = ((struct artsDb *)depv[i].ptr) - 1;
      ARTS_DEBUG("internalLCSync %lu %p", depv[i].guid, db);
      internalLCSyncGPU(depv[i].guid, db);
    }
#endif
  }
}

/**
 * Helper: Handle twin-diff release for WRITE mode
 * Returns true if release was handled, false otherwise
 */
static inline bool artsTryReleaseTwinDiff(artsGuid_t guid, artsEdtDep_t *dep) {
  struct artsDb *db = ((struct artsDb *)dep->ptr) - 1;
  unsigned int owner = artsGuidGetRank(guid);
  bool isOwner = (owner == artsGlobalRankId);

  // Check if twin was allocated and use twin-diff
  if (db->twinFlags & ARTS_DB_HAS_TWIN) {
    void *working = (void *)(db + 1);
    void *twin = db->twin;
    size_t dbSize = db->header.size - sizeof(struct artsDb);

    // Quick estimate of dirty ratio
    float dirtyRatio = artsEstimateDirtyRatio(working, twin, dbSize);

    ARTS_DEBUG("DB[Id:%lu, Guid:%lu] estimated dirty ratio: %.2f%% "
               "(isOwner=%d)",
               db->arts_id, guid, dirtyRatio * 100.0, isOwner);

    // If dirty ratio is too high, fall back to full DB send (non-owner only)
    if (dirtyRatio > ARTS_DIFF_DIRTY_THRESHOLD && !isOwner) {
      ARTS_INFO("DB[Id:%lu, Guid:%lu] dirty ratio %.2f%% exceeds threshold, "
                "using full DB send",
                db->arts_id, guid, dirtyRatio * 100.0);
      artsRemoteUpdateDb(guid, true);
      INCREMENT_OWNER_UPDATES_PERFORMED_BY(1);
      INCREMENT_TWIN_DIFF_SKIPPED_BY(1);
    } else {
      // Compute diffs and send partial update (non-owner) or just update
      // progress (owner)
      struct artsDiffList *diffs = artsComputeDiffs(working, twin, dbSize);
      if (diffs && diffs->regionCount > 0) {
        ARTS_INFO("DB[Id:%lu, Guid:%lu] using twin-diff: %u regions, "
                  "%u bytes (%.2f%% of %zu)",
                  db->arts_id, guid, diffs->regionCount, diffs->totalBytes,
                  diffs->dirtyRatio * 100.0, dbSize);

        if (!isOwner) {
          // Non-owner: Send partial update immediately (no coalescing)
          artsRemotePartialUpdateDb(guid, diffs, working);
          artsFreeDiffList(diffs);
        } else {
          // Owner:Just update progress frontier (canonical DB already has
          // changes)
          artsProgressFrontier(db, artsGlobalRankId);
          artsFreeDiffList(diffs);
        }

        INCREMENT_OWNER_UPDATES_PERFORMED_BY(1);
      } else {
        // No diffs - either signal owner (non-owner) or just update progress
        // (owner)
        ARTS_DEBUG("DB[Id:%lu, Guid:%lu] no diffs detected", db->arts_id, guid);
        if (!isOwner)
          artsRemoteUpdateDb(guid, false);
        else
          artsProgressFrontier(db, artsGlobalRankId);

        INCREMENT_OWNER_UPDATES_PERFORMED_BY(1);
      }
    }

    // Free twin after use. Only the owner decrements the latch here; non-owners
    // defer latch decrement until the controller applies their update.
    artsDbFreeTwin(db);

    if (isOwner)
      artsDbDecrementLatch(guid);
    return true;
  }

  // No twin allocated - fall back to traditional release
  if (isOwner) {
    // Owner without twin: traditional release
    artsProgressFrontier(db, artsGlobalRankId);
    artsDbDecrementLatch(guid);
  } else {
    // Non-owner without twin: full DB send
    artsRemoteUpdateDb(guid, true);
    INCREMENT_OWNER_UPDATES_PERFORMED_BY(1);
    INCREMENT_TWIN_DIFF_SKIPPED_BY(1);
  }
  return true;
}

void releaseDbs(unsigned int depc, artsEdtDep_t *depv, bool gpu) {
  for (int i = 0; i < depc; i++) {
    ARTS_DEBUG("Releasing DB[Guid:%lu] [Mode:%s]", depv[i].guid,
               getTypeName(depv[i].mode));
    unsigned int owner = artsGuidGetRank(depv[i].guid);

    /// Check compiler-inferred acquireMode if provided, otherwise use
    /// allocation mode
    artsType_t effectiveMode = artsGetEffectiveMode(&depv[i]);

    if (depv[i].guid != NULL_GUID && effectiveMode == ARTS_DB_WRITE) {
      if (depv[i].mode == ARTS_DB_PIN) {
        ARTS_DEBUG("Pinned DB write release (no frontier update)");
        artsDbDecrementLatch(depv[i].guid);
      } else if (depv[i].useTwinDiff) {
        artsTryReleaseTwinDiff(depv[i].guid, &depv[i]);
      } else if (owner == artsGlobalRankId) {
        struct artsDb *db = ((struct artsDb *)depv[i].ptr - 1);
        artsProgressFrontier(db, artsGlobalRankId);
        artsDbDecrementLatch(depv[i].guid);
      } else {
        struct artsDb *db = ((struct artsDb *)depv[i].ptr) - 1;
        if (db && (db->header.size - sizeof(struct artsDb)) == 176128) {
          ARTS_INFO("Release WRITE non-owner DB[Id:%lu, Guid:%lu, Size:%lu] "
                    "sending update to owner %u",
                    db->arts_id, depv[i].guid, db->header.size, owner);
        }
        artsRemoteUpdateDb(depv[i].guid, true);
        INCREMENT_OWNER_UPDATES_PERFORMED_BY(1);
        INCREMENT_TWIN_DIFF_SKIPPED_BY(1);
      }
    } else if (depv[i].guid != NULL_GUID && effectiveMode == ARTS_DB_READ) {
      // READ mode: NO latch decrement, NO owner update
      ARTS_DEBUG("DB[Guid:%lu] released in READ mode (no owner update, no "
                 "latch decrement)",
                 depv[i].guid);
      INCREMENT_OWNER_UPDATES_SAVED_BY(1);
    } else if (depv[i].mode == ARTS_DB_PIN) {
      artsDbDecrementLatch(depv[i].guid);
    } else if (depv[i].mode == ARTS_DB_ONCE_LOCAL ||
               depv[i].mode == ARTS_DB_ONCE) {
      artsRouteTableInvalidateItem(depv[i].guid);
    } else if (depv[i].mode == ARTS_PTR) {
      // Only free explicit buffers (guid == NULL). ESD slices point into DBs.
      if (depv[i].guid == NULL_GUID)
        artsFree(depv[i].ptr);
    } else if (!gpu && depv[i].mode == ARTS_DB_LC) {
      struct artsDb *db = ((struct artsDb *)depv[i].ptr) - 1;
      artsReaderUnlock(&db->reader);
    } else {
      if (artsRouteTableReturnDb(depv[i].guid, depv[i].mode != ARTS_DB_PIN)) {
        ARTS_DEBUG("FREED A COPY - DB[Guid:%lu]", depv[i].guid);
      }
    }
  }
}

bool artsAddDbDuplicate(struct artsDb *db, unsigned int rank,
                        struct artsEdt *edt, artsGuid_t edtGuid,
                        unsigned int slot, artsType_t mode) {
  bool write = (mode == ARTS_DB_WRITE);
  bool exclusive = false;
  if (edt && edtGuid == NULL_GUID)
    edtGuid = edt->currentEdt;
  return artsPushDbToList((struct artsDbList *)db->dbList, rank, write,
                          exclusive, artsGuidGetRank(db->guid) == rank, false,
                          edt, edtGuid, slot, mode);
}

void internalGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot,
                       unsigned int offset, unsigned int size,
                       unsigned int rank) {
  if (rank == artsGlobalRankId) {
    struct artsDb *db = (struct artsDb *)artsRouteTableLookupItem(dbGuid);
    if (db) {
      void *data = (void *)(((char *)(db + 1)) + offset);
      void *ptr = artsMalloc(size);
      memcpy(ptr, data, size);
      ARTS_INFO("Getting DB[Guid:%lu] From: %p", dbGuid, data);
      if (edtGuid != NULL_GUID)
        artsSignalEdtPtr(edtGuid, slot, ptr, size);
      artsMetricsTriggerEvent(artsGetBW, artsThread, size);
    } else {
      assert(edtGuid != NULL_GUID && "DB not found and no EDT to signal");
      ARTS_INFO("Getting OO-DB[Guid:%lu] From: %p", dbGuid, NULL);
      artsOutOfOrderGetFromDb(edtGuid, dbGuid, slot, offset, size);
    }
  } else {
    ARTS_DEBUG("Sending DB[Guid:%lu] to Rank %u", dbGuid, rank);
    assert(edtGuid != NULL_GUID && "DB not found and no EDT to signal");
    artsRemoteGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
  }
}

void artsGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot,
                   unsigned int offset, unsigned int size) {
  GET_DB_COUNTER_START();
  unsigned int rank = artsGuidGetRank(dbGuid);
  internalGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
  GET_DB_COUNTER_STOP();
}

void artsGetFromDbAt(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot,
                     unsigned int offset, unsigned int size,
                     unsigned int rank) {
  GET_DB_COUNTER_START();
  internalGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
  GET_DB_COUNTER_STOP();
}

void internalPutInDb(void *ptr, artsGuid_t edtGuid, artsGuid_t dbGuid,
                     unsigned int slot, unsigned int offset, unsigned int size,
                     artsGuid_t epochGuid, unsigned int rank) {
  if (rank == artsGlobalRankId) {
    struct artsDb *db = (struct artsDb *)artsRouteTableLookupItem(dbGuid);
    if (db) {
      // Do this so when we increment finished we can check the term status
      incrementQueueEpoch(epochGuid);
      globalShutdownGuidIncQueue();
      void *data = (void *)(((char *)(db + 1)) + offset);
      memcpy(data, ptr, size);
      if (edtGuid != NULL_GUID)
        artsSignalEdt(edtGuid, slot, dbGuid);
      incrementFinishedEpoch(epochGuid);
      globalShutdownGuidIncFinished();
      artsMetricsTriggerEvent(artsPutBW, artsThread, size);
    } else {
      void *cpyPtr = artsMalloc(size);
      memcpy(cpyPtr, ptr, size);
      artsOutOfOrderPutInDb(cpyPtr, edtGuid, dbGuid, slot, offset, size,
                            epochGuid);
    }
  } else {
    void *cpyPtr = artsMalloc(size);
    memcpy(cpyPtr, ptr, size);
    artsRemotePutInDb(cpyPtr, edtGuid, dbGuid, slot, offset, size, epochGuid,
                      rank);
  }
}

void artsPutInDbAt(void *ptr, artsGuid_t edtGuid, artsGuid_t dbGuid,
                   unsigned int slot, unsigned int offset, unsigned int size,
                   unsigned int rank) {
  PUT_DB_COUNTER_START();
  artsGuid_t epochGuid = artsGetCurrentEpochGuid();
  ARTS_DEBUG("Epoch [Guid:%lu]", epochGuid);
  incrementActiveEpoch(epochGuid);
  globalShutdownGuidIncActive();
  internalPutInDb(ptr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
  PUT_DB_COUNTER_STOP();
}

void artsPutInDb(void *ptr, artsGuid_t edtGuid, artsGuid_t dbGuid,
                 unsigned int slot, unsigned int offset, unsigned int size) {
  PUT_DB_COUNTER_START();
  unsigned int rank = artsGuidGetRank(dbGuid);
  artsGuid_t epochGuid = artsGetCurrentEpochGuid();
  ARTS_DEBUG("Epoch [Guid:%lu]", epochGuid);
  incrementActiveEpoch(epochGuid);
  globalShutdownGuidIncActive();
  internalPutInDb(ptr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
  PUT_DB_COUNTER_STOP();
}

void artsPutInDbEpoch(void *ptr, artsGuid_t epochGuid, artsGuid_t dbGuid,
                      unsigned int offset, unsigned int size) {
  PUT_DB_COUNTER_START();
  unsigned int rank = artsGuidGetRank(dbGuid);
  incrementActiveEpoch(epochGuid);
  globalShutdownGuidIncActive();
  internalPutInDb(ptr, NULL_GUID, dbGuid, 0, offset, size, epochGuid, rank);
  PUT_DB_COUNTER_STOP();
}
