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
#include "arts/introspection/Metrics.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/RT.h"
#include "arts/runtime/compute/EdtFunctions.h"
#include "arts/runtime/memory/DbList.h"
#include "arts/runtime/network/RemoteFunctions.h"
#include "arts/runtime/sync/TerminationDetection.h"
#include "arts/system/ArtsPrint.h"
#include "arts/utils/Atomics.h"

#ifdef USE_GPU
#include "arts/gpu/GpuRuntime.cuh"
#endif

artsTypeName;

// inline void * artsDbMalloc(artsType_t mode, unsigned int size)
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
                          uint64_t packetSize, artsType_t mode) {
  struct artsHeader *header = (struct artsHeader *)addr;
  header->type = mode;
  header->size = packetSize;

  struct artsDb *dbRes = (struct artsDb *)header;
  dbRes->guid = guid;
  dbRes->version = 0;
  dbRes->reader = 0;
  dbRes->writer = 0;
  dbRes->copyCount = 0;
  if (mode != ARTS_DB_PIN) {
    dbRes->dbList = artsNewDbList();
  }
  if (mode == ARTS_DB_LC) {
    void *shadowCopy = (void *)(((char *)addr) + packetSize);
    memcpy(shadowCopy, addr, sizeof(struct artsDb));
  }
#ifdef USE_SMART_DB
  dbRes->eventGuid = artsPersistentEventCreate(artsGuidGetRank(guid), 0, guid);
#endif
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
  unsigned int dbSize = size + sizeof(struct artsDb);

  void *ptr = artsMallocWithType(dbSize, artsDbMemorySize);
  if (ptr) {
    guid = artsGuidCreateForRank(artsGlobalRankId, mode);
    artsDbCreateInternal(guid, ptr, size, dbSize, mode);
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
  unsigned int dbSize = size + sizeof(struct artsDb);

  void *ptr = artsMallocWithType(dbSize, artsDbMemorySize);
  if (ptr) {
    guid = artsGuidCreateForRank(artsGlobalRankId, mode);
    artsDbCreateInternal(guid, ptr, size, dbSize, mode);
    // change false to true to force a manual DB delete
    artsRouteTableAddItem(ptr, guid, artsGlobalRankId, false);
    *addr = (artsPtr_t)((struct artsDb *)ptr + 1);
  }
  DB_CREATE_COUNTER_STOP();
  return guid;
}

// Guid must be for a local DB only
void *artsDbCreateWithGuid(artsGuid_t guid, uint64_t size) {
  DB_CREATE_COUNTER_START();
  artsType_t mode = artsGuidGetType(guid);

  void *ptr = NULL;
  if (artsIsGuidLocal(guid)) {
    unsigned int dbSize = size + sizeof(struct artsDb);

    ptr = artsMallocWithType(dbSize, artsDbMemorySize);
    if (ptr) {
      artsDbCreateInternal(guid, ptr, size, dbSize, mode);
      if (artsRouteTableAddItemRace(ptr, guid, artsGlobalRankId, false)) {
        artsRouteTableFireOO(guid, artsOutOfOrderHandler);
      }
      ptr = (void *)((struct artsDb *)ptr + 1);
    }
  }
  ARTS_INFO("Creating DB [Guid: %lu] [Mode: %s] [Ptr: %p] [Route: %d]", guid,
            getTypeName(mode), ptr, artsGuidGetRank(guid));
  DB_CREATE_COUNTER_STOP();
  return ptr;
}

void *artsDbCreateWithGuidAndData(artsGuid_t guid, void *data, uint64_t size) {
  DB_CREATE_COUNTER_START();
  artsType_t mode = artsGuidGetType(guid);
  void *ptr = NULL;
  if (artsIsGuidLocal(guid)) {
    unsigned int dbSize = size + sizeof(struct artsDb);

    ptr = artsMallocWithType(dbSize, artsDbMemorySize);

    if (ptr) {
      artsDbCreateInternal(guid, ptr, size, dbSize, mode);
      void *dbData = (void *)((struct artsDb *)ptr + 1);
      memcpy(dbData, data, size);
      if (artsRouteTableAddItemRace(ptr, guid, artsGlobalRankId, false))
        artsRouteTableFireOO(guid, artsOutOfOrderHandler);
      ptr = dbData;
    }
  }
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

#ifdef USE_SMART_DB
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
#endif

/**********************DB MEMORY MODEL*************************************/
// Side Effects: edt depcNeeded will be incremented, ptr will be updated,
//   and launches out of order handleReadyEdt
// Returns false on out of order and true otherwise
void acquireDbs(struct artsEdt *edt) {
  artsEdtDep_t *depv = (artsEdtDep_t *)artsGetDepv(edt);
  edt->depcNeeded = edt->depc + 1;
  ARTS_INFO(
      "Acquiring %u DBs for EDT [Guid: %lu], depcNeeded initialized to %u",
      edt->depc, edt->currentEdt, edt->depcNeeded);
  for (int i = 0; i < edt->depc; i++) {
    if (depv[i].guid && depv[i].ptr == NULL) {
      struct artsDb *dbFound = NULL;
      int owner = artsGuidGetRank(depv[i].guid);
      ARTS_INFO("Slot %d: DB [Guid: %lu, mode=%u, owner=%d, currentRank=%u]", i,
                depv[i].guid, depv[i].mode, owner, artsGlobalRankId);
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
            ARTS_DEBUG("DB [Guid: %lu] out of order request for LC_SYNC not "
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
            if (artsAddDbDuplicate(dbTemp, artsGlobalRankId, edt, i,
                                   depv[i].mode)) {
              ARTS_DEBUG("Adding duplicate DB [Guid: %lu]", depv[i].guid);
              // Owner rank and we have the valid copy
              if (validRank == artsGlobalRankId) {
                dbFound = dbTemp;
                artsAtomicSub(&edt->depcNeeded, 1U);
              }
              // Owner rank but someone else has valid copy
              else {
                if (depv[i].mode == ARTS_DB_READ ||
                    depv[i].mode == ARTS_DB_GPU_READ ||
                    depv[i].mode == ARTS_DB_GPU_WRITE ||
                    depv[i].mode == ARTS_DB_LC ||
                    depv[i].mode == ARTS_DB_LC_NO_COPY ||
                    depv[i].mode == ARTS_DB_GPU_MEMSET)
                  artsRemoteDbRequest(depv[i].guid, validRank, edt, i,
                                      depv[i].mode, true);
                else
                  artsRemoteDbFullRequest(depv[i].guid, validRank, edt, i,
                                          depv[i].mode);
              }
            } else {
              ARTS_DEBUG("Duplicate not added DB [Guid: %lu]", depv[i].guid);
            }
          }
          // The Db hasn't been created yet
          else {
            ARTS_DEBUG("DB [Guid: %lu] out of order request slot %u",
                       depv[i].guid, i);
            artsOutOfOrderHandleDbRequest(depv[i].guid, edt, i, true);
          }
        } else {
          int validRank = -1;
          struct artsDb *dbTemp = (struct artsDb *)artsRouteTableLookupDb(
              depv[i].guid, &validRank, true);
          // We have found an entry
          ARTS_INFO("acquireDbs: Non-owner case for DB [Guid: %lu, mode=%u, "
                    "owner=%d, validRank=%d, dbTemp=%p]",
                    depv[i].guid, depv[i].mode, owner, validRank,
                    (void *)dbTemp);
          if (dbTemp) {
            dbFound = dbTemp;
            artsAtomicSub(&edt->depcNeeded, 1U);
            ARTS_INFO("  Found local copy, decremented depcNeeded");
          }

          if (depv[i].mode == ARTS_DB_WRITE) {
            // We can't aggregate read requests for cdag write
            ARTS_INFO("  WRITE mode - sending full DB request to rank %d",
                      owner);
            artsRemoteDbFullRequest(depv[i].guid, owner, edt, i, depv[i].mode);
          } else if (!dbTemp) {
            // We can aggregate read requests for reads
            ARTS_INFO("  READ mode, no local copy - sending aggregated request "
                      "to rank %d",
                      owner);
            artsRemoteDbRequest(depv[i].guid, owner, edt, i, depv[i].mode,
                                true);
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
      ARTS_DEBUG("DB [Guid: %lu, Ptr: %p] acquired", depv[i].guid, depv[i].ptr);
    } else {
      artsAtomicSub(&edt->depcNeeded, 1U);
    }
  }
  ARTS_INFO("EDT [Guid: %lu] has finished acquiring DBs", edt->currentEdt);
}

void prepDbs(unsigned int depc, artsEdtDep_t *depv, bool gpu) {
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].guid != NULL_GUID && depv[i].mode == ARTS_DB_WRITE) {
      artsRemoteUpdateRouteTable(depv[i].guid, -1);
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

void releaseDbs(unsigned int depc, artsEdtDep_t *depv, bool gpu) {
  for (int i = 0; i < depc; i++) {
    ARTS_DEBUG("Releasing DB [Guid: %lu] [Mode: %s]", depv[i].guid,
               getTypeName(depv[i].mode));
    unsigned int owner = artsGuidGetRank(depv[i].guid);
    if (depv[i].guid != NULL_GUID && depv[i].mode == ARTS_DB_WRITE) {
      if (owner == artsGlobalRankId) {
        struct artsDb *db = ((struct artsDb *)depv[i].ptr - 1);
        artsProgressFrontier(db, artsGlobalRankId);
#ifdef USE_SMART_DB
        artsDbDecrementLatch(depv[i].guid);
#endif
      } else {
        artsRemoteUpdateDb(depv[i].guid, true);
      }
    } else if (depv[i].guid != NULL_GUID && depv[i].mode == ARTS_DB_READ) {
      artsDbDecrementLatch(depv[i].guid);
    } else if (depv[i].mode == ARTS_DB_PIN) {
#ifdef USE_SMART_DB
      artsDbDecrementLatch(depv[i].guid);
#endif
    } else if (depv[i].mode == ARTS_DB_ONCE_LOCAL ||
               depv[i].mode == ARTS_DB_ONCE) {
      artsRouteTableInvalidateItem(depv[i].guid);
    } else if (depv[i].mode == ARTS_PTR) {
      artsFree(depv[i].ptr);
    } else if (!gpu && depv[i].mode == ARTS_DB_LC) {
      struct artsDb *db = ((struct artsDb *)depv[i].ptr) - 1;
      artsReaderUnlock(&db->reader);
    } else {
      if (artsRouteTableReturnDb(depv[i].guid, depv[i].mode != ARTS_DB_PIN)) {
        ARTS_DEBUG("FREED A COPY - DB [Guid: %lu]", depv[i].guid);
      }
    }
  }
}

bool artsAddDbDuplicate(struct artsDb *db, unsigned int rank,
                        struct artsEdt *edt, unsigned int slot,
                        artsType_t mode) {
  bool write = (mode == ARTS_DB_WRITE);
  bool exclusive = false;
  return artsPushDbToList((struct artsDbList *)db->dbList, rank, write,
                          exclusive, artsGuidGetRank(db->guid) == rank, false,
                          edt, slot, mode);
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
      ARTS_INFO("Getting DB [Guid: %lu] From: %p", dbGuid, data);
      if (edtGuid != NULL_GUID)
        artsSignalEdtPtr(edtGuid, slot, ptr, size);
      artsMetricsTriggerEvent(artsGetBW, artsThread, size);
    } else {
      assert(edtGuid != NULL_GUID && "DB not found and no EDT to signal");
      ARTS_INFO("Getting OO-DB [Guid: %lu] From: %p", dbGuid, NULL);
      artsOutOfOrderGetFromDb(edtGuid, dbGuid, slot, offset, size);
    }
  } else {
    ARTS_DEBUG("Sending DB [Guid: %lu] to Rank %u", dbGuid, rank);
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
  ARTS_DEBUG("Epoch [Guid: %lu]", epochGuid);
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
  ARTS_DEBUG("Epoch [Guid: %lu]", epochGuid);
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
