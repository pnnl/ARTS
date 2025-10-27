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
#include "arts/gas/Guid.h"

#include "arts/arts.h"
#include "arts/runtime/Globals.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"

uint64_t numTables = 0;
uint64_t keysPerThread = 0;
uint64_t globalGuidOn = 0;
uint64_t minGlobalGuidThread = 0;
uint64_t maxGlobalGuidThread = 0;

void setGlobalGuidOn() { globalGuidOn = ((uint64_t)1) << 40; }

uint64_t *artsGuidGeneratorGetKey(unsigned int route, unsigned int type) {
  return &artsNodeInfo
              .keys[artsThreadInfo.groupId][route * ARTS_LAST_TYPE + type];
}

artsGuid_t artsGuidCreateForRankInternal(unsigned int route, unsigned int type,
                                         unsigned int guidCount) {
  artsGuid guid;
  if (globalGuidOn) {
    // Safeguard against wrap around
    if (globalGuidOn > guidCount) {
      guid.fields.key = globalGuidOn - guidCount;
      globalGuidOn -= guidCount;
    } else {
      ARTS_INFO("Parallel Start out of guid keys");
      artsDebugGenerateSegFault();
    }
  } else {
    uint64_t *key = artsGuidGeneratorGetKey(route, type);
    uint64_t value = *key;
    if (value + guidCount < keysPerThread) {
      guid.fields.key =
          value + keysPerThread *
                      artsNodeInfo.globalGuidThreadId[artsThreadInfo.groupId];
      (*key) += guidCount;
    } else {
      ARTS_INFO("Out of guid keys");
      artsDebugGenerateSegFault();
    }
  }
  guid.fields.type = type;
  guid.fields.rank = route;
  INCREMENT_GUID_ALLOC_COUNTER_BY(guidCount);
  return (artsGuid_t)guid.bits;
}

artsGuid_t artsGuidCreateForRank(unsigned int route, unsigned int type) {
  return artsGuidCreateForRankInternal(route, type, 1);
}

void setGuidGeneratorAfterParallelStart() {
  unsigned int numOfTables = artsNodeInfo.workerThreadCount + 1;
  keysPerThread = globalGuidOn / (numOfTables * artsGlobalRankCount);
  globalGuidOn = 0;
}

void artsGuidKeyGeneratorInit() {
  numTables = (artsGlobalRankCount == 1) ? artsNodeInfo.workerThreadCount
                                         : artsNodeInfo.workerThreadCount + 1;
  uint64_t localId = (artsThreadInfo.worker) ? artsThreadInfo.threadId
                                             : artsNodeInfo.workerThreadCount;
  minGlobalGuidThread = numTables * artsGlobalRankId;
  maxGlobalGuidThread = (artsGlobalRankCount == 1)
                            ? minGlobalGuidThread + numTables
                            : minGlobalGuidThread + numTables - 1;
  //    globalGuidThreadId  = minGlobalGuidThread + localId;
  artsNodeInfo.globalGuidThreadId[artsThreadInfo.groupId] =
      minGlobalGuidThread + localId;

  //    ARTS_INFO("numTables: %lu localId: %lu minGlobalGuidThread: %lu
  //    maxGlobalGuidThread: %lu globalGuidThreadId: %lu", numTables, localId,
  //    minGlobalGuidThread, maxGlobalGuidThread, globalGuidThreadId); keys =
  //    artsMalloc(sizeof(uint64_t) * ARTS_LAST_TYPE * artsGlobalRankCount);
  artsNodeInfo.keys[artsThreadInfo.groupId] = (uint64_t *)artsMalloc(
      sizeof(uint64_t) * ARTS_LAST_TYPE * artsGlobalRankCount);
  for (unsigned int i = 0; i < ARTS_LAST_TYPE * artsGlobalRankCount; i++)
    artsNodeInfo.keys[artsThreadInfo.groupId][i] = 1;
  //        keys[i] = 1;
}

// I think this might be a problem.  If all keys start at 1 then when we cast
// type, it can override a legitimate guid of another type
artsGuid_t artsGuidCast(artsGuid_t guid, artsType_t type) {
  artsGuid addressInfo = (artsGuid){.bits = guid};
  addressInfo.fields.type = (unsigned int)type;
  return (artsGuid_t)addressInfo.bits;
}

artsType_t artsGuidGetType(artsGuid_t guid) {
  INCREMENT_GUID_LOOKUP_COUNTER_BY(1);
  artsGuid addressInfo = (artsGuid){.bits = guid};
  return (artsType_t)addressInfo.fields.type;
}

unsigned int artsGuidGetRank(artsGuid_t guid) {
  INCREMENT_GUID_LOOKUP_COUNTER_BY(1);
  artsGuid addressInfo = (artsGuid){.bits = guid};
  return addressInfo.fields.rank;
}

bool artsIsGuidLocal(artsGuid_t guid) {
  return (artsGlobalRankId == artsGuidGetRank(guid));
}

uint64_t artsGetGuidKey(artsGuid_t guid) {
  artsGuid addressInfo = (artsGuid){.bits = guid};
  return addressInfo.fields.key;
}

artsGuid_t artsReserveGuidRoute(artsType_t type, unsigned int route) {
  artsGuid_t guid = NULL_GUID;
  if (route == -1)
    route = artsGlobalRankId;
  route = route % artsGlobalRankCount;
  if (type > ARTS_NULL && type < ARTS_LAST_TYPE) {
    guid = artsGuidCreateForRankInternal(route, (unsigned int)type, 1);
    // ARTS_INFO("Allocation Guid %u", guid);
  } else {
    ARTS_INFO("Invalid type %u", type);
  }
  //    if(route == artsGlobalRankId)
  //        artsRouteTableAddItem(NULL, guid, artsGlobalRankId, false);
  return guid;
}

artsGuid_t *artsReserveGuidsRoundRobin(unsigned int size, artsType_t type) {
  artsGuid_t *guids = NULL;
  if (type > ARTS_NULL && type < ARTS_LAST_TYPE) {
    guids = (artsGuid_t *)artsMalloc(size * sizeof(artsGuid_t));
    for (unsigned int i = 0; i < size; i++) {
      unsigned int route = i % artsGlobalRankCount;
      guids[i] = artsGuidCreateForRank(route, (unsigned int)type);
    }
  }
  return guids;
}

artsGuidRange *artsNewGuidRangeNode(artsType_t type, unsigned int size,
                                    unsigned int route) {
  if (route == -1)
    route = artsGlobalRankId;
  artsGuidRange *range = NULL;
  if (size && type > ARTS_NULL && type < ARTS_LAST_TYPE) {
    range = (artsGuidRange *)artsCalloc(1, sizeof(artsGuidRange));
    range->size = size;
    range->startGuid =
        artsGuidCreateForRankInternal(route, (unsigned int)type, size);
    //        if(artsIsGuidLocalExt(range->startGuid))
    //        {
    //            artsGuid temp = (artsGuid) range->startGuid;
    //            for(unsigned int i=0; i<size; i++)
    //            {
    //                artsRouteTableAddItem(NULL, temp.bits, route);
    //                temp.fields.key++;
    //            }
    //        }
  }
  return range;
}

artsGuidRange *artsNewGuidRangeNodeHash(artsType_t type, unsigned int size,
                                        unsigned int route,
                                        unsigned int hashSize) {
  artsGuidRange *range = NULL;
  if (size && type > ARTS_NULL && type < ARTS_LAST_TYPE) {
    range = (artsGuidRange *)artsCalloc(1, sizeof(artsGuidRange));
    range->size = size;
    range->startGuid = artsGuidCreateForRankInternal(route, (unsigned int)type,
                                                     size + hashSize);
    artsGuid temp = (artsGuid){.bits = range->startGuid};
    for (unsigned int i = 0; i < hashSize; i++) {
      if (temp.fields.key % hashSize == 0) {
        range->startGuid = (artsGuid_t)temp.bits;
        break;
      }
      temp.fields.key++;
    }
  }
  return range;
}

artsGuid_t artsGetGuid(artsGuidRange *range, unsigned int index) {
  if (!range || index >= range->size) {
    return NULL_GUID;
  }
  artsGuid ret = (artsGuid){.bits = range->startGuid};
  ret.fields.key += index;
  return ret.bits;
}

artsGuid_t artsGuidRangeNext(artsGuidRange *range) {
  artsGuid_t ret = NULL_GUID;
  if (range) {
    if (range->index < range->size)
      ret = artsGetGuid(range, range->index++);
  }
  return ret;
}

bool artsGuidRangeHasNext(artsGuidRange *range) {
  if (range)
    return (range->index < range->size);
  return false;
}

void artsGuidRangeResetIter(artsGuidRange *range) {
  if (range)
    range->index = 0;
}

bool artsIsInGuidRange(artsGuidRange *range, artsGuid_t guid) {
  artsGuid startGuid = (artsGuid){.bits = range->startGuid};
  artsGuid toCheck = (artsGuid){.bits = guid};

  if (startGuid.fields.rank != toCheck.fields.rank)
    return false;

  if (startGuid.fields.type != toCheck.fields.type)
    return false;

  if (startGuid.fields.key <= toCheck.fields.key &&
      toCheck.fields.key < startGuid.fields.key + range->index)
    return true;

  return false;
}

uint64_t artsHashGuidKey(artsGuid_t guid) {
  uint64_t key = artsGetGuidKey(guid);
  return key % (uint64_t)artsNodeInfo.gpu;
}
