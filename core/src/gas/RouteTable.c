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

#include "arts/gas/RouteTable.h"

#include <stdlib.h>

#include "arts/arts.h"
#include "arts/gas/Guid.h"
#include "arts/gas/OutOfOrder.h"
#include "arts/runtime/Globals.h"
#include "arts/runtime/memory/DbFunctions.h"
#include "arts/runtime/memory/DbList.h"
#include "arts/system/ArtsPrint.h"
#include "arts/system/Debug.h"
#include "arts/utils/Atomics.h"

#define initInvalidateSize 128
#define guidLockSize 1024
volatile unsigned int guidLock[guidLockSize] = {0};

void setItem(artsRouteItem_t *item, void *data) { item->data = data; }

void freeItem(artsRouteItem_t *item) {
  // artsType_t type = artsGuidGetType(item->key);
  // if (type > ARTS_BUFFER && type < ARTS_LAST_TYPE)
  //   artsDbFree(item->data);
  // else
  //   artsFree(item->data);
  artsOutOfOrderListDelete(&item->ooList);
  item->data = NULL;
  item->key = 0;
  item->lock = 0;
  item->touched = 0;
}

bool markReserve(artsRouteItem_t *item, bool markUse) {
  if (markUse) {
    uint64_t mask = reservedItem + 1;
    return !artsAtomicCswapU64(&item->lock, 0, mask);
  }
  return !artsAtomicFetchOrU64(&item->lock, reservedItem);
}

bool markRequested(artsRouteItem_t *item) {
  uint64_t local, temp;
  while (1) {
    local = item->lock;
    if ((local & reservedItem) || (local & deleteItem)) {
      return false;
    }
    temp = local | reservedItem;
    if (local == artsAtomicCswapU64(&item->lock, local, temp))
      return true;
  }
}

bool markWrite(artsRouteItem_t *item) {
  uint64_t local, temp;
  while (1) {
    local = item->lock;
    if (local & reservedItem) {
      temp = (local & ~reservedItem) | availableItem;
      if (local == artsAtomicCswapU64(&item->lock, local, temp))
        return true;
    } else
      return false;
  }
}

bool markDelete(artsRouteItem_t *item) {
  uint64_t res = artsAtomicFetchOrU64(&item->lock, deleteItem);
  return (res & deleteItem) != 0;
}

bool tryMarkDelete(artsRouteItem_t *item, uint64_t countVal) {
  uint64_t compVal = availableItem + countVal;
  uint64_t newVal = (availableItem | deleteItem);
  uint64_t oldVal = artsAtomicCswapU64(&item->lock, compVal, newVal);
  return (compVal == oldVal);
}

void printState(artsRouteItem_t *item) {
  if (item) {
    uint64_t local = item->lock;
    if (isReq(local)) {
      ARTS_INFO("%lu: reserved-available %p %s", item->key, local,
                getTypeName(artsGuidGetType(item->key)));
    } else if (isRes(local)) {
      ARTS_INFO("%lu: reserved %p %s", item->key, local,
                getTypeName(artsGuidGetType(item->key)));
    } else if (isAvail(local)) {
      ARTS_INFO("%lu: available %p %s", item->key, local,
                getTypeName(artsGuidGetType(item->key)));
    } else if (isDel(local)) {
      ARTS_INFO("%lu: deleted %p %s", item->key, local,
                getTypeName(artsGuidGetType(item->key)));
    }
  } else {
    ARTS_INFO("NULL ITEM");
  }
}

// 11000 & 11100 = 11000, 10000 & 11100 = 10000, 11100 & 11100 = 11000
bool checkItemState(artsRouteItem_t *item, itemState_t state) {
  if (item) {
    uint64_t local = item->lock;
    switch (state) {
    case reservedKey:
      return isRes(local);

    case requestedKey:
      return isReq(local);

    case availableKey:
      return isAvail(local);

    case allocatedKey:
      return isRes(local) || isAvail(local) || isReq(local);

    case deletedKey:
      return isDel(local);

    case anyKey:
      return local != 0;

    default:
      return false;
    }
  }
  return false;
}

inline bool checkMinItemState(artsRouteItem_t *item, itemState_t state) {
  if (item) {
    uint64_t local = item->lock;
    itemState_t actualState = noKey;

    if (isDel(local))
      actualState = deletedKey;

    else if (isRes(local))
      actualState = reservedKey;

    else if (isReq(local))
      actualState = requestedKey;

    else if (isAvail(local))
      actualState = availableKey;

    return (actualState && actualState >= state);
  }
  return false;
}

itemState_t getItemState(artsRouteItem_t *item) {
  if (item) {
    uint64_t local = item->lock;

    if (isRes(local))
      return reservedKey;
    if (isAvail(local))
      return availableKey;

    if (isReq(local))
      return requestedKey;

    if (isDel(local))
      return deletedKey;
  }
  return noKey;
}

bool incItem(artsRouteItem_t *item, unsigned int count, artsGuid_t key,
             artsRouteTable_t *routeTable) {
  while (1) {
    uint64_t local = item->lock;
    if (!(local & deleteItem) && checkMaxItem(local) && item->key == key) {
      if (local == artsAtomicCswapU64(&item->lock, local, local + count)) {
        if (item->key != key) // This is for an ABA problem
        {
          ARTS_DEBUG("The key changed on us from %lu -> %lu", key, item->key);
          decItem(routeTable, item);
          return false;
        }
        return true;
      }
    } else
      break;
  }
  return false;
}

bool decItem(artsRouteTable_t *routeTable, artsRouteItem_t *item) {
  uint64_t local = artsAtomicSubU64(&item->lock, 1);
  if (getCount(local) == 0) {
    if (shouldDelete(local)) {
      routeTable->freeFunc(item);
      return true;
    }
  }
  return false;
}

uint64_t urand64() {
  uint64_t hi = lrand48();
  uint64_t md = lrand48();
  uint64_t lo = lrand48();
  uint64_t res = (hi << 42) + (md << 21) + lo;
  return res;
}

#define hash64(x, y) ((uint64_t)(x) * y)

static inline uint64_t getRouteTableKey(uint64_t x, unsigned int shift) {
  uint64_t hash = 14695981039346656037U;
  switch (shift) {
  /*case 5:
      hash *= 31;
  case 6:
      hash *= 61;
  case 7:
      hash *= 127;
  case 8:
      hash *= 251;
  case 9:
      hash *= 509;*/
  case 10:
    hash *= 1021;
  case 11:
    hash *= 2039;
  case 12:
    hash *= 4093;
  case 13:
    hash *= 8191;
  case 14:
    hash *= 16381;
  case 15:
    hash *= 32749;
  case 16:
    hash *= 65521;
  case 17:
    hash *= 131071;
  case 18:
    hash *= 262139;
  case 19:
    hash *= 524287;
  case 20:
    hash *= 1048573;
  case 21:
    hash *= 2097143;
  case 22:
    hash *= 4194301;
  case 31:
    hash *= 2147483647;
  case 32:
    hash *= 4294967291;
  }

  return (hash64(x, hash) >> (64 - shift)) * collisionResolves;
}
extern uint64_t numTables;
extern uint64_t maxGuid;
extern uint64_t keysPerThread;
extern uint64_t minGlobalGuidThread;
extern uint64_t maxGlobalGuidThread;

static inline artsRouteTable_t *artsGetRouteTable(artsGuid_t guid) {
  artsGuid raw = (artsGuid){.bits = guid};
  uint64_t key = raw.fields.key;
  if (keysPerThread) {
    uint64_t globalThread = (key / keysPerThread);
    if (minGlobalGuidThread <= globalThread &&
        globalThread < maxGlobalGuidThread)
      return artsNodeInfo.routeTable[globalThread - minGlobalGuidThread];
  }
  return artsNodeInfo.remoteRouteTable;
}

artsRouteTable_t *artsNewRouteTable(unsigned int routeTableSize,
                                    unsigned int shift) {
  artsRouteTable_t *routeTable =
      (artsRouteTable_t *)artsCalloc(1, sizeof(artsRouteTable_t));
  routeTable->data = (artsRouteItem_t *)artsCallocAlign(
      collisionResolves * routeTableSize, sizeof(artsRouteItem_t), 16);
  routeTable->size = routeTableSize;
  routeTable->shift = shift;
  routeTable->setFunc = setItem;
  routeTable->freeFunc = freeItem;
  routeTable->newFunc = artsNewRouteTable;
  return routeTable;
}

artsRouteItem_t *artsRouteTableSearchForKey(artsRouteTable_t *routeTable,
                                            artsGuid_t key, itemState_t state) {
  artsRouteTable_t *current = routeTable;
  artsRouteTable_t *next;
  uint64_t keyVal;
  while (current) {
    keyVal = getRouteTableKey((uint64_t)key, current->shift);
    for (int i = 0; i < collisionResolves; i++) {
      if (checkItemState(&current->data[keyVal], state)) {
        if (current->data[keyVal].key == key) {
          return &current->data[keyVal];
        }
      }
      keyVal++;
    }
    artsReaderLock(&current->readerLock, &current->writerLock);
    next = current->next;
    artsReaderUnlock(&current->readerLock);
    current = next;
  }
  return NULL;
}

artsRouteItem_t *artsRouteTableSearchForEmpty(artsRouteTable_t *routeTable,
                                              artsGuid_t key, bool markUsed) {
  artsRouteTable_t *current = routeTable;
  artsRouteTable_t *next;
  uint64_t keyVal;
  while (current != NULL) {
    keyVal = getRouteTableKey((uint64_t)key, current->shift);
    for (int i = 0; i < collisionResolves; i++) {
      if (!current->data[keyVal].lock) {
        if (markReserve(&current->data[keyVal], markUsed)) {
          current->data[keyVal].key = key;
          return &current->data[keyVal];
        }
      }
      keyVal++;
    }

    artsReaderLock(&current->readerLock, &current->writerLock);
    next = current->next;
    artsReaderUnlock(&current->readerLock);

    if (!next) {
      if (artsWriterTryLock(&current->readerLock, &current->writerLock)) {
        next = current->next =
            current->newFunc(2 * current->size, current->shift + 1);
        artsWriterUnlock(&current->writerLock);
      } else {
        artsReaderLock(&current->readerLock, &current->writerLock);
        next = current->next;
        artsReaderUnlock(&current->readerLock);
      }
    }
    current = next;
  }
  PRINTF(
      "Route table search in impossible state: producing a segfault now %p ...",
      routeTable);
  artsDebugGenerateSegFault();
  return NULL;
}

void *internalRouteTableAddItem(artsRouteTable_t *routeTable, void *item,
                                artsGuid_t key, unsigned int rank, bool used) {
  artsRouteItem_t *location =
      artsRouteTableSearchForEmpty(routeTable, key, used);
  routeTable->setFunc(location, item);
  location->rank = rank;
  markWrite(location);
  return location;
}

void *artsRouteTableAddItem(void *item, artsGuid_t key, unsigned int rank,
                            bool used) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  return internalRouteTableAddItem(routeTable, item, key, rank, used);
}

bool internalRouteTableRemoveItem(artsRouteTable_t *routeTable,
                                  artsGuid_t key) {
  artsRouteItem_t *item =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (item) {
    markDelete(item);
    if (shouldDelete(item->lock)) {
      routeTable->freeFunc(item);
    }
  }
  return 0;
}

bool artsRouteTableRemoveItem(artsGuid_t key) {
  // artsRouteTable_t *routeTable = artsGetRouteTable(key);
  // return internalRouteTableRemoveItem(routeTable, key);
  return artsRouteTableInvalidateItem(key);
}

// This just doesn't delete the item itself... It is for DB rename
bool artsRouteTableHideItem(artsGuid_t key) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *item =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (item) {
    item->data = NULL;
  }
  return 0;
}

// This locks the guid so it is useful when multiple people have the guid ahead
// of time The guid doesn't need to be locked if no one knows about it
artsRouteItem_t *internalRouteTableAddItemRace(bool *addedItem,
                                               artsRouteTable_t *routeTable,
                                               void *item, artsGuid_t key,
                                               unsigned int rank, bool usedRes,
                                               bool usedAvail,
                                               unsigned int toAddOnCreation) {
  unsigned int pos = (unsigned int)(((uint64_t)key) % (uint64_t)guidLockSize);
  *addedItem = false;
  artsRouteItem_t *found = NULL;
  while (!found) {
    if (guidLock[pos] == 0) {
      if (!artsAtomicCswap(&guidLock[pos], 0U, 1U)) {
        found = artsRouteTableSearchForKey(routeTable, key, allocatedKey);
        if (found) {
          if (checkItemState(found, reservedKey)) {
            routeTable->setFunc(found, item);
            found->rank = rank;
            markWrite(found);
            if (usedRes)
              incItem(found, 1, found->key, routeTable);
            *addedItem = true;
          } else if (usedAvail && checkItemState(found, availableKey))
            incItem(found, 1, found->key, routeTable);
        } else {
          found = (artsRouteItem_t *)internalRouteTableAddItem(
              routeTable, item, key, rank, usedRes);
          if (toAddOnCreation)
            incItem(found, toAddOnCreation, found->key, routeTable);
          *addedItem = true;
        }
        guidLock[pos] = 0U;
      }
    } else {
      found = artsRouteTableSearchForKey(routeTable, key, availableKey);
      if (found && usedAvail)
        incItem(found, 1, found->key, routeTable);
    }
  }
  //    ARTS_INFO("found: %lu %p", key, found);
  return found;
}

artsRouteItem_t *
internalRouteTableAddDeletedItemRace(artsRouteTable_t *routeTable, void *item,
                                     artsGuid_t key, unsigned int rank) {
  unsigned int pos = (unsigned int)(((uint64_t)key) % (uint64_t)guidLockSize);
  artsRouteItem_t *found = NULL;
  while (!found) {
    if (guidLock[pos] == 0) {
      if (!artsAtomicCswap(&guidLock[pos], 0U, 1U)) {
        found = artsRouteTableSearchForEmpty(routeTable, key, false);
        routeTable->setFunc(found, item);
        found->rank = rank;
        markDelete(found);
        markWrite(found);
        guidLock[pos] = 0U;
      }
    }
  }
  return found;
}

bool artsRouteTableAddItemRace(void *item, artsGuid_t key, unsigned int rank,
                               bool used) {
  bool ret;
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  internalRouteTableAddItemRace(&ret, routeTable, item, key, rank, used, false,
                                0);
  return ret;
}

// This is used for the send aggregation
bool artsRouteTableReserveItemRace(artsGuid_t key, artsRouteItem_t **item,
                                   bool used) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  unsigned int pos = (unsigned int)(((uint64_t)key) % (uint64_t)guidLockSize);
  bool ret = false;
  *item = NULL;
  while (!(*item)) {
    if (guidLock[pos] == 0) {
      if (!artsAtomicCswap(&guidLock[pos], 0U, 1U)) {
        *item = artsRouteTableSearchForKey(routeTable, key, allocatedKey);
        if (!(*item)) {
          *item = artsRouteTableSearchForEmpty(routeTable, key, used);
          ret = true;
        } else {
          if (used)
            incItem(*item, 1, (*item)->key, routeTable);
        }
        guidLock[pos] = 0U;
      }
    } else {
      artsRouteItem_t *temp =
          artsRouteTableSearchForKey(routeTable, key, allocatedKey);
      if (temp && used)
        incItem(temp, 1, temp->key, routeTable);
      *item = temp;
    }
  }
  //    printState(artsRouteTableSearchForKey(routeTable, key, anyKey));
  return ret;
}

// This does the send aggregation
bool artsRouteTableAddSent(artsGuid_t key, void *edt, unsigned int slot,
                           bool aggregate) {
  artsRouteItem_t *item = NULL;
  bool sendReq;
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  // I shouldn't be able to get to here if the db hasn't already been created
  // and I am the owner node thus item can't be null... or so it should be
  if (artsGuidGetRank(key) == artsGlobalRankId) {
    item = artsRouteTableSearchForKey(routeTable, key, allocatedKey);
    sendReq = markRequested(item);
  } else {
    sendReq = artsRouteTableReserveItemRace(key, &item, true);
    if (!sendReq && !incItem(item, 1, item->key, routeTable)) {
      ARTS_INFO("Item marked for deletion before it has arrived %u...",
                sendReq);
    }
  }
  artsOutOfOrderHandleDbRequestWithOOList(&item->ooList, &item->data,
                                          (struct artsEdt *)edt, slot);
  return sendReq || !aggregate;
}

void *artsRouteTableLookupItem(artsGuid_t key) {
  void *ret = NULL;
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *location =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (location)
    ret = location->data;
  return ret;
}

itemState_t artsRouteTableLookupItemWithState(artsGuid_t key, void ***data,
                                              itemState_t min, bool inc) {
  void *ret = NULL;
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *location = artsRouteTableSearchForKey(routeTable, key, min);
  if (location) {
    if (inc) {
      if (!incItem(location, 1, location->key, routeTable)) {
        *data = NULL;
        return noKey;
      }
    }
    *data = &location->data;
    return getItemState(location);
  }
  return noKey;
}

void *internalRouteTableLookupDb(artsRouteTable_t *routeTable, artsGuid_t key,
                                 int *rank, unsigned int **touched) {
  *rank = -1;
  void *ret = NULL;
  *touched = NULL;
  artsRouteItem_t *location =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (location) {
    *rank = location->rank;
    if (incItem(location, 1, location->key, routeTable)) {
      ret = location->data;
      *touched = &location->touched;
    }
  }
  return ret;
}

unsigned int internalIncDbVersion(volatile unsigned int *touched) {
  return artsAtomicAdd(touched, 1);
}

void *artsRouteTableLookupDb(artsGuid_t key, int *rank, bool touch) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  unsigned int *touched;
  void *data = internalRouteTableLookupDb(routeTable, key, rank, &touched);
  if (data) {
    if (touch)
      internalIncDbVersion(touched);
  }
  return data;
}

bool internalRouteTableReturnDb(artsRouteTable_t *routeTable, artsGuid_t key,
                                bool markToDelete, bool doDelete) {
  artsRouteItem_t *location =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (location) {
    // Only mark it for deletion if it is the last one
    // Why make it unusable to other if there is still other
    // tasks that may benifit
    // True True
    if (markToDelete && doDelete) {
      // This should work if there is only one outstanding left... me.  The
      // decItem needs to sub 1 to delete
      tryMarkDelete(location, 1);
      return decItem(routeTable, location);
    }
    // True False
    if (markToDelete && !doDelete) {
      decItem(routeTable, location);
      tryMarkDelete(location, 0);
      return false;
    }
    // False True || False False
    return decItem(routeTable, location);
  }
  return false;
}

bool artsRouteTableReturnDb(artsGuid_t key, bool markToDelete) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  bool isRemote = artsGuidGetRank(key) != artsGlobalRankId;
  return internalRouteTableReturnDb(routeTable, key, markToDelete, isRemote);
}

int artsRouteTableLookupRank(artsGuid_t key) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *location =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (location)
    return location->rank;
  return -1;
}

int artsRouteTableSetRank(artsGuid_t key, int rank) {
  int ret = -1;
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *location =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (location) {
    ret = location->rank;
    location->rank = rank;
  }
  return ret;
}

void artsRouteTableFireOO(artsGuid_t key, void (*callback)(void *, void *)) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *item =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (item != NULL)
    artsOutOfOrderListFireCallback(&item->ooList, item->data, callback);
}

bool artsRouteTableAddOO(artsGuid_t key, void *data, bool inc) {
  artsRouteItem_t *item = NULL;
  if (artsRouteTableReserveItemRace(key, &item, true) ||
      checkItemState(item, reservedKey)) {
    if (inc)
      incItem(item, 1, item->key, artsGetRouteTable(key));
    bool res = artsOutOfOrderListAddItem(&item->ooList, data);

    return res;
  }
  if (inc)
    incItem(item, 1, item->key, artsGetRouteTable(key));
  return false;
}

bool artsRouteTableAddOOExisting(artsGuid_t key, void *data, bool inc) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *item =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (item) {
    if (inc)
      incItem(item, 1, item->key, routeTable);
    bool res = artsOutOfOrderListAddItem(&item->ooList, data);
    return res;
  }
  return false;
}

void artsRouteTableResetOO(artsGuid_t key) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *item = artsRouteTableSearchForKey(routeTable, key, anyKey);
  artsOutOfOrderListReset(&item->ooList);
}

void **artsRouteTableGetOOList(artsGuid_t key,
                               struct artsOutOfOrderList **list) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *item =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (item != NULL) {
    *list = &item->ooList;
    return &item->data;
  }
  return NULL;
}

// This is just a wrapper for outside consumption...
void **artsRouteTableReserve(artsGuid_t key, bool *dec, itemState_t *state) {
  bool res;
  *dec = false;
  artsRouteItem_t *item = NULL;
  while (1) {
    res = artsRouteTableReserveItemRace(key, &item, true);
    if (!res) {
      // Check to make sure we can use it
      if (incItem(item, 1, item->key, artsGetRouteTable(key))) {
        *dec = true;
        break;
      }
      // If we were not keep trying...
    } else // we were successful in reserving
      break;
  }
  if (item)
    *state = getItemState(item);
  return &item->data;
}

artsRouteItem_t *getItemFromData(artsGuid_t key, void *data) {
  if (data) {
    artsRouteItem_t *item =
        (artsRouteItem_t *)((char *)data - sizeof(artsGuid_t));
    if (key == item->key)
      return item;
  }
  return NULL;
}

void artsRouteTableDecItem(artsGuid_t key, void *data) {
  if (data) {
    artsRouteTable_t *routeTable = artsGetRouteTable(key);
    decItem(routeTable, getItemFromData(key, data));
  }
}

artsRouteTableIterator *artsNewRouteTableIterator(artsRouteTable_t *table) {
  artsRouteTableIterator *ret =
      (artsRouteTableIterator *)artsCalloc(1, sizeof(artsRouteTableIterator));
  ret->table = table;
  return ret;
}

void artsResetRouteTableIterator(artsRouteTableIterator *iter,
                                 artsRouteTable_t *table) {
  iter->table = table;
  iter->index = 0;
}

artsRouteItem_t *artsRouteTableIterate(artsRouteTableIterator *iter) {
  artsRouteTable_t *current = iter->table;
  artsRouteTable_t *next;
  while (current != NULL) {
    for (uint64_t i = iter->index; i < current->size * collisionResolves; i++) {
      // artsPrintItem(&current->data[i]);
      if (current->data[i].lock) {
        iter->index = i + 1;
        iter->table = current;
        return &current->data[i];
      }
    }
    iter->index = 0;
    artsReaderLock(&current->readerLock, &current->writerLock);
    next = current->next;
    artsReaderUnlock(&current->readerLock);
    current = next;
  }
  return NULL;
}

void artsPrintItem(artsRouteItem_t *item) {
  if (item) {
    uint64_t local = item->lock;
    ARTS_INFO(
        "GUID: %lu DATA: %p RANK: %u LOCK: %p COUNTERS: %lu Res: %u Req: %u "
        "Avail: %u Del: %u",
        item->key, item->data, item->rank, local, getCount(local),
        isRes(local) != 0, isReq(local) != 0, isAvail(local) != 0,
        isDel(local) != 0);
  }
}

uint64_t artsCleanUpRouteTable(artsRouteTable_t *routeTable) {
  uint64_t freeSize = 0;
  artsRouteTableIterator iter;
  artsResetRouteTableIterator(&iter, routeTable);

  artsRouteItem_t *item = artsRouteTableIterate(&iter);
  while (item) {
    // artsPrintItem(item);
    artsType_t type = artsGuidGetType(item->key);
    // These are DB types
    if (type > ARTS_BUFFER && type < ARTS_LAST_TYPE) {
      struct artsDb *db = (struct artsDb *)item->data;
      if (db) {
        if (!artsAtomicSub(&db->copyCount, 1)) {
          freeSize += db->header.size;
          artsDbFree(db);
          freeItem(item);
        }
      }
    }
    item = artsRouteTableIterate(&iter);
  }
  return freeSize;
}

void artsCleanUpDbs() {
  uint64_t freeSize = 0;
  for (unsigned int i = 0; i < artsNodeInfo.totalThreadCount; i++)
    freeSize += artsCleanUpRouteTable(artsNodeInfo.routeTable[i]);
  // artsCleanUpRouteTable(artsNodeInfo.remoteRouteTable);
  ARTS_INFO("Cleaned %lu bytes", freeSize);
}

// To cleanup
// --------------------------------------------------------------------------->

bool artsRouteTableUpdateItem(artsGuid_t key, void *data, unsigned int rank,
                              itemState_t state) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  bool ret = false;
  artsRouteItem_t *found = NULL;
  while (!found) {
    found = artsRouteTableSearchForKey(routeTable, key, state);
    if (found) {
      found->data = data;
      found->rank = rank;
      markWrite(found);
      ret = true;
    }
  }
  return ret;
}

bool artsRouteTableInvalidateItem(artsGuid_t key) {
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  return internalRouteTableRemoveItem(routeTable, key);
}

void artsRouteTableAddRankDuplicate(artsGuid_t key, unsigned int rank) {}

struct artsDbFrontierIterator *
artsRouteTableGetRankDuplicates(artsGuid_t key, unsigned int rank) {
  struct artsDbFrontierIterator *iter = NULL;
  artsRouteTable_t *routeTable = artsGetRouteTable(key);
  artsRouteItem_t *location =
      artsRouteTableSearchForKey(routeTable, key, availableKey);
  if (location) {
    if (rank != -1) {
      // Blocks until the OO is done firing
      artsOutOfOrderListReset(&location->ooList);
      location->rank = rank;
    }
    struct artsDb *db = (struct artsDb *)location->data;
    iter = artsCloseFrontier((struct artsDbList *)db->dbList);
  }
  return iter;
}
