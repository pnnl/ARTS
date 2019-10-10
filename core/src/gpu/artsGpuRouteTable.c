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

#include "artsGpuRouteTable.h"
#include "artsAtomics.h"
#include "artsOutOfOrder.h"
#include "artsGuid.h"
#include "artsGlobals.h"
#include "artsDbList.h"
#include "artsDebug.h"
#include "artsCounter.h"
#include "artsGpuStream.h"

 #define DPRINTF(...)
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

//Must be thread local
__thread uint64_t gpuItemSizeBypass = 0;

void setGpuItem(artsRouteItem_t * item, void * data)
{
    DPRINTF("gpuItemSizeBypass: %lu\n", gpuItemSizeBypass);
    artsItemWrapper_t * wrapper = (artsItemWrapper_t*) item->data;
    wrapper->realData = data;
    wrapper->size = gpuItemSizeBypass;
    gpuItemSizeBypass = 0;
}

artsRouteTable_t * artsGpuNewRouteTable(unsigned int routeTableSize, unsigned int shift)
{
    unsigned int totalElems = collisionResolves * routeTableSize;
    artsGpuRouteTable_t * gpuRouteTable = (artsGpuRouteTable_t *) artsCalloc(sizeof(artsGpuRouteTable_t));
    gpuRouteTable->routingTable.data = (artsRouteItem_t *) artsCalloc(totalElems * sizeof(artsRouteItem_t)); 
    gpuRouteTable->routingTable.size = routeTableSize;
    gpuRouteTable->routingTable.shift = shift;
    gpuRouteTable->routingTable.setFunc = setGpuItem;
    gpuRouteTable->routingTable.freeFunc = freeGpuItem;
    gpuRouteTable->routingTable.newFunc = artsGpuNewRouteTable;

    gpuRouteTable->wrappers = (artsItemWrapper_t *) artsCalloc(totalElems * sizeof(artsItemWrapper_t));
    for(unsigned int i=0; i<totalElems; i++)
        gpuRouteTable->routingTable.data[i].data = &gpuRouteTable->wrappers[i];
        
    return &gpuRouteTable->routingTable;
}

uint64_t artsGpuLookupDb(artsGuid_t key)
{
    uint64_t ret = 0;
    for (int i=0; i<artsNodeInfo.gpu; ++i)
    {
        artsRouteTable_t * gpuRouteTable = artsNodeInfo.gpuRouteTable[i];
        artsRouteItem_t * location = artsRouteTableSearchForKey(gpuRouteTable, key, availableKey);
        if(location)
            ret |= 1<<i;
    }
    return ret;
}

void * artsGpuRouteTableAddItemRace(void * item, uint64_t size, artsGuid_t key, unsigned int gpuId)
{
    //This is a bypass thread local variable to make the api nice...
    gpuItemSizeBypass = size;
    artsRouteTable_t * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    bool ret;
    artsRouteItem_t * entry = internalRouteTableAddItemRace(&ret, routeTable, item, key, artsGlobalRankId, true, true);
    artsItemWrapper_t * wrapper = (artsItemWrapper_t*) entry->data;
    return (void *)wrapper->realData;
}

artsItemWrapper_t * artsGpuRouteTableReserveItemRace(bool * added, uint64_t size, artsGuid_t key, unsigned int gpuId)
{
    //This is a bypass thread local variable to make the api nice...
    gpuItemSizeBypass = size;
    artsRouteTable_t * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    artsRouteItem_t * entry = internalRouteTableAddItemRace(added, routeTable, NULL, key, artsGlobalRankId, true, true);
    artsItemWrapper_t * wrapper = (artsItemWrapper_t*) entry->data;
    return wrapper;
}

void * artsGpuRouteTableAddItemToDeleteRace(void * item, uint64_t size, artsGuid_t key, unsigned int gpuId)
{
    //This is a bypass thread local variable to make the api nice...
    gpuItemSizeBypass = size;
    artsRouteTable_t * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    artsRouteItem_t * entry = internalRouteTableAddDeletedItemRace(routeTable, item, key, artsGlobalRankId);
    artsItemWrapper_t * wrapper = (artsItemWrapper_t*) entry->data;
    return (void *)wrapper->realData;
}

void * artsGpuRouteTableLookupDb(artsGuid_t key, int gpuId)
{
    int dummyRank;
    artsRouteTable_t * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    artsItemWrapper_t * wrapper = (artsItemWrapper_t*) internalRouteTableLookupDb(routeTable, key, &dummyRank);
    return (wrapper) ? (void*) wrapper->realData : NULL;
}

bool artsGpuRouteTableReturnDb(artsGuid_t key, bool markToDelete, unsigned int gpuId)
{
    artsRouteTable_t * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    return internalRouteTableReturnDb(routeTable, key, markToDelete, false);
}

bool artsGpuInvalidateRouteTables(artsGuid_t key, unsigned int keepOnThisGpu)
{
    bool ret = 0;
    for(unsigned int i=0; i<artsNodeInfo.gpu; i++)
    {
        if(i != keepOnThisGpu)
            ret |= internalRouteTableRemoveItem(artsNodeInfo.gpuRouteTable[i], key);
    }
    return ret;
}

/*This takes three parameters to regulate what is deleted.  This will only clean up DBs!
1.  sizeToClean - this is the desired space to clean up.  The gc will continue untill it
    it reaches this size or it has made a full pass across the RT.  Passing -1 will make the gc
    clean up the entire RT.
2.  cleanZeros - this flag indicates if we should delete data that is not being used by anyone.
    Will delete up to sizeToClean.
3.  gpuId - the id of which GPU this RT belongs.  This is the contiguous id [0 - numGpus-1].
    Pass -1 for a host RT.
Returns the size of the memory freed!
*/
uint64_t artsGpuCleanUpRouteTable(unsigned int sizeToClean, bool cleanZeros, unsigned int gpuId)
{
    uint64_t freedSize = 0;
    artsRouteTable_t * routeTable = artsNodeInfo.gpuRouteTable[gpuId];
    artsGpuRouteTable_t * gpuRouteTable = (artsGpuRouteTable_t*) routeTable;
    //Only one person can be running the gc at a time...
    if(artsTryLock(&gpuRouteTable->gcLock))
    {
        artsRouteTableIterator iter;
        artsResetRouteTableIterator(&iter, routeTable);

        artsRouteItem_t * item = artsRouteTableIterate(&iter);
        while(item && freedSize < sizeToClean)
        {
            // artsPrintItem(item);
            artsItemWrapper_t * wrapper = (artsItemWrapper_t *) item->data;
            uint64_t size = wrapper->size;
            DPRINTF("size %u\n", size);
            if(isDel(item->lock))
            {
                uint64_t compVal = (availableItem | deleteItem);
                uint64_t newVal  = (availableItem | deleteItem) + 1;
                uint64_t oldVal  = artsAtomicCswapU64(&item->lock, compVal, newVal);
                if((compVal == oldVal) && decItem(routeTable, item))
                    freedSize+=size;
            }
            else if(cleanZeros && !getCount(item->lock))
            {
                uint64_t compVal =  availableItem;
                uint64_t newVal  = (availableItem | deleteItem) + 1;
                uint64_t oldVal  = artsAtomicCswapU64(&item->lock, compVal, newVal);
                if((compVal == oldVal) && decItem(routeTable, item))
                    freedSize+=size;
            }
            item = artsRouteTableIterate(&iter);
        }
        artsUnlock(&gpuRouteTable->gcLock);
    }
    return freedSize;
}

uint64_t artsGpuFreeAll(unsigned int gpuId)
{
    uint64_t freedSize = 0;
    artsRouteTable_t * routeTable = artsNodeInfo.gpuRouteTable[gpuId];

    artsRouteTableIterator iter;
    artsResetRouteTableIterator(&iter, routeTable);

    artsRouteItem_t * item = artsRouteTableIterate(&iter);
    while(item)
    {
        artsItemWrapper_t * wrapper = (artsItemWrapper_t *) item->data;
        freedSize += wrapper->size;
        freeGpuItem(item);
        item = artsRouteTableIterate(&iter);
    }
    return freedSize;
}