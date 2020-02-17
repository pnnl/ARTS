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

#include <stdio.h>
#include <stdlib.h>
#include "arts.h"
#include "artsRouteTable.h"
#include "artsGlobals.h"
#include "artsAtomics.h"

#define MYSIZE 10

void printRT()
{
    artsRouteTableIterator * iter = artsNewRouteTableIterator(artsNodeInfo.routeTable[0]);
    artsRouteItem_t * item = artsRouteTableIterate(iter);
    while(item)
    {
        artsPrintItem(item);
        item = artsRouteTableIterate(iter);
    }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    printf("Init per node\n");
    artsGuidRange * range = artsNewGuidRangeNode(ARTS_EDT, MYSIZE, nodeId);
    for(uint64_t i=0; i<MYSIZE; i++)
    {
        artsRouteItem_t * location = artsRouteTableAddItem((void*)range, artsGuidRangeNext(range), nodeId, 0);
        if(!i)
        {
            PRINTF("SWAPPING\n");
            artsAtomicCswapU64(&location->lock, availableItem, (availableItem | deleteItem));
        }
    }
    
    printRT();

    int rank;
    artsGuid_t guid = artsGetGuid(range, 0);
    artsRouteTableLookupDb(guid, &rank, true);
    artsRouteTableReturnDb(guid, true);

    void * ptr = artsRouteTableLookupItem(guid);
    PRINTF("Lookup %lu %p\n", guid, ptr);
    artsPrintItem(getItemFromData(guid, ptr));

    ptr = artsRouteTableLookupDb(guid, &rank, true);
    PRINTF("DB Lookup %lu %p\n", guid, ptr);
    artsPrintItem(getItemFromData(guid, ptr));

    artsRouteItem_t * location = artsRouteTableAddItem((void*)range, guid, nodeId, 0);
    // artsAtomicCswapU64(&location->lock, availableItem, (availableItem | deleteItem));

    ptr = artsRouteTableLookupItem(guid);
    PRINTF("Lookup2 %lu %p\n", guid, ptr);
    artsPrintItem(getItemFromData(guid, ptr));

    ptr = artsRouteTableLookupDb(guid, &rank, true);
    PRINTF("DB Lookup2 %lu %p\n", guid, ptr);
    artsPrintItem(getItemFromData(guid, ptr));

    artsShutdown();
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
