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
#include "artsOutOfOrderList.h"
#include "artsAtomics.h"
#include "artsDebug.h"

#define DPRINTF
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

#define fireLock 1U
#define resetLock 2U

bool readerOOTryLock(struct artsOutOfOrderList * list)
{
    while(1)
    {
        if(list->writerLock == fireLock)
            return false;
        while(list->writerLock == resetLock);
        artsAtomicFetchAdd(&list->readerLock, 1U);
        if(list->writerLock==0)
            break;
        artsAtomicSub(&list->readerLock, 1U);
    }
    return true;
}

inline void readerOOLock(struct artsOutOfOrderList *  list)
{
    while(1)
    {
        while(list->writerLock);
        artsAtomicFetchAdd(&list->readerLock, 1U);
        if(list->writerLock==0)
            break;
        artsAtomicSub(&list->readerLock, 1U);
    }
}

void readerOOUnlock(struct artsOutOfOrderList *  list)
{
    artsAtomicSub(&list->readerLock, 1U);
}

void writerOOLock(struct artsOutOfOrderList *  list, unsigned int lockType)
{
    while(artsAtomicCswap(&list->writerLock, 0U, lockType) == 0U);
    while(list->readerLock);
    return;
}

bool writerTryOOLock(struct artsOutOfOrderList *  list, unsigned int lockType)
{
    while(1)
    {
        unsigned int temp = artsAtomicCswap(&list->writerLock, 0U, lockType);
        if(temp == 0U)
        {
            while(list->readerLock);
            break;
        }
        if(temp == lockType)
            return false;
    }
    return true;
}

void writerOOUnlock(struct artsOutOfOrderList *  list)
{
    artsAtomicSwap(&list->writerLock, 0U);
}

bool artsOOisFired(struct artsOutOfOrderList *  list)
{
    return list->isFired;
}

bool artsOutOfOrderListAddItem(struct artsOutOfOrderList * addToMe, void * item)
{
    if(!readerOOTryLock(addToMe))
    {
        return false;
    }
    if(artsOOisFired(addToMe))
    {
        readerOOUnlock(addToMe);
        return false;
    }
    unsigned int pos = artsAtomicFetchAdd(&addToMe->count, 1U);

    DPRINTF("ADDING to OO LIST %u %u %p\n", pos, addToMe->count, &addToMe->count);
    unsigned int numElements = pos / OOPERELEMENT;
    unsigned int elementPos = pos % OOPERELEMENT;

    volatile struct artsOutOfOrderElement * current = &addToMe->head;
    for(unsigned int i=0; i<numElements; i++)
    {
        if(!current->next)
        {
            if(i+1 == numElements && elementPos == 0 )
            {
                current->next = artsCalloc(sizeof(struct artsOutOfOrderElement));
            }
            else
                while(!current->next);
        }
        current=current->next;
    }
    if(artsAtomicCswapPtr((volatile void**)&current->array[elementPos], (void*)0, item))
        PRINTF("OO pos not empty...\n");
    readerOOUnlock(addToMe);
    return true;
}

void artsOutOfOrderListReset(struct artsOutOfOrderList * list)
{
    if(writerTryOOLock(list, resetLock))
    {
        list->isFired = false;
        if(list->count)
        {
            PRINTF("Reseting but OO is not empty\n");
        }
        writerOOUnlock(list);
    }
}

void deleteOOElements(struct artsOutOfOrderElement * current)
{
    struct artsOutOfOrderElement * trail = NULL;
    while(current)
    {
        for(unsigned int i=0; i<OOPERELEMENT; i++)
        {
            while(current->array[i]);
        }
        trail = current;
        current = (struct artsOutOfOrderElement *) current->next;
        artsFree(trail);
    }
}

//Not threadsafe
void artsOutOfOrderListDelete(struct artsOutOfOrderList * list)
{
    deleteOOElements((struct artsOutOfOrderElement *) list->head.next);
    list->head.next = NULL;
    list->isFired = false;
    list->count = 0;
}

void artsOutOfOrderListFireCallback(struct artsOutOfOrderList * fireMe, void * localGuidAddress,  void (* callback )( void *, void * ))
{
    if(writerTryOOLock(fireMe, fireLock))
    {
        DPRINTF("FIRING OO LIST %u\n", fireMe->count);
        fireMe->isFired = true;
        unsigned int pos = fireMe->count;
        unsigned int j = 0;
        for(volatile struct artsOutOfOrderElement * current=&fireMe->head; current; current=current->next)
        {
            for(unsigned int i=0; i<OOPERELEMENT; i++)
            {
                if(j<pos)
                {
                    volatile void * item = NULL;
                    while(!item)
                    {
                        item = artsAtomicSwapPtr((volatile void **)&current->array[i], (void*)0);
                    }
                    callback((void *)item, localGuidAddress);
                    j++;
                }
            }
            if(j == pos)
                break;
            while(!current->next);
        }
        fireMe->count = 0;
        struct artsOutOfOrderElement * p = (struct artsOutOfOrderElement *) fireMe->head.next;
        fireMe->head.next = NULL;
        writerOOUnlock(fireMe);
        deleteOOElements(p);
    }
}
