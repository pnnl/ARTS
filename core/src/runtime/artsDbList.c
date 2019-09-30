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
#include "artsDbList.h"
#include "artsGlobals.h"
#include "artsAtomics.h"
#include "artsRemoteFunctions.h"
#include "artsRouteTable.h"
#include "artsOutOfOrder.h"
#include "artsRuntime.h"
#include "artsEdtFunctions.h"

#define writeSet     0x80000000
#define exclusiveSet 0x40000000

#define DPRINTF(...)

void frontierLock(volatile unsigned int * lock)
{
    DPRINTF("Llocking frontier: >>>>>>> %p\n", lock);
    unsigned int local, temp;
    while(1)
    {
        local = *lock;
        if((local & 1U) == 0)
        {
            temp = artsAtomicCswap(lock, local, local | 1U);
            if(temp == local)
            {
                return;
            }
        }
    }
}

void frontierUnlock(volatile unsigned int * lock)
{
    DPRINTF("unlocking frontier: <<<<<<< %p\n", lock);
    unsigned int mask = writeSet | exclusiveSet;
    artsAtomicFetchAnd(lock, mask);
}

bool frontierAddReadLock(volatile unsigned int * lock)
{
    DPRINTF("Rlocking frontier: >>>>>>> %p %u\n", lock, *lock);
    unsigned int local, temp;
    while(1)
    {
        local = *lock;
        if((local & exclusiveSet) != 0) //Make sure exclusive not set first
            return false;
        if((local & 1U) == 0)
        {
            temp = artsAtomicCswap(lock, local, local | 1U);
            if(temp == local)
            {
                return true;
            }
        }
    }
}

//Returns true if there is no write in the frontier, false if there is
bool frontierAddWriteLock(volatile unsigned int * lock)
{
    DPRINTF("Wlocking frontier: >>>>>>> %p\n", lock);
    unsigned int local, temp;
    while(1)
    {
        local = *lock;
        if((local & exclusiveSet) != 0) //Make sure exclusive not set first
            return false;
        if((local & writeSet) != 0) //Make sure write not set first
            return false;
        if((local & 1U) == 0) //Wait for lock to be free
        {
            temp = artsAtomicCswap(lock, local, local | writeSet | 1U);
            if(temp == local)
            {
                return true;
            }
        }
    }
}

bool frontierAddExclusiveLock(volatile unsigned int * lock)
{
    DPRINTF("Elocking frontier: >>>>>>> %p\n", lock);
    unsigned int local, temp;
    while(1)
    {
        local = *lock;
        if((local & exclusiveSet) != 0) //Make sure exclusive not set first
            return false;
        if((local & writeSet) != 0) //Make sure write not set first
            return false;
        if((local & 1U) == 0) //We reserved the write, now wait for lock to be free
        {
            temp = artsAtomicCswap(lock, local, local | exclusiveSet | writeSet | 1U);
            if(temp == local)
            {
                return true;
            }
        }
        return true;
    }
}

void readerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(1)
    {
        while(*writer);
        artsAtomicFetchAdd(reader, 1U);
        if(*writer==0)
            break;
        artsAtomicSub(reader, 1U);
    }
}

void readerUnlock(volatile unsigned int * reader)
{
    artsAtomicSub(reader, 1U);
}

void writerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(artsAtomicCswap(writer, 0U, 1U) == 0U);
    while(*reader);
    return;
}

bool nonBlockingWriteLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    if(artsAtomicCswap(writer, 0U, 1U) == 0U)
    {
        while(*reader);
        return true;
    }
    else
    {
        return false;
    }
}

void writerUnlock(volatile unsigned int * writer)
{
    artsAtomicSwap(writer, 0U);
}

struct artsDbElement * artsNewDbElement()
{
    struct artsDbElement * ret = (struct artsDbElement*) artsCalloc(sizeof(struct artsDbElement));
    return ret;
}

struct artsDbFrontier * artsNewDbFrontier()
{
    struct artsDbFrontier * ret = (struct artsDbFrontier*) artsCalloc(sizeof(struct artsDbFrontier));
    return ret;
}

//This should be done before being released into the wild
struct artsDbList * artsNewDbList()
{
    struct artsDbList * ret = (struct artsDbList *) artsCalloc(sizeof(struct artsDbList));
    ret->head = ret->tail = artsNewDbFrontier();
    return ret;
}

void artsDeleteDbElement(struct artsDbElement * head)
{
    struct artsDbElement * trail;
    struct artsDbElement * current;
    while(current)
    {
        trail = current;
        current = current->next;
        artsFree(trail);
    }
}

void artsDeleteLocalDelayedEdt(struct artsLocalDelayedEdt * head)
{
    struct artsLocalDelayedEdt * trail;
    struct artsLocalDelayedEdt * current;
    while(current)
    {
        trail = current;
        current = current->next;
        artsFree(trail);
    }
}

void artsDeleteDbFrontier(struct artsDbFrontier * frontier)
{
    if(frontier->list.next)
        artsDeleteDbElement(frontier->list.next);
    if(frontier->localDelayed.next)
        artsDeleteLocalDelayedEdt(frontier->localDelayed.next);
    artsFree(frontier);
}

bool artsPushDbToElement(struct artsDbElement * head, unsigned int position, unsigned int data)
{
    unsigned int j = 0;
    for(struct artsDbElement * current=head; current; current=current->next)
    {
        for(unsigned int i=0; i<DBSPERELEMENT; i++)
        {
            if(j<position)
            {
                if(current->array[i] == data)
                {
                    return false;
                }
                j++;
            }
            else
            {
                current->array[i] = data;
                return true;
            }
        }
        if(!current->next)
            current->next = artsNewDbElement();
    }
    // Need to mark unreachable
    return false;
}

void artsPushDelayedEdt(struct artsLocalDelayedEdt * head, unsigned int position, struct artsEdt * edt, unsigned int slot, artsType_t mode)
{
    unsigned int numElements = position / DBSPERELEMENT;
    unsigned int elementPos = position % DBSPERELEMENT;
    struct artsLocalDelayedEdt * current = head;
    for(unsigned int i=0; i<numElements; i++)
    {
        if(!current->next)
            current->next = artsCalloc(sizeof(struct artsLocalDelayedEdt));
        current=current->next;
    }
    current->edt[elementPos] = edt;
    current->slot[elementPos] = slot;
    current->mode[elementPos] = mode;
}

bool artsPushDbToFrontier(struct artsDbFrontier * frontier, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct artsEdt * edt, unsigned int slot, artsType_t mode, bool * unique)
{
    if(bypass)
    {
        DPRINTF("A\n");
        frontierLock(&frontier->lock);
        DPRINTF("B\n");
    }
    else if(exclusive && !frontierAddExclusiveLock(&frontier->lock))
    {
        DPRINTF("Failed artsPushDbToFrontier 1\n");
        return false;
    }
    else if(write && !frontierAddWriteLock(&frontier->lock))
    {
        DPRINTF("Failed artsPushDbToFrontier 2\n");
        return false;
    }
    else if(!exclusive && !write && !frontierAddReadLock(&frontier->lock))
    {
        DPRINTF("Failed artsPushDbToFrontier 3\n");
        return false;
    }

    if(artsPushDbToElement(&frontier->list, frontier->position, data))
        frontier->position++;
//    else
//    {
//        if(*unique && !local)
//            PRINTF("agging the write after read\n");
//        *unique = local;
//
//    }

    DPRINTF("*unique %u w: %u e: %u l: %u b: %u rank: %u\n", *unique, write, exclusive, local, bypass, data);

    if(exclusive || (write && !local))
    {
        frontier->exNode = data;
        frontier->exEdt = edt;
        frontier->exSlot = slot;
        frontier->exMode = mode;
    }
    else if(local)
    {
        artsPushDelayedEdt(&frontier->localDelayed, frontier->localPosition++, edt, slot, mode);
    }

    frontierUnlock(&frontier->lock);
    return true;
}

//Returns if the push is to the head frontier
/* A read after write from the same node would send duplicate copies of DB.
 * To fix this, if the node is remote, we only return true if the adding the
 * rank to the frontier is unique.  If the db is local then we return if the DB
 * is added to the first frontier reguardless of if there are duplicates.
 */
bool artsPushDbToList(struct artsDbList * dbList, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct artsEdt * edt, unsigned int slot, artsType_t mode)
{
    if(!dbList->head)
    {
        if(nonBlockingWriteLock(&dbList->reader, &dbList->writer))
        {
            dbList->head = dbList->tail = artsNewDbFrontier();
            writerUnlock(&dbList->writer);
        }
    }
    readerLock(&dbList->reader, &dbList->writer);
    struct artsDbFrontier * frontier = dbList->head;
    bool ret = true;
    for(struct artsDbFrontier * frontier = dbList->head; frontier; frontier=frontier->next)
    {
        DPRINTF("FRONT: %p, W: %u E: %u L: %u B: %u %p\n", frontier, write, exclusive, local, bypass, edt);
        if(artsPushDbToFrontier(frontier, data, write, exclusive, local, bypass, edt, slot, mode, &ret))
            break;
        else
        {
            if(!frontier->next)
            {
                struct artsDbFrontier * newFrontier = artsNewDbFrontier();
                if(artsAtomicCswapPtr((void*)&frontier->next, NULL, newFrontier))
                {
                    artsDeleteDbFrontier(newFrontier);
                    while(!frontier->next);
                }
            }
        }
        ret = false;
    }
    readerUnlock(&dbList->reader);
    return ret;
}

unsigned int artsCurrentFrontierSize(struct artsDbList * dbList)
{
    unsigned int size;
    readerLock(&dbList->reader, &dbList->writer);
    if(dbList->head)
    {
        frontierLock(&dbList->head->lock);
        size = dbList->head->position;
        frontierUnlock(&dbList->head->lock);
    }
    readerUnlock(&dbList->head->lock);
    return size;
}

struct artsDbFrontierIterator * artsDbFrontierIterCreate(struct artsDbFrontier * frontier)
{
    struct artsDbFrontierIterator * iter = NULL;
    if(frontier && frontier->position)
    {
        iter = (struct artsDbFrontierIterator *) artsCalloc(sizeof(struct artsDbFrontierIterator));
        iter->frontier = frontier;
        iter->currentElement = &frontier->list;
    }
    // Need to mark unreachable
    return NULL;
}

unsigned int artsDbFrontierIterSize(struct artsDbFrontierIterator * iter)
{
    return iter->frontier->position;
}

bool artsDbFrontierIterNext(struct artsDbFrontierIterator * iter, unsigned int * next)
{
    if(iter->currentIndex < iter->frontier->position)
    {
        *next = iter->currentElement->array[iter->currentIndex++ % DBSPERELEMENT];
        if(!(iter->currentIndex % DBSPERELEMENT))
        {
            iter->currentElement = iter->currentElement->next;
        }
        return true;
    }
    return false;
}

bool artsDbFrontierIterHasNext(struct artsDbFrontierIterator * iter)
{
    return (iter->currentIndex < iter->frontier->position);
}

void artsDbFrontierIterDelete(struct artsDbFrontierIterator * iter)
{
    artsFree(iter->frontier);
    artsFree(iter);
}

struct artsDbFrontierIterator * artsCloseFrontier(struct artsDbList * dbList)
{
    struct artsDbFrontierIterator * iter = NULL;
    readerLock(&dbList->reader, &dbList->writer);
    struct artsDbFrontier * frontier = dbList->head;
    if(frontier)
    {
        frontierLock(&frontier->lock);

        artsAtomicFetchOr(&frontier->lock, exclusiveSet | writeSet | 1U);
        iter = artsDbFrontierIterCreate(frontier);

        frontierUnlock(&frontier->lock);
    }
    readerUnlock(&dbList->reader);
    return iter;

}

void artsSignalFrontierRemote(struct artsDbFrontier * frontier, struct artsDb * db, unsigned int getFrom)
{
    frontierLock(&frontier->lock);

    if(frontier->exEdt)
    {
        if(frontier->exNode == getFrom)
            artsRemoteSendAlreadyLocal(getFrom, db->guid, frontier->exEdt, frontier->exSlot, frontier->exMode);
        else if(frontier->exNode != artsGlobalRankId)
            artsRemoteDbForwardFull(frontier->exNode, getFrom, db->guid, frontier->exEdt, frontier->exSlot, frontier->exMode);
        else
            artsRemoteDbFullRequest(db->guid, getFrom, frontier->exEdt, frontier->exSlot, frontier->exMode);

    }

    struct artsDbFrontierIterator * iter = artsDbFrontierIterCreate(frontier);
    if(iter)
    {
        unsigned int node;
        while(artsDbFrontierIterNext(iter, &node))
        {
            if(node != artsGlobalRankId && !(frontier->exEdt &&  node == frontier->exNode))
            {
                artsRemoteDbForward(node, getFrom, db->guid, ARTS_DB_READ); //Don't care about mode
            }
        }
    }

    if(frontier->localPosition)
    {
        struct artsLocalDelayedEdt * current = &frontier->localDelayed;
        for(unsigned int i=0; i<frontier->localPosition; i++)
        {
            unsigned int pos = i % DBSPERELEMENT;
            struct artsEdt * edt = current->edt[pos];
            unsigned int slot = current->slot[pos];
            //send through aggregation
            artsRemoteDbRequest(db->guid, getFrom, edt, slot, ARTS_DB_READ, true);
            if(pos+1 == DBSPERELEMENT)
                current = current->next;
        }
    }

    if(artsPushDbToElement(&frontier->list, frontier->position, getFrom))
        frontier->position++;
    frontierUnlock(&frontier->lock);
}

void artsSignalFrontierLocal(struct artsDbFrontier * frontier, struct artsDb * db)
{
    frontierLock(&frontier->lock);

    if(frontier->exEdt)
    {
        if(frontier->exNode == artsGlobalRankId)
        {
            struct artsEdt * edt = frontier->exEdt;
            artsEdtDep_t * depv = artsGetDepv(edt);
            depv[frontier->exSlot].ptr = db+1;
            if(artsAtomicSub(&edt->depcNeeded,1U) == 0)
                artsHandleRemoteStolenEdt(edt);
        }
        else
            artsRemoteDbFullSendNow(frontier->exNode, db, frontier->exEdt, frontier->exSlot, frontier->exMode);
    }

    struct artsDbFrontierIterator * iter = artsDbFrontierIterCreate(frontier);
    if(iter)
    {
        unsigned int node;
        while(artsDbFrontierIterNext(iter, &node))
        {
            if(node != artsGlobalRankId && !(frontier->exEdt &&  node == frontier->exNode))
            {
                artsRemoteDbSendNow(node, db);
                PRINTF("Progress Local sending to %u\n", node);
            }
        }
    }


    if(frontier->localPosition)
    {
        struct artsLocalDelayedEdt * current = &frontier->localDelayed;
        for(unsigned int i=0; i<frontier->localPosition; i++)
        {
            unsigned int pos = i % DBSPERELEMENT;
            struct artsEdt * edt = current->edt[pos];
            //This is prob wrong now with GPUs
            artsEdtDep_t * depv = artsGetDepv(edt);
            depv[current->slot[pos]].ptr = db+1;

            if(artsAtomicSub(&edt->depcNeeded,1U) == 0)
            {
                artsHandleRemoteStolenEdt(edt);
            }

            if(pos+1 == DBSPERELEMENT)
                current = current->next;
        }
    }
    frontierUnlock(&frontier->lock);
}

void artsProgressFrontier(struct artsDb * db, unsigned int rank)
{
    struct artsDbList * dbList = db->dbList;
    writerLock(&dbList->reader, &dbList->writer);
    struct artsDbFrontier * tail = dbList->head;
    if(dbList->head)
    {
        PRINTF("HEAD: %p -> NEXT: %p\n", dbList->head, dbList->head->next);
        dbList->head = (struct artsDbFrontier *) dbList->head->next;
        if(dbList->head)
        {
            if(rank==artsGlobalRankId)
                artsSignalFrontierLocal(dbList->head, db);
            else
                artsSignalFrontierRemote(dbList->head, db, rank);
        }
    }
    writerUnlock(&dbList->writer);
    //This should be safe since the writer lock ensures all readers are done
    if(tail)
        artsDeleteDbFrontier(tail);
}

struct artsDbFrontierIterator * artsProgressAndGetFrontier(struct artsDbList * dbList)
{
    writerLock(&dbList->reader, &dbList->writer);
    struct artsDbFrontier * tail = dbList->head;
    dbList->head = (struct artsDbFrontier *) dbList->head->next;
    writerUnlock(&dbList->writer);
    //This should be safe since the writer lock ensures all readers are done
    return artsDbFrontierIterCreate(tail);
}

/*Not sure if we need then...**************************************************/

unsigned int * makeCopy(struct artsDbFrontier * frontier)
{
    unsigned int * array = artsCalloc(sizeof(unsigned int) * frontier->position);
    unsigned int numElements = frontier->position / DBSPERELEMENT;
    unsigned int lastPos = frontier->position % DBSPERELEMENT;
    struct artsDbElement * current = &frontier->list;
    unsigned int k = 0;
    for(unsigned int i=0; i<numElements; i++)
    {
        unsigned int bound = (i+1 == numElements) ? lastPos : DBSPERELEMENT;
        for(unsigned int j=0; j<bound; j++)
        {
            array[k++] = current->array[j];
        }
        current = current->next;
    }
    return array;
}

void quicksort(unsigned int * array, unsigned int length)
{
  if (length < 2)
      return;

  unsigned int pivot = array[length/2];
  unsigned int i, j;
  for(i = 0, j = length-1; ; i++, j--)
  {
    while(array[i] < pivot)
        i++;
    while(array[j] > pivot)
        j--;

    if (i >= j) break;
    unsigned int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }

  quicksort(array, i);
  quicksort(array+i, length-i);
}

void replaceWithCopy(struct artsDbFrontier * frontier, unsigned int * array)
{
    unsigned int numElements = frontier->position / DBSPERELEMENT;
    unsigned int lastPos = frontier->position % DBSPERELEMENT;
    struct artsDbElement * current = &frontier->list;
    unsigned int k = 0;
    for(unsigned int i=0; i<numElements; i++)
    {
        unsigned int bound = (i+1 == numElements) ? lastPos : DBSPERELEMENT;
        for(unsigned int j=0; j<bound; j++)
        {
            current->array[j] = array[k++];
        }
        current = current->next;
    }
}

void sortSingleElement(struct artsDbElement * current, unsigned int size)
{
    unsigned int * array = current->array;
    for(unsigned int i=1; i<size; i++)
    {
        unsigned int temp = array[i];
        unsigned int j = i;
        while(j>0 && temp < array[j-1])
        {
            array[j] = array[j-1];
            j--;
        }
        array[j] = temp;
    }
}

void artsDbFrontierSort(struct artsDbFrontier * frontier)
{
    frontierLock(&frontier->lock);
    if(frontier->position > 1)
    {
        if(frontier->position <= DBSPERELEMENT)
            sortSingleElement(&frontier->list, frontier->position);
        else
        {
            unsigned int * copy = makeCopy(frontier);
            quicksort(copy, frontier->position);
            replaceWithCopy(frontier, copy);
            artsFree(copy);
        }
    }
    frontierUnlock(&frontier->lock);
}
