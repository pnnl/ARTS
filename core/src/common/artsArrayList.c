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
#include "artsArrayList.h"

artsArrayListElement * artsNewArrayListElement(uint64_t start, size_t elementSize, size_t arrayLength)
{
    artsArrayListElement * ret = (artsArrayListElement*) artsMalloc(sizeof(artsArrayListElement) + elementSize * arrayLength);  
    ret->start = start;
    ret->next = NULL;
    ret->array = (void*)(1+ret);
    return ret;
}

artsArrayList * artsNewArrayList(size_t elementSize, size_t arrayLength)
{
    artsArrayList * ret = (artsArrayList*) artsMalloc(sizeof(artsArrayList));
    ret->elementSize = elementSize;
    ret->arrayLength = arrayLength;
    ret->head = ret->current = artsNewArrayListElement(0, elementSize, arrayLength);
    ret->index = 0;
    ret->lastRequest = 0;
    ret->lastRequestPtr = ret->head->array;
    return ret;
}

void artsDeleteArrayList(artsArrayList * aList)
{
    artsArrayListElement * trail;
    artsArrayListElement * current = aList->head;
    while(current)
    {
        trail = current;
        current = current->next;
        artsFree(trail);
    }
    artsFree(aList);    
}

uint64_t artsPushToArrayList(artsArrayList * aList, void * element)
{
    uint64_t index = aList->index;
    if(!(aList->index % aList->arrayLength) && aList->index)
    {
        if(!aList->current->next)
            aList->current->next = artsNewArrayListElement(aList->current->start+aList->arrayLength, aList->elementSize, aList->arrayLength);
        aList->current = aList->current->next;
    }
    uint64_t offset =  aList->index - aList->current->start;
    void * ptr = (void*)((char*)aList->current->array + offset*aList->elementSize);
    memcpy(ptr, element, aList->elementSize);
    aList->index++;
    return index;
}

void artsResetArrayList(artsArrayList * aList)
{
    aList->current = aList->head;
    aList->index = 0;
    aList->lastRequest = 0;
    aList->lastRequestPtr = aList->head->array;
}

uint64_t artsLengthArrayList(artsArrayList * aList)
{
    return aList->index;
}

void * artsGetFromArrayList(artsArrayList * aList, uint64_t index)
{
    if(aList)
    {
        //Fastest Path
        if(index==aList->lastRequest)
            return aList->lastRequestPtr;

        if(index < aList->index)
        {           
            aList->lastRequest = index;

            //Faster Path
            if(aList->index < aList->arrayLength)
            {
                aList->lastRequestPtr = (void*) ((char*)aList->head->array + index * aList->elementSize);
                return aList->lastRequestPtr;
            }

            //Slow Path
            artsArrayListElement * node = aList->head;
            while(node && index >= node->start + aList->arrayLength )
                node = node->next;
            if(node)
            {
                uint64_t offset =  index - node->start;
                aList->lastRequestPtr = (void*) ((char*)node->array + offset * aList->elementSize);
                return aList->lastRequestPtr; 
            }
        }
    }
    return NULL;
}

    artsArrayListIterator * artsNewArrayListIterator(artsArrayList * aList)
    {
        artsArrayListIterator * iter = artsMalloc(sizeof(artsArrayListIterator));
        iter->index = 0;
        iter->last = aList->index;
        iter->elementSize = aList->elementSize;
        iter->arrayLength = aList->arrayLength;
        iter->current = aList->head;

        return iter;
    }
    
    void * artsArrayListNext(artsArrayListIterator * iter)
    {
        void * ret = NULL;
        if(iter)
        {
            if(iter->index < iter->last)
            {
                if(!(iter->index % iter->arrayLength) && iter->index)
                {
                    iter->current = iter->current->next;
                }
                if(iter->current)
                {
                    ret = (void*) ((char*)iter->current->array + (iter->index - iter->current->start) * iter->elementSize);
                    iter->index++;
                }
            }
        }
        return ret;
    }
    
    bool artsArrayListHasNext(artsArrayListIterator * iter)
    {
        return (iter->index < iter->last);
    }
    
    void artsDeleteArrayListIterator(artsArrayListIterator * iter)
    {
        artsFree(iter);
    }
