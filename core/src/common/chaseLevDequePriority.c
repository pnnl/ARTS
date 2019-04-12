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

// Copyright (c) 2013 Amanieu d'Antras
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include "artsDeque.h"
#include "artsAtomics.h"

struct circularArray
{
    struct circularArray * next;
    unsigned int size;
    void ** segment;
}__attribute__ ((aligned(64)));

struct artsDeque
{
    volatile uint64_t  top;
    char pad1[56];
    volatile uint64_t bottom;
    char pad2[56];
    struct circularArray * volatile activeArray;
    char pad3[56];
    volatile unsigned int leftLock;
    char pad4[60];
    volatile unsigned int rightLock;
    char pad5[60];
    struct circularArray * head;
    volatile unsigned int push;
    volatile unsigned int pop;
    volatile unsigned int steal;
    unsigned int priority;
    struct artsDeque * volatile left;
    struct artsDeque * volatile right;
}__attribute__ ((aligned(64)));

static inline struct circularArray * 
newCircularArray(unsigned int size)
{
    struct circularArray * array 
        = artsCalloc( sizeof(struct circularArray) + sizeof(void*) * size); 
//    memset(array,0,sizeof(struct circularArray) + sizeof(void*) * size);
    array->size = size;
    array->segment = (void**)(array+1);
    array->next = NULL;
    return array;
}


bool
artsDequeEmpty(struct artsDeque * deque)
{
    //don't really know what this is for
    //return (deque->bottom == deque->top);
    return false;
}

void
artsDequeClear(struct artsDeque *deque)
{
    deque->top = deque->bottom;
}

unsigned int 
artsDequeSize(struct artsDeque *deque)
{
    return deque->bottom - deque->top;
}

static inline void * 
getCircularArray(struct circularArray * array, uint64_t i)
{
    return array->segment[i%array->size];
}

static inline void 
putCircularArray(struct circularArray * array, uint64_t i, void * object)
{
    array->segment[i%array->size] = object;
}

static inline struct circularArray *
growCircularArray(struct circularArray * array, uint64_t b, uint64_t t)
{
    struct circularArray * a = newCircularArray(array->size*2);
    array->next = a;
    uint64_t i;
    for(i=t; i<b; i++)
        putCircularArray(a, i, getCircularArray(array, i));
    return a;
}

static inline void
artsDequeNewInit(struct artsDeque * deque, unsigned int size)
{
    deque->top = 1;
    deque->bottom = 1;
    deque->activeArray = newCircularArray(size);
    deque->head = deque->activeArray;
    deque->push = 0;
    deque->pop = 0;
    deque->steal = 0;
    deque->priority = 0;
    deque->left = deque->right=NULL;
    deque->leftLock = deque->rightLock = 0;
}

struct artsDeque * 
artsDequeNew(unsigned int size)
{
    struct artsDeque * deque = artsCalloc(sizeof(struct artsDeque));
    artsDequeNewInit(deque, size);
    return deque;
}

void
artsDequeDelete(struct artsDeque *deque)
{
    struct circularArray * trail, * current = deque->head;
    while(current)
    {
        trail = current;
        current = current->next;
        artsFree(trail);
    }
//    free(deque);
}
struct artsDeque * findTheDeque( struct artsDeque * deque, unsigned int priority)
{
    struct artsDeque * next = deque;
    
    while(next->priority != priority)
    {
        if(priority < next->priority)
        {
            unsigned int old;
            if(next->leftLock == 0)
            {
                old = artsAtomicCswap(&next->leftLock, 0U, 1U);
                if(old == 0U)
                {
                    struct artsDeque * nextDeque = artsDequeNew(8);
                    nextDeque->priority = priority;
                    next->left = nextDeque;
                    return next->left;
                }
            }
            while(next->left==NULL){}
            next = next->left;
        }
        else
        {
            unsigned int old;
            if(next->rightLock == 0)
            {
                old = artsAtomicCswap(&next->rightLock, 0U, 1U);
                if(old == 0U)
                {
                    struct artsDeque * nextDeque = artsDequeNew(8);
                    nextDeque->priority = priority;
                    next->right = nextDeque;
                    return next->right;
                }
            }
            while(next->right==NULL){}
            next = next->right;
            
        }
    }

    return next;
}

bool 
artsDequePushFront(struct artsDeque *deque, void *item, unsigned int priority)
{
    deque = findTheDeque(deque, priority);
    struct circularArray * a = deque->activeArray;
    uint64_t b = deque->bottom;
    uint64_t t = deque->top;
    if(b >= a->size - 1 + t)
    {
        a = growCircularArray(a, b, t);
        deque->activeArray = a;
    }
    putCircularArray(a, b, item);
    HW_MEMORY_FENCE();
    deque->bottom=b+1;
    return true;
}

void *
artsDequePopFront(struct artsDeque *deque)
{
    void * o = NULL;
    
    if(deque->left)
        o = artsDequePopFront(deque->left);
    
    if(!o)
    {

        uint64_t b = --deque->bottom;
        HW_MEMORY_FENCE();
        uint64_t t = deque->top;
        
        o = getCircularArray(deque->activeArray, b);
        if(t > b)
        {
            deque->bottom = t;

            o = NULL;
        }
        else
        {
            //Success
            getCircularArray(deque->activeArray, b);
            if(b <= t)
            {
                if(artsAtomicCswapU64(&deque->top, t, t+1) != t)
                    o = NULL;
                deque->bottom = t+1;
            }
        }

        if(!o && deque->right)
            o = artsDequePopFront(deque->right);
    }
    return o;
}

void *
artsDequePopBack(struct artsDeque *deque)
{
    void * o = NULL;
   
    if(deque->left)
        o = artsDequePopBack(deque->left);
    
    if(!o)
    {
        uint64_t t = deque->top;
        HW_MEMORY_FENCE();
        uint64_t b = deque->bottom;
        if(t < b)
        {
            o = getCircularArray(deque->activeArray, t);
            uint64_t temp = artsAtomicCswapU64(&deque->top, t, t+1);
            if(temp!=t)
            {
                o=NULL;
            }
        }
        if(!o && deque->right)
            o = artsDequePopBack(deque->right);
    }
    return o;
}

struct artsDeque *
artsDequeListNew(unsigned int listSize, unsigned int dequeSize)
{
    struct artsDeque *dequeList =
        (struct artsDeque *) artsCalloc( listSize * sizeof (struct artsDeque) );
    int i = 0;
    for (i = 0; i < listSize; i++)
        artsDequeNewInit(&dequeList[i]  , dequeSize);

    return dequeList;
}

struct artsDeque *
artsDequeListGetDeque(struct artsDeque *dequeList, unsigned int position)
{
    return dequeList+position;
}

void 
artsDequeListDelete(void *dequeList)
{
//    artsDeque * ptr = (artsDeque*) dequeList;
//    for (i = 0; i < listSize; i++)
//        artsDequeDelete( dequeList+i  , dequeSize);
//    artsFree(dequeList);
}
