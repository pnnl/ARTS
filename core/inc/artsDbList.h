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
#ifndef ARTSDBLIST_H
#define ARTSDBLIST_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
    
#define DBSPERELEMENT 8

struct artsDbElement
{
    struct artsDbElement * next;
    unsigned int array[DBSPERELEMENT];
};

struct artsLocalDelayedEdt
{
    struct artsLocalDelayedEdt * next;
    struct artsEdt * edt[DBSPERELEMENT];
    unsigned int slot[DBSPERELEMENT];
    artsType_t mode[DBSPERELEMENT];
};

struct artsDbFrontier
{
    struct artsDbElement list;
    unsigned int position;
    struct artsDbFrontier * next;
    volatile unsigned int lock;
    
    /* 
     * This is because we can't aggregate exclusive requests
     * and we need to store them somewhere.  There will only
     * be at most one per frontier.
     */
    unsigned int exNode;
    struct artsEdt * exEdt;
    unsigned int exSlot;
    artsType_t exMode;
    
    /* 
     * This is dumb, but we need somewhere to store requests
     * that are from the guid owner but cannot be satisfied
     * because of the memory model
     */
    unsigned int localPosition;
    struct artsLocalDelayedEdt localDelayed;
};

struct artsDbList
{
    struct artsDbFrontier * head;
    struct artsDbFrontier * tail;
    volatile unsigned int reader;
    volatile unsigned int writer;
};

struct artsDbFrontierIterator
{
    struct artsDbFrontier * frontier;
    unsigned int currentIndex;
    struct artsDbElement * currentElement;
};

struct artsDbList * artsNewDbList();
unsigned int artsCurrentFrontierSize(struct artsDbList * dbList);
struct artsDbFrontierIterator * artsDbFrontierIterCreate(struct artsDbFrontier * frontier);
unsigned int artsDbFrontierIterSize(struct artsDbFrontierIterator * iter);
bool artsDbFrontierIterNext(struct artsDbFrontierIterator * iter, unsigned int * next);
bool artsDbFrontierIterHasNext(struct artsDbFrontierIterator * iter);
void artsDbFrontierIterDelete(struct artsDbFrontierIterator * iter);
void artsProgressFrontier(struct artsDb * db, unsigned int rank);
struct artsDbFrontierIterator * artsProgressAndGetFrontier(struct artsDbList * dbList);
bool artsPushDbToList(struct artsDbList * dbList, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct artsEdt * edt, unsigned int slot, artsType_t mode);
struct artsDbFrontierIterator * artsCloseFrontier(struct artsDbList * dbList);
#ifdef __cplusplus
}
#endif

#endif /* ARTSDBLIST_H */

