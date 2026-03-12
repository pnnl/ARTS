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
#include "arts/utils/LinkList.h"

#include "arts/arts.h"
#include "arts/utils/Atomics.h"

void artsLinkListNew(struct artsLinkList *list) {
  list->headPtr = list->tailPtr = NULL;
}

void artsLinkListDelete(void *linkList) {
  struct artsLinkList *list = (struct artsLinkList *)linkList;
  struct artsLinkListItem *last;
  while (list->headPtr != NULL) {
    last = (struct artsLinkListItem *)list->headPtr;
    list->headPtr = list->headPtr->next;
    artsFree(last);
  }
  artsFree(linkList);
}

struct artsLinkList *artsLinkListGroupNew(unsigned int listSize) {
  struct artsLinkList *linkList =
      (struct artsLinkList *)artsCalloc(listSize, sizeof(struct artsLinkList));
  for (int i = 0; i < listSize; i++) {
    artsLinkListNew(&linkList[i]);
  }
  return linkList;
}

void *artsLinkListNewItem(unsigned int size) {
  struct artsLinkListItem *newItem = (struct artsLinkListItem *)artsCalloc(
      1, sizeof(struct artsLinkListItem) + size);
  newItem->next = NULL;
  if (size) {
    return (void *)(newItem + 1);
  }
  return NULL;
}

void artsLinkListDeleteItem(void *toDelete) {
  struct artsLinkListItem *item = ((struct artsLinkListItem *)toDelete) - 1;
  artsFree(item);
}

inline struct artsLinkList *artsLinkListGet(struct artsLinkList *linkList,
                                            unsigned int position) {
  return (struct artsLinkList *)(linkList + position);
}

inline unsigned artsLinkListGetSize(struct artsLinkList *linkList) {
  unsigned size = 0;
  struct artsLinkListItem *head = NULL;
  artsLock(&linkList->lock);
  head = linkList->headPtr;
  while (head != NULL) {
    size++;
    head = head->next;
  }
  artsUnlock(&linkList->lock);
  return size;
}

inline uint8_t artsLinkListIsEmpty(struct artsLinkList *linkList) {
  struct artsLinkListItem *head = NULL;
  artsLock(&linkList->lock);
  head = linkList->headPtr;
  artsUnlock(&linkList->lock);
  return (head == NULL);
}

void *artsLinkListGetFrontData(struct artsLinkList *linkList) {
  void *data = NULL;
  artsLock(&linkList->lock);
  data = linkList->headPtr + 1;
  artsUnlock(&linkList->lock);
  return data;
}

void *artsLinkListGetTailData(struct artsLinkList *linkList) {
  void *data = NULL;
  artsLock(&linkList->lock);
  data = linkList->tailPtr + 1;
  artsUnlock(&linkList->lock);
  return data;
}

void artsLinkListPushBack(struct artsLinkList *list, void *item) {
  struct artsLinkListItem *newItem = (struct artsLinkListItem *)item;
  newItem -= 1;
  artsLock(&list->lock);
  if (list->headPtr == NULL) {
    list->headPtr = list->tailPtr = newItem;
  } else {
    list->tailPtr->next = newItem;
    list->tailPtr = newItem;
  }
  artsUnlock(&list->lock);
}

void *artsLinkListPopFront(struct artsLinkList *list, void **freePos) {
  void *data = NULL;
  if (freePos)
    *freePos = NULL;
  artsLock(&list->lock);
  if (list->headPtr) {
    data = (void *)(list->headPtr + 1);
    list->headPtr = list->headPtr->next;
  }
  artsUnlock(&list->lock);
  return data;
}
