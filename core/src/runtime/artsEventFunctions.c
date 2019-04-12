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
#include "artsEventFunctions.h"
#include "artsAtomics.h"
#include "artsGlobals.h"
#include "artsGuid.h"
#include "artsRemoteFunctions.h"
#include "artsEdtFunctions.h"
#include "artsOutOfOrder.h"
#include "artsRouteTable.h"
#include "artsDebug.h"
#include "artsCounter.h"
#include "artsIntrospection.h"

extern __thread struct artsEdt * currentEdt;

bool artsEventCreateInternal(artsGuid_t * guid, unsigned int route, unsigned int dependentCount, unsigned int latchCount, bool destroyOnFire) {
    unsigned int eventSize = sizeof (struct artsEvent) + sizeof (struct artsDependent) * dependentCount;
    ARTSSETMEMSHOTTYPE(artsEventMemorySize);
    void * eventPacket = artsCalloc(eventSize);
    ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);

    if (eventSize) {
        struct artsEvent *event = eventPacket;
        event->header.type = ARTS_EVENT;
        event->header.size = eventSize;
        event->dependentCount = 0;
        event->dependent.size = dependentCount;
        event->latchCount = latchCount;
        event->destroyOnFire = (destroyOnFire) ? dependentCount : -1;
        event->data = NULL_GUID;

        if (route == artsGlobalRankId) {
            if (*guid) {
                artsRouteTableAddItem(eventPacket, *guid, artsGlobalRankId, false);
                artsRouteTableFireOO(*guid, artsOutOfOrderHandler);
            } else {
                *guid = artsGuidCreateForRank(route, ARTS_EVENT);
                artsRouteTableAddItem(eventPacket, *guid, artsGlobalRankId, false);
            }
        } else
            artsRemoteMemoryMove(route, *guid, eventPacket, eventSize, ARTS_REMOTE_EVENT_MOVE_MSG, artsFree);

        return true;
    }
    return false;
}

artsGuid_t artsEventCreate(unsigned int route, unsigned int latchCount) {
    ARTSEDTCOUNTERTIMERSTART(eventCreateCounter);
    artsGuid_t guid = NULL_GUID;
    artsEventCreateInternal(&guid, route, INITIAL_DEPENDENT_SIZE, latchCount, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(eventCreateCounter);
    return guid;
}

artsGuid_t artsEventCreateWithGuid(artsGuid_t guid, unsigned int latchCount) {
    ARTSEDTCOUNTERTIMERSTART(eventCreateCounter);
    unsigned int route = artsGuidGetRank(guid);
    bool ret = artsEventCreateInternal(&guid, route, INITIAL_DEPENDENT_SIZE, latchCount, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(eventCreateCounter);
    return (ret) ? guid : NULL_GUID;
}

void artsEventFree(struct artsEvent * event) {
    struct artsDependentList * trail, * current = event->dependent.next;
    while (current) {
        trail = current;
        current = current->next;
        artsFree(trail);
    }
    artsFree(event);
}

void artsEventDestroy(artsGuid_t guid) {
    struct artsEvent * event = (struct artsEvent*) artsRouteTableLookupItem(guid);
    if (event != NULL)
    {
        artsRouteTableRemoveItem(guid);
        artsEventFree(event);
    }
}

void artsEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot) {
    ARTSEDTCOUNTERTIMERSTART(signalEventCounter);
    if (currentEdt && currentEdt->invalidateCount > 0) {
        artsOutOfOrderEventSatisfySlot(currentEdt->currentEdt, eventGuid, dataGuid, slot);
        return;
    }

    struct artsEvent * event = (struct artsEvent *) artsRouteTableLookupItem(eventGuid);
    if (!event) {
        unsigned int rank = artsGuidGetRank(eventGuid);
        if (rank != artsGlobalRankId) {
            artsRemoteEventSatisfySlot(eventGuid, dataGuid, slot);
        } else {
            artsOutOfOrderEventSatisfySlot(eventGuid, eventGuid, dataGuid, slot);
        }
    } 
    else {
        if (event->fired) {
            PRINTF("ARTS_EVENT_LATCH_T already fired guid: %lu data: %lu slot: %u\n", eventGuid, dataGuid, slot);
            artsDebugGenerateSegFault();
        }

        unsigned int res;
        if (slot == ARTS_EVENT_LATCH_INCR_SLOT) {
            res = artsAtomicAdd(&event->latchCount, 1U);
        } else if (slot == ARTS_EVENT_LATCH_DECR_SLOT) {
            if (dataGuid != NULL_GUID)
                event->data = dataGuid;
            res = artsAtomicSub(&event->latchCount, 1U);
        } else {
            PRINTF("Bad latch slot %u\n", slot);
            artsDebugGenerateSegFault();
        }

        if (!res) {
            if (artsAtomicSwapBool(&event->fired, true)) {
                PRINTF("ARTS_EVENT_LATCH_T already fired guid: %lu data: %lu slot: %u\n", eventGuid, dataGuid, slot);
                artsDebugGenerateSegFault();
            } else {
                struct artsDependentList *dependentList = &event->dependent;
                struct artsDependent *dependent = event->dependent.dependents;
                int i, j;
                unsigned int lastKnown = artsAtomicFetchAdd(&event->dependentCount, 0U);
                event->pos = lastKnown + 1;
                i = 0;
                int totalSize = 0;
                while (i < lastKnown) {
                    j = i - totalSize;
                    while (i < lastKnown && j < dependentList->size) {
                        while (!dependent[j].doneWriting);
                        if (dependent[j].type == ARTS_EDT) {
                            artsSignalEdt(dependent[j].addr, dependent[j].slot, event->data);
                        } else if (dependent[j].type == ARTS_EVENT) {
#ifdef COUNT
                            //THIS IS A TEMP FIX... problem is recursion...
                            artsCounterTimerEndIncrement(artsGetCounter((artsThreadInfo.currentEdtGuid) ? signalEventCounterOn : signalEventCounter));
                            uint64_t start = artsCounterGetStartTime(artsGetCounter((artsThreadInfo.currentEdtGuid) ? signalEventCounterOn : signalEventCounter));
#endif
                            artsEventSatisfySlot(dependent[j].addr, event->data, dependent[j].slot);
#ifdef COUNT
                            //THIS IS A TEMP FIX... problem is recursion...
                            artsCounterSetEndTime(artsGetCounter((artsThreadInfo.currentEdtGuid) ? signalEventCounterOn : signalEventCounter), start);
#endif
                        } else if (dependent[j].type == ARTS_CALLBACK) {
                            artsEdtDep_t arg;
                            arg.guid = event->data;
                            arg.ptr = artsRouteTableLookupItem(event->data);
                            arg.mode = ARTS_NULL;
                            dependent[j].callback(arg);
                        }
                        j++;
                        i++;
                    }
                    totalSize += dependentList->size;
                    while (i < lastKnown && dependentList->next == NULL);
                    dependentList = dependentList->next;
                    dependent = dependentList->dependents;
                }
                if (!event->destroyOnFire) {
                    artsEventFree(event);
                    artsRouteTableRemoveItem(eventGuid);
                }
            }
        }
    }
    artsUpdatePerformanceMetric(artsEventSignalThroughput, artsThread, 1, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(signalEventCounter);
}

struct artsDependent * artsDependentGet(struct artsDependentList * head, int position) {
    struct artsDependentList * list = head;
    volatile struct artsDependentList * temp;

    while (1) {
        //totalSize += list->size;

        if (position >= list->size) {
            if (position - list->size == 0) {
                if (list->next == NULL) {
                    temp = artsCalloc(sizeof (struct artsDependentList) + sizeof ( struct artsDependent) *list->size * 2);
                    temp->size = list->size * 2;

                    list->next = (struct artsDependentList *) temp;
                }
            }

            //EXPONENTIONAL BACK OFF THIS
            while (list->next == NULL) {
            }

            position -= list->size;
            list = list->next;
        } else
            break;
    }

    return list->dependents + position;
}

void artsAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot) {
    artsType_t mode = artsGuidGetType(destination);
    struct artsHeader *sourceHeader = artsRouteTableLookupItem(source);
    if (sourceHeader == NULL) {
        unsigned int rank = artsGuidGetRank(source);
        if (rank != artsGlobalRankId) {
            artsRemoteAddDependence(source, destination, slot, mode, rank);
        } else {
            artsOutOfOrderAddDependence(source, destination, slot, mode, source);
        }
        return;
    }

    struct artsEvent *event = (struct artsEvent *) sourceHeader;
    if (mode == ARTS_EDT) {
        struct artsDependentList *dependentList = &event->dependent;
        struct artsDependent *dependent;
        unsigned int position = artsAtomicFetchAdd(&event->dependentCount, 1U);
        dependent = artsDependentGet(dependentList, position);
        dependent->type = ARTS_EDT;
        dependent->addr = destination;
        dependent->slot = slot;
        COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
        dependent->doneWriting = true;

        unsigned int destroyEvent = (event->destroyOnFire != -1) ? artsAtomicSub(&event->destroyOnFire, 1U) : 1;
        if (event->fired) {
            while (event->pos == 0);
            if (position >= event->pos - 1) {
                artsSignalEdt(destination, slot, event->data);
                if (!destroyEvent) {
                    artsEventFree(event);
                    artsRouteTableRemoveItem(source);
                }
            }
        }
    } else if (mode == ARTS_EVENT) {
        struct artsDependentList *dependentList = &event->dependent;
        struct artsDependent *dependent;
        unsigned int position = artsAtomicFetchAdd(&event->dependentCount, 1U);
        dependent = artsDependentGet(dependentList, position);
        dependent->type = ARTS_EVENT;
        dependent->addr = destination;
        dependent->slot = slot;
        COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
        dependent->doneWriting = true;

        unsigned int destroyEvent = (event->destroyOnFire != -1) ? artsAtomicSub(&event->destroyOnFire, 1U) : 1;
        if (event->fired) {
            while (event->pos == 0);
            if (event->pos - 1 <= position) {
                artsEventSatisfySlot(destination, event->data, slot);
                if (!destroyEvent) {
                    artsEventFree(event);
                    artsRouteTableRemoveItem(source);
                }
            }
        }
    }
    return;
}

void artsAddLocalEventCallback(artsGuid_t source, eventCallback_t callback) {
    struct artsEvent *event = (struct artsEvent *) artsRouteTableLookupItem(source);
    if (event && artsGuidGetType(source) == ARTS_EVENT) {
        struct artsDependentList *dependentList = &event->dependent;
        struct artsDependent *dependent;
        unsigned int position = artsAtomicFetchAdd(&event->dependentCount, 1U);
        dependent = artsDependentGet(dependentList, position);
        dependent->type = ARTS_CALLBACK;
        dependent->callback = callback;
        dependent->addr = NULL_GUID;
        dependent->slot = 0;
        COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
        dependent->doneWriting = true;

        unsigned int destroyEvent = (event->destroyOnFire != -1) ? artsAtomicSub(&event->destroyOnFire, 1U) : 1;
        if (event->fired) {
            while (event->pos == 0);
            if (event->pos - 1 <= position) {
                artsEdtDep_t arg;
                arg.guid = event->data;
                arg.ptr = artsRouteTableLookupItem(event->data);
                arg.mode = ARTS_NULL;
                callback(arg);
                if (!destroyEvent) {
                    artsEventFree(event);
                    artsRouteTableRemoveItem(source);
                }
            }
        }
    }
}

bool artsIsEventFired(artsGuid_t event) {
    bool fired = false;
    struct artsEvent * actualEvent = (struct artsEvent *) artsRouteTableLookupItem(event);
    if (actualEvent)
        fired = actualEvent->fired;
    return fired;
}
