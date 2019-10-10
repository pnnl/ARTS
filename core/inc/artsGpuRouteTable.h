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
#ifndef ARTSGPUROUTETABLE_H
#define ARTSGPUROUTETABLE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "artsRouteTable.h"

typedef struct 
{
    uint64_t size;
    uint64_t timestamp;
    volatile void * realData;
} artsItemWrapper_t;

typedef struct
{
    artsRouteTable_t routingTable;
    artsItemWrapper_t * wrappers;
    volatile unsigned int gcLock;
} artsGpuRouteTable_t;

artsRouteTable_t * artsGpuNewRouteTable(unsigned int routeTableSize, unsigned int shift);

uint64_t artsGpuLookupDb(artsGuid_t key);
void * artsGpuRouteTableAddItemRace(void * item, uint64_t size, artsGuid_t key, unsigned int gpuId);
artsItemWrapper_t * artsGpuRouteTableReserveItemRace(bool * added, uint64_t size, artsGuid_t key, unsigned int gpuId);
void * artsGpuRouteTableAddItemToDeleteRace(void * item, uint64_t size, artsGuid_t key, unsigned int gpuId);
void * artsGpuRouteTableLookupDb(artsGuid_t key, int gpuId);
bool artsGpuRouteTableReturnDb(artsGuid_t key, bool markToDelete, unsigned int gpuId);
bool artsGpuInvalidateRouteTables(artsGuid_t key, unsigned int keepOnThisGpu);
uint64_t artsGpuCleanUpRouteTable(unsigned int sizeToClean, bool cleanZeros, unsigned int gpuId);
uint64_t artsGpuFreeAll(unsigned int gpuId);
#ifdef __cplusplus
}
#endif

#endif
