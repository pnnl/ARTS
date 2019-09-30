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
#include "artsRT.h"
#include "artsDebug.h"
#include "artsGlobals.h"
#include "artsCounter.h"
#include "artsIntrospection.h"

static void zeroMemory(char * addr, size_t size )
{
    for(size_t i=0; i< size; i++ )
        addr[i]=0;
}

void * artsMalloc(size_t size)
{
    ARTSEDTCOUNTERTIMERSTART(mallocMemory);
    size+=sizeof(uint64_t);
    void * address;
    if(posix_memalign(&address, 16, size))
    {
        PRINTF("Out of Memory\n");
        artsDebugGenerateSegFault();
    }
    uint64_t * temp = (uint64_t*) address;
    *temp = size;
    address = (void*)(temp+1);
    if(artsThreadInfo.mallocTrace)
        artsUpdatePerformanceMetric(artsMallocBW, artsThread, size, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(mallocMemory);
    return address;
}

void * artsRealloc(void * ptr, size_t size)
{
    uint64_t * temp = (uint64_t*) ptr;
    temp--;
    void * addr = realloc(temp, size + sizeof(uint64_t));
    temp = (uint64_t *) addr;
    *temp = size + sizeof(uint64_t);
    return ++temp;
}

void * artsCalloc(size_t size)
{
    ARTSEDTCOUNTERTIMERSTART(callocMemory);
    size+=sizeof(uint64_t);
    void * address;
    if(posix_memalign(&address, 16, size))
    {
        PRINTF("Out of Memory\n");
        artsDebugGenerateSegFault();
    }
    zeroMemory(address,size);
    uint64_t * temp = (uint64_t*) address;
    *temp = size;
    address = (void*)(temp+1);
    if(artsThreadInfo.mallocTrace)
        artsUpdatePerformanceMetric(artsMallocBW, artsThread, size, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(callocMemory);
    return address;
}

void artsFree(void *ptr)
{
    ARTSEDTCOUNTERTIMERSTART(freeMemory);
    uint64_t * temp = (uint64_t*) ptr;
    temp--;
    uint64_t size = (*temp);
    free(temp);
    if(artsThreadInfo.mallocTrace)
        artsUpdatePerformanceMetric(artsFreeBW, artsThread, size, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(freeMemory);
}

void * artsMallocAlign(size_t size, size_t align)
{
    if(!size || align < ALIGNMENT || align % 2)
        return NULL;

    void * ptr = artsMalloc(size + align);
    memset(ptr, 0, align);
    if(ptr)
    {
        char * temp = ptr;
        *temp = 'a';
        ptr = (void*)(temp+1);
        uintptr_t mask = ~(uintptr_t)(align - 1);
        ptr = (void *)(((uintptr_t)ptr + align - 1) & mask);
    }
    return ptr;
}

void * artsCallocAlign(size_t size, size_t align)
{
    if(!size || align < ALIGNMENT || align % 2)
        return NULL;

    void * ptr = artsCalloc(size + align);
    if(ptr)
    {
        char * temp = ptr;
        *temp = 1;
        ptr = (void*)(temp+1);
        uintptr_t mask = ~(uintptr_t)(align - 1);
        ptr = (void *)(((uintptr_t)ptr + align - 1) & mask);
    }
    return ptr;
}

void artsFreeAlign(void * ptr)
{
    char * trail = (char*)ptr - 1;
    while(!(*trail))
        trail--;
    artsFree(trail);
}
